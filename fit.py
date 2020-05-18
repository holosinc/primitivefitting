import torch
import torch.optim
import matplotlib.pyplot as plt
import draw
import voxel
from utils import argmin, argmax, partial, map_range
from models import SphereModel, BoxModel, CylinderModel, LossType
import math

class NumericalInstabilityException(Exception):
   pass

def optimize(points, model, loss_type=LossType.BEST_EFFORT):
    num_steps = 500

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: map_range(i, 0.0, num_steps - 1, 0.1, 0.005))
    model.train()

    #with torch.autograd.detect_anomaly():
    for i in range(num_steps):
        model.lambda_ = map_range(i, 0, num_steps - 1, 0.5, 8.0)

        optimizer.zero_grad()
        loss = model(points, loss_type=loss_type)

        if math.isnan(loss.item()):
            raise NumericalInstabilityException("Loss score was NaN. Consider using torch.autograd.detect_anomaly() to track down the source of the NaN")

        loss.backward()

        # Occasionally the gradient can blow up, so clip it here to make
        # sure we don't jump too far around the parameter space
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        #print("Pos", model.position, model.position.grad)
        #print("Rot", model.rotation, model.rotation.grad)
        #print("Inv Sca", model.inverse_scale, model.inverse_scale.grad)

        optimizer.step()
        model.normalize_rotation()
        model.abs_scale()

        #if isinstance(model, BoxModel):
            #print("Box loss:", loss)
        #elif isinstance(model, SphereModel):
            #print("Sphere loss:", loss)
        #elif isinstance(model, CylinderModel):
            #print("Cylinder loss:", loss)

        scheduler.step()

    # Don't use the raw loss score since the different geometric models may have different incomparable loss scores
    # (like comparing apples to oranges)
    return model.exact_forward(points, loss_type=loss_type)

def points_to_voxel_grid(points, voxel_size):
    integer_points = torch.round(points / voxel_size).long()
    mins = torch.min(integer_points, dim=0).values
    maxes = torch.max(integer_points, dim=0).values
    size = (maxes - mins) + 1

    offset = -mins
    integer_points = integer_points + offset.unsqueeze(0)

    voxel_grid = torch.zeros(tuple(size), dtype=torch.bool)

    # Assign the point cloud to the new voxel grid
    voxel.batch_set(voxel_grid, integer_points, True)

    voxel_grid = voxel.fill_holes(voxel_grid)

    return (voxel_grid, offset)

def restore_point_coordinates(model, offset, voxel_size):
    model.position.data -= offset
    model.position.data *= voxel_size
    model.inverse_scale.data /= voxel_size

def fit_points(points, voxel_size, max_num_fitted_models=5, use_spheres=True, use_boxes=True, use_cylinders=False,
               visualize_intermediate=False, loss_type=LossType.BEST_EFFORT, use_fuzzy_containment=True, use_cuda=False,
               cuda_device=None):
    (voxel_grid, offset) = points_to_voxel_grid(points, voxel_size)
    if use_cuda:
        offset = offset.cuda(device=cuda_device)
    models = fit_voxel_grid(voxel_grid, max_num_fitted_models=max_num_fitted_models, use_spheres=use_spheres,
                            use_boxes=use_boxes, use_cylinders=use_cylinders, visualize_intermediate=visualize_intermediate,
                            loss_type=loss_type, use_fuzzy_containment=use_fuzzy_containment, use_cuda=use_cuda,
                            cuda_device=cuda_device)
    for model in models:
        restore_point_coordinates(model, offset, voxel_size)
    return models

def fit_voxel_grid(voxel_grid, max_num_fitted_models=5, use_spheres=True, use_boxes=True, use_cylinders=False,
                   visualize_intermediate=False, loss_type=LossType.BEST_EFFORT, use_fuzzy_containment=True,
                   use_cuda=False, cuda_device=None):
    fitted_models = []

    voxels_remaining = voxel_grid.clone()
    connected_components = voxel.connected_components(voxels_remaining)

    i = 0
    while len(connected_components) > 0 and i < max_num_fitted_models:
        #print("Number of connected components: " + str(len(connected_components)))

        component = argmax(connected_components, lambda component: component.shape[0])

        component_points = component.float()
        if use_cuda:
            component_points = component_points.cuda(device=cuda_device)

        lambda_ = 1.0

        potential_models = []
        if use_spheres:
            potential_models.append(SphereModel(component_points, lambda_))
        if use_boxes:
            potential_models.append(BoxModel(component_points, lambda_))
        if use_cylinders:
            potential_models.append(CylinderModel(component_points, lambda_))

        if use_cuda:
            for model in potential_models:
                model.cuda(device=cuda_device)

        best_model = argmin(potential_models, partial(optimize, component_points, loss_type=loss_type))

        if visualize_intermediate:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            draw.draw_voxels(ax, voxels_remaining)
            best_model.draw(ax)
            plt.show()

        points_inside_mask = best_model.exact_containment(component_points)
        if use_fuzzy_containment:
            best_model.lambda_ = 10.0
            points_inside_mask |= (best_model.containment(component_points) >= 0.5)
        points_outside_mask = ~points_inside_mask

        #print(best_model)
        #print("Number of points exactly inside: " + str(points_inside_mask.sum()))
        #print("Number of points exactly outside: " + str(points_outside_mask.sum()))

        indices_covered = component[points_inside_mask, :]
        voxel.batch_set(voxels_remaining, indices_covered, False)

        if points_inside_mask.sum().item() > 0:
            fitted_models.append(best_model)
        else:
            # We failed to fit to any of the voxels in this component, so just ignore it
            # Todo: try splitting the component up and fit to those pieces
            voxel.batch_set(voxels_remaining, component, False)

        i += 1

        connected_components = voxel.connected_components(voxels_remaining)

    return fitted_models