import torch
import torch.optim
import matplotlib.pyplot as plt
import draw
import voxel
from utils import argmin, argmax, partial, map_range
from models import LossType, SphereModel, AxisAlignedCuboid, CuboidModel, CapsuleModel
import math
from PSOptimizer import ParticleSwarmOptimizer

class NumericalInstabilityException(Exception):
   pass

def optimize(points, model, loss_type=LossType.BEST_MATCH):
    num_steps = 500

    optimizers = [torch.optim.SGD(optimizer_param.params, lr=0.1, momentum=0.9) for optimizer_param in model.optimizer_config]
    schedulers = [(lambda x: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: map_range(i, 0.0, num_steps - 1, x.start_lr, x.end_lr)))(optimizer_param)
                  for (optimizer, optimizer_param) in zip(optimizers, model.optimizer_config)]

    model.train()

    # Uncomment the line below in the event of an anomaly detection
    #with torch.autograd.detect_anomaly():
    for i in range(num_steps):
        model.lambda_ = map_range(i, 0, num_steps - 1, 0.01, 8.0)

        for optimizer in optimizers:
            optimizer.zero_grad()

        loss = model(points, loss_type=loss_type)
        print("Loss: " + str(loss))

        if math.isnan(loss.item()):
            raise NumericalInstabilityException("Loss score was NaN. Consider using torch.autograd.detect_anomaly() to track down the source of the NaN")

        loss.backward()

        # Occasionally the gradient can blow up, so clip it here to make
        # sure we don't jump too far around the parameter space
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        for optimizer in optimizers:
            optimizer.step()

        model.normalize()

        for scheduler in schedulers:
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

    #voxel_grid = voxel.fill_holes(voxel_grid)
    voxel_grid = voxel.fill_holes_sequential(voxel_grid)

    return (voxel_grid, offset)

def restore_point_coordinates(model, offset, voxel_size):
    model.translate(-offset)
    model.uniform_scale(voxel_size)

def fit_points(points, voxel_size, max_num_fitted_models=5, use_sphere=True, use_cuboid=True, use_capsule=False,
               visualize_intermediate=False, loss_type=LossType.BEST_EFFORT, use_cuda=False,
               cuda_device=None, component_threshold=0.05):
    (voxel_grid, offset) = points_to_voxel_grid(points, voxel_size)
    if use_cuda:
        offset = offset.cuda(device=cuda_device)
    models = fit_voxel_grid(voxel_grid, max_num_fitted_models=max_num_fitted_models, use_sphere=use_sphere,
                            use_cuboid=use_cuboid, use_capsule=use_capsule, visualize_intermediate=visualize_intermediate,
                            loss_type=loss_type, use_cuda=use_cuda,
                            cuda_device=cuda_device, component_threshold=component_threshold)
    for model in models:
        restore_point_coordinates(model, offset, voxel_size)
    return models

def fit_voxel_grid(voxel_grid, max_num_fitted_models=5, use_sphere=True, use_cuboid=True, use_capsule=True,
                   visualize_intermediate=False, loss_type=LossType.BEST_EFFORT,
                   use_cuda=False, cuda_device=None, component_threshold=0.05):
    fitted_models = []

    num_voxels_total = voxel_grid.sum().item()

    voxels_remaining = voxel_grid.clone()
    connected_components = voxel.connected_components(voxels_remaining)

    i = 0
    while len(connected_components) > 0 and i < max_num_fitted_models:
        component = argmax(connected_components, lambda comp: comp.shape[0])

        if (float(component.shape[0]) / num_voxels_total) <= component_threshold:
            break

        component_points = component.float()
        if use_cuda:
            component_points = component_points.cuda(device=cuda_device)

        lambda_ = 1.0

        potential_models = []
        if use_sphere:
            potential_models.append(SphereModel(component_points, lambda_))
        if use_cuboid:
            potential_models.append(CuboidModel(component_points, lambda_))
        if use_capsule:
            potential_models.append(CapsuleModel(component_points, lambda_))

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