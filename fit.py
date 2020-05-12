import torch
import torch.optim
import matplotlib.pyplot as plt
import draw
import voxel
from utils import argmin, argmax, partial, map_range
from models import SphereModel, BoxModel, CylinderModel
from torchext import numerically_stable_sigmoid
import math

#point_cloud_file = "C:\\Users\\Caleb Helbling\\Documents\\holosproject\\voxelizervs\Debug\\bunnyP00625.txt"
#voxel_size = 0.00625

point_cloud_file = "C:\\Users\\Caleb Helbling\\Documents\\holosproject\\voxelizervs\Debug\\monkeyP05.txt"
voxel_size = 0.05

#point_cloud_file = "C:\\Users\\Caleb Helbling\\Documents\\holosproject\\voxelizervs\Debug\\tram2P0.txt"
#voxel_size = 2.0

#point_cloud_file = "C:\\Users\\Caleb Helbling\\Documents\\holosproject\\voxelizervs\Debug\\dragonP05.txt"
#voxel_size = 0.05

#point_cloud_file = "C:\\Users\\Caleb Helbling\\Documents\\holosproject\\voxelizervs\Debug\\beeP05.txt"
#voxel_size = 0.05

with open(point_cloud_file, 'r') as f:
    point_strs = f.readlines()

# Filter out empty lines
point_strs = [line.strip() for line in point_strs]
point_strs = [line for line in point_strs if line != ""]

points = torch.tensor([list(map(float, line.split())) for line in point_strs])

integer_points = torch.round(points / voxel_size).long()
mins = torch.min(integer_points, dim=0).values
maxes = torch.max(integer_points, dim=0).values

original_size = (maxes - mins) + 1
size = original_size

offset = -mins.unsqueeze(0)

integer_points = integer_points + offset

voxel_grid = torch.zeros(tuple(size), dtype=torch.bool)

# Assign the point cloud to the new voxel grid
voxel.batch_set(voxel_grid, integer_points, True)

voxel_grid = voxel.fill_holes(voxel_grid)

# Now that we have a filled 3D object, convert the voxel grid back to point form for the rest of the algorithm

torch.set_printoptions(profile="full")

inside_points = voxel.voxels_to_indices(voxel_grid).float()
outside_points = voxel.voxels_to_indices(~voxel_grid).float()

class NumericalInstabilityException(Exception):
   pass

def optimize(inside_points, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    prev_loss = None

    num_steps = 500

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: map_range(i, 0.0, num_steps - 1, 0.1, 0.005))

    num_points = inside_points.shape[0]

    def volume_bonus(vol):
        return numerically_stable_sigmoid((10.0 / num_points) * (vol - (num_points / 2.0)))

    #with torch.autograd.detect_anomaly():
    for i in range(num_steps):
        model.lambda_ = map_range(i, 0, num_steps - 1, 0.5, 8.0)

        optimizer.zero_grad()

        jaccard_index = model(inside_points)
        bonus = volume_bonus(model.volume())

        loss = -jaccard_index - bonus

        if math.isnan(loss.item()):
            raise NumericalInstabilityException("Loss score was NaN. Consider using torch.autograd.detect_anomaly() to track down the source of the NaN")

        loss.backward()

        # Occasionally the gradient can blow up, so clip it here to make
        # sure we don't jump too far around the parameter space
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        print("Pos", model.position, model.position.grad)
        print("Rot", model.rotation, model.rotation.grad)
        print("Inv Sca", model.inverse_scale, model.inverse_scale.grad)

        optimizer.step()
        model.normalize_rotation()
        model.abs_scale()
        #model.max_scale(1.0)

        print("jaccard_index", jaccard_index)
        print("bonus", bonus)

        if isinstance(model, BoxModel):
            print("Box loss:", loss)
        elif isinstance(model, SphereModel):
            print("Sphere loss:", loss)
        elif isinstance(model, CylinderModel):
            print("Cylinder loss:", loss)

        #if prev_loss is not None and abs(loss.item() - prev_loss) < 0.00001:
            #prev_loss = loss.item()
            #break

        prev_loss = loss.item()
        scheduler.step()

    # Don't use the raw loss score since the different geometric models may have different incomparable loss scores
    # (like comparing apples to oranges). Make a comparable loss score by using exact containment here
    return -float(model.exact_containment(inside_points).sum().item()) / model.volume().item() - volume_bonus(model.volume()).item()

fitted_models = []

voxels_remaining = voxel_grid.clone()
connected_components = voxel.connected_components(voxels_remaining)
max_num_fitted_models = 5
i = 0

while len(connected_components) > 0 and i < max_num_fitted_models:
    print("Number of connected components: " + str(len(connected_components)))

    component = argmax(connected_components, lambda component: component.shape[0])

    points = component.float()

    lambda_ = 1.0
    best_model = argmin([SphereModel(points, lambda_), BoxModel(points, lambda_), CylinderModel(points, lambda_)], partial(optimize, points))
    #best_model = argmin([SphereModel(points, lambda_)], partial(optimize, points))
    #best_model = argmin([BoxModel(points, lambda_)], partial(optimize, points))
    #best_model = argmin([CylinderModel(points, lambda_)], partial(optimize, points))

    """
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    draw.draw_voxels(ax2, voxels_remaining)

    best_model.position.data += 0.5
    if isinstance(best_model, BoxModel):
        draw.draw_cube(ax2, best_model)
    elif isinstance(best_model, SphereModel):
        draw.draw_sphere(ax2, best_model)
    elif isinstance(best_model, CylinderModel):
        draw.draw_cylinder(ax2, best_model)
    best_model.position.data -= 0.5

    plt.show()
    """

    points_inside_mask = best_model.exact_containment(points)
    best_model.lambda_ = 10.0
    points_inside_mask |= (best_model.containment(points) >= 0.5)
    points_outside_mask = ~points_inside_mask

    print(best_model)
    print("Number of points exactly inside: " + str(points_inside_mask.sum()))
    print("Number of points exactly outside: " + str(points_outside_mask.sum()))

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

print("Final models")
for m in fitted_models:
    print(str(m))

#ax.set_aspect("equal")

fig = plt.figure()
ax = fig.gca(projection='3d')
draw.draw_voxels(ax, voxel_grid)

for m in fitted_models:
    if isinstance(m, BoxModel):
        m.position.data += 0.5
        draw.draw_cube(ax, m)
    elif isinstance(m, SphereModel):
        m.position.data += 0.5
        draw.draw_sphere(ax, m)
    elif isinstance(m, CylinderModel):
        m.position.data += 0.5
        draw.draw_cylinder(ax, m)

plt.show()