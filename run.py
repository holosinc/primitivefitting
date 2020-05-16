import torch
import fit
import matplotlib.pyplot as plt
import draw
from models import LossType
import os.path

point_cloud_file = None
voxel_size = None
while point_cloud_file is None:
    val = input("Which model would you like to fit primitives to? (Input a number)\n0. Deer\n1. Bunny\n2. Monkey \n3. Tram\n4. Dragon\n5. Custom\n")
    if val == "0":
        point_cloud_file = "examples/deerP25.txt"
        voxel_size = 0.25
    elif val == "1":
        point_cloud_file = "examples/bunnyP00625.txt"
        voxel_size = 0.00625
    elif val == "2":
        point_cloud_file = "examples/monkeyP05.txt"
        voxel_size = 0.05
    elif val == "3":
        point_cloud_file = "examples/tram2P0.txt"
        voxel_size = 2.0
    elif val == "4":
        point_cloud_file = "examples/dragonP05.txt"
        voxel_size = 0.05
    elif val == "5":
        while point_cloud_file is None:
            point_cloud_file = input("Enter the path to the point text file\n")
            if os.path.isfile(point_cloud_file):
                while voxel_size is None:
                    voxel_size_str = input("Enter the size of a voxel for this point data\n")
                    try:
                        voxel_size = float(voxel_size_str)
                    except ValueError:
                        print("Invalid number entered")
            else:
                point_cloud_file = None
                print("This path is not a file")
    else:
        print("Invalid model selected. Type a number to continue")

max_num_fitted_models = None
while max_num_fitted_models is None:
    val = input("How many number of primitives do you want to fit to the model? (Input a number)\n")
    try:
        val = int(val)
        if val >= 1:
            max_num_fitted_models = val
        else:
            print("Invalid number entered")
    except ValueError:
        print("Invalid number entered")

def use_prompt(primitive_name):
    while True:
        val = input("Would you like to use " + primitive_name + " when fitting the model? (y/n)\n")
        if val == "y" or val == "yes":
            return True
        elif val == "n" or val == "no":
            return False
        else:
            print("Invalid value entered")

use_boxes = use_prompt("boxes")
use_spheres = use_prompt("spheres")
use_cylinders = use_prompt("cylinders")

loss_type = None
while loss_type is None:
    val = input("What loss type would you like to use? (Input a number)\n0. Best effort\n1. Best match\n")
    if val == "0":
        loss_type = LossType.BEST_EFFORT
    elif val == "1":
        loss_type = LossType.BEST_MATCH
    else:
        print("Invalid value entered")

visualize_intermediate = None
while visualize_intermediate is None:
    val = input("Would you like to visualize intermediate results? (y/n)\n")
    if val == "y" or val == "yes":
        visualize_intermediate = True
    elif val == "n" or val == "no":
        visualize_intermediate = False
    else:
        print("Invalid value entered")

print("Reading point file")

with open(point_cloud_file, 'r') as f:
    point_strs = f.readlines()

# Filter out empty lines
point_strs = [line.strip() for line in point_strs]
point_strs = [line for line in point_strs if line != ""]

points = torch.tensor([list(map(float, line.split())) for line in point_strs])

print("Points processed. Converting points to voxel grid")
(voxel_grid, _) = fit.points_to_voxel_grid(points, voxel_size)
print("Voxel grid created. Fitting primitives to voxel grid")
fitted_models = fit.fit_voxel_grid(voxel_grid, max_num_fitted_models=max_num_fitted_models, use_boxes=use_boxes, use_spheres=use_spheres,
                                   use_cylinders=use_cylinders, loss_type=loss_type, visualize_intermediate=visualize_intermediate)

print("Primitive fitting complete")

torch.set_printoptions(profile="full")

print("Final models")
for m in fitted_models:
    print(str(m))

print("Visualizing result")

fig = plt.figure()
ax = fig.gca(projection='3d')
draw.draw_voxels(ax, voxel_grid)
for m in fitted_models:
    m.draw(ax)
plt.show()