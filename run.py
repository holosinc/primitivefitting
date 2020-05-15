import torch
import fit
import matplotlib.pyplot as plt
import draw

point_cloud_file = "C:\\Users\\Caleb Helbling\\Documents\\holosproject\\voxelizervs\Debug\\bunnyP00625.txt"
voxel_size = 0.00625

#point_cloud_file = "C:\\Users\\Caleb Helbling\\Documents\\holosproject\\voxelizervs\Debug\\monkeyP05.txt"
#voxel_size = 0.05

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

(voxel_grid, _) = fit.points_to_voxel_grid(points, voxel_size)
fitted_models = fit.fit_voxel_grid(voxel_grid)

torch.set_printoptions(profile="full")

print("Final models")
for m in fitted_models:
    print(str(m))

#ax.set_aspect("equal")

fig = plt.figure()
ax = fig.gca(projection='3d')
draw.draw_voxels(ax, voxel_grid)
for m in fitted_models:
    m.draw(ax)
plt.show()