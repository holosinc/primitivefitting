from itertools import product, combinations
import numpy as np
import torch

def draw_cube(ax, model):
    # draw cube
    r = [-1.0, 1.0]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            points = list(zip(s, e))
            points_torch = torch.tensor(points, dtype=torch.float).t()
            transformed_points = model.transform(points_torch).detach().t().numpy()
            ax.plot3D(transformed_points[0], transformed_points[1], transformed_points[2], color="r")

def draw_voxels(ax, voxels):
    ax.voxels(filled=voxels.numpy())