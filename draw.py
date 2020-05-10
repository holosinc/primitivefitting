from itertools import product, combinations
import numpy as np
import torch

def draw_cube(ax, model):
    r = [-1.0, 1.0]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            points = list(zip(s, e))
            points_torch = torch.tensor(points, dtype=torch.float).t()
            transformed_points = model.transform(points_torch).detach().t().numpy()
            ax.plot3D(transformed_points[0], transformed_points[1], transformed_points[2], color="r")

def draw_sphere(ax, model):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    x2 = torch.tensor(x, dtype=torch.float).view(-1)
    y2 = torch.tensor(y, dtype=torch.float).view(-1)
    z2 = torch.tensor(z, dtype=torch.float).view(-1)
    points_torch = torch.stack([x2, y2, z2]).t()
    transformed_points = model.transform(points_torch).detach().t()

    x3 = transformed_points[0].view(x.shape)
    y3 = transformed_points[1].view(y.shape)
    z3 = transformed_points[2].view(z.shape)

    ax.plot_wireframe(x3.numpy(), y3.numpy(), z3.numpy(), color="r")

def draw_cylinder(ax, model):
    y = np.linspace(-1.0, 1.0, 10)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, y_grid = np.meshgrid(theta, y)
    x_grid = np.cos(theta_grid)
    z_grid = np.sin(theta_grid)

    x2 = torch.tensor(x_grid, dtype=torch.float).view(-1)
    y2 = torch.tensor(y_grid, dtype=torch.float).view(-1)
    z2 = torch.tensor(z_grid, dtype=torch.float).view(-1)
    points_torch = torch.stack([x2, y2, z2]).t()
    transformed_points = model.transform(points_torch).detach().t()

    x3 = transformed_points[0].view(x_grid.shape)
    y3 = transformed_points[1].view(y_grid.shape)
    z3 = transformed_points[2].view(z_grid.shape)

    ax.plot_wireframe(x3.numpy(), y3.numpy(), z3.numpy(), color="r")

def draw_voxels(ax, voxels):
    ax.voxels(filled=voxels.numpy())