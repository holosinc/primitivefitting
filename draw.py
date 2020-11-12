from itertools import product, combinations
import numpy as np
import torch

def draw_cube(ax, model):
    r = [-1.0, 1.0]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            points = list(zip(s, e))
            points_torch = torch.tensor(points, dtype=torch.float, device=model.get_device()).t()
            transformed_points = model.transform(points_torch).detach().cpu().t().numpy()
            ax.plot3D(transformed_points[0], transformed_points[1], transformed_points[2], color="r")

def draw_sphere(ax, model):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    device = model.get_device()
    x2 = torch.tensor(x, dtype=torch.float, device=device).view(-1)
    y2 = torch.tensor(y, dtype=torch.float, device=device).view(-1)
    z2 = torch.tensor(z, dtype=torch.float, device=device).view(-1)
    points_torch = torch.stack([x2, y2, z2]).t()
    transformed_points = model.transform(points_torch).detach().t()

    x3 = transformed_points[0].view(x.shape)
    y3 = transformed_points[1].view(y.shape)
    z3 = transformed_points[2].view(z.shape)

    ax.plot_wireframe(x3.cpu().numpy(), y3.cpu().numpy(), z3.cpu().numpy(), color="r")

def draw_cylinder(ax, model):
    y = np.linspace(-1.0, 1.0, 10)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, y_grid = np.meshgrid(theta, y)
    x_grid = np.cos(theta_grid)
    z_grid = np.sin(theta_grid)

    device = model.get_device()
    x2 = torch.tensor(x_grid, dtype=torch.float, device=device).view(-1)
    y2 = torch.tensor(y_grid, dtype=torch.float, device=device).view(-1)
    z2 = torch.tensor(z_grid, dtype=torch.float, device=device).view(-1)
    points_torch = torch.stack([x2, y2, z2]).t()
    transformed_points = model.transform(points_torch).detach().t()

    x3 = transformed_points[0].view(x_grid.shape)
    y3 = transformed_points[1].view(y_grid.shape)
    z3 = transformed_points[2].view(z_grid.shape)

    ax.plot_wireframe(x3.cpu().numpy(), y3.cpu().numpy(), z3.cpu().numpy(), color="r")

def equalize_aspect_ratio(ax, voxels):
    half_upper_bound = max(voxels.shape[0], voxels.shape[1], voxels.shape[2]) / 2.0
    center_x = voxels.shape[0] / 2.0
    center_y = voxels.shape[1] / 2.0
    center_z = voxels.shape[2] / 2.0
    ax.set_xlim(center_x - half_upper_bound, center_x + half_upper_bound)
    ax.set_ylim(center_y - half_upper_bound, center_y + half_upper_bound)
    ax.set_zlim(center_z - half_upper_bound, center_z + half_upper_bound)

def draw_voxels(ax, voxels, eq_aspect_ratio=True):
    ax.voxels(filled=voxels.numpy())
    if eq_aspect_ratio:
        equalize_aspect_ratio(ax, voxels)

def draw_line(ax, pa, pb, color="r"):
    points = np.stack([pa, pb]).transpose()
    ax.plot3D(points[0], points[1], points[2], color=color)