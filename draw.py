import numpy as np

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