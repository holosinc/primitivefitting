import os
import torch
import fit
from models import LossType
import time
import json
import torch.nn as nn
import math
import voxel
import shutil

if os.path.exists("analysis"):
    answer = input("Analysis already exists. Do you want to delete the previous analysis and start over?")
    if answer != "yes" and answer != "y":
        exit(0)
    shutil.rmtree("analysis")

os.mkdir("analysis")

voxel_model_dir = r"C:\Users\Caleb Helbling\Documents\holosproject\polymodels\voxel"
max_resolution = "50.txt"

def listdirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def load_voxel_grid(file_name):
    with open(file_name, 'r') as f:
        point_strs = f.readlines()
    point_strs = [line.strip() for line in point_strs]
    dimensions = list(map(int, point_strs[0].split()))
    origin = list(map(float, point_strs[1].split()))
    scale = list(map(float, point_strs[2].split()))[0]
    sparse_points = torch.tensor([list(map(int, line.split())) for line in point_strs[3:]])
    grid = torch.zeros(dimensions, dtype=torch.bool)
    voxel.batch_set(grid, sparse_points, True)
    return (grid, scale, origin)

def analyze(point_file_path, max_resolution_file_path):
    (voxel_grid, voxel_size, offset) = load_voxel_grid(point_file_path)

    max_num_fitted_models = 10
    use_cuboids = True
    use_spheres = True
    use_capsules = True
    loss_type = LossType.BEST_MATCH
    visualize_intermediate = False
    use_cuda = False

    start_time = time.perf_counter()
    fitted_models = fit.fit_voxel_grid(voxel_grid, max_num_fitted_models=max_num_fitted_models, use_cuboid=use_cuboids,
                                       use_sphere=use_spheres,
                                       use_capsule=use_capsules, loss_type=loss_type,
                                       visualize_intermediate=visualize_intermediate,
                                       use_cuda=use_cuda)
    end_time = time.perf_counter()
    duration = end_time - start_time

    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    draw.draw_voxels(ax, voxel_grid)
    for m in fitted_models:
        m.draw(ax)
    plt.show()
    """

    (max_voxel_grid, max_voxel_size, max_offset) = load_voxel_grid(max_resolution_file_path)

    padding0 = math.ceil(max_voxel_grid.shape[0] / 2)
    padding1 = math.ceil(max_voxel_grid.shape[1] / 2)
    padding2 = math.ceil(max_voxel_grid.shape[2] / 2)

    # Add some padding so that any fitted primitives outside of the normal grid
    # will be taken into account
    max_voxel_grid = nn.functional.pad(max_voxel_grid, [padding2, padding2, padding1, padding1, padding0, padding0])

    for m in fitted_models:
        m.uniform_scale(voxel_size / max_voxel_size)
        m.translate(torch.tensor([padding0, padding1, padding2], dtype=torch.float))

    max_indices = torch.cartesian_prod(torch.arange(0, max_voxel_grid.shape[0]), torch.arange(0, max_voxel_grid.shape[1]),
                                       torch.arange(0, max_voxel_grid.shape[2]))
    max_indices_float = max_indices.float()

    covered_by_models = torch.zeros_like(max_voxel_grid, dtype=torch.bool)

    for m in fitted_models:
        covered = max_indices[m.exact_containment(max_indices_float)]
        voxel.batch_set(covered_by_models, covered, True)

    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    draw.draw_voxels(ax, covered_by_models)
    for m in fitted_models:
        m.draw(ax)
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    draw.draw_voxels(ax, max_voxel_grid)
    for m in fitted_models:
        m.draw(ax)
    plt.show()
    """

    print(fitted_models)

    overall_jaccard_index = float((max_voxel_grid & covered_by_models).sum()) / float((max_voxel_grid | covered_by_models).sum())
    print("Overall Jaccard Index: " + str(overall_jaccard_index))
    print("Done")
    return (overall_jaccard_index, duration, int(voxel_grid.sum()))

for model_name in listdirs(voxel_model_dir)[0:1]:
    analysis_results_path = os.path.join("analysis", model_name + ".json")
    analysis_results = {}
    for point_file_name in os.listdir(os.path.join(voxel_model_dir, model_name)):
        resolution_str = os.path.splitext(point_file_name)[0]
        max_resolution_file_path = os.path.join(voxel_model_dir, model_name, max_resolution)
        if point_file_name.endswith(".txt"):
            point_file_path = os.path.join(voxel_model_dir, model_name, point_file_name)
            (jaccard_index, duration, num_voxels) = analyze(point_file_path, max_resolution_file_path)
            analysis_results[resolution_str] = {"jaccard_index": jaccard_index, "duration": duration, "num_voxels": num_voxels}
    with open(analysis_results_path, 'w+') as f:
        json.dump(analysis_results, f)
