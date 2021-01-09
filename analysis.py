import os
import torch
import fit
from models import LossType
import time
import matplotlib.pyplot as plt
import draw

os.mkdir("analysis")

voxel_model_dir = r"C:\Users\Caleb Helbling\Documents\holosproject\polymodels\voxel"

def listdirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def analyze(point_file_path, analysis_results_path):
    with open(point_file_path, 'r') as f:
        point_strs = f.readlines()
    point_strs = [line.strip() for line in point_strs]
    voxel_size = float(point_strs[0])
    points = torch.tensor([list(map(float, line.split())) for line in point_strs[1:]])
    (voxel_grid, offset) = fit.points_to_voxel_grid(points, voxel_size)

    max_num_fitted_models = 10
    use_boxes = True
    use_spheres = True
    use_cylinders = True
    loss_type = LossType.BEST_MATCH
    visualize_intermediate = False
    fuzzy_containment = False
    use_cuda = False

    start_time = time.perf_counter()
    fitted_models = fit.fit_voxel_grid(voxel_grid, max_num_fitted_models=max_num_fitted_models, use_boxes=use_boxes,
                                       use_spheres=use_spheres,
                                       use_cylinders=use_cylinders, loss_type=loss_type,
                                       visualize_intermediate=visualize_intermediate,
                                       use_cuda=use_cuda)
    end_time = time.perf_counter()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    draw.draw_voxels(ax, voxel_grid)
    for m in fitted_models:
        m.draw(ax)
    plt.show()

    print(fitted_models)

for model_name in listdirs(voxel_model_dir):
    os.mkdir(os.path.join("analysis", model_name))
    for point_file_name in os.listdir(os.path.join(voxel_model_dir, model_name)):
        if point_file_name.endswith(".txt"):
            point_file_path = os.path.join(voxel_model_dir, model_name, point_file_name)
            analysis_results_path = os.path.join("analysis", model_name, point_file_name)
            analyze(point_file_path, analysis_results_path)