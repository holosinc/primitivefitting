import torch
import itertools

def valid_coord(voxel_grid, voxel):
    return voxel[0] >= 0 and voxel[0] < voxel_grid.shape[0] and voxel[1] >= 0 and voxel[1] < voxel_grid.shape[1] and voxel[2] >= 0 and voxel[2] < voxel_grid.shape[2]

def voxel_neighbors(voxel):
    (x, y, z) = voxel
    return [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)]

def connected_components(voxel_grid):
    visited = torch.zeros_like(voxel_grid, dtype=torch.bool)

    def explore_component(start_voxel):
        ret = []
        voxels_to_visit = [start_voxel]
        while len(voxels_to_visit) > 0:
            voxel = voxels_to_visit.pop()
            if valid_coord(voxel_grid, voxel) and not get(visited, voxel) and get(voxel_grid, voxel):
                ret.append(voxel)
                set(visited, voxel, True)
                for neighbor in voxel_neighbors(voxel):
                    voxels_to_visit.append(neighbor)
        return torch.tensor(ret, dtype=torch.LongTensor)

    components = []
    for voxel in itertools.product(range(voxel_grid.shape[0]), range(voxel_grid.shape[1]), range(voxel_grid.shape[2])):
        if not get(visited, voxel) and get(voxel_grid, voxel):
            components.append(explore_component(voxel))

    return components

# This function returns a n x 3 vector, where each row in the tensor is an index of a filled voxel position
def voxels_to_indices(voxel_grid):
    r1 = torch.arange(0, voxel_grid.shape[0]).unsqueeze(1).unsqueeze(2).expand_as(voxel_grid)
    idx0 = torch.masked_select(r1, voxel_grid)
    r2 = torch.arange(0, voxel_grid.shape[1]).unsqueeze(0).unsqueeze(2).expand_as(voxel_grid)
    idx1 = torch.masked_select(r2, voxel_grid)
    r3 = torch.arange(0, voxel_grid.shape[2]).unsqueeze(0).unsqueeze(1).expand_as(voxel_grid)
    idx2 = torch.masked_select(r3, voxel_grid)
    return torch.stack([idx0, idx1, idx2], dim=1)

def batch_set(voxel_grid, indices, value):
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = value

def get(voxel_grid, voxel):
    (x, y, z) = voxel
    return voxel_grid[x, y, z].item()

def set(voxel_grid, voxel, value):
    (x, y, z) = voxel
    voxel_grid[x, y, z] = value

# Returns a new voxel grid with all holes in the voxel grid filled
def fill_holes(voxel_grid):
    visited = torch.zeros_like(voxel_grid, dtype=torch.bool)
    ret = torch.ones_like(voxel_grid, dtype=torch.bool)

    voxels_to_visit = []

    # Iterate over all 6 sides and insert flood fill seeds at locations that do not have voxels
    for coord in itertools.product([0, voxel_grid.shape[0] - 1], range(0, voxel_grid.shape[1]), range(0, voxel_grid.shape[2])):
        if not get(voxel_grid, coord):
            voxels_to_visit.append(coord)

    for coord in itertools.product(range(0, voxel_grid.shape[0]), [0, voxel_grid.shape[1] - 1], range(0, voxel_grid.shape[2])):
        if not get(voxel_grid, coord):
            voxels_to_visit.append(coord)

    for coord in itertools.product(range(0, voxel_grid.shape[0]), range(0, voxel_grid.shape[1]), [0, voxel_grid.shape[2] - 1]):
        if not get(voxel_grid, coord):
            voxels_to_visit.append(coord)

    while len(voxels_to_visit) > 0:
        voxel = voxels_to_visit.pop()
        if valid_coord(voxel_grid, voxel) and not get(visited, voxel):
            set(visited, voxel, True)
            if not get(voxel_grid, voxel):
                set(ret, voxel, False)
                for neighbor in voxel_neighbors(voxel):
                    voxels_to_visit.append(neighbor)

    return ret
