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
            if valid_coord(voxel_grid, voxel) and not get_voxel(visited, voxel) and get_voxel(voxel_grid, voxel):
                ret.append(voxel)
                set_voxel(visited, voxel, True)
                for neighbor in voxel_neighbors(voxel):
                    voxels_to_visit.append(neighbor)
        return torch.tensor(ret, dtype=torch.long)

    components = []
    for voxel in itertools.product(range(voxel_grid.shape[0]), range(voxel_grid.shape[1]), range(voxel_grid.shape[2])):
        if not get_voxel(visited, voxel) and get_voxel(voxel_grid, voxel):
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

def get_voxel(voxel_grid, voxel):
    (x, y, z) = voxel
    return voxel_grid[x, y, z].item()

def set_voxel(voxel_grid, voxel, value):
    (x, y, z) = voxel
    voxel_grid[x, y, z] = value

def batch_valid_coord(voxel_grid, voxels):
    x = voxels[:, 0]
    y = voxels[:, 1]
    z = voxels[:, 2]
    return (x >= 0) & (x < voxel_grid.shape[0]) & (y >= 0) & (y < voxel_grid.shape[1]) & (z >= 0) & (z < voxel_grid.shape[2])

def batch_neighbors(voxels):
    return torch.cat([voxels + torch.tensor([1, 0, 0], dtype=torch.long),
                      voxels + torch.tensor([-1, 0, 0], dtype=torch.long),
                      voxels + torch.tensor([0, 1, 0], dtype=torch.long),
                      voxels + torch.tensor([0, -1, 0], dtype=torch.long),
                      voxels + torch.tensor([0, 0, 1], dtype=torch.long),
                      voxels + torch.tensor([0, 0, -1], dtype=torch.long)])

def fill_holes(voxel_grid):
    visited = torch.zeros_like(voxel_grid, dtype=torch.bool)
    ret = torch.ones_like(voxel_grid, dtype=torch.bool)

    voxels_to_visit_set = set()

    # Iterate over all 6 sides and insert flood fill seeds at locations that do not have voxels
    for coord in itertools.product([0, voxel_grid.shape[0] - 1], range(0, voxel_grid.shape[1]), range(0, voxel_grid.shape[2])):
        if not get_voxel(voxel_grid, coord):
            voxels_to_visit_set.add(coord)

    for coord in itertools.product(range(0, voxel_grid.shape[0]), [0, voxel_grid.shape[1] - 1], range(0, voxel_grid.shape[2])):
        if not get_voxel(voxel_grid, coord):
            voxels_to_visit_set.add(coord)

    for coord in itertools.product(range(0, voxel_grid.shape[0]), range(0, voxel_grid.shape[1]), [0, voxel_grid.shape[2] - 1]):
        if not get_voxel(voxel_grid, coord):
            voxels_to_visit_set.add(coord)

    voxels_to_visit = torch.tensor(list(voxels_to_visit_set), dtype=torch.long)

    while voxels_to_visit.shape[0] > 0:
        voxels_to_visit = voxels_to_visit[batch_valid_coord(voxel_grid, voxels_to_visit)]
        voxels_to_visit = voxels_to_visit[~(visited[voxels_to_visit[:, 0], voxels_to_visit[:, 1], voxels_to_visit[:, 2]])]
        visited[voxels_to_visit[:, 0], voxels_to_visit[:, 1], voxels_to_visit[:, 2]] = True
        voxels_to_visit = voxels_to_visit[~(voxel_grid[voxels_to_visit[:, 0], voxels_to_visit[:, 1], voxels_to_visit[:, 2]])]
        ret[voxels_to_visit[:, 0], voxels_to_visit[:, 1], voxels_to_visit[:, 2]] = False
        voxels_to_visit = torch.unique(voxels_to_visit, dim=0)
        if voxels_to_visit.shape[0] > 0:
            voxels_to_visit = batch_neighbors(voxels_to_visit)

    return ret

# Returns a new voxel grid with all holes in the voxel grid filled
def fill_holes_old(voxel_grid):
    visited = torch.zeros_like(voxel_grid, dtype=torch.bool)
    ret = torch.ones_like(voxel_grid, dtype=torch.bool)

    voxels_to_visit = set()

    # Iterate over all 6 sides and insert flood fill seeds at locations that do not have voxels
    for coord in itertools.product([0, voxel_grid.shape[0] - 1], range(0, voxel_grid.shape[1]), range(0, voxel_grid.shape[2])):
        if not get_voxel(voxel_grid, coord):
            voxels_to_visit.add(coord)

    for coord in itertools.product(range(0, voxel_grid.shape[0]), [0, voxel_grid.shape[1] - 1], range(0, voxel_grid.shape[2])):
        if not get_voxel(voxel_grid, coord):
            voxels_to_visit.add(coord)

    for coord in itertools.product(range(0, voxel_grid.shape[0]), range(0, voxel_grid.shape[1]), [0, voxel_grid.shape[2] - 1]):
        if not get_voxel(voxel_grid, coord):
            voxels_to_visit.add(coord)

    while len(voxels_to_visit) > 0:
        voxel = voxels_to_visit.pop()
        if valid_coord(voxel_grid, voxel) and not get_voxel(visited, voxel):
            set_voxel(visited, voxel, True)
            if not get_voxel(voxel_grid, voxel):
                set_voxel(ret, voxel, False)
                for neighbor in voxel_neighbors(voxel):
                    voxels_to_visit.add(neighbor)

    return ret
