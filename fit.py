import torch
import torch.nn as nn
import torch.optim
import sklearn.cluster
import matplotlib.pyplot as plt
import draw
import math

point_cloud_file = "C:\\Users\\Caleb Helbling\\Documents\\holosproject\\voxelizervs\Debug\\mesh_voxelized_res.txt"
voxel_size = 2.0

with open(point_cloud_file, 'r') as f:
    point_strs = f.readlines()

# Filter out empty lines
point_strs = [line.strip() for line in point_strs]
point_strs = [line for line in point_strs if line != ""]

points = torch.tensor([list(map(float, line.split())) for line in point_strs])

integer_points = torch.round(points / voxel_size).long()
mins = torch.min(integer_points, dim=0).values
maxes = torch.max(integer_points, dim=0).values

original_size = (maxes - mins) + 1
# Give some padding around the object by multiplying the size by 2
size = original_size + 10

offset = -mins.unsqueeze(0) + (size - original_size) / 2

integer_points = integer_points + offset

voxel_grid = torch.zeros(tuple(size), dtype=torch.bool)

# Assign the point cloud to the new voxel grid
voxel_grid[integer_points[:, 0], integer_points[:, 1], integer_points[:, 2]] = 1

# Flood fill from the outside
# The result will be that any voxels not reachable from the inside will be set to 1 (filled)
visited_grid = torch.zeros_like(voxel_grid, dtype=torch.bool)
flood_filled_grid = torch.ones_like(voxel_grid, dtype=torch.bool)
voxels_to_visit = [[0, 0, 0]]
while len(voxels_to_visit) > 0:
    visited = voxels_to_visit.pop()
    if visited[0] >= 0 and visited[0] < size[0].item() and visited[1] >= 0 and visited[1] < size[1].item() and visited[2] >= 0 and visited[2] < size[2].item() and not visited_grid[visited[0], visited[1], visited[2]].item():
        visited_grid[visited[0], visited[1], visited[2]] = True
        if not voxel_grid[visited[0], visited[1], visited[2]].item():
            flood_filled_grid[visited[0], visited[1], visited[2]] = False
            voxels_to_visit.append([visited[0] + 1, visited[1], visited[2]])
            voxels_to_visit.append([visited[0] - 1, visited[1], visited[2]])
            voxels_to_visit.append([visited[0], visited[1] + 1, visited[2]])
            voxels_to_visit.append([visited[0], visited[1] - 1, visited[2]])
            voxels_to_visit.append([visited[0], visited[1], visited[2] + 1])
            voxels_to_visit.append([visited[0], visited[1], visited[2] - 1])

voxel_grid = flood_filled_grid

fig = plt.figure()
ax = fig.gca(projection='3d')

draw.draw_voxels(ax, voxel_grid)

# Now that we have a filled 3D object, convert the voxel grid back to point form for the rest of the algorithm

# This function returns a n x 3 vector, where each row in the tensor is an index of a filled voxel position
def voxels_to_indices(voxel_grid):
    r1 = torch.arange(0, voxel_grid.shape[0]).unsqueeze(1).unsqueeze(2).expand_as(voxel_grid)
    idx0 = torch.masked_select(r1, voxel_grid)
    r2 = torch.arange(0, voxel_grid.shape[1]).unsqueeze(0).unsqueeze(2).expand_as(voxel_grid)
    idx1 = torch.masked_select(r2, voxel_grid)
    r3 = torch.arange(0, voxel_grid.shape[2]).unsqueeze(0).unsqueeze(1).expand_as(voxel_grid)
    idx2 = torch.masked_select(r3, voxel_grid)
    return torch.stack([idx0, idx1, idx2], dim=1)

# From https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
def quaternion_to_rotation_matrix(quaternion):
    qr = quaternion[0]
    qi = quaternion[1]
    qj = quaternion[2]
    qk = quaternion[3]
    s = 1.0 / quaternion.pow(2).sum()
    row1 = torch.cat([(1.0 - 2.0 * s * (qj.pow(2) + qk.pow(2))).unsqueeze(0), (2.0 * s * (qi * qj - qk * qr)).unsqueeze(0), (2.0 * s * (qi * qk + qj * qr)).unsqueeze(0)])
    row2 = torch.cat([(2.0 * s * (qi * qj + qk * qr)).unsqueeze(0), (1.0 - 2.0 * s * (qi.pow(2) * qk.pow(2))).unsqueeze(0), (2.0 * s * (qj * qk - qi * qr)).unsqueeze(0)])
    row3 = torch.cat([(2.0 * s * (qi * qk - qj * qr)).unsqueeze(0), (2.0 * s * (qj * qk + qi * qr)).unsqueeze(0), (1.0 - 2.0 * s * (qi.pow(2) + qj.pow(2))).unsqueeze(0)])
    return torch.stack([row1, row2, row3])

identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])
identity_matrix = torch.eye(3)
assert(torch.abs(identity_matrix - quaternion_to_rotation_matrix(identity_quaternion)).sum().item() == 0.0)

# See https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
def numerically_stable_sigmoid(x):
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    z = torch.zeros_like(x)
    z[pos_mask] = torch.exp(-x[pos_mask])
    z[neg_mask] = torch.exp(x[neg_mask])
    top = torch.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1.0 + z)

def invert_rotation_matrix(rot_matrix):
    # The inverse of a rotation matrix is merely the transpose
    return rot_matrix.t()

def differentiable_leq_one(x, lambda_=1.0):
    return numerically_stable_sigmoid(lambda_ * (1.0 - x))

def differentiable_geq_neg_one(x, lambda_=1.0):
    return numerically_stable_sigmoid(lambda_ * (x + 1.0))

def invert_rotation(quaternion, points):
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    inverse_rotation_matrix = invert_rotation_matrix(rotation_matrix)
    return inverse_rotation_matrix.unsqueeze(0).matmul(points.unsqueeze(2)).squeeze(2)

def rotate(quaternion, points):
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    return rotation_matrix.unsqueeze(0).matmul(points.unsqueeze(2)).squeeze(2)

def translate(position, points):
    return points + position.unsqueeze(0)

def invert_translate(position, points):
    return points - position.unsqueeze(0)

def invert_scale(scale, points):
    return scale.unsqueeze(0) * points

def scale(scale_, points):
    return scale_.unsqueeze(0) * points

def norm2(m, dim=0):
    return m.pow(2).sum(dim)

def clamp01(x):
    return numerically_stable_sigmoid(10.0 * (x - 0.5))

class PrimitiveModel(nn.Module):
    def __init__(self, init_points, randomize=True):
        super().__init__()
        self.position = nn.Parameter(torch.randn((3,)))
        self.rotation = nn.Parameter(torch.randn((4,)))
        self.inverse_scale = nn.Parameter(torch.randn((3,)))

        inside_point_center = torch.mean(init_points, dim=0)
        max_distance_from_center = torch.max(torch.norm(init_points - (inside_point_center.unsqueeze(0)), dim=1))

        self.position.data[0] = inside_point_center[0].item()
        self.position.data[1] = inside_point_center[1].item()
        self.position.data[2] = inside_point_center[2].item()

        #avg_distance_from_center = torch.mean(torch.norm(init_points - (inside_point_center.unsqueeze(0)), dim=1))
        #self.inverse_scale.data[0] = 1.0 / avg_distance_from_center
        #self.inverse_scale.data[1] = 1.0 / avg_distance_from_center
        #self.inverse_scale.data[2] = 1.0 / avg_distance_from_center

        self.inverse_scale.data[0] = 1.0 / max_distance_from_center
        self.inverse_scale.data[1] = 1.0 / max_distance_from_center
        self.inverse_scale.data[2] = 1.0 / max_distance_from_center

        self.rotation.data[0] = identity_quaternion[0].item()
        self.rotation.data[1] = identity_quaternion[1].item()
        self.rotation.data[2] = identity_quaternion[2].item()
        self.rotation.data[3] = identity_quaternion[3].item()

    def get_scale(self):
        return 1.0 / self.inverse_scale

    def forward(self, points):
        # Assume that the voxels have a volume of 1
        num_contained_voxels = self.count_containment(points)
        jaccard_index = num_contained_voxels / self.volume()
        return clamp01(jaccard_index)

    def inverse_transform(self, points):
        return scale(self.inverse_scale, invert_rotation(self.rotation, invert_translate(self.position, points)))

    def transform(self, points):
        return translate(self.position, rotate(self.rotation, scale(self.get_scale(), points)))

    def normalize_rotation(self):
        n = torch.norm(self.rotation).item()
        if n == 0.0:
            self.rotation.data[0] = identity_quaternion[0].item()
            self.rotation.data[1] = identity_quaternion[1].item()
            self.rotation.data[2] = identity_quaternion[2].item()
            self.rotation.data[3] = identity_quaternion[3].item()
        else:
            self.rotation.data[0] /= n
            self.rotation.data[1] /= n
            self.rotation.data[2] /= n
            self.rotation.data[3] /= n

    def clamp_inverse_scale(self, value):
        self.inverse_scale.data = torch.clamp(self.inverse_scale.data, -value, value)

    def volume(self):
        raise NotImplementedError("Volume not implemented")

    def count_containment(self, points):
        return self.containment(points).sum()

    def exact_containment(self, points):
        raise NotImplementedError("Exact containment not implemented")

    def containment(self, points):
        raise NotImplementedError("Containment not implemented")

    def abs_scale(self):
        self.inverse_scale.data.abs_()

    def __str__(self):
        return "Position: " + str(self.position) + "\nRotation: " + str(self.rotation) + "\nScale: " + str(self.get_scale())

class SphereModel(PrimitiveModel):
    def __init__(self, init_points, lambda_):
        super().__init__(init_points)
        self.lambda_ = lambda_

    def exact_containment(self, points):
        transformed_points = self.inverse_transform(points)
        distance_from_origin = torch.norm(transformed_points, dim=1)
        return distance_from_origin <= 1.0

    def containment(self, points):
        transformed_points = self.inverse_transform(points)
        distance_from_origin = torch.norm(transformed_points, dim=1)
        return differentiable_leq_one(distance_from_origin, lambda_=self.lambda_)

    def volume(self):
        scale = self.get_scale()
        return 4.0 / 3.0 * math.pi * scale.prod()

    def __str__(self):
        return "Sphere Model\n" + super().__str__()

class BoxModel(PrimitiveModel):
    def __init__(self, init_points, lambda_):
        super().__init__(init_points)
        self.lambda_ = lambda_

    def exact_containment(self, points):
        transformed_points = self.inverse_transform(points)
        face1 = transformed_points[:, 0] <= 1.0
        face2 = transformed_points[:, 0] >= -1.0
        face3 = transformed_points[:, 1] <= 1.0
        face4 = transformed_points[:, 1] >= -1.0
        face5 = transformed_points[:, 2] <= 1.0
        face6 = transformed_points[:, 2] >= -1.0
        return face1 & face2 & face3 & face4 & face5 & face6

    def containment(self, points):
        transformed_points = self.inverse_transform(points)
        face1 = differentiable_leq_one(transformed_points[:, 0], lambda_=self.lambda_)
        face2 = differentiable_geq_neg_one(transformed_points[:, 0], lambda_=self.lambda_)
        face3 = differentiable_leq_one(transformed_points[:, 1], lambda_=self.lambda_)
        face4 = differentiable_geq_neg_one(transformed_points[:, 1], lambda_=self.lambda_)
        face5 = differentiable_leq_one(transformed_points[:, 2], lambda_=self.lambda_)
        face6 = differentiable_geq_neg_one(transformed_points[:, 2], lambda_=self.lambda_)
        #return (face1 * face2 * face3 * face4 * face5 * face6).pow(1.0 / 6.0)
        return face1 * face2 * face3 * face4 * face5 * face6

    def volume(self):
        scale = self.get_scale()
        return 8.0 * scale.prod()

    def __str__(self):
        return "Box Model\n" + super().__str__()

class CylinderModel(PrimitiveModel):
    def __init__(self, init_points, lambda_):
        super().__init__(init_points)
        self.lambda_ = lambda_

    def exact_containment(self, points):
        transformed_points = self.inverse_transform(points)
        face_top = transformed_points[:, 1] <= 1.0
        face_bottom = transformed_points[:, 1] >= -1.0
        points_xz = points[:, [0, 2]]
        distance_from_axis = torch.norm(points_xz, dim=1)
        curved_side = distance_from_axis <= 1.0
        return face_top & face_bottom & curved_side

    def containment(self, points):
        transformed_points = self.inverse_transform(points)
        face_top = differentiable_leq_one(transformed_points[:, 1], lambda_=self.lambda_)
        face_bottom = differentiable_geq_neg_one(transformed_points[:, 1], lambda_=self.lambda_)
        points_xz = transformed_points[:, [0, 2]]
        distance_from_axis = torch.norm(points_xz, dim=1)
        curved_side = differentiable_leq_one(distance_from_axis, lambda_=self.lambda_)
        #return (face_top * face_bottom * curved_side).pow(1.0 / 3.0)
        return face_top * face_bottom * curved_side

    def __str__(self):
        return "Cylinder Model\n" + super().__str__()

    def volume(self):
        scale = self.get_scale()
        return 2.0 * math.pi * scale.prod()

torch.set_printoptions(profile="full")

inside_points = voxels_to_indices(voxel_grid).float()
outside_points = voxels_to_indices(~voxel_grid).float()

#model = SphereModel(inside_points, 0.1)
#model = BoxModel(inside_points, 1.0)
#model = CylinderModel(inside_points, 1.0)

def map_range(x, x0, x1, y0, y1):
    return y0 + (x - x0) * ((y1 - y0) / (x1 - x0))

def optimize(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    model.train()

    prev_loss = None

    num_steps = 250

    #with torch.autograd.detect_anomaly():
    for i in range(num_steps):
        model.lambda_ = map_range(i, 0, num_steps - 1, 0.5, 8.0)

        optimizer.zero_grad()

        num_points = inside_points.shape[0]
        vol = model.volume()
        volume_bonus = numerically_stable_sigmoid((10.0 / num_points) * (vol - (num_points / 2.0)))

        loss = -model(inside_points) - volume_bonus
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        print("Pos", model.position, model.position.grad)
        print("Rot", model.rotation, model.rotation.grad)
        print("Sca", model.inverse_scale, model.inverse_scale.grad)

        optimizer.step()
        model.normalize_rotation()
        model.abs_scale()
        print("Loss:", loss)

        #if prev_loss is not None and abs(loss.item() - prev_loss) < 0.00001:
            #prev_loss = loss.item()
            #break

        prev_loss = loss.item()

    return prev_loss

    #return model.exact_containment(inside_points).sum() * inside_points_multiplier + model.exact_containment(outside_points).sum() * outside_points_multiplier

def argmin(lst, f):
    min_score = None
    min_elem = None
    for elem in lst:
        score = f(elem)
        if min_score is None or score < min_score:
            min_score = score
            min_elem = elem
    return min_elem

fitted_models = []

lambda_ = 5.0
for _ in range(4):
    potential_models = []

    #number_means = 4
    #kmeans = sklearn.cluster.KMeans(n_clusters=number_means)
    #mean_indices = torch.tensor(kmeans.fit_predict(inside_points.numpy()), dtype=torch.long)
    #for mean_idx in range(number_means):
        #mean_inside_points = inside_points[mean_indices == mean_idx, :]
        #potential_models.append(SphereModel(mean_inside_points, lambda_))
        #potential_models.append(BoxModel(mean_inside_points, lambda_))
        #potential_models.append(CylinderModel(mean_inside_points, lambda_))

    #best_model = argmin(potential_models, optimize)

    best_model = argmin([SphereModel(inside_points, lambda_), BoxModel(inside_points, lambda_), CylinderModel(inside_points, lambda_)], optimize)
    #best_model = argmin(potential_models, optimize)
    #best_model = SphereModel(inside_points, lambda_)

    #best_model = BoxModel(inside_points, lambda_)
    #best_model = SphereModel(inside_points, lambda_)
    #best_model = CylinderModel(inside_points, lambda_)
    #optimize(best_model)

    fitted_models.append(best_model)

    points_exactly_inside = best_model.exact_containment(inside_points)
    points_exactly_outside = ~points_exactly_inside

    print(best_model)
    print("Number of points exactly inside: " + str(points_exactly_inside.sum()))
    print("Number of points exactly outside: " + str(points_exactly_outside.sum()))

    #outside_points = torch.cat([outside_points, inside_points[points_exactly_inside, :]], dim=0)
    inside_points = inside_points[points_exactly_outside, :]

    #best_model.lambda_ = 0.25
    #points_approx_inside = best_model(inside_points) >= 0.4
    #points_approx_outside = ~points_approx_inside

    #print("Number of points approx inside: " + str(points_approx_inside.sum()))
    #print("Number of points approx outside: " + str(points_approx_outside.sum()))

    #outside_points = torch.cat([outside_points, inside_points[points_approx_inside, :]], dim=0)
    #inside_points = inside_points[points_approx_outside, :]

    if inside_points.shape[0] == 0:
        break

print("Final models")
for m in fitted_models:
    print(str(m))

#ax.set_aspect("equal")

for m in fitted_models:
    if isinstance(m, BoxModel):
        m.position.data += 0.5
        draw.draw_cube(ax, m)
    elif isinstance(m, SphereModel):
        m.position.data += 0.5
        draw.draw_sphere(ax, m)
    elif isinstance(m, CylinderModel):
        m.position.data += 0.5
        draw.draw_cylinder(ax, m)

plt.show()