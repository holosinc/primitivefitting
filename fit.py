import torch
import torch.nn as nn
import torch.optim
import sklearn.cluster
import matplotlib.pyplot as plt
import draw
import math
import voxel

use_cuda = False

def to_device(x):
    if use_cuda:
        return x.cuda()
    else:
        return x

point_cloud_file = "C:\\Users\\Caleb Helbling\\Documents\\holosproject\\voxelizervs\Debug\\bunnyP00625.txt"
voxel_size = 0.00625

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
size = original_size

offset = -mins.unsqueeze(0)

integer_points = integer_points + offset

voxel_grid = torch.zeros(tuple(size), dtype=torch.bool)

# Assign the point cloud to the new voxel grid
voxel.batch_set(voxel_grid, integer_points, True)

voxel_grid = voxel.fill_holes(voxel_grid)

# Now that we have a filled 3D object, convert the voxel grid back to point form for the rest of the algorithm

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

def trough_curve(x, center, trough_size):
    return torch.exp(-(((x - center) / trough_size).pow(4.0)))

class PrimitiveModel(nn.Module):
    def __init__(self, init_points, randomize=True):
        super().__init__()
        self.position = nn.Parameter(torch.randn((3,)))
        self.rotation = nn.Parameter(torch.randn((4,)))
        self.inverse_scale = nn.Parameter(torch.randn((3,)))

        inside_point_center = torch.mean(init_points, dim=0)
        #max_distance_from_center = torch.max(torch.norm(init_points - (inside_point_center.unsqueeze(0)), dim=1))
        max_distance_from_center = torch.max(torch.abs(init_points - (inside_point_center.unsqueeze(0))), dim=0).values

        self.position.data[0] = inside_point_center[0].item()
        self.position.data[1] = inside_point_center[1].item()
        self.position.data[2] = inside_point_center[2].item()

        #avg_distance_from_center = torch.mean(torch.norm(init_points - (inside_point_center.unsqueeze(0)), dim=1))
        #self.inverse_scale.data[0] = 1.0 / avg_distance_from_center
        #self.inverse_scale.data[1] = 1.0 / avg_distance_from_center
        #self.inverse_scale.data[2] = 1.0 / avg_distance_from_center

        self.inverse_scale.data[0] = 1.0 / max(max_distance_from_center[0].item(), 0.5)
        self.inverse_scale.data[1] = 1.0 / max(max_distance_from_center[1].item(), 0.5)
        self.inverse_scale.data[2] = 1.0 / max(max_distance_from_center[2].item(), 0.5)

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
        return (face1 * face2 * face3 * face4 * face5 * face6).pow(1.0 / 6.0)
        #return face1 * face2 * face3 * face4 * face5 * face6

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
        return (face_top * face_bottom * curved_side).pow(1.0 / 3.0)
        #return face_top * face_bottom * curved_side

    def __str__(self):
        return "Cylinder Model\n" + super().__str__()

    def volume(self):
        scale = self.get_scale()
        return 2.0 * math.pi * scale.prod()

torch.set_printoptions(profile="full")

inside_points = voxel.voxels_to_indices(voxel_grid).float()
outside_points = voxel.voxels_to_indices(~voxel_grid).float()

def map_range(x, x0, x1, y0, y1):
    return y0 + (x - x0) * ((y1 - y0) / (x1 - x0))

def optimize(inside_points, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    prev_loss = None

    num_steps = 500

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: map_range(i, 0.0, num_steps - 1, 0.1, 0.005))

    #with torch.autograd.detect_anomaly():
    for i in range(num_steps):
        model.lambda_ = map_range(i, 0, num_steps - 1, 0.5, 8.0)

        optimizer.zero_grad()

        num_points = inside_points.shape[0]
        vol = model.volume()
        volume_bonus = numerically_stable_sigmoid((10.0 / num_points) * (vol - (num_points / 2.0)))
        #volume_bonus = numerically_stable_sigmoid(10.0 * (vol - 2.0))
        """
        trough_low = 0.5 * num_points
        trough_high = 1.5 * num_points
        trough_center = (trough_low + trough_high) / 2.0
        trough_size = trough_high - trough_low
        volume_bonus = trough_curve(vol, trough_center, trough_size)
        """

        loss = -model(inside_points) - volume_bonus
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        #print("Pos", model.position, model.position.grad)
        #print("Rot", model.rotation, model.rotation.grad)
        #print("Sca", model.inverse_scale, model.inverse_scale.grad)

        optimizer.step()
        model.normalize_rotation()
        model.abs_scale()
        #print("Loss:", loss)

        #if prev_loss is not None and abs(loss.item() - prev_loss) < 0.00001:
            #prev_loss = loss.item()
            #break

        prev_loss = loss.item()
        scheduler.step()


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

def argmax(lst, f):
    max_score = None
    max_elem = None
    for elem in lst:
        score = f(elem)
        if max_score is None or score > max_score:
            max_score = score
            max_elem = elem
    return max_elem

# Partial function application
def partial(f, *args, **kwargs):
    def ret(*more_args, **morekwargs):
        kwargs.update(morekwargs)
        return f(*(args + more_args), **kwargs)
    return ret

fitted_models = []

voxels_remaining = voxel_grid.clone()
connected_components = voxel.connected_components(voxels_remaining)
max_num_fitted_models = 15
i = 0

while len(connected_components) > 0 and i < max_num_fitted_models:
    print("Number of connected components: " + str(len(connected_components)))

    component = argmax(connected_components, lambda component: component.shape[0])

    points = component.float()

    lambda_ = 1.0
    #best_model = argmin([SphereModel(points, lambda_), BoxModel(points, lambda_), CylinderModel(points, lambda_)], partial(optimize, points))
    best_model = argmin([BoxModel(points, lambda_)], partial(optimize, points))

    """
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    draw.draw_voxels(ax2, voxels_remaining)

    best_model.position.data += 0.5
    if isinstance(best_model, BoxModel):
        draw.draw_cube(ax2, best_model)
    elif isinstance(best_model, SphereModel):
        draw.draw_sphere(ax2, best_model)
    elif isinstance(best_model, CylinderModel):
        draw.draw_cylinder(ax2, best_model)
    best_model.position.data -= 0.5

    plt.show()
    """

    points_inside_mask = best_model.exact_containment(points)
    best_model.lambda_ = 10.0
    points_inside_mask |= (best_model.containment(points) >= 0.6)
    points_outside_mask = ~points_inside_mask

    print(best_model)
    print("Number of points exactly inside: " + str(points_inside_mask.sum()))
    print("Number of points exactly outside: " + str(points_outside_mask.sum()))

    indices_covered = component[points_inside_mask, :]
    voxel.batch_set(voxels_remaining, indices_covered, False)

    if points_inside_mask.sum().item() > 0:
        fitted_models.append(best_model)
    else:
        # We failed to fit to any of the voxels in this component, so just ignore it
        # Todo: try splitting the component up and fit to those pieces
        voxel.batch_set(voxels_remaining, points_inside_mask, False)

    i += 1

    connected_components = voxel.connected_components(voxels_remaining)

"""
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

    #best_model = argmin([SphereModel(inside_points, lambda_), BoxModel(inside_points, lambda_), CylinderModel(inside_points, lambda_)], optimize)
    #best_model = argmin(potential_models, optimize)
    #best_model = SphereModel(inside_points, lambda_)

    best_model = BoxModel(inside_points, lambda_)
    #best_model = SphereModel(inside_points, lambda_)
    #best_model = CylinderModel(inside_points, lambda_)
    optimize(best_model)

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
"""

print("Final models")
for m in fitted_models:
    print(str(m))

#ax.set_aspect("equal")

fig = plt.figure()
ax = fig.gca(projection='3d')
draw.draw_voxels(ax, voxel_grid)

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