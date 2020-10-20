from enum import Enum
import torch
import torch.nn as nn
import math
from torchext import clamp01, differentiable_leq_one, differentiable_geq_neg_one, numerically_stable_sigmoid
from three_d import identity_quaternion, scale, invert_rotation, translate, invert_translate, rotate
import draw

class LossType(Enum):
    BEST_EFFORT = 0
    BEST_MATCH = 1

class ContainmentType(Enum):
    DEFAULT = 0
    EXACT = 1
    FUZZY = 2

class PrimitiveModel(nn.Module):
    def __init__(self, init_points):
        super().__init__()
        self.position = nn.Parameter(torch.randn((3,)))
        self.rotation = nn.Parameter(torch.randn((4,)))
        self.inverse_scale = nn.Parameter(torch.randn((3,)))

        inside_point_center = torch.mean(init_points, dim=0)
        max_distance_from_center = torch.max(torch.abs(init_points - (inside_point_center.unsqueeze(0))), dim=0).values

        self.position.data[0] = inside_point_center[0].item()
        self.position.data[1] = inside_point_center[1].item()
        self.position.data[2] = inside_point_center[2].item()

        self.inverse_scale.data[0] = 1.0 / max(max_distance_from_center[0].item(), 0.5)
        self.inverse_scale.data[1] = 1.0 / max(max_distance_from_center[1].item(), 0.5)
        self.inverse_scale.data[2] = 1.0 / max(max_distance_from_center[2].item(), 0.5)

        self.rotation.data[0] = identity_quaternion[0].item()
        self.rotation.data[1] = identity_quaternion[1].item()
        self.rotation.data[2] = identity_quaternion[2].item()
        self.rotation.data[3] = identity_quaternion[3].item()

        self.ranges = {
            "position": (torch.min(init_points, dim=0).values, torch.max(init_points, dim=0).values),
            "rotation": (torch.tensor([-1.0, -1.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0, 1.0])),
            "inverse_scale": (self.inverse_scale.clone(), torch.tensor([2.0, 2.0, 2.0]))
        }

    def get_device(self):
        if self.position.is_cuda:
            return self.position.get_device()
        else:
            return None

    def get_scale(self):
        return 1.0 / self.inverse_scale

    def compute_loss(self, points, containment_type=ContainmentType.DEFAULT, loss_type=LossType.BEST_EFFORT):
        # Assume that the voxels have a volume of 1
        if containment_type == ContainmentType.DEFAULT:
            num_contained_voxels = self.count_containment(points)
        elif containment_type == ContainmentType.EXACT:
            num_contained_voxels = self.count_exact_containment(points).detach()
        elif containment_type == ContainmentType.FUZZY:
            num_contained_voxels = self.count_fuzzy_containment(points).detach()
        else:
            raise NotImplementedError("Containment type not implemented")
        volume = self.volume()
        if containment_type != ContainmentType.DEFAULT:
            volume = volume.detach()
        num_points = float(points.shape[0])
        if loss_type == LossType.BEST_EFFORT:
            best_effort_index = num_contained_voxels / volume
            best_effort_score = clamp01(best_effort_index)
            volume_bonus = numerically_stable_sigmoid((10.0 / num_points) * (volume - num_points / 2.0))
            return -(best_effort_score + volume_bonus)
        elif loss_type == LossType.BEST_MATCH:
            jaccard_index = num_contained_voxels / (volume + num_points - num_contained_voxels)
            return -jaccard_index

    def forward(self, points, loss_type=LossType.BEST_EFFORT):
        return self.compute_loss(points, containment_type=ContainmentType.DEFAULT, loss_type=loss_type)

    def exact_forward(self, points, loss_type=LossType.BEST_EFFORT):
        return float(self.compute_loss(points, containment_type=ContainmentType.EXACT, loss_type=loss_type))

    def fuzzy_forward(self, points, loss_type=LossType.BEST_EFFORT):
        return float(self.compute_loss(points, containment_type=ContainmentType.FUZZY, loss_type=loss_type))

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

    def volume(self):
        raise NotImplementedError("Volume not implemented")

    def count_containment(self, points):
        return self.containment(points).sum()

    def exact_containment(self, points):
        raise NotImplementedError("Exact containment not implemented")

    def fuzzy_containment(self, points, lambda_=10.0, threshold=0.5):
        points_inside_mask = self.exact_containment(points)
        prev_lambda = self.lambda_
        self.lambda_ = lambda_
        points_inside_mask |= (self.containment(points) >= threshold)
        self.lambda_ = prev_lambda
        return points_inside_mask

    def count_fuzzy_containment(self, points, lambda_=10.0, threshold=0.5):
        return self.fuzzy_containment(points, lambda_=lambda_, threshold=threshold).sum()

    def count_exact_containment(self, points):
        return self.exact_containment(points).sum()

    def containment(self, points):
        raise NotImplementedError("Containment not implemented")

    def abs_scale(self):
        self.inverse_scale.data.abs_()

    def __str__(self):
        return "Position: " + str(self.position) + "\nRotation: " + str(self.rotation) + "\nScale: " + str(self.get_scale())

    def draw(self, ax):
        raise NotImplementedError("Draw not implemented")

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

    def draw(self, ax):
        # Move over 0.5 since integer coordinates in the model represent center of voxels,
        # but when drawing integer coordinates are voxel corners
        self.position.data += 0.5
        draw.draw_sphere(ax, self)
        self.position.data -= 0.5

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
        return (face1 * face2 * face3 * face4 * face5 * face6 + 0.00001).pow(1.0 / 6.0)

    def volume(self):
        scale = self.get_scale()
        return 8.0 * scale.prod()

    def __str__(self):
        return "Box Model\n" + super().__str__()

    def draw(self, ax):
        self.position.data += 0.5
        draw.draw_cube(ax, self)
        self.position.data -= 0.5

class CylinderModel(PrimitiveModel):
    def __init__(self, init_points, lambda_):
        super().__init__(init_points)
        self.lambda_ = lambda_

    def exact_containment(self, points):
        transformed_points = self.inverse_transform(points)
        face_top = transformed_points[:, 1] <= 1.0
        face_bottom = transformed_points[:, 1] >= -1.0
        points_xz = transformed_points[:, [0, 2]]
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
        return (face_top * face_bottom * curved_side + 0.00001).pow(1.0 / 3.0)
        #return torch.min(torch.min(face_top, face_bottom), curved_side)

    def __str__(self):
        return "Cylinder Model\n" + super().__str__()

    def volume(self):
        scale = self.get_scale()
        return 2.0 * math.pi * scale.prod()

    def draw(self, ax):
        self.position.data += 0.5
        draw.draw_cylinder(ax, self)
        self.position.data -= 0.5