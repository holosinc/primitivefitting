from enum import Enum
import torch
import torch.nn as nn
import math
from torchext import clamp01, differentiable_leq_one, differentiable_geq_neg_one, numerically_stable_sigmoid
from three_d import identity_quaternion, scale, invert_rotation, translate, invert_translate, rotate

class LossType(Enum):
    BEST_EFFORT = 0
    BEST_MATCH = 1

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

    def compute_loss(self, points, exact=False, loss_type=LossType.BEST_EFFORT):
        # Assume that the voxels have a volume of 1
        if exact:
            num_contained_voxels = self.count_exact_containment(points).detach()
        else:
            num_contained_voxels = self.count_containment(points)
        volume = self.volume()
        if exact:
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
        return self.compute_loss(points, exact=False, loss_type=loss_type)

    def exact_forward(self, points, loss_type=LossType.BEST_EFFORT):
        return float(self.compute_loss(points, exact=True, loss_type=loss_type))

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

    def count_exact_containment(self, points):
        return self.exact_containment(points).sum()

    def containment(self, points):
        raise NotImplementedError("Containment not implemented")

    def abs_scale(self):
        self.inverse_scale.data.abs_()

    def max_scale(self, c):
        c_recip = 1.0 / c
        self.inverse_scale.data[0] = min(self.inverse_scale.data[0].item(), c_recip)
        self.inverse_scale.data[1] = min(self.inverse_scale.data[0].item(), c_recip)
        self.inverse_scale.data[2] = min(self.inverse_scale.data[0].item(), c_recip)

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