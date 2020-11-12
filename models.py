from enum import Enum
import torch
import torch.nn as nn
import math
import torchext
from torchext import clamp01, differentiable_leq_one, differentiable_geq_neg_one, numerically_stable_sigmoid,\
    differentiable_leq_c, differentiable_geq_c, inverse_softplus, get_device, perpendicular_vector
from three_d import identity_quaternion, scale, invert_rotation, translate, invert_translate, rotate
import draw
import numpy as np
from sklearn.cluster import KMeans

class LossType(Enum):
    BEST_EFFORT = 0
    BEST_MATCH = 1

class ContainmentType(Enum):
    DEFAULT = 0
    EXACT = 1
    FUZZY = 2

class PrimitiveModel(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

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

    def draw(self, ax):
        raise NotImplementedError("Draw not implemented")

    def normalize(self):
        pass

    def forward(self, points, loss_type=LossType.BEST_EFFORT):
        return self.compute_loss(points, containment_type=ContainmentType.DEFAULT, loss_type=loss_type)

    def exact_forward(self, points, loss_type=LossType.BEST_EFFORT):
        return float(self.compute_loss(points, containment_type=ContainmentType.EXACT, loss_type=loss_type))

    def fuzzy_forward(self, points, loss_type=LossType.BEST_EFFORT):
        return float(self.compute_loss(points, containment_type=ContainmentType.FUZZY, loss_type=loss_type))

class OptimizerPreference:
    def __init__(self, start_lr, end_lr, params):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.params = params

class CapsuleModel(PrimitiveModel):
    def __init__(self, init_points, lambda_):
        super(CapsuleModel, self).__init__(lambda_)

        self.p1 = nn.Parameter(torch.randn(3))
        self.p2 = nn.Parameter(torch.randn(3))
        self.radius_param = nn.Parameter(torch.randn(1))

        kmeans = KMeans(n_clusters=2).fit(init_points.numpy())
        cluster_a = torch.tensor(kmeans.cluster_centers_[0])
        cluster_b = torch.tensor(kmeans.cluster_centers_[1])

        n_hat = torchext.normalize(cluster_b - cluster_a)
        v = init_points - cluster_a.unsqueeze(0)
        v_projected = v @ n_hat
        v_horizontal = v - (v_projected.unsqueeze(1) * n_hat.unsqueeze(0))
        horizontal_dist = torch.norm(v_horizontal, 2, dim=1)

        mean_horizontal_dist = torch.mean(horizontal_dist)

        self.p1.data[0] = cluster_a[0].item()
        self.p1.data[1] = cluster_a[1].item()
        self.p1.data[2] = cluster_a[2].item()

        self.p2.data[0] = cluster_b[0].item()
        self.p2.data[1] = cluster_b[1].item()
        self.p2.data[2] = cluster_b[2].item()

        self.radius_param.data[0] = self.inverse_radius(mean_horizontal_dist).item()

        #inside_point_center = torch.mean(init_points, dim=0)

        #self.p1.data[0] = inside_point_center[0].item() - 1.0
        #self.p1.data[1] = inside_point_center[1].item()
        #self.p1.data[2] = inside_point_center[2].item()

        #self.p2.data[0] = inside_point_center[0].item() + 1.0
        #self.p2.data[1] = inside_point_center[1].item()
        #self.p2.data[2] = inside_point_center[2].item()

        #self.radius_param.data[0] = self.inverse_radius(torch.tensor(1.0)).item()

        self.optimizer_config = [OptimizerPreference(15.0, 7.0, [self.p1, self.p2]),
                                 OptimizerPreference(5.0, 2.0, [self.radius_param])]

    def radius(self):
        return torch.nn.functional.softplus(self.radius_param) + 0.5

    def inverse_radius(self, r):
        return inverse_softplus(r - 0.5)

    def containment(self, points):
        n = self.p2 - self.p1
        h = torch.norm(n, 2)
        if h.item() == 0.0:
            n_hat = n
        else:
            n_hat = n / h

        v = points - self.p1.unsqueeze(0)
        w = points - self.p2.unsqueeze(0)

        # Project each point onto the central line segment of the capsule to determine if
        # the point lies between the top and bottom cylinder caps
        v_projected = v @ n_hat

        bottom = differentiable_geq_c(v_projected, 0.0, lambda_=self.lambda_)
        top = differentiable_leq_c(v_projected, h, lambda_=self.lambda_)

        # Subtract off the component of v in the direction of the central line segment to determine
        # how far the point is from the central line horizontally
        v_horizontal = v - (v_projected.unsqueeze(1) * n_hat.unsqueeze(0))
        horizontal_dist = torch.norm(v_horizontal, 2, dim=1)
        radius = self.radius()
        # Is the point within radius distance from the center?
        side = differentiable_leq_c(horizontal_dist, radius, lambda_=self.lambda_)

        # Now test if the point is within the bottom sphere
        bottom_sphere = differentiable_leq_c(torch.norm(v, 2, dim=1), radius, lambda_=self.lambda_)
        top_sphere = differentiable_leq_c(torch.norm(w, 2, dim=1), radius, lambda_=self.lambda_)

        cylinder = top * bottom * side

        # A simple application of De Morgan's law, that is NOT ((NOT X) AND (NOT Y) AND (NOT Z)) == X OR Y OR Z
        capsule = 1.0 - ((1.0 - cylinder) * (1.0 - bottom_sphere) * (1.0 - top_sphere))

        return capsule

    def exact_containment(self, points):
        n = self.p2 - self.p1
        h = torch.norm(n, 2)
        if h.item() == 0.0:
            n_hat = n
        else:
            n_hat = n / h

        v = points - self.p1.unsqueeze(0)
        w = points - self.p2.unsqueeze(0)

        # Project each point onto the central line segment of the capsule to determine if
        # the point lies between the top and bottom cylinder caps
        v_projected = v @ n_hat

        bottom = v_projected >= 0.0
        top = v_projected <= h

        # Subtract off the component of v in the direction of the central line segment to determine
        # how far the point is from the central line horizontally
        v_horizontal = v - (v_projected.unsqueeze(1) * n_hat.unsqueeze(0))
        horizontal_dist = torch.norm(v_horizontal, 2, dim=1)
        radius = self.radius()
        # Is the point within radius distance from the center?
        side = horizontal_dist <= radius

        # Now test if the point is within the bottom sphere
        bottom_sphere = torch.norm(v, 2, dim=1) <= radius
        top_sphere = torch.norm(w, 2, dim=1) <= radius

        cylinder = top & bottom & side

        capsule = bottom_sphere | top_sphere | cylinder

        return capsule

    def volume(self):
        h = torch.norm(self.p2 - self.p1, 2)
        radius = self.radius()

        capsule_volume = h * math.pi * radius ** 2
        sphere_volume = (4.0 / 3.0) * math.pi * radius ** 3

        return capsule_volume + sphere_volume

    def draw(self, ax):
        radius = self.radius().item()

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)

        x = x * radius
        y = y * radius
        z = z * radius

        ax.plot_wireframe(x + self.p1.data[0].item(), y + self.p1.data[1].item(), z + self.p1.data[2].item(), color="g")
        ax.plot_wireframe(x + self.p2.data[0].item(), y + self.p2.data[1].item(), z + self.p2.data[2].item(), color="g")

        n = self.p2 - self.p1
        h = torch.norm(n, 2)
        if h.item() > 0.0:
            n_hat = n / h

            q = torchext.normalize(perpendicular_vector(n_hat))
            s = torchext.normalize(torch.cross(n_hat, q))

            p1 = self.p1.detach().numpy()
            p2 = self.p2.detach().numpy()

            q = (q * radius).detach().numpy()
            s = (s * radius).detach().numpy()

            for theta in np.mgrid[0:2 * np.pi:20j]:
                offset = np.sin(theta) * q + np.cos(theta) * s
                draw.draw_line(ax, p1 + offset, p2 + offset, color="g")


            """
            draw.draw_line(ax, (self.p1 + q).detach().numpy(), (self.p2 + q).detach().numpy())
            draw.draw_line(ax, (self.p1 - q).detach().numpy(), (self.p2 - q).detach().numpy())
            draw.draw_line(ax, (self.p1 + s).detach().numpy(), (self.p2 + s).detach().numpy())
            draw.draw_line(ax, (self.p1 - s).detach().numpy(), (self.p2 - s).detach().numpy())
            """

class CuboidModel(PrimitiveModel):
    def __init__(self, init_points, lambda_):
        super(CuboidModel, self).__init__(lambda_)
        # This models has 13 parameters, which is 4 more than the minimum for the cuboid
        # This is done out of the hope that the extra parameters will assist in keeping the cuboid
        # from getting stuck in local minima
        self.min_corner_param = nn.Parameter(torch.randn(3))
        self.max_corner_param = nn.Parameter(torch.randn(3))
        self.position = nn.Parameter(torch.randn(3))
        self.rotation = nn.Parameter(torch.randn(4))

        inside_point_center = torch.mean(init_points, dim=0)
        min_corner_0 = torch.mean(init_points[init_points[:, 0] <= inside_point_center[0]])
        min_corner_1 = torch.mean(init_points[init_points[:, 1] <= inside_point_center[1]])
        min_corner_2 = torch.mean(init_points[init_points[:, 2] <= inside_point_center[2]])
        min_delta = torch.tensor([min_corner_0, min_corner_1, min_corner_2]) - inside_point_center
        min_delta = torch.min(min_delta, torch.tensor([-0.3, -0.3, -0.3]))
        min_corner = self.inverse_min_corner(min_delta)

        max_corner_0 = torch.mean(init_points[init_points[:, 0] > inside_point_center[0]])
        max_corner_1 = torch.mean(init_points[init_points[:, 1] > inside_point_center[1]])
        max_corner_2 = torch.mean(init_points[init_points[:, 2] > inside_point_center[2]])
        max_delta = torch.tensor([max_corner_0, max_corner_1, max_corner_2]) - inside_point_center
        max_delta = torch.max(max_delta, torch.tensor([0.3, 0.3, 0.3]))
        max_corner = self.inverse_max_corner(max_delta)

        self.min_corner_param.data[0] = min_corner[0].item()
        self.min_corner_param.data[1] = min_corner[1].item()
        self.min_corner_param.data[2] = min_corner[2].item()

        self.max_corner_param.data[0] = max_corner[0].item()
        self.max_corner_param.data[1] = max_corner[1].item()
        self.max_corner_param.data[2] = max_corner[2].item()

        """
        delta = init_points - inside_point_center.unsqueeze(0)
        min_distance_from_center = torch.min(delta, dim=0).values
        max_distance_from_center = torch.max(delta, dim=0).values

        min_distance_from_center = torch.min(min_distance_from_center, torch.tensor([-0.3, -0.3, -0.3]))
        min_distance_from_center_remapped = self.inverse_min_corner(min_distance_from_center)
        max_distance_from_center = torch.max(max_distance_from_center, torch.tensor([0.3, 0.3, 0.3]))
        max_distance_from_center_remapped = self.inverse_max_corner(max_distance_from_center)

        self.min_corner_param.data[0] = min_distance_from_center_remapped[0].item()
        self.min_corner_param.data[1] = min_distance_from_center_remapped[1].item()
        self.min_corner_param.data[2] = min_distance_from_center_remapped[2].item()

        self.max_corner_param.data[0] = max_distance_from_center_remapped[0].item()
        self.max_corner_param.data[1] = max_distance_from_center_remapped[1].item()
        self.max_corner_param.data[2] = max_distance_from_center_remapped[2].item()
        """

        self.position.data[0] = inside_point_center[0].item()
        self.position.data[1] = inside_point_center[1].item()
        self.position.data[2] = inside_point_center[2].item()

        self.rotation.data[0] = identity_quaternion[0].item()
        self.rotation.data[1] = identity_quaternion[1].item()
        self.rotation.data[2] = identity_quaternion[2].item()
        self.rotation.data[3] = identity_quaternion[3].item()

        self.optimizer_config = [OptimizerPreference(15.0, 10.0, [self.min_corner_param, self.max_corner_param, self.position]),
                                 OptimizerPreference(0.05, 0.01, [self.rotation])]

        # TODO: Ranges

    def containment(self, points):
        min_corner = self.min_corner()
        max_corner = self.max_corner()
        transformed_points = invert_rotation(self.rotation, invert_translate(self.position, points))

        face1 = differentiable_leq_c(transformed_points[:, 0], max_corner[0], lambda_=self.lambda_)
        face2 = differentiable_geq_c(transformed_points[:, 0], min_corner[0], lambda_=self.lambda_)
        face3 = differentiable_leq_c(transformed_points[:, 1], max_corner[1], lambda_=self.lambda_)
        face4 = differentiable_geq_c(transformed_points[:, 1], min_corner[1], lambda_=self.lambda_)
        face5 = differentiable_leq_c(transformed_points[:, 2], max_corner[2], lambda_=self.lambda_)
        face6 = differentiable_geq_c(transformed_points[:, 2], min_corner[2], lambda_=self.lambda_)
        # return (face1 * face2 * face3 * face4 * face5 * face6 + 0.00001).pow(1.0 / 6.0)
        return face1 * face2 * face3 * face4 * face5 * face6

    def inverse_min_corner(self, x):
        return inverse_softplus(-x - 0.25)

    def inverse_max_corner(self, x):
        return inverse_softplus(x - 0.25)

    def min_corner(self):
        return -(torch.nn.functional.softplus(self.min_corner_param) + 0.25)

    def max_corner(self):
        return torch.nn.functional.softplus(self.max_corner_param) + 0.25

    def exact_containment(self, points):
        min_corner = self.min_corner()
        max_corner = self.max_corner()
        transformed_points = invert_rotation(self.rotation, invert_translate(self.position, points))

        face1 = transformed_points[:, 0] <= max_corner[0]
        face2 = transformed_points[:, 0] >= min_corner[0]
        face3 = transformed_points[:, 1] <= max_corner[1]
        face4 = transformed_points[:, 1] >= min_corner[1]
        face5 = transformed_points[:, 2] <= max_corner[2]
        face6 = transformed_points[:, 2] >= min_corner[2]

        return face1 & face2 & face3 & face4 & face5 & face6

    def volume(self):
        return torch.abs(self.max_corner() - self.min_corner()).prod()

    def draw(self, ax):
        min_corner = self.min_corner()
        max_corner = self.max_corner()

        # Square 1
        p1 = min_corner
        p2 = torch.tensor([max_corner[0], min_corner[1], min_corner[2]])
        p3 = torch.tensor([max_corner[0], max_corner[1], min_corner[2]])
        p4 = torch.tensor([min_corner[0], max_corner[1], min_corner[2]])

        # Square 2
        p5 = torch.tensor([min_corner[0], min_corner[1], max_corner[2]])
        p6 = torch.tensor([max_corner[0], min_corner[1], max_corner[2]])
        p7 = max_corner
        p8 = torch.tensor([min_corner[0], max_corner[1], max_corner[2]])

        points = translate(self.position, rotate(self.rotation, torch.stack([p1, p2, p3, p4, p5, p6, p7, p8]))).detach().numpy()

        p1 = points[0]
        p2 = points[1]
        p3 = points[2]
        p4 = points[3]
        p5 = points[4]
        p6 = points[5]
        p7 = points[6]
        p8 = points[7]

        # Draw square 1
        draw.draw_line(ax, p1, p2)
        draw.draw_line(ax, p2, p3)
        draw.draw_line(ax, p3, p4)
        draw.draw_line(ax, p4, p1)

        # Draw square 2
        draw.draw_line(ax, p5, p6)
        draw.draw_line(ax, p6, p7)
        draw.draw_line(ax, p7, p8)
        draw.draw_line(ax, p8, p5)

        # Draw lines connecting the squares
        draw.draw_line(ax, p1, p5)
        draw.draw_line(ax, p2, p6)
        draw.draw_line(ax, p3, p7)
        draw.draw_line(ax, p4, p8)

    def normalize(self):
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


class AxisAlignedCuboid(PrimitiveModel):
    def __init__(self, init_points, lambda_):
        super().__init__(lambda_)
        self.min_corner = nn.Parameter(torch.randn(3))
        self.max_corner = nn.Parameter(torch.randn(3))
        # self.rotation takes local space points into world space
        #self.rotation = nn.Parameter(torch.randn((4,)))

        inside_point_center = torch.mean(init_points, dim=0)
        max_distance_from_center = torch.max(torch.abs(init_points - (inside_point_center.unsqueeze(0))), dim=0).values
        p1 = inside_point_center - max_distance_from_center
        p2 = inside_point_center + max_distance_from_center

        self.min_corner.data[0] = p1[0].item()
        self.min_corner.data[1] = p1[1].item()
        self.min_corner.data[2] = p1[2].item()

        self.max_corner.data[0] = p2[0].item()
        self.max_corner.data[1] = p2[1].item()
        self.max_corner.data[2] = p2[2].item()

        self.rotation.data[0] = identity_quaternion[0].item()
        self.rotation.data[1] = identity_quaternion[1].item()
        self.rotation.data[2] = identity_quaternion[2].item()
        self.rotation.data[3] = identity_quaternion[3].item()

        self.optimizer_config = [OptimizerPreference(15.0, 7.0, [self.min_corner, self.max_corner])]

        pos_range = (torch.min(init_points, dim=0).values, torch.max(init_points, dim=0).values)
        self.ranges = {
            "min_corner": pos_range,
            "max_corner": pos_range
        }

    def containment(self, points):
        min_corner = torch.min(self.min_corner, self.max_corner)
        max_corner = torch.max(self.min_corner, self.max_corner)

        face1 = differentiable_leq_c(points[:, 0], max_corner[0], lambda_=self.lambda_)
        face2 = differentiable_geq_c(points[:, 0], min_corner[0], lambda_=self.lambda_)
        face3 = differentiable_leq_c(points[:, 1], max_corner[1], lambda_=self.lambda_)
        face4 = differentiable_geq_c(points[:, 1], min_corner[1], lambda_=self.lambda_)
        face5 = differentiable_leq_c(points[:, 2], max_corner[2], lambda_=self.lambda_)
        face6 = differentiable_geq_c(points[:, 2], min_corner[2], lambda_=self.lambda_)
        #return (face1 * face2 * face3 * face4 * face5 * face6 + 0.00001).pow(1.0 / 6.0)
        return face1 * face2 * face3 * face4 * face5 * face6

    def exact_containment(self, points):
        min_corner = torch.min(self.min_corner, self.max_corner)
        max_corner = torch.max(self.min_corner, self.max_corner)

        face1 = points[:, 0] <= max_corner[0]
        face2 = points[:, 0] >= min_corner[0]
        face3 = points[:, 1] <= max_corner[1]
        face4 = points[:, 1] >= min_corner[1]
        face5 = points[:, 2] <= max_corner[2]
        face6 = points[:, 2] >= min_corner[2]
        return face1 & face2 & face3 & face4 & face5 & face6

    def volume(self):
        return torch.abs(self.max_corner - self.min_corner).prod()

    def get_device(self):
        return get_device(self.min_corner)

    def draw(self, ax):
        min_corner = torch.min(self.min_corner, self.max_corner)
        max_corner = torch.max(self.min_corner, self.max_corner)

        # Square 1
        p1 = min_corner
        p2 = np.array([max_corner[0], min_corner[1], min_corner[2]])
        p3 = np.array([max_corner[0], max_corner[1], min_corner[2]])
        p4 = np.array([min_corner[0], max_corner[1], min_corner[2]])

        # Square 2
        p5 = np.array([min_corner[0], min_corner[1], max_corner[2]])
        p6 = np.array([max_corner[0], min_corner[1], max_corner[2]])
        p7 = max_corner
        p8 = np.array([min_corner[0], max_corner[1], max_corner[2]])

        # Draw square 1
        draw.draw_line(ax, p1, p2)
        draw.draw_line(ax, p2, p3)
        draw.draw_line(ax, p3, p4)
        draw.draw_line(ax, p4, p1)

        # Draw square 2
        draw.draw_line(ax, p5, p6)
        draw.draw_line(ax, p6, p7)
        draw.draw_line(ax, p7, p8)
        draw.draw_line(ax, p8, p5)

        # Draw lines connecting the squares
        draw.draw_line(ax, p1, p5)
        draw.draw_line(ax, p2, p6)
        draw.draw_line(ax, p3, p7)
        draw.draw_line(ax, p4, p8)

    def normalize(self):
        min_corner = torch.min(self.min_corner, self.max_corner)
        max_corner = torch.max(self.min_corner, self.max_corner)

        self.min_corner.data[0] = min_corner[0].item()
        self.min_corner.data[1] = min_corner[1].item()
        self.min_corner.data[2] = min_corner[2].item()

        self.max_corner.data[0] = max_corner[0].item()
        self.max_corner.data[1] = max_corner[1].item()
        self.max_corner.data[2] = max_corner[2].item()

class SphereModel(PrimitiveModel):
    def __init__(self, init_points, lambda_):
        super().__init__(lambda_)
        self.position = nn.Parameter(torch.randn(3))
        self.radius = nn.Parameter(torch.randn(1))

        inside_point_center = torch.mean(init_points, dim=0)
        max_distance_from_center = torch.max(torch.norm(init_points - (inside_point_center.unsqueeze(0)), dim=1))

        self.position.data[0] = inside_point_center[0].item()
        self.position.data[1] = inside_point_center[1].item()
        self.position.data[2] = inside_point_center[2].item()

        self.radius.data[0] = max_distance_from_center.item()

        self.optimizer_config = [OptimizerPreference(15.0, 7.0, [self.position]), OptimizerPreference(7.0, 3.0, [self.radius])]

        self.ranges = {
            "position": (torch.min(init_points, dim=0).values, torch.max(init_points, dim=0).values),
            "radius": (torch.tensor([0.5]), torch.tensor([2.0 * max_distance_from_center.item()]))
        }

    def containment(self, points):
        transformed_points = invert_translate(self.position, points)
        distance_from_origin = torch.norm(transformed_points, dim=1)
        return differentiable_leq_c(distance_from_origin, self.radius, lambda_=self.lambda_)

    def exact_containment(self, points):
        transformed_points = invert_translate(self.position, points)
        distance_from_origin = torch.norm(transformed_points, dim=1)
        return distance_from_origin <= self.radius

    def volume(self):
        return (4.0 / 3.0) * math.pi * (self.radius ** 3)

    def get_device(self):
        if self.position.is_cuda:
            return self.position.get_device()
        else:
            return None

    def normalize(self):
        self.radius.data.abs_()

    def draw(self, ax):
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)

        device = self.get_device()
        x2 = torch.tensor(x, dtype=torch.float, device=device).view(-1)
        y2 = torch.tensor(y, dtype=torch.float, device=device).view(-1)
        z2 = torch.tensor(z, dtype=torch.float, device=device).view(-1)
        points_torch = torch.stack([x2, y2, z2]).t()
        transformed_points = ((self.radius * points_torch) + self.position.unsqueeze(0)).detach().t()

        x3 = transformed_points[0].view(x.shape)
        y3 = transformed_points[1].view(y.shape)
        z3 = transformed_points[2].view(z.shape)

        ax.plot_wireframe(x3.cpu().numpy(), y3.cpu().numpy(), z3.cpu().numpy(), color="b")

# Unit primitive models are rescaled to unit coordinates before checking containment
# this may be problematic when used in conjunction with the sigmoid function because
# the scale may directly impact the containment calculation
class UnitPrimitiveModel(PrimitiveModel):
    def __init__(self, init_points, lambda_):
        super().__init__(lambda_)
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

    def inverse_transform(self, points):
        return scale(self.inverse_scale, invert_rotation(self.rotation, invert_translate(self.position, points)))

    def transform(self, points):
        return translate(self.position, rotate(self.rotation, scale(self.get_scale(), points)))

    def normalize(self):
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
        self.inverse_scale.data.abs_()

    def __str__(self):
        return "Position: " + str(self.position) + "\nRotation: " + str(self.rotation) + "\nScale: " + str(self.get_scale())

class EllipsoidModel(UnitPrimitiveModel):
    def __init__(self, init_points, lambda_):
        super().__init__(init_points, lambda_)

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

class BoxModel(UnitPrimitiveModel):
    def __init__(self, init_points, lambda_):
        super().__init__(init_points, lambda_)

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

class CylinderModel(UnitPrimitiveModel):
    def __init__(self, init_points, lambda_):
        super().__init__(init_points, lambda_)

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