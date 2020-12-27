import torch

# From https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
def quaternion_to_rotation_matrix(quaternion):
    qr = quaternion[0]
    qi = quaternion[1]
    qj = quaternion[2]
    qk = quaternion[3]
    s = 1.0 / quaternion.pow(2).sum()
    row1 = torch.cat([(1.0 - 2.0 * s * (qj.pow(2) + qk.pow(2))).unsqueeze(0), (2.0 * s * (qi * qj - qk * qr)).unsqueeze(0), (2.0 * s * (qi * qk + qj * qr)).unsqueeze(0)])
    row2 = torch.cat([(2.0 * s * (qi * qj + qk * qr)).unsqueeze(0), (1.0 - 2.0 * s * (qi.pow(2) + qk.pow(2))).unsqueeze(0), (2.0 * s * (qj * qk - qi * qr)).unsqueeze(0)])
    row3 = torch.cat([(2.0 * s * (qi * qk - qj * qr)).unsqueeze(0), (2.0 * s * (qj * qk + qi * qr)).unsqueeze(0), (1.0 - 2.0 * s * (qi.pow(2) + qj.pow(2))).unsqueeze(0)])
    return torch.stack([row1, row2, row3])

identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])


def rotation_matrix_to_quaternion(a):
    trace = a[0,0] + a[1,1] + a[2,2]
    if trace.item() > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (a[2,1] - a[1,2]) * s
        y = (a[0,2] - a[2,0]) * s
        z = (a[1,0] - a[0,1]) * s
    else:
        if a[0,0] > a[1,1] and a[0,0] > a[2,2]:
            s = 2.0 * torch.sqrt(1.0 + a[0,0] - a[1,1] - a[2,2])
            w = (a[2,1] - a[1,2]) / s
            x = 0.25 * s
            y = (a[0,1] + a[1,0]) / s
            z = (a[0,2] + a[2,0]) / s
        elif a[1,1] > a[2,2]:
            s = 2.0 * torch.sqrt(1.0 + a[1,1] - a[0,0] - a[2,2])
            w = (a[0,2] - a[2,0]) / s
            x = (a[0,1] + a[1,0]) / s
            y = 0.25 * s
            z = (a[1,2] + a[2,1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + a[2,2] - a[0,0] - a[1,1])
            w = (a[1,0] - a[0,1]) / s
            x = (a[0,2] + a[2,0]) / s
            y = (a[1,2] + a[2,1]) / s
            z = 0.25 * s
    return torch.tensor([w, x, y, z])



def invert_rotation_matrix(rot_matrix):
    # The inverse of a rotation matrix is merely the transpose
    return rot_matrix.t()

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