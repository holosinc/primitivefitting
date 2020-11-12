import torch

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

def differentiable_leq_one(x, lambda_=1.0):
    return numerically_stable_sigmoid(lambda_ * (1.0 - x))

def differentiable_geq_neg_one(x, lambda_=1.0):
    return numerically_stable_sigmoid(lambda_ * (x + 1.0))

def differentiable_leq_c(x, c, lambda_=1.0):
    return numerically_stable_sigmoid(lambda_ * (c - x))

def differentiable_geq_c(x, c, lambda_=1.0):
    return numerically_stable_sigmoid(lambda_ * (x - c))

def norm2(m, dim=0):
    return m.pow(2).sum(dim)

def clamp01(x):
    return numerically_stable_sigmoid(10.0 * (x - 0.5))

def trough_curve(x, center, trough_size):
    return torch.exp(-(((x - center) / trough_size).pow(4.0)))

def inverse_softplus(x):
    ret = torch.log(torch.exp(x) - 1.0)
    over_twenty = x > 20
    ret[over_twenty] = x[over_twenty]
    return ret

def get_device(x):
    if x.is_cuda:
        return x.get_device()
    else:
        return None

def perpendicular_vector(v):
    if v[1].item() == 0 and v[2].item() == 0:
        if v[0].item() == 0:
            raise ValueError('zero vector')
        else:
            return torch.cross(v, torch.tensor([0.0, 1.0, 0.0], device=get_device(v)))
    return torch.cross(v, torch.tensor([1.0, 0.0, 0.0], device=get_device(v)))

def normalize(v):
    return v / torch.norm(v, 2)