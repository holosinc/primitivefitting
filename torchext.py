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

def norm2(m, dim=0):
    return m.pow(2).sum(dim)

def clamp01(x):
    return numerically_stable_sigmoid(10.0 * (x - 0.5))

def trough_curve(x, center, trough_size):
    return torch.exp(-(((x - center) / trough_size).pow(4.0)))