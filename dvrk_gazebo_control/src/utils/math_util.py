import torch
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


def quaternion_error(desired, current, square=False, numpy=False, flatten=False):
    q_c = quat_conjugate(current)
    q_r = quat_mul(desired, q_c)
    error = q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
    if square:
        error = error**2
    if numpy:
        error = error.numpy()
    if flatten:
        error = error.flatten()
    return error


def position_error(desired, current, square=False, numpy=False, flatten=False):
    error = desired - current
    if square:
        error = error**2
    if numpy:
        error = error.numpy()
    if flatten:
        error = error.flatten()
    return error