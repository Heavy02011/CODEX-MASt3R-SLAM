import torch

def _bilinear_sample(img, u, v):
    h, w = img.shape[0], img.shape[1]
    u = float(u); v = float(v)
    u11 = int(torch.floor(torch.tensor(u)))
    v11 = int(torch.floor(torch.tensor(v)))
    du = u - u11
    dv = v - v11
    w11 = du * dv
    w12 = (1.0 - du) * dv
    w21 = du * (1.0 - dv)
    w22 = (1.0 - du) * (1.0 - dv)
    r11 = img[v11 + 1, u11 + 1]
    r12 = img[v11 + 1, u11]
    r21 = img[v11, u11 + 1]
    r22 = img[v11, u11]
    return (
        w11 * r11 +
        w12 * r12 +
        w21 * r21 +
        w22 * r22
    )

def iter_proj(rays_img_with_grad, pts_3d_norm, p_init, max_iter, lambda_init, cost_thresh):
    b, h, w, _ = rays_img_with_grad.shape
    n = pts_3d_norm.shape[1]
    device = rays_img_with_grad.device
    p_new = torch.zeros_like(p_init)
    converged = torch.zeros(b, n, dtype=torch.bool, device=device)
    for bi in range(b):
        for ni in range(n):
            u = float(p_init[bi, ni, 0])
            v = float(p_init[bi, ni, 1])
            u = max(1.0, min(w - 2.0, u))
            v = max(1.0, min(h - 2.0, v))
            lam = lambda_init
            for _ in range(max_iter):
                sample = _bilinear_sample
                vec = sample(rays_img_with_grad[bi], u, v)
                r = vec[:3]
                gx = vec[3:6]
                gy = vec[6:9]
                r = r / torch.linalg.norm(r)
                err = r - pts_3d_norm[bi, ni]
                cost = torch.dot(err, err)
                A00 = torch.dot(gx, gx) + lam
                A01 = torch.dot(gx, gy)
                A11 = torch.dot(gy, gy) + lam
                b0 = -torch.dot(err, gx)
                b1 = -torch.dot(err, gy)
                det = A00 * A11 - A01 * A01
                delta_u = (A11 * b0 - A01 * b1) / det
                delta_v = (-A01 * b0 + A00 * b1) / det
                u_new = max(1.0, min(w - 2.0, u + delta_u))
                v_new = max(1.0, min(h - 2.0, v + delta_v))
                vec2 = sample(rays_img_with_grad[bi], u_new, v_new)
                r2 = vec2[:3]
                r2 = r2 / torch.linalg.norm(r2)
                err2 = r2 - pts_3d_norm[bi, ni]
                new_cost = torch.dot(err2, err2)
                if new_cost < cost:
                    u, v = u_new, v_new
                    lam *= 0.1
                    if new_cost < cost_thresh:
                        converged[bi, ni] = True
                else:
                    lam *= 10.0
                    if cost < cost_thresh:
                        converged[bi, ni] = True
            p_new[bi, ni, 0] = u
            p_new[bi, ni, 1] = v
    return p_new, converged

def refine_matches(D11, D21, p1, radius, dilation_max):
    b, n = p1.shape[:2]
    h, w = D11.shape[1:3]
    p1_new = p1.clone()
    for bi in range(b):
        for ni in range(n):
            u0 = int(p1[bi, ni, 0])
            v0 = int(p1[bi, ni, 1])
            u_new, v_new = u0, v0
            for d in range(dilation_max, 0, -1):
                rd = radius * d
                diam = 2 * rd + 1
                max_score = float('-inf')
                for i in range(0, diam, d):
                    for j in range(0, diam, d):
                        u = u0 - rd + i
                        v = v0 - rd + j
                        if 0 <= u < w and 0 <= v < h:
                            score = torch.dot(D21[bi, ni], D11[bi, v, u])
                            if score > max_score:
                                max_score = score
                                u_new, v_new = u, v
                u0, v0 = u_new, v_new
            p1_new[bi, ni, 0] = u_new
            p1_new[bi, ni, 1] = v_new
    return (p1_new,)

def gauss_newton_rays(*args, **kwargs):
    raise NotImplementedError("gauss_newton_rays not available without CUDA")

def gauss_newton_calib(*args, **kwargs):
    raise NotImplementedError("gauss_newton_calib not available without CUDA")
