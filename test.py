import numpy as np
import torch

import raytracing

# quad mesh
vertices = np.array([[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]], dtype=np.float32)
triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)

rt = raytracing.RayTracer(vertices, triangles)

print(rt)

# intersect with ray
rays_o = torch.FloatTensor([[0, 0, -1], [0, 0, -1], [0, 0, -1]]).cuda()
rays_d = torch.FloatTensor([[0, 0, 1], [1, 0, 1], [2, 0, 1]]).cuda()

rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)

print(rays_o, rays_d)

rt.trace(rays_o, rays_d)

print(rays_o, rays_d)