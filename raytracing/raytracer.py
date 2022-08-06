
import numpy as np
import torch

# CUDA extension
import _raytracing as _backend

class RayTracer():
    def __init__(self, vertices, triangles):
        # vertices: np.ndarray, [N, 3]
        # triangles: np.ndarray, [M, 3]

        assert triangles.shape[0] > 8
        
        # implementation
        self.impl = _backend.create_raytracer(vertices, triangles)

    def trace(self, rays_o, rays_d):
        # rays_o: torch.Tensor, cuda, float, [N, 3]
        # rays_d: torch.Tensor, cuda, float, [N, 3]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        rays_o = rays_o.float().contiguous()
        rays_d = rays_d.float().contiguous()

        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        # print('trace', rays_o.shape, rays_d.shape)

        # inplace write intersections back to rays_o
        self.impl.trace(rays_o, rays_d) # [N, 3]

        rays_o = rays_o.view(*prefix, 3)
        rays_d = rays_d.view(*prefix, 3)

        return rays_o, rays_d