
import numpy as np
import torch

# CUDA extension
import _raytracing as _backend

class RayTracer():
    def __init__(self, vertices, triangles):
        # vertices: np.ndarray, [N, 3]
        # triangles: np.ndarray, [M, 3]
        
        # implementation
        self.impl = _backend.create_raytracer(vertices, triangles)

    def __call__(self, rays_o, rays_d):
        # rays_o: torch.Tensor, cuda, float, [N, 3]
        # rays_d: torch.Tensor, cuda, float, [N, 3]

        # inplace write intersections back to rays_o
        self.impl.trace(rays_o, rays_d) # [N, 3]

        return rays_o