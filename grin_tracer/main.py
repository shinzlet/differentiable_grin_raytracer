# Create a coordinate grid
# Create an empty composition tensor as a list of z planes 

import numpy as np
import torch

from grin_tracer.coordinates import Coordinates
from grin_tracer.ray_bundle import RayBundle
from grin_tracer.optic import Optic

# We use mm as our unit, although this is scale invariant. The factory
# can manufacture a 3mm x 1" x 1" optic.
# coords = Coordinates(
#     0, 3, 50,
#     -12.7, 12.7, 50,
#     -12.7, 12.7, 50
# )

# validation: maxwell's fisheye
coords = Coordinates(
    0, 1, 40,
    -1, 1, 40,
    -1, 1, 40
)

def sampler(n: int):
    # Produce rays that emerge from a point at z = -1 and focus to a point at z = 1 with random
    # angular distribution.
    injection_angles = torch.rand(n, 1) * 2 * np.pi + 0.1
    angle_magnitudes = torch.rand(n, 1) * 2
    input_rays = RayBundle(
        torch.Tensor([0]).double(),
        torch.Tensor([
            [0] * n,
            [0] * n,
            torch.cos(injection_angles) * angle_magnitudes,
            torch.sin(injection_angles) * angle_magnitudes
        ]).double())
    
    output_rays = RayBundle(
        torch.Tensor([1]).double(),
        torch.Tensor([
            [0] * n,
            [0] * n,
            -torch.cos(injection_angles) * angle_magnitudes,
            -torch.sin(injection_angles) * angle_magnitudes
        ]).double())
    return input_rays, output_rays

optic = Optic(coords)

from grin_tracer.interactive_trainer import InteractiveTrainer
trainer = InteractiveTrainer(optic, sampler, n_rays=4096, n_rays_visualized=10)
trainer.run()

# r2 = coords.xx**2 + coords.yy**2 + coords.zz**2
# fisheye = np.ones_like(r2) * 1.5
# # fisheye[r2 < 1] = 2 / (1 + r2[r2 < 1])
# fisheye = _to_composition(fisheye)
# torch.autograd.set_detect_anomaly(True)
# optic.composition = torch.from_numpy(fisheye).requires_grad_()
# N = 1
# input_rays = RayBundle(torch.Tensor([-1]).double(), torch.Tensor([[0] * N, [0] * N, [np.cos(theta) for theta in np.linspace(0, 2 * np.pi, N, endpoint=False)], [np.sin(theta) for theta in np.linspace(0, 2 * np.pi, N, endpoint=False)]]).double())
# ray_sequence = optic.propagate_rays(input_rays, keep_paths=True)
# optic.visualize_rays(ray_sequence)
