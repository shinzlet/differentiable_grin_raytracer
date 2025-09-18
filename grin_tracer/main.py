# Create a coordinate grid
# Create an empty composition tensor as a list of z planes 

import numpy as np
import torch

from grin_tracer.coordinates import Coordinates
from grin_tracer.ray_bundle import RayBundle
from grin_tracer.optic import Optic
from grin_tracer.interactive_trainer import InteractiveTrainer

# We use mm as our unit, although this is scale invariant. The factory
# can manufacture a 3mm x 1" x 1" optic.
# coords = Coordinates(
#     0, 3, 50,
#     -12.7, 12.7, 50,
#     -12.7, 12.7, 50
# )

coords = Coordinates(
    0, 1, 40,
    -1, 1, 40,
    -1, 1, 40
)

def loss(index: torch.Tensor, target_rays: RayBundle, propagated_rays: RayBundle, coords: Coordinates, iteration: int) -> torch.Tensor:
    # target_rays and propagated_rays should have the same number of rays
    assert target_rays.rays.shape[1] == propagated_rays.rays.shape[1]
    # [2, NRays]: [[dy**2, dx**2], [dy**2, dx**2], ...]
    delta_xy_squared = (target_rays.rays[0:2] - propagated_rays.rays[0:2])**2
    
    # [1, NRays]: [hypot1, hypot2, ...]
    hypots = torch.sqrt(torch.sum(delta_xy_squared, dim=0))
    # We sum and then normalize to the max hypot to make the scale of the angle loss (-1 to 1)
    # and the position loss roughly comparable.
    pos_loss = torch.sum(hypots) / (coords.x_range ** 2 + coords.y_range ** 2) ** 0.5

    return pos_loss

    # We also want to penalize rays that are not pointing in the right direction.
    # We do this by computing the angle between the two direction vectors.
    # The direction vectors are (1, p, q) where p = dy/dz and q = dx/dz.
    # The cosine of the angle between two vectors a and b is given by:
    # First we take the dot product of all of these vectors:
    dot_products = (1 + target_rays.rays[2] * propagated_rays.rays[2] + target_rays.rays[3] * propagated_rays.rays[3])
    # Then we take the magnitude of each vector:
    target_magnitudes = torch.sqrt(1 + target_rays.rays[2]**2 + target_rays.rays[3]**2)
    propagated_magnitudes = torch.sqrt(1 + propagated_rays.rays[2]**2 + propagated_rays.rays[3]**2)
    # Now we can compute the cosine of the angle between each pair of vectors:
    cos_angles = dot_products / (target_magnitudes * propagated_magnitudes)
    # We want a value that is large for large angles, so we use 1 - cos(angle):
    angle_losses = 1 - cos_angles
    angle_loss = torch.sum(angle_losses)

    # This weighting is arbitrary. We use a schedule to prefer position at the start and angle at the end.
    # 50/50 is achieved asymptotically, time constant is 1000 iterations
    angle_weight = 0.5# * (1 - np.exp(-iteration / 1000))
    loss = (1 - angle_weight) * pos_loss + angle_weight * angle_loss * 2

    return loss

def post_update_composition_regularizer(composition: torch.Tensor, iteration: int) -> torch.Tensor:
    # To kill voxels that aren't actively contributing, we just multiply the entire composition by a factor < 1.
    # Voxels that are contributing will be replenished by the gradient update, while voxels that aren't will decay to 0.
    # For early iterations, where rays are everywhere, the decay factor is impactful (starting at 0.9). However,
    # as the training progresses and rays become more focused, we want to reduce the decay to allow the meaningful
    # composition to stabilize. We use an exponential decay schedule to achieve this.
    decay_factor = 0.90 + 0.10 * (1 - np.exp(-iteration / 1000))
    with torch.no_grad():
        composition *= decay_factor
    return composition

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

trainer = InteractiveTrainer(
    optic,
    sampler,
    n_rays=8192,
    n_rays_visualized=10,
    loss_func=loss,
    post_update_composition_regularizer=post_update_composition_regularizer
)

trainer.run()
