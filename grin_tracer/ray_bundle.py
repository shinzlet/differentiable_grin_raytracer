import torch
from dataclasses import dataclass

@dataclass
class RayBundle:
    z: torch.Tensor

    # shape: [4, n_rays]. rays[i] = (ray i y, ray i x, ray i dydz, ray i dxdz)
    rays: torch.Tensor

    def clone(self) -> 'RayBundle':
        return RayBundle(
            self.z.clone(),
            self.rays.clone()
        )

    def detach(self) -> 'RayBundle':
        return RayBundle(
            self.z.detach(),
            self.rays.detach()
        )
