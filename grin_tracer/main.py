# Create a coordinate grid
# Create an empty composition tensor as a list of z planes 

import numpy as np
import torch

from grin_tracer.coordinates import Coordinates
from grin_tracer.ray_bundle import RayBundle
from grin_tracer.torch_utils import interp_value_2d, spatial_gradient

# TODO: replace this so that it takes into account the material palette and wavelength
def _to_index(composition):
    if isinstance(composition, torch.Tensor):
        arctan = torch.arctan
    elif isinstance(composition, np.ndarray):
        arctan = np.arctan
    # Since -pi/2 < arctan < pi/2, this guarantees 1 < concentration < 1.5
    concentration = 1.5 + arctan(composition) / np.pi
    return concentration

def _to_composition(index):
    if isinstance(index, torch.Tensor):
        tan, clip = torch.tan, torch.clip
    elif isinstance(index, np.ndarray):
        tan, clip = np.tan, np.clip
    # We enforce 0 < concentration < 1, which guarantees finite values
    # for `composition`:
    eps = 1e-6
    concentration = clip(index - 1, eps, 1-eps)
    composition = tan(np.pi*(concentration - 0.5))
    return composition

class Refractive3DOptic:
    coord: Coordinates

    def __init__(self, coords: Coordinates):
        self.coords = coords
        self.composition = torch.zeros(coords.shape, dtype=torch.float64)

    def gradient_update(self):
        # Create input ray bundle
        input_rays, output_rays = RayBundle(), RayBundle()
        # propagate the rays through
        propagated_rays = self.propagate_rays(input_rays, keep_paths=False)[-1]
        # compute the loss of each ray

        # compute the global loss
        # call backward to compute the gradient
        # apply the gradient to the composition tensor
        pass

    def propagate_rays(self, ray_bundle: RayBundle, keep_paths: bool = False) -> list[RayBundle]:
        # compute the index field from the composition
        n = _to_index(self.composition)
        nz, ny, nx = spatial_gradient(n, self.coords)
        n_half = (n[1:] + n[:-1]) / 2
        nz_half = (nz[1:] + nz[:-1]) / 2
        ny_half = (ny[1:] + ny[:-1]) / 2
        nx_half = (nx[1:] + nx[:-1]) / 2
        
        prop_bundle = ray_bundle.clone()
        ret = [prop_bundle.detach().clone()] if keep_paths else []

        # RayBundle should be a z scalar and a 4xNRays tensor as y, x, dydz, dxdz. This will march
        # the bundle in-place
        def _rhs_march(rays: torch.Tensor, z0: int, plus_half: bool = False):
            # rays is a torch tensor from RayBundle.rays of shape (4, NRays)
            # We need to sample the gradient of the index at each point in the ray bundle.
            # The x and y values will be different for each point, but they all occur in the
            # same z plane. Because we use rk4, the z value is either an integer or half integer.
            # For integers we use the value exactly, for half integers we linearly interpolate
            # two planes.
            if plus_half:
                n_plane = n_half[z0]
                nz_plane = nz_half[z0]
                ny_plane = ny_half[z0]
                nx_plane = nx_half[z0]
            else:
                n_plane = n[z0]
                nz_plane = nz[z0]
                ny_plane = ny[z0]
                nx_plane = nx[z0]
            
            # Now, all of our values can be 2 dimensionally interpolated from this plane.
            ray_points = rays[:2] # [[y1 y2 y3 y4 ...], [x1 x2 x3 x4 ...]]
            n_points = interp_value_2d(n_plane, ray_points, self.coords)
            nz_points = interp_value_2d(nz_plane, ray_points, self.coords)
            ny_points = interp_value_2d(ny_plane, ray_points, self.coords)
            nx_points = interp_value_2d(nx_plane, ray_points, self.coords)

            # This is not actually momentum, it's 1 + p^2 + q^2 which is a quantity that
            # appears a lot in the ODEs
            momentum = 1 + rays[2]**2 + rays[3]**2

            # ODEs:
            yp = rays[2]
            xp = rays[3]
            dydz_p = (momentum / n_points) * (ny_points - rays[2] * nz_points)
            dxdz_p = (momentum / n_points) * (nx_points - rays[3] * nz_points)

            return torch.stack([yp, xp, dydz_p, dxdz_p])

        # z_idx == 0: the ray is on the *back face* of the optic - it has not yet entered the glass
        # for each z fencepost
        # TODO: Deal with the boundary case
        for z_idx in range(self.coords.nz - 1):
            
            # rk4 steps
            # Given:
            # y_n   = state at step n
            # t_n   = time (or independent variable) at step n
            # h     = step size
            # f(t,y) = right-hand side (RHS), i.e., dy/dt = f(t, y)

            # Compute the four stages:
            # k1 = f(t_n,             y_n)
            # k2 = f(t_n + h/2,       y_n + (h/2)*k1)
            # k3 = f(t_n + h/2,       y_n + (h/2)*k2)
            # k4 = f(t_n + h,         y_n + h*k3)

            # Combine:
            # y_{n+1} = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            # t_{n+1} = t_n + h
            k1 = _rhs_march(prop_bundle.rays, z_idx, plus_half=False)
            k2 = _rhs_march(prop_bundle.rays + 0.5 * self.coords.dz * k1, z_idx, plus_half=True)
            k3 = _rhs_march(prop_bundle.rays + 0.5 * self.coords.dz * k2, z_idx, plus_half=True)
            k4 = _rhs_march(prop_bundle.rays + self.coords.dz * k3, z_idx, plus_half=False)

            prop_bundle.rays += (self.coords.dz / 6) * (k1 + 2*k2 + 2*k3 + k4)

            prop_bundle.z += self.coords.dz
            if keep_paths:
                ret.append(prop_bundle.detach().clone())
        
        # Final weird thing: we need to propagate one more step to get to the front face, but we
        # can't use the same numerics to do it because it is a boundary. We will use a single
        # step of Euler's method.
        k1 = _rhs_march(prop_bundle.rays, self.coords.nz - 2, plus_half=False)
        prop_bundle.rays += self.coords.dz * k1
        prop_bundle.z += self.coords.dz
        # We always keep the last step, because it is the output of the entire propagation
        ret.append(prop_bundle)

        return ret

    def visualize_rays(self, ray_bundles: list[RayBundle]):
        import napari

        index = _to_index(self.composition).detach().cpu().numpy()
        viewer = napari.Viewer()
        viewer.add_image(
            index,
            name="Refractive Index",
            scale=(self.coords.dz, self.coords.dy, self.coords.dx),
            translate=(self.coords.z_min_center, self.coords.y_min_center, self.coords.x_min_center),
            colormap="plasma",
            contrast_limits=(1.0, 2.0),
        )

        # each elment of rays is a RayBundle at a different z plane. We need to unpack them
        # into a list of (z,y,x) points for each ray.
        # ray_bundles[0].rays.shape[1] == number of rays in a bundle
        for ray_idx in range(ray_bundles[0].rays.shape[1]):
            # len(ray_bundles) == number of z planes
            ray = np.zeros((len(ray_bundles), 3), dtype=np.float64) # (z steps, 3)
            for bundle_idx in range(len(ray_bundles)):
                ray[bundle_idx, 0] = ray_bundles[bundle_idx].z
                ray[bundle_idx, 1:3] = ray_bundles[bundle_idx].rays[0:2, ray_idx].detach().cpu().numpy()

            # viewer.add_points(ray, size=0.2, face_color='red', name=f"Ray {ray_idx}")
            viewer.add_shapes(ray, shape_type='path', edge_color='white', name=f"Ray {ray_idx} Path", edge_width=0.01)

        viewer.show()
        napari.run()

# We use mm as our unit, although this is scale invariant. The factory
# can manufacture a 3mm x 1" x 1" optic.
# coords = Coordinates(
#     0, 3, 50,
#     -12.7, 12.7, 50,
#     -12.7, 12.7, 50
# )

# validation: maxwell's fisheye
coords = Coordinates(
    -1, 1, 100,
    -1, 1, 100,
    -1, 1, 100
)

optic = Refractive3DOptic(coords)
r2 = coords.xx**2 + coords.yy**2 + coords.zz**2
fisheye = np.ones_like(r2)
fisheye[r2 < 1] = 2 / (1 + r2[r2 < 1])
fisheye = _to_composition(fisheye)
optic.composition = torch.from_numpy(fisheye)

N = 2**15
input_rays = RayBundle(torch.Tensor([-1]).double(), torch.Tensor([[0] * N, [0] * N, [np.cos(theta) for theta in np.linspace(0, 2 * np.pi, N, endpoint=False)], [np.sin(theta) for theta in np.linspace(0, 2 * np.pi, N, endpoint=False)]]).double())
import time
start_time = time.time()
ray_sequence = optic.propagate_rays(input_rays, keep_paths=True)
print(f"Propagated {N} rays through {coords.nz} planes in {time.time() - start_time:.2f} seconds")
# optic.visualize_rays(ray_sequence)
