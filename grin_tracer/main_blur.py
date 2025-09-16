# Create a coordinate grid
# Create an empty composition tensor as a list of z planes 

import numpy as np
import torch
import torch.nn.functional as F

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

def _loss(target_rays: RayBundle, propagated_rays: RayBundle, coords: Coordinates, iteration: int) -> torch.Tensor:
    # target_rays and propagated_rays should have the same number of rays
    assert target_rays.rays.shape[1] == propagated_rays.rays.shape[1]
    # [2, NRays]: [[dy**2, dx**2], [dy**2, dx**2], ...]
    delta_xy_squared = (target_rays.rays[0:2] - propagated_rays.rays[0:2])**2
    
    # [1, NRays]: [hypot1, hypot2, ...]
    hypots = torch.sqrt(torch.sum(delta_xy_squared, dim=0))
    # We sum and then normalize to the max hypot to make the scale of the angle loss (-1 to 1)
    # and the position loss roughly comparable.
    pos_loss = torch.sum(hypots) / (coords.x_range ** 2 + coords.y_range ** 2) ** 0.5

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
    angle_weight = 0.5 * (1 - np.exp(-iteration / 1000))
    loss = (1 - angle_weight) * pos_loss + angle_weight * angle_loss * 2

    return loss

class Refractive3DOptic:
    coords: Coordinates

    def __init__(self, coords: Coordinates, step_size: float = 0.02):
        self.coords = coords
        self.composition = torch.zeros(coords.shape, dtype=torch.float64, requires_grad=True)
        self._iteration = 0
        self._losses = []
        self._step_size = step_size  # fixed step size

    def _gaussian_kernel_1d(self, size: int, sigma: float, device, dtype):
        """Create 1D Gaussian kernel."""
        x = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return kernel

    def _gaussian_kernel_3d(self, ksize, sigma, device, dtype):
        """Create separable 3D Gaussian kernel as outer product."""
        kx = self._gaussian_kernel_1d(ksize[2], sigma[2], device, dtype)
        ky = self._gaussian_kernel_1d(ksize[1], sigma[1], device, dtype)
        kz = self._gaussian_kernel_1d(ksize[0], sigma[0], device, dtype)
        kernel_3d = torch.einsum("i,j,k->ijk", kz, ky, kx)
        kernel_3d /= kernel_3d.sum()
        return kernel_3d

    def _gaussian_blur3d(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Apply 3D Gaussian blur to a [Z, Y, X] tensor.
        Kernel size ~1/10 of each axis, sigma ~ kernel_size/3.
        """
        sz = max(3, int(round(self.coords.shape[0] / 10)))
        sy = max(3, int(round(self.coords.shape[1] / 10)))
        sx = max(3, int(round(self.coords.shape[2] / 10)))

        # Force odd sizes
        sz += (sz + 1) % 2
        sy += (sy + 1) % 2
        sx += (sx + 1) % 2

        sigma = (sz / 3.0, sy / 3.0, sx / 3.0)

        kernel = self._gaussian_kernel_3d((sz, sy, sx), sigma, vol.device, vol.dtype)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]

        v = vol.unsqueeze(0).unsqueeze(0)  # [1,1,Z,Y,X]
        pad = (sx // 2, sx // 2, sy // 2, sy // 2, sz // 2, sz // 2)
        v = F.pad(v, pad, mode="replicate")
        out = F.conv3d(v, kernel, stride=1)
        return out.squeeze(0).squeeze(0)

    def gradient_update(self, sampler, n_rays: int = 100):
        input_rays, output_rays = sampler(n_rays)
        propagated_rays, ray_penalties = self.propagate_rays(input_rays, keep_paths=False)
        loss = _loss(output_rays, propagated_rays[-1], self.coords, self._iteration)
        loss = loss + 150 * torch.sum(ray_penalties) / (input_rays.rays.shape[1] * self.coords.shape[0])

        if self.composition.grad is not None:
            self.composition.grad.zero_()
        loss.backward()

        with torch.no_grad():
            grad = self.composition.grad
            if grad is None:
                return
            blurred_grad = self._gaussian_blur3d(grad)
            self.composition -= self._step_size * blurred_grad

        self._losses.append(loss.item())
        self._iteration += 1

    def propagate_rays(self, ray_bundle: RayBundle, keep_paths: bool = False) -> tuple[list[RayBundle], torch.Tensor]:
        # compute the index field from the composition
        n = _to_index(self.composition)
        nz, ny, nx = spatial_gradient(n, self.coords)
        n_half = (n[1:] + n[:-1]) / 2
        nz_half = (nz[1:] + nz[:-1]) / 2
        ny_half = (ny[1:] + ny[:-1]) / 2
        nx_half = (nx[1:] + nx[:-1]) / 2
        
        prop_bundle = ray_bundle.clone()
        ret = [prop_bundle.detach().clone()] if keep_paths else []
        
        # We accumulate penalties for each ray based on "niceness" criteria. Examples
        # could be things like distance from the center or extreme angles. These can
        # be added to loss functions later.
        ray_pentalties = torch.zeros(ray_bundle.rays.shape[1], dtype=torch.float64)

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
            # For the index, the "free space" around the optic is index = 1, so we replace
            # zeros with 1. For the gradients, zero is well defined (and air is homogeneous),
            # so we don't do replacement.
            n_points = interp_value_2d(n_plane, ray_points, self.coords, padding_mode="zeros", replace_zeros=1.0)
            nz_points = interp_value_2d(nz_plane, ray_points, self.coords, padding_mode="zeros")
            ny_points = interp_value_2d(ny_plane, ray_points, self.coords, padding_mode="zeros")
            nx_points = interp_value_2d(nx_plane, ray_points, self.coords, padding_mode="zeros")

            # This is not actually momentum, it's 1 + p^2 + q^2 which is a quantity that
            # appears a lot in the ODEs
            momentum = 1 + rays[2]**2 + rays[3]**2

            # ODEs:
            yp = rays[2]
            xp = rays[3]
            # We clamp the derivatives to avoid blowup to NaN (which kills backprop). Right now,
            # we don't penalize these rays: they would be easy to spot in debug visualization, and
            # likely are so uncontrollable that their loss is enormous already.
            dydz_p = torch.clamp((momentum / n_points) * (ny_points - rays[2] * nz_points), -10, 10)
            dxdz_p = torch.clamp((momentum / n_points) * (nx_points - rays[3] * nz_points), -10, 10)

            return torch.stack([yp, xp, dydz_p, dxdz_p])
        
        def _boundary_penalty(ordinates, vmin, vmax, softness=0.02):
            # center/half_width define a normalized “distance to wall”
            center = (vmin + vmax) / 2
            half   = (vmax - vmin) / 2
            # distance > 0 means heading toward the wall
            dist = torch.abs((ordinates - center) / half)  # 0 at center, 1 at walls
            # Softplus gives smooth >0 slope everywhere, sharper near walls as dist→1
            return F.softplus((dist - 0.9) / softness)     # start rising ~10% from wall

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

            prop_bundle.rays = prop_bundle.rays + (self.coords.dz / 6) * (k1 + 2*k2 + 2*k3 + k4)

            # Penalize rays:
            ray_pentalties = ray_pentalties + _boundary_penalty(prop_bundle.rays[0], self.coords.y_min_face, self.coords.y_max_face)
            ray_pentalties = ray_pentalties + _boundary_penalty(prop_bundle.rays[1], self.coords.x_min_face, self.coords.x_max_face)

            prop_bundle.z = prop_bundle.z + self.coords.dz
            if keep_paths:
                ret.append(prop_bundle.detach().clone())
        
        # Final weird thing: we need to propagate one more step to get to the front face, but we
        # can't use the same numerics to do it because it is a boundary. We will use a single
        # step of Euler's method.
        k1 = _rhs_march(prop_bundle.rays, self.coords.nz - 2, plus_half=False)
        prop_bundle.rays = prop_bundle.rays + self.coords.dz * k1
        prop_bundle.z = prop_bundle.z + self.coords.dz
        # Penalize rays for the final step
        ray_pentalties = ray_pentalties + _boundary_penalty(prop_bundle.rays[0], self.coords.y_min_face, self.coords.y_max_face)
        ray_pentalties = ray_pentalties + _boundary_penalty(prop_bundle.rays[1], self.coords.x_min_face, self.coords.x_max_face)
        # We always keep the last step, because it is the output of the entire propagation
        ret.append(prop_bundle)

        return ret, ray_pentalties

    def visualize_rays(self, ray_bundles: list[RayBundle], viewer):
        with torch.no_grad():
            index = _to_index(self.composition).detach().cpu().numpy()

            viewer.add_image(
                index,
                name="Refractive Index",
                scale=(self.coords.dz, self.coords.dy, self.coords.dx),
                translate=(self.coords.z_min_center, self.coords.y_min_center, self.coords.x_min_center),
                colormap="plasma",
                contrast_limits=(1.0, 2.0),
            )

            overshoot = 0.3 # 30% overshoot ray extension
            overshoot_length = overshoot * self.coords.z_range
            overshoot_planes = int(np.ceil(overshoot_length / self.coords.dz))

            # each elment of rays is a RayBundle at a different z plane. We need to unpack them
            # into a list of (z,y,x) points for each ray.
            # ray_bundles[0].rays.shape[1] == number of rays in a bundle
            for ray_idx in range(ray_bundles[0].rays.shape[1]):
                # len(ray_bundles) == number of z planes
                ray = np.zeros((len(ray_bundles) + overshoot_planes, 3), dtype=np.float64) # (z steps, 3)
                for bundle_idx in range(len(ray_bundles)):
                    ray[bundle_idx, 0] = ray_bundles[bundle_idx].z
                    ray[bundle_idx, 1:] = ray_bundles[bundle_idx].rays[0:2, ray_idx].detach().cpu().numpy()
                
                # Naively extrapolate extra z planes just for visualization
                exit_pos_z = ray_bundles[-1].z
                exit_pos_yx = ray_bundles[-1].rays[0:2, ray_idx]
                exit_pos_dydz_dxdz = ray_bundles[-1].rays[2:, ray_idx]
                for overshoot_z_idx in range(overshoot_planes):
                    delta_z = self.coords.dz * overshoot_z_idx
                    ray[-overshoot_z_idx-1][0] = exit_pos_z + delta_z
                    ray[-overshoot_z_idx-1][1:] = exit_pos_yx + exit_pos_dydz_dxdz * delta_z

                # viewer.add_points(ray, size=0.2, face_color='red', name=f"Ray {ray_idx}")
                viewer.add_shapes(ray, shape_type='path', edge_color='white', name=f"Ray {ray_idx} Path", edge_width=0.01)

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
    -1, 1, 100,
    -1, 1, 100
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

optic = Refractive3DOptic(coords)
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

import napari
import matplotlib.pyplot as plt
counter = 0
first_loop = True
while True:
    try:
        print(f"Iteration {optic._iteration}")
        if counter == 0:
            if not first_loop:
                input_rays, output_rays = sampler(10)
                # print(input_rays)
                ray_sequence, _ = optic.propagate_rays(input_rays, keep_paths=True)
                # print(ray_sequence)
                viewer = napari.Viewer()
                viewer.layers.clear()
                optic.visualize_rays(ray_sequence, viewer)
                viewer.show()
                napari.run()

                plt.plot(optic._losses)        
                plt.yscale('log')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Loss over Iterations')
                plt.grid(True)
                plt.show()

            counter = input("how many iterations to run? ")
            counter = int(counter)
            first_loop = False

        optic.gradient_update(sampler, n_rays=1000)
        counter -= 1
    except KeyboardInterrupt:
        try:
            input("Keyboard interrupt: forcing visualization. Press Enter to continue, or Ctrl-C again to quit.")
            counter = 0
        except KeyboardInterrupt:
            print("Exiting")
            break

napari.run()
