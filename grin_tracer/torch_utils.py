import torch
import torch.nn.functional as F

from grin_tracer.coordinates import Coordinates
import numpy as np

def interp_value_2d(index_yx: torch.Tensor,
                    pts_yx: torch.Tensor,
                    coords: Coordinates,
                    padding_mode: str = "border") -> torch.Tensor:
    """
    Bilinear interpolation on a 2D plane given arbitrary (y, x) points.

    Args:
        index_yx: (Y, X) tensor — the source plane to sample from.
        pts_yx:   (M, 2) tensor of points as (y, x) in the original coordinate
                  system where y in [coords.y_center_min, coords.y_center_max]
                  and x in [coords.x_center_min, coords.x_center_max].
        coords:   Coordinates object, used to normalize to [-1, 1] for grid_sample.
        padding_mode: Passed to grid_sample ("border", "zeros", "reflection").

    Returns:
        (M,) tensor of interpolated values. Grad flows into index_yx if it requires_grad.
    """
    # (1, 1, Y, X) for grid_sample
    V = index_yx[None, None]

    # Split input points
    y = pts_yx[0]
    x = pts_yx[1]

    # Normalize to [-1, 1]
    y_norm = (2.0 * (y - coords.y_min_center) / (coords.y_max_center - coords.y_min_center)) - 1.0
    x_norm = (2.0 * (x - coords.x_min_center) / (coords.x_max_center - coords.x_min_center)) - 1.0

    # grid_sample expects (x, y) order
    grid_xy = torch.stack((x_norm, y_norm), dim=-1).to(device=V.device, dtype=V.dtype)
    grid = grid_xy.view(1, 1, -1, 2)

    # Sample: (1, 1, 1, M) → flatten to (M,)
    vals = F.grid_sample(
        V, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True
    ).view(-1)

    return vals

def spatial_gradient(
        field: torch.Tensor | np.ndarray,
        coords: Coordinates) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
    """
    Produces a finite difference approximation of the spatial gradient inside of a scalar field
    with the same size as the input array (done by using different finite differences at the boundary).
    """

    dfdz, dfdy, dfdx = torch.zeros_like(field), torch.zeros_like(field), torch.zeros_like(field)
    dz, dy, dx = coords.dz, coords.dy, coords.dx

    # Fill the center values using a second order approximation
    dfdz[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2 * dz)
    dfdy[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2 * dy)
    dfdx[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2 * dx)

    # Fill the boundaries using a first order approximation
    dfdz[0, :, :] = (field[1, :, :] - field[0, :, :]) / dz
    dfdz[-1, :, :] = (field[-1, :, :] - field[-2, :, :]) / dz

    dfdy[:, 0, :] = (field[:, 1, :] - field[:, 0, :]) / dy
    dfdy[:, -1, :] = (field[:, -1, :] - field[:, -2, :]) / dy

    dfdx[:, :, 0] = (field[:, :, 1] - field[:, :, 0]) / dx
    dfdx[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / dx

    return dfdz, dfdy, dfdx
