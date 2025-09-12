# Interpolate over a anisotropic, rectilinear grid (both value and gradients)

import numpy as np
from grin_tracer.torch_utils import interp_value
import napari

x = np.linspace(-12.7, 12.7, 20)
y = np.linspace(-12.7, 12.7, 20)
z = np.linspace(0, 3, 50)

dz, dy, dx = z[1] - z[0], y[1] - y[0], x[1] - x[0]
zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

# This is an arbitrary field that's just built to make it easy to
# see the values and gradients along the proper directions.
# The field increases linearly along x. It is parabolic around
# y=0 (the center of the y axis). It is parabolic (but only the
# +ve half, so monotonically increasing) in the z axis.
field = xx / np.max(xx) + yy ** 2 / np.max(yy) ** 2 + zz ** 2 / np.max(zz) ** 2
dfdz, dfdy, dfdx = np.zeros_like(field), np.zeros_like(field), np.zeros_like(field)

# a b c d e f g
# _ x x x x x _
# _ c-a/2 ... g-e/2 _
#   2-0/2 ... -1 - -3 / 2

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

viewer = napari.Viewer()
scale = [dz, dy, dx]
viewer.add_image(field, colormap="plasma", scale=scale, name="Scalar Field")
viewer.add_image(dfdz, colormap="blue", scale=scale, name="df / dz")
viewer.add_image(dfdy, colormap="green", scale=scale, name="df / dy")
viewer.add_image(dfdx, colormap="red", scale=scale, name="df / dx")

viewer.show()
napari.run()
