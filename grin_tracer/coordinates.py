import numpy as np

class Coordinates:
    """
    Represents a 3D lattice of physical rectilinear cells. `n` is the number of cells
    along a given axis - the axis_min_face and axis_max_face values are the physical extents
    of the prism. (i.e. it would fit in a slot of size (axis_max_face - axis_min_face)).

    The actual values you might assign to those cells would be a scalar field sampled at the
    cell centers: so there is a half-cell offset from the min_face value to the first cell center,
    which is what the x_min_center, etc. properties return.

    To be concrete:

    ```python
    coords = Coordinates(-1, 1, 2, -1, 1, 2, -1, 1, 2)
    coords.z_min_face  # -1
    coords.z_max_face  #  1
    coords.nz          #  2
    coords.dz          #  1
    coords.z_min_center # -0.5
    coords.z_max_center #  0.5
    coords.z # array([-0.5, 0.5])
    coords.zz # np.meshgrid(coords.z, coords.y, coords.x, indexing="ij")[0]
    ```
    """

    def __init__(self,
                 z_min_face: float, z_max_face: float, nz: int,
                 y_min_face: float, y_max_face: float, ny: int,
                 x_min_face: float, x_max_face: float, nx: int):
        self._z_min_face = z_min_face
        self._z_max_face = z_max_face
        self._nz = nz
        self._dz = (z_max_face - z_min_face) / nz
        self._z = np.linspace(self.z_min_center, self.z_max_center, nz)

        self._y_min_face = y_min_face
        self._y_max_face = y_max_face
        self._ny = ny
        self._dy = (y_max_face - y_min_face) / ny
        self._y = np.linspace(self.y_min_center, self.y_max_center, ny)

        self._x_min_face = x_min_face
        self._x_max_face = x_max_face
        self._nx = nx
        self._dx = (x_max_face - x_min_face) / nx
        self._x = np.linspace(self.x_min_center, self.x_max_center, nx)

        self._zz, self._yy, self._xx = np.meshgrid(self.z, self.y, self.x, indexing="ij")

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Returns (nz, ny, nx)
        """
        return (self.nz, self.ny, self.nx)

    @property
    def z_min_face(self) -> float:
        return self._z_min_face
    
    @property
    def z_max_face(self) -> float:
        return self._z_max_face
    
    @property
    def z_min_center(self) -> float:
        return self._z_min_face + 0.5 * self._dz

    @property
    def z_max_center(self) -> float:
        return self._z_max_face - 0.5 * self._dz

    @property
    def nz(self) -> int:
        return self._nz

    @property
    def dz(self) -> float:
        return self._dz

    @property
    def z(self) -> np.ndarray:
        return self._z

    @property
    def zz(self) -> np.ndarray:
        return self._zz

    @property
    def y_min_face(self) -> float:
        return self._y_min_face
    
    @property
    def y_max_face(self) -> float:
        return self._y_max_face
    
    @property
    def y_min_center(self) -> float:
        return self._y_min_face + 0.5 * self._dy

    @property
    def y_max_center(self) -> float:
        return self._y_max_face - 0.5 * self._dy

    @property
    def ny(self) -> int:
        return self._ny

    @property
    def dy(self) -> float:
        return self._dy

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def yy(self) -> np.ndarray:
        return self._yy

    @property
    def x_min_face(self) -> float:
        return self._x_min_face

    @property
    def x_max_face(self) -> float:
        return self._x_max_face
    
    @property
    def x_min_center(self) -> float:
        return self._x_min_face + 0.5 * self._dx

    @property
    def x_max_center(self) -> float:
        return self._x_max_face - 0.5 * self._dx

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def xx(self) -> np.ndarray:
        return self._xx
    
    @property
    def x_range(self) -> float:
        return self._x_max_face - self._x_min_face
    
    @property
    def y_range(self) -> float:
        return self._y_max_face - self._y_min_face
    
    @property
    def z_range(self) -> float:
        return self._z_max_face - self._z_min_face
