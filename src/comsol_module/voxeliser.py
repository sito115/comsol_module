import logging
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pyvista as pv
from pyvista import VectorLike
# from tqdm import tqdm

logger = logging.getLogger()


@dataclass
class Voxel:
    """Voxelization utility for COMSOL datasets.

    Handles the conversion of regular point-based grids into PyVista
    :class:`pyvista.ImageData` objects (voxels).

    Attributes:
        mesh: The input mesh containing the regular point grid.
        nx, ny, nz: Number of voxels in each dimension.
        dx, dy, dz: Voxel spacing in each dimension.
        origin: Coordinate of the bottom-left-front corner of the grid.
        xs, ys, zs: Sorted unique coordinate values for each dimension.

    """
    mesh: pv.PolyData
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    origin: tuple[float, float, float]
    xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    ys: np.ndarray = field(default_factory=lambda: np.empty(0))
    zs: np.ndarray = field(default_factory=lambda: np.empty(0))

    @classmethod
    def from_mesh(cls, mesh: pv.PolyData) -> Self:
        """Create a Voxel instance from a mesh with regular point centers.

        The mesh points are assumed to represent the centers of the voxels.
        The regularity of the grid is checked during initialization.

        Args:
            mesh: The PyVista PolyData mesh to voxelize.

        Returns:
            A new :class:`Voxel` instance.

        Raises:
            AssertionError: If the grid is not regular.

        """
        xs = np.unique(mesh.points[:, 0])
        ys = np.unique(mesh.points[:, 1])
        zs = np.unique(mesh.points[:, 2])

        nx, ny, nz = len(xs), len(ys), len(zs)
        logger.debug(nx, ny, nz)

        dxs = np.diff(xs)
        dys = np.diff(ys)
        dzs = np.diff(zs)
        logger.debug(dxs, dys, dzs)
        # check regularity
        assert np.all(np.allclose(dxs[0], dxs, atol=1e-3))
        assert np.all(np.allclose(dys[0], dys, atol=1e-3))
        assert np.all(np.allclose(dzs[0], dzs, atol=1e-3))
        dx = int(dxs[0])
        dy = int(dys[0])
        dz = int(dzs[0])

        origin = (
            np.round(xs.min()) - dx / 2,
            np.round(ys.min()) - dy / 2,
            np.round(zs.min()) - dz / 2,
        )

        return cls(
            mesh=mesh,
            nx=nx,
            ny=ny,
            nz=nz,
            dx=dx,
            dy=dy,
            dz=dz,
            origin=origin,
            xs=xs,
            ys=ys,
            zs=zs,
        )

    def create_image_data(self) -> pv.ImageData:
        """Create an empty PyVista ImageData grid matching the voxel geometry.

        Returns:
            A new :class:`pyvista.ImageData` instance.

        """
        grid = pv.ImageData(
            dimensions=(
                self.nx + 1,
                self.ny + 1,
                self.nz + 1,
            ),  # cells -> points (cells = points - 1 in vtk/pyvista)
            spacing=(self.dx, self.dy, self.dz),
            origin=self.origin,
        )
        assert grid.n_cells == self.mesh.n_points
        return grid

    def map_point_data_to_cells(self, grid: pv.ImageData) -> pv.ImageData:
        """Map point values from the source mesh to cell values of a voxel grid.

        This performs a lookup based on coordinates to correctly place
        point data into the 3D array of the image data.

        Args:
            grid: The target PyVista ImageData grid.

        Returns:
            The input grid updated with cell data from the mesh.

        Raises:
            ValueError: If the number of points in the mesh doesn't match
                the number of cells in the grid.

        """
        if grid.n_cells != self.mesh.n_points:
            raise ValueError(
                f"Number of points and cells must match! ({self.mesh.n_points} vs {grid.n_cells})"
            )
        ix = np.searchsorted(self.xs, self.mesh.points[:, 0])
        iy = np.searchsorted(self.ys, self.mesh.points[:, 1])
        iz = np.searchsorted(self.zs, self.mesh.points[:, 2])

        for key, vals in self.mesh.point_data.items():
            cells = np.empty((self.nx, self.ny, self.nz), dtype=vals.dtype)
            cells[ix, iy, iz] = vals
            grid.cell_data[key] = cells.ravel(order="F")

        return grid


if __name__ == "__main__":
    # TODO: Move test to separate test folder.

    mesh = pv.wrap(
        pv.read("/Users/thomassimader/Downloads/RegularGrid_Spreadsheet_Rows.vtu")
    )
    logger.debug(type(mesh))
    voxel = Voxel.from_mesh(mesh)
    grid = voxel.create_image_data()

    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, color="b")  # (scalars=)
    # plotter.add_mesh(grid.points, show_edges=True, opacity=0.4, color="r")
    # plotter.add_mesh(grid, show_edges=True, opacity=0.4)
    # plotter.show_bounds()
    # plotter.show_axes_all()
    # plotter.show()

    field_name = "Temperature_@_t=2.9E12"
    # grid.cell_data[field_name] = mesh.point_data[field_name].ravel(
    #     order="F"
    # )  # Fortran order
    grid = voxel.map_point_data_to_cells(grid)

    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars=field_name, opacity=0.8, show_edges=True)
    # plotter.add_mesh(grid.points, show_edges=True, opacity=0.4, color="r")
    plotter.add_mesh(
        mesh,
        scalars=field_name,
        style="points",
        point_size=10,
    )
    plotter.show_bounds()
    plotter.show_axes_all()
    plotter.show()
