"""Interactive display functions for convex hull results and point clouds.

All visual parameters (colours, sizes, opacity) are defined as module-level
constants below.  Functions accept an optional *plotter* argument so callers
can reuse an existing Plotter or inspect the built scene without opening a
window (pass ``interactive=False``).
"""

from __future__ import annotations

from collections.abc import Sequence

import pyvista as pv

from domain.entities import HullResult3D, Point3D
from rendering.scene import (
    hull_normals_to_polydata,
    hull_to_polydata,
    points_to_polydata,
)

# ---------------------------------------------------------------------------
# Visual constants — edit here, not inside functions
# ---------------------------------------------------------------------------
_HULL_COLOR: str = "lightgray"
_HULL_OPACITY: float = 0.5
_EDGE_COLOR: str = "dimgray"
_EDGE_LINE_WIDTH: float = 1.5
_NORMAL_COLOR: str = "tomato"
_NORMAL_LINE_WIDTH: float = 2.0
_HULL_VERTEX_COLOR: str = "teal"
_INTERIOR_POINT_COLOR: str = "gray"
_POINT_COLOR: str = "steelblue"
_DEFAULT_POINT_SIZE: float = 8.0
_HULL_VERTEX_SIZE_FACTOR: float = 1.5


def show_hull(
    result: HullResult3D,
    *,
    show_normals: bool = False,
    interactive: bool = True,
    plotter: pv.Plotter | None = None,
) -> pv.Plotter:
    """Render a HullResult3D in an interactive (or off-screen) PyVista window.

    Adds semi-transparent hull faces and wireframe edges unconditionally.
    An optional third actor renders outward face-normal segments.  When
    *interactive* is ``False`` the Plotter is returned without calling
    ``.show()``; this mode is intended for tests and pipeline usage.

    Args:
        result: The convex hull to display.
        show_normals: If ``True``, add outward face-normal line segments.
        interactive: If ``True``, call ``plotter.show()`` before returning.
        plotter: Re-use an existing Plotter.  When ``None`` a new Plotter is
            created whose window title is
            ``"<algorithm> | n=<n_points_input> | h=<n_vertices_hull>"``.

    Returns:
        The Plotter instance (with ``.show()`` already called if *interactive*).
    """
    if plotter is None:
        m = result.metadata
        title = f"{m.algorithm} | n={m.n_points_input} | h={m.n_vertices_hull}"
        plotter = pv.Plotter(title=title)

    hull_pd = hull_to_polydata(result)
    plotter.add_mesh(hull_pd, color=_HULL_COLOR, opacity=_HULL_OPACITY)
    plotter.add_mesh(
        hull_pd.extract_all_edges(),
        color=_EDGE_COLOR,
        line_width=_EDGE_LINE_WIDTH,
    )
    if show_normals:
        normals_pd = hull_normals_to_polydata(result)
        plotter.add_mesh(normals_pd, color=_NORMAL_COLOR, line_width=_NORMAL_LINE_WIDTH)

    if interactive:
        plotter.show()
    return plotter


def show_points(
    points: Sequence[Point3D],
    *,
    interactive: bool = True,
    plotter: pv.Plotter | None = None,
) -> pv.Plotter:
    """Render a raw point cloud in an interactive (or off-screen) PyVista window.

    Adds exactly one actor: the point cloud coloured with :data:`_POINT_COLOR`.

    Args:
        points: Points to display.
        interactive: If ``True``, call ``plotter.show()`` before returning.
        plotter: Re-use an existing Plotter.  A new one is created when
            ``None``.

    Returns:
        The Plotter instance.
    """
    if plotter is None:
        plotter = pv.Plotter()

    cloud = points_to_polydata(points)
    plotter.add_mesh(
        cloud,
        color=_POINT_COLOR,
        point_size=_DEFAULT_POINT_SIZE,
        render_points_as_spheres=True,
    )

    if interactive:
        plotter.show()
    return plotter


def show_hull_with_points(
    result: HullResult3D,
    *,
    show_normals: bool = False,
    point_size: float = _DEFAULT_POINT_SIZE,
    interactive: bool = True,
    plotter: pv.Plotter | None = None,
) -> pv.Plotter:
    """Render a HullResult3D together with all its input points, colour-coded.

    Hull vertices are rendered in crimson (slightly larger); interior points in
    gray.  Adds all actors from :func:`show_hull` plus up to two point actors.

    Args:
        result: The convex hull to display.
        show_normals: If ``True``, add outward face-normal line segments.
        point_size: Size of the point glyphs (world units).
        interactive: If ``True``, call ``plotter.show()`` before returning.
        plotter: Re-use an existing Plotter.  A new one is created when
            ``None``.

    Returns:
        The Plotter instance.
    """
    if plotter is None:
        plotter = pv.Plotter()

    plotter = show_hull(
        result,
        show_normals=show_normals,
        interactive=False,
        plotter=plotter,
    )

    hull_pts = [result.points[i] for i in sorted(result.vertex_indices)]
    interior_pts = [p for i, p in enumerate(result.points) if i not in result.vertex_indices]

    if hull_pts:
        plotter.add_mesh(
            points_to_polydata(hull_pts),
            color=_HULL_VERTEX_COLOR,
            point_size=point_size * _HULL_VERTEX_SIZE_FACTOR,
            render_points_as_spheres=True,
        )
    if interior_pts:
        plotter.add_mesh(
            points_to_polydata(interior_pts),
            color=_INTERIOR_POINT_COLOR,
            point_size=point_size,
            render_points_as_spheres=True,
        )

    if interactive:
        plotter.show()
    return plotter


