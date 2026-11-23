"""
theta_star.py
Theta* Any-Angle Path Planner for 2D Occupancy Grids.

Theta* is a variant of A* that allows any-angle paths by checking
line-of-sight between non-adjacent nodes. This produces smooth,
near-optimal paths without the jagged staircase artifacts of grid-based A*.
"""

import numpy as np
import heapq


def _line_of_sight(grid, r0, c0, r1, c1):
    """
    Bresenham's line algorithm to check if there is an unobstructed
    line of sight between grid cells (r0, c0) and (r1, c1).

    Returns True if the path is clear (no occupied cells along the line).
    """
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    r, c = r0, c0
    rows, cols = grid.shape

    while True:
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if grid[r, c] == 1:
            return False
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr

    return True


def _heuristic(r0, c0, r1, c1):
    """Euclidean distance heuristic."""
    return np.sqrt(float((r1 - r0) ** 2 + (c1 - c0) ** 2))


def theta_star(grid, start, goal):
    """
    Theta* any-angle path planner on a 2D occupancy grid.

    The key innovation over A*: when expanding a node, Theta* checks
    line-of-sight from the PARENT of the current node to each neighbor.
    If visible, it connects parent -> neighbor directly, creating smooth
    any-angle paths that skip intermediate grid nodes.

    Args:
        grid: 2D numpy array (0=free, 1=obstacle)
        start: (row, col) tuple
        goal: (row, col) tuple

    Returns:
        List of (row, col) waypoints from start to goal.
        Empty list if no path exists.
    """
    rows, cols = grid.shape
    sr, sc = start
    gr, gc = goal

    if grid[sr, sc] == 1 or grid[gr, gc] == 1:
        return []

    g_score = np.full((rows, cols), np.inf)
    g_score[sr, sc] = 0.0

    parent = {}
    parent[(sr, sc)] = (sr, sc)

    open_set = []
    heapq.heappush(open_set, (_heuristic(sr, sc, gr, gc), sr, sc))

    closed = set()

    # 8-connected neighborhood
    neighbors_delta = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    while open_set:
        _, cr, cc = heapq.heappop(open_set)
        current = (cr, cc)

        if current == (gr, gc):
            # Reconstruct path by walking parent pointers
            path = []
            node = (gr, gc)
            while node != (sr, sc):
                path.append(node)
                node = parent[node]
            path.append((sr, sc))
            path.reverse()
            return path

        if current in closed:
            continue
        closed.add(current)

        for dr, dc in neighbors_delta:
            nr, nc = cr + dr, cc + dc
            neighbor = (nr, nc)

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if grid[nr, nc] == 1:
                continue
            if neighbor in closed:
                continue

            # Theta* core: check line-of-sight from parent(current) to neighbor
            p = parent[current]
            if _line_of_sight(grid, p[0], p[1], nr, nc):
                # Any-angle shortcut: connect parent directly to neighbor
                new_g = g_score[p[0], p[1]] + _heuristic(p[0], p[1], nr, nc)
                if new_g < g_score[nr, nc]:
                    g_score[nr, nc] = new_g
                    parent[neighbor] = p
                    f_score = new_g + _heuristic(nr, nc, gr, gc)
                    heapq.heappush(open_set, (f_score, nr, nc))
            else:
                # Fallback to standard A* update through current node
                move_cost = _heuristic(cr, cc, nr, nc)
                new_g = g_score[cr, cc] + move_cost
                if new_g < g_score[nr, nc]:
                    g_score[nr, nc] = new_g
                    parent[neighbor] = current
                    f_score = new_g + _heuristic(nr, nc, gr, gc)
                    heapq.heappush(open_set, (f_score, nr, nc))

    return []


def build_occupancy_grid(arena_w, arena_h, resolution, obstacles, robot_radius):
    """
    Build a 2D occupancy grid from obstacle definitions.
    Obstacles are inflated by the robot radius to create a configuration-space
    representation where the robot can be treated as a point.

    Args:
        arena_w, arena_h: Arena dimensions in meters.
        resolution: Grid cell size in meters.
        obstacles: List of obstacle dicts with 'type' and shape params.
        robot_radius: Robot radius for inflation.

    Returns:
        2D numpy array (0=free, 1=occupied).
    """
    rows = int(arena_h / resolution)
    cols = int(arena_w / resolution)
    grid = np.zeros((rows, cols), dtype=np.int8)

    inflate = robot_radius + 0.02  # small safety margin

    # Pre-compute world coordinate grids (vectorized)
    col_coords = np.arange(cols) * resolution
    row_coords = np.arange(rows) * resolution
    xx, yy = np.meshgrid(col_coords, row_coords)

    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy, r = obs["x"], obs["y"], obs["r"]
            inflated_r = r + inflate
            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
            grid[dist_sq <= inflated_r ** 2] = 1

        elif obs["type"] == "rect":
            x_min = obs["x_min"] - inflate
            y_min = obs["y_min"] - inflate
            x_max = obs["x_max"] + inflate
            y_max = obs["y_max"] + inflate
            mask = (xx >= x_min) & (xx <= x_max) & (yy >= y_min) & (yy <= y_max)
            grid[mask] = 1

    return grid


def world_to_grid(x, y, resolution):
    """Convert world (x, y) coordinates to grid (row, col)."""
    col = int(x / resolution)
    row = int(y / resolution)
    return (row, col)


def grid_to_world(row, col, resolution):
    """Convert grid (row, col) to world (x, y) coordinates."""
    x = col * resolution
    y = row * resolution
    return (x, y)


def plan_path(arena_w, arena_h, resolution, obstacles, robot_radius, start_xy, goal_xy):
    """
    High-level function: build the occupancy grid and run Theta*.

    Args:
        arena_w, arena_h: Arena dimensions in meters.
        resolution: Grid resolution in meters.
        obstacles: List of obstacle dicts.
        robot_radius: Robot radius for grid inflation.
        start_xy: (x, y) start position in world coordinates.
        goal_xy: (x, y) goal position in world coordinates.

    Returns:
        List of (x, y) waypoints in world coordinates, or empty list.
    """
    grid = build_occupancy_grid(arena_w, arena_h, resolution, obstacles, robot_radius)

    start_grid = world_to_grid(start_xy[0], start_xy[1], resolution)
    goal_grid = world_to_grid(goal_xy[0], goal_xy[1], resolution)

    rows, cols = grid.shape
    start_grid = (
        max(0, min(start_grid[0], rows - 1)),
        max(0, min(start_grid[1], cols - 1)),
    )
    goal_grid = (
        max(0, min(goal_grid[0], rows - 1)),
        max(0, min(goal_grid[1], cols - 1)),
    )

    if grid[start_grid[0], start_grid[1]] == 1:
        return []
    if grid[goal_grid[0], goal_grid[1]] == 1:
        return []

    path_grid = theta_star(grid, start_grid, goal_grid)

    if not path_grid:
        return []

    path_world = [grid_to_world(r, c, resolution) for r, c in path_grid]
    return path_world
