"""
qbot_env.py
Custom Gymnasium Environment for QBot NanoRL Training.

Implements a self-contained 2D differential-drive robot simulator with:
- Vectorized LiDAR raycasting (72 beams)
- Procedural map generation with static and dynamic obstacles
- Reward function matching the Final Formulation (v12)
- Curriculum-stage-aware parameters
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from theta_star import plan_path


class QBotEnv(gym.Env):
    """
    QBot Navigation Environment.

    Observation Space (76,):
        [0:72]  - 72 LiDAR distance readings, normalized to [0, 1]
        [72]    - Relative distance to current waypoint, normalized [0, 1]
        [73]    - Relative heading to current waypoint, normalized [-1, 1]
        [74]    - Current linear velocity, normalized [0, 1]
        [75]    - Current angular velocity, normalized [-1, 1]

    Action Space (2,):
        [0] - Linear velocity command, mapped from [-1,1] to [0, v_max]
        [1] - Angular velocity command, mapped from [-1,1] to [-omega_max, omega_max]
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # --- Arena ---
        self.arena_width = 10.0
        self.arena_height = 10.0

        # --- QBot Platform Parameters ---
        self.robot_radius = 0.175
        self.v_max = 0.50
        self.omega_max = 1.0
        self.dt = 0.1

        # --- LiDAR Parameters ---
        self.n_lidar_rays = 72
        self.d_max = 3.5
        self.d_safe = 0.4
        # Pre-compute ray angle offsets (fixed relative to robot heading)
        self.ray_angles = np.linspace(0, 2 * np.pi, self.n_lidar_rays, endpoint=False)

        # --- Reward Hyperparameters (Final Formulation v12) ---
        self.k = 15.0
        self.lam = 0.2
        self.gamma_ppo = 0.999
        self.w_t_prime = 0.25
        self.R_goal = 1000.0
        self.P_collision = 500.0
        self.P_timeout = 500.0

        # --- Curriculum Defaults (Stage 1) ---
        self.curriculum_stage = 1
        self.t_max = 2400
        self.n_dynamic_range = (0, 1)
        self.v_dyn_range = (0.05, 0.15)
        self.goal_tolerance = 0.5
        self.n_static_obstacles = 5

        # --- Gymnasium Spaces ---
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(76,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # --- Theta* Grid Parameters ---
        self.grid_resolution = 0.05

        # --- Runtime State (initialized in reset) ---
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.robot_v = 0.0
        self.robot_omega = 0.0

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.static_obstacles = []
        self.dynamic_obstacles = []
        self.waypoints = []
        self.current_wp_idx = 0

        self.d_path_0 = 1.0
        self.d_hat_prev = 1.0
        self.step_count = 0

        # Rendering
        self.fig = None
        self.ax = None

    # ------------------------------------------------------------------
    # Curriculum Interface
    # ------------------------------------------------------------------
    def set_curriculum_stage(self, stage):
        """Called by CurriculumCallback to advance difficulty."""
        self.curriculum_stage = stage
        if stage == 1:
            self.t_max = 2400
            self.n_dynamic_range = (0, 1)
            self.v_dyn_range = (0.05, 0.15)
            self.goal_tolerance = 0.5
        elif stage == 2:
            self.t_max = 1200
            self.n_dynamic_range = (1, 2)
            self.v_dyn_range = (0.05, 0.20)
            self.goal_tolerance = 0.35
        elif stage == 3:
            self.t_max = 800
            self.n_dynamic_range = (2, 3)
            self.v_dyn_range = (0.05, 0.25)
            self.goal_tolerance = 0.25
        elif stage == 4:
            self.t_max = 600
            self.n_dynamic_range = (2, 4)
            self.v_dyn_range = (0.05, 0.30)
            self.goal_tolerance = 0.15

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_attempts = 50
        for _ in range(max_attempts):
            self._generate_map()

            # Run Theta* global planner
            all_obstacles = self.static_obstacles  # plan only on static map
            path = plan_path(
                self.arena_width,
                self.arena_height,
                self.grid_resolution,
                all_obstacles,
                self.robot_radius,
                (self.robot_x, self.robot_y),
                (self.goal_x, self.goal_y),
            )

            if not path:
                continue

            self.waypoints = path
            self.current_wp_idx = min(1, len(path) - 1)  # skip start node

            d_path_0 = self._compute_d_path()
            if d_path_0 < 1.0:
                continue

            self.d_path_0 = d_path_0
            self.d_hat_prev = 1.0  # D_hat(0) = D_path(0)/D_path(0) = 1
            self.step_count = 0
            self.robot_v = 0.0
            self.robot_omega = 0.0

            lidar = self._cast_lidar()
            obs = self._build_observation(lidar)
            return obs, {}

        # Fallback: place robot and goal in open space, straight path
        self.static_obstacles = []
        self.dynamic_obstacles = []
        self.robot_x = 2.0
        self.robot_y = 5.0
        self.robot_theta = 0.0
        self.goal_x = 8.0
        self.goal_y = 5.0
        self.waypoints = [(2.0, 5.0), (8.0, 5.0)]
        self.current_wp_idx = 1
        self.d_path_0 = 6.0
        self.d_hat_prev = 1.0
        self.step_count = 0
        self.robot_v = 0.0
        self.robot_omega = 0.0

        lidar = self._cast_lidar()
        obs = self._build_observation(lidar)
        return obs, {}

    def step(self, action):
        self.step_count += 1

        # 1. Map action to physical motor commands
        a0 = float(np.clip(action[0], -1.0, 1.0))
        a1 = float(np.clip(action[1], -1.0, 1.0))
        v = ((a0 + 1.0) / 2.0) * self.v_max   # [0, v_max]
        omega = a1 * self.omega_max              # [-omega_max, omega_max]

        # 2. Update robot kinematics (Euler integration)
        self.robot_x += v * np.cos(self.robot_theta) * self.dt
        self.robot_y += v * np.sin(self.robot_theta) * self.dt
        self.robot_theta += omega * self.dt
        # Normalize angle to [-pi, pi]
        self.robot_theta = ((self.robot_theta + np.pi) % (2 * np.pi)) - np.pi
        self.robot_v = v
        self.robot_omega = omega

        # 3. Update dynamic obstacles
        self._update_dynamic_obstacles()

        # 4. Check collision
        collision = self._check_collision()

        # 5. Cast LiDAR
        lidar = self._cast_lidar()

        # 6. Advance waypoints
        self._advance_waypoints()

        # 7. Compute normalized remaining path distance
        d_path_current = self._compute_d_path()
        d_hat_current = d_path_current / self.d_path_0 if self.d_path_0 > 0 else 0.0

        # 8. Determine reward and terminal state
        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        if collision:
            # HIGHEST PRIORITY: collision penalty
            reward = -self.P_collision
            terminated = True
            info["is_success"] = False
            info["termination"] = "collision"

        elif self._goal_reached():
            # Goal reached
            reward = self.R_goal
            terminated = True
            info["is_success"] = True
            info["termination"] = "goal"

        elif self.step_count >= self.t_max:
            # Timeout
            reward = -self.P_timeout
            truncated = True
            info["is_success"] = False
            info["termination"] = "timeout"

        else:
            # Dense step reward (Equation 8 of Final Formulation)
            # Base time penalty (strictly negative)
            reward = -self.w_t_prime

            # LiDAR clearance reward
            min_lidar = float(np.min(lidar))
            clearance = np.clip(min_lidar, 0.0, self.d_safe) / self.d_safe
            reward += self.lam * clearance

            # Normalized progress shaping
            reward += self.k * (self.d_hat_prev - self.gamma_ppo * d_hat_current)

        # Update previous D_hat for next step
        self.d_hat_prev = d_hat_current

        # Build observation
        obs = self._build_observation(lidar)

        # Optional rendering
        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Map Generation
    # ------------------------------------------------------------------
    def _generate_map(self):
        """Procedurally generate a random map with obstacles."""
        margin = self.robot_radius + 0.5  # keep robot/goal away from walls

        # Random robot start
        self.robot_x = self.np_random.uniform(margin, self.arena_width - margin)
        self.robot_y = self.np_random.uniform(margin, self.arena_height - margin)
        self.robot_theta = self.np_random.uniform(-np.pi, np.pi)

        # Random goal (must be > 1.0m from start)
        for _ in range(100):
            self.goal_x = self.np_random.uniform(margin, self.arena_width - margin)
            self.goal_y = self.np_random.uniform(margin, self.arena_height - margin)
            dist = np.sqrt(
                (self.goal_x - self.robot_x) ** 2
                + (self.goal_y - self.robot_y) ** 2
            )
            if dist >= 2.0:
                break

        # Generate static obstacles
        self.static_obstacles = []
        for _ in range(self.n_static_obstacles):
            obs = self._random_obstacle(static=True)
            if obs is not None:
                self.static_obstacles.append(obs)

        # Generate dynamic obstacles
        n_dyn = self.np_random.integers(
            self.n_dynamic_range[0], self.n_dynamic_range[1] + 1
        )
        self.dynamic_obstacles = []
        for _ in range(n_dyn):
            obs = self._random_dynamic_obstacle()
            if obs is not None:
                self.dynamic_obstacles.append(obs)

    def _random_obstacle(self, static=True):
        """Generate a single random static obstacle (circle or rectangle)."""
        margin = 0.5
        shape_type = self.np_random.choice(["circle", "rect"])

        for _ in range(20):
            if shape_type == "circle":
                if static:
                    r = self.np_random.uniform(0.2, 0.8)
                else:
                    r = self.np_random.uniform(0.2, 0.6)
                cx = self.np_random.uniform(r + margin, self.arena_width - r - margin)
                cy = self.np_random.uniform(r + margin, self.arena_height - r - margin)

                # Check it doesn't overlap robot or goal
                d_robot = np.sqrt((cx - self.robot_x) ** 2 + (cy - self.robot_y) ** 2)
                d_goal = np.sqrt((cx - self.goal_x) ** 2 + (cy - self.goal_y) ** 2)
                safe_dist = r + self.robot_radius + 0.3
                if d_robot > safe_dist and d_goal > safe_dist:
                    return {"type": "circle", "x": cx, "y": cy, "r": r}

            elif shape_type == "rect":
                w = self.np_random.uniform(0.3, 0.8)
                h = self.np_random.uniform(0.3, 0.8)
                x_min = self.np_random.uniform(margin, self.arena_width - w - margin)
                y_min = self.np_random.uniform(margin, self.arena_height - h - margin)
                x_max = x_min + w
                y_max = y_min + h

                # Check center distance to robot and goal
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                d_robot = np.sqrt((cx - self.robot_x) ** 2 + (cy - self.robot_y) ** 2)
                d_goal = np.sqrt((cx - self.goal_x) ** 2 + (cy - self.goal_y) ** 2)
                safe_dist = max(w, h) / 2 + self.robot_radius + 0.3
                if d_robot > safe_dist and d_goal > safe_dist:
                    return {
                        "type": "rect",
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                    }

        return None

    def _random_dynamic_obstacle(self):
        """Generate a single random dynamic obstacle (circle)."""
        margin = 0.5
        for _ in range(20):
            r = self.np_random.uniform(0.2, 0.6)
            cx = self.np_random.uniform(r + margin, self.arena_width - r - margin)
            cy = self.np_random.uniform(r + margin, self.arena_height - r - margin)

            d_robot = np.sqrt((cx - self.robot_x) ** 2 + (cy - self.robot_y) ** 2)
            if d_robot > r + self.robot_radius + 0.5:
                speed = self.np_random.uniform(*self.v_dyn_range)
                angle = self.np_random.uniform(0, 2 * np.pi)
                return {
                    "x": cx,
                    "y": cy,
                    "r": r,
                    "vx": speed * np.cos(angle),
                    "vy": speed * np.sin(angle),
                    "steps_until_turn": self.np_random.integers(20, 60),
                }
        return None

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def _update_dynamic_obstacles(self):
        """Move dynamic obstacles with random-walk + wall bouncing."""
        for obs in self.dynamic_obstacles:
            obs["x"] += obs["vx"] * self.dt
            obs["y"] += obs["vy"] * self.dt

            # Bounce off arena walls
            r = obs["r"]
            if obs["x"] - r < 0:
                obs["x"] = r
                obs["vx"] = abs(obs["vx"])
            elif obs["x"] + r > self.arena_width:
                obs["x"] = self.arena_width - r
                obs["vx"] = -abs(obs["vx"])

            if obs["y"] - r < 0:
                obs["y"] = r
                obs["vy"] = abs(obs["vy"])
            elif obs["y"] + r > self.arena_height:
                obs["y"] = self.arena_height - r
                obs["vy"] = -abs(obs["vy"])

            # Random direction change
            obs["steps_until_turn"] -= 1
            if obs["steps_until_turn"] <= 0:
                speed = np.sqrt(obs["vx"] ** 2 + obs["vy"] ** 2)
                angle = self.np_random.uniform(0, 2 * np.pi)
                obs["vx"] = speed * np.cos(angle)
                obs["vy"] = speed * np.sin(angle)
                obs["steps_until_turn"] = self.np_random.integers(20, 60)

    def _check_collision(self):
        """Check if the robot collides with any obstacle or arena wall."""
        rx, ry, rr = self.robot_x, self.robot_y, self.robot_radius

        # Wall collision
        if rx < rr or rx > self.arena_width - rr:
            return True
        if ry < rr or ry > self.arena_height - rr:
            return True

        # Static obstacle collision
        for obs in self.static_obstacles:
            if obs["type"] == "circle":
                dist = np.sqrt((rx - obs["x"]) ** 2 + (ry - obs["y"]) ** 2)
                if dist < rr + obs["r"]:
                    return True
            elif obs["type"] == "rect":
                # Closest point on AABB to robot center
                cx = np.clip(rx, obs["x_min"], obs["x_max"])
                cy = np.clip(ry, obs["y_min"], obs["y_max"])
                dist = np.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
                if dist < rr:
                    return True

        # Dynamic obstacle collision
        for obs in self.dynamic_obstacles:
            dist = np.sqrt((rx - obs["x"]) ** 2 + (ry - obs["y"]) ** 2)
            if dist < rr + obs["r"]:
                return True

        return False

    # ------------------------------------------------------------------
    # LiDAR Raycasting (Vectorized)
    # ------------------------------------------------------------------
    def _cast_lidar(self):
        """Cast 72 LiDAR rays, return raw distances (meters)."""
        angles = self.robot_theta + self.ray_angles
        dirs_x = np.cos(angles)
        dirs_y = np.sin(angles)

        distances = np.full(self.n_lidar_rays, self.d_max)

        # Arena walls (4 line segments)
        w, h = self.arena_width, self.arena_height
        wall_segments = [
            (0, 0, w, 0),  # bottom
            (0, h, w, h),  # top
            (0, 0, 0, h),  # left
            (w, 0, w, h),  # right
        ]
        for x1, y1, x2, y2 in wall_segments:
            d = self._ray_segment_batch(dirs_x, dirs_y, x1, y1, x2, y2)
            distances = np.minimum(distances, d)

        # Static obstacles
        for obs in self.static_obstacles:
            if obs["type"] == "circle":
                d = self._ray_circle_batch(dirs_x, dirs_y, obs["x"], obs["y"], obs["r"])
                distances = np.minimum(distances, d)
            elif obs["type"] == "rect":
                # Decompose AABB into 4 line segments
                xn, yn = obs["x_min"], obs["y_min"]
                xx, yx = obs["x_max"], obs["y_max"]
                segs = [
                    (xn, yn, xx, yn),  # bottom
                    (xn, yx, xx, yx),  # top
                    (xn, yn, xn, yx),  # left
                    (xx, yn, xx, yx),  # right
                ]
                for x1, y1, x2, y2 in segs:
                    d = self._ray_segment_batch(dirs_x, dirs_y, x1, y1, x2, y2)
                    distances = np.minimum(distances, d)

        # Dynamic obstacles (circles only)
        for obs in self.dynamic_obstacles:
            d = self._ray_circle_batch(dirs_x, dirs_y, obs["x"], obs["y"], obs["r"])
            distances = np.minimum(distances, d)

        return distances

    def _ray_segment_batch(self, dirs_x, dirs_y, x1, y1, x2, y2):
        """Vectorized ray-segment intersection for all 72 rays at once."""
        seg_dx = x2 - x1
        seg_dy = y2 - y1

        # Cross product denominator
        denom = dirs_x * seg_dy - dirs_y * seg_dx
        parallel = np.abs(denom) < 1e-12
        safe_denom = np.where(parallel, 1.0, denom)

        diff_x = x1 - self.robot_x
        diff_y = y1 - self.robot_y

        t = (diff_x * seg_dy - diff_y * seg_dx) / safe_denom
        s = (diff_x * dirs_y - diff_y * dirs_x) / safe_denom

        valid = ~parallel & (t > 1e-6) & (s >= 0.0) & (s <= 1.0)

        result = np.full(self.n_lidar_rays, np.inf)
        result[valid] = t[valid]
        return result

    def _ray_circle_batch(self, dirs_x, dirs_y, cx, cy, cr):
        """Vectorized ray-circle intersection for all 72 rays at once."""
        oc_x = self.robot_x - cx
        oc_y = self.robot_y - cy

        # Quadratic coefficients (a=1 for unit direction vectors)
        b = 2.0 * (dirs_x * oc_x + dirs_y * oc_y)
        c_val = oc_x ** 2 + oc_y ** 2 - cr ** 2

        disc = b ** 2 - 4.0 * c_val

        result = np.full(self.n_lidar_rays, np.inf)
        mask = disc >= 0
        if not np.any(mask):
            return result

        sqrt_disc = np.sqrt(disc[mask])
        b_m = b[mask]

        t1 = (-b_m - sqrt_disc) / 2.0
        t2 = (-b_m + sqrt_disc) / 2.0

        t_best = np.where(t1 > 1e-6, t1, np.where(t2 > 1e-6, t2, np.inf))
        result[mask] = t_best
        return result

    # ------------------------------------------------------------------
    # Path & Waypoint Logic
    # ------------------------------------------------------------------
    def _advance_waypoints(self):
        """Advance to next waypoint if close enough (intermediate only)."""
        while self.current_wp_idx < len(self.waypoints) - 1:
            wp = self.waypoints[self.current_wp_idx]
            dist = np.sqrt(
                (self.robot_x - wp[0]) ** 2 + (self.robot_y - wp[1]) ** 2
            )
            if dist < 0.3:
                self.current_wp_idx += 1
            else:
                break

    def _goal_reached(self):
        """Check if robot is within goal_tolerance of the goal."""
        dist = np.sqrt(
            (self.robot_x - self.goal_x) ** 2
            + (self.robot_y - self.goal_y) ** 2
        )
        return dist < self.goal_tolerance

    def _compute_d_path(self):
        """
        Compute total remaining path distance from robot to goal via waypoints.
        D_path(t) = dist(robot, current_wp) + sum(remaining segment lengths)
        """
        if self.current_wp_idx >= len(self.waypoints):
            return 0.0

        wp = self.waypoints[self.current_wp_idx]
        d = np.sqrt((self.robot_x - wp[0]) ** 2 + (self.robot_y - wp[1]) ** 2)

        for i in range(self.current_wp_idx, len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]
            d += np.sqrt((wp2[0] - wp1[0]) ** 2 + (wp2[1] - wp1[1]) ** 2)

        return d

    # ------------------------------------------------------------------
    # Observation Builder
    # ------------------------------------------------------------------
    def _build_observation(self, lidar):
        """Build the 76-element observation vector."""
        obs = np.zeros(76, dtype=np.float32)

        # [0:72] LiDAR normalized to [0, 1]
        obs[0:72] = np.clip(lidar / self.d_max, 0.0, 1.0)

        # Current waypoint (or goal if all waypoints consumed)
        if self.current_wp_idx < len(self.waypoints):
            wp_x, wp_y = self.waypoints[self.current_wp_idx]
        else:
            wp_x, wp_y = self.goal_x, self.goal_y

        # [72] Relative distance to waypoint, normalized [0, 1]
        wp_dist = np.sqrt(
            (self.robot_x - wp_x) ** 2 + (self.robot_y - wp_y) ** 2
        )
        obs[72] = np.clip(wp_dist / 10.0, 0.0, 1.0)

        # [73] Relative heading to waypoint, normalized [-1, 1]
        dx = wp_x - self.robot_x
        dy = wp_y - self.robot_y
        abs_angle = np.arctan2(dy, dx)
        rel_angle = abs_angle - self.robot_theta
        rel_angle = ((rel_angle + np.pi) % (2 * np.pi)) - np.pi
        obs[73] = np.clip(rel_angle / np.pi, -1.0, 1.0)

        # [74] Current linear velocity, normalized [0, 1]
        obs[74] = np.clip(self.robot_v / self.v_max, 0.0, 1.0)

        # [75] Current angular velocity, normalized [-1, 1]
        obs[75] = np.clip(self.robot_omega / self.omega_max, -1.0, 1.0)

        return obs

    # ------------------------------------------------------------------
    # Rendering (Optional, matplotlib-based)
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))

        self.ax.clear()
        self.ax.set_xlim(0, self.arena_width)
        self.ax.set_ylim(0, self.arena_height)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor("#1a1a2e")
        self.ax.set_title(
            f"Stage {self.curriculum_stage} | Step {self.step_count}/{self.t_max}",
            color="white",
            fontsize=12,
        )

        # Draw static obstacles
        for obs in self.static_obstacles:
            if obs["type"] == "circle":
                circle = mpatches.Circle(
                    (obs["x"], obs["y"]), obs["r"], color="#e94560", alpha=0.8
                )
                self.ax.add_patch(circle)
            elif obs["type"] == "rect":
                rect = mpatches.Rectangle(
                    (obs["x_min"], obs["y_min"]),
                    obs["x_max"] - obs["x_min"],
                    obs["y_max"] - obs["y_min"],
                    color="#e94560",
                    alpha=0.8,
                )
                self.ax.add_patch(rect)

        # Draw dynamic obstacles
        for obs in self.dynamic_obstacles:
            circle = mpatches.Circle(
                (obs["x"], obs["y"]), obs["r"], color="#ff6b35", alpha=0.8
            )
            self.ax.add_patch(circle)

        # Draw Theta* path
        if self.waypoints:
            wx = [p[0] for p in self.waypoints]
            wy = [p[1] for p in self.waypoints]
            self.ax.plot(wx, wy, "c--", linewidth=1, alpha=0.5)

        # Draw goal
        goal_circle = mpatches.Circle(
            (self.goal_x, self.goal_y),
            self.goal_tolerance,
            color="#00ff41",
            alpha=0.3,
        )
        self.ax.add_patch(goal_circle)
        self.ax.plot(self.goal_x, self.goal_y, "g*", markersize=15)

        # Draw robot
        robot_circle = mpatches.Circle(
            (self.robot_x, self.robot_y),
            self.robot_radius,
            color="#0096ff",
            alpha=0.9,
        )
        self.ax.add_patch(robot_circle)
        # Heading arrow
        arrow_len = 0.3
        self.ax.arrow(
            self.robot_x,
            self.robot_y,
            arrow_len * np.cos(self.robot_theta),
            arrow_len * np.sin(self.robot_theta),
            head_width=0.08,
            head_length=0.05,
            fc="white",
            ec="white",
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.ax = None
