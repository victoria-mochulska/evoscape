import numpy as np

from evoscape.modules.module_class import Node, UnstableNode


class LandscapePhaseAnalysisBase:
    def _allows_limit_cycles(self):
        return not all(isinstance(module, (Node, UnstableNode)) for module in self.module_list)

    def analytic_jacobian(self, t, q):
        raise NotImplementedError(
            "Provide analytic Jacobian"
        )

    @staticmethod
    def _coerce_phase_coordinate_array(q):
        coords = np.asarray(q, dtype=float)
        if coords.ndim == 1:
            if coords.shape[0] != 2:
                raise ValueError("Expected q to have shape (2,) for a single point.")
            return coords.reshape(2, 1), True
        if coords.ndim < 2 or coords.shape[0] != 2:
            raise ValueError("Expected q to have shape (2, ...) for phase analysis.")
        return coords, False

    @staticmethod
    def _classify_eigenvalues(eigvals, tol):
        real_parts = np.real(eigvals)
        if np.all(real_parts < -tol):
            return "attractor"
        if np.all(real_parts > tol):
            return "repeller"
        if np.any(real_parts < -tol) and np.any(real_parts > tol):
            return "saddle"
        return "center_or_degenerate"

    def numerical_jacobian(self, t, q, h=1e-4, use_jax=False):
        coords, squeeze_output = self._coerce_phase_coordinate_array(q)
        jac = np.zeros((2, 2) + coords.shape[1:], dtype=float)

        for axis in range(2):
            delta = np.zeros_like(coords)
            delta[axis] = h
            flow_plus = np.asarray(self(t, coords + delta, return_potentials=False, use_jax=use_jax), dtype=float)
            flow_minus = np.asarray(self(t, coords - delta, return_potentials=False, use_jax=use_jax), dtype=float)
            jac[:, axis] = (flow_plus - flow_minus) / (2.0 * h)

        if squeeze_output:
            return jac.reshape(2, 2)
        return jac

    def jacobian(self, t, q, method="auto", h=1e-4, use_jax=False):
        method = method.lower()
        if method in ("numeric", "numerical"):
            return self.numerical_jacobian(t, q, h=h, use_jax=use_jax)
        if method == "analytic":
            return self.analytic_jacobian(t, q)
        if method != "auto":
            raise ValueError("method must be one of 'auto', 'analytic', or 'numeric'.")

        try:
            return self.analytic_jacobian(t, q)
        except NotImplementedError:
            return self.numerical_jacobian(t, q, h=h, use_jax=use_jax)

    def _flow_vector(self, t, point, use_jax=False):
        coords = np.asarray(point, dtype=float).reshape(2, 1)
        flow = np.asarray(self(t, coords, return_potentials=False, use_jax=use_jax), dtype=float)
        return flow.reshape(2)

    def _flow_vector_from_pars(self, pars, point, use_jax=False):
        coords = np.asarray(point, dtype=float).reshape(2, 1)
        flow = np.asarray(self._eval_flow(pars, coords, return_potentials=False, use_jax=use_jax), dtype=float)
        return flow.reshape(2)

    def _numerical_jacobian_from_pars(self, pars, q, h=1e-4, use_jax=False):
        coords, squeeze_output = self._coerce_phase_coordinate_array(q)
        jac = np.zeros((2, 2) + coords.shape[1:], dtype=float)

        for axis in range(2):
            delta = np.zeros_like(coords)
            delta[axis] = h
            flow_plus = np.asarray(
                self._eval_flow(pars, coords + delta, return_potentials=False, use_jax=use_jax),
                dtype=float,
            )
            flow_minus = np.asarray(
                self._eval_flow(pars, coords - delta, return_potentials=False, use_jax=use_jax),
                dtype=float,
            )
            jac[:, axis] = (flow_plus - flow_minus) / (2.0 * h)

        if squeeze_output:
            return jac.reshape(2, 2)
        return jac

    def _rk4_step(self, t, q, dt, use_jax=False):
        k1 = np.asarray(self(t, q, return_potentials=False, use_jax=use_jax), dtype=float)
        k2 = np.asarray(self(t, q + 0.5 * dt * k1, return_potentials=False, use_jax=use_jax), dtype=float)
        k3 = np.asarray(self(t, q + 0.5 * dt * k2, return_potentials=False, use_jax=use_jax), dtype=float)
        k4 = np.asarray(self(t, q + dt * k3, return_potentials=False, use_jax=use_jax), dtype=float)
        return q + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def _rk4_step_batched(self, pars, q, dt, use_jax=False):
        k1 = np.asarray(self._eval_flow(pars, q, return_potentials=False, use_jax=use_jax), dtype=float)
        k2 = np.asarray(self._eval_flow(pars, q + 0.5 * dt * k1, return_potentials=False, use_jax=use_jax), dtype=float)
        k3 = np.asarray(self._eval_flow(pars, q + 0.5 * dt * k2, return_potentials=False, use_jax=use_jax), dtype=float)
        k4 = np.asarray(self._eval_flow(pars, q + dt * k3, return_potentials=False, use_jax=use_jax), dtype=float)
        return q + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    @staticmethod
    def _jax_modules():
        try:
            import jax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            import sys
        except ImportError as exc:
            raise ImportError("use_jax=True requires jax. Install with `poetry install -E jax`.") from exc
        landscape_module = sys.modules.get("evoscape.landscapes.landscape_class")
        if landscape_module is not None:
            landscape_module.jnp = jnp
        return jax, jnp

    def _integrate_manifold_branches_batched_jax(
        self,
        pars,
        starts,
        dt,
        n_steps,
        x_range,
        y_range,
        fixed_point_positions=None,
        exclude_indices=None,
        termination_tol=1e-2,
        velocity_tol=1e-3,
    ):
        jax, jnp = self._jax_modules()
        starts = np.asarray(starts, dtype=float)
        n_branches = starts.shape[1]
        x_min, x_max = sorted(map(float, x_range))
        y_min, y_max = sorted(map(float, y_range))
        x_pad = 0.1 * max(x_max - x_min, termination_tol * 10.0)
        y_pad = 0.1 * max(y_max - y_min, termination_tol * 10.0)
        dt_scale = max(abs(float(dt)), 1e-12)
        has_fixed_points = fixed_point_positions is not None and np.asarray(fixed_point_positions).size > 0
        fixed_points_jax = (
            jnp.asarray(fixed_point_positions, dtype=float)
            if has_fixed_points
            else jnp.empty((0, 2), dtype=float)
        )
        exclude_indices_jax = (
            jnp.asarray(exclude_indices, dtype=int)
            if exclude_indices is not None
            else -jnp.ones(n_branches, dtype=int)
        )
        jax_pars = {
            key: (jnp.asarray(value, dtype=float) if key != "empty" else value)
            for key, value in pars.items()
        }
        fp_column_indices = jnp.arange(fixed_points_jax.shape[0])

        def rk4(q):
            k1 = self._eval_flow(jax_pars, q, return_potentials=False, use_jax=True)
            k2 = self._eval_flow(jax_pars, q + 0.5 * dt * k1, return_potentials=False, use_jax=True)
            k3 = self._eval_flow(jax_pars, q + 0.5 * dt * k2, return_potentials=False, use_jax=True)
            k4 = self._eval_flow(jax_pars, q + dt * k3, return_potentials=False, use_jax=True)
            return q + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        def scan_step(state, step):
            current, active, lengths, step_velocity, terminated_fp, terminated_boundary, terminated_nonfinite = state
            safe_current = jnp.where(active[None, :], current, 0.0)
            next_positions = rk4(safe_current)
            finite = jnp.all(jnp.isfinite(next_positions), axis=0)
            live = active & finite
            nonfinite = active & ~finite
            speeds = jnp.linalg.norm(next_positions - safe_current, axis=0) / dt_scale
            boundary = live & (
                (next_positions[0] < x_min - x_pad)
                | (next_positions[0] > x_max + x_pad)
                | (next_positions[1] < y_min - y_pad)
                | (next_positions[1] > y_max + y_pad)
            )
            nearest = -jnp.ones(n_branches, dtype=int)
            converged = jnp.zeros(n_branches, dtype=bool)
            if has_fixed_points:
                distances = jnp.linalg.norm(next_positions.T[:, None, :] - fixed_points_jax[None, :, :], axis=2)
                distances = jnp.where(
                    fp_column_indices[None, :] == exclude_indices_jax[:, None],
                    jnp.inf,
                    distances,
                )
                nearest = jnp.argmin(distances, axis=1)
                nearest_distance = distances[jnp.arange(n_branches), nearest]
                converged = (
                    live
                    & ~boundary
                    & jnp.isfinite(nearest_distance)
                    & (nearest_distance <= termination_tol)
                    & (speeds <= velocity_tol)
                )

            finished = nonfinite | boundary | converged
            active = active & ~finished
            current = jnp.where(active[None, :], next_positions, jnp.nan)
            lengths = jnp.where(live, step + 2, lengths)
            step_velocity = jnp.where(live, speeds, step_velocity)
            terminated_fp = jnp.where(converged, nearest, terminated_fp)
            terminated_boundary = terminated_boundary | boundary
            terminated_nonfinite = terminated_nonfinite | nonfinite
            stored_positions = jnp.where(live[None, :], next_positions, jnp.nan)
            return (
                current,
                active,
                lengths,
                step_velocity,
                terminated_fp,
                terminated_boundary,
                terminated_nonfinite,
            ), stored_positions

        @jax.jit
        def run_scan(starts_jax):
            state = (
                starts_jax,
                jnp.ones(n_branches, dtype=bool),
                jnp.ones(n_branches, dtype=int),
                jnp.full(n_branches, jnp.nan, dtype=float),
                -jnp.ones(n_branches, dtype=int),
                jnp.zeros(n_branches, dtype=bool),
                jnp.zeros(n_branches, dtype=bool),
            )
            return jax.lax.scan(scan_step, state, jnp.arange(int(n_steps)))

        final_state, trajectory_steps = run_scan(jnp.asarray(starts, dtype=float))
        _, _, lengths, step_velocity, terminated_fp, terminated_boundary, terminated_nonfinite = final_state
        trajectories_buffer = np.concatenate(
            (starts[None, :, :], np.asarray(trajectory_steps, dtype=float)),
            axis=0,
        )
        lengths = np.asarray(lengths, dtype=int)
        trajectories = [trajectories_buffer[:lengths[idx], :, idx].copy() for idx in range(n_branches)]
        endpoints = np.array([trajectory[-1].copy() for trajectory in trajectories], dtype=float)
        return {
            "trajectories": trajectories,
            "endpoints": endpoints,
            "step_velocity": np.asarray(step_velocity, dtype=float),
            "terminated_fixed_point_index": np.asarray(terminated_fp, dtype=int),
            "terminated_at_boundary": np.asarray(terminated_boundary, dtype=bool),
            "terminated_nonfinite": np.asarray(terminated_nonfinite, dtype=bool),
        }

    def _integrate_manifold_branches_batched(
        self,
        pars,
        starts,
        dt,
        n_steps,
        x_range,
        y_range,
        fixed_point_positions=None,
        exclude_indices=None,
        termination_tol=1e-2,
        velocity_tol=1e-3,
        use_jax=False,
    ):
        starts, _ = self._coerce_phase_coordinate_array(starts)
        n_branches = starts.shape[1]
        if n_branches == 0:
            return {
                "trajectories": [],
                "endpoints": np.empty((0, 2), dtype=float),
                "step_velocity": np.empty((0,), dtype=float),
                "terminated_fixed_point_index": np.empty((0,), dtype=int),
                "terminated_at_boundary": np.empty((0,), dtype=bool),
                "terminated_nonfinite": np.empty((0,), dtype=bool),
            }

        x_min, x_max = sorted(map(float, x_range))
        y_min, y_max = sorted(map(float, y_range))
        x_pad = 0.1 * max(x_max - x_min, termination_tol * 10.0)
        y_pad = 0.1 * max(y_max - y_min, termination_tol * 10.0)
        dt_scale = max(abs(float(dt)), 1e-12)

        if fixed_point_positions is not None:
            fixed_point_positions = np.asarray(fixed_point_positions, dtype=float)
            if fixed_point_positions.ndim != 2:
                fixed_point_positions = fixed_point_positions.reshape(-1, 2)
        if exclude_indices is not None:
            exclude_indices = np.asarray(exclude_indices, dtype=int).reshape(-1)
            if exclude_indices.size != n_branches:
                raise ValueError("exclude_indices must have one entry per branch.")
        if use_jax:
            return self._integrate_manifold_branches_batched_jax(
                pars,
                starts,
                dt,
                n_steps,
                x_range,
                y_range,
                fixed_point_positions=fixed_point_positions,
                exclude_indices=exclude_indices,
                termination_tol=termination_tol,
                velocity_tol=velocity_tol,
            )

        trajectories_buffer = np.full((int(n_steps) + 1, 2, n_branches), np.nan, dtype=float)
        trajectories_buffer[0] = starts
        lengths = np.ones(n_branches, dtype=int)
        current = starts.copy()
        active = np.ones(n_branches, dtype=bool)
        step_velocity = np.full(n_branches, np.nan, dtype=float)
        terminated_fixed_point_index = np.full(n_branches, -1, dtype=int)
        terminated_at_boundary = np.zeros(n_branches, dtype=bool)
        terminated_nonfinite = np.zeros(n_branches, dtype=bool)

        for step in range(int(n_steps)):
            if not np.any(active):
                break

            active_indices = np.flatnonzero(active)
            current_active = current[:, active_indices]
            next_active = self._rk4_step_batched(pars, current_active, dt, use_jax=use_jax)
            finite_mask = np.all(np.isfinite(next_active), axis=0)

            if np.any(~finite_mask):
                nonfinite_indices = active_indices[~finite_mask]
                terminated_nonfinite[nonfinite_indices] = True
                active[nonfinite_indices] = False
                current[:, nonfinite_indices] = np.nan

            if not np.any(finite_mask):
                continue

            live_indices = active_indices[finite_mask]
            live_current = current_active[:, finite_mask]
            live_next = next_active[:, finite_mask]
            trajectories_buffer[step + 1][:, live_indices] = live_next
            lengths[live_indices] = step + 2
            speeds = np.linalg.norm(live_next - live_current, axis=0) / dt_scale
            step_velocity[live_indices] = speeds
            current[:, live_indices] = live_next

            finished = np.zeros(live_indices.size, dtype=bool)
            boundary = (
                (live_next[0] < x_min - x_pad)
                | (live_next[0] > x_max + x_pad)
                | (live_next[1] < y_min - y_pad)
                | (live_next[1] > y_max + y_pad)
            )
            if np.any(boundary):
                boundary_indices = live_indices[boundary]
                terminated_at_boundary[boundary_indices] = True
                finished |= boundary

            if fixed_point_positions is not None and fixed_point_positions.size:
                check_mask = ~finished
                if np.any(check_mask):
                    check_indices = live_indices[check_mask]
                    check_points = live_next[:, check_mask].T
                    distances = np.linalg.norm(
                        check_points[:, None, :] - fixed_point_positions[None, :, :],
                        axis=2,
                    )
                    if exclude_indices is not None:
                        check_excludes = exclude_indices[check_indices]
                        valid_excludes = (
                            (check_excludes >= 0)
                            & (check_excludes < distances.shape[1])
                        )
                        if np.any(valid_excludes):
                            distances[
                                np.arange(check_indices.size)[valid_excludes],
                                check_excludes[valid_excludes],
                            ] = np.inf
                    nearest = np.argmin(distances, axis=1)
                    nearest_distance = distances[np.arange(check_indices.size), nearest]
                    converged = (
                        np.isfinite(nearest_distance)
                        & (nearest_distance <= termination_tol)
                        & (speeds[check_mask] <= velocity_tol)
                    )
                    if np.any(converged):
                        converged_indices = check_indices[converged]
                        terminated_fixed_point_index[converged_indices] = nearest[converged]
                        check_positions = np.flatnonzero(check_mask)
                        finished[check_positions[converged]] = True

            if np.any(finished):
                finished_indices = live_indices[finished]
                active[finished_indices] = False
                current[:, finished_indices] = np.nan

        trajectories = [trajectories_buffer[:lengths[idx], :, idx].copy() for idx in range(n_branches)]
        endpoints = np.array([trajectory[-1].copy() for trajectory in trajectories], dtype=float)
        return {
            "trajectories": trajectories,
            "endpoints": endpoints,
            "step_velocity": step_velocity,
            "terminated_fixed_point_index": terminated_fixed_point_index,
            "terminated_at_boundary": terminated_at_boundary,
            "terminated_nonfinite": terminated_nonfinite,
        }

    def _integrate_manifold_branch(
        self,
        t,
        start,
        dt,
        n_steps,
        x_range,
        y_range,
        fixed_point_positions=None,
        exclude_index=None,
        termination_tol=1e-2,
        velocity_tol=1e-3,
        return_metadata=False,
        use_jax=False,
    ):
        exclude_indices = None if exclude_index is None else np.array([int(exclude_index)], dtype=int)
        result = self._integrate_manifold_branches_batched(
            self._get_all_pars(t),
            np.asarray(start, dtype=float).reshape(2, 1),
            dt,
            n_steps,
            x_range,
            y_range,
            fixed_point_positions=fixed_point_positions,
            exclude_indices=exclude_indices,
            termination_tol=termination_tol,
            velocity_tol=velocity_tol,
            use_jax=use_jax,
        )
        trajectory = np.asarray(result["trajectories"][0], dtype=float)
        if not return_metadata:
            return trajectory

        terminated_fixed_point_index = int(result["terminated_fixed_point_index"][0])
        return {
            "trajectory": trajectory.copy(),
            "endpoint": np.asarray(result["endpoints"][0], dtype=float).copy(),
            "step_velocity": float(result["step_velocity"][0]),
            "terminated_fixed_point_index": None if terminated_fixed_point_index < 0 else terminated_fixed_point_index,
            "terminated_at_boundary": bool(result["terminated_at_boundary"][0]),
            "terminated_nonfinite": bool(result["terminated_nonfinite"][0]),
        }

    @staticmethod
    def _detect_cycle_signature(traj, dt, fp_tol, vel_tol):
        if traj.shape[0] < 16:
            return None

        deltas = np.diff(traj, axis=0)
        step_norms = np.linalg.norm(deltas, axis=1)
        if step_norms.size == 0:
            return None

        mean_speed = step_norms.mean() / max(dt, 1e-12)
        centroid = traj.mean(axis=0)
        radial = np.linalg.norm(traj - centroid, axis=1)
        mean_radius = float(radial.mean())
        radius_span = float(radial.max() - radial.min())

        if mean_speed < vel_tol:
            return None
        if mean_radius < fp_tol * 2.0 or max(mean_radius, radius_span) < fp_tol:
            return None

        max_lag = min(traj.shape[0] - 8, 512)
        if max_lag < 8:
            return None

        candidate_lags = np.arange(8, max_lag + 1)
        recurrence = np.linalg.norm(traj[-1] - traj[-1 - candidate_lags], axis=1)
        best_idx = int(np.argmin(recurrence))
        best_lag = int(candidate_lags[best_idx])
        best_distance = float(recurrence[best_idx])
        recurrence_tol = max(fp_tol * 2.5, step_norms.mean() * 2.0, 1e-3)
        if best_distance > recurrence_tol:
            return None

        segment_len = min(max(6, best_lag // 2), 16)
        current_segment = traj[-segment_len:]
        previous_segment = traj[-segment_len - best_lag:-best_lag]
        if previous_segment.shape[0] != segment_len:
            return None
        segment_error = float(np.mean(np.linalg.norm(current_segment - previous_segment, axis=1)))
        current_radius = np.linalg.norm(current_segment - centroid, axis=1).mean()
        previous_radius = np.linalg.norm(previous_segment - centroid, axis=1).mean()
        if segment_error > max(fp_tol * 3.0, step_norms.mean() * 2.0):
            return None
        if abs(current_radius - previous_radius) > max(fp_tol * 2.0, 0.25 * mean_radius):
            return None

        cycle_trajectory = traj[-best_lag - 1:].copy()
        cycle_centroid = cycle_trajectory.mean(axis=0)
        cycle_radial = np.linalg.norm(cycle_trajectory - cycle_centroid, axis=1)
        cycle_mean_radius = float(cycle_radial.mean())
        cycle_radius_span = float(cycle_radial.max() - cycle_radial.min())

        return {
            "center": cycle_centroid,
            "radius_mean": cycle_mean_radius,
            "radius_span": cycle_radius_span,
            "period_steps": best_lag,
            "period": best_lag * dt,
            "trajectory": cycle_trajectory,
            "recurrence_distance": best_distance,
            "segment_error": segment_error,
        }


    @staticmethod
    def _mark_grid_point(mask, iy, ix, thickness=1):
        y0 = max(0, int(iy) - int(thickness))
        y1 = min(mask.shape[0], int(iy) + int(thickness) + 1)
        x0 = max(0, int(ix) - int(thickness))
        x1 = min(mask.shape[1], int(ix) + int(thickness) + 1)
        mask[y0:y1, x0:x1] = True

    def _mark_curve_on_grid(self, curve, x_coords, y_coords, mask, thickness=1):
        curve = np.asarray(curve, dtype=float)
        if curve.ndim != 2 or curve.shape[0] < 1:
            return

        x_min = float(x_coords[0])
        y_min = float(y_coords[0])
        dx = max(float(np.mean(np.diff(x_coords))), 1e-12)
        dy = max(float(np.mean(np.diff(y_coords))), 1e-12)
        sample_spacing = 0.5 * min(dx, dy)

        for point in curve:
            ix = int(np.rint((point[0] - x_min) / dx))
            iy = int(np.rint((point[1] - y_min) / dy))
            if 0 <= ix < mask.shape[1] and 0 <= iy < mask.shape[0]:
                self._mark_grid_point(mask, iy, ix, thickness=thickness)

        for start, end in zip(curve[:-1], curve[1:]):
            segment = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
            length = float(np.linalg.norm(segment))
            n_samples = max(2, int(np.ceil(length / sample_spacing)) + 1)
            for alpha in np.linspace(0.0, 1.0, n_samples):
                point = (1.0 - alpha) * start + alpha * end
                ix = int(np.rint((point[0] - x_min) / dx))
                iy = int(np.rint((point[1] - y_min) / dy))
                if 0 <= ix < mask.shape[1] and 0 <= iy < mask.shape[0]:
                    self._mark_grid_point(mask, iy, ix, thickness=thickness)

    @staticmethod
    def _label_grid_regions(open_mask):
        labels = -np.ones(open_mask.shape, dtype=int)
        region_id = 0
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for start_y, start_x in np.argwhere(open_mask):
            if labels[start_y, start_x] != -1:
                continue

            stack = [(int(start_y), int(start_x))]
            labels[start_y, start_x] = region_id
            while stack:
                y, x = stack.pop()
                for dy, dx in neighbors:
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= open_mask.shape[0] or nx < 0 or nx >= open_mask.shape[1]:
                        continue
                    if not open_mask[ny, nx] or labels[ny, nx] != -1:
                        continue
                    labels[ny, nx] = region_id
                    stack.append((ny, nx))
            region_id += 1

        return labels

    @staticmethod
    def _select_region_samples(region_mask, max_samples=5):
        coords = np.argwhere(region_mask)
        if coords.size == 0:
            return []

        centroid = coords.mean(axis=0)
        distances = np.linalg.norm(coords - centroid[None, :], axis=1)
        order = np.argsort(distances)
        ordered_coords = coords[order]

        if ordered_coords.shape[0] <= max_samples:
            return [tuple(map(int, coord)) for coord in ordered_coords]

        sample_positions = np.linspace(0, ordered_coords.shape[0] - 1, max_samples, dtype=int)
        return [tuple(map(int, ordered_coords[pos])) for pos in sample_positions]

    @staticmethod
    def _fill_barrier_labels(labels, boundary_mask):
        filled = np.asarray(labels, dtype=int).copy()
        unresolved = boundary_mask & (filled < 0)
        if not np.any(unresolved):
            return filled

        neighbors = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )

        for _ in range(max(filled.shape)):
            pending = np.argwhere(unresolved)
            if pending.size == 0:
                break

            changed = False
            for y, x in pending:
                neighbor_labels = []
                for dy, dx in neighbors:
                    ny = int(y + dy)
                    nx = int(x + dx)
                    if ny < 0 or ny >= filled.shape[0] or nx < 0 or nx >= filled.shape[1]:
                        continue
                    if filled[ny, nx] >= 0:
                        neighbor_labels.append(int(filled[ny, nx]))
                if not neighbor_labels:
                    continue
                values, counts = np.unique(neighbor_labels, return_counts=True)
                filled[int(y), int(x)] = int(values[np.argmax(counts)])
                unresolved[int(y), int(x)] = False
                changed = True

            if not changed:
                break

        return filled

    @staticmethod
    def _build_fixed_point_attractors(fixed_points):
        points = np.asarray(fixed_points["points"], dtype=float)
        attracting_indices = np.flatnonzero(np.asarray(fixed_points["attracting_mask"], dtype=bool))

        attractors = []
        fixed_label_map = {}
        for fixed_index in attracting_indices:
            label_id = len(attractors)
            fixed_label_map[int(fixed_index)] = label_id
            attractors.append(
                {
                    "id": label_id,
                    "type": "fixed_point",
                    "point": points[fixed_index].copy(),
                    "fixed_point_index": int(fixed_index),
                    "eigenvalues": np.asarray(fixed_points["eigenvalues"][fixed_index]).copy(),
                }
            )
        return attractors, fixed_label_map

    def _trace_saddle_manifold_geometry(
        self,
        t,
        fixed_points,
        x_range,
        y_range,
        step_size,
        n_steps,
        perturbation,
        stability_tol,
        termination_tol,
        velocity_tol,
        use_jax=False,
    ):
        points = np.asarray(fixed_points["points"], dtype=float)
        span = max(abs(float(x_range[1]) - float(x_range[0])), abs(float(y_range[1]) - float(y_range[0])), 1.0)
        seed_offset = float(perturbation) if perturbation is not None else 1e-3 * span
        branch_dt = abs(float(step_size))

        saddle_entries = []
        stable_seeds = []
        unstable_seeds = []
        branch_excludes = []
        saddle_indices = np.flatnonzero(np.asarray(fixed_points["stability"], dtype=object) == "saddle")
        for fixed_index in saddle_indices:
            point = points[fixed_index]
            jacobian = np.asarray(fixed_points["jacobians"][fixed_index], dtype=float).reshape(2, 2)
            eigenvalues, eigenvectors = np.linalg.eig(jacobian)

            stable_candidates = np.flatnonzero(np.real(eigenvalues) < -stability_tol)
            unstable_candidates = np.flatnonzero(np.real(eigenvalues) > stability_tol)
            if stable_candidates.size == 0 or unstable_candidates.size == 0:
                continue

            stable_index = int(stable_candidates[np.argmin(np.real(eigenvalues[stable_candidates]))])
            unstable_index = int(unstable_candidates[np.argmax(np.real(eigenvalues[unstable_candidates]))])

            stable_vector = np.asarray(np.real(eigenvectors[:, stable_index]), dtype=float)
            unstable_vector = np.asarray(np.real(eigenvectors[:, unstable_index]), dtype=float)
            stable_norm = np.linalg.norm(stable_vector)
            unstable_norm = np.linalg.norm(unstable_vector)
            if stable_norm <= 0.0 or unstable_norm <= 0.0:
                continue

            stable_vector /= stable_norm
            unstable_vector /= unstable_norm

            for sign in (-1.0, 1.0):
                stable_seeds.append(point + sign * seed_offset * stable_vector)
                unstable_seeds.append(point + sign * seed_offset * unstable_vector)
                branch_excludes.append(int(fixed_index))

            saddle_entries.append(
                {
                    "fixed_point_index": int(fixed_index),
                    "point": point.copy(),
                    "jacobian": jacobian,
                    "eigenvalues": np.asarray(eigenvalues).copy(),
                    "stable_eigenvalue": eigenvalues[stable_index],
                    "unstable_eigenvalue": eigenvalues[unstable_index],
                    "stable_vector": stable_vector.copy(),
                    "unstable_vector": unstable_vector.copy(),
                }
            )

        if stable_seeds:
            pars = self._get_all_pars(t)
            branch_excludes = np.asarray(branch_excludes, dtype=int)
            stable_batch = self._integrate_manifold_branches_batched(
                pars,
                np.column_stack(stable_seeds),
                -branch_dt,
                n_steps,
                x_range,
                y_range,
                fixed_point_positions=points,
                exclude_indices=branch_excludes,
                termination_tol=termination_tol,
                velocity_tol=velocity_tol,
                use_jax=use_jax,
            )
            unstable_batch = self._integrate_manifold_branches_batched(
                pars,
                np.column_stack(unstable_seeds),
                branch_dt,
                n_steps,
                x_range,
                y_range,
                fixed_point_positions=points,
                exclude_indices=branch_excludes,
                termination_tol=termination_tol,
                velocity_tol=velocity_tol,
                use_jax=use_jax,
            )
        else:
            stable_batch = {
                "trajectories": [],
            }
            unstable_batch = {
                "trajectories": [],
                "endpoints": np.empty((0, 2), dtype=float),
                "step_velocity": np.empty((0,), dtype=float),
                "terminated_fixed_point_index": np.empty((0,), dtype=int),
                "terminated_at_boundary": np.empty((0,), dtype=bool),
                "terminated_nonfinite": np.empty((0,), dtype=bool),
            }

        saddles = []
        branch_index = 0
        for entry in saddle_entries:
            point = np.asarray(entry["point"], dtype=float)
            stable_branches = []
            unstable_branches = []
            unstable_meta = []
            for _ in (-1.0, 1.0):
                stable_branch = np.asarray(stable_batch["trajectories"][branch_index], dtype=float)
                unstable_branch = np.asarray(unstable_batch["trajectories"][branch_index], dtype=float)
                stable_branches.append(np.vstack((point.copy(), stable_branch)))
                unstable_branches.append(np.vstack((point.copy(), unstable_branch)))
                unstable_target = int(unstable_batch["terminated_fixed_point_index"][branch_index])
                unstable_meta.append(
                    {
                        "trajectory": unstable_branch.copy(),
                        "endpoint": np.asarray(unstable_batch["endpoints"][branch_index], dtype=float).copy(),
                        "step_velocity": float(unstable_batch["step_velocity"][branch_index]),
                        "terminated_fixed_point_index": None if unstable_target < 0 else unstable_target,
                        "terminated_at_boundary": bool(unstable_batch["terminated_at_boundary"][branch_index]),
                        "terminated_nonfinite": bool(unstable_batch["terminated_nonfinite"][branch_index]),
                    }
                )
                branch_index += 1

            saddle = dict(entry)
            saddle["stable"] = stable_branches
            saddle["unstable"] = unstable_branches
            saddle["_unstable_meta"] = unstable_meta
            saddles.append(saddle)

        return {
            "saddles": saddles,
            "fixed_points": fixed_points,
        }

    def _classify_unstable_branch_connection(
        self,
        branch,
        fixed_points,
        attractors,
        x_range,
        y_range,
        fp_tol,
        termination_tol=1e-2,
        source_fixed_point_index=None,
        branch_meta=None,
    ):
        branch = np.asarray(branch, dtype=float)
        if branch.ndim != 2 or branch.shape[0] == 0:
            return {"target_type": None, "target_ref": None, "endpoint": np.full(2, np.nan, dtype=float)}

        endpoint = np.asarray(branch[-1], dtype=float).copy()
        terminated_fixed_point_index = None
        if branch_meta is not None:
            endpoint = np.asarray(branch_meta.get("endpoint", endpoint), dtype=float).reshape(2)
            if branch_meta.get("terminated_fixed_point_index") is not None:
                terminated_fixed_point_index = int(branch_meta["terminated_fixed_point_index"])

        points = np.asarray(fixed_points["points"], dtype=float)
        if terminated_fixed_point_index is None and points.size and np.all(np.isfinite(endpoint)):
            distances = np.linalg.norm(points - endpoint[None, :], axis=1)
            if source_fixed_point_index is not None and 0 <= int(source_fixed_point_index) < distances.size:
                distances[int(source_fixed_point_index)] = np.inf
            nearest_index = int(np.argmin(distances))
            if np.isfinite(distances[nearest_index]):
                proximity_tol = max(float(termination_tol), float(fp_tol) * 4.0)
                if distances[nearest_index] <= proximity_tol:
                    terminated_fixed_point_index = nearest_index

        if terminated_fixed_point_index is not None:
            fixed_index = int(terminated_fixed_point_index)
            if np.asarray(fixed_points["attracting_mask"], dtype=bool)[fixed_index]:
                return {
                    "target_type": "attractor",
                    "target_ref": fixed_index,
                    "endpoint": endpoint,
                }
            if str(np.asarray(fixed_points["stability"], dtype=object)[fixed_index]) == "saddle":
                return {
                    "target_type": "saddle",
                    "target_ref": fixed_index,
                    "endpoint": endpoint,
                }
            return {"target_type": None, "target_ref": None, "endpoint": endpoint}

        escaped = False
        if branch_meta is not None:
            escaped = bool(branch_meta.get("terminated_at_boundary", False) or branch_meta.get("terminated_nonfinite", False))
        if x_range is not None and y_range is not None and np.all(np.isfinite(endpoint)):
            x_min, x_max = sorted(map(float, x_range))
            y_min, y_max = sorted(map(float, y_range))
            x_pad = 0.1 * max(x_max - x_min, termination_tol * 10.0)
            y_pad = 0.1 * max(y_max - y_min, termination_tol * 10.0)
            escaped = escaped or (
                endpoint[0] < x_min - x_pad
                or endpoint[0] > x_max + x_pad
                or endpoint[1] < y_min - y_pad
                or endpoint[1] > y_max + y_pad
            )
        if escaped or not self._allows_limit_cycles():
            return {"target_type": None, "target_ref": None, "endpoint": endpoint}

        cycle_match = self._match_endpoint_to_cycle_attractor(endpoint, attractors, fp_tol)
        if cycle_match is None:
            return {"target_type": None, "target_ref": None, "endpoint": endpoint}

        return {"target_type": "cycle", "target_ref": int(cycle_match["id"]), "endpoint": endpoint}

    @staticmethod
    def _match_endpoint_to_cycle_attractor(endpoint, attractors, fp_tol):
        endpoint = np.asarray(endpoint, dtype=float).reshape(2)
        if not np.all(np.isfinite(endpoint)):
            return None

        best_match = None
        best_distance = np.inf
        for attractor in attractors:
            if attractor.get("type") != "cycle":
                continue

            traj = np.asarray(attractor.get("trajectory"), dtype=float)
            if traj.ndim != 2 or traj.shape[0] == 0:
                continue

            distances = np.linalg.norm(traj - endpoint[None, :], axis=1)
            min_distance = float(np.min(distances))
            if traj.shape[0] >= 2:
                step_scale = float(np.mean(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
            else:
                step_scale = 0.0

            distance_tol = max(
                float(fp_tol) * 6.0,
                3.0 * step_scale,
                0.1 * float(attractor.get("radius_mean", 0.0)),
                5e-2,
            )
            if min_distance > distance_tol:
                continue
            if min_distance < best_distance:
                best_distance = min_distance
                best_match = attractor

        return best_match

    @staticmethod
    def _build_phase_objects(attractors, saddles):
        objects = []
        for attractor in sorted(attractors, key=lambda attractor: int(attractor["id"])):
            object_id = int(attractor["id"])
            if attractor["type"] == "fixed_point":
                obj = {
                    "id": object_id,
                    "type": "attractor",
                    "fixed_point_index": int(attractor["fixed_point_index"]),
                }
                if attractor.get("node_id") is not None:
                    obj["node_id"] = int(attractor["node_id"])
            elif attractor["type"] == "cycle":
                obj = {
                    "id": object_id,
                    "type": "cycle",
                    "center": np.asarray(attractor["center"], dtype=float).copy(),
                    "period": float(attractor["period"]),
                    "radius_mean": float(attractor["radius_mean"]),
                    "radius_span": float(attractor["radius_span"]),
                }
                if attractor.get("node_id") is not None:
                    obj["node_id"] = int(attractor["node_id"])
            else:
                continue
            objects.append(obj)

        for saddle in sorted(saddles, key=lambda saddle: int(saddle["id"])):
            objects.append(
                {
                    "id": int(saddle["id"]),
                    "type": "saddle",
                    "fixed_point_index": int(saddle["fixed_point_index"]),
                }
            )

        return objects

    def _annotate_saddle_connections(
        self,
        saddle_manifolds,
        fixed_points,
        attractors,
        fp_tol,
        x_range=None,
        y_range=None,
        termination_tol=None,
    ):
        if x_range is None or y_range is None:
            raise ValueError("x_range and y_range are required for saddle connection annotation.")
        termination_tol = float(fp_tol if termination_tol is None else termination_tol)
        saddles = list(saddle_manifolds.get("saddles", ()))
        fixed_label_map = {
            int(attractor["fixed_point_index"]): int(attractor["id"])
            for attractor in attractors
            if attractor["type"] == "fixed_point"
        }

        pending_connections = []
        for saddle in saddles:
            source_fixed_point_index = int(saddle["fixed_point_index"])
            branch_meta_list = saddle.get("_unstable_meta", ())
            branch_connections = []
            for branch_index, branch in enumerate(saddle.get("unstable", ())):
                branch_meta = branch_meta_list[branch_index] if branch_index < len(branch_meta_list) else None
                branch_connections.append(
                    self._classify_unstable_branch_connection(
                        branch,
                        fixed_points,
                        attractors,
                        x_range,
                        y_range,
                        fp_tol,
                        termination_tol=termination_tol,
                        source_fixed_point_index=source_fixed_point_index,
                        branch_meta=branch_meta,
                    )
                )
            pending_connections.append(branch_connections)

        saddle_index_map = {}
        first_saddle_index = len(attractors)
        for saddle_offset, saddle in enumerate(saddles):
            saddle_index = first_saddle_index + saddle_offset
            saddle["id"] = saddle_index
            saddle_index_map[int(saddle["fixed_point_index"])] = saddle_index

        for saddle, branch_connections in zip(saddles, pending_connections):
            unstable_connections = []
            for connection in branch_connections:
                target_type = connection["target_type"]
                target_ref = connection["target_ref"]
                if target_type == "attractor":
                    target_index = int(fixed_label_map.get(int(target_ref), -1))
                    if target_index < 0:
                        target_type = None
                elif target_type == "cycle":
                    target_index = int(target_ref)
                elif target_type == "saddle":
                    target_index = int(saddle_index_map.get(int(target_ref), -1))
                    if target_index < 0:
                        target_type = None
                else:
                    target_index = -1
                    target_type = None

                unstable_connections.append(
                    {
                        "target_index": target_index,
                        "target_type": target_type,
                        "endpoint": np.asarray(connection["endpoint"], dtype=float).copy(),
                    }
                )

            saddle["unstable_connections"] = unstable_connections
            saddle.pop("_unstable_meta", None)

        saddle_manifolds["objects"] = self._build_phase_objects(attractors, saddles)
        return saddle_manifolds

    def _match_or_register_cycle_attractor(self, signature, attractors, fp_tol):
        cycle_clusters = [attractor for attractor in attractors if attractor["type"] == "cycle"]
        for cluster in cycle_clusters:
            center_delta = np.linalg.norm(signature["center"] - cluster["center"])
            radius_delta = abs(signature["radius_mean"] - cluster["radius_mean"])
            period_delta = abs(signature["period_steps"] - cluster["period_steps"])
            period_tol = max(8, int(0.5 * max(signature["period_steps"], cluster["period_steps"])))
            if (
                center_delta <= max(fp_tol * 12.0, 0.75)
                and radius_delta <= max(fp_tol * 12.0, 0.75)
                and period_delta <= period_tol
            ):
                if signature["recurrence_distance"] < cluster["recurrence_distance"]:
                    cluster["recurrence_distance"] = signature["recurrence_distance"]
                    cluster["trajectory"] = signature["trajectory"]
                return cluster

        label_id = len(attractors)
        cluster = {
            "id": label_id,
            "type": "cycle",
            "center": signature["center"],
            "radius_mean": signature["radius_mean"],
            "radius_span": signature["radius_span"],
            "period_steps": signature["period_steps"],
            "period": signature["period"],
            "recurrence_distance": signature["recurrence_distance"],
            "trajectory": signature["trajectory"],
            "members": [],
        }
        attractors.append(cluster)
        return cluster

    def _classify_seed_attractors_batched_jax(
        self,
        pars,
        seeds,
        fixed_points,
        dt,
        n_steps,
        fp_tol,
        vel_tol,
        transient_fraction=0.5,
        cycle_window=128,
        x_range=None,
        y_range=None,
    ):
        jax, jnp = self._jax_modules()
        seeds = np.asarray(seeds, dtype=float)
        n_seeds = seeds.shape[1]
        points = np.asarray(fixed_points["points"], dtype=float)
        attracting_indices = np.flatnonzero(np.asarray(fixed_points["attracting_mask"], dtype=bool))
        attractor_points = points[attracting_indices] if attracting_indices.size else np.empty((0, 2), dtype=float)
        has_attractors = attractor_points.size > 0

        tail_len = max(int(cycle_window), 32)
        convergence_start = max(4, int(transient_fraction * n_steps))
        convergence_start = min(convergence_start, max(n_steps - 1, 0))
        tail_start = max(0, int(n_steps) - tail_len)
        dt_scale = max(abs(float(dt)), 1e-12)
        has_bounds = x_range is not None and y_range is not None
        if has_bounds:
            x_min, x_max = sorted(map(float, x_range))
            y_min, y_max = sorted(map(float, y_range))
            x_pad = 0.25 * max(x_max - x_min, fp_tol * 10.0)
            y_pad = 0.25 * max(y_max - y_min, fp_tol * 10.0)
        else:
            x_min = x_max = y_min = y_max = x_pad = y_pad = 0.0

        jax_pars = {
            key: (jnp.asarray(value, dtype=float) if key != "empty" else value)
            for key, value in pars.items()
        }
        attractor_points_jax = jnp.asarray(attractor_points, dtype=float)
        attracting_indices_jax = jnp.asarray(attracting_indices, dtype=int)

        def rk4(q):
            k1 = self._eval_flow(jax_pars, q, return_potentials=False, use_jax=True)
            k2 = self._eval_flow(jax_pars, q + 0.5 * dt * k1, return_potentials=False, use_jax=True)
            k3 = self._eval_flow(jax_pars, q + 0.5 * dt * k2, return_potentials=False, use_jax=True)
            k4 = self._eval_flow(jax_pars, q + dt * k3, return_potentials=False, use_jax=True)
            return q + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        def scan_step(state, step):
            positions, active, fixed_indices, escaped, nonfinite, tail_history, tail_index, stored_steps = state
            safe_positions = jnp.where(active[None, :], positions, 0.0)
            next_positions = rk4(safe_positions)
            finite = jnp.all(jnp.isfinite(next_positions), axis=0)
            live = active & finite
            nonfinite_now = active & ~finite
            speeds = jnp.linalg.norm(next_positions - safe_positions, axis=0) / dt_scale

            boundary = jnp.zeros(n_seeds, dtype=bool)
            if has_bounds:
                boundary = live & (
                    (next_positions[0] < x_min - x_pad)
                    | (next_positions[0] > x_max + x_pad)
                    | (next_positions[1] < y_min - y_pad)
                    | (next_positions[1] > y_max + y_pad)
                )

            converged = jnp.zeros(n_seeds, dtype=bool)
            fixed_updates = -jnp.ones(n_seeds, dtype=int)
            if has_attractors:
                distances = jnp.linalg.norm(next_positions.T[:, None, :] - attractor_points_jax[None, :, :], axis=2)
                nearest = jnp.argmin(distances, axis=1)
                nearest_distance = distances[jnp.arange(n_seeds), nearest]
                unresolved = live & ~boundary
                converged = (
                    (step >= convergence_start)
                    & unresolved
                    & (nearest_distance <= fp_tol)
                    & (speeds <= vel_tol)
                )
                fixed_updates = attracting_indices_jax[nearest]

            finished = nonfinite_now | boundary | converged
            active = active & ~finished
            positions = jnp.where(active[None, :], next_positions, jnp.nan)
            fixed_indices = jnp.where(converged, fixed_updates, fixed_indices)
            escaped = escaped | boundary
            nonfinite = nonfinite | nonfinite_now

            should_store = step >= tail_start
            tail_history = jax.lax.cond(
                should_store,
                lambda history: history.at[tail_index].set(positions),
                lambda history: history,
                tail_history,
            )
            tail_index = jnp.where(should_store, (tail_index + 1) % tail_len, tail_index)
            stored_steps = jnp.where(should_store, jnp.minimum(stored_steps + 1, tail_len), stored_steps)
            return (
                positions,
                active,
                fixed_indices,
                escaped,
                nonfinite,
                tail_history,
                tail_index,
                stored_steps,
            ), None

        @jax.jit
        def run_scan(seeds_jax):
            state = (
                seeds_jax,
                jnp.ones(n_seeds, dtype=bool),
                -jnp.ones(n_seeds, dtype=int),
                jnp.zeros(n_seeds, dtype=bool),
                jnp.zeros(n_seeds, dtype=bool),
                jnp.full((tail_len, 2, n_seeds), jnp.nan, dtype=float),
                jnp.asarray(0, dtype=int),
                jnp.asarray(0, dtype=int),
            )
            return jax.lax.scan(scan_step, state, jnp.arange(int(n_steps)))

        final_state, _ = run_scan(jnp.asarray(seeds, dtype=float))
        _, _, fixed_point_indices, escaped_mask, nonfinite_mask, tail_history, tail_index, stored_steps = final_state
        tail_history = np.asarray(tail_history, dtype=float)
        tail_index = int(np.asarray(tail_index))
        stored_steps = int(np.asarray(stored_steps))
        if stored_steps == 0:
            chronological_tail = np.empty((0, 2, n_seeds), dtype=float)
        elif stored_steps < tail_len:
            chronological_tail = tail_history[:stored_steps].copy()
        else:
            chronological_tail = np.concatenate(
                (tail_history[tail_index:], tail_history[:tail_index]),
                axis=0,
            )

        fixed_point_indices = np.asarray(fixed_point_indices, dtype=int)
        escaped_mask = np.asarray(escaped_mask, dtype=bool)
        nonfinite_mask = np.asarray(nonfinite_mask, dtype=bool)
        if attractor_points.size and chronological_tail.shape[0]:
            unresolved_indices = np.flatnonzero(
                (fixed_point_indices < 0)
                & ~escaped_mask
                & ~nonfinite_mask
            )
            if unresolved_indices.size:
                final_points = chronological_tail[-1][:, unresolved_indices].T
                distances = np.linalg.norm(
                    final_points[:, None, :] - attractor_points[None, :, :],
                    axis=2,
                )
                nearest = np.argmin(distances, axis=1)
                nearest_distance = distances[np.arange(unresolved_indices.size), nearest]
                close_to_fixed = nearest_distance < max(fp_tol * 4.0, 0.1)
                if np.any(close_to_fixed):
                    fixed_point_indices[unresolved_indices[close_to_fixed]] = attracting_indices[nearest[close_to_fixed]]

        return {
            "fixed_point_indices": fixed_point_indices,
            "chronological_tail": chronological_tail,
            "escaped_mask": escaped_mask,
            "nonfinite_mask": nonfinite_mask,
        }

    def _classify_seed_attractors_batched(
        self,
        pars,
        seeds,
        fixed_points,
        dt,
        n_steps,
        fp_tol,
        vel_tol,
        transient_fraction=0.5,
        cycle_window=128,
        x_range=None,
        y_range=None,
        use_jax=False,
    ):
        seeds, _ = self._coerce_phase_coordinate_array(seeds)
        n_seeds = seeds.shape[1]
        if n_seeds == 0:
            return {
                "fixed_point_indices": np.empty((0,), dtype=int),
                "chronological_tail": np.empty((0, 2, 0), dtype=float),
                "escaped_mask": np.empty((0,), dtype=bool),
                "nonfinite_mask": np.empty((0,), dtype=bool),
            }
        if use_jax:
            return self._classify_seed_attractors_batched_jax(
                pars,
                seeds,
                fixed_points,
                dt,
                n_steps,
                fp_tol,
                vel_tol,
                transient_fraction=transient_fraction,
                cycle_window=cycle_window,
                x_range=x_range,
                y_range=y_range,
            )

        points = np.asarray(fixed_points["points"], dtype=float)
        attracting_indices = np.flatnonzero(np.asarray(fixed_points["attracting_mask"], dtype=bool))
        attractor_points = points[attracting_indices] if attracting_indices.size else np.empty((0, 2), dtype=float)

        positions = seeds.copy()
        active = np.ones(n_seeds, dtype=bool)
        fixed_point_indices = np.full(n_seeds, -1, dtype=int)
        escaped_mask = np.zeros(n_seeds, dtype=bool)
        nonfinite_mask = np.zeros(n_seeds, dtype=bool)

        tail_len = max(int(cycle_window), 32)
        tail_history = np.full((tail_len, 2, n_seeds), np.nan, dtype=float)
        tail_index = 0
        stored_steps = 0

        convergence_start = max(4, int(transient_fraction * n_steps))
        convergence_start = min(convergence_start, max(n_steps - 1, 0))
        tail_start = max(0, int(n_steps) - tail_len)
        dt_scale = max(abs(float(dt)), 1e-12)

        if x_range is not None and y_range is not None:
            x_min, x_max = sorted(map(float, x_range))
            y_min, y_max = sorted(map(float, y_range))
            x_pad = 0.25 * max(x_max - x_min, fp_tol * 10.0)
            y_pad = 0.25 * max(y_max - y_min, fp_tol * 10.0)
        else:
            x_min = x_max = y_min = y_max = None
            x_pad = y_pad = 0.0

        for step in range(int(n_steps)):
            if not np.any(active):
                break

            active_indices = np.flatnonzero(active)
            current = positions[:, active_indices]
            next_positions = self._rk4_step_batched(pars, current, dt, use_jax=use_jax)
            finite_mask = np.all(np.isfinite(next_positions), axis=0)

            if np.any(~finite_mask):
                invalid_indices = active_indices[~finite_mask]
                nonfinite_mask[invalid_indices] = True
                active[invalid_indices] = False
                positions[:, invalid_indices] = np.nan

            if np.any(finite_mask):
                live_indices = active_indices[finite_mask]
                live_current = current[:, finite_mask]
                live_positions = next_positions[:, finite_mask]
                speeds = np.linalg.norm(live_positions - live_current, axis=0) / dt_scale
                positions[:, live_indices] = live_positions

                boundary = np.zeros(live_indices.size, dtype=bool)
                if x_min is not None:
                    boundary = (
                        (live_positions[0] < x_min - x_pad)
                        | (live_positions[0] > x_max + x_pad)
                        | (live_positions[1] < y_min - y_pad)
                        | (live_positions[1] > y_max + y_pad)
                    )
                if np.any(boundary):
                    boundary_indices = live_indices[boundary]
                    escaped_mask[boundary_indices] = True
                    active[boundary_indices] = False
                    positions[:, boundary_indices] = np.nan

                if step >= convergence_start and attractor_points.size:
                    unresolved_mask = ~boundary
                    if np.any(unresolved_mask):
                        unresolved_indices = live_indices[unresolved_mask]
                        unresolved_positions = live_positions[:, unresolved_mask]
                        unresolved_speeds = speeds[unresolved_mask]
                        distances = np.linalg.norm(
                            unresolved_positions.T[:, None, :] - attractor_points[None, :, :],
                            axis=2,
                        )
                        nearest = np.argmin(distances, axis=1)
                        nearest_distance = distances[np.arange(unresolved_indices.size), nearest]
                        converged = (nearest_distance <= fp_tol) & (unresolved_speeds <= vel_tol)
                        if np.any(converged):
                            converged_indices = unresolved_indices[converged]
                            fixed_point_indices[converged_indices] = attracting_indices[nearest[converged]]
                            active[converged_indices] = False
                            positions[:, converged_indices] = np.nan

            if step >= tail_start:
                tail_history[tail_index] = positions
                tail_index = (tail_index + 1) % tail_len
                stored_steps = min(stored_steps + 1, tail_len)

        if stored_steps == 0:
            chronological_tail = np.empty((0, 2, n_seeds), dtype=float)
        elif stored_steps < tail_len:
            chronological_tail = tail_history[:stored_steps].copy()
        else:
            chronological_tail = np.concatenate(
                (tail_history[tail_index:], tail_history[:tail_index]),
                axis=0,
            )

        if attractor_points.size and chronological_tail.shape[0]:
            unresolved_indices = np.flatnonzero(
                (fixed_point_indices < 0)
                & ~escaped_mask
                & ~nonfinite_mask
            )
            if unresolved_indices.size:
                final_points = chronological_tail[-1][:, unresolved_indices].T
                distances = np.linalg.norm(
                    final_points[:, None, :] - attractor_points[None, :, :],
                    axis=2,
                )
                nearest = np.argmin(distances, axis=1)
                nearest_distance = distances[np.arange(unresolved_indices.size), nearest]
                close_to_fixed = nearest_distance < max(fp_tol * 4.0, 0.1)
                if np.any(close_to_fixed):
                    fixed_point_indices[unresolved_indices[close_to_fixed]] = attracting_indices[nearest[close_to_fixed]]

        return {
            "fixed_point_indices": fixed_point_indices,
            "chronological_tail": chronological_tail,
            "escaped_mask": escaped_mask,
            "nonfinite_mask": nonfinite_mask,
        }

    def find_fixed_points(
        self,
        t,
        x_range,
        y_range,
        n_grid=25,
        root_tol=1e-9,
        dedup_tol=1e-6,
        stability_tol=1e-6,
        use_jax=False,
    ):
        try:
            from scipy.optimize import root
        except ImportError as exc:
            raise ImportError(
                "find_fixed_points(...) requires scipy. Add scipy to the environment first."
            ) from exc

        x_min, x_max = sorted(map(float, x_range))
        y_min, y_max = sorted(map(float, y_range))
        x_seeds = np.linspace(x_min, x_max, n_grid)
        y_seeds = np.linspace(y_min, y_max, n_grid)
        residual_tol = max(root_tol * 1000.0, 1e-6)
        pars = self._get_all_pars(t)
        if use_jax:
            jax, jnp = self._jax_modules()
            jax_pars = {
                key: (jnp.asarray(value, dtype=float) if key != "empty" else value)
                for key, value in pars.items()
            }

            def flow_point_jax(point):
                coords = point.reshape(2, 1)
                return self._eval_flow(jax_pars, coords, return_potentials=False, use_jax=True).reshape(2)

            def jac_point_jax(point):
                jac_h = 1e-4
                deltas = jac_h * jnp.eye(2, dtype=float)
                flow_plus = jax.vmap(lambda delta: flow_point_jax(point + delta))(deltas)
                flow_minus = jax.vmap(lambda delta: flow_point_jax(point - delta))(deltas)
                return ((flow_plus - flow_minus) / (2.0 * jac_h)).T

            flow_point_jit = jax.jit(flow_point_jax)
            jac_point_jit = jax.jit(jac_point_jax)

            def flow_point(point):
                return np.asarray(flow_point_jit(jnp.asarray(point, dtype=float)), dtype=float)

            def jac_point(point):
                return np.asarray(jac_point_jit(jnp.asarray(point, dtype=float)), dtype=float)
        else:
            def flow_point(point):
                return self._flow_vector_from_pars(pars, point)

            def jac_point(point):
                return self._numerical_jacobian_from_pars(pars, point)

        unique_points = []
        residual_norms = []

        for x0 in x_seeds:
            for y0 in y_seeds:
                guess = np.array([x0, y0], dtype=float)
                try:
                    solution = root(
                        flow_point,
                        guess,
                        jac=jac_point,
                        tol=root_tol,
                    )
                except Exception:
                    continue

                if not solution.success:
                    continue

                point = np.asarray(solution.x, dtype=float)
                if point.shape != (2,) or not np.all(np.isfinite(point)):
                    continue
                if (
                    point[0] < x_min - dedup_tol
                    or point[0] > x_max + dedup_tol
                    or point[1] < y_min - dedup_tol
                    or point[1] > y_max + dedup_tol
                ):
                    continue

                residual = float(np.linalg.norm(flow_point(point)))
                if residual > residual_tol:
                    continue

                duplicate_index = None
                for idx, existing in enumerate(unique_points):
                    if np.linalg.norm(point - existing) <= dedup_tol:
                        duplicate_index = idx
                        break

                if duplicate_index is None:
                    unique_points.append(point)
                    residual_norms.append(residual)
                elif residual < residual_norms[duplicate_index]:
                    unique_points[duplicate_index] = point
                    residual_norms[duplicate_index] = residual

        if not unique_points:
            return {
                "points": np.empty((0, 2), dtype=float),
                "residual_norms": np.empty((0,), dtype=float),
                "jacobians": np.empty((0, 2, 2), dtype=float),
                "eigenvalues": np.empty((0, 2), dtype=complex),
                "stability": np.empty((0,), dtype=object),
                "attracting_mask": np.empty((0,), dtype=bool),
            }

        points = np.asarray(unique_points, dtype=float)
        residual_norms = np.asarray(residual_norms, dtype=float)
        order = np.lexsort((points[:, 1], points[:, 0]))
        points = points[order]
        residual_norms = residual_norms[order]

        jacobians = np.zeros((points.shape[0], 2, 2), dtype=float)
        eigenvalues = np.zeros((points.shape[0], 2), dtype=complex)
        stability = []
        attracting_mask = np.zeros(points.shape[0], dtype=bool)

        for idx, point in enumerate(points):
            jac = np.asarray(jac_point(point), dtype=float).reshape(2, 2)
            eigvals = np.linalg.eigvals(jac)
            fixed_type = self._classify_eigenvalues(eigvals, stability_tol)
            jacobians[idx] = jac
            eigenvalues[idx] = eigvals
            stability.append(fixed_type)
            attracting_mask[idx] = fixed_type == "attractor"

        return {
            "points": points,
            "residual_norms": residual_norms,
            "jacobians": jacobians,
            "eigenvalues": eigenvalues,
            "stability": np.asarray(stability, dtype=object),
            "attracting_mask": attracting_mask,
        }

    def find_saddle_manifolds(
        self,
        t,
        fixed_points=None,
        x_range=None,
        y_range=None,
        step_size=0.03,
        n_steps=700,
        perturbation=None,
        stability_tol=1e-6,
        termination_tol=1e-2,
        velocity_tol=1e-3,
        use_jax=False,
    ):
        if fixed_points is None:
            if x_range is None or y_range is None:
                raise ValueError("x_range and y_range are required when fixed_points is not provided.")
            fixed_points = self.find_fixed_points(t, x_range, y_range, use_jax=use_jax)

        points = np.asarray(fixed_points["points"], dtype=float)
        if x_range is None:
            if not points.size:
                raise ValueError("x_range is required when no fixed points are available.")
            x_min = float(np.min(points[:, 0]))
            x_max = float(np.max(points[:, 0]))
            x_pad = max(0.5, 0.25 * max(x_max - x_min, 1.0))
            x_range = (x_min - x_pad, x_max + x_pad)
        if y_range is None:
            if not points.size:
                raise ValueError("y_range is required when no fixed points are available.")
            y_min = float(np.min(points[:, 1]))
            y_max = float(np.max(points[:, 1]))
            y_pad = max(0.5, 0.25 * max(y_max - y_min, 1.0))
            y_range = (y_min - y_pad, y_max + y_pad)

        saddle_manifolds = self._trace_saddle_manifold_geometry(
            t,
            fixed_points,
            x_range,
            y_range,
            step_size,
            n_steps,
            perturbation,
            stability_tol,
            termination_tol,
            velocity_tol,
            use_jax=use_jax,
        )
        attractors, _ = self._build_fixed_point_attractors(fixed_points)
        self._annotate_saddle_connections(
            saddle_manifolds,
            fixed_points,
            attractors,
            fp_tol=termination_tol,
            x_range=x_range,
            y_range=y_range,
            termination_tol=termination_tol,
        )
        return saddle_manifolds

    def find_attractor_basins_manifold(self, phase_result, fill_boundary=True):
        region_ids = np.asarray(phase_result["region_ids"], dtype=int)
        boundary_mask = np.asarray(phase_result["boundary_mask"], dtype=bool)
        region_attractor_ids = np.asarray(phase_result["region_attractor_ids"], dtype=int)
        basin_labels = -np.ones(region_ids.shape, dtype=int)
        for region_id, attractor_id in enumerate(region_attractor_ids):
            if int(attractor_id) < 0:
                continue
            basin_labels[region_ids == int(region_id)] = int(attractor_id)
        unresolved_mask = basin_labels < 0
        if fill_boundary and np.any(boundary_mask):
            basin_labels = self._fill_barrier_labels(basin_labels, boundary_mask)
            unresolved_mask = basin_labels < 0

        x_coords = np.asarray(phase_result["x_coords"], dtype=float).copy()
        y_coords = np.asarray(phase_result["y_coords"], dtype=float).copy()
        node_labels = np.full(basin_labels.shape, -1, dtype=int)
        for attractor in phase_result["attractors"]:
            node_id = attractor.get("node_id")
            if node_id is None:
                continue
            node_labels[basin_labels == int(attractor["id"])] = int(node_id)

        return {
            "basin_labels": basin_labels,
            "unresolved_mask": unresolved_mask,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "node_labels": node_labels,
        }

    def find_phase_objects_manifold(
        self,
        t,
        xx,
        yy,
        fixed_points=None,
        saddle_manifolds=None,
        dt=0.05,
        n_steps=800,
        fp_tol=1e-2,
        vel_tol=1e-3,
        transient_fraction=0.5,
        cycle_window=128,
        manifold_linewidth=0,
        max_region_samples=5,
        use_jax=False,
    ):
        xx = np.asarray(xx, dtype=float)
        yy = np.asarray(yy, dtype=float)
        if xx.shape != yy.shape:
            raise ValueError("xx and yy must have the same shape.")

        x_range = (float(np.min(xx)), float(np.max(xx)))
        y_range = (float(np.min(yy)), float(np.max(yy)))

        if fixed_points is None:
            fixed_points = self.find_fixed_points(t, x_range, y_range, use_jax=use_jax)
        if saddle_manifolds is None:
            saddle_manifolds = self._trace_saddle_manifold_geometry(
                t,
                fixed_points,
                x_range,
                y_range,
                0.03,
                n_steps,
                None,
                1e-6,
                1e-2,
                1e-3,
                use_jax=use_jax,
            )

        points = np.asarray(fixed_points["points"], dtype=float)
        attracting_indices = np.flatnonzero(np.asarray(fixed_points["attracting_mask"], dtype=bool))
        attractors, fixed_label_map = self._build_fixed_point_attractors(fixed_points)

        x_coords = np.asarray(xx[0], dtype=float)
        y_coords = np.asarray(yy[:, 0], dtype=float)
        boundary_mask = np.zeros(xx.shape, dtype=bool)

        for saddle in saddle_manifolds.get("saddles", ()):
            saddle_point = np.asarray(saddle["point"], dtype=float)
            if x_coords.size > 1:
                ix = int(np.rint((saddle_point[0] - x_coords[0]) / (x_coords[1] - x_coords[0])))
            else:
                ix = 0
            if y_coords.size > 1:
                iy = int(np.rint((saddle_point[1] - y_coords[0]) / (y_coords[1] - y_coords[0])))
            else:
                iy = 0
            if 0 <= ix < boundary_mask.shape[1] and 0 <= iy < boundary_mask.shape[0]:
                self._mark_grid_point(boundary_mask, iy, ix, thickness=manifold_linewidth)
            for branch in saddle.get("stable", ()):
                self._mark_curve_on_grid(
                    branch,
                    x_coords,
                    y_coords,
                    boundary_mask,
                    thickness=manifold_linewidth,
                )

        region_mask = ~boundary_mask
        region_ids = self._label_grid_regions(region_mask)

        attracting_cells = {}
        if points.size:
            dx = float(x_coords[1] - x_coords[0]) if x_coords.size > 1 else 1.0
            dy = float(y_coords[1] - y_coords[0]) if y_coords.size > 1 else 1.0
            for fixed_index in attracting_indices:
                point = points[fixed_index]
                ix = int(np.rint((point[0] - x_coords[0]) / max(dx, 1e-12)))
                iy = int(np.rint((point[1] - y_coords[0]) / max(dy, 1e-12)))
                if 0 <= ix < xx.shape[1] and 0 <= iy < xx.shape[0]:
                    attracting_cells[int(fixed_index)] = (iy, ix)

        if region_ids.size:
            region_attractor_ids = np.full(int(np.max(region_ids)) + 1, -1, dtype=int)
        else:
            region_attractor_ids = np.empty((0,), dtype=int)

        sample_records = []
        for region_id in np.unique(region_ids):
            if region_id < 0:
                continue

            region = region_ids == region_id
            region_attractors = [
                fixed_index
                for fixed_index, (iy, ix) in attracting_cells.items()
                if region[iy, ix]
            ]
            if len(region_attractors) == 1:
                region_attractor_ids[int(region_id)] = int(fixed_label_map[int(region_attractors[0])])
                continue

            sample_indices = self._select_region_samples(region, max_samples=max_region_samples)
            for sample_rank, (iy, ix) in enumerate(sample_indices):
                sample_records.append((int(region_id), int(sample_rank), int(iy), int(ix)))

        if sample_records:
            sample_seeds = np.array(
                [[xx[iy, ix], yy[iy, ix]] for _, _, iy, ix in sample_records],
                dtype=float,
            ).T
            sample_results = self._classify_seed_attractors_batched(
                self._get_all_pars(t),
                sample_seeds,
                fixed_points,
                dt,
                n_steps,
                fp_tol,
                vel_tol,
                transient_fraction=transient_fraction,
                cycle_window=cycle_window,
                x_range=x_range,
                y_range=y_range,
                use_jax=use_jax,
            )
            fixed_point_indices = np.asarray(sample_results["fixed_point_indices"], dtype=int)
            chronological_tail = np.asarray(sample_results["chronological_tail"], dtype=float)

            for sample_index, (region_id, _, _, _) in enumerate(sample_records):
                if region_attractor_ids[int(region_id)] >= 0:
                    continue

                fixed_index = int(fixed_point_indices[sample_index])
                if fixed_index >= 0:
                    region_attractor_ids[int(region_id)] = int(fixed_label_map[fixed_index])
                    continue

                if not self._allows_limit_cycles() or chronological_tail.shape[0] < 16:
                    continue

                traj = chronological_tail[:, :, sample_index]
                finite_rows = np.all(np.isfinite(traj), axis=1)
                traj = traj[finite_rows]
                if traj.shape[0] != chronological_tail.shape[0] or traj.shape[0] < 16:
                    continue

                signature = self._detect_cycle_signature(traj, dt, fp_tol, vel_tol)
                if signature is None:
                    continue

                cluster = self._match_or_register_cycle_attractor(signature, attractors, fp_tol)
                region_attractor_ids[int(region_id)] = int(cluster["id"])

        if hasattr(self, "_map_attractors_to_nodes"):
            attractor_node_ids = self._map_attractors_to_nodes(
                attractors,
                x_coords=x_coords,
                y_coords=y_coords,
            )
            for attractor in attractors:
                node_id = attractor_node_ids.get(int(attractor["id"]))
                if node_id is not None:
                    attractor["node_id"] = int(node_id)

        self._annotate_saddle_connections(
            saddle_manifolds,
            fixed_points,
            attractors,
            fp_tol=fp_tol,
            x_range=x_range,
            y_range=y_range,
            termination_tol=fp_tol,
        )

        return {
            "attractors": attractors,
            "fixed_points": fixed_points,
            "saddle_manifolds": saddle_manifolds,
            "objects": saddle_manifolds.get("objects", ()),
            "boundary_mask": boundary_mask,
            "region_ids": region_ids,
            "region_attractor_ids": region_attractor_ids,
            "x_coords": x_coords.copy(),
            "y_coords": y_coords.copy(),
            "xx_shape": tuple(int(dim) for dim in xx.shape),
            "method": "manifold",
        }

    def find_attractor_basins(
        self,
        t,
        xx,
        yy,
        fixed_points=None,
        method="manifold",
        use_jax=False,
        **kwargs,
    ):
        method = str(method).lower()
        if method != "manifold":
            raise ValueError("method must be 'manifold'.")
        phase_result = kwargs.pop("phase_result", None)
        fill_boundary = kwargs.pop("fill_boundary", True)
        if phase_result is None:
            phase_result = self.find_phase_objects_manifold(
                t,
                xx,
                yy,
                fixed_points=fixed_points,
                use_jax=use_jax,
                **kwargs,
            )
        elif kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise ValueError(
                f"Unexpected kwargs when phase_result is provided: {unexpected}"
            )
        return self.find_attractor_basins_manifold(phase_result, fill_boundary=fill_boundary)

    @staticmethod
    def get_catastrophe_info(phase_result):
        fixed_points = phase_result.get("fixed_points")
        saddle_manifolds = phase_result.get("saddle_manifolds")
        objects = phase_result.get("objects")
        if fixed_points is None or saddle_manifolds is None or objects is None:
            raise ValueError(
                "get_catastrophe_info(...) expects the output of find_phase_objects_manifold(...)."
            )

        points = np.asarray(fixed_points["points"], dtype=float)
        object_info = []
        n_cycles = 0
        exported_ids = {}
        for obj in objects:
            obj_type = str(obj["type"])
            object_id = int(obj["id"])
            export_id = int(obj.get("node_id", object_id)) if obj_type in {"attractor", "cycle"} else object_id
            exported_ids[object_id] = export_id
            if obj_type == "cycle":
                n_cycles += 1
                cycle_info = {
                    "id": export_id,
                    "type": obj_type,
                    "location": np.asarray(obj["center"], dtype=float).copy(),
                    "radius": float(obj["radius_mean"]),
                    "period": float(obj["period"]),
                }
                if obj.get("node_id") is not None:
                    cycle_info["node_id"] = int(obj["node_id"])
                object_info.append(cycle_info)
                continue

            fixed_index = int(obj["fixed_point_index"])
            obj_info = {
                "id": export_id,
                "type": obj_type,
                "location": points[fixed_index].copy(),
            }
            if obj_type == "attractor" and obj.get("node_id") is not None:
                obj_info["node_id"] = int(obj["node_id"])
            object_info.append(obj_info)

        unstable_connections = []
        for saddle in saddle_manifolds.get("saddles", ()):
            source_index = int(saddle["id"])
            for connection in saddle.get("unstable_connections", ()):
                target_index = int(connection["target_index"])
                if connection["target_type"] in {"attractor", "cycle"}:
                    target_index = exported_ids.get(target_index, target_index)
                unstable_connections.append(
                    {
                        "connection": (
                            source_index,
                            target_index,
                        ),
                        "target_type": connection["target_type"],
                    }
                )

        return {
            "n_fp": int(points.shape[0]),
            "n_cycles": n_cycles,
            "objects": object_info,
            "unstable_connections": unstable_connections,
        }
