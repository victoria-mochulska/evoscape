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

    def numerical_jacobian(self, t, q, h=1e-4):
        coords, squeeze_output = self._coerce_phase_coordinate_array(q)
        jac = np.zeros((2, 2) + coords.shape[1:], dtype=float)

        for axis in range(2):
            delta = np.zeros_like(coords)
            delta[axis] = h
            flow_plus = np.asarray(self(t, coords + delta, return_potentials=False), dtype=float)
            flow_minus = np.asarray(self(t, coords - delta, return_potentials=False), dtype=float)
            jac[:, axis] = (flow_plus - flow_minus) / (2.0 * h)

        if squeeze_output:
            return jac.reshape(2, 2)
        return jac

    def jacobian(self, t, q, method="auto", h=1e-4):
        method = method.lower()
        if method in ("numeric", "numerical"):
            return self.numerical_jacobian(t, q, h=h)
        if method == "analytic":
            return self.analytic_jacobian(t, q)
        if method != "auto":
            raise ValueError("method must be one of 'auto', 'analytic', or 'numeric'.")

        try:
            return self.analytic_jacobian(t, q)
        except NotImplementedError:
            return self.numerical_jacobian(t, q, h=h)

    def _flow_vector(self, t, point):
        coords = np.asarray(point, dtype=float).reshape(2, 1)
        flow = np.asarray(self(t, coords, return_potentials=False), dtype=float)
        return flow.reshape(2)

    def _rk4_step(self, t, q, dt):
        k1 = np.asarray(self(t, q, return_potentials=False), dtype=float)
        k2 = np.asarray(self(t, q + 0.5 * dt * k1, return_potentials=False), dtype=float)
        k3 = np.asarray(self(t, q + 0.5 * dt * k2, return_potentials=False), dtype=float)
        k4 = np.asarray(self(t, q + dt * k3, return_potentials=False), dtype=float)
        return q + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

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
    ):
        x_min, x_max = sorted(map(float, x_range))
        y_min, y_max = sorted(map(float, y_range))
        x_pad = 0.1 * max(x_max - x_min, termination_tol * 10.0)
        y_pad = 0.1 * max(y_max - y_min, termination_tol * 10.0)

        point = np.asarray(start, dtype=float).reshape(2)
        trajectory = [point.copy()]

        for _ in range(int(n_steps)):
            current = point.reshape(2, 1)
            next_point = self._rk4_step(t, current, dt).reshape(2)
            if not np.all(np.isfinite(next_point)):
                break

            trajectory.append(next_point.copy())
            step_velocity = np.linalg.norm(next_point - point) / max(abs(dt), 1e-12)
            point = next_point

            if (
                point[0] < x_min - x_pad
                or point[0] > x_max + x_pad
                or point[1] < y_min - y_pad
                or point[1] > y_max + y_pad
            ):
                break

            if fixed_point_positions is not None and fixed_point_positions.size:
                distances = np.linalg.norm(fixed_point_positions - point[None, :], axis=1)
                if exclude_index is not None and 0 <= int(exclude_index) < distances.size:
                    distances[int(exclude_index)] = np.inf
                nearest_index = int(np.argmin(distances))
                if np.isfinite(distances[nearest_index]):
                    if distances[nearest_index] <= termination_tol and step_velocity <= velocity_tol:
                        break

        return np.asarray(trajectory, dtype=float)

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

        return {
            "center": centroid,
            "radius_mean": mean_radius,
            "radius_span": radius_span,
            "period_steps": best_lag,
            "period": best_lag * dt,
            "trajectory": traj.copy(),
            "recurrence_distance": best_distance,
            "segment_error": segment_error,
        }

    @staticmethod
    def _trajectory_set_distance(traj, reference_traj, sample_count=64):
        if traj.size == 0 or reference_traj.size == 0:
            return np.inf, np.inf

        if traj.shape[0] > sample_count:
            traj_idx = np.linspace(0, traj.shape[0] - 1, sample_count, dtype=int)
            traj_sample = traj[traj_idx]
        else:
            traj_sample = traj

        if reference_traj.shape[0] > sample_count:
            ref_idx = np.linspace(0, reference_traj.shape[0] - 1, sample_count, dtype=int)
            reference_sample = reference_traj[ref_idx]
        else:
            reference_sample = reference_traj

        distances = np.linalg.norm(
            traj_sample[:, None, :] - reference_sample[None, :, :],
            axis=2,
        )
        nearest = np.min(distances, axis=1)
        return float(np.mean(nearest)), float(np.max(nearest))

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

    def _classify_seed_attractor(
        self,
        t,
        seed,
        fixed_points,
        attractors,
        dt,
        n_steps,
        fp_tol,
        vel_tol,
        transient_fraction=0.5,
        cycle_window=128,
        x_range=None,
        y_range=None,
    ):
        points = np.asarray(fixed_points["points"], dtype=float)
        attracting_indices = np.flatnonzero(np.asarray(fixed_points["attracting_mask"], dtype=bool))
        attractor_points = points[attracting_indices] if attracting_indices.size else np.empty((0, 2), dtype=float)
        fixed_label_map = {
            attractor.get("fixed_point_index"): attractor["id"]
            for attractor in attractors
            if attractor["type"] == "fixed_point"
        }

        point = np.asarray(seed, dtype=float).reshape(2)
        tail_len = max(int(cycle_window), 32)
        tail_history = []
        convergence_start = max(4, int(transient_fraction * n_steps))

        if x_range is not None and y_range is not None:
            x_min, x_max = sorted(map(float, x_range))
            y_min, y_max = sorted(map(float, y_range))
            x_pad = 0.25 * max(x_max - x_min, fp_tol * 10.0)
            y_pad = 0.25 * max(y_max - y_min, fp_tol * 10.0)
        else:
            x_min = x_max = y_min = y_max = None
            x_pad = y_pad = 0.0

        for step in range(int(n_steps)):
            current = point.reshape(2, 1)
            next_point = self._rk4_step(t, current, dt).reshape(2)
            if not np.all(np.isfinite(next_point)):
                return None

            speed = np.linalg.norm(next_point - point) / max(abs(dt), 1e-12)
            point = next_point

            if x_min is not None:
                if (
                    point[0] < x_min - x_pad
                    or point[0] > x_max + x_pad
                    or point[1] < y_min - y_pad
                    or point[1] > y_max + y_pad
                ):
                    return None

            if step >= convergence_start and attractor_points.size:
                distances = np.linalg.norm(attractor_points - point[None, :], axis=1)
                nearest = int(np.argmin(distances))
                if distances[nearest] <= fp_tol and speed <= vel_tol:
                    fixed_index = int(attracting_indices[nearest])
                    return {
                        "type": "fixed_point",
                        "label": int(fixed_label_map[fixed_index]),
                        "fixed_point_index": fixed_index,
                    }

            if step >= max(0, n_steps - tail_len):
                tail_history.append(point.copy())

        if not tail_history:
            return None

        traj = np.asarray(tail_history, dtype=float)
        if traj.shape[0] < 16:
            return None

        if attractor_points.size:
            end_distance = np.min(np.linalg.norm(traj[-1][None, :] - attractor_points, axis=1))
            if end_distance < max(fp_tol * 4.0, 0.1):
                nearest = int(np.argmin(np.linalg.norm(attractor_points - traj[-1][None, :], axis=1)))
                fixed_index = int(attracting_indices[nearest])
                return {
                    "type": "fixed_point",
                    "label": int(fixed_label_map[fixed_index]),
                    "fixed_point_index": fixed_index,
                }

        if not self._allows_limit_cycles():
            return None

        signature = self._detect_cycle_signature(traj, dt, fp_tol, vel_tol)
        if signature is None:
            return None

        cluster = self._match_or_register_cycle_attractor(signature, attractors, fp_tol)
        return {"type": "cycle", "label": int(cluster["id"]), "cluster": cluster}

    def find_fixed_points(
        self,
        t,
        x_range,
        y_range,
        n_grid=25,
        root_tol=1e-9,
        dedup_tol=1e-6,
        stability_tol=1e-6,
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

        unique_points = []
        residual_norms = []

        for x0 in x_seeds:
            for y0 in y_seeds:
                guess = np.array([x0, y0], dtype=float)
                try:
                    solution = root(
                        lambda p: self._flow_vector(t, p),
                        guess,
                        jac=lambda p: self.jacobian(t, p, method="auto"),
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

                residual = float(np.linalg.norm(self._flow_vector(t, point)))
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
            jac = np.asarray(self.jacobian(t, point, method="auto"), dtype=float).reshape(2, 2)
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
        n_steps=600,
        perturbation=None,
        stability_tol=1e-6,
        termination_tol=1e-2,
        velocity_tol=1e-3,
    ):
        if fixed_points is None:
            if x_range is None or y_range is None:
                raise ValueError("x_range and y_range are required when fixed_points is not provided.")
            fixed_points = self.find_fixed_points(t, x_range, y_range)

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

        span = max(abs(float(x_range[1]) - float(x_range[0])), abs(float(y_range[1]) - float(y_range[0])), 1.0)
        seed_offset = float(perturbation) if perturbation is not None else 1e-3 * span
        branch_dt = abs(float(step_size))

        saddles = []
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

            stable_branches = []
            unstable_branches = []
            for sign in (-1.0, 1.0):
                stable_seed = point + sign * seed_offset * stable_vector
                unstable_seed = point + sign * seed_offset * unstable_vector

                stable_branch = self._integrate_manifold_branch(
                    t,
                    stable_seed,
                    -branch_dt,
                    n_steps,
                    x_range,
                    y_range,
                    fixed_point_positions=points,
                    exclude_index=int(fixed_index),
                    termination_tol=termination_tol,
                    velocity_tol=velocity_tol,
                )
                unstable_branch = self._integrate_manifold_branch(
                    t,
                    unstable_seed,
                    branch_dt,
                    n_steps,
                    x_range,
                    y_range,
                    fixed_point_positions=points,
                    exclude_index=int(fixed_index),
                    termination_tol=termination_tol,
                    velocity_tol=velocity_tol,
                )

                stable_branches.append(np.vstack((point.copy(), stable_branch)))
                unstable_branches.append(np.vstack((point.copy(), unstable_branch)))

            saddles.append(
                {
                    "fixed_point_index": int(fixed_index),
                    "point": point.copy(),
                    "jacobian": jacobian,
                    "eigenvalues": np.asarray(eigenvalues).copy(),
                    "stable_eigenvalue": eigenvalues[stable_index],
                    "unstable_eigenvalue": eigenvalues[unstable_index],
                    "stable_vector": stable_vector.copy(),
                    "unstable_vector": unstable_vector.copy(),
                    "stable": stable_branches,
                    "unstable": unstable_branches,
                }
            )

        return {"saddles": saddles, "fixed_points": fixed_points}

    def find_attractor_basins_heuristic(
        self,
        t,
        xx,
        yy,
        fixed_points=None,
        dt=0.05,
        n_steps=800,
        fp_tol=1e-2,
        vel_tol=1e-3,
        transient_fraction=0.5,
        cycle_window=128,
    ):
        xx = np.asarray(xx, dtype=float)
        yy = np.asarray(yy, dtype=float)
        if xx.shape != yy.shape:
            raise ValueError("xx and yy must have the same shape.")

        if fixed_points is None:
            fixed_points = self.find_fixed_points(
                t,
                (float(np.min(xx)), float(np.max(xx))),
                (float(np.min(yy)), float(np.max(yy))),
            )

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
                    "stability": str(fixed_points["stability"][fixed_index]),
                    "eigenvalues": np.asarray(fixed_points["eigenvalues"][fixed_index]).copy(),
                }
            )

        positions = np.vstack((xx.ravel(), yy.ravel())).astype(float)
        n_points = positions.shape[1]
        labels = np.full(n_points, -1, dtype=int)
        active = np.ones(n_points, dtype=bool)

        x_min, x_max = float(np.min(xx)), float(np.max(xx))
        y_min, y_max = float(np.min(yy)), float(np.max(yy))
        x_pad = 0.25 * max(x_max - x_min, fp_tol * 10.0)
        y_pad = 0.25 * max(y_max - y_min, fp_tol * 10.0)

        tail_len = max(int(cycle_window), 32)
        tail_history = np.empty((tail_len, 2, n_points), dtype=float)
        tail_index = 0
        stored_steps = 0

        attractor_points = points[attracting_indices] if attracting_indices.size else np.empty((0, 2), dtype=float)
        convergence_start = max(4, int(transient_fraction * n_steps))
        convergence_start = min(convergence_start, max(n_steps - 1, 0))
        tail_start = max(0, n_steps - tail_len)

        for step in range(n_steps):
            if not np.any(active):
                break

            active_indices = np.flatnonzero(active)
            current = positions[:, active_indices]
            next_positions = self._rk4_step(t, current, dt)
            velocities = (next_positions - current) / max(dt, 1e-12)
            positions[:, active_indices] = next_positions

            escaped = (
                ~np.all(np.isfinite(next_positions), axis=0)
                | (next_positions[0] < x_min - x_pad)
                | (next_positions[0] > x_max + x_pad)
                | (next_positions[1] < y_min - y_pad)
                | (next_positions[1] > y_max + y_pad)
            )
            if np.any(escaped):
                active[active_indices[escaped]] = False

            if step >= convergence_start and attractor_points.size:
                live_mask = ~escaped
                live_indices = active_indices[live_mask]
                if live_indices.size:
                    live_positions = next_positions[:, live_mask]
                    speed = np.linalg.norm(velocities[:, live_mask], axis=0)
                    deltas = live_positions.T[:, None, :] - attractor_points[None, :, :]
                    distances = np.linalg.norm(deltas, axis=2)
                    nearest = np.argmin(distances, axis=1)
                    nearest_distance = distances[np.arange(live_indices.size), nearest]
                    converged = (nearest_distance <= fp_tol) & (speed <= vel_tol)
                    if np.any(converged):
                        converged_indices = live_indices[converged]
                        labels[converged_indices] = np.array(
                            [fixed_label_map[int(attracting_indices[idx])] for idx in nearest[converged]],
                            dtype=int,
                        )
                        active[converged_indices] = False

            if step >= tail_start:
                tail_history[tail_index] = positions
                tail_index = (tail_index + 1) % tail_len
                stored_steps = min(stored_steps + 1, tail_len)

        if stored_steps == 0:
            chronological_tail = positions[None, :, :].copy()
        elif stored_steps < tail_len:
            chronological_tail = tail_history[:stored_steps].copy()
        else:
            chronological_tail = np.concatenate(
                (tail_history[tail_index:], tail_history[:tail_index]),
                axis=0,
            )

        cycle_clusters = []
        if self._allows_limit_cycles():
            unresolved_indices = np.flatnonzero(labels == -1)
            for point_index in unresolved_indices:
                traj = chronological_tail[:, :, point_index]
                finite_rows = np.all(np.isfinite(traj), axis=1)
                traj = traj[finite_rows]
                if traj.shape[0] != chronological_tail.shape[0] or traj.shape[0] < 16:
                    continue

                if attractor_points.size:
                    end_distance = np.min(np.linalg.norm(traj[-1][None, :] - attractor_points, axis=1))
                    if end_distance < max(fp_tol * 4.0, 0.1):
                        continue

                signature = self._detect_cycle_signature(traj, dt, fp_tol, vel_tol)
                if signature is None:
                    continue

                cluster_match = None
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
                        cluster_match = cluster
                        break

                if cluster_match is None:
                    label_id = len(attractors)
                    cluster_match = {
                        "id": label_id,
                        "type": "cycle",
                        "center": signature["center"],
                        "radius_mean": signature["radius_mean"],
                        "radius_span": signature["radius_span"],
                        "period_steps": signature["period_steps"],
                        "period": signature["period"],
                        "recurrence_distance": signature["recurrence_distance"],
                        "trajectory": signature["trajectory"],
                        "members": [int(point_index)],
                    }
                    attractors.append(cluster_match)
                    cycle_clusters.append(cluster_match)
                else:
                    cluster_match["members"].append(int(point_index))
                    if signature["recurrence_distance"] < cluster_match["recurrence_distance"]:
                        cluster_match["recurrence_distance"] = signature["recurrence_distance"]
                        cluster_match["trajectory"] = signature["trajectory"]

                labels[point_index] = cluster_match["id"]

            if cycle_clusters:
                remaining_indices = np.flatnonzero(labels == -1)
                for point_index in remaining_indices:
                    traj = chronological_tail[:, :, point_index]
                    finite_rows = np.all(np.isfinite(traj), axis=1)
                    traj = traj[finite_rows]
                    if traj.shape[0] != chronological_tail.shape[0] or traj.shape[0] < 16:
                        continue

                    if attractor_points.size:
                        end_distance = np.min(np.linalg.norm(traj[-1][None, :] - attractor_points, axis=1))
                        if end_distance < max(fp_tol * 4.0, 0.1):
                            continue

                    step_norms = np.linalg.norm(np.diff(traj, axis=0), axis=1)
                    if step_norms.size == 0 or step_norms.mean() / max(dt, 1e-12) < vel_tol * 0.5:
                        continue

                    centroid = traj.mean(axis=0)
                    radial = np.linalg.norm(traj - centroid, axis=1)
                    radius_mean = float(radial.mean())
                    if radius_mean < fp_tol:
                        continue

                    for cluster in cycle_clusters:
                        center_delta = np.linalg.norm(centroid - cluster["center"])
                        radius_delta = abs(radius_mean - cluster["radius_mean"])
                        if (
                            center_delta <= max(fp_tol * 20.0, 1.0)
                            and radius_delta <= max(fp_tol * 20.0, 1.0)
                        ):
                            cluster["members"].append(int(point_index))
                            labels[point_index] = cluster["id"]
                            break

            if cycle_clusters:
                remaining_indices = np.flatnonzero(labels == -1)
                for point_index in remaining_indices:
                    traj = chronological_tail[:, :, point_index]
                    finite_rows = np.all(np.isfinite(traj), axis=1)
                    traj = traj[finite_rows]
                    if traj.shape[0] != chronological_tail.shape[0] or traj.shape[0] < 16:
                        continue

                    if attractor_points.size:
                        end_distance = np.min(np.linalg.norm(traj[-1][None, :] - attractor_points, axis=1))
                        if end_distance < max(fp_tol * 4.0, 0.1):
                            continue

                    step_norms = np.linalg.norm(np.diff(traj, axis=0), axis=1)
                    if step_norms.size == 0 or step_norms.mean() / max(dt, 1e-12) < vel_tol * 0.5:
                        continue

                    centroid = traj.mean(axis=0)
                    radial = np.linalg.norm(traj - centroid, axis=1)
                    radius_mean = float(radial.mean())
                    if radius_mean < fp_tol:
                        continue

                    best_cluster = None
                    best_score = np.inf
                    for cluster in cycle_clusters:
                        center_delta = np.linalg.norm(centroid - cluster["center"])
                        radius_delta = abs(radius_mean - cluster["radius_mean"])
                        if (
                            center_delta > max(fp_tol * 30.0, 1.5)
                            or radius_delta > max(fp_tol * 30.0, 1.5)
                        ):
                            continue

                        mean_distance, max_distance = self._trajectory_set_distance(
                            traj,
                            np.asarray(cluster["trajectory"], dtype=float),
                            sample_count=64,
                        )
                        distance_tol = max(
                            fp_tol * 20.0,
                            0.35 * max(cluster["radius_mean"], radius_mean),
                            0.2,
                        )
                        if mean_distance > distance_tol or max_distance > distance_tol * 2.5:
                            continue

                        score = mean_distance + 0.25 * center_delta + 0.25 * radius_delta
                        if score < best_score:
                            best_score = score
                            best_cluster = cluster

                    if best_cluster is not None:
                        best_cluster["members"].append(int(point_index))
                        labels[point_index] = best_cluster["id"]

            if len(cycle_clusters) == 1:
                fallback_cluster = cycle_clusters[0]
                remaining_indices = np.flatnonzero(labels == -1)
                for point_index in remaining_indices:
                    traj = chronological_tail[:, :, point_index]
                    finite_rows = np.all(np.isfinite(traj), axis=1)
                    traj = traj[finite_rows]
                    if traj.shape[0] != chronological_tail.shape[0] or traj.shape[0] < 16:
                        continue

                    if attractor_points.size:
                        end_distance = np.min(np.linalg.norm(traj[-1][None, :] - attractor_points, axis=1))
                        if end_distance < max(fp_tol * 4.0, 0.1):
                            continue

                    step_norms = np.linalg.norm(np.diff(traj, axis=0), axis=1)
                    if step_norms.size == 0 or step_norms.mean() / max(dt, 1e-12) < vel_tol * 0.5:
                        continue

                    centroid = traj.mean(axis=0)
                    radial = np.linalg.norm(traj - centroid, axis=1)
                    radius_mean = float(radial.mean())
                    if radius_mean < fp_tol:
                        continue

                    center_delta = np.linalg.norm(centroid - fallback_cluster["center"])
                    radius_delta = abs(radius_mean - fallback_cluster["radius_mean"])
                    if (
                        center_delta > max(fp_tol * 40.0, 2.0)
                        or radius_delta > max(fp_tol * 40.0, 2.0)
                    ):
                        continue

                    fallback_cluster["members"].append(int(point_index))
                    labels[point_index] = fallback_cluster["id"]

        labels = labels.reshape(xx.shape)
        unresolved_mask = labels == -1

        return {
            "labels": labels,
            "attractors": attractors,
            "fixed_points": fixed_points,
            "unresolved_mask": unresolved_mask,
            "method": "heuristic",
        }

    def find_attractor_basins_manifold(
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
        fill_boundary=True,
    ):
        xx = np.asarray(xx, dtype=float)
        yy = np.asarray(yy, dtype=float)
        if xx.shape != yy.shape:
            raise ValueError("xx and yy must have the same shape.")

        x_range = (float(np.min(xx)), float(np.max(xx)))
        y_range = (float(np.min(yy)), float(np.max(yy)))

        if fixed_points is None:
            fixed_points = self.find_fixed_points(t, x_range, y_range)
        if saddle_manifolds is None:
            saddle_manifolds = self.find_saddle_manifolds(
                t,
                fixed_points=fixed_points,
                x_range=x_range,
                y_range=y_range,
            )

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
                    "stability": str(fixed_points["stability"][fixed_index]),
                    "eigenvalues": np.asarray(fixed_points["eigenvalues"][fixed_index]).copy(),
                }
            )

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
        labels = -np.ones(xx.shape, dtype=int)

        attracting_cells = {}
        if points.size:
            dx = float(x_coords[1] - x_coords[0]) if x_coords.size > 1 else 1.0
            dy = float(y_coords[1] - y_coords[0]) if y_coords.size > 1 else 1.0
            for fixed_index in attracting_indices:
                point = points[fixed_index]
                ix = int(np.rint((point[0] - x_coords[0]) / max(dx, 1e-12)))
                iy = int(np.rint((point[1] - y_coords[0]) / max(dy, 1e-12)))
                if 0 <= ix < labels.shape[1] and 0 <= iy < labels.shape[0]:
                    attracting_cells[int(fixed_index)] = (iy, ix)

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
                labels[region] = int(fixed_label_map[int(region_attractors[0])])
                continue

            sample_indices = self._select_region_samples(region, max_samples=max_region_samples)
            label_id = None
            for iy, ix in sample_indices:
                seed = np.array([xx[iy, ix], yy[iy, ix]], dtype=float)
                attractor = self._classify_seed_attractor(
                    t,
                    seed,
                    fixed_points,
                    attractors,
                    dt,
                    n_steps,
                    fp_tol,
                    vel_tol,
                    transient_fraction=transient_fraction,
                    cycle_window=cycle_window,
                    x_range=x_range,
                    y_range=y_range,
                )
                if attractor is None:
                    continue
                label_id = int(attractor["label"])
                break

            if label_id is not None:
                labels[region] = label_id

        unresolved_mask = labels < 0
        if fill_boundary and np.any(boundary_mask):
            labels = self._fill_barrier_labels(labels, boundary_mask)
            unresolved_mask = labels < 0

        return {
            "labels": labels,
            "attractors": attractors,
            "fixed_points": fixed_points,
            "unresolved_mask": unresolved_mask,
            "saddle_manifolds": saddle_manifolds,
            "boundary_mask": boundary_mask,
            "region_ids": region_ids,
            "method": "manifold",
        }

    def find_attractor_basins(
        self,
        t,
        xx,
        yy,
        fixed_points=None,
        method="manifold",
        **kwargs,
    ):
        method = str(method).lower()
        if method == "heuristic":
            return self.find_attractor_basins_heuristic(
                t,
                xx,
                yy,
                fixed_points=fixed_points,
                **kwargs,
            )
        if method == "manifold":
            return self.find_attractor_basins_manifold(
                t,
                xx,
                yy,
                fixed_points=fixed_points,
                **kwargs,
            )
        raise ValueError("method must be 'manifold' or 'heuristic'.")
