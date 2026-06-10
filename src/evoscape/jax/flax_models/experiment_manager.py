"""
ExperimentManager
=================
Save and load Flax NNX experiments.

Directory layout:
    experiments/
    └── exp_001_<name>/
        ├── config.json        # hyperparameters, loss names, notes
        ├── loss_vals.json     # total loss value per epoch
        ├── checkpoints/
        │   ├── step_0050/
        │   └── ...
        └── final/
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import orbax.checkpoint as ocp
from flax import nnx


# ==============================================================================
# HELPERS
# ==============================================================================

def _next_exp_id(root: Path) -> str:
    existing = [
        d.name for d in root.iterdir()
        if d.is_dir() and d.name[:3].isdigit()
    ] if root.exists() else []
    if not existing:
        return "001"
    return f"{max(int(n[:3]) for n in existing) + 1:03d}"


def _json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _save_model(model: nnx.Module, path: Path) -> None:
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path / "state", state)
    checkpointer.wait_until_finished()


def _load_model(model: nnx.Module, path: Path) -> nnx.Module:
    path = path.resolve()
    graphdef, abstract_state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(path / "state", abstract_state)
    return nnx.merge(graphdef, state)


# ==============================================================================
# MAIN CLASS
# ==============================================================================

class ExperimentManager:

    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nnx.Module,
        config: dict[str, Any],
        loss_names: list[str],
        loss_vals: list[float],
        name: str = "",
        notes: str = "",
        checkpoints: dict[int, nnx.Module] | None = None,
    ) -> Path:
        """
        Save a complete experiment.

        Parameters
        ----------
        model       : trained nnx.Module
        config      : hyperparameters and data info
        loss_names  : names of the loss components used, e.g. ["mmd", "encoding"]
        loss_vals   : total loss value per epoch
        name        : optional name suffix for the experiment folder
        notes       : free-form comment
        checkpoints : optional dict {epoch: model_snapshot}
        """
        exp_id  = _next_exp_id(self.root)
        exp_dir = self.root / (f"exp_{exp_id}_{name}" if name else f"exp_{exp_id}")
        exp_dir.mkdir(parents=True)

        config["_timestamp"] = datetime.now().isoformat()
        config["_notes"]     = notes
        config["_losses"]    = loss_names
        (exp_dir / "config.json").write_text(
            json.dumps(_json_serializable(config), indent=2)
        )

        (exp_dir / "loss_vals.json").write_text(
            json.dumps([float(v) for v in loss_vals], indent=2)
        )

        _save_model(model, exp_dir / "final")

        if checkpoints:
            for epoch, snap in checkpoints.items():
                _save_model(snap, exp_dir / "checkpoints" / f"step_{epoch:04d}")

        print(f"[ExperimentManager] saved → {exp_dir}")
        return exp_dir

    def load(
        self,
        exp_name: str,
        model_template: nnx.Module,
        checkpoint_epoch: int | None = None,
    ) -> tuple[nnx.Module, dict, list[float]]:
        """
        Load an experiment.

        Parameters
        ----------
        exp_name         : folder name or numeric prefix (e.g. '001')
        model_template   : blank instance with the same architecture
        checkpoint_epoch : if given, loads an intermediate checkpoint instead of final

        Returns
        -------
        (model, config, loss_vals)
        """
        exp_dir   = self._resolve(exp_name)
        config    = json.loads((exp_dir / "config.json").read_text())
        loss_vals = json.loads((exp_dir / "loss_vals.json").read_text())

        if checkpoint_epoch is not None:
            ckpt_path = exp_dir / "checkpoints" / f"step_{checkpoint_epoch:04d}"
            if not ckpt_path.exists():
                available = sorted(p.name for p in (exp_dir / "checkpoints").iterdir()) \
                    if (exp_dir / "checkpoints").exists() else []
                raise FileNotFoundError(
                    f"No checkpoint at epoch {checkpoint_epoch}. Available: {available}"
                )
            model = _load_model(model_template, ckpt_path)
        else:
            model = _load_model(model_template, exp_dir / "final")

        print(f"[ExperimentManager] loaded ← {exp_dir}")
        return model, config, loss_vals

    def load_all_checkpoints(
        self,
        exp_name: str,
        model_template: nnx.Module,
    ) -> list[tuple[int, nnx.Module]]:
        """
        Load all intermediate checkpoints for an experiment.

        Returns
        -------
        List of (epoch, model) sorted by epoch.
        """
        exp_dir  = self._resolve(exp_name)
        ckpt_dir = exp_dir / "checkpoints"
        if not ckpt_dir.exists():
            return []

        result = []
        for path in sorted(ckpt_dir.iterdir()):
            try:
                epoch = int(path.name.replace("step_", ""))
                model = _load_model(model_template, path)
                result.append((epoch, model))
            except (ValueError, FileNotFoundError):
                pass

        return result
    
    def load_all_landscape_flax(
        self,
        exp_name: str,
        model_template: nnx.Module,
    ) -> list[nnx.Module]:
        checkpoints = self.load_all_checkpoints(exp_name, model_template)
        landscapes = [checkpoint[1].landscape_flax for checkpoint in checkpoints]
        return landscapes

    def _resolve(self, exp_name: str) -> Path:
        direct = self.root / exp_name
        if direct.exists():
            return direct
        # also match bare numeric prefix: "001" → "exp_001_..."
        matches = [
            d for d in self.root.iterdir()
            if d.name.startswith(exp_name) or d.name.startswith(f"exp_{exp_name}")
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous prefix '{exp_name}': {[m.name for m in matches]}")
        raise FileNotFoundError(f"Experiment '{exp_name}' not found in {self.root}")

    def summary(self, top_n: int = 3, convergence_threshold: float = None) -> None:
        """
        Print a summary of all saved experiments, ranked by final loss.

        Parameters
        ----------
        top_n                  : number of best experiments to highlight
        convergence_threshold  : if given, reports the epoch at which loss first
                                dropped below this value
        """
        exps = sorted(
            [d for d in self.root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if not exps:
            print("No experiments found.")
            return

        rows = []
        for exp_dir in exps:
            try:
                config    = json.loads((exp_dir / "config.json").read_text())
                loss_vals = json.loads((exp_dir / "loss_vals.json").read_text())

                loss_final = float(loss_vals[-1])
                loss_min   = float(min(loss_vals))
                epoch_min  = int(np.argmin(loss_vals))
                stability  = float(np.std(loss_vals[-20:]))  # std over last 20 epochs

                conv_epoch = None
                if convergence_threshold is not None:
                    below = [i for i, v in enumerate(loss_vals) if v < convergence_threshold]
                    conv_epoch = below[0] if below else None

                rows.append({
                    "name":        exp_dir.name,
                    "timestamp":   config.get("_timestamp", "")[:19],
                    "notes":       config.get("_notes", ""),
                    "loss_final":  loss_final,
                    "loss_min":    loss_min,
                    "epoch_min":   epoch_min,
                    "stability":   stability,
                    "conv_epoch":  conv_epoch,
                })
            except Exception as e:
                print(f"  [could not read {exp_dir.name}: {e}]")

        rows.sort(key=lambda r: r["loss_final"])
        best_names = {r["name"] for r in rows[:top_n]}

        # header
        conv_col = "  conv_epoch" if convergence_threshold is not None else ""
        print(f"\n{'':2} {'Name':<40} {'Date':<20} {'loss_final':>10} {'loss_min':>10} {'epoch_min':>10} {'stability':>10}{conv_col}  Notes")
        print("-" * (115 + (13 if convergence_threshold else 0)))

        for i, r in enumerate(rows):
            marker = f"#{i+1} " if r["name"] in best_names else "   "
            conv_str = f"  {r['conv_epoch'] if r['conv_epoch'] is not None else '—':>10}" if convergence_threshold is not None else ""
            print(
                f"{marker}"
                f"{r['name']:<40} "
                f"{r['timestamp']:<20} "
                f"{r['loss_final']:>10.4f} "
                f"{r['loss_min']:>10.4f} "
                f"{r['epoch_min']:>10d} "
                f"{r['stability']:>10.4f}"
                f"{conv_str}"
                f"  {r['notes'][:40]}"
            )

        print(f"\nTop {top_n}:")
        for i, r in enumerate(rows[:top_n]):
            print(f"  #{i+1} {r['name']}  —  loss_final={r['loss_final']:.4f}, loss_min={r['loss_min']:.4f} (epoch {r['epoch_min']})")