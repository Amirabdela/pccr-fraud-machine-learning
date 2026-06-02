"""
Model versioning and registry module.
Tracks trained model versions, metadata, and allows side-by-side metric comparison.
"""

from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime

from src.utils import setup_logger

logger = setup_logger()

_DEFAULT_REGISTRY_PATH = "models/registry.json"


class ModelRegistry:
    """
    Lightweight JSON-backed model registry for tracking experiment versions.

    Each registered entry stores model metadata (metrics, params, timestamps)
    so you can compare runs and select the best performing version.

    Usage:
        registry = ModelRegistry()
        registry.register("random_forest_v2", metrics={...}, params={...}, path="models/rf_v2.joblib")
        registry.list_versions()
        registry.compare(["random_forest_v1", "random_forest_v2"])
    """

    def __init__(self, registry_path: str = _DEFAULT_REGISTRY_PATH):
        self.registry_path = registry_path
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"versions": {}}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.registry_path) or ".", exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)
        logger.info(f"Registry saved → {self.registry_path}")

    def register(
        self,
        version_name: str,
        metrics: Dict[str, float],
        params: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a new model version in the registry.

        Args:
            version_name: Unique identifier for this version (e.g. 'rf_smote_v3').
            metrics: Dictionary of evaluation metrics (f1, precision, recall, mcc, roc_auc, etc.).
            params: Hyperparameters used for this version.
            model_path: Filesystem path to the serialised .joblib file.
            tags: Optional labels (e.g. ['baseline', 'production_candidate']).
            notes: Free-form notes about this experiment.
            overwrite: Whether to overwrite an existing entry with the same name.
        """
        if version_name in self._data["versions"] and not overwrite:
            logger.warning(
                f"Version '{version_name}' already in registry. "
                "Pass overwrite=True to replace it."
            )
            return

        entry = {
            "registered_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "params": params or {},
            "model_path": model_path,
            "tags": tags or [],
            "notes": notes or "",
        }
        self._data["versions"][version_name] = entry
        self._save()
        logger.info(f"Registered model version: '{version_name}'")

    def list_versions(self) -> List[Dict]:
        """Return all registered versions as a list of dicts."""
        versions = []
        for name, entry in self._data["versions"].items():
            versions.append({"version": name, **entry})
        versions.sort(key=lambda x: x.get("registered_at", ""), reverse=True)
        return versions

    def get(self, version_name: str) -> Optional[Dict]:
        """Retrieve a single version entry by name."""
        return self._data["versions"].get(version_name)

    def delete(self, version_name: str) -> bool:
        """Remove a version from the registry."""
        if version_name in self._data["versions"]:
            del self._data["versions"][version_name]
            self._save()
            logger.info(f"Deleted version: '{version_name}'")
            return True
        logger.warning(f"Version '{version_name}' not found in registry.")
        return False

    def compare(
        self,
        version_names: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a Markdown comparison table for the specified versions.

        Args:
            version_names: Versions to compare. Defaults to all registered versions.
            metrics: Metric keys to include. Defaults to all metrics in the first entry.

        Returns:
            str: Markdown-formatted comparison table.
        """
        if not self._data["versions"]:
            return "No versions registered yet."

        target_versions = version_names or list(self._data["versions"].keys())
        entries = {
            v: self._data["versions"][v]
            for v in target_versions
            if v in self._data["versions"]
        }

        if not entries:
            return "None of the specified versions found in registry."

        # Determine metrics to display
        if metrics is None:
            all_metric_keys: set = set()
            for e in entries.values():
                all_metric_keys.update(e.get("metrics", {}).keys())
            metrics = sorted(all_metric_keys)

        # Build table
        header = "| Metric | " + " | ".join(target_versions) + " |"
        divider = "|--------|" + "--------|" * len(target_versions)
        rows = [header, divider]

        for metric in metrics:
            row = f"| **{metric}** |"
            for v in target_versions:
                val = entries.get(v, {}).get("metrics", {}).get(metric, "–")
                if isinstance(val, float):
                    row += f" {val:.4f} |"
                else:
                    row += f" {val} |"
            rows.append(row)

        # Add timestamp row
        rows.append("| Registered |" + "".join(
            f" {entries.get(v, {}).get('registered_at', '–')[:10]} |"
            for v in target_versions
        ))

        return "\n".join(rows)

    def best_version(self, metric: str = "f1", higher_is_better: bool = True) -> Optional[str]:
        """Find the best registered version by a given metric."""
        if not self._data["versions"]:
            return None

        best_name, best_val = None, None
        for name, entry in self._data["versions"].items():
            val = entry.get("metrics", {}).get(metric)
            if val is None:
                continue
            if best_val is None:
                best_name, best_val = name, val
            elif higher_is_better and val > best_val:
                best_name, best_val = name, val
            elif not higher_is_better and val < best_val:
                best_name, best_val = name, val

        logger.info(f"Best version by {metric}: '{best_name}' ({best_val:.4f})")
        return best_name
