from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from sfar.data import paper_name


class ResultWriter:
    def __init__(self, output_dir: str | Path, run_name: str | None, args: dict, config: dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or timestamp
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.args = args
        self.config = config

    def dataset_dir(self, dataset: str) -> Path:
        path = self.output_dir / paper_name(dataset).lower() / self.run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, path: str | Path, payload: Dict[str, Any]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        return path

    def save_tensor(self, dataset: str, filename: str, tensor: torch.Tensor) -> str:
        path = self.dataset_dir(dataset) / filename
        torch.save(tensor.cpu(), path)
        return str(path.relative_to(self.output_dir))

    def save_dataset_result(self, dataset: str, result: Dict[str, Any]) -> None:
        dataset_dir = self.dataset_dir(dataset)
        self.write_json(dataset_dir / "args.json", self.args)
        self.write_json(dataset_dir / "config.json", self.config)
        self.write_json(dataset_dir / "result.json", result)
        self._write_reconstruction_csv(dataset, result)
        self._write_classification_csv(dataset, result)

    def save_all_results(self, results: Dict[str, Any]) -> None:
        self.write_json(self.output_dir / f"results_{self.run_name}.json", results)
        self._write_summary_csv(results)

    def _write_reconstruction_csv(self, dataset: str, result: Dict[str, Any]) -> None:
        rows = [
            {"metric": metric, "value": value}
            for metric, value in result["feature_reconstruction"].items()
        ]
        self._write_csv(self.dataset_dir(dataset) / "feature_reconstruction.csv", rows, ["metric", "value"])

    def _write_classification_csv(self, dataset: str, result: Dict[str, Any]) -> None:
        fold_rows: List[Dict[str, Any]] = []
        mean_rows: List[Dict[str, Any]] = []
        for classifier, payload in result["node_classification"].items():
            for metric, value in payload["mean"].items():
                mean_rows.append(
                    {
                        "classifier": classifier,
                        "metric": metric,
                        "mean": value,
                        "std": payload["std"][metric],
                    }
                )
            for fold_id, fold in enumerate(payload["folds"]):
                row = {"classifier": classifier, "fold": fold_id}
                row.update(fold)
                fold_rows.append(row)

        self._write_csv(
            self.dataset_dir(dataset) / "node_classification.csv",
            mean_rows,
            ["classifier", "metric", "mean", "std"],
        )
        self._write_csv(
            self.dataset_dir(dataset) / "node_classification_folds.csv",
            fold_rows,
            ["classifier", "fold", "acc"],
        )

    def _write_summary_csv(self, results: Dict[str, Any]) -> None:
        rows: List[Dict[str, Any]] = []
        for dataset_name, result in results.items():
            for metric, value in result["feature_reconstruction"].items():
                rows.append(
                    {
                        "dataset": dataset_name,
                        "task": "feature_reconstruction",
                        "classifier": "",
                        "metric": metric,
                        "mean": value,
                        "std": "",
                    }
                )
            for classifier, payload in result["node_classification"].items():
                for metric, value in payload["mean"].items():
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "task": "node_classification",
                            "classifier": classifier,
                            "metric": metric,
                            "mean": value,
                            "std": payload["std"][metric],
                        }
                    )

        self._write_csv(
            self.output_dir / f"summary_{self.run_name}.csv",
            rows,
            ["dataset", "task", "classifier", "metric", "mean", "std"],
        )

    @staticmethod
    def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
