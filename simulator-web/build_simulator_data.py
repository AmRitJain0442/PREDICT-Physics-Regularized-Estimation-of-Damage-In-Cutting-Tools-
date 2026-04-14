import json
from pathlib import Path

import pandas as pd


def main():
    root = Path("..") / "results" / "cross_dataset_research"
    metrics = json.loads((root / "metrics_summary.json").read_text(encoding="utf-8"))
    life = pd.read_csv(root / "tables" / "nuaa_simulated_life_table.csv")

    markers = {
        "PHM2010": pd.read_csv(root / "tables" / "phm2010_failure_markers.csv").head(5).to_dict(orient="records"),
        "NASA": pd.read_csv(root / "tables" / "nasa_failure_markers.csv").head(5).to_dict(orient="records"),
        "UniWear": pd.read_csv(root / "tables" / "uniwear_failure_markers.csv").head(5).to_dict(orient="records"),
    }

    payload = {
        "generated_from": str(root),
        "thresholds_mm": {
            "PHM2010": metrics["PHM2010"]["failure_threshold_mm"],
            "NASA": metrics["NASA"]["failure_threshold_mm"],
            "UniWear": metrics["UniWear"]["failure_threshold_mm"],
        },
        "nuaa_life_table": life.to_dict(orient="records"),
        "taylor_like_equation": metrics["Simulator_NUAA"]["taylor_like_equation"],
        "best_setting": metrics["Simulator_NUAA"]["best_life_setting"],
        "worst_setting": metrics["Simulator_NUAA"]["worst_life_setting"],
        "dataset_metrics": {
            k: {
                "best_model": v.get("best_model"),
                "best_rmse": v.get("best_rmse"),
                "best_r2": v.get("best_r2"),
                "classification_roc_auc_mean": v.get("classification_roc_auc_mean"),
                "classification_f1_mean": v.get("classification_f1_mean"),
            }
            for k, v in metrics.items()
            if k in ["PHM2010", "NASA", "UniWear"]
        },
        "top_markers": markers,
    }

    out = Path("simulator_data.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
