import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(
    r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm"
)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUN_DIR = ARTIFACTS_DIR / "deepfm_run"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(
        page_title="Criteo DeepFM Dashboard",
        layout="wide",
    )

    st.title("Criteo CTR DeepFM Dashboard")
    st.caption("Read-only dashboard over local artifacts for experiment review.")

    metrics_path = RUN_DIR / "metrics.json"
    preprocess_summary_path = ARTIFACTS_DIR / "preprocess_summary.json"
    feature_config_path = ARTIFACTS_DIR / "feature_config.json"
    valid_pred_path = RUN_DIR / "valid_predictions.parquet"

    if not metrics_path.exists():
        st.error(f"Missing metrics file: {metrics_path}")
        return

    metrics = load_json(metrics_path)
    preprocess_summary = (
        load_json(preprocess_summary_path) if preprocess_summary_path.exists() else None
    )
    feature_config = (
        load_json(feature_config_path) if feature_config_path.exists() else None
    )
    valid_predictions = (
        pd.read_parquet(valid_pred_path) if valid_pred_path.exists() else None
    )

    final_valid = metrics["final_valid"]
    best_epoch = metrics["best_epoch"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Best Valid ROC-AUC", f"{best_epoch['valid_metrics']['roc_auc']:.4f}")
    metric_cols[1].metric("Best Valid PR-AUC", f"{best_epoch['valid_metrics']['pr_auc']:.4f}")
    metric_cols[2].metric("Best Valid LogLoss", f"{best_epoch['valid_metrics']['log_loss']:.4f}")
    metric_cols[3].metric("Final Valid ROC-AUC", f"{final_valid['roc_auc']:.4f}")

    history_df = pd.DataFrame(
        [
            {
                "epoch": row["epoch"],
                "train_bce_loss": row["train_bce_loss"],
                "train_roc_auc": row["train_metrics"]["roc_auc"],
                "valid_roc_auc": row["valid_metrics"]["roc_auc"],
                "valid_pr_auc": row["valid_metrics"]["pr_auc"],
                "valid_log_loss": row["valid_metrics"]["log_loss"],
            }
            for row in metrics["history"]
        ]
    )

    st.subheader("Training Curves")
    st.line_chart(
        history_df.set_index("epoch")[
            ["train_bce_loss", "valid_log_loss", "train_roc_auc", "valid_roc_auc"]
        ]
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Training Config")
        st.json(metrics["training_config"])

        if preprocess_summary is not None:
            st.subheader("Preprocess Summary")
            st.json(preprocess_summary)

    with right_col:
        if feature_config is not None:
            st.subheader("Feature Layout")
            feature_summary = {
                "dense_feature_count": len(feature_config["dense_features"]),
                "dense_bucket_feature_count": len(feature_config["dense_bucket_features"]),
                "sparse_feature_count": len(feature_config["sparse_features"]),
                "top_sparse_vocab_sizes": dict(
                    sorted(
                        feature_config["sparse_vocab_sizes"].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:10]
                ),
            }
            st.json(feature_summary)

        if valid_predictions is not None:
            st.subheader("Validation Prediction Snapshot")
            st.dataframe(valid_predictions.head(30), use_container_width=True)
            st.bar_chart(
                valid_predictions.assign(
                    prediction_bucket=pd.cut(
                        valid_predictions["prediction"],
                        bins=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
                        include_lowest=True,
                    ).astype(str)
                )
                .groupby("prediction_bucket", as_index=False)
                .size()
                .set_index("prediction_bucket")
            )


if __name__ == "__main__":
    main()
