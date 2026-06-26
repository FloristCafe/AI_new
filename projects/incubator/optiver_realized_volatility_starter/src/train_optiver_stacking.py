import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from train_optiver_baseline import build_model, make_bootstrap_sample
from train_optiver_dual_cnn import (
    DualBranchCNN,
    OptiverTensorDataset,
    inverse_target_scale,
    run_epoch,
    standardize_tensor,
)


DEFAULT_FEATURE_TABLE = Path(
    r"D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7-8-9-10-11-13-14-15-16-17-18-19-20-21-22-23-24_times_200\features_knn\optiver_features_knn.parquet"
)
DEFAULT_TENSOR_DIR = Path(
    r"D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7-8-9-10-11-13-14-15-16-17-18-19-20-21-22-23-24_times_200\cnn_tensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a standard Optiver stacking model using OOF predictions from "
            "LightGBM, MLP and dual-branch CNN, then fit a meta regressor."
        )
    )
    parser.add_argument(
        "--feature-table",
        type=str,
        default=str(DEFAULT_FEATURE_TABLE),
        help="Path to the Optiver feature parquet used by tabular models.",
    )
    parser.add_argument(
        "--tensor-dir",
        type=str,
        default=str(DEFAULT_TENSOR_DIR),
        help="Directory containing book/trade tensors aligned with the same sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for stacking outputs. Defaults to feature table sibling stacking_models.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Outer chronological validation ratio based on unique time_id groups.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="GroupKFold split count used to build OOF predictions on the outer train set.",
    )
    parser.add_argument(
        "--meta-model",
        type=str,
        default="ridge",
        choices=["ridge", "lightgbm"],
        help="Meta learner used on OOF base predictions.",
    )
    parser.add_argument(
        "--meta-ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge alpha when --meta-model ridge.",
    )
    parser.add_argument(
        "--meta-use-augmented-features",
        action="store_true",
        help="If set, add prediction gaps and simple summary stats to the meta features.",
    )
    parser.add_argument(
        "--bagging-size",
        type=int,
        default=3,
        help="Bootstrap bagging size for LightGBM and MLP base models.",
    )
    parser.add_argument(
        "--lgbm-num-leaves",
        type=int,
        default=15,
        help="LightGBM num_leaves for the tabular base model.",
    )
    parser.add_argument(
        "--lgbm-max-depth",
        type=int,
        default=4,
        help="LightGBM max_depth for the tabular base model.",
    )
    parser.add_argument(
        "--lgbm-min-child-samples",
        type=int,
        default=40,
        help="LightGBM min_child_samples for the tabular base model.",
    )
    parser.add_argument(
        "--lgbm-learning-rate",
        type=float,
        default=0.03,
        help="LightGBM learning_rate for the tabular base model.",
    )
    parser.add_argument(
        "--lgbm-n-estimators",
        type=int,
        default=400,
        help="LightGBM boosting rounds for the tabular base model.",
    )
    parser.add_argument(
        "--mlp-hidden-layers",
        type=str,
        default="64,32",
        help="Comma-separated hidden layers for the tabular MLP base model.",
    )
    parser.add_argument(
        "--mlp-alpha",
        type=float,
        default=0.005,
        help="MLP alpha for the tabular base model.",
    )
    parser.add_argument(
        "--mlp-learning-rate-init",
        type=float,
        default=5e-4,
        help="MLP learning_rate_init for the tabular base model.",
    )
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=800,
        help="MLP max_iter for the tabular base model.",
    )
    parser.add_argument(
        "--cnn-mode",
        type=str,
        default="dual",
        choices=["book_only", "trade_only", "dual"],
        help="CNN branch mode used as the sequence base learner.",
    )
    parser.add_argument(
        "--cnn-epochs",
        type=int,
        default=30,
        help="Epochs for each CNN base learner fit.",
    )
    parser.add_argument(
        "--cnn-batch-size",
        type=int,
        default=64,
        help="Mini-batch size for the CNN base learner.",
    )
    parser.add_argument(
        "--cnn-learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate for the CNN base learner.",
    )
    parser.add_argument(
        "--cnn-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the CNN base learner.",
    )
    parser.add_argument(
        "--cnn-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension after CNN branch pooling.",
    )
    parser.add_argument(
        "--cnn-dropout",
        type=float,
        default=0.2,
        help="Dropout used in the CNN prediction head.",
    )
    parser.add_argument(
        "--book-weight",
        type=float,
        default=0.7,
        help="Book branch weight in dual CNN mode.",
    )
    parser.add_argument(
        "--trade-weight",
        type=float,
        default=0.3,
        help="Trade branch weight in dual CNN mode.",
    )
    parser.add_argument(
        "--cnn-internal-valid-ratio",
        type=float,
        default=0.1,
        help="Internal chronological validation ratio inside each CNN fit.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for CNN training.",
    )
    return parser.parse_args()


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.where(y_true == 0, 1e-8, y_true)
    return float(np.sqrt(np.mean(np.square((y_true - y_pred) / denominator))))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "rmspe": rmspe(y_true, y_pred),
    }


def load_multimodal_dataset(
    feature_table_path: Path,
    tensor_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    feature_df = pd.read_parquet(feature_table_path).copy()
    feature_df = feature_df.sort_values(["stock_id", "time_id"]).reset_index(drop=True)

    book_tensor = np.load(tensor_dir / "book_tensor.npy").astype(np.float32)
    trade_tensor = np.load(tensor_dir / "trade_tensor.npy").astype(np.float32)
    tensor_target = np.load(tensor_dir / "target.npy").astype(np.float32)
    tensor_stock_id = np.load(tensor_dir / "stock_id.npy")
    tensor_time_id = np.load(tensor_dir / "time_id.npy")

    tensor_df = pd.DataFrame(
        {
            "stock_id": tensor_stock_id,
            "time_id": tensor_time_id,
            "target_tensor": tensor_target,
            "tensor_index": np.arange(len(tensor_target)),
        }
    )
    tensor_df = tensor_df.sort_values(["stock_id", "time_id"]).reset_index(drop=True)

    merged = feature_df.merge(
        tensor_df,
        on=["stock_id", "time_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No overlapping rows found between feature table and CNN tensors.")

    if not np.allclose(
        merged["target"].to_numpy(dtype=np.float32),
        merged["target_tensor"].to_numpy(dtype=np.float32),
        atol=1e-8,
    ):
        raise ValueError("Target mismatch detected between feature table and tensor targets.")

    tensor_index = merged["tensor_index"].to_numpy()
    aligned_book = book_tensor[tensor_index]
    aligned_trade = trade_tensor[tensor_index]

    merged = merged.drop(columns=["target_tensor", "tensor_index"])
    sort_order = np.lexsort(
        (
            merged["stock_id"].to_numpy(),
            merged["time_id"].to_numpy(),
        )
    )
    merged = merged.iloc[sort_order].reset_index(drop=True)
    aligned_book = aligned_book[sort_order]
    aligned_trade = aligned_trade[sort_order]
    return merged, aligned_book, aligned_trade


def split_outer_train_valid(
    df: pd.DataFrame,
    book_tensor: np.ndarray,
    trade_tensor: np.ndarray,
    valid_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_time_ids = np.array(sorted(df["time_id"].unique()))
    if len(unique_time_ids) < 2:
        raise ValueError("Need at least two unique time_id groups to create train/valid split.")

    split_index = max(1, int(len(unique_time_ids) * (1 - valid_ratio)))
    split_index = min(split_index, len(unique_time_ids) - 1)

    train_time_ids = set(unique_time_ids[:split_index].tolist())
    valid_time_ids = set(unique_time_ids[split_index:].tolist())

    train_mask = df["time_id"].isin(train_time_ids).to_numpy()
    valid_mask = df["time_id"].isin(valid_time_ids).to_numpy()

    train_df = df.loc[train_mask].reset_index(drop=True)
    valid_df = df.loc[valid_mask].reset_index(drop=True)
    book_train = book_tensor[train_mask]
    book_valid = book_tensor[valid_mask]
    trade_train = trade_tensor[train_mask]
    trade_valid = trade_tensor[valid_mask]
    return train_df, valid_df, book_train, book_valid, trade_train, trade_valid


def train_bagged_predict(
    model_name: str,
    args: argparse.Namespace,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_predict: pd.DataFrame,
) -> np.ndarray:
    if args.bagging_size <= 1:
        model = build_model(model_name, args)
        model.fit(X_train, y_train)
        pred = model.predict(X_predict)
        if model_name == "mlp":
            pred = np.clip(pred, 0.0, None)
        return pred

    preds: list[np.ndarray] = []
    for bag_idx in range(args.bagging_size):
        bag_args = copy.deepcopy(args)
        seed = 42 + bag_idx
        X_boot, y_boot = make_bootstrap_sample(X_train, y_train, seed)
        model = build_model(model_name, bag_args)
        model.fit(X_boot, y_boot)
        bag_pred = model.predict(X_predict)
        if model_name == "mlp":
            bag_pred = np.clip(bag_pred, 0.0, None)
        preds.append(bag_pred)
    return np.mean(preds, axis=0)


def make_cnn_internal_split(
    groups: np.ndarray,
    valid_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.array(sorted(np.unique(groups)))
    if len(unique_groups) < 2:
        train_idx = np.arange(len(groups))
        return train_idx, train_idx

    split_index = max(1, int(len(unique_groups) * (1 - valid_ratio)))
    split_index = min(split_index, len(unique_groups) - 1)
    train_groups = set(unique_groups[:split_index].tolist())
    valid_groups = set(unique_groups[split_index:].tolist())

    train_idx = np.flatnonzero(np.isin(groups, list(train_groups)))
    valid_idx = np.flatnonzero(np.isin(groups, list(valid_groups)))
    if len(valid_idx) == 0:
        valid_idx = train_idx
    return train_idx, valid_idx


def train_cnn_predict(
    args: argparse.Namespace,
    book_fit_raw: np.ndarray,
    trade_fit_raw: np.ndarray,
    y_fit: np.ndarray,
    groups_fit: np.ndarray,
    book_predict_raw: np.ndarray,
    trade_predict_raw: np.ndarray,
) -> np.ndarray:
    internal_train_idx, internal_valid_idx = make_cnn_internal_split(
        groups_fit,
        args.cnn_internal_valid_ratio,
    )

    book_inner_train_raw = book_fit_raw[internal_train_idx]
    book_inner_valid_raw = book_fit_raw[internal_valid_idx]
    trade_inner_train_raw = trade_fit_raw[internal_train_idx]
    trade_inner_valid_raw = trade_fit_raw[internal_valid_idx]
    y_inner_train = y_fit[internal_train_idx]
    y_inner_valid = y_fit[internal_valid_idx]

    book_inner_train, book_inner_valid, _, _ = standardize_tensor(
        book_inner_train_raw,
        book_inner_valid_raw,
    )
    trade_inner_train, trade_inner_valid, _, _ = standardize_tensor(
        trade_inner_train_raw,
        trade_inner_valid_raw,
    )

    _, book_predict_scaled, book_mean, book_std = standardize_tensor(
        book_inner_train_raw,
        book_predict_raw,
    )
    _, trade_predict_scaled, trade_mean, trade_std = standardize_tensor(
        trade_inner_train_raw,
        trade_predict_raw,
    )

    target_mean = float(y_inner_train.mean())
    target_std = float(y_inner_train.std())
    if target_std < 1e-8:
        target_std = 1.0
    y_inner_train_scaled = (y_inner_train - target_mean) / target_std
    y_inner_valid_scaled = (y_inner_valid - target_mean) / target_std

    train_dataset = OptiverTensorDataset(
        book_inner_train,
        trade_inner_train,
        y_inner_train_scaled,
    )
    valid_dataset = OptiverTensorDataset(
        book_inner_valid,
        trade_inner_valid,
        y_inner_valid_scaled,
    )
    predict_dataset = OptiverTensorDataset(
        book_predict_scaled,
        trade_predict_scaled,
        np.zeros(len(book_predict_scaled), dtype=np.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.cnn_batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.cnn_batch_size,
        shuffle=False,
    )
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=args.cnn_batch_size,
        shuffle=False,
    )

    weight_sum = args.book_weight + args.trade_weight
    book_weight = args.book_weight / weight_sum if weight_sum > 0 else 1.0
    trade_weight = args.trade_weight / weight_sum if weight_sum > 0 else 0.0

    model = DualBranchCNN(
        mode=args.cnn_mode,
        book_channels=book_fit_raw.shape[1],
        trade_channels=trade_fit_raw.shape[1],
        hidden_dim=args.cnn_hidden_dim,
        dropout=args.cnn_dropout,
        book_weight=book_weight,
        trade_weight=trade_weight,
    ).to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.cnn_learning_rate,
        weight_decay=args.cnn_weight_decay,
    )

    best_valid_rmse = float("inf")
    best_predict_scaled = None

    for _ in range(args.cnn_epochs):
        run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=args.device,
        )
        _, valid_pred_scaled, _ = run_epoch(
            model=model,
            loader=valid_loader,
            optimizer=None,
            device=args.device,
        )
        _, predict_pred_scaled, _ = run_epoch(
            model=model,
            loader=predict_loader,
            optimizer=None,
            device=args.device,
        )

        valid_pred = inverse_target_scale(valid_pred_scaled, target_mean, target_std)
        valid_pred = np.clip(valid_pred, 0.0, None)
        valid_rmse = float(np.sqrt(mean_squared_error(y_inner_valid, valid_pred)))
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_predict_scaled = predict_pred_scaled.copy()

    predict_pred = inverse_target_scale(best_predict_scaled, target_mean, target_std)
    predict_pred = np.clip(predict_pred, 0.0, None)
    return predict_pred


def build_meta_feature_frame(
    pred_lightgbm: np.ndarray,
    pred_mlp: np.ndarray,
    pred_cnn: np.ndarray,
    use_augmented: bool,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "pred_lightgbm": pred_lightgbm,
            "pred_mlp": pred_mlp,
            "pred_cnn": pred_cnn,
        }
    )
    if use_augmented:
        frame["pred_mean"] = frame.mean(axis=1)
        frame["pred_std"] = frame.std(axis=1)
        frame["pred_lgbm_mlp_gap"] = frame["pred_lightgbm"] - frame["pred_mlp"]
        frame["pred_lgbm_cnn_gap"] = frame["pred_lightgbm"] - frame["pred_cnn"]
        frame["pred_mlp_cnn_gap"] = frame["pred_mlp"] - frame["pred_cnn"]
        frame["pred_lgbm_mlp_abs_gap"] = np.abs(frame["pred_lgbm_mlp_gap"])
        frame["pred_lgbm_cnn_abs_gap"] = np.abs(frame["pred_lgbm_cnn_gap"])
        frame["pred_mlp_cnn_abs_gap"] = np.abs(frame["pred_mlp_cnn_gap"])
    return frame


def build_meta_model(args: argparse.Namespace):
    if args.meta_model == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=args.meta_ridge_alpha)),
            ]
        )

    try:
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lightgbm is not installed in the current environment. Install it first with "
            "pip install lightgbm"
        ) from exc

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    num_leaves=15,
                    min_child_samples=20,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    verbosity=-1,
                    force_col_wise=True,
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    feature_table_path = Path(args.feature_table)
    tensor_dir = Path(args.tensor_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else feature_table_path.parent / "stacking_models"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df, book_tensor, trade_tensor = load_multimodal_dataset(feature_table_path, tensor_dir)
    train_df, valid_df, book_train, book_valid, trade_train, trade_valid = split_outer_train_valid(
        df,
        book_tensor,
        trade_tensor,
        args.valid_ratio,
    )

    feature_cols = [c for c in train_df.columns if c not in {"stock_id", "time_id", "target"}]
    X_train = train_df[feature_cols].reset_index(drop=True)
    y_train = train_df["target"].to_numpy(dtype=np.float32)
    groups_train = train_df["time_id"].to_numpy()

    X_valid = valid_df[feature_cols].reset_index(drop=True)
    y_valid = valid_df["target"].to_numpy(dtype=np.float32)

    unique_groups = np.unique(groups_train)
    if len(unique_groups) < 2:
        raise ValueError("Need at least two unique train time_id groups for OOF stacking.")
    n_splits = min(args.n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 after considering available groups.")

    gkf = GroupKFold(n_splits=n_splits)

    oof_pred_lightgbm = np.zeros(len(train_df), dtype=np.float32)
    oof_pred_mlp = np.zeros(len(train_df), dtype=np.float32)
    oof_pred_cnn = np.zeros(len(train_df), dtype=np.float32)

    for fold_idx, (fit_idx, holdout_idx) in enumerate(gkf.split(X_train, y_train, groups_train), start=1):
        X_fit = X_train.iloc[fit_idx].reset_index(drop=True)
        y_fit_series = pd.Series(y_train[fit_idx])
        X_holdout = X_train.iloc[holdout_idx].reset_index(drop=True)

        oof_pred_lightgbm[holdout_idx] = train_bagged_predict(
            model_name="lightgbm",
            args=args,
            X_train=X_fit,
            y_train=y_fit_series,
            X_predict=X_holdout,
        )
        oof_pred_mlp[holdout_idx] = train_bagged_predict(
            model_name="mlp",
            args=args,
            X_train=X_fit,
            y_train=y_fit_series,
            X_predict=X_holdout,
        )
        oof_pred_cnn[holdout_idx] = train_cnn_predict(
            args=args,
            book_fit_raw=book_train[fit_idx],
            trade_fit_raw=trade_train[fit_idx],
            y_fit=y_train[fit_idx],
            groups_fit=groups_train[fit_idx],
            book_predict_raw=book_train[holdout_idx],
            trade_predict_raw=trade_train[holdout_idx],
        )
        print(f"[Fold {fold_idx}/{n_splits}] OOF predictions finished.")

    valid_pred_lightgbm = train_bagged_predict(
        model_name="lightgbm",
        args=args,
        X_train=X_train,
        y_train=pd.Series(y_train),
        X_predict=X_valid,
    )
    valid_pred_mlp = train_bagged_predict(
        model_name="mlp",
        args=args,
        X_train=X_train,
        y_train=pd.Series(y_train),
        X_predict=X_valid,
    )
    valid_pred_cnn = train_cnn_predict(
        args=args,
        book_fit_raw=book_train,
        trade_fit_raw=trade_train,
        y_fit=y_train,
        groups_fit=groups_train,
        book_predict_raw=book_valid,
        trade_predict_raw=trade_valid,
    )

    meta_train_X = build_meta_feature_frame(
        pred_lightgbm=oof_pred_lightgbm,
        pred_mlp=oof_pred_mlp,
        pred_cnn=oof_pred_cnn,
        use_augmented=args.meta_use_augmented_features,
    )
    meta_valid_X = build_meta_feature_frame(
        pred_lightgbm=valid_pred_lightgbm,
        pred_mlp=valid_pred_mlp,
        pred_cnn=valid_pred_cnn,
        use_augmented=args.meta_use_augmented_features,
    )

    meta_model = build_meta_model(args)
    meta_model.fit(meta_train_X, y_train)
    meta_train_pred = meta_model.predict(meta_train_X)
    meta_valid_pred = meta_model.predict(meta_valid_X)
    meta_train_pred = np.clip(meta_train_pred, 0.0, None)
    meta_valid_pred = np.clip(meta_valid_pred, 0.0, None)

    metrics = {
        "train": evaluate_predictions(y_train, meta_train_pred),
        "valid": evaluate_predictions(y_valid, meta_valid_pred),
        "component_metrics": {
            "lightgbm_oof_train": evaluate_predictions(y_train, oof_pred_lightgbm),
            "mlp_oof_train": evaluate_predictions(y_train, oof_pred_mlp),
            "cnn_oof_train": evaluate_predictions(y_train, oof_pred_cnn),
            "lightgbm_valid": evaluate_predictions(y_valid, valid_pred_lightgbm),
            "mlp_valid": evaluate_predictions(y_valid, valid_pred_mlp),
            "cnn_valid": evaluate_predictions(y_valid, valid_pred_cnn),
        },
        "metadata": {
            "feature_table_path": str(feature_table_path),
            "tensor_dir": str(tensor_dir),
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "valid_ratio": args.valid_ratio,
            "n_splits": n_splits,
            "meta_model": args.meta_model,
            "meta_ridge_alpha": args.meta_ridge_alpha,
            "meta_feature_columns": list(meta_train_X.columns),
            "meta_use_augmented_features": args.meta_use_augmented_features,
            "bagging_size": args.bagging_size,
            "lgbm_num_leaves": args.lgbm_num_leaves,
            "lgbm_max_depth": args.lgbm_max_depth,
            "lgbm_min_child_samples": args.lgbm_min_child_samples,
            "lgbm_learning_rate": args.lgbm_learning_rate,
            "lgbm_n_estimators": args.lgbm_n_estimators,
            "mlp_hidden_layers": args.mlp_hidden_layers,
            "mlp_alpha": args.mlp_alpha,
            "mlp_learning_rate_init": args.mlp_learning_rate_init,
            "mlp_max_iter": args.mlp_max_iter,
            "cnn_mode": args.cnn_mode,
            "cnn_epochs": args.cnn_epochs,
            "cnn_batch_size": args.cnn_batch_size,
            "cnn_learning_rate": args.cnn_learning_rate,
            "cnn_weight_decay": args.cnn_weight_decay,
            "cnn_hidden_dim": args.cnn_hidden_dim,
            "cnn_dropout": args.cnn_dropout,
            "book_weight": args.book_weight,
            "trade_weight": args.trade_weight,
            "cnn_internal_valid_ratio": args.cnn_internal_valid_ratio,
            "device": args.device,
            "train_metric_note": "train metrics are computed after fitting the meta model on OOF base predictions.",
        },
    }

    metrics_path = output_dir / "stacking_metrics.json"
    oof_pred_path = output_dir / "stacking_oof_predictions.csv"
    valid_pred_path = output_dir / "stacking_valid_predictions.csv"

    oof_out = train_df[["stock_id", "time_id", "target"]].copy()
    oof_out["prediction_lightgbm_oof"] = oof_pred_lightgbm
    oof_out["prediction_mlp_oof"] = oof_pred_mlp
    oof_out["prediction_cnn_oof"] = oof_pred_cnn
    oof_out["prediction_stacking"] = meta_train_pred

    valid_out = valid_df[["stock_id", "time_id", "target"]].copy()
    valid_out["prediction_lightgbm"] = valid_pred_lightgbm
    valid_out["prediction_mlp"] = valid_pred_mlp
    valid_out["prediction_cnn"] = valid_pred_cnn
    valid_out["prediction_stacking"] = meta_valid_pred

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    oof_out.to_csv(oof_pred_path, index=False)
    valid_out.to_csv(valid_pred_path, index=False)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Metrics saved to: {metrics_path}")
    print(f"OOF predictions saved to: {oof_pred_path}")
    print(f"Validation predictions saved to: {valid_pred_path}")


if __name__ == "__main__":
    main()
