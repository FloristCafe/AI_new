import argparse
import json
from pathlib import Path

import torch

from deepfm_model import DeepFM


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DeepFM checkpoint to ONNX for Netron inspection."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\artifacts\deepfm_run\deepfm_model_best.pt",
        help="Path to the trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        default=r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\artifacts\feature_config.json",
        help="Path to the feature config json.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\artifacts\deepfm_run\deepfm_model_best.onnx",
        help="Output ONNX file path.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\artifacts\deepfm_run\metrics.json",
        help="Optional metrics.json used to infer architecture hyperparameters.",
    )
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--disable-fm",
        action="store_true",
        help="Disable the FM branch if the checkpoint was trained with this ablation.",
    )
    parser.add_argument(
        "--disable-deep",
        action="store_true",
        help="Disable the deep branch if the checkpoint was trained with this ablation.",
    )
    parser.add_argument(
        "--learn-global-bias",
        action="store_true",
        help="Set if the checkpoint was trained with a learnable global bias.",
    )
    parser.add_argument(
        "--fm-embedding-init-std",
        type=float,
        default=0.01,
        help="Kept for architectural parity when reconstructing the model.",
    )
    parser.add_argument(
        "--fm-scale",
        type=float,
        default=0.1,
        help="FM scaling factor used by the checkpoint architecture.",
    )
    return parser.parse_args(argv)


def load_feature_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model(feature_config: dict, args: argparse.Namespace) -> DeepFM:
    dense_cols = feature_config["dense_features"]
    dense_bucket_cols = feature_config["dense_bucket_features"]
    sparse_cols = feature_config["sparse_features"]
    dense_bucket_vocab_sizes = [
        feature_config["dense_bucket_rules"][col.replace("_bucket", "")]["vocab_size"]
        for col in dense_bucket_cols
    ]
    sparse_vocab_sizes = [
        feature_config["sparse_vocab_sizes"][col] for col in sparse_cols
    ]

    return DeepFM(
        dense_feature_count=len(dense_cols),
        dense_bucket_vocab_sizes=dense_bucket_vocab_sizes,
        sparse_vocab_sizes=sparse_vocab_sizes,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        global_bias_init=0.0,
        learnable_global_bias=args.learn_global_bias,
        use_fm=not args.disable_fm,
        use_deep=not args.disable_deep,
        fm_embedding_init_std=args.fm_embedding_init_std,
        fm_scale=args.fm_scale,
    )


def maybe_apply_training_config(args: argparse.Namespace) -> argparse.Namespace:
    metrics_path = Path(args.metrics_path)
    if not metrics_path.exists():
        return args

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics_payload = json.load(f)

    training_config = metrics_payload.get("training_config", {})
    args.embedding_dim = int(training_config.get("embedding_dim", args.embedding_dim))
    args.dropout = float(training_config.get("dropout", args.dropout))
    args.fm_scale = float(training_config.get("fm_scale", args.fm_scale))
    args.learn_global_bias = bool(
        training_config.get("learn_global_bias", args.learn_global_bias)
    )
    args.disable_fm = not bool(training_config.get("use_fm", not args.disable_fm))
    args.disable_deep = not bool(training_config.get("use_deep", not args.disable_deep))
    return args


def run_export(args: argparse.Namespace) -> dict[str, str]:
    args = maybe_apply_training_config(args)
    checkpoint_path = Path(args.checkpoint_path)
    feature_config_path = Path(args.feature_config)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_config = load_feature_config(feature_config_path)
    model = build_model(feature_config, args)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dense_dim = len(feature_config["dense_features"])
    dense_bucket_dim = len(feature_config["dense_bucket_features"])
    sparse_dim = len(feature_config["sparse_features"])

    dense_x = torch.zeros((1, dense_dim), dtype=torch.float32)
    dense_bucket_x = torch.zeros((1, dense_bucket_dim), dtype=torch.long)
    sparse_x = torch.zeros((1, sparse_dim), dtype=torch.long)

    torch.onnx.export(
        model,
        (dense_x, dense_bucket_x, sparse_x),
        str(output_path),
        input_names=["dense_x", "dense_bucket_x", "sparse_x"],
        output_names=["logits"],
        dynamic_axes={
            "dense_x": {0: "batch_size"},
            "dense_bucket_x": {0: "batch_size"},
            "sparse_x": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
    )

    return {
        "checkpoint_path": str(checkpoint_path),
        "feature_config_path": str(feature_config_path),
        "output_path": str(output_path),
    }


def main() -> None:
    args = parse_args()
    result = run_export(args)
    print(f"ONNX export finished: {result['output_path']}")


if __name__ == "__main__":
    main()
