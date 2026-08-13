from pathlib import Path

from preprocess_criteo_deepfm import parse_args as parse_preprocess_args
from preprocess_criteo_deepfm import run_preprocess
from train_deepfm import parse_args as parse_train_args
from train_deepfm import run_training
from export_deepfm_onnx import parse_args as parse_export_args
from export_deepfm_onnx import run_export


PROJECT_ROOT = Path(
    r"D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm"
)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def preprocess_node() -> dict:
    args = parse_preprocess_args([])
    return run_preprocess(args)


def train_node(preprocess_outputs: dict) -> dict:
    args = parse_train_args([])
    args.train_path = preprocess_outputs["train_path"]
    args.valid_path = preprocess_outputs["valid_path"]
    args.feature_config = preprocess_outputs["feature_config_path"]
    return run_training(args)


def export_node(training_outputs: dict, preprocess_outputs: dict) -> dict:
    args = parse_export_args([])
    args.checkpoint_path = training_outputs["best_model_path"]
    args.metrics_path = training_outputs["metrics_path"]
    args.feature_config = preprocess_outputs["feature_config_path"]
    return run_export(args)
