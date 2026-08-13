from kedro.pipeline import Pipeline

from criteo_ctr_baseline_deepfm_kedro.pipelines import create_all_pipelines


def register_pipelines() -> dict[str, Pipeline]:
    return create_all_pipelines()
