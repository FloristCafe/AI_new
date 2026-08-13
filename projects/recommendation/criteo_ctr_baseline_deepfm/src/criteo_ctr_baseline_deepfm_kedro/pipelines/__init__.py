from kedro.pipeline import Pipeline
from kedro.pipeline import pipeline as compose_pipeline

from criteo_ctr_baseline_deepfm_kedro.pipelines.export.pipeline import (
    create_pipeline as create_export_pipeline,
)
from criteo_ctr_baseline_deepfm_kedro.pipelines.preprocessing.pipeline import (
    create_pipeline as create_preprocessing_pipeline,
)
from criteo_ctr_baseline_deepfm_kedro.pipelines.training.pipeline import (
    create_pipeline as create_training_pipeline,
)


def create_all_pipelines() -> dict[str, Pipeline]:
    preprocessing = create_preprocessing_pipeline()
    training = create_training_pipeline()
    export = create_export_pipeline()
    default_pipeline = compose_pipeline([preprocessing, training, export])
    return {
        "__default__": default_pipeline,
        "preprocessing": preprocessing,
        "training": training,
        "export": export,
    }
