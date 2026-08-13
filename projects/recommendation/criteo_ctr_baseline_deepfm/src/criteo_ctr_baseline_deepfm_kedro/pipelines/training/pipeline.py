from kedro.pipeline import Pipeline, node, pipeline

from kedro_nodes import train_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_node,
                inputs="preprocess_outputs",
                outputs="training_outputs",
                name="train_criteo_deepfm",
            )
        ]
    )
