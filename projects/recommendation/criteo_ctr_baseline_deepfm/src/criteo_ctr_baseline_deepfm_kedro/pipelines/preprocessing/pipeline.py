from kedro.pipeline import Pipeline, node, pipeline

from kedro_nodes import preprocess_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_node,
                inputs=None,
                outputs="preprocess_outputs",
                name="preprocess_criteo_deepfm",
            )
        ]
    )
