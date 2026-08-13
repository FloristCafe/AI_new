from kedro.pipeline import Pipeline, node, pipeline

from kedro_nodes import export_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=export_node,
                inputs=["training_outputs", "preprocess_outputs"],
                outputs="onnx_export_outputs",
                name="export_deepfm_onnx",
            )
        ]
    )
