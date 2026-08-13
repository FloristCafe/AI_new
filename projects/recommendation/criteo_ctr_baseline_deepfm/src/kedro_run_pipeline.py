from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner

from kedro_pipeline import create_pipeline


def main() -> None:
    pipeline: Pipeline = create_pipeline()
    runner = SequentialRunner()
    outputs = runner.run(pipeline, {})
    print("Kedro-style pipeline finished.")
    print(outputs)


if __name__ == "__main__":
    main()
