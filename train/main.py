from .config import config_from_args
from .orchestrator import TrainingRun


def main(argv=None) -> str:
    config = config_from_args(argv)
    return TrainingRun(config).execute()


if __name__ == "__main__":
    main()
