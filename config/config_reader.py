import yaml
from pathlib import Path

CONFIG_PATH: Path = Path(Path(__file__).parent.parent, 'config', 'config.yaml')


def read_config():
    with open(CONFIG_PATH, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

config = read_config()