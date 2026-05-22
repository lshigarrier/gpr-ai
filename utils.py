import sys
import yaml
from types import SimpleNamespace
from pathlib import Path

def get_conf(verbose=True):
    filename = sys.argv[1]
    config_path = Path('.') / f'{filename}.yaml'
    conf = yaml.safe_load(config_path.read_text())
    conf = SimpleNamespace(**conf)
    if verbose:
        print('-' * 70)
        for key, value in vars(conf).items():
            print(f'{key} : {value}')
        print('-' * 70)
    return conf