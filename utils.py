import sys
import yaml
import re
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


def dms_to_decimal(dms_str):
    # Extrait les nombres et la lettre de direction
    # Ex: "48°46'34.52\"N" -> ('48', '46', '34.52', 'N')
    parts = re.split(r'[°\'"]+', dms_str.strip())

    deg = float(parts[0])
    mnt = float(parts[1])
    sec = float(parts[2])
    direction = parts[3].upper()

    # Calcul principal
    decimal = deg + (mnt / 60) + (sec / 3600)

    # Inversion du signe pour Sud et Ouest
    if direction in ['S', 'W', 'O']:
        decimal *= -1

    return decimal
