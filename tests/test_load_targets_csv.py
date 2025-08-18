import os
import csv
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.modules.setdefault('cv2', types.ModuleType('cv2'))
from app.calibrate import _load_targets_csv


def write_csv(path, header, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_load_targets_csv_case_insensitive_upper(tmp_path):
    config_dir = tmp_path
    header = ['row', 'col', 'R', 'G', 'B']
    rows = [
        [0, 0, 10, 20, 30],
        [0, 1, 40, 50, 60],
    ]
    write_csv(config_dir / 'ref_colors.csv', header, rows)

    target_rgb, labels = _load_targets_csv(str(config_dir), expected_n=2)

    assert labels == ['P01', 'P02']
    assert target_rgb.tolist() == [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]


def test_load_targets_csv_case_insensitive_lower(tmp_path):
    config_dir = tmp_path
    header = ['row', 'col', 'r', 'g', 'b']
    rows = [
        [0, 0, 15, 25, 35],
        [0, 1, 45, 55, 65],
    ]
    write_csv(config_dir / 'ref_colors.csv', header, rows)

    target_rgb, labels = _load_targets_csv(str(config_dir), expected_n=2)

    assert labels == ['P01', 'P02']
    assert target_rgb.tolist() == [[15.0, 25.0, 35.0], [45.0, 55.0, 65.0]]
