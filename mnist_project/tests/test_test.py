import torch
from tests import _PATH_DATA


def test_load_data():
    torch.load(f'{_PATH_DATA}/processed/test_processed.pt')
    assert 1 == 1
