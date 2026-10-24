import numpy as np

from data_processor import label_run, run_split_masks, window_run


def test_label_run_respects_onset():
    y = label_run(fault=4, n_samples=10, onset=6)
    assert y.tolist() == [0] * 6 + [4] * 4
    assert label_run(0, 5, 3).tolist() == [0] * 5


def test_window_run_shapes_and_labels():
    data = np.arange(30, dtype=np.float32)[:, None] * np.ones((1, 3), np.float32)
    labels = label_run(2, 30, 20)
    x, y = window_run(data, labels, window=10, stride=5)
    assert x.shape == (5, 10, 3)
    assert x[1, 0, 0] == 5 and x[1, -1, 0] == 14  # second window covers samples 5..14
    assert y.tolist() == [0, 0, 0, 2, 2]  # window ends at 9, 14, 19, 24, 29


def test_window_run_too_short():
    x, y = window_run(np.zeros((5, 3), np.float32), np.zeros(5, np.int8), 10, 1)
    assert len(x) == 0 and len(y) == 0


def test_split_is_by_run():
    runs = np.repeat(np.arange(1, 21), 7)
    train, val = run_split_masks(runs)
    assert not set(runs[train]) & set(runs[val])
    assert set(runs[val]) == {5, 10, 15, 20}
