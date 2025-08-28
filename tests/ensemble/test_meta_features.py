import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.ensemble.meta_features import MetaFeaturesBuilder


@pytest.fixture
def tmp_oof_files(tmp_path):
    """Create temporary OOF preds and targets .npy files."""
    preds = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.0, 2.0, 3.0])

    preds_path = tmp_path / "oof_preds.npy"
    targets_path = tmp_path / "oof_targets.npy"

    np.save(preds_path, preds)
    np.save(targets_path, targets)

    return str(preds_path), str(targets_path)


def test_load_oof_success(tmp_oof_files):
    preds_path, targets_path = tmp_oof_files
    builder = MetaFeaturesBuilder(preds_path, targets_path, [])

    df = builder.load_oof()

    assert "oof_pred" in df.columns
    assert "target" in df.columns
    assert len(df) == 3
    assert df.iloc[0]["oof_pred"] == 1.0


def test_load_oof_shape_mismatch(tmp_path):
    preds = np.array([1.0, 2.0])
    targets = np.array([1.0, 2.0, 3.0])

    preds_path = tmp_path / "oof_preds.npy"
    targets_path = tmp_path / "oof_targets.npy"
    np.save(preds_path, preds)
    np.save(targets_path, targets)

    builder = MetaFeaturesBuilder(str(preds_path), str(targets_path), [])

    with pytest.raises(ValueError, match="Shape mismatch"):
        builder.load_oof()


def test_load_features_with_npy_and_csv(tmp_path):
    # npy feature
    npy_path = tmp_path / "feat.npy"
    np.save(npy_path, np.array([[1], [2], [3]]))

    # csv feature
    csv_path = tmp_path / "feat.csv"
    pd.DataFrame({"f2": [4, 5, 6]}).to_csv(csv_path, index=False)

    builder = MetaFeaturesBuilder("dummy_preds.npy", "dummy_targets.npy", [str(npy_path), str(csv_path)])
    df = builder.load_features()

    assert df.shape == (3, 2)
    assert list(df.columns) == [0, "f2"]


def test_load_features_unsupported_file(tmp_path):
    bad_file = tmp_path / "feat.txt"
    bad_file.write_text("bad data")

    builder = MetaFeaturesBuilder("p.npy", "t.npy", [str(bad_file)])
    with pytest.raises(ValueError, match="Unsupported feature file type"):
        builder.load_features()


def test_load_features_length_mismatch(tmp_path):
    f1 = tmp_path / "f1.npy"
    f2 = tmp_path / "f2.npy"

    np.save(f1, np.array([[1], [2]]))
    np.save(f2, np.array([[3], [4], [5]]))

    builder = MetaFeaturesBuilder("p.npy", "t.npy", [str(f1), str(f2)])
    with pytest.raises(ValueError, match="Feature length mismatch"):
        builder.load_features()


@patch("src.ensemble.meta_features.pd.DataFrame.to_csv")
def test_build_and_save(mock_to_csv, tmp_oof_files, tmp_path):
    preds_path, targets_path = tmp_oof_files

    # create extra feature file
    feat_path = tmp_path / "extra.npy"
    np.save(feat_path, np.array([[10], [20], [30]]))

    builder = MetaFeaturesBuilder(preds_path, targets_path, [str(feat_path)])
    out_csv = tmp_path / "meta.csv"

    df_meta = builder.build(save_path=str(out_csv))

    assert "oof_pred" in df_meta.columns
    assert "target" in df_meta.columns
    assert df_meta.shape[0] == 3
    mock_to_csv.assert_called_once()
