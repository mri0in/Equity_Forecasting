# src/ensemble/meta_features.py

"""
Combine out-of-fold (oof) predictions along with handcrafted features to build
the meta-learner dataset.
"""

import os
from typing import Union, List
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetaFeaturesBuilder:
    """
    Builds a DataFrame of meta-features by merging OOF predictions
    from base models with additional engineered features.
    """

    def __init__(
        self,
        oof_preds_path: str,
        oof_targets_path: str,
        feature_paths: List[str]
    ) -> None:
        """
        Initialize with paths to OOF preds, targets, and handcrafted features.

        Args:
            oof_preds_path (str): Path to .npy file of OOF predictions.
            oof_targets_path (str): Path to .npy file of true OOF targets.
            feature_paths (List[str]): List of CSV/NPY files containing extra features.
        """
        self.oof_preds_path = oof_preds_path
        self.oof_targets_path = oof_targets_path
        self.feature_paths = feature_paths

    def load_oof(self) -> pd.DataFrame:
        """
        Load OOF predictions and targets into a DataFrame.

        Returns:
            pd.DataFrame: Columns ['oof_pred', 'target'] with length = n_samples.
        """
        # Load arrays
        preds = np.load(self.oof_preds_path)
        targets = np.load(self.oof_targets_path)
        logger.info("Loaded OOF preds (%s) and targets (%s)",
                    preds.shape, targets.shape)

        if preds.shape != targets.shape:
            msg = f"Shape mismatch: preds {preds.shape}, targets {targets.shape}"
            logger.error(msg)
            raise ValueError(msg)

        # Build DataFrame
        df = pd.DataFrame({
            "oof_pred": preds.flatten(),
            "target": targets.flatten()
        })
        return df

    def load_features(self) -> pd.DataFrame:
        """
        Load all additional feature files and concatenate them column-wise.

        Supports .npy (NumPy) and .csv (Pandas).

        Returns:
            pd.DataFrame: Combined feature DataFrame.
        """
        dfs = []
        for path in self.feature_paths:
            if path.endswith(".npy"):
                arr = np.load(path)
                df = pd.DataFrame(arr)
                logger.info("Loaded .npy feature %s with shape %s", path, arr.shape)
            elif path.endswith(".csv"):
                df = pd.read_csv(path)
                logger.info("Loaded .csv feature %s with shape %s", path, df.shape)
            else:
                msg = f"Unsupported feature file type: {path}"
                logger.error(msg)
                raise ValueError(msg)
            dfs.append(df)

        # Ensure same length
        lengths = [len(df) for df in dfs]
        if not all(l == lengths[0] for l in lengths):
            msg = f"Feature length mismatch: {lengths}"
            logger.error(msg)
            raise ValueError(msg)

        # Concatenate side-by-side
        features_df = pd.concat(dfs, axis=1)
        logger.info("Combined features shape: %s", features_df.shape)
        return features_df

    def build(
        self,
        save_path: Union[str, None] = None
    ) -> pd.DataFrame:
        """
        Build the meta-feature DataFrame and optionally save to disk.

        Args:
            save_path (str | None): If provided, path (CSV) to save the combined DataFrame.

        Returns:
            pd.DataFrame: The assembled meta-feature table with columns:
                           ['oof_pred', 'target', ...additional features...]
        """
        df_oof = self.load_oof()
        df_feat = self.load_features()

        # Merge horizontally
        df_meta = pd.concat([df_oof, df_feat.reset_index(drop=True)], axis=1)
        logger.info("Built meta-feature DataFrame with shape %s", df_meta.shape)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df_meta.to_csv(save_path, index=False)
            logger.info("Saved meta-features to %s", save_path)

        return df_meta
