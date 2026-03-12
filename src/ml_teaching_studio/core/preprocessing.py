"""Preprocessing pipeline creation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, OneHotEncoder, StandardScaler


@dataclass
class PreprocessingOptions:
    scale_numeric: bool = True
    normalize_numeric: bool = False
    impute_missing: bool = True
    encode_categorical: bool = True
    use_pca: bool = False
    pca_components: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_dense(data: Any) -> np.ndarray:
    return data.toarray() if hasattr(data, "toarray") else np.asarray(data)


def build_preprocessor(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    options: PreprocessingOptions,
) -> ColumnTransformer:
    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    numeric_steps: list[tuple[str, BaseEstimator | str]] = []
    categorical_steps: list[tuple[str, BaseEstimator | str]] = []

    if options.impute_missing:
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
        categorical_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    else:
        numeric_steps.append(("imputer", "passthrough"))
        categorical_steps.append(("imputer", "passthrough"))

    if options.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    if options.encode_categorical and categorical_columns:
        categorical_steps.append(
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        )
        categorical_transformer = Pipeline(steps=categorical_steps)
    else:
        categorical_transformer = "passthrough"

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric_transformer, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_transformer, categorical_columns))

    if not transformers:
        raise ValueError("No features remain after preprocessing setup.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_training_pipeline(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    estimator: BaseEstimator,
    options: PreprocessingOptions,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("preprocessor", build_preprocessor(dataframe, feature_columns, options)),
    ]
    if options.normalize_numeric:
        steps.append(("normalizer", Normalizer()))
    if options.use_pca and options.pca_components and options.pca_components > 0:
        steps.append(("dense", FunctionTransformer(_to_dense, accept_sparse=True)))
        steps.append(("reducer", PCA(n_components=options.pca_components)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def extract_feature_names(
    pipeline: Pipeline,
    feature_columns: list[str],
    options: PreprocessingOptions,
) -> list[str]:
    if options.use_pca and options.pca_components and options.pca_components > 0:
        return [f"PC{index + 1}" for index in range(options.pca_components)]

    preprocessor = pipeline.named_steps["preprocessor"]
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out(feature_columns))
    return feature_columns.copy()


def describe_preprocessing(options: PreprocessingOptions) -> str:
    parts = []
    parts.append("scaling enabled" if options.scale_numeric else "no numeric scaling")
    parts.append("row normalization enabled" if options.normalize_numeric else "no row normalization")
    parts.append("imputation enabled" if options.impute_missing else "missing values left unchanged")
    parts.append("categorical encoding enabled" if options.encode_categorical else "categorical encoding disabled")
    if options.use_pca and options.pca_components:
        parts.append(f"PCA enabled with {options.pca_components} components")
    else:
        parts.append("no dimensionality reduction")
    return ", ".join(parts)
