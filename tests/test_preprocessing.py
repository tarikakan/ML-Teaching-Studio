from __future__ import annotations

from ml_teaching_studio.core.datasets import load_builtin_dataset
from ml_teaching_studio.core.model_factory import create_model
from ml_teaching_studio.core.preprocessing import (
    PreprocessingOptions,
    build_training_pipeline,
    extract_feature_names,
)


def test_preprocessing_pipeline_fits_with_pca() -> None:
    bundle = load_builtin_dataset("Synthetic Regression")
    options = PreprocessingOptions(use_pca=True, pca_components=2)
    model = create_model("Linear Regression")
    pipeline = build_training_pipeline(bundle.dataframe, bundle.feature_columns, model, options)
    pipeline.fit(bundle.dataframe[bundle.feature_columns], bundle.dataframe[bundle.target_column])
    feature_names = extract_feature_names(pipeline, bundle.feature_columns, options)
    assert feature_names == ["PC1", "PC2"]
