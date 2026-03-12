"""Central registry of supported models and hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .classification import CLASSIFICATION_BUILDERS
from .regression import REGRESSION_BUILDERS


@dataclass(frozen=True)
class HyperparameterSpec:
    name: str
    param_type: str
    default: Any
    widget: str
    description: str
    role: str
    recommended: str
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    choices: tuple[Any, ...] = ()
    log_scale: bool = False


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task_type: str
    family: str
    constructor: Callable[..., Any]
    description: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    important_hyperparameters: tuple[str, ...]
    hyperparameters: dict[str, HyperparameterSpec] = field(default_factory=dict)
    supports_probability: bool = False
    supports_feature_importance: bool = False
    supports_coefficients: bool = False


def _int_spec(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
    description: str,
    role: str,
    recommended: str,
) -> HyperparameterSpec:
    return HyperparameterSpec(
        name=name,
        param_type="int",
        default=default,
        widget="spin",
        min_value=minimum,
        max_value=maximum,
        step=1,
        description=description,
        role=role,
        recommended=recommended,
    )


def _float_spec(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
    description: str,
    role: str,
    recommended: str,
    *,
    step: float = 0.1,
    log_scale: bool = False,
) -> HyperparameterSpec:
    return HyperparameterSpec(
        name=name,
        param_type="float",
        default=default,
        widget="double",
        min_value=minimum,
        max_value=maximum,
        step=step,
        description=description,
        role=role,
        recommended=recommended,
        log_scale=log_scale,
    )


def _choice_spec(
    name: str,
    default: str,
    choices: tuple[str, ...],
    description: str,
    role: str,
    recommended: str,
) -> HyperparameterSpec:
    return HyperparameterSpec(
        name=name,
        param_type="choice",
        default=default,
        widget="combo",
        description=description,
        role=role,
        recommended=recommended,
        choices=choices,
    )


def _text_spec(
    name: str,
    default: str,
    description: str,
    role: str,
    recommended: str,
) -> HyperparameterSpec:
    return HyperparameterSpec(
        name=name,
        param_type="text",
        default=default,
        widget="line",
        description=description,
        role=role,
        recommended=recommended,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Linear Regression": ModelSpec(
        name="Linear Regression",
        task_type="regression",
        family="Linear Models",
        constructor=REGRESSION_BUILDERS["Linear Regression"],
        description="Fits a straight-line relationship between features and the target.",
        strengths=("Fast baseline", "Easy coefficient interpretation"),
        weaknesses=("Cannot model strong non-linearity", "Sensitive to correlated features"),
        important_hyperparameters=(),
        hyperparameters={},
        supports_coefficients=True,
    ),
    "Ridge Regression": ModelSpec(
        name="Ridge Regression",
        task_type="regression",
        family="Linear Models",
        constructor=REGRESSION_BUILDERS["Ridge Regression"],
        description="Linear regression with L2 regularization to stabilize coefficients.",
        strengths=("Handles multicollinearity", "Good regularized baseline"),
        weaknesses=("Still linear", "Alpha needs tuning"),
        important_hyperparameters=("alpha",),
        hyperparameters={
            "alpha": _float_spec(
                "alpha",
                1.0,
                0.001,
                100.0,
                "Strength of L2 regularization.",
                "Larger values shrink coefficients more aggressively.",
                "Start near 1.0, then sweep on a log scale.",
                step=0.1,
                log_scale=True,
            )
        },
        supports_coefficients=True,
    ),
    "Lasso Regression": ModelSpec(
        name="Lasso Regression",
        task_type="regression",
        family="Linear Models",
        constructor=REGRESSION_BUILDERS["Lasso Regression"],
        description="Linear regression with L1 regularization that can push some coefficients to zero.",
        strengths=("Performs feature selection", "Readable sparse models"),
        weaknesses=("Can underfit if alpha is too large", "Sensitive to scaling"),
        important_hyperparameters=("alpha",),
        hyperparameters={
            "alpha": _float_spec(
                "alpha",
                0.1,
                0.0001,
                10.0,
                "Strength of L1 regularization.",
                "Higher values zero-out more coefficients.",
                "Start small if you expect many useful features.",
                step=0.01,
                log_scale=True,
            )
        },
        supports_coefficients=True,
    ),
    "Decision Tree Regressor": ModelSpec(
        name="Decision Tree Regressor",
        task_type="regression",
        family="Trees",
        constructor=REGRESSION_BUILDERS["Decision Tree Regressor"],
        description="Splits the feature space into regions and predicts by region averages.",
        strengths=("Captures non-linearity", "No scaling required"),
        weaknesses=("Can overfit quickly", "Unstable across small data changes"),
        important_hyperparameters=("max_depth", "min_samples_split", "min_samples_leaf", "criterion"),
        hyperparameters={
            "max_depth": _int_spec(
                "max_depth",
                5,
                0,
                20,
                "Maximum depth of the tree. Use 0 for automatic depth.",
                "Controls how many nested decisions the tree can make.",
                "Shallow trees are safer for small datasets.",
            ),
            "min_samples_split": _int_spec(
                "min_samples_split",
                2,
                2,
                20,
                "Minimum samples required before splitting an internal node.",
                "Higher values make splits harder and reduce complexity.",
                "Increase this if the tree memorizes noise.",
            ),
            "min_samples_leaf": _int_spec(
                "min_samples_leaf",
                1,
                1,
                20,
                "Minimum samples allowed in a leaf.",
                "Higher values smooth predictions by forcing larger leaves.",
                "Useful for noisy regression targets.",
            ),
            "criterion": _choice_spec(
                "criterion",
                "squared_error",
                ("squared_error", "friedman_mse", "absolute_error"),
                "Function used to measure split quality.",
                "Changes what kind of error reduction the tree prefers.",
                "Keep squared_error as the default unless you need robustness to outliers.",
            ),
        },
        supports_feature_importance=True,
    ),
    "Random Forest Regressor": ModelSpec(
        name="Random Forest Regressor",
        task_type="regression",
        family="Ensembles",
        constructor=REGRESSION_BUILDERS["Random Forest Regressor"],
        description="Averaging many decision trees reduces variance compared with a single tree.",
        strengths=("Strong default performance", "Robust on tabular data"),
        weaknesses=("Less interpretable than one tree", "Can be slower"),
        important_hyperparameters=("n_estimators", "max_depth", "max_features", "min_samples_leaf"),
        hyperparameters={
            "n_estimators": _int_spec(
                "n_estimators",
                200,
                10,
                600,
                "Number of trees in the forest.",
                "More trees usually reduce variance but cost more time.",
                "Use 100 to 300 for most classroom demos.",
            ),
            "max_depth": _int_spec(
                "max_depth",
                8,
                0,
                30,
                "Maximum depth of each tree. Use 0 for automatic depth.",
                "Deeper trees can model finer patterns but overfit more easily.",
                "Moderate depth is a good teaching default.",
            ),
            "max_features": _choice_spec(
                "max_features",
                "sqrt",
                ("sqrt", "log2", "1.0"),
                "How many features each split can consider.",
                "Smaller subsets decorrelate trees and can improve generalization.",
                "Start with sqrt for classification-style behavior and 1.0 for more flexible trees.",
            ),
            "min_samples_leaf": _int_spec(
                "min_samples_leaf",
                1,
                1,
                20,
                "Minimum samples per leaf.",
                "Higher values make each tree smoother and less noisy.",
                "Raise this when you see large train-test gaps.",
            ),
        },
        supports_feature_importance=True,
    ),
    "Support Vector Regressor": ModelSpec(
        name="Support Vector Regressor",
        task_type="regression",
        family="Kernel Methods",
        constructor=REGRESSION_BUILDERS["Support Vector Regressor"],
        description="Uses a margin-based objective and optional kernels to fit smooth functions.",
        strengths=("Can model complex patterns", "Works well after scaling"),
        weaknesses=("Sensitive to hyperparameters", "Slower on larger datasets"),
        important_hyperparameters=("C", "kernel", "gamma"),
        hyperparameters={
            "C": _float_spec(
                "C",
                1.0,
                0.01,
                100.0,
                "Penalty for training errors.",
                "Higher C pushes harder to fit the training data.",
                "Sweep C on a log scale to see flexibility change.",
                step=0.1,
                log_scale=True,
            ),
            "kernel": _choice_spec(
                "kernel",
                "rbf",
                ("rbf", "linear", "poly"),
                "Kernel used to transform similarity between points.",
                "The kernel decides the shape of the function family.",
                "Use rbf first, then linear as a simpler baseline.",
            ),
            "gamma": _text_spec(
                "gamma",
                "scale",
                "Kernel width parameter for non-linear kernels.",
                "Higher gamma makes each training point influence a smaller neighborhood.",
                "Keep scale first, then try numeric values in sweeps.",
            ),
            "epsilon": _float_spec(
                "epsilon",
                0.1,
                0.01,
                2.0,
                "Width of the no-penalty tube around predictions.",
                "Larger epsilon ignores small errors and creates a smoother function.",
                "Increase epsilon when the data is noisy.",
                step=0.01,
            ),
        },
    ),
    "Gradient Boosting Regressor": ModelSpec(
        name="Gradient Boosting Regressor",
        task_type="regression",
        family="Boosting",
        constructor=REGRESSION_BUILDERS["Gradient Boosting Regressor"],
        description="Builds many shallow trees sequentially, each correcting earlier mistakes.",
        strengths=("Strong performance on tabular data", "Captures non-linearity"),
        weaknesses=("Can overfit with aggressive settings", "Training is sequential"),
        important_hyperparameters=("n_estimators", "learning_rate", "max_depth"),
        hyperparameters={
            "n_estimators": _int_spec(
                "n_estimators",
                150,
                10,
                500,
                "Number of boosting stages.",
                "More stages increase capacity if the learning rate is small enough.",
                "Pair this with learning_rate instead of tuning them separately.",
            ),
            "learning_rate": _float_spec(
                "learning_rate",
                0.05,
                0.01,
                1.0,
                "Contribution of each new tree.",
                "Lower values learn more slowly and usually need more trees.",
                "Start with 0.05 or 0.1.",
                step=0.01,
                log_scale=True,
            ),
            "max_depth": _int_spec(
                "max_depth",
                3,
                1,
                8,
                "Depth of each weak learner tree.",
                "Deeper learners capture interactions but raise variance.",
                "Keep trees shallow for teaching additive boosting behavior.",
            ),
        },
        supports_feature_importance=True,
    ),
    "MLP Regressor": ModelSpec(
        name="MLP Regressor",
        task_type="regression",
        family="Neural Networks",
        constructor=REGRESSION_BUILDERS["MLP Regressor"],
        description="Feed-forward neural network for learning non-linear regression functions.",
        strengths=("Flexible", "Learns complex smooth patterns"),
        weaknesses=("Sensitive to scaling", "Less transparent and harder to tune"),
        important_hyperparameters=(
            "hidden_layer_sizes",
            "activation",
            "learning_rate_init",
            "alpha",
            "batch_size",
            "max_iter",
        ),
        hyperparameters={
            "hidden_layer_sizes": _text_spec(
                "hidden_layer_sizes",
                "100,50",
                "Comma-separated hidden layer sizes.",
                "Controls network width and depth.",
                "Start with one or two moderate layers, not very deep stacks.",
            ),
            "activation": _choice_spec(
                "activation",
                "relu",
                ("relu", "tanh", "logistic"),
                "Activation function used in hidden layers.",
                "Changes the non-linear behavior of neurons.",
                "ReLU is a practical default for most demos.",
            ),
            "learning_rate_init": _float_spec(
                "learning_rate_init",
                0.001,
                0.0001,
                0.1,
                "Initial optimization step size.",
                "Too large can destabilize training; too small can stall learning.",
                "Start around 0.001.",
                step=0.0005,
                log_scale=True,
            ),
            "alpha": _float_spec(
                "alpha",
                0.0001,
                0.00001,
                0.1,
                "L2 regularization strength.",
                "Higher alpha dampens large weights.",
                "Increase alpha when the network overfits.",
                step=0.0001,
                log_scale=True,
            ),
            "batch_size": _int_spec(
                "batch_size",
                32,
                8,
                256,
                "Mini-batch size during training.",
                "Smaller batches add noise; larger batches are steadier.",
                "Use 32 or 64 unless the dataset is tiny.",
            ),
            "max_iter": _int_spec(
                "max_iter",
                500,
                100,
                2000,
                "Maximum optimization iterations.",
                "Higher values allow more learning time.",
                "Raise this before assuming the architecture is wrong.",
            ),
        },
    ),
    "Logistic Regression": ModelSpec(
        name="Logistic Regression",
        task_type="classification",
        family="Linear Models",
        constructor=CLASSIFICATION_BUILDERS["Logistic Regression"],
        description="Linear classifier that estimates class probabilities from a weighted sum of features.",
        strengths=("Fast baseline", "Probabilities are interpretable"),
        weaknesses=("Linear decision boundary", "Needs scaled numeric features"),
        important_hyperparameters=("C", "penalty", "solver"),
        hyperparameters={
            "C": _float_spec(
                "C",
                1.0,
                0.01,
                100.0,
                "Inverse regularization strength.",
                "Larger C means weaker regularization and a more flexible model.",
                "Sweep C on a log scale.",
                step=0.1,
                log_scale=True,
            ),
            "penalty": _choice_spec(
                "penalty",
                "l2",
                ("l2", "l1"),
                "Regularization type.",
                "L1 can zero out coefficients; L2 shrinks them smoothly.",
                "Use L2 first, then explore L1 for sparse models.",
            ),
            "solver": _choice_spec(
                "solver",
                "lbfgs",
                ("lbfgs", "liblinear", "saga"),
                "Optimization algorithm.",
                "Some solvers support more penalties or scale better.",
                "lbfgs is a robust default for L2; liblinear is reliable for L1.",
            ),
            "max_iter": _int_spec(
                "max_iter",
                1000,
                100,
                5000,
                "Maximum solver iterations.",
                "Raise it if you see convergence warnings.",
                "1000 is generous for classroom-sized datasets.",
            ),
        },
        supports_probability=True,
        supports_coefficients=True,
    ),
    "K-Nearest Neighbors": ModelSpec(
        name="K-Nearest Neighbors",
        task_type="classification",
        family="Instance-Based",
        constructor=CLASSIFICATION_BUILDERS["K-Nearest Neighbors"],
        description="Classifies a point by looking at nearby labeled examples.",
        strengths=("Easy to visualize", "Very intuitive"),
        weaknesses=("Sensitive to scaling", "Prediction can be slow"),
        important_hyperparameters=("n_neighbors", "weights", "metric"),
        hyperparameters={
            "n_neighbors": _int_spec(
                "n_neighbors",
                5,
                1,
                50,
                "Number of nearby points used to vote.",
                "Small values are flexible; large values are smoother.",
                "Use 3 to 15 for many small demos.",
            ),
            "weights": _choice_spec(
                "weights",
                "uniform",
                ("uniform", "distance"),
                "How neighbors vote.",
                "Distance weighting gives closer points more influence.",
                "Try distance when class borders look noisy.",
            ),
            "metric": _choice_spec(
                "metric",
                "minkowski",
                ("minkowski", "euclidean", "manhattan"),
                "Distance formula used to define neighborhood.",
                "Different metrics change what counts as close.",
                "Use euclidean after scaling unless the data suggests otherwise.",
            ),
        },
        supports_probability=True,
    ),
    "Decision Tree Classifier": ModelSpec(
        name="Decision Tree Classifier",
        task_type="classification",
        family="Trees",
        constructor=CLASSIFICATION_BUILDERS["Decision Tree Classifier"],
        description="Learns a series of if-then splits to separate classes.",
        strengths=("Highly visual", "Handles mixed feature effects"),
        weaknesses=("Overfits easily", "Can be unstable"),
        important_hyperparameters=("max_depth", "min_samples_split", "min_samples_leaf", "criterion"),
        hyperparameters={
            "max_depth": _int_spec(
                "max_depth",
                5,
                0,
                20,
                "Maximum tree depth. Use 0 for automatic depth.",
                "Controls how detailed the decision rules can become.",
                "Shallow trees are easier to interpret.",
            ),
            "min_samples_split": _int_spec(
                "min_samples_split",
                2,
                2,
                20,
                "Minimum samples before making a split.",
                "Higher values prevent tiny branches.",
                "Increase it when the tree memorizes outliers.",
            ),
            "min_samples_leaf": _int_spec(
                "min_samples_leaf",
                1,
                1,
                20,
                "Minimum samples per leaf.",
                "Larger leaves produce smoother class regions.",
                "Raise this to control variance.",
            ),
            "criterion": _choice_spec(
                "criterion",
                "gini",
                ("gini", "entropy", "log_loss"),
                "How split quality is measured.",
                "Changes what impurity reduction means.",
                "Use gini first; entropy is useful for comparison.",
            ),
        },
        supports_feature_importance=True,
        supports_probability=True,
    ),
    "Random Forest Classifier": ModelSpec(
        name="Random Forest Classifier",
        task_type="classification",
        family="Ensembles",
        constructor=CLASSIFICATION_BUILDERS["Random Forest Classifier"],
        description="Combines many trees trained on random subsets to improve stability.",
        strengths=("Strong default classifier", "Works well on tabular data"),
        weaknesses=("Less transparent than one tree", "Bigger models take longer"),
        important_hyperparameters=("n_estimators", "max_depth", "max_features", "min_samples_leaf"),
        hyperparameters={
            "n_estimators": _int_spec(
                "n_estimators",
                200,
                10,
                600,
                "Number of trees in the ensemble.",
                "More trees reduce variance up to a point.",
                "100 to 300 is a strong default range.",
            ),
            "max_depth": _int_spec(
                "max_depth",
                8,
                0,
                30,
                "Maximum tree depth. Use 0 for automatic depth.",
                "Deeper trees let each learner fit finer details.",
                "Cap depth if training accuracy becomes perfect too quickly.",
            ),
            "max_features": _choice_spec(
                "max_features",
                "sqrt",
                ("sqrt", "log2", "1.0"),
                "Number of features considered at each split.",
                "Lower values encourage more diverse trees.",
                "sqrt is the common starting point.",
            ),
            "min_samples_leaf": _int_spec(
                "min_samples_leaf",
                1,
                1,
                20,
                "Minimum samples per leaf.",
                "Higher values smooth the forest.",
                "Increase it when the model is too noisy.",
            ),
        },
        supports_feature_importance=True,
        supports_probability=True,
    ),
    "Support Vector Machine": ModelSpec(
        name="Support Vector Machine",
        task_type="classification",
        family="Kernel Methods",
        constructor=CLASSIFICATION_BUILDERS["Support Vector Machine"],
        description="Finds a separating boundary with maximum margin and optional kernel tricks.",
        strengths=("Powerful on medium-sized data", "Excellent boundary control"),
        weaknesses=("Requires scaling", "Hyperparameters matter a lot"),
        important_hyperparameters=("C", "kernel", "gamma"),
        hyperparameters={
            "C": _float_spec(
                "C",
                1.0,
                0.01,
                100.0,
                "Penalty for classification errors.",
                "Higher C weakens regularization and fits training points more tightly.",
                "Sweep across orders of magnitude.",
                step=0.1,
                log_scale=True,
            ),
            "kernel": _choice_spec(
                "kernel",
                "rbf",
                ("rbf", "linear", "poly"),
                "Kernel function.",
                "Defines the geometry of the separator.",
                "rbf is flexible; linear is the simplest comparison point.",
            ),
            "gamma": _text_spec(
                "gamma",
                "scale",
                "Neighborhood width for non-linear kernels.",
                "Larger gamma makes influence more local and boundaries wigglier.",
                "Use scale first, then test numeric values.",
            ),
        },
        supports_probability=True,
    ),
    "Naive Bayes": ModelSpec(
        name="Naive Bayes",
        task_type="classification",
        family="Probabilistic",
        constructor=CLASSIFICATION_BUILDERS["Naive Bayes"],
        description="Probabilistic classifier that assumes features are conditionally independent.",
        strengths=("Very fast", "Works surprisingly well on some problems"),
        weaknesses=("Strong independence assumption", "Less flexible on correlated features"),
        important_hyperparameters=("var_smoothing",),
        hyperparameters={
            "var_smoothing": _float_spec(
                "var_smoothing",
                1e-9,
                1e-12,
                1e-3,
                "Added variance for numerical stability.",
                "Higher values smooth class distributions more strongly.",
                "Usually leave this near the default unless the model is unstable.",
                step=1e-6,
                log_scale=True,
            )
        },
        supports_probability=True,
    ),
    "Gradient Boosting Classifier": ModelSpec(
        name="Gradient Boosting Classifier",
        task_type="classification",
        family="Boosting",
        constructor=CLASSIFICATION_BUILDERS["Gradient Boosting Classifier"],
        description="Sequentially adds small trees that correct previous classification errors.",
        strengths=("Strong performance", "Useful for showing additive learning"),
        weaknesses=("Can overfit with deep learners", "Slower than bagging"),
        important_hyperparameters=("n_estimators", "learning_rate", "max_depth"),
        hyperparameters={
            "n_estimators": _int_spec(
                "n_estimators",
                150,
                10,
                500,
                "Number of boosting stages.",
                "More stages raise capacity, especially with low learning rate.",
                "Tune together with learning_rate.",
            ),
            "learning_rate": _float_spec(
                "learning_rate",
                0.1,
                0.01,
                1.0,
                "Contribution of each new tree.",
                "Lower values learn cautiously; higher values can overshoot.",
                "0.05 to 0.1 is a practical start.",
                step=0.01,
                log_scale=True,
            ),
            "max_depth": _int_spec(
                "max_depth",
                3,
                1,
                8,
                "Depth of each weak learner tree.",
                "Deeper trees fit stronger interactions.",
                "Keep learners shallow to show boosting behavior clearly.",
            ),
        },
        supports_feature_importance=True,
        supports_probability=True,
    ),
    "AdaBoost Classifier": ModelSpec(
        name="AdaBoost Classifier",
        task_type="classification",
        family="Boosting",
        constructor=CLASSIFICATION_BUILDERS["AdaBoost Classifier"],
        description="Reweights hard examples so later weak learners focus on them more.",
        strengths=("Good teaching model for iterative reweighting", "Simple ensemble"),
        weaknesses=("Sensitive to noisy labels", "Less strong than newer boosting methods"),
        important_hyperparameters=("n_estimators", "learning_rate"),
        hyperparameters={
            "n_estimators": _int_spec(
                "n_estimators",
                100,
                10,
                400,
                "Number of weak learners.",
                "More rounds add capacity.",
                "Try 50 to 150 first.",
            ),
            "learning_rate": _float_spec(
                "learning_rate",
                1.0,
                0.01,
                2.0,
                "Weight given to each learner.",
                "Higher values make each step more aggressive.",
                "Decrease it if the model becomes unstable.",
                step=0.05,
                log_scale=True,
            ),
        },
        supports_probability=True,
    ),
    "MLP Classifier": ModelSpec(
        name="MLP Classifier",
        task_type="classification",
        family="Neural Networks",
        constructor=CLASSIFICATION_BUILDERS["MLP Classifier"],
        description="Feed-forward neural network that learns layered non-linear class boundaries.",
        strengths=("Flexible and expressive", "Good for teaching neural basics"),
        weaknesses=("Needs scaling", "Less interpretable than simpler models"),
        important_hyperparameters=(
            "hidden_layer_sizes",
            "activation",
            "learning_rate_init",
            "alpha",
            "batch_size",
            "max_iter",
        ),
        hyperparameters={
            "hidden_layer_sizes": _text_spec(
                "hidden_layer_sizes",
                "100",
                "Comma-separated hidden layer sizes.",
                "Controls model capacity and representation depth.",
                "Begin with one hidden layer before stacking more.",
            ),
            "activation": _choice_spec(
                "activation",
                "relu",
                ("relu", "tanh", "logistic"),
                "Activation function in hidden layers.",
                "Changes the shape of the learned representation.",
                "ReLU is a sensible default.",
            ),
            "learning_rate_init": _float_spec(
                "learning_rate_init",
                0.001,
                0.0001,
                0.1,
                "Initial optimizer step size.",
                "Too large can overshoot; too small can learn very slowly.",
                "Use 0.001 as the first baseline.",
                step=0.0005,
                log_scale=True,
            ),
            "alpha": _float_spec(
                "alpha",
                0.0001,
                0.00001,
                0.1,
                "L2 penalty on weights.",
                "Higher alpha discourages very large weights.",
                "Raise it when the network memorizes training data.",
                step=0.0001,
                log_scale=True,
            ),
            "batch_size": _int_spec(
                "batch_size",
                32,
                8,
                256,
                "Mini-batch size.",
                "Changes the noise level of optimization.",
                "32 or 64 is a good educational baseline.",
            ),
            "max_iter": _int_spec(
                "max_iter",
                400,
                100,
                2000,
                "Maximum training iterations.",
                "More iterations give the optimizer more time to converge.",
                "Increase this before concluding the architecture fails.",
            ),
        },
        supports_probability=True,
    ),
}


def list_models(task_type: str | None = None) -> list[ModelSpec]:
    models = list(MODEL_REGISTRY.values())
    if task_type:
        models = [spec for spec in models if spec.task_type == task_type]
    return sorted(models, key=lambda spec: spec.name)


def list_model_names(task_type: str | None = None) -> list[str]:
    return [spec.name for spec in list_models(task_type)]


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model: {name}") from exc


def default_model_for_task(task_type: str) -> str:
    return "Logistic Regression" if task_type == "classification" else "Linear Regression"
