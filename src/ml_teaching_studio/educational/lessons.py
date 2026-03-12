"""Lesson content for the theory module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lesson:
    slug: str
    title: str
    summary: str
    html: str
    action_label: str | None = None
    action_page: str | None = None


def _bullets(items: list[str]) -> str:
    return "".join(f"<li>{item}</li>" for item in items)


def _lesson_html(
    *,
    title: str,
    overview: str,
    example: str,
    why_it_matters: list[str],
    app_connection: list[str],
    mistakes: list[str],
    questions: list[str],
) -> str:
    return f"""
    <h2>{title}</h2>
    <p>{overview}</p>
    <h3>Concrete Example</h3>
    <p>{example}</p>
    <h3>Why It Matters</h3>
    <ul>{_bullets(why_it_matters)}</ul>
    <h3>Use This Inside The App</h3>
    <ul>{_bullets(app_connection)}</ul>
    <h3>Common Beginner Mistakes</h3>
    <ul>{_bullets(mistakes)}</ul>
    <h3>Questions To Ask Yourself</h3>
    <ul>{_bullets(questions)}</ul>
    """


LESSONS = [
    Lesson(
        slug="what-is-ml",
        title="What Is Machine Learning?",
        summary="Machine learning learns patterns from examples instead of hard-coded rules.",
        action_label="Open Datasets",
        action_page="Datasets",
        html=_lesson_html(
            title="What Is Machine Learning?",
            overview=(
                "Machine learning builds rules from data. Instead of writing every decision by hand, you give a model "
                "examples and let it estimate a pattern that links inputs to outputs. The important question is not "
                "only whether the model fits the examples you showed it, but whether it can generalize to new cases."
            ),
            example=(
                "If you want to predict house prices, you do not manually write a rule for every neighborhood and every "
                "size. You provide historical examples and let the model learn how features such as square footage, "
                "location, and age relate to price."
            ),
            why_it_matters=[
                "It changes software from rule-writing to pattern-estimation.",
                "The learned rule is always uncertain, so evaluation matters as much as training.",
                "Different models learn different kinds of patterns, which is why model comparison is essential.",
            ],
            app_connection=[
                "Use the Datasets page to inspect what information is actually available.",
                "Use the Training page to see how a learned rule performs on unseen data.",
                "Use Compare Runs to see that two models can fit the same dataset in very different ways.",
            ],
            mistakes=[
                "Treating a trained model as if it discovered a perfect law.",
                "Assuming a high training score means the model learned the right pattern.",
                "Ignoring whether the data even contains useful signal for the target.",
            ],
            questions=[
                "What pattern is the model trying to learn from this dataset?",
                "How will I tell the difference between memorization and generalization?",
                "What would a reasonable baseline be before trying a more complex model?",
            ],
        ),
    ),
    Lesson(
        slug="supervised-vs-unsupervised",
        title="Supervised vs Unsupervised Learning",
        summary="Supervised learning uses known targets; unsupervised learning looks for structure without labels.",
        action_label="Try a Synthetic Dataset",
        action_page="Datasets",
        html=_lesson_html(
            title="Supervised vs Unsupervised Learning",
            overview=(
                "In supervised learning, each training example includes the answer you want the model to predict. "
                "In unsupervised learning, there is no target column, so the goal is to discover structure such as "
                "clusters, directions of variation, or compressed representations."
            ),
            example=(
                "Classifying flower species from measurements is supervised because the species label is known. "
                "Using PCA to compress many numeric features into two visual dimensions is unsupervised because "
                "the method is not trying to predict the target."
            ),
            why_it_matters=[
                "It determines whether you are predicting an answer or exploring structure.",
                "It changes what success means: prediction quality versus useful organization of data.",
                "Some unsupervised tools are still valuable inside a supervised workflow, especially for visualization.",
            ],
            app_connection=[
                "Most training tools in ML-Teaching Studio are supervised.",
                "PCA plots in the Datasets and Visualizations pages show how unsupervised tools can still support understanding.",
                "Use synthetic datasets to see how visible structure sometimes makes a supervised problem easier.",
            ],
            mistakes=[
                "Assuming unsupervised plots directly prove a classifier will perform well.",
                "Thinking PCA or clustering can replace target-aware evaluation.",
                "Calling every data-analysis step machine learning, even when it is only visualization.",
            ],
            questions=[
                "Do I have a target column I am trying to predict?",
                "Am I evaluating prediction quality or only looking for structure?",
                "How does this exploratory view help my later supervised choices?",
            ],
        ),
    ),
    Lesson(
        slug="classification-vs-regression",
        title="Classification vs Regression",
        summary="Classification predicts categories; regression predicts continuous values.",
        action_label="Open Models",
        action_page="Models",
        html=_lesson_html(
            title="Classification vs Regression",
            overview=(
                "Classification predicts a discrete label such as spam versus not spam, while regression predicts a "
                "continuous value such as price or demand. The task type determines which models, metrics, and visuals "
                "make sense."
            ),
            example=(
                "Predicting whether a patient has a disease is classification. Predicting blood pressure level is regression. "
                "The first needs confusion matrices and ROC-style reasoning, while the second needs residual analysis."
            ),
            why_it_matters=[
                "The same model family can behave differently depending on the task.",
                "Accuracy is meaningful for labels but meaningless for a continuous target.",
                "The right plot depends on whether you are separating classes or fitting a numeric relationship.",
            ],
            app_connection=[
                "The Models page separates supported classification and regression models.",
                "The Training page changes its available plots when you switch task type.",
                "Interactive model demos let you compare decision boundaries for classification and fitted curves for regression.",
            ],
            mistakes=[
                "Using classification metrics on regression problems.",
                "Forcing a numeric target into classification without a real reason.",
                "Assuming a model that works well on one task type will behave similarly on the other.",
            ],
            questions=[
                "Is my target fundamentally a category or a quantity?",
                "What does a correct prediction look like for this task?",
                "Which error types matter most in this setting?",
            ],
        ),
    ),
    Lesson(
        slug="train-validation-test",
        title="Training, Validation, and Test Sets",
        summary="Separate data roles keep evaluation honest.",
        action_label="Open Training",
        action_page="Training",
        html=_lesson_html(
            title="Training, Validation, and Test Sets",
            overview=(
                "The training set teaches the model, validation is used to choose settings, and the test set estimates "
                "how well the final choice generalizes. If the same data is used for everything, the model selection "
                "process becomes overly optimistic."
            ),
            example=(
                "Suppose you try ten SVM settings and keep the best one. If you choose and report performance on the same "
                "test split, the final number partly reflects lucky tuning to that split instead of only true generalization."
            ),
            why_it_matters=[
                "It prevents you from mistaking tuned performance for genuine performance.",
                "It makes model comparison fair because every model is judged on unseen data.",
                "It explains why train and test metrics should be read together, not separately.",
            ],
            app_connection=[
                "The Training page reports train and test metrics side by side.",
                "The Hyperparameter Lab helps you see how choosing settings can overfit the split if you are careless.",
                "Compare Runs lets you store the split-related choices that matter for fair comparisons.",
            ],
            mistakes=[
                "Tuning on the same data used for final reporting.",
                "Only reading the best single number from the experiment.",
                "Changing several parts of the pipeline at once and forgetting which split was used.",
            ],
            questions=[
                "Which data taught the model and which data judged it?",
                "Could this score be inflated by repeated tuning on one split?",
                "If I reran the experiment with a different seed, would the conclusion still hold?",
            ],
        ),
    ),
    Lesson(
        slug="feature-scaling",
        title="Feature Scaling and Normalization",
        summary="Some models compare distances or optimize weights, so input scale matters.",
        action_label="Open Training",
        action_page="Training",
        html=_lesson_html(
            title="Feature Scaling and Normalization",
            overview=(
                "Some models depend strongly on the relative scale of features. If one feature is measured in thousands "
                "and another in tenths, the large-scale feature can dominate distance calculations or gradient-based optimization."
            ),
            example=(
                "In KNN, a difference of 500 in income can overwhelm a difference of 2 in education years unless the "
                "features are scaled. The model then acts as if income is almost the only feature that matters."
            ),
            why_it_matters=[
                "Distance-based models such as KNN use scale directly.",
                "SVM and logistic regression often train more reliably when features are scaled.",
                "Interpretation of coefficients becomes cleaner when features are on comparable scales.",
            ],
            app_connection=[
                "Turn scaling on and off in the Training page and compare the outcome.",
                "Use the Interactive Example compare mode to contrast a scale-sensitive model against a more scale-robust one.",
                "Look at feature histograms before deciding whether raw scales are very different.",
            ],
            mistakes=[
                "Tuning KNN or SVM before fixing scaling.",
                "Assuming tree models need the same scaling care as linear or distance-based models.",
                "Confusing feature scaling with row normalization and using them interchangeably.",
            ],
            questions=[
                "Could one feature dominate because of units alone?",
                "Does this model compare distances or optimize coefficients directly?",
                "Did preprocessing change the result more than the model itself?",
            ],
        ),
    ),
    Lesson(
        slug="encoding",
        title="Encoding Categorical Variables",
        summary="Models need numeric inputs, so categorical values usually must be encoded.",
        action_label="Open Datasets",
        action_page="Datasets",
        html=_lesson_html(
            title="Encoding Categorical Variables",
            overview=(
                "Most scikit-learn models expect numeric features. Categorical values such as city names or product types "
                "must usually be transformed into numeric form without accidentally inventing false order or distance."
            ),
            example=(
                "If a feature has values red, blue, and green, mapping them to 1, 2, and 3 suggests a numeric ordering that "
                "does not exist. One-hot encoding avoids this by creating one indicator column per category."
            ),
            why_it_matters=[
                "Incorrect encoding can inject false meaning into the data.",
                "Different encodings change how models interpret category relationships.",
                "Real tabular datasets often combine numeric and categorical information.",
            ],
            app_connection=[
                "CSV import lets you bring in mixed-type data.",
                "The Training page can apply one-hot encoding in preprocessing.",
                "Use Compare Runs to see whether encoded categorical information changes model rankings.",
            ],
            mistakes=[
                "Assigning arbitrary integers to unordered categories and calling it done.",
                "Forgetting that one-hot encoding increases feature count.",
                "Blaming the model when the real problem is an invalid representation of the input.",
            ],
            questions=[
                "Do these categories have a meaningful order?",
                "Will this encoding change the geometry seen by the model?",
                "How much did the preprocessing step change performance?",
            ],
        ),
    ),
    Lesson(
        slug="missing-data",
        title="Missing Data Handling",
        summary="Missing values can break training or bias results unless you handle them intentionally.",
        action_label="Open Datasets",
        action_page="Datasets",
        html=_lesson_html(
            title="Missing Data Handling",
            overview=(
                "Missing values are common in practical datasets. A model can fail entirely, or it can silently learn from "
                "a distorted subset, if missing data is ignored without thought."
            ),
            example=(
                "If expensive homes are more likely to have missing tax information, dropping all rows with missing values "
                "may change the distribution of the target and bias the training set."
            ),
            why_it_matters=[
                "Missingness can remove useful data or distort the sample.",
                "Different imputation strategies create different assumptions about the data.",
                "Some models are more tolerant than others, but every approach changes the learning problem.",
            ],
            app_connection=[
                "Use the missing-value overview before deciding how to preprocess.",
                "Turn imputation on and off in Training to see whether the result is sensitive to missingness.",
                "Save runs with different preprocessing choices and compare them directly.",
            ],
            mistakes=[
                "Dropping rows without checking what kind of cases disappear.",
                "Assuming the imputed values are true observations.",
                "Treating missing-data handling as a minor technical cleanup step instead of a modeling choice.",
            ],
            questions=[
                "How much data is missing and where?",
                "Could the missingness itself be informative?",
                "Did the handling choice change both score and interpretation?",
            ],
        ),
    ),
    Lesson(
        slug="bias-variance",
        title="Bias-Variance Tradeoff",
        summary="Simple models can miss patterns; very flexible models can chase noise.",
        action_label="Open Hyperparameter Lab",
        action_page="Hyperparameter Lab",
        html=_lesson_html(
            title="Bias-Variance Tradeoff",
            overview=(
                "Bias is error from being too simple; variance is error from reacting too strongly to the particular training sample. "
                "Many hyperparameters control where a model sits on this spectrum."
            ),
            example=(
                "A shallow decision tree may miss a curved boundary entirely, which is high bias. A very deep tree may wrap around noise in the "
                "training data, which is high variance."
            ),
            why_it_matters=[
                "It explains why the best model is rarely the most flexible one.",
                "It connects hyperparameters to behavior instead of treating them as magic numbers.",
                "It helps you interpret train-test gaps as a learning signal, not just a bad surprise.",
            ],
            app_connection=[
                "Use Hyperparameter Lab to sweep depth, k, C, alpha, or learning rate and watch the curve change.",
                "Use the Interactive Example compare mode to contrast a simple baseline against a flexible model on the same pattern.",
                "Look for the region where test performance peaks before the train-test gap becomes too wide.",
            ],
            mistakes=[
                "Treating every training improvement as genuine progress.",
                "Using only one split and concluding too much about model stability.",
                "Changing several flexibility controls at once, which hides the real cause of the shift.",
            ],
            questions=[
                "Is the model currently too rigid or too reactive?",
                "Which hyperparameter is changing flexibility most directly?",
                "Did the test score improve for the same reason the training score improved?",
            ],
        ),
    ),
    Lesson(
        slug="overfitting-underfitting",
        title="Overfitting and Underfitting",
        summary="Compare training and test performance to see whether the model generalizes.",
        action_label="Open Visualizations",
        action_page="Visualizations",
        html=_lesson_html(
            title="Overfitting and Underfitting",
            overview=(
                "Underfitting happens when the model is too simple to capture the main signal. Overfitting happens when it "
                "captures details that do not generalize well beyond the training data."
            ),
            example=(
                "A KNN classifier with very large k can smooth away real local structure and underfit. A decision tree with unrestricted depth "
                "can carve tiny regions around training points and overfit."
            ),
            why_it_matters=[
                "It is one of the central reasons train and test scores differ.",
                "It turns hyperparameter tuning into reasoning about behavior, not blind search.",
                "It explains why visual plots are useful even when you already have metrics.",
            ],
            app_connection=[
                "Use learning curves and validation curves in Visualizations and Hyperparameter Lab.",
                "In the Interactive Example section, compare the same model at default versus tuned settings.",
                "Read the explanation panels, which explicitly comment on likely overfitting or underfitting patterns.",
            ],
            mistakes=[
                "Calling every low test score overfitting, even when the train score is also low.",
                "Looking only at accuracy or R² without checking the train-test gap.",
                "Assuming more parameters or more trees always means overfitting.",
            ],
            questions=[
                "Are both train and test weak, or only test?",
                "What changed visually when the model became more flexible?",
                "Would more data help, or is the model choice itself the problem?",
            ],
        ),
    ),
    Lesson(
        slug="cross-validation",
        title="Cross-Validation",
        summary="Cross-validation reduces dependence on a single train/test split.",
        action_label="Open Hyperparameter Lab",
        action_page="Hyperparameter Lab",
        html=_lesson_html(
            title="Cross-Validation",
            overview=(
                "A single train/test split can be lucky or unlucky. Cross-validation repeats the train-evaluate cycle on "
                "different folds so you can estimate performance more stably."
            ),
            example=(
                "On a small dataset, one split might leave most hard cases in the training set, making the test score look better than usual. "
                "Another split might do the opposite. Cross-validation averages across those accidents."
            ),
            why_it_matters=[
                "It reduces overconfidence in one split.",
                "It is especially important when datasets are small or noisy.",
                "It supports fairer hyperparameter tuning because settings are judged across multiple folds.",
            ],
            app_connection=[
                "The Hyperparameter Lab also supports repeated experiments across seeds, which teaches a related stability idea.",
                "Compare Runs can store different seeds so you can inspect how much rankings change.",
                "If the best setting changes wildly across repetitions, your conclusion is fragile.",
            ],
            mistakes=[
                "Treating one split as the final truth.",
                "Using cross-validation results and final test results interchangeably.",
                "Ignoring variability and only reporting the best average score.",
            ],
            questions=[
                "How stable is this conclusion across different splits?",
                "Is the dataset large enough that one split is already reliable?",
                "Did a small gain survive repeated evaluation?",
            ],
        ),
    ),
    Lesson(
        slug="classification-metrics",
        title="Metrics for Classification",
        summary="Accuracy is useful, but class balance and ranking quality matter too.",
        action_label="Open Training",
        action_page="Training",
        html=_lesson_html(
            title="Metrics for Classification",
            overview=(
                "Classification metrics answer different questions. Accuracy counts overall correctness, precision asks how "
                "trustworthy positive predictions are, recall asks how many positives were found, and ROC-style metrics measure ranking quality."
            ),
            example=(
                "If only 5% of examples are positive, a classifier can get 95% accuracy by predicting the majority class all the time. "
                "That score looks strong while the model is actually useless for finding positives."
            ),
            why_it_matters=[
                "Different applications care about different error types.",
                "Class imbalance can make one metric look good while another reveals failure.",
                "Metrics and plots should be interpreted together, not in isolation.",
            ],
            app_connection=[
                "Use confusion matrix, ROC, and precision-recall views after training.",
                "Compare saved runs on the metric that matches the real teaching goal, not automatically on accuracy.",
                "Interactive classification demos help you connect decision-boundary shape to the final metrics.",
            ],
            mistakes=[
                "Reporting only accuracy on imbalanced data.",
                "Using ROC AUC as the only story when precision at useful thresholds matters more.",
                "Forgetting that threshold choice changes precision and recall.",
            ],
            questions=[
                "Which class mistakes are most costly here?",
                "Is the dataset balanced enough that accuracy is informative?",
                "Does the plot show class confusion that one summary number hides?",
            ],
        ),
    ),
    Lesson(
        slug="regression-metrics",
        title="Metrics for Regression",
        summary="Regression metrics measure error size and explained variance.",
        action_label="Open Training",
        action_page="Training",
        html=_lesson_html(
            title="Metrics for Regression",
            overview=(
                "Regression quality is usually described with multiple metrics because different metrics react differently to "
                "large errors, average errors, and variance explained."
            ),
            example=(
                "RMSE punishes a few large misses more than MAE does. Two models can have similar MAE but very different RMSE if one of them "
                "sometimes makes very large mistakes."
            ),
            why_it_matters=[
                "No single regression metric captures every practical concern.",
                "Residual structure can reveal problems that summary metrics hide.",
                "A good R² does not automatically mean the model is reliable on all parts of the range.",
            ],
            app_connection=[
                "Use prediction-vs-true and residual plots together with RMSE, MAE, and R².",
                "Compare models with both performance and training cost in the comparison views.",
                "Interactive regression demos help you see how a fitted curve can look too smooth or too jagged even before the final metric tells you why.",
            ],
            mistakes=[
                "Treating R² as the only metric worth reading.",
                "Ignoring whether errors grow systematically for larger targets.",
                "Calling a model good because the average error is low while a few large misses remain unacceptable.",
            ],
            questions=[
                "Are the errors small on average and also well behaved across the range?",
                "Do a few extreme mistakes dominate the metric?",
                "What does the residual plot say that the summary score does not?",
            ],
        ),
    ),
    Lesson(
        slug="interpretability",
        title="Feature Importance and Interpretability",
        summary="Interpretability tools help learners connect model output back to features.",
        action_label="Open Visualizations",
        action_page="Visualizations",
        html=_lesson_html(
            title="Feature Importance and Interpretability",
            overview=(
                "Interpretability tools help you understand what the model relied on. They do not automatically explain why the "
                "real-world target behaves that way, and they do not prove causality."
            ),
            example=(
                "A random forest may rank income as highly important for predicting spending, but that does not mean income causes every spending pattern. "
                "It only means the model found that feature useful for prediction."
            ),
            why_it_matters=[
                "It connects abstract model behavior back to concrete inputs.",
                "It helps detect suspicious shortcuts or over-reliance on a small set of features.",
                "It teaches that prediction and explanation are related but not identical goals.",
            ],
            app_connection=[
                "Use coefficient plots for linear models and feature importance plots for tree-based models.",
                "Compare two models that perform similarly but rely on different features.",
                "Read explanations together with preprocessing choices, because scaling and encoding affect interpretability.",
            ],
            mistakes=[
                "Treating feature importance as proof of causation.",
                "Comparing raw coefficients from unscaled features as if they were directly comparable.",
                "Ignoring correlation, which can spread importance across several related features.",
            ],
            questions=[
                "Which features drive the prediction and does that make sense?",
                "Could preprocessing or correlation be distorting the interpretation?",
                "Would a different model family tell a similar interpretability story?",
            ],
        ),
    ),
    Lesson(
        slug="neural-networks",
        title="Basic Idea of Neural Networks",
        summary="Neural networks stack layers of learned transformations.",
        action_label="Open Models",
        action_page="Models",
        html=_lesson_html(
            title="Basic Idea of Neural Networks",
            overview=(
                "A neural network learns several layers of transformations. Earlier layers build intermediate representations, "
                "and later layers use those representations to make the final prediction."
            ),
            example=(
                "An MLP on tabular data can combine several input features into hidden features that are more useful for predicting the target than any one raw input alone."
            ),
            why_it_matters=[
                "It introduces a model family whose flexibility comes from architecture and optimization choices.",
                "It shows why hyperparameters such as hidden layer size, learning rate, and regularization matter together.",
                "It provides a contrast with simpler classical models that are easier to interpret but less flexible.",
            ],
            app_connection=[
                "Use the Models page to compare MLP models with linear models and trees.",
                "Use Hyperparameters and Hyperparameter Lab to see how hidden size and alpha affect performance.",
                "Interactive comparison mode is useful for contrasting an MLP with a simpler baseline on the same synthetic problem.",
            ],
            mistakes=[
                "Assuming a bigger network is automatically better.",
                "Ignoring scaling and then blaming the model for unstable training.",
                "Judging the network before checking whether max_iter or learning rate are reasonable.",
            ],
            questions=[
                "Is extra flexibility really needed for this problem?",
                "Did the network improve test performance or only training performance?",
                "Would a simpler model be easier to explain with similar accuracy?",
            ],
        ),
    ),
    Lesson(
        slug="reading-curves",
        title="Reading Learning and Validation Curves",
        summary="Curves often teach more than a single best score.",
        action_label="Open Visualizations",
        action_page="Visualizations",
        html=_lesson_html(
            title="Reading Learning and Validation Curves",
            overview=(
                "Curves show how performance changes with sample size or hyperparameter value. They are often more educational "
                "than one final score because they reveal direction, stability, and tradeoffs."
            ),
            example=(
                "A learning curve with a large train-test gap suggests variance. A validation curve where test performance peaks in the middle "
                "suggests that both too little and too much flexibility hurt generalization."
            ),
            why_it_matters=[
                "It reveals whether more data is likely to help.",
                "It makes bias-variance behavior visible.",
                "It prevents overreacting to a single lucky setting.",
            ],
            app_connection=[
                "Use learning curves in Training and Visualizations after fitting a model.",
                "Use validation curves and heatmaps in the Hyperparameter Lab.",
                "Compare the shape of curves across different model families, not only their peak values.",
            ],
            mistakes=[
                "Reading only the best point and ignoring the rest of the curve.",
                "Assuming a noisy best value is more meaningful than a broad stable region.",
                "Ignoring whether the train-test gap widens as flexibility increases.",
            ],
            questions=[
                "Where does test performance stop improving?",
                "Is there a stable region or only a fragile best point?",
                "Does this curve suggest underfitting, overfitting, or simple lack of data?",
            ],
        ),
    ),
    Lesson(
        slug="model-selection",
        title="Hyperparameter Tuning and Model Selection",
        summary="Fair comparison means controlling preprocessing, splits, and metrics.",
        action_label="Compare Runs",
        action_page="Compare Runs",
        html=_lesson_html(
            title="Hyperparameter Tuning and Model Selection",
            overview=(
                "Model selection is not just choosing the highest number. Fair selection means controlling preprocessing, "
                "feature choices, data splits, metric choice, and random seed so that differences are interpretable."
            ),
            example=(
                "If one model is trained with scaling and another is not, the score difference may partly reflect preprocessing fairness rather than true model superiority."
            ),
            why_it_matters=[
                "It turns experiments into meaningful comparisons instead of memory-based guesses.",
                "It helps explain why a model performed better, not only whether it did.",
                "It teaches learners to reason about cost, stability, and interpretability alongside raw score.",
            ],
            app_connection=[
                "Save runs and compare them side by side in Compare Runs.",
                "Use Interactive Example compare mode to separate model-family differences from hyperparameter differences.",
                "When using the Hyperparameter Lab, keep the scenario fixed so changes can be attributed to the studied settings.",
            ],
            mistakes=[
                "Changing model, preprocessing, and split all at once and then drawing a strong conclusion.",
                "Comparing scores that were produced with different metrics.",
                "Ignoring training time, inference time, and interpretability when the raw metric gain is small.",
            ],
            questions=[
                "Was this comparison fair?",
                "What exactly changed between the two runs?",
                "Is the winning model better because of fit quality, better preprocessing, or lower variance?",
            ],
        ),
    ),
]

LESSON_BY_SLUG = {lesson.slug: lesson for lesson in LESSONS}
