"""Optional integrations using public SHAP APIs, also usable with ensembles."""
import numpy as np


class ShapExplainer:
    """Callable SHAP adapter with stable class names on returned Explanations.

    ``explainer`` exposes the underlying public shap.Explainer instance.
    """

    def __init__(self, explainer, output_names):
        self.explainer = explainer
        self.output_names = output_names

    def __call__(self, x, **kwargs):
        if not hasattr(x, 'columns'):
            x = np.asarray(x)
        values = self.explainer(x, **kwargs)
        if self.output_names is not None:
            values.output_names = self.output_names
        return values


def shap_explainer(model, background, *, algorithm='auto', feature_names=None,
                   seed=None):
    """Explain predictions (regression) or all class probabilities (classification).

    Returns a ShapExplainer, callable on a batch to produce shap.Explanation.
    Uses SHAP's model-agnostic Independent masker and exact/permutation engines.
    Background is explicit and must contain representative, training-only rows.
    This also accepts sklearn ensembles and Pipelines exposing predict/proba.
    Install libtree-gbdt[explain] to use this optional integration.
    """
    try:
        import shap
    except ImportError as error:
        raise ImportError("install libtree-gbdt[explain] to use SHAP") from error
    if algorithm not in ('auto', 'exact', 'permutation'):
        raise ValueError('algorithm must be auto, exact or permutation')
    if hasattr(model, '__sklearn_is_fitted__') and not model.__sklearn_is_fitted__():
        raise RuntimeError('model is not fitted or is closed')
    if feature_names is None and hasattr(background, 'columns'):
        feature_names = [str(name) for name in background.columns]
    values = np.asarray(background, dtype=np.float64)
    if (values.ndim != 2 or not values.shape[0] or not values.shape[1]
            or np.isinf(values).any()):
        raise ValueError('background must be a nonempty finite-or-NaN feature matrix')
    if hasattr(model, 'n_features_in_') and values.shape[1] != model.n_features_in_:
        raise ValueError('background feature count does not match fitted model')
    if feature_names is not None:
        if (isinstance(feature_names, str) or len(feature_names) != values.shape[1]
                or any(not isinstance(name, str) for name in feature_names)):
            raise ValueError('feature_names must contain one string per feature')
        feature_names = list(feature_names)
    classifier = hasattr(model, 'predict_proba')
    predict = model.predict_proba if classifier else model.predict
    output_names = [str(label) for label in model.classes_] if classifier else None
    # Keep every explicitly selected background row; no hidden 100-row sampling.
    # Identity link preserves the actual probability/value scale and additivity.
    masker_data = background if hasattr(background, 'columns') else values
    masker = shap.maskers.Independent(masker_data, max_samples=len(values))
    explainer = shap.Explainer(predict, masker, algorithm=algorithm,
                               feature_names=feature_names, seed=seed)
    # SHAP Exact does not consistently forward output_names across supported
    # versions. Attach labels through the public Explanation property instead.
    return ShapExplainer(explainer, output_names)
