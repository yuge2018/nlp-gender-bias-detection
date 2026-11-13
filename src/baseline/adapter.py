"""P2 adapter used by the demo (P6).

``load_predictor`` never trains: a saved run directory loads its vectorizer and
classifier and produces label=1 scores in input order. IDs, labels and target
groups are not accepted by this interface.
"""
from src.common.model_api import MockPredictor, Predictor

from .model import predict_proba_batched


class BaselinePredictor:
    """Offline predictor for one saved P2 baseline model."""

    def __init__(self, model, metadata):
        self._model = model
        self.metadata = metadata
        self.is_mock = bool(metadata.get("is_mock", False))
        self.threshold = float(metadata["threshold"])
        self.model_name = str(metadata.get("model_name", "char_tfidf_logreg"))
        self.batch_size = int(metadata.get("config", {}).get("batch_size", 256))

    def predict_proba(self, texts):
        if not isinstance(texts, (list, tuple)):
            raise ValueError("texts must be a list of strings")
        if not texts:
            return []
        return predict_proba_batched(self._model, list(texts), self.batch_size)


def load_predictor(model_dir: str, *, mock: bool = False) -> Predictor:
    """Load a saved baseline offline; ``mock=True`` keeps the constant substitute."""
    if mock:
        return MockPredictor()
    from .model import load_model

    model, metadata = load_model(model_dir)
    return BaselinePredictor(model, metadata)
