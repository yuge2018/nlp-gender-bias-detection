"""Character TF-IDF + logistic-regression bundle for the P2 baseline.

Only ``text`` is ever passed to the vectorizer. The vectorizer is fitted on the
training split alone, and the trained model is saved so that prediction can run
offline without retraining.
"""
import importlib
import json
import platform
import subprocess
from pathlib import Path

from src.common.contracts import VERSION, write_json

from .config import METHOD_LOGREG, METHOD_MAJORITY

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "model_metadata.json"


def _sklearn_components():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - exercised only without scikit-learn
        raise ImportError("the real P2 baseline needs scikit-learn; install requirements/p2.txt") from exc
    return TfidfVectorizer, LogisticRegression


def _joblib():
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("saving/loading the P2 baseline needs joblib; install requirements/p2.txt") from exc
    return joblib


def _validate_texts(texts):
    if not isinstance(texts, (list, tuple)):
        raise ValueError("texts must be a list of strings")
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"texts[{index}]: every text must be a non-empty string")
    return list(texts)


def _validate_labels(labels, expected):
    if not isinstance(labels, (list, tuple)) or len(labels) != expected:
        raise ValueError("labels must align with texts")
    for index, label in enumerate(labels):
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"labels[{index}]: label must be integer 0/1")
    return list(labels)


class BaselineModel:
    """Either a fitted vectorizer+classifier or the majority-class reference (B0)."""

    def __init__(self, config, vectorizer=None, classifier=None, majority_label=None):
        self.config = config
        self.method = config["method"]
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.majority_label = majority_label

    @classmethod
    def fit(cls, config, texts, labels, seed):
        texts = _validate_texts(texts)
        labels = _validate_labels(labels, len(texts))
        if not texts:
            raise ValueError("cannot train on an empty dataset")
        if config["method"] == METHOD_MAJORITY:
            positives = sum(labels)
            majority_label = 1 if positives * 2 > len(labels) else 0
            return cls(config, majority_label=majority_label)
        if config["method"] != METHOD_LOGREG:
            raise ValueError(f"unsupported method {config['method']}")
        TfidfVectorizer, LogisticRegression = _sklearn_components()
        vectorizer = TfidfVectorizer(analyzer=config["analyzer"], ngram_range=tuple(config["ngram_range"]), max_features=config["max_features"], lowercase=False)
        matrix = vectorizer.fit_transform(texts)
        classifier = LogisticRegression(C=config["C"], class_weight=config["class_weight"], max_iter=config["max_iter"], solver=config["solver"], random_state=seed)
        classifier.fit(matrix, labels)
        return cls(config, vectorizer=vectorizer, classifier=classifier, majority_label=None)

    @property
    def is_majority(self):
        return self.vectorizer is None

    def vocabulary_size(self):
        return 0 if self.vectorizer is None else len(self.vectorizer.vocabulary_)

    def predict_proba(self, texts):
        texts = _validate_texts(texts)
        if self.vectorizer is None:
            score = 1.0 if self.majority_label == 1 else 0.0
            return [score] * len(texts)
        matrix = self.vectorizer.transform(texts)
        scores = self.classifier.predict_proba(matrix)[:, 1]
        return [float(score) for score in scores]

    def top_features(self, limit=20):
        if self.vectorizer is None:
            return {"positive": [], "negative": [], "note": "majority-class reference has no lexical weights"}
        names = self.vectorizer.get_feature_names_out()
        weights = self.classifier.coef_[0]
        order = sorted(range(len(names)), key=lambda index: weights[index])
        negative = [{"feature": names[index], "weight": round(float(weights[index]), 6)} for index in order[:limit]]
        positive = [{"feature": names[index], "weight": round(float(weights[index]), 6)} for index in order[-limit:][::-1]]
        return {"positive": positive, "negative": negative}


def predict_proba_batched(model, texts, batch_size):
    """Predict in input order; batch boundaries must not change a row's score."""
    texts = _validate_texts(texts)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not texts:
        return []
    scores = []
    for start in range(0, len(texts), batch_size):
        scores.extend(model.predict_proba(texts[start:start + batch_size]))
    if len(scores) != len(texts):
        raise RuntimeError("batched prediction did not preserve the number of inputs")
    return scores


def environment():
    """Recorded versions for run metadata; unavailable optional modules become null."""
    import numpy

    versions = {"python": platform.python_version(), "implementation": platform.python_implementation(), "platform": platform.platform(), "machine": platform.machine(), "numpy": numpy.__version__}
    for name in ("scipy", "sklearn", "joblib"):
        try:
            module = importlib.import_module(name)
        except ImportError:  # pragma: no cover
            versions[name] = None
        else:
            versions[name] = getattr(module, "__version__", "unknown")
    return versions


def repository_state(root):
    """Record the code version used for a run when the checkout is a Git repository."""
    root = str(root)
    try:
        commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", root, "status", "--porcelain"], capture_output=True, text=True, timeout=10, check=True).stdout.strip())
        return {"code_commit": commit, "code_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"code_commit": "UNKNOWN", "code_dirty": None}


def save_model(model_dir, model, metadata):
    base = Path(model_dir)
    base.mkdir(parents=True, exist_ok=True)
    _joblib().dump(model, base / MODEL_FILENAME)
    metadata = dict(metadata)
    metadata["schema_version"] = VERSION
    write_json(base / METADATA_FILENAME, metadata)
    return base


def load_model(model_dir):
    base = Path(model_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"{base}: model directory not found")
    model_path = base / MODEL_FILENAME
    metadata_path = base / METADATA_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(f"{model_path}: saved model not found")
    if not metadata_path.is_file():
        raise ValueError(f"{metadata_path}: model metadata not found")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{metadata_path}: invalid JSON ({exc})") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != VERSION:
        raise ValueError(f"{metadata_path}: unsupported model metadata")
    if metadata.get("method") not in (METHOD_LOGREG, METHOD_MAJORITY):
        raise ValueError(f"{metadata_path}: unknown baseline method")
    _sklearn_components()
    model = _joblib().load(model_path)
    if not isinstance(model, BaselineModel):
        raise ValueError(f"{model_path}: unexpected saved object type")
    return model, metadata


