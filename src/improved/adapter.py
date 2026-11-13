"""Adapter contract for P6: real encoder predictor plus the frozen mock substitute."""
import json
from pathlib import Path

from src.common.model_api import MockPredictor, Predictor


class EncoderPredictor:
    """Loads a model saved by `python -m src.improved train` fully offline.

    is_mock stays False; predict_proba returns label=1 scores in [0,1] preserving order.
    """

    is_mock = False

    def __init__(self, model_dir):
        path = Path(model_dir)
        meta_path = path / "model_metadata.json"
        if not path.is_dir() or not meta_path.is_file():
            raise FileNotFoundError(f"{model_dir}: no trained improved model here (model_metadata.json missing); run the train CLI first")
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        config = self.metadata["config"]
        self.max_length = config["max_length"]
        self.batch_size = config.get("predict_batch_size", 64)

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True).to(self.device).eval()

    def predict_proba(self, texts: list) -> list:
        from .engine import score_texts

        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every text must be a non-empty string")
        # 推理不扰动权重，也不做概率校准：分数是 softmax 的 label=1 分量。
        return score_texts(self.model, self.tokenizer, texts, max_length=self.max_length,
                           device=self.device, batch_size=self.batch_size, amp_enabled=False)


def load_predictor(model_dir: str, *, mock: bool = False) -> Predictor:
    if mock:
        return MockPredictor()
    return EncoderPredictor(model_dir)
