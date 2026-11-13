from src.common.model_api import MockPredictor, Predictor


def load_predictor(model_dir: str, *, mock: bool = False) -> Predictor:
    if mock:
        return MockPredictor()
    raise NotImplementedError("Real model loading is assigned in docs/roles/")
