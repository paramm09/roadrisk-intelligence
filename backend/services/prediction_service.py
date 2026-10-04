"""Thin wrapper around ml.predict.Predictor -> HTTP-shaped results.

Every number in the response comes from a REAL trained artifact; nothing is
re-trained at request time (load artifact -> preprocess -> predict -> return).
Invalid input raises ValueError (mapped to HTTP 422 by main.py); a request for
a model that does not exist raises ModelNotAvailableError (HTTP 400).
"""
from __future__ import annotations

from ..schemas import BatchPredictionRequest, PredictionRequest
from .model_service import ModelService


class PredictionService:
    def __init__(self, model_service: ModelService):
        self.model = model_service

    def predict(self, request: PredictionRequest) -> dict:
        record = request.model_dump(exclude_none=True)
        model_id = record.pop("model", None)          # selection, not a model feature
        artifact, name = self.model.resolve_model(model_id)
        return self.model.predictor.predict(record, artifact=artifact, model_label=name)

    def predict_batch(self, request: BatchPredictionRequest) -> dict:
        # Resolve every requested model FIRST so an invalid id fails the whole batch (400)
        # before any prediction is produced.
        groups: dict[tuple[str | None, str], list[tuple[int, dict]]] = {}
        for i, r in enumerate(request.records):
            record = r.model_dump(exclude_none=True)
            model_id = record.pop("model", None)
            key = self.model.resolve_model(model_id)
            groups.setdefault(key, []).append((i, record))

        results: dict[int, dict] = {}
        for (artifact, name), items in groups.items():
            outs = self.model.predictor.predict_batch(
                [rec for _, rec in items], artifact=artifact, model_label=name)
            for (i, _), out in zip(items, outs):
                results[i] = out
        ordered = [results[i] for i in range(len(request.records))]
        return {"count": len(ordered), "predictions": ordered}
