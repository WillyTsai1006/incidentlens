"""Synthetic source for inspection only; not an executable serving application."""

MODEL_NAME = "demo-4b-model"


def worker_main(worker_id):
    model = load_model(MODEL_NAME)
    serve(model)
