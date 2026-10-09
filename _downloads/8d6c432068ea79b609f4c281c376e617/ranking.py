"""Prepared data: x[N,J,F], ranking[N,K]; top-k from the full menu."""

import torch
from torch import nn
from torchdcm import ObservationBatch, LikelihoodModel
from torchdcm.events import ranking_log_prob, event_log_probs


class Ranking(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(n_features, dtype=torch.float64))

    def forward(self, data):
        return ranking_log_prob(
            data["x"] @ self.beta, data["ranking"], data.payload.get("available")
        )

    def predict_proba(self, data):
        return event_log_probs(
            data["x"] @ self.beta, data.payload.get("available")
        ).exp()

    predict = predict_proba


def fit(data):
    result = LikelihoodModel(Ranking(data["x"].shape[-1])).fit(data)
    return result, result.predict_proba()


if __name__ == "__main__":
    torch.manual_seed(21)
    x = torch.randn(400, 4, 2, dtype=torch.float64)
    utility = x @ torch.tensor([0.7, -0.4], dtype=torch.float64)
    noise = -torch.log(-torch.log(torch.rand_like(utility)))
    ranking = (utility + noise).argsort(-1, descending=True)[:, :2]
    result, prediction = fit(
        ObservationBatch({"x": x, "ranking": ranking}, len(x), "ranking")
    )
    print(result.summary())
    print(
        "first-position probability sums:",
        prediction.sum(-1).min(),
        prediction.sum(-1).max(),
    )
