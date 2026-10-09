"""Ranking emissions plus hidden states; one independent record per sequence."""

import torch
from torch import nn
from torchdcm import ObservationBatch, LikelihoodModel
from torchdcm.events import ranking_log_prob
from torchdcm.markov import markov_eliminate


class RankedHMM(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
        self.switch = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float64))
        self.register_buffer(
            "state_scale", torch.tensor([0.3, 1.7], dtype=torch.float64)
        )

    def forward(self, data):
        return self.predict(data).log_likelihood

    def predict(self, data, smooth=False):
        utility = self.beta * data["x"][..., None, :] * self.state_scale[:, None]
        emission = ranking_log_prob(utility, data["ranking"][..., None, :])
        transition = torch.diag(self.switch)
        return markov_eliminate(
            emission,
            self.switch.new_zeros(2),
            transition,
            lengths=data["lengths"],
            smooth=smooth,
        )


def fit(data):
    result = LikelihoodModel(RankedHMM()).fit(data)
    history = result.predict(smooth=True)
    return result, history.forecast(torch.diag(result.export_module().switch))


if __name__ == "__main__":
    torch.manual_seed(22)
    n, t, j = 250, 5, 3
    x = torch.randn(n, t, j, dtype=torch.float64)
    transition = torch.diag(torch.tensor([1.0, 0.7], dtype=torch.float64)).softmax(-1)
    state = torch.zeros(n, t, dtype=torch.long)
    state[:, 0] = torch.randint(2, (n,))
    for time in range(1, t):
        state[:, time] = torch.multinomial(transition[state[:, time - 1]], 1).squeeze(
            -1
        )
    utility = 0.8 * x * torch.tensor([0.3, 1.7], dtype=torch.float64)[state][..., None]
    noise = -torch.log(-torch.log(torch.rand_like(utility)))
    ranking = (utility + noise).argsort(-1, descending=True)[..., :2]
    data = ObservationBatch(
        {"x": x, "ranking": ranking, "lengths": torch.full((n,), t)}, n, "sequence"
    )
    result, forecast = fit(data)
    print(result.summary())
    print(
        "next-state probability sums:",
        forecast.sum(-1).min().item(),
        forecast.sum(-1).max().item(),
    )
