"""Demand-only random coefficients, endogenous prices, market-clustered GMM."""

import numpy as np
import torch
from torchdcm import BLPData, BLPDemand, GMMModel


def run(data):
    moments = BLPDemand([-0.8, 0.4, -0.9], [np.log(0.3)])
    return GMMModel(moments, max_iter=500).fit(data, steps=2, groups=data.market)


def make_data(n_markets=40, products=3):
    rng = np.random.default_rng(42)
    n = n_markets * products
    quality = rng.normal(size=n)
    cost = rng.normal(size=n)
    z = np.column_stack([np.ones(n), quality, cost, cost**2 - 1, quality * cost])
    xi = rng.normal(size=n) * 0.15
    xi = xi - z @ np.linalg.lstsq(z, xi, rcond=None)[0]
    price = 2 + 0.3 * quality + 0.6 * cost + 0.5 * xi
    x = np.column_stack([np.ones(n), quality, price])
    beta = np.array([-1.0, 0.6, -1.2])
    nodes, weights = np.polynomial.hermite.hermgauss(11)
    draws = np.sqrt(2) * nodes
    weights = weights / np.sqrt(np.pi)
    shares = []
    for market in range(n_markets):
        rows = slice(market * products, (market + 1) * products)
        utility = (x[rows] @ beta + xi[rows])[:, None] + price[rows, None] * 0.4 * draws
        exponent = np.exp(utility)
        shares.extend((exponent / (1 + exponent.sum(0))) @ weights)

    def tensor(value):
        return torch.tensor(np.asarray(value), dtype=torch.float64)

    return BLPData(
        tensor(x),
        tensor(price[:, None]),
        tensor(shares),
        tensor(z),
        torch.arange(n_markets).repeat_interleave(products),
        tensor(draws[:, None]),
        tensor(weights),
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    data = make_data()
    result = run(data)
    print(result.summary())
    print("Step status:", [(s["success"], s["gradient_norm"]) for s in result.steps])
    fitted = result.export_module()
    xi = fitted.mean_utilities(data) - data.x @ fitted.beta
    print(
        "Share inversion error:",
        float((fitted.predict(data, xi) - data.shares).abs().max().detach()),
    )
    print(
        "Random coefficient scale:",
        fitted.share_model.log_sigma.exp().detach().tolist(),
    )
