import torch

from torchdcm import Beta, ChoiceDataset, MultinomialLogit, UtilitySpec
from torchdcm.datasets import make_swissmetro_like

df = make_swissmetro_like(n_obs=300, seed=7)
data = ChoiceDataset.from_wide(
    df,
    alternatives=["TRAIN", "SM", "CAR"],
    choice="choice",
    variables={
        "time": {"TRAIN": "time_train", "SM": "time_sm", "CAR": "time_car"},
        "cost": {"TRAIN": "cost_train", "SM": "cost_sm", "CAR": "cost_car"},
    },
    availability={
        "TRAIN": "avail_train",
        "SM": "avail_sm",
        "CAR": "avail_car",
    },
    individual_id="person_id",
)

spec = UtilitySpec()
spec.utility(
    "TRAIN",
    Beta("ASC_TRAIN")
    + Beta("B_TIME", init=-0.01) * "time"
    + Beta("B_COST", init=-0.1) * "cost",
)
spec.utility(
    "SM",
    Beta("B_TIME", init=-0.01) * "time"
    + Beta("B_COST", init=-0.1) * "cost",
)
spec.utility(
    "CAR",
    Beta("ASC_CAR")
    + Beta("B_TIME", init=-0.01) * "time"
    + Beta("B_COST", init=-0.1) * "cost",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
result = MultinomialLogit(spec, device=device).fit(data, cov_type="cluster", groups="person_id")
print(result.summary())
