---
name: torchdcm-extension
description: Implement and validate TorchDCM mechanisms with minimal application code, reusing observation, latent-state, equilibrium, and estimation APIs. Use for utilities, probability/density laws, sequences, allocations, posterior targets, or moment extensions in this repository.
---

# Extend TorchDCM

Read [the extension guide](../../../docs/extension.md) and the closest linked
example. Paths below are relative to the software repository root.
Use the project's Python environment (`.venv/bin/python` when present).

## Decide what changes

- Existing law and linear utility: use `UtilitySpec` and the existing model.
- Other observed responses: read [general mechanisms](../../../docs/general.md).
  Use `ObservationBatch` + `LikelihoodModel`; return one log contribution per
  declared independent unit. Do not fabricate a chosen alternative or edit `_fit.py`.
- Ranking, hidden states, allocation, attention, continuous outcomes or equilibria:
  compose the public operators in that guide before adding a new operator.
- Composite likelihood: declare `objective_kind="composite"`; use sandwich
  inference. Posterior and moment-inequality targets have separate result types.
- Simple custom function: `ModuleChoiceModel.from_callable(function, parameters)`;
  the hook takes `(parameters, data)`. Avoid an unnecessary per-application model class.
- Trainable networks, buffers, parametrizations: supply an `nn.Module` to `ModuleChoiceModel`.
- Utility-only module: `ModuleChoiceModel.from_utility`; it adds the Logit law.
  Export fitted weights with `export_utility(result.params)`.
- Dynamic choice, BLP or path models: read [structural interfaces](../../../docs/structural.md).
  Supply payoff/transition/share modules; reuse the public estimator and solver interfaces.
- Moment estimation: `GMMModel` or `GMMModel.from_callable`; return row-level moments.
  Do not reuse likelihood Hessian covariance, AIC or pseudo-R2 for GMM.
- History or inner-solve mechanisms: read [advanced interfaces](../../../docs/advanced.md).
  Reuse `HistoryUtility` or `implicit_root`/`implicit_minimize` before writing a
  per-application recurrence, solver derivative, or estimator.
- Weighted mixtures, multiple latent axes, or choice + measurement density:
  use the [composition API](../../../docs/composition.md); return its objective
  from the module and retain the original choice branch for prediction.
- New factorization: combine `LogFactor`, `join`, `sum_by`, `integrate` and the
  sequence operator. Preserve the statistical unit; propose core changes only
  when these operations demonstrably cannot express the required mechanism.

Keep new task code in an application module. Edit core code only for a demonstrated
missing operation, with a regression test. Preserve existing model-specific recovery
and initialization policies.

## Kernel invariants

- `ModuleChoiceModel`: long-row conditional log probabilities; normalize per
  choice set/draw; unavailable rows `-inf`.
- `LikelihoodModel`: per-record observed log probability/density or scoped factor.
  Declare mass/density/mixed; densities need not be <=1. Frequency counts stay
  outside integration. Whole sequences/rankings are units, not individual stages.
- Register parameters and fixed draws; deterministic, mutation-free `forward`.
- Panel draws shared across occasions; constant individual frequency weights.
- Use the wrapper's aggregation, derivatives, fit, covariance, and prediction.
- Report registered parameter coordinates. Use `delta_method` for scalar transforms.
- Use `cov_type="none"` when dense Hessian inference is inappropriate; say so.
- Check prediction conditioning and economic quantities for the requested model.

## Finish with evidence

1. Independent numerical formula and derivative checks.
2. Relevant grouping, availability, and constraint tests.
3. `python -m pytest -q`; runnable example; convergence status.
4. Full task code count, including imported task helpers and core edits.
   Separate model definition/calls from synthetic generation and experiment-only export/CLI.
5. If timing is requested: matched conditions and inference work, repeated runs.

For composed models, declare latent-axis identity and observation/person scope;
join shared-latent factors before integration, count personal measurements once,
and keep frequency weights outside integration. Do not choice-normalize densities.
Reuse existing probability kernels before writing their mathematics again.
Minimize each application's added code; core-library total SLOC is informational.

For structural extensions, preserve the statistical target: BellmanUtility fits
choices conditional on observed states; transition and initial-state densities
need explicit factors for a joint likelihood. BLP share reconstruction is an
inversion check; prediction with `xi=None` means zero structural errors. A custom
share module may need a different solver; a valid tensor shape does not establish
contraction or identification. Path partitions cover absorbing walks, not simple
paths or arbitrary combinatorial constraints. The current implicit layer is dense.

Human and AI contributions use the same contract: keep new mechanism code in its
own module, register parameters, document input/output shape and conditioning,
and add one independent numerical check. Avoid per-application parameter flattening,
optimizer loops or copied covariance assembly. New shared abstractions should
replace demonstrated duplication across actual models.

History heads use pre-choice state and exogenous features; transition inputs are
occasion-local. One-step forecasts must not read query choices. Inner layers need
an explicit smooth residual/objective and regular local root; do not claim that
autograd differentiates an arbitrary opaque or discrete solver. Check second
derivatives independently before reporting Hessian-based inference.

Prefer fewer required components and readable code. Do not replace a tested fast
path solely to make all built-in models inherit one class.

For HMM compositions, integrate permanent taste outside the full sequence and add
personal measurements once. Keep forecast conditioning explicit. RI support changes,
MDCEV active-set switches and multiple game/pricing equilibria are not covered by a
generic smooth derivative claim. For posterior targets, include priors and transform
Jacobians and inspect sampling diagnostics. For moment regions, declare independent
rows and known bounds; do not generalize fixed-grid coverage to adaptive grids.
