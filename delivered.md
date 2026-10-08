# ImitationForge — Delivery Notes

**Author:** 晨星 (CJX0712) · **Version:** 0.1.0 · **License:** MIT
**Repository:** https://github.com/CJX0712/imitationforge

## What was delivered

A complete, tested, open-source Python system for **imitation learning** (behavior
cloning → DAGGER → ensemble + uncertainty-gated expert query), built around a
single flagship method **ImiteFuse**.

| Layer | Module | Status |
|-------|--------|--------|
| Deterministic core | `core/{seed,config,errors,types}.py` | ✅ global seed entry + per-env offset |
| Synthetic control envs + optimal experts | `envs/{gridworld,pendulum,cartpole}.py`, `envs/common.py` | ✅ 3 envs, expert rate ≥ 0.9 |
| Policy learners (Tier-0 sklearn / Tier-1 numpy fallback) | `policy/learners.py` | ✅ SklearnPolicy/NumpyPolicy/Ensemble/NN |
| Imitation methods (7 registered) | `methods/{bc,nn,dagger,imitafuse}.py` + `registry.py` | ✅ BC/DAGGER/ImiteFuse + ablations |
| Benchmark engine | `pipeline/pipeline.py` + `cli.py` | ✅ run_benchmark → benchmark.json |
| Demo + determinism check | `examples/run_demo.py` | ✅ ≤60s, bit_identical |
| Docs | `README.md`, `docs/architecture.md`, `docs/model_card.md` | ✅ |
| Packaging / CI / Docker | `pyproject`-style `requirements.lock.txt`, `Makefile`, `.github/workflows/ci.yml`, `Dockerfile` | ✅ |
| Secret preflight + 3-tier push | `preflight.py`, `gh_push.py` | ✅ |

## Flagship method — ImiteFuse

1. **DAGGER dataset aggregation** (Ross et al. 2011): roll out the current
   policy, collect states from its *true visited distribution*, and have the
   expert label them → repairs distribution shift.
2. **K-policy ensemble**: bootstrap-fit `K` members; `predict = mean`,
   `uncertainty = variance`.
3. **Uncertainty-gated expert query**: query the expert for the `K` *most
   uncertain* states within the query budget → maximal information gain.

## Benchmark results (default config, seed=7)

Metric: success rate (higher = better). ImiteFuse ≥ BC in **all 3 envs**,
mean Δ ≈ **+0.139** (threshold 0.10).

| env | BC | nn | DAGGER | **ImiteFuse** | Δ(flagship−BC) |
|-----|---:|---:|---:|---:|---:|
| gridworld | 1.000 | 0.667 | 1.000 | **1.000** | +0.000 |
| pendulum (sparse demos, hard) | 0.583 | 0.417 | 0.667 | **1.000** | **+0.417** ✅ |
| cartpole | 1.000 | 0.000 | 1.000 | **1.000** | +0.000 |

Key insight: a smooth, well-covered expert is always cloned perfectly by a strong
MLP (BC=1.00); BC only fails on **hard tasks** (sparse demos / unstable dynamics
exposed). Pendulum's sparse demos are exactly that sweet spot, where ImiteFuse
wins big.

## Quality gates (all green)

- `python -m ruff check .` → **All checks passed!** (ruff 0.16.10)
- `python -m pytest -q` → **17 passed**
- Coverage → **88%** (threshold 80%)
- Determinism → `bit_identical=True` (two runs, byte-identical core metrics)
- Secret preflight → **passed** (no hardcoded credentials)
- CI: lint + pytest + cov + demo smoke, matrix py3.11–3.13

## How to reproduce

```bash
pip install -r requirements.lock.txt
make lint        # ruff check + format --check
make test        # pytest (17)
make demo        # end-to-end → benchmark.json (+ determinism re-check)
```
