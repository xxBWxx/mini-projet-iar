import json
import warnings
import numpy as np
from pathlib import Path
from glob import glob


def load_evals(logdir: Path):
    """Load EvalCallback outputs; return (timesteps, results) or (None, None) if missing/empty."""
    fns = glob(str(logdir / "*_eval" / "evaluations.npz"))
    if not fns:
        return None, None
    data = np.load(fns[0])
    ts = data.get("timesteps")
    res = data.get("results")
    if ts is None or res is None or len(ts) == 0 or res.size == 0:
        return None, None
    return np.asarray(ts).squeeze(), np.asarray(res)


def ci95(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return np.nan, np.nan
    if x.size == 1:
        return float(x[0]), 0.0
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return m, 1.96 * se


def auc(ts, ys):
    """Per-seed AUC of the learning curve (mean over episodes already applied)."""
    ts = np.asarray(ts, float)
    ys = np.asarray(ys, float)
    if ts.size < 2 or ys.size < 2:
        return 0.0
    return float(np.trapz(ys, ts))


def aggregate_learning_curves(root: Path):
    """
    Returns dict algo -> (aligned_timesteps, matrix_of_seed_means)
    - Skips seeds with no evals
    - Aligns on common timesteps if possible, otherwise truncates to min length
    - Warns if nothing found
    """
    alg2curves = {}
    for alg in ["DDPG", "TD3"]:
        seeds_dirs = sorted(root.glob(f"{alg}_seed*"))
        ts_list, means_list, missing = [], [], []
        for d in seeds_dirs:
            ts, res = load_evals(d)
            if ts is None:
                missing.append(d.name)
                continue
            # mean over episodes at each eval point
            means = res.mean(axis=1)
            ts_list.append(np.asarray(ts))
            means_list.append(np.asarray(means))

        if not ts_list:
            warnings.warn(
                f"No evaluations found for {alg} under {root}. "
                f"Missing/empty: {missing}"
            )
            continue

        # try to align by intersection of timesteps
        common = set(ts_list[0])
        for ts in ts_list[1:]:
            common &= set(ts)
        if common:
            common = np.array(sorted(common))
            aligned = []
            for ts, means in zip(ts_list, means_list):
                mask = np.in1d(ts, common)
                aligned.append(means[mask])
            mat = np.vstack(aligned)
            alg2curves[alg] = (common, mat)
        else:
            # fallback: truncate to min length preserving order
            L = min(len(ts) for ts in ts_list)
            # choose ts from the seed with the smallest length
            ts_ref = ts_list[np.argmin([len(ts) for ts in ts_list])][:L]
            mat = np.vstack([m[:L] for m in means_list])
            alg2curves[alg] = (ts_ref, mat)

        if missing:
            warnings.warn(
                f"Skipped {len(missing)} {alg} seeds without evals: {missing}"
            )

    if not alg2curves:
        raise RuntimeError(
            f"No evaluation data found under {root}. "
            "Ensure EvalCallback ran (eval_freq ≤ TIMESTEPS) and log_path matches your loader."
        )
    return alg2curves


def summarize_bias(root: Path):
    ddpg_bias, td3_bias = [], []
    for p in root.glob("*_seed*/result.json"):
        r = json.loads(Path(p).read_text())
        if r["algo"] == "DDPG":
            # single critic -> 'bias'
            ddpg_bias.append(r["bias_metrics"]["bias"])
        else:
            # TD3 has bias_q1, bias_q2, bias_qmin; report min
            td3_bias.append(r["bias_metrics"]["bias_qmin"])
    return np.array(ddpg_bias), np.array(td3_bias)
