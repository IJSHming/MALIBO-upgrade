"""2D Airfoil 7-Method Main & Ablation Benchmark Runner (NACA 0012, Re=3.5e6, AoA=3.5 deg, D=18, NFE=300, 10 seeds).

Group 1 (Main Comparison — Baselines vs. Proposed Full Method):
  - Vanilla-TuRBO  (frozen baseline)
  - MALIBO         (frozen baseline, Paper-Exact)
  - LLM-Meta-TuRBO (frozen baseline)
  - CoLT-MALIBO    (Full upgraded MALIBO: Obj-TR + Con-TR + LLM)

Group 2 (Ablation Study — Full Method vs. Single-Component Removals):
  - CoLT-MALIBO    (Full)
  - w/o LLM        (Removes LLM offset injection & meta-control)
  - w/o Obj-TR     (Removes TuRBO objective trust-region box)
  - w/o Con-TR     (Removes SCBO Matérn-5/2 constrained feasibility surrogate)

Saves unified 7-method results to `results/ablation_2d_results.json` and `.npz`.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import multiprocessing as mp
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
import torch

from src.case_2d.airfoil_utils import get_kulfan_bounds, get_naca0012_theta
from src.case_2d.evaluator import AerodynamicEvaluator
from src.case_2d.llm_agent import LLMMALIBO2DAgent
from src.case_2d.malibo_core import load_2d_malibo_model
from src.case_2d.upgraded_malibo import UpgradedMALIBO2DOptimizer

BENCHMARK_SEEDS = [42, 123, 256, 777, 1024, 2024, 3407, 5555, 8888, 9999]
# (use_trust_region, use_llm, use_feas_surrogate)
VARIANT_SPECS = {
    "CoLT-MALIBO": (True, True, True),
    "w/o LLM": (True, False, True),
    "w/o Obj-TR": (False, True, True),
    "w/o Con-TR": (True, True, False),
}
LEGACY_KEY_MAP = {
    "LLM-TR-MALIBO": "CoLT-MALIBO",
    "TR-MALIBO": "w/o LLM",
    "LLM-MALIBO": "w/o Obj-TR",
}
METHOD_ORDER = [
    "Vanilla-TuRBO",
    "MALIBO",
    "LLM-Meta-TuRBO",
    "CoLT-MALIBO",
    "w/o LLM",
    "w/o Obj-TR",
    "w/o Con-TR",
]


def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _run_2d_variant_seed(args: Tuple[str, int, float, float, int, str, str, str, str]) -> Dict[str, Any]:
    variant, seed, reynolds, alpha, max_evals, ckpt_path, llm_url, llm_model, device_str = args
    use_tr, use_llm, use_feas = VARIANT_SPECS[variant]
    device = torch.device(device_str)
    meta_model = load_2d_malibo_model(ckpt_path, device=device)

    evaluator = AerodynamicEvaluator(reynolds=reynolds, alpha=alpha, max_evals=max_evals)
    x0 = get_naca0012_theta()
    lb, ub = get_kulfan_bounds(x0)

    llm_agent = (
        LLMMALIBO2DAgent(base_url=llm_url, model=llm_model, use_trust_region=use_tr)
        if use_llm
        else None
    )

    opt = UpgradedMALIBO2DOptimizer(
        meta_model=meta_model,
        evaluator=evaluator,
        use_trust_region=use_tr,
        use_llm=use_llm,
        use_feas_surrogate=use_feas,
        use_gamma_annealing=True,
        infeas_failure_weight=2,
        x0=x0,
        bounds=(lb, ub),
        max_evals=max_evals,
        seed=seed,
        llm_agent=llm_agent,
        device=device,
    )
    return opt.run()


def _save_2d_results(
    results: Dict[str, List[Dict[str, Any]]],
    condition: Dict[str, Any],
    output_json: str,
    output_npz: str,
):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    base_scores = np.array([r["best_score"] for r in results["MALIBO"]], dtype=np.float64)
    turbo_scores = np.array([r["best_score"] for r in results["Vanilla-TuRBO"]], dtype=np.float64)

    summary = []
    for name in METHOD_ORDER:
        rows = results.get(name, [])
        if not rows:
            continue
        sc = np.array([r["best_score"] for r in rows], dtype=np.float64)
        s15 = np.array([r["history"][14] for r in rows], dtype=np.float64)
        feas = sum(1 for r in rows if r.get("detailed", {}).get("is_feasible", True))
        times = [r["elapsed_seconds"] for r in rows if "elapsed_seconds" in r]
        summary.append({
            "method": name,
            "n_seeds": len(rows),
            "step15_mean": float(np.mean(s15)),
            "step15_std": float(np.std(s15)),
            "mean": float(np.mean(sc)),
            "std": float(np.std(sc)),
            "median": float(np.median(sc)),
            "min": float(np.min(sc)),
            "max": float(np.max(sc)),
            "delta_vs_malibo": float(np.mean(sc) - np.mean(base_scores)),
            "delta_vs_turbo": float(np.mean(sc) - np.mean(turbo_scores)),
            "feasibility_rate": feas / len(rows),
            "mean_elapsed_seconds": float(np.mean(times)) if times else None,
        })

    payload = {
        "condition": condition,
        "summary": summary,
        "results": results,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    npz_dict: Dict[str, Any] = {"seeds": np.array(condition["seeds"])}
    for name in METHOD_ORDER:
        rows = results.get(name, [])
        if not rows:
            continue
        key = name.lower().replace("/", "").replace(" ", "_").replace("-", "_")
        npz_dict[f"{key}_scores"] = np.array([r["best_score"] for r in rows], dtype=np.float64)
        npz_dict[f"{key}_histories"] = np.array([r["history"] for r in rows], dtype=np.float64)
    np.savez_compressed(output_npz, **npz_dict)


def print_summary(results: Dict[str, List[Dict[str, Any]]]):
    print("\n" + "=" * 116)
    print(" 2D Airfoil Ablation Summary (NACA0012, Re=3.5e6, AoA=3.5 deg, D=18, NFE=300)")
    print("=" * 116)
    print(f"{'Method':<16} | {'Step15':<11} | {'Final Mean ± SD':<17} | {'Median [Min, Max]':<22} | {'Δ vs MALIBO':<12} | {'Δ vs TuRBO':<12} | {'Time':<7}")
    print("-" * 116)
    m_base = np.mean([r["best_score"] for r in results["MALIBO"]])
    t_base = np.mean([r["best_score"] for r in results["Vanilla-TuRBO"]])
    for name in METHOD_ORDER:
        rows = results.get(name, [])
        if not rows:
            continue
        sc = np.array([r["best_score"] for r in rows])
        s15 = np.array([r["history"][14] for r in rows])
        times = [r["elapsed_seconds"] for r in rows if "elapsed_seconds" in r]
        t_str = f"{np.mean(times):5.1f}s" if times else "  --"
        print(
            f"{name:<16} | {np.mean(s15):5.2f}±{np.std(s15):4.2f} | "
            f"{np.mean(sc):6.2f} ± {np.std(sc):5.2f}   | "
            f"{np.median(sc):6.2f} [{np.min(sc):5.1f}, {np.max(sc):5.1f}] | "
            f"{np.mean(sc)-m_base:+6.2f} ({(np.mean(sc)-m_base)/m_base*100:+4.1f}%) | "
            f"{np.mean(sc)-t_base:+6.2f} ({(np.mean(sc)-t_base)/t_base*100:+4.1f}%) | {t_str}"
        )
    print("=" * 116)


def main():
    parser = argparse.ArgumentParser(description="2D Upgraded MALIBO Ablation Benchmark")
    parser.add_argument("--baseline_json", type=str, default="baselines/baseline_2d_results.json")
    parser.add_argument("--checkpoint", type=str, default="weights/malibo_2d_checkpoint.pt")
    parser.add_argument("--output_json", type=str, default="results/ablation_2d_results.json")
    parser.add_argument("--output_npz", type=str, default="results/ablation_2d_results.npz")
    parser.add_argument("--llm_url", type=str, default="http://127.0.0.1:30000/v1")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--variants", type=str, default="CoLT-MALIBO,w/o LLM,w/o Obj-TR,w/o Con-TR")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in BENCHMARK_SEEDS))
    parser.add_argument("--n_workers", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    with open(args.baseline_json, "r", encoding="utf-8") as f:
        base_blob = json.load(f)

    condition = dict(base_blob["condition"])
    reynolds = float(condition.get("reynolds", 3.5e6))
    alpha = float(condition.get("alpha", 3.5))
    max_evals = int(condition.get("max_evals", 300))

    results: Dict[str, List[Dict[str, Any]]] = {
        "Vanilla-TuRBO": base_blob["results"]["Vanilla-TuRBO"],
        "MALIBO": base_blob["results"]["MALIBO"],
        "LLM-Meta-TuRBO": base_blob["results"]["LLM-Meta-TuRBO"],
    }

    if not args.force_rerun and os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            prev = json.load(f).get("results", {})
        for old_k, new_k in LEGACY_KEY_MAP.items():
            if old_k in prev and new_k not in prev:
                migrated = []
                for row in prev[old_k]:
                    r_copy = dict(row)
                    r_copy["method"] = new_k
                    migrated.append(r_copy)
                results[new_k] = migrated
        for v in VARIANT_SPECS:
            if v in prev:
                results[v] = prev[v]

    for v in VARIANT_SPECS:
        results.setdefault(v, [])

    ctx = mp.get_context("spawn")

    for variant in variants:
        if variant not in VARIANT_SPECS:
            raise ValueError(f"Unknown variant: {variant}")
        existing_seeds = {r["seed"] for r in results[variant]}
        pending = [s for s in seeds if s not in existing_seeds]
        print(f"\n>>> [2D {variant}] pending seeds ({len(pending)}/{len(seeds)}): {pending}", flush=True)
        if not pending:
            continue

        task_args = [
            (variant, s, reynolds, alpha, max_evals, args.checkpoint, args.llm_url, args.llm_model, args.device)
            for s in pending
        ]
        n_proc = min(args.n_workers, len(pending))
        t_var = time.time()
        if n_proc == 1:
            _worker_init()
            for ta in task_args:
                res = _run_2d_variant_seed(ta)
                results[variant].append(res)
                n_int = len(res.get("interventions", []))
                print(
                    f"    [{variant} Seed {res['seed']:4d}] Step15={res['history'][14]:.2f} | "
                    f"Best={res['best_score']:.2f} | Feas={res['detailed']['is_feasible']} | "
                    f"Calls={n_int} | Time={res['elapsed_seconds']:.1f}s",
                    flush=True,
                )
                results[variant].sort(key=lambda r: seeds.index(r["seed"]) if r["seed"] in seeds else r["seed"])
                _save_2d_results(results, condition, args.output_json, args.output_npz)
        else:
            with ctx.Pool(processes=n_proc, initializer=_worker_init) as pool:
                for res in pool.imap_unordered(_run_2d_variant_seed, task_args):
                    results[variant].append(res)
                    n_int = len(res.get("interventions", []))
                    print(
                        f"    [{variant} Seed {res['seed']:4d}] Step15={res['history'][14]:.2f} | "
                        f"Best={res['best_score']:.2f} | Feas={res['detailed']['is_feasible']} | "
                        f"Calls={n_int} | Time={res['elapsed_seconds']:.1f}s",
                        flush=True,
                    )
                    results[variant].sort(key=lambda r: BENCHMARK_SEEDS.index(r["seed"]) if r["seed"] in BENCHMARK_SEEDS else r["seed"])
                    _save_2d_results(results, condition, args.output_json, args.output_npz)

        sc = [r["best_score"] for r in results[variant]]
        print(
            f"<<< [2D {variant} DONE in {time.time() - t_var:.1f}s] "
            f"Mean={np.mean(sc):.2f} ± {np.std(sc):.2f} | Med={np.median(sc):.2f} | Max={np.max(sc):.2f}",
            flush=True,
        )

    _save_2d_results(results, condition, args.output_json, args.output_npz)
    print_summary(results)


if __name__ == "__main__":
    main()
