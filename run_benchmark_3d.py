"""3D NREL Phase VI Blade 7-Method Main & Ablation Benchmark Runner (V=7.0 m/s, 72 rpm, D=63, NFE=300, 10 seeds).

Group 1 (Main Comparison — Baselines vs. Proposed Full Method):
  - TuRBO          (frozen baseline, TuRBO-1)
  - MALIBO         (frozen baseline, Paper-Exact)
  - LLM-Meta-TuRBO (frozen baseline)
  - CoLT-MALIBO    (Full upgraded MALIBO: Obj-TR + Con-TR + LLM)

Group 2 (Ablation Study — Full Method vs. Single-Component Removals):
  - CoLT-MALIBO    (Full)
  - w/o LLM        (Removes LLM offset injection & meta-control)
  - w/o Obj-TR     (Removes TuRBO objective trust-region box)
  - w/o Con-TR     (Removes SCBO Matérn-5/2 constrained feasibility surrogate)

Saves unified 7-method results to `results/ablation_3d_results.json` and `.npz`.
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
import torch

from src.case_3d.blade_geometry import BASELINE_PITCH_DEG, BEM_NUM_SECTIONS
from src.case_3d.evaluator import WindTurbineBladeEvaluator
from src.case_3d.llm_agent import LLMMALIBO3DAgent
from src.case_3d.malibo_core import load_3d_malibo_model
from src.case_3d.upgraded_malibo import UpgradedMALIBOBladeOptimizer

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
    "TuRBO",
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


def _run_3d_variant_seed(
    args: Tuple[str, int, float, float, float, int, int, str, str, str, str]
) -> Dict[str, Any]:
    (
        variant,
        seed,
        wind_speed,
        rpm,
        pitch_deg,
        n_bem,
        max_evals,
        ckpt_path,
        llm_url,
        llm_model,
        device_str,
    ) = args
    use_tr, use_llm, use_feas = VARIANT_SPECS[variant]
    device = torch.device(device_str)
    meta_model = load_3d_malibo_model(ckpt_path, device=device)

    ev = WindTurbineBladeEvaluator(
        wind_speed=wind_speed,
        rpm=rpm,
        pitch_angle_deg=pitch_deg,
        n_bem_stations=n_bem,
        max_evals=max_evals,
    )
    x0 = ev.baseline_x0
    llm_agent = (
        LLMMALIBO3DAgent(base_url=llm_url, model=llm_model, use_trust_region=use_tr)
        if use_llm
        else None
    )

    opt = UpgradedMALIBOBladeOptimizer(
        meta_model=meta_model,
        evaluator=ev,
        use_trust_region=use_tr,
        use_llm=use_llm,
        use_feas_surrogate=use_feas,
        use_gamma_annealing=True,
        infeas_failure_weight=2,
        x0=x0,
        max_evals=max_evals,
        seed=seed,
        llm_agent=llm_agent,
        device=device,
    )
    return opt.run()


def _save_3d_results(
    results: Dict[str, List[Dict[str, Any]]],
    metadata: Dict[str, Any],
    output_json: str,
    output_npz: str,
):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    base_power = float(metadata["baseline_power_kW"])
    m_base = np.mean([r["best_score"] for r in results["MALIBO"]])
    t_base = np.mean([r["best_score"] for r in results["TuRBO"]])

    summary = []
    for name in METHOD_ORDER:
        rows = results.get(name, [])
        if not rows:
            continue
        sc = np.array([r["best_score"] for r in rows], dtype=np.float64)
        s15 = np.array([r["history"][14] for r in rows], dtype=np.float64)
        cps = np.array([r["detailed"]["Cp"] for r in rows], dtype=np.float64)
        thrusts = np.array([r["detailed"]["thrust_kN"] for r in rows], dtype=np.float64)
        times = np.array([r["elapsed_seconds"] for r in rows], dtype=np.float64)
        feas = sum(1 for r in rows if r["detailed"].get("is_feasible", False))
        summary.append({
            "method": name,
            "n_seeds": len(rows),
            "step15_mean": float(np.mean(s15)),
            "step15_std": float(np.std(s15)),
            "mean_power_kW": float(np.mean(sc)),
            "std_power_kW": float(np.std(sc)),
            "median_power_kW": float(np.median(sc)),
            "min_power_kW": float(np.min(sc)),
            "max_power_kW": float(np.max(sc)),
            "gain_vs_baseline_pct": float((np.mean(sc) - base_power) / base_power * 100.0),
            "delta_vs_malibo_kW": float(np.mean(sc) - m_base),
            "delta_vs_turbo_kW": float(np.mean(sc) - t_base),
            "mean_Cp": float(np.mean(cps)),
            "mean_thrust_kN": float(np.mean(thrusts)),
            "feasibility_rate": feas / len(rows),
            "mean_elapsed_seconds": float(np.mean(times)),
        })

    payload = {
        "metadata": metadata,
        "summary": summary,
        "results": results,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    npz_dict: Dict[str, Any] = {"seeds": np.array(metadata["seeds"])}
    for name in METHOD_ORDER:
        rows = results.get(name, [])
        if not rows:
            continue
        key = name.lower().replace("/", "").replace(" ", "_").replace("-", "_")
        npz_dict[f"{key}_scores"] = np.array([r["best_score"] for r in rows], dtype=np.float64)
        npz_dict[f"{key}_histories"] = np.array([r["history"] for r in rows], dtype=np.float64)
        npz_dict[f"{key}_cps"] = np.array([r["detailed"]["Cp"] for r in rows], dtype=np.float64)
        npz_dict[f"{key}_times"] = np.array([r["elapsed_seconds"] for r in rows], dtype=np.float64)
    np.savez_compressed(output_npz, **npz_dict)


def print_summary(results: Dict[str, List[Dict[str, Any]]], baseline_power: float):
    print("\n" + "=" * 124)
    print(" 3D NREL Phase VI Ablation Summary (V=7.0 m/s, 72 rpm, 31 full-cosine BEM, D=63, NFE=300)")
    print("=" * 124)
    print(f"{'Method':<16} | {'Step15':<13} | {'Power Mean ± SD (kW)':<20} | {'Med [Min, Max]':<22} | {'Gain/Base':<10} | {'Thrust':<8} | {'Time':<7}")
    print("-" * 124)
    for name in METHOD_ORDER:
        rows = results.get(name, [])
        if not rows:
            continue
        sc = np.array([r["best_score"] for r in rows])
        s15 = np.array([r["history"][14] for r in rows])
        th = np.array([r["detailed"]["thrust_kN"] for r in rows])
        tm = np.array([r["elapsed_seconds"] for r in rows])
        gain = (np.mean(sc) - baseline_power) / baseline_power * 100.0
        print(
            f"{name:<16} | {np.mean(s15):6.4f}±{np.std(s15):5.4f} | "
            f"{np.mean(sc):6.4f} ± {np.std(sc):6.4f}     | "
            f"{np.median(sc):6.4f} [{np.min(sc):5.3f},{np.max(sc):5.3f}] | "
            f"{gain:+7.2f}%  | {np.mean(th):6.4f}kN | {np.mean(tm):5.1f}s"
        )
    print("=" * 124)


def main():
    parser = argparse.ArgumentParser(description="3D Upgraded MALIBO Ablation Benchmark")
    parser.add_argument("--baseline_json", type=str, default="baselines/baseline_3d_results.json")
    parser.add_argument("--checkpoint", type=str, default="weights/malibo_3d_checkpoint.pt")
    parser.add_argument("--output_json", type=str, default="results/ablation_3d_results.json")
    parser.add_argument("--output_npz", type=str, default="results/ablation_3d_results.npz")
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

    metadata = dict(base_blob["metadata"])
    wind_speed = float(metadata.get("wind_speed", 7.0))
    rpm = float(metadata.get("rpm", 72.0))
    pitch_deg = float(metadata.get("pitch_angle_deg", BASELINE_PITCH_DEG))
    n_bem = int(metadata.get("n_bem_stations", BEM_NUM_SECTIONS))
    max_evals = int(metadata.get("max_evals", 300))
    baseline_power = float(metadata["baseline_power_kW"])

    results: Dict[str, List[Dict[str, Any]]] = {
        "TuRBO": base_blob["results"]["TuRBO"],
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
        print(f"\n>>> [3D {variant}] pending seeds ({len(pending)}/{len(seeds)}): {pending}", flush=True)
        if not pending:
            continue

        task_args = [
            (
                variant, s, wind_speed, rpm, pitch_deg, n_bem, max_evals,
                args.checkpoint, args.llm_url, args.llm_model, args.device,
            )
            for s in pending
        ]
        n_proc = min(args.n_workers, len(pending))
        t_var = time.time()
        if n_proc == 1:
            _worker_init()
            for ta in task_args:
                res = _run_3d_variant_seed(ta)
                results[variant].append(res)
                det = res["detailed"]
                n_int = len(res.get("intervention_history", []))
                print(
                    f"    [{variant} Seed {res['seed']:4d}] Step15={res['history'][14]:.4f} | "
                    f"Best={res['best_score']:.4f} kW (+{(res['best_score']-baseline_power)/baseline_power*100:.2f}%) | "
                    f"T={det['thrust_kN']:.4f}kN | Feas={det['is_feasible']} | Calls={n_int} | Time={res['elapsed_seconds']:.1f}s",
                    flush=True,
                )
                results[variant].sort(key=lambda r: BENCHMARK_SEEDS.index(r["seed"]) if r["seed"] in BENCHMARK_SEEDS else r["seed"])
                _save_3d_results(results, metadata, args.output_json, args.output_npz)
        else:
            with ctx.Pool(processes=n_proc, initializer=_worker_init) as pool:
                for res in pool.imap_unordered(_run_3d_variant_seed, task_args):
                    results[variant].append(res)
                    det = res["detailed"]
                    n_int = len(res.get("intervention_history", []))
                    print(
                        f"    [{variant} Seed {res['seed']:4d}] Step15={res['history'][14]:.4f} | "
                        f"Best={res['best_score']:.4f} kW (+{(res['best_score']-baseline_power)/baseline_power*100:.2f}%) | "
                        f"T={det['thrust_kN']:.4f}kN | Feas={det['is_feasible']} | Calls={n_int} | Time={res['elapsed_seconds']:.1f}s",
                        flush=True,
                    )
                    results[variant].sort(key=lambda r: BENCHMARK_SEEDS.index(r["seed"]) if r["seed"] in BENCHMARK_SEEDS else r["seed"])
                    _save_3d_results(results, metadata, args.output_json, args.output_npz)

        sc = [r["best_score"] for r in results[variant]]
        print(
            f"<<< [3D {variant} DONE in {time.time() - t_var:.1f}s] "
            f"Mean={np.mean(sc):.4f} ± {np.std(sc):.4f} kW | Med={np.median(sc):.4f} | Max={np.max(sc):.4f}",
            flush=True,
        )

    _save_3d_results(results, metadata, args.output_json, args.output_npz)
    print_summary(results, baseline_power)


if __name__ == "__main__":
    main()
