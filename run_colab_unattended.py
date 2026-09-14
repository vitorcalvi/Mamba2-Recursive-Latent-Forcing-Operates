"""run_colab_unattended.py — Unattended End-to-End Gate Runner for Google Colab.

Executes:
1. Hardware & Environment Audit (A100 VRAM, CUDA sm_80)
2. Gate G1: Track 1 (RLR) Anti-Cheat & Reasoner Validation
3. Gate G2: Track 2 (SSM-MBRL) vs GRU-RSSM Comparison Grid
4. Gate G3: Track 1 ↔ Track 2 Bridge 3-Arm Ablation Grid
5. Master Markdown Report Compilation
6. Automatic Compute-Unit Termination via google.colab.runtime.unassign()

Supports:
  --mode rapid   (Option B: 3 seeds, ~1.5 hours on A100) [DEFAULT]
  --mode full    (Option A: 5 seeds, ~4-5 hours on A100)
  --dry-run      (Local fast test without heavy compute or auto-shutdown)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from eval.gate_g1 import GateG1
from eval.gate_g2 import GateG2
from eval.gate_g3 import GateG3


def parse_args():
    parser = argparse.ArgumentParser(description="Unattended RLR & MBRL Gate Pipeline")
    parser.add_argument(
        "--mode",
        choices=["rapid", "full"],
        default="rapid",
        help="rapid (3 seeds, ~1.5h) or full (5 seeds, ~4-5h)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save report and logs (default: Google Drive or local ./rlr_runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast smoke tests without heavy training or Colab auto-shutdown",
    )
    return parser.parse_args()


class UnattendedPipeline:
    def __init__(self, mode: str = "rapid", output_dir: str | None = None, dry_run: bool = False):
        self.mode = mode
        self.dry_run = dry_run
        self.is_colab = "google.colab" in sys.modules or os.path.exists("/content")

        if output_dir:
            self.out_dir = Path(output_dir)
        elif self.is_colab and os.path.exists("/content/drive/MyDrive"):
            self.out_dir = Path("/content/drive/MyDrive/RLR_Runs")
        else:
            self.out_dir = Path(_ROOT) / "rlr_runs"

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.out_dir / "pipeline.log"
        self.report_file = self.out_dir / "Master_Gate_Report.md"

        self.log(f"=== Pipeline Initialized ===")
        self.log(f"Mode: {self.mode.upper()} {'(DRY-RUN)' if self.dry_run else ''}")
        self.log(f"Output Directory: {self.out_dir}")
        self.log(f"Colab Environment Detected: {self.is_colab}")

    def log(self, message: str) -> None:
        ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
        formatted = f"{ts} {message}"
        print(formatted, flush=True)
        try:
            with open(self.log_file, "a") as f:
                f.write(formatted + "\n")
        except Exception:
            pass

    def phase0_audit_hardware(self) -> dict:
        self.log("\n--- Phase 0: Hardware & CUDA Audit ---")
        
        # Ensure critical dependencies are present
        reqs = ["einops", "gymnasium", "popgym", "transformers"]
        missing = []
        for r in reqs:
            try:
                __import__(r)
            except ImportError:
                missing.append(r)

        if missing:
            self.log(f"Installing missing dependencies: {missing}...")
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", *missing],
                check=True,
            )
            self.log("Dependencies installed successfully.")

        device_name = "CPU"
        vram_gb = 0.0
        cuda_ok = torch.cuda.is_available()

        if cuda_ok:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            cap = torch.cuda.get_device_capability(0)
            self.log(f"GPU Detected: {device_name} ({vram_gb:.2f} GB VRAM, Compute Capability: sm_{cap[0]}{cap[1]})")
        else:
            self.log("CUDA not available. Running on CPU with fallback kernels.")

        try:
            from t2.world_model import MAMBA_AVAILABLE
        except Exception:
            MAMBA_AVAILABLE = False

        if MAMBA_AVAILABLE:
            self.log("Mamba SSM Dynamics Core: Compiled CUDA kernels active.")
        else:
            self.log("Mamba SSM Dynamics Core: Native Pure-PyTorch scan active (Safe & stable).")

        audit = {
            "cuda_available": cuda_ok,
            "device": device_name,
            "vram_gb": vram_gb,
            "pytorch_version": torch.__version__,
            "mamba_ssm_compiled": MAMBA_AVAILABLE,
        }
        return audit

    def phase1_gate_g1(self) -> dict:
        self.log("\n--- Phase 1: Gate G1 Evaluation (Track 1 RLR Anti-Cheat) ---")
        t0 = time.perf_counter()
        device = "cuda" if torch.cuda.is_available() and not self.dry_run else "cpu"

        gate = GateG1()
        res = gate.evaluate(device=device)
        elapsed = time.perf_counter() - t0

        self.log(f"Gate G1 Completed in {elapsed:.1f}s. Passed: {res.passed}")
        return {"result": res, "elapsed": elapsed}

    def phase2_gate_g2(self) -> dict:
        self.log("\n--- Phase 2: Gate G2 Evaluation (Track 2 SSM-MBRL vs GRU-RSSM) ---")
        t0 = time.perf_counter()
        n_seeds = 1 if self.dry_run else (3 if self.mode == "rapid" else 5)
        fast_mode = self.dry_run or (self.mode == "rapid")

        gate = GateG2(n_seeds=n_seeds)
        res = gate.evaluate(fast_mode=fast_mode)
        elapsed = time.perf_counter() - t0

        self.log(f"Gate G2 Completed in {elapsed:.1f}s. Passed: {res.passed}")
        return {"result": res, "elapsed": elapsed}

    def phase3_gate_g3(self) -> dict:
        self.log("\n--- Phase 3: Gate G3 Evaluation (Track 1 ↔ Track 2 Bridge 3-Arm Ablation) ---")
        t0 = time.perf_counter()
        n_seeds = 1 if self.dry_run else (3 if self.mode == "rapid" else 5)
        fast_mode = self.dry_run or (self.mode == "rapid")

        gate = GateG3(env_name="RepeatPrevious", n_seeds=n_seeds)
        res = gate.evaluate(fast_mode=fast_mode)
        elapsed = time.perf_counter() - t0

        self.log(f"Gate G3 Completed in {elapsed:.1f}s. Passed: {res.passed}")
        return {"result": res, "elapsed": elapsed}

    def phase4_synthesize_report(self, audit: dict, g1: dict, g2: dict, g3: dict) -> str:
        self.log("\n--- Phase 4: Compiling Master Markdown Report ---")
        now = time.strftime("%Y-%m-%d %H:%M:%S UTC")

        res1, res2, res3 = g1["result"], g2["result"], g3["result"]

        report = f"""# Recursive Latent Reasoner (RLR) — Master Gate Report
**Generated:** {now}  
**Execution Mode:** {self.mode.upper()} {'(DRY-RUN)' if self.dry_run else ''}  
**Hardware Accelerator:** {audit['device']} ({audit['vram_gb']:.2f} GB VRAM)  
**Total Wall-Clock:** {g1['elapsed'] + g2['elapsed'] + g3['elapsed']:.1f}s  

---

## Executive Gate Status

| Gate | Domain | Core Objective | Verdict | Wall-Clock |
|:---|:---|:---|:---|:---|
| **Gate G1** | Track 1 (RLR) | 5-Probe Anti-Cheat & Scratchpad Utilization | {'✅ PASSED' if res1.passed else '🛑 FAILED'} | {g1['elapsed']:.1f}s |
| **Gate G2** | Track 2 (SSM-MBRL) | Mamba-2 vs GRU-RSSM Parity/Win on POPGym | {'✅ PASSED' if res2.passed else '🛑 FAILED'} | {g2['elapsed']:.1f}s |
| **Gate G3** | Bridge Gate | +15% Hard-Mode Gain over Baseline | {'✅ PASSED' if res3.passed else '🛑 FAILED'} | {g3['elapsed']:.1f}s |

---

## Detailed Gate Reports

### Gate G1 (Track 1 Reasoner Proof)
{res1.summary}

---

### Gate G2 (Track 2 SSM World Model)
{res2.summary}

---

### Gate G3 (T1↔T2 Bridge 3-Arm Ablation)
{res3.summary}

---

## Empirical Decision Record
- **Gate G1 Assessment**: {'Demonstrates multi-hop latent reasoning.' if res1.passed else 'Fails reasoning criteria; model operates as sequential tape-reader.'}
- **Gate G2 Assessment**: {'Mamba-2 matches/exceeds GRU dynamics on memory benchmarks.' if res2.passed else 'Mamba-2 underperforms standard GRU.'}
- **Gate G3 Assessment**: {'Latent scratchpad bridge provides validated empirical boost.' if res3.passed else 'Scratchpad bridge does not exceed baseline ablations; negative result confirmed.'}
"""
        with open(self.report_file, "w") as f:
            f.write(report)

        self.log(f"Report written to: {self.report_file}")
        return report

    def phase5_terminate_runtime(self) -> None:
        if self.dry_run:
            self.log("\n[DRY-RUN] Skipping Colab VM unassignment.")
            return

        if self.is_colab:
            self.log("\n[COLAB] Unassigning VM to conserve compute units...")
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as e:
                self.log(f"Warning: Could not unassign runtime: {e}")
        else:
            self.log("\n[LOCAL] Pipeline finished successfully.")

    def run(self) -> None:
        try:
            audit = self.phase0_audit_hardware()
            g1 = self.phase1_gate_g1()
            g2 = self.phase2_gate_g2()
            g3 = self.phase3_gate_g3()
            report = self.phase4_synthesize_report(audit, g1, g2, g3)
            print("\n" + "=" * 72)
            print(report)
            print("=" * 72 + "\n")
        except Exception as e:
            self.log(f"FATAL PIPELINE ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.phase5_terminate_runtime()


if __name__ == "__main__":
    args = parse_args()
    pipeline = UnattendedPipeline(mode=args.mode, output_dir=args.output_dir, dry_run=args.dry_run)
    pipeline.run()
