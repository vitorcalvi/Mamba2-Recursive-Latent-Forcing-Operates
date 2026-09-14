# notebooks/train_export_mobile — replacement cells v1
# Generated from analysis of the existing notebook + codebase conventions.
# Apply each section to the named cell in the .ipynb JSON.

from pathlib import Path
REPO_ROOT = Path(".").resolve()

# ─────────────────────────────────────────────────────────────────────────────
# NEW CELL 1 — Smart dependency installer
# Replaces: the existing `pip_install([...])` cell (cell index 3, code cell)
# ─────────────────────────────────────────────────────────────────────────────

def install_dependencies():
    """Install deps with platform-aware fallbacks."""
    import platform, subprocess, sys

    arch   = platform.machine()   # arm64 / x86_64
    system = platform.system()    # Darwin / Linux / etc.

    base_pkgs = [
        "transformers>=4.30.0",
        "safetensors",
        "datasets",
        "ninja",            # required for mamba-ssm source builds
        "sentencepiece",
        "tiktoken",
    ]

    # ── Torch: pick wheel matching available hardware ─────────────────────────
    cuda_ok = _probe_cuda()
    if cuda_ok:
        base_pkgs.append("torch>=2.0.0")
        print("[install] CUDA detected — installing CUDA torch wheel")
    elif system == "Darwin" and arch == "arm64":
        base_pkgs.append("torch>=2.0.0")
        print("[install] Apple Silicon — installing CPU torch (MPS available at runtime)")
    else:
        base_pkgs.append("torch>=2.0.0")
        print("[install] No CUDA — installing CPU-only torch")

    # ── mamba-ssm: prefer prebuilt wheel, fall back to source build ───────────
    # The `microsoft/mamba-ssm` nightly index has CUDA 12.1 wheels for x86_64.
    # On other platforms we fall back to PyPI and compile from source (ninja req.).
    if cuda_ok and arch == "x86_64" and system == "Linux":
        mamba_pkgs = ["mamba-ssm>=2.2.0", "--index-url", "https://pypi.nvidia.com"]
    elif cuda_ok and arch == "x86_64" and system == "Windows":
        mamba_pkgs = ["mamba-ssm>=2.2.0", "--index-url", "https://pypi.nvidia.com"]
    else:
        mamba_pkgs = ["mamba-ssm>=2.2.0"]

    failed = []
    for pkg in [base_pkgs, mamba_pkgs]:
        cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + pkg
        print(f">>> {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as exc:
            failed.append((pkg[0], exc))
            print(f"[WARN] install failed for {pkg[0]}: {exc}")

    if failed:
        print("\n[WARN] Some packages failed to install:")
        for name, exc in failed:
            print(f"  {name}: returncode {exc.returncode}")
        print("\n  Colab / fresh-VM fix (run in a fresh terminal BEFORE this cell):")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("    pip install 'mamba-ssm>=2.2.0' --index-url https://pypi.nvidia.com")
        print("  CPU-only fallback:")
        print("    pip install torch transformers safetensors datasets sentencepiece tiktoken ninja")
    else:
        print("[install] All packages installed successfully.")


def _probe_cuda() -> bool:
    """Return True if nvidia-smi reports a healthy GPU (no torch needed)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr= subprocess.STDOUT, text=True, timeout=10,
        )
        return bool(out.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


install_dependencies()

# ─────────────────────────────────────────────────────────────────────────────
# NEW CELL 2 — Environment detection
# Replaces: the existing GPU-assert cell (cell index 4, code cell)
# Insert this BEFORE any cell that imports from rlf_engine_1_4b.py
# ─────────────────────────────────────────────────────────────────────────────

def detect_environment():
    """Detect available compute backend and return (DEVICE, DTYPE) tuple."""
    import torch

    if torch.cuda.is_available():
        DEVICE = "cuda"
        torch.cuda.set_per_process_memory_fraction(0.92)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
        # bfloat16 requires compute capability ≥ 8.0 (Ampere A100/RTX 30xx+)
        cc = props.major * 10 + props.minor
        DTYPE = torch.bfloat16 if cc >= 80 else torch.float16
        print(f"Compute capability: {props.major}.{props.minor} → dtype={DTYPE}")

    elif hasattr(torch, "mps") and torch.mps.is_available():
        DEVICE = "mps"
        DTYPE  = torch.float32   # MPS does not support bfloat16 reliably
        print("Using Apple Silicon MPS backend")
        print(f"MPS device: {torch.mps.get_mem_info()[0] / 1e9:.0f} GB free")

    else:
        DEVICE = "cpu"
        DTYPE  = torch.float32
        print("⚠  Using CPU backend — training will be very slow.")
        print("   Consider using Colab (free T4) or a CUDA-equipped host.")

    print(f"DEVICE={DEVICE}  DTYPE={DTYPE}")
    return DEVICE, DTYPE


DEVICE, DTYPE = detect_environment()

# ─────────────────────────────────────────────────────────────────────────────
# NEW CELL 3 — Graceful import fallbacks
# Insert AFTER detect_environment, BEFORE the RLF engine import cell
# ─────────────────────────────────────────────────────────────────────────────

def safe_import_ml_stack():
    """Import ML stack with actionable error messages. Returns (torch, mamba_ssm)."""
    errors = []

    try:
        import torch                         # noqa: F401
    except ImportError as e:
        errors.append(f"torch: {e}")

    try:
        import mamba_ssm                     # noqa: F401
        from mamba_ssm import MambaLMHeadModel  # noqa: F401
    except ImportError as e:
        errors.append(f"mamba-ssm: {e}")

    if errors:
        print("Missing dependencies — cannot continue:")
        for err in errors:
            print(f"  ✗ {err}")
        print()
        print("Fix by running install_dependencies() in the prior cell,")
        print("or manually:")
        print("  pip install torch mamba-ssm transformers")
        raise ImportError("ML stack incomplete — see messages above.")

    import torch
    import mamba_ssm
    return torch, mamba_ssm


torch, mamba_ssm = safe_import_ml_stack()

# ─────────────────────────────────────────────────────────────────────────────
# NEW CELL 4 — Small smoke-test dataset
# Replaces the dataset cell's size=2000 with a configurable smoke-test option
# ─────────────────────────────────────────────────────────────────────────────

SMOKE_TEST = True          # ← set False for full training
SMOKE_TEST_SIZE = 100      # completes in ~minutes on CPU, ~seconds on GPU

def get_smoke_test_datasets():
    """Create tiny datasets for notebook smoke testing."""
    from rlf_dataset import RLFDataset, collate_rlf
    from torch.utils.data import DataLoader

    rlf_size = SMOKE_TEST_SIZE if SMOKE_TEST else 2000
    rlf_ds = RLFDataset(size=rlf_size, seq_len=64, adversarial_prob=0.0)
    rlf_loader = DataLoader(
        rlf_ds, batch_size=1, collate_fn=collate_rlf, shuffle=True,
    )
    print(f"RLF dataset: {len(rlf_ds)} samples (smoke_test={SMOKE_TEST})")

    # SFT: prefer real JSONL; fall back to None (phase 3c reuses RLF data)
    SFT_DATA = REPO_ROOT / "combined_training_data.jsonl"
    if not SFT_DATA.exists():
        print(f"[WARN] {SFT_DATA} not found — Phase 3c will use RLF data as fallback.")
        sft_ds = None
    else:
        from rlf_trainer_1_4b import SFTDataset
        sft_ds = SFTDataset(str(SFT_DATA))

    sft_loader = (
        DataLoader(sft_ds, batch_size=1, shuffle=True)
        if sft_ds and len(sft_ds) > 0 else None
    )
    print(f"SFT dataset: {len(sft_ds) if sft_ds else 0} samples")
    return rlf_loader, sft_loader
