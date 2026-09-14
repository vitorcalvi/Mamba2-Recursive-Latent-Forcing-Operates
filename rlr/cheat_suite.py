"""cheat_suite.py — Detect whether RLF is reasoning or shortcutting via
prompt structure, loop collapse, tape replay, or synthetic-syntax artifacts.
Six probes plus a suite runner and a mock-model unit test."""

from __future__ import annotations
import random, statistics
from dataclasses import dataclass, field
from typing import Any
def _make_var_chain_fallback(rng: random.Random, hops: int, mode: str = "clean") -> tuple[str, list[str]]:
    words = ["Blue", "Red", "Cat", "Dog", "Sun", "Moon", "Fire", "Star", "Gold", "Ice", "Alpha"]
    val = rng.choice(words)
    entities = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(4)) for _ in range(hops + 1)]
    facts = [f"{entities[0]}={val}."] + [f"{entities[i]}={entities[i-1]}." for i in range(1, hops + 1)]
    query = entities[-1]
    rng.shuffle(facts)
    prompt = " ".join(facts) + f" What is {query}?"
    return prompt, [val] * hops + ["§"]

try:
    from rlf_dataset import make_var_chain  # type: ignore
except Exception:
    make_var_chain = _make_var_chain_fallback

if make_var_chain is None:
    make_var_chain = _make_var_chain_fallback

# ── Shared container ──────────────────────────────────────────────────────────
@dataclass
class ProbeResult:
    probe_name: str
    passed: bool
    score: float
    details: dict = field(default_factory=dict)
    def to_row(self) -> str:
        return f"| {self.probe_name} | {'✅' if self.passed else '❌'} | {self.score:.4f} |"

# ── Adapter ───────────────────────────────────────────────────────────────────
def _run(model, prompt, **kw):
    return model.run(prompt, **kw) if hasattr(model, "run") else model(prompt, **kw)

def _norm_token(s):
    return str(s).split()[0].rstrip(".,;:") if s else ""

def _acc(model, cases, device, **kw):
    if not cases: return 0.0
    h = 0
    for c in cases:
        p = c["prompt"] if isinstance(c, dict) else c.prompt
        e = c["expected"] if isinstance(c, dict) else c.expected
        out = _run(model, p, device=device, **kw)
        pred = (out.get("answer") or "").strip()
        if pred and _norm_token(pred) == _norm_token(e):
            h += 1
    return h / len(cases)

# ── 1. PromptShuffleProbe ─────────────────────────────────────────────────────
class PromptShuffleProbe:
    """Accuracy invariant under assignment-order permutation ⇒ exploiting
    dataset artifact, not reasoning."""
    def __init__(self, n_shuffles=4, max_degradation=0.15):
        self.n_shuffles, self.max_degradation = n_shuffles, max_degradation
    def evaluate(self, model, cases, device):
        base = _acc(model, cases, device)
        rng = random.Random(0xC0FFEE)
        accs = []
        for _ in range(self.n_shuffles):
            pert = []
            for c in cases:
                p = c["prompt"] if isinstance(c, dict) else c.prompt
                e = c["expected"] if isinstance(c, dict) else c.expected
                toks = p.split(); rng.shuffle(toks)
                pert.append({"prompt": " ".join(toks), "expected": e})
            accs.append(_acc(model, pert, device))
        avg = sum(accs) / len(accs) if accs else 0.0
        deg = max(0.0, base - avg)
        return ProbeResult("prompt_shuffle", deg <= self.max_degradation, deg,
            {"base_acc": base, "avg_shuffled_acc": avg, "per_shuffle": accs})

# ── 2. PerLoopDecodeInspector ─────────────────────────────────────────────────
def _seq_rep(toks):
    best = run = 1
    for i in range(1, len(toks)):
        run = run + 1 if toks[i] == toks[i-1] else 1
        best = max(best, run)
    return best if toks else 0

class PerLoopDecodeInspector:
    """Detect loop collapse (all loops identical) and tape-replay
    (sequential repetition of the same token across loops)."""
    def __init__(self, min_conf_stdev=0.05, max_seq_rep=2):
        self.min_conf_stdev, self.max_seq_rep = min_conf_stdev, max_seq_rep
    def evaluate(self, model, cases, device):
        stdevs, reps, per = [], [], []
        for c in cases:
            p = c["prompt"] if isinstance(c, dict) else c.prompt
            out = _run(model, p, device=device, return_trace=True)
            tr = out.get("per_loop") or []
            confs = [float(L.get("confidence", 0.0)) for L in tr]
            toks = [L.get("token") for L in tr]
            sd = statistics.pstdev(confs) if len(confs) > 1 else 0.0
            rp = _seq_rep(toks)
            stdevs.append(sd); reps.append(rp)
            per.append({"conf_stdev": sd, "seq_rep": rp, "n_loops": len(tr)})
        avg_sd = sum(stdevs) / len(stdevs) if stdevs else 0.0
        mx = max(reps) if reps else 0
        collapse = avg_sd < self.min_conf_stdev
        replay = mx > self.max_seq_rep
        return ProbeResult("per_loop_decode", not (collapse or replay), avg_sd,
            {"avg_conf_stdev": avg_sd, "max_seq_repetition": mx,
             "loop_collapse": collapse, "tape_replay": replay, "per_case": per})

# ── 3. LoopRemovalAblation ────────────────────────────────────────────────────
class LoopRemovalAblation:
    """If accuracy does not drop when looping is disabled, loops are decorative."""
    def __init__(self, min_delta=0.10): self.min_delta = min_delta
    def evaluate(self, model, cases, device):
        full = _acc(model, cases, device, max_loops=8)
        lob = _acc(model, cases, device, max_loops=1)
        d = full - lob
        return ProbeResult("loop_removal", d > self.min_delta, d,
            {"full_acc": full, "lobotomy_acc": lob, "min_delta": self.min_delta})

# ── 4. SemanticShiftEval ──────────────────────────────────────────────────────
BABI_PARAPHRASES = [
    ("Mary went to the kitchen. John went to the garden. Where is Mary?", "kitchen"),
    ("The cat is on the table. The dog is under the bed. Where is the cat?", "table"),
    ("Tom has a red ball. He gave it to Sara. Who has the ball now?", "Sara"),
    ("There are 3 birds on a branch. 2 fly away. How many are left?", "1"),
    ("A box has 7 apples. Tom eats 3. How many remain?", "4"),
    ("Alice is taller than Bob. Bob is taller than Eve. Who is shortest?", "Eve"),
    ("The key is in the drawer. The drawer is in the kitchen. Where is the key?", "kitchen"),
]

class SemanticShiftEval:
    """bAbI-style NL probes that share no surface form with the synthetic
    'A=B. C=A. What is C?' template. Guards empty prediction."""
    def evaluate(self, model, device):
        ok = 0; per = []
        for prompt, exp in BABI_PARAPHRASES:
            out = _run(model, prompt, device=device)
            pred = (out.get("answer") or "").strip()
            s = 1.0 if (pred and exp.lower() in pred.lower()) else 0.0
            ok += int(s)
            per.append({"prompt": prompt[:60], "expected": exp, "pred": pred, "score": s})
        n = len(BABI_PARAPHRASES)
        acc = ok / n if n else 0.0
        return ProbeResult("semantic_shift", acc >= 0.5, acc,
            {"n": n, "correct": ok, "per_q": per})

# ── 5. OODHopExtrapolation ────────────────────────────────────────────────────
class OODHopExtrapolation:
    """Chains with 9-12 hops. Training horizon: 1-8. Memorisation cannot
    reach above-chance accuracy here."""
    def __init__(self, n_per_hop=16, train_horizon=8):
        self.n_per_hop, self.train_horizon = n_per_hop, train_horizon
    def evaluate(self, model, device, test_hops=(9, 10, 11, 12)):
        if make_var_chain is None:
            return ProbeResult("ood_hop_extrapolation", False, 0.0,
                {"error": "make_var_chain unavailable (mamba_ssm missing)"})
        rng = random.Random(0xDEADBEEF)
        by_hop = {h: [] for h in test_hops}
        for h in test_hops:
            for _ in range(self.n_per_hop):
                prompt, chain = make_var_chain(rng, h, mode="clean")
                out = _run(model, prompt, device=device)
                pred = (out.get("answer") or "").strip()
                hit = 1.0 if (pred and _norm_token(pred) == _norm_token(chain[0])) else 0.0
                by_hop[h].append(hit)
        per = {h: (sum(v)/len(v) if v else 0.0) for h, v in by_hop.items()}
        overall = sum(per.values()) / len(per) if per else 0.0
        return ProbeResult("ood_hop_extrapolation", overall >= 0.25, overall,
            {"train_horizon": self.train_horizon, "test_hops": list(test_hops),
             "per_hop_acc": per})

# ── 6. CheatDetectionSuite ────────────────────────────────────────────────────
class CheatDetectionSuite:
    def __init__(self, eval_cases=None, test_hops=(9, 10, 11, 12)):
        self.shuffler = PromptShuffleProbe()
        self.inspector = PerLoopDecodeInspector()
        self.ablator = LoopRemovalAblation()
        self.semantic = SemanticShiftEval()
        self.ood = OODHopExtrapolation()
        self.eval_cases, self.test_hops = eval_cases, test_hops
    def _default_cases(self):
        rng = random.Random(42); out = []
        if make_var_chain is None:
            return [{"prompt": "A=Alpha. B=A. What is B?", "expected": "Alpha"}]
        for _ in range(32):
            p, c = make_var_chain(rng, hops=4, mode="clean")
            out.append({"prompt": p, "expected": c[0]})
        return out
    def run_all(self, model, device, eval_cases=None):
        cases = eval_cases or self.eval_cases or self._default_cases()
        return [
            self.shuffler.evaluate(model, cases, device),
            self.inspector.evaluate(model, cases, device),
            self.ablator.evaluate(model, cases, device),
            self.semantic.evaluate(model, device),
            self.ood.evaluate(model, device, self.test_hops),
        ]
    @staticmethod
    def summary(results):
        lines = ["| Probe | Pass | Score |", "|---|---|---|"]
        lines += [r.to_row() for r in results]
        lines += ["", f"**{sum(1 for r in results if r.passed)}/{len(results)} probes passed.**"]
        return "\n".join(lines)

# ── Self-contained unit test with a mock model ────────────────────────────────
class _MockModel:
    """Deterministic mock. Flags: prompt_inv, loop_collapser, tape_replayer."""
    def __init__(self, **flags): self.flags = flags
    def run(self, prompt, device="cpu", **kw):
        if self.flags.get("loop_collapser"):
            return {"answer": "X", "per_loop": [{"token": 7, "confidence": 0.5}] * 4}
        if self.flags.get("tape_replayer"):
            return {"answer": "Alpha", "per_loop": [{"token": 9, "confidence": 0.9}] * 4}
        max_loops = kw.get("max_loops", 16)

        # Handle bAbI paraphrases
        for p, a in [("Mary", "kitchen"), ("cat", "table"), ("ball", "Sara"),
                     ("birds", "1"), ("apples", "4"), ("shortest", "Eve"), ("key", "kitchen")]:
            if p in prompt:
                return {"answer": a, "per_loop": [{"token": 1, "confidence": 0.8}, {"token": 2, "confidence": 0.9}]}

        # If max_loops <= 1 (lobotomy), recurrence is disabled -> cannot chain
        if max_loops <= 1:
            return {"answer": "lobotomized", "per_loop": [{"token": 1, "confidence": 0.2}]}

        import re
        pairs = re.findall(r"(\w+)=([^\.\s]+)", prompt)
        env = dict(pairs)
        m = re.search(r"(\w+)\?", prompt)
        if m and m.group(1) in env:
            curr = m.group(1)
            hops = 0
            while curr in env and hops < max_loops:
                curr = env[curr]
                hops += 1
            answer = curr
        else:
            m_first = [t for t in prompt.split() if "=" in t]
            answer = m_first[0].split("=", 1)[1].rstrip(".") if m_first else "Alpha"

        if self.flags.get("prompt_inv"):
            answer = "Alpha"

        return {
            "answer": answer,
            "per_loop": [
                {"token": 1, "confidence": 0.45},
                {"token": 2, "confidence": 0.68},
                {"token": 3, "confidence": 0.92},
            ],
        }

if __name__ == "__main__":
    device = "cpu"
    print("Running CheatDetectionSuite on mock models…\n")
    suite = CheatDetectionSuite(eval_cases=[
        {"prompt": "A=Alpha. B=A. What is B?", "expected": "Alpha"},
    ])
    for label, mock in [("Honest model", _MockModel()),
                        ("Prompt-invariant cheater", _MockModel(prompt_inv=True)),
                        ("Loop collapser", _MockModel(loop_collapser=True)),
                        ("Tape replayer", _MockModel(tape_replayer=True))]:
        print(f"=== {label} ===")
        print(suite.summary(suite.run_all(mock, device)))
        print()
    print("All probes executed against the mock harness.")
