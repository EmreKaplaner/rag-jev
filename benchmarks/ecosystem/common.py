"""Reproducibility and cross-process paid-request accounting."""

import fcntl
import math
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.public.common import digest, read_json, write_json

ROOT = Path("artifacts/ecosystem-v1")
SEED = "rag-jev-ecosystem-pilot-2026-09-19"
ARMS = ["baseline", "bge", "jev_rerank", "jev_filter", "bge_jev_filter"]


def sample(ids, count):
    return sorted(set(ids), key=lambda x: digest([SEED, str(x)]))[:count]


def fingerprint_files(paths):
    import hashlib

    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths)}


def protocol():
    p = read_json(ROOT / "protocol.json")
    if fingerprint_files(p["code"]) != p["code"]:
        raise ValueError("Frozen inference code changed; create a new campaign")
    return p


class Ledger:
    """Reserve BEFORE calling; unknown charges remain reserved and block retries.

    File lock covers accounting only, so workers/processes can issue requests concurrently.
    Reservations are deliberately larger than bounded request costs. This is a local
    usage-based cap, not access to the provider's invoice or account-wide budget.
    """

    def __init__(self, root=ROOT, ceiling=25.0):
        self.root = Path(root)
        self.ceiling = ceiling
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with (self.root / "ledger.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / "ledger.json"
            data = read_json(path) if path.exists() else {}
            yield data
            write_json(path, data)

    def reserve(self, key, fingerprint, maximum):
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("positive finite reservation required")
        with self.locked() as data:
            if any(r["status"] == "reservation_exceeded" for r in data.values()):
                raise ValueError("Actual usage exceeded a reservation; reconcile before continuing")
            if key in data:
                raise ValueError("Request already attempted; never silently retry")
            committed = sum(r.get("cost_usd", r["reserved_usd"]) for r in data.values())
            if committed + maximum > self.ceiling:
                raise ValueError("Campaign API budget exhausted")
            data[key] = {
                "fingerprint": fingerprint,
                "reserved_usd": maximum,
                "started_at": datetime.now(UTC).isoformat(),
                "status": "started",
            }

    def finish(self, key, cost, status="complete"):
        with self.locked() as data:
            row = data[key]
            if row["status"] != "started":
                raise ValueError("Request already finalized")
            row["status"] = status
            if cost is not None:
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError("Invalid returned cost")
                row["cost_usd"] = cost
                if cost > row["reserved_usd"]:
                    # Preserve the actual liability, then fail all further work.
                    row["status"] = "reservation_exceeded"

    def summary(self):
        with self.locked() as data:
            known = sum(r.get("cost_usd", 0) for r in data.values())
            unknown = [r for r in data.values() if "cost_usd" not in r]
            reserved = sum(r["reserved_usd"] for r in unknown)
            return {
                "ceiling_usd": self.ceiling,
                "known_usage_cost_usd": known,
                "unreconciled_requests": len(unknown),
                "reserved_unknown_usd": reserved,
                "remaining_after_reservations_usd": self.ceiling - known - reserved,
                "requests": len(data),
            }


def select_indices(arm, count, jev=None, bge=None, post=None, threshold=0.2, top_k=10):
    """Five frozen policies; post scores align with BGE's independently selected top-k."""
    base = list(range(count))
    if arm == "baseline":
        return base[:top_k]
    if arm in {"bge", "bge_jev_filter"}:
        if bge is None or len(bge) != count:
            raise ValueError("BGE scores missing or misaligned")
        order = sorted(base, key=lambda i: (-bge[i], i))[:top_k]
        if arm == "bge":
            return order
        if post is None or len(post) != len(order):
            raise ValueError("Fresh scores on BGE context required")
        return [i for i, score in zip(order, post, strict=True) if score >= threshold]
    if arm not in {"jev_rerank", "jev_filter"} or jev is None or len(jev) != count:
        raise ValueError("Invalid policy or Jev scores")
    order = sorted(base, key=lambda i: (-jev[i], i))
    if arm == "jev_filter":
        order = [i for i in order if jev[i] >= threshold]
    return order[:top_k]
