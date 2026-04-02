"""Idea gatekeeper: two-tier filtering to prevent GPU waste without killing good ideas.

Tier 1 (rule-based, instant, conservative):
  - EXACT duplicate: same config hash + same seed as a completed experiment
  - Known-dead combos from RESEARCH_RULES.md (share_all_norms, tie_gate, d=2/hd=2)
  These are safe to filter — they are literally the same experiment repeated.

Tier 2 (LLM agent, periodic, nuanced):
  - Reviews borderline ideas that Tier 1 didn't catch
  - Has full context: leaderboard, recent code changes, experiment distribution
  - Can reason about whether a "similar" config exercises a new code path
  - Decides: APPROVE, SKIP (with reason), or PRIORITIZE (boost priority)

Design principle: NEVER filter an idea that might exercise a new code path.
False negatives (letting a bad idea through) cost 25min of GPU time.
False positives (killing a good idea) cost an entire research direction.
"""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

from fsm.engine import guard, action, Context

logger = logging.getLogger("fsm")


# ---------------------------------------------------------------------------
# Tier 1: Rule-based. Only catches EXACT duplicates and proven-dead configs.
# Conservative by design — if in doubt, let it through.
# ---------------------------------------------------------------------------

# These are absolute — confirmed dead in RESEARCH_RULES.md
PROVEN_DEAD = [
    {"share_norms": True, "share_ln_f": True},   # share all 3 norms = always 0%
    {"tie_gate": True},                            # gate=up in SwiGLU = always 0%
    {"d_model": 2, "head_dim": 2},                # untrainable at any param count
]


def _config_fingerprint(config: dict) -> str:
    """Full config hash INCLUDING seed — only exact duplicates match."""
    cfg = {k: v for k, v in sorted(config.items())}
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def _config_structure_hash(config: dict) -> str:
    """Config hash EXCLUDING seed — for detecting same-architecture retries."""
    cfg = {k: v for k, v in sorted(config.items()) if k != "seed"}
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _has_novel_keys(config: dict, known_keys: set) -> bool:
    """True if config has keys not seen in any completed experiment.

    Novel keys likely exercise new code paths added by code evolution.
    NEVER filter ideas with novel keys.
    """
    return bool(set(config.keys()) - known_keys)


def _tier1_check(idea_id: str, config: dict,
                 completed_fingerprints: set,
                 known_keys: set) -> Optional[str]:
    """Returns rejection reason or None. ONLY rejects certainties."""

    # NEVER reject ideas with novel config keys (code evolution output)
    if _has_novel_keys(config, known_keys):
        return None

    # Reject exact duplicates (same config + same seed)
    fp = _config_fingerprint(config)
    if fp in completed_fingerprints:
        return f"exact duplicate (fingerprint {fp})"

    # Reject proven-dead configs
    for dead in PROVEN_DEAD:
        if all(config.get(k) == v for k, v in dead.items()):
            return f"proven dead: {dead}"

    return None


# ---------------------------------------------------------------------------
# Tier 2 context builder: gives the LLM agent full picture
# ---------------------------------------------------------------------------

def _build_gatekeeper_context(results_dir: Path, queued_ideas: list) -> str:
    """Build context for the LLM gatekeeper with full experiment distribution."""
    lines = ["# Idea Gatekeeper Context\n"]

    # Leaderboard summary
    report = results_dir / "report.md"
    if report.exists():
        content = report.read_text(encoding="utf-8")
        # Extract top 10 from report
        table_lines = [l for l in content.split("\n") if l.startswith("|") and "idea-" in l]
        if table_lines:
            lines.append("## Top Results")
            for l in table_lines[:10]:
                lines.append(l)
            lines.append("")

    # Experiment distribution
    lake_path = results_dir.parent / "idea_lake.db"
    if not lake_path.exists():
        lake_path = results_dir / "idea_lake.db"
    if lake_path.exists():
        try:
            conn = sqlite3.connect(str(lake_path), timeout=5)

            # Approach family distribution
            rows = conn.execute(
                "SELECT approach_family, COUNT(*), "
                "SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) "
                "FROM ideas GROUP BY approach_family"
            ).fetchall()
            if rows:
                lines.append("## Approach Distribution")
                for family, total, completed in rows:
                    lines.append(f"- {family or 'other'}: {total} total, {completed} completed")
                lines.append("")

            # Accuracy distribution
            rows = conn.execute(
                "SELECT idea_id, title, eval_metrics FROM ideas "
                "WHERE status='completed' AND eval_metrics IS NOT NULL "
                "ORDER BY rowid DESC LIMIT 20"
            ).fetchall()
            if rows:
                lines.append("## Recent Completions")
                for iid, title, metrics_str in rows:
                    try:
                        m = json.loads(metrics_str) if metrics_str else {}
                        acc = m.get("accuracy", "?")
                        lines.append(f"- {iid}: {title[:50]} → acc={acc}")
                    except Exception:
                        pass
                lines.append("")
            conn.close()
        except Exception:
            pass

    # Recent code changes (from git)
    try:
        import subprocess
        result = subprocess.run(
            ["git", "log", "--oneline", "-5", "--", "train.py"],
            capture_output=True, text=True, timeout=5,
            cwd=str(results_dir.parent))
        if result.stdout.strip():
            lines.append("## Recent train.py Changes")
            lines.append(result.stdout.strip())
            lines.append("")
    except Exception:
        pass

    # The ideas to review
    lines.append("## Ideas to Review")
    for idea in queued_ideas:
        lines.append(f"\n### {idea['id']}: {idea['title']}")
        if idea.get("hypothesis"):
            lines.append(f"Hypothesis: {idea['hypothesis']}")
        lines.append(f"Config keys: {', '.join(sorted(idea['config'].keys()))}")
        # Highlight novel keys
        if idea.get("novel_keys"):
            lines.append(f"**NOVEL KEYS**: {', '.join(idea['novel_keys'])}")
        lines.append(f"Seed: {idea['config'].get('seed', '?')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Guards & Actions
# ---------------------------------------------------------------------------

@guard("has_garbage_ideas")
def has_garbage_ideas(ctx: Context) -> str | None:
    """True when queued ideas contain Tier 1 filterable garbage."""
    lake_path = ctx.results_dir.parent / "idea_lake.db"
    if not lake_path.exists():
        lake_path = ctx.results_dir / "idea_lake.db"
    if not lake_path.exists():
        return None

    # Build completed fingerprints and known keys
    completed_fingerprints = set()
    known_keys = set()
    try:
        import yaml
        conn = sqlite3.connect(str(lake_path), timeout=5)
        rows = conn.execute(
            "SELECT config FROM ideas WHERE status = 'completed'"
        ).fetchall()
        for (config_str,) in rows:
            if not config_str:
                continue
            try:
                config = yaml.safe_load(config_str)
                if isinstance(config, dict):
                    completed_fingerprints.add(_config_fingerprint(config))
                    known_keys.update(config.keys())
            except Exception:
                continue

        # Check queued ideas
        rows = conn.execute(
            "SELECT idea_id, config FROM ideas WHERE status = 'queued'"
        ).fetchall()
        conn.close()
    except Exception:
        return None

    garbage_count = 0
    for idea_id, config_str in rows:
        if not config_str:
            continue
        try:
            config = yaml.safe_load(config_str)
            if not isinstance(config, dict):
                continue
        except Exception:
            continue
        if _tier1_check(idea_id, config, completed_fingerprints, known_keys):
            garbage_count += 1

    if garbage_count > 0:
        return f"{garbage_count} exact duplicates/dead configs in queue (of {len(rows)} queued)"
    return None


@action("filter_garbage_ideas")
def filter_garbage_ideas(ctx: Context):
    """Tier 1: Skip only exact duplicates and proven-dead configs."""
    lake_path = ctx.results_dir.parent / "idea_lake.db"
    if not lake_path.exists():
        lake_path = ctx.results_dir / "idea_lake.db"
    if not lake_path.exists():
        return

    completed_fingerprints = set()
    known_keys = set()
    try:
        import yaml
        conn = sqlite3.connect(str(lake_path), timeout=5)
        rows = conn.execute(
            "SELECT config FROM ideas WHERE status = 'completed'"
        ).fetchall()
        for (config_str,) in rows:
            if not config_str:
                continue
            try:
                config = yaml.safe_load(config_str)
                if isinstance(config, dict):
                    completed_fingerprints.add(_config_fingerprint(config))
                    known_keys.update(config.keys())
            except Exception:
                continue

        rows = conn.execute(
            "SELECT idea_id, config FROM ideas WHERE status = 'queued'"
        ).fetchall()
    except Exception:
        return

    filtered = 0
    for idea_id, config_str in rows:
        if not config_str:
            continue
        try:
            config = yaml.safe_load(config_str)
            if not isinstance(config, dict):
                continue
        except Exception:
            continue

        reason = _tier1_check(idea_id, config, completed_fingerprints, known_keys)
        if reason:
            try:
                conn.execute(
                    "UPDATE ideas SET status = 'skipped' WHERE idea_id = ?",
                    (idea_id,))
                idea_dir = ctx.results_dir / idea_id
                idea_dir.mkdir(exist_ok=True)
                (idea_dir / "metrics.json").write_text(json.dumps({
                    "status": "SKIPPED",
                    "skip_reason": reason,
                    "skipped_by": "tier1_filter",
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }, indent=2), encoding="utf-8")
                filtered += 1
                logger.info("Tier1 filtered %s: %s", idea_id, reason)
            except Exception as e:
                logger.error("Failed to filter %s: %s", idea_id, e)

    if filtered > 0:
        try:
            conn.commit()
        except Exception:
            pass
    try:
        conn.close()
    except Exception:
        pass

    ctx.vars["last_filtered_count"] = filtered
    ctx.vars["total_filtered"] = ctx.vars.get("total_filtered", 0) + filtered
    if filtered > 0:
        logger.info("Tier1 filtered %d exact duplicates/dead configs (total: %d)",
                     filtered, ctx.vars["total_filtered"])
