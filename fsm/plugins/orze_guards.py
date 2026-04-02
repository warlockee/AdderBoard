"""Reusable guards for orze experiment pipelines."""

import json
import time
from pathlib import Path

from fsm.engine import guard, Context


def _get_best_accuracy(results_dir: Path) -> tuple:
    """Return (best_accuracy, total_completed)."""
    best = 0.0
    completed = 0
    for d in results_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            if m.get("status") != "COMPLETED":
                continue
            completed += 1
            acc = m.get("accuracy", 0.0)
            if acc > best:
                best = acc
        except Exception:
            continue
    return best, completed


def _get_status(results_dir: Path) -> dict:
    """Read status.json."""
    path = results_dir / "status.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@guard("plateau_detected")
def plateau_detected(ctx: Context) -> str | None:
    """True when best metric hasn't improved in `plateau_window` completions."""
    best, completed = _get_best_accuracy(ctx.results_dir)
    window = ctx.vars.get("plateau_window", 20)

    # Track best in vars
    if best > ctx.vars.get("best_accuracy", 0.0):
        ctx.vars["best_accuracy"] = best
        ctx.vars["best_at_completion"] = completed

    best_at = ctx.vars.get("best_at_completion", 0)
    if completed < window:
        return None
    if (completed - best_at) >= window:
        return f"no improvement in {completed - best_at} experiments (best={best:.4f})"
    return None


@guard("has_improvement")
def has_improvement(ctx: Context) -> str | None:
    """True when accuracy improved over recorded best."""
    best, _ = _get_best_accuracy(ctx.results_dir)
    prev = ctx.vars.get("best_accuracy", 0.0)
    if best > prev:
        return f"accuracy improved: {prev:.4f} → {best:.4f}"
    return None


@guard("attempts_exhausted")
def attempts_exhausted(ctx: Context) -> str | None:
    """True when evolution attempts >= max_attempts."""
    attempts = ctx.vars.get("evolution_attempts", 0)
    max_att = ctx.vars.get("max_attempts", 3)
    if attempts >= max_att:
        return f"exhausted {attempts}/{max_att} evolution attempts"
    return None


@guard("attempts_remaining")
def attempts_remaining(ctx: Context) -> str | None:
    """True when evolution attempts < max_attempts."""
    attempts = ctx.vars.get("evolution_attempts", 0)
    max_att = ctx.vars.get("max_attempts", 3)
    if attempts < max_att:
        return f"attempt {attempts + 1}/{max_att}"
    return None


@guard("trigger_consumed")
def trigger_consumed(ctx: Context) -> str | None:
    """True when the trigger file no longer exists (evolution finished)."""
    trigger_name = ctx.vars.get("trigger_file", "_trigger_code_evolution")
    trigger = ctx.results_dir / trigger_name
    if not trigger.exists():
        return "trigger consumed (evolution cycle completed)"
    return None


@guard("is_deadlocked")
def is_deadlocked(ctx: Context) -> str | None:
    """True when paused + idle + empty queue for >5 minutes."""
    status = _get_status(ctx.results_dir)
    active = len(status.get("active", []))
    queued = status.get("queue_depth", 0)
    if active > 0 or queued > 0:
        return None

    # Check how long in current state
    history = ctx.state.get("history", [])
    if not history:
        return None
    last_time = history[-1].get("time", "")
    try:
        import datetime
        entered = datetime.datetime.fromisoformat(last_time)
        elapsed_min = (datetime.datetime.now() - entered).total_seconds() / 60
        if elapsed_min > 5:
            return f"deadlocked {elapsed_min:.0f}min: 0 active, 0 queued"
    except Exception:
        pass
    return None


@guard("queue_low")
def queue_low(ctx: Context) -> str | None:
    """True when queue depth < threshold."""
    status = _get_status(ctx.results_dir)
    queued = status.get("queue_depth", 0)
    threshold = ctx.vars.get("queue_low_threshold", 5)
    if queued < threshold:
        return f"queue low: {queued} < {threshold}"
    return None


@guard("failure_rate_high")
def failure_rate_high(ctx: Context) -> str | None:
    """True when recent failure rate exceeds threshold."""
    window = ctx.vars.get("fail_window", 10)
    threshold = ctx.vars.get("fail_threshold", 0.5)
    results_dir = ctx.results_dir

    recent = []
    for d in sorted(results_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            recent.append(m.get("status", "UNKNOWN"))
            if len(recent) >= window:
                break
        except Exception:
            continue

    if len(recent) < window:
        return None
    fail_rate = sum(1 for s in recent if s == "FAILED") / len(recent)
    if fail_rate >= threshold:
        return f"failure rate {fail_rate:.0%} >= {threshold:.0%} (last {window})"
    return None


@guard("always")
def always(ctx: Context) -> str | None:
    """Always passes. Use for unconditional transitions with actions."""
    return "unconditional"


@guard("family_imbalanced")
def family_imbalanced(ctx: Context) -> str | None:
    """True when approach family distribution is heavily skewed.

    Checks idea_lake.db for completed experiments. If any single family
    accounts for more than imbalance_threshold (default 60%) of completed
    ideas and total completed >= min_for_meta (default 30), returns a reason.
    """
    import sqlite3

    threshold = ctx.vars.get("imbalance_threshold", 0.6)
    min_completed = ctx.vars.get("min_for_meta", 30)
    lake_path = ctx.results_dir.parent / "idea_lake.db"
    if not lake_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(lake_path), timeout=5)
        rows = conn.execute(
            "SELECT approach_family, COUNT(*) as cnt FROM ideas "
            "WHERE status = 'completed' GROUP BY approach_family"
        ).fetchall()
        conn.close()
    except Exception:
        return None

    if not rows:
        return None
    total = sum(r[1] for r in rows)
    if total < min_completed:
        return None

    for family, count in rows:
        share = count / total
        if share >= threshold:
            return (f"family imbalance: {family or 'other'} has "
                    f"{share:.0%} of {total} completed ideas")
    return None


@guard("meta_cooldown_elapsed")
def meta_cooldown_elapsed(ctx: Context) -> str | None:
    """True when enough time has passed since the last meta-research run.

    Uses vars['last_meta_time'] (epoch) and vars['meta_cooldown_sec']
    (default 3600 = 1 hour).
    """
    cooldown = ctx.vars.get("meta_cooldown_sec", 3600)
    last = ctx.vars.get("last_meta_time", 0)
    elapsed = time.time() - last
    if elapsed >= cooldown:
        return f"meta cooldown elapsed ({elapsed / 60:.0f}min >= {cooldown / 60:.0f}min)"
    return None


@guard("bug_fixer_fixes_exhausted")
def bug_fixer_fixes_exhausted(ctx: Context) -> str | None:
    """True when bug fixer has hit the max fixes per hour limit."""
    max_fixes = ctx.vars.get("max_fixes_per_hour", 3)
    fix_count = ctx.vars.get("fixes_this_hour", 0)
    if fix_count >= max_fixes:
        return f"bug fixer exhausted: {fix_count}/{max_fixes} fixes this hour"
    return None


@guard("recent_fix_succeeded")
def recent_fix_succeeded(ctx: Context) -> str | None:
    """True when the most recent bug fix resolved the issue.

    Checks bug_fixer_issues/ for the most recent response file and
    sees if failure_rate has decreased since the fix.
    """
    issues_dir = ctx.results_dir / "bug_fixer_issues"
    if not issues_dir.exists():
        return None

    # Check if failure rate has decreased since last fix
    prev_fail_rate = ctx.vars.get("pre_fix_fail_rate", None)
    if prev_fail_rate is None:
        return None

    # Compute current failure rate
    window = ctx.vars.get("fail_window", 10)
    results_dir = ctx.results_dir
    recent = []
    for d in sorted(results_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            recent.append(m.get("status", "UNKNOWN"))
            if len(recent) >= window:
                break
        except Exception:
            continue

    if len(recent) < window:
        return None
    current_rate = sum(1 for s in recent if s == "FAILED") / len(recent)
    if current_rate < prev_fail_rate:
        return f"fix succeeded: failure rate {prev_fail_rate:.0%} -> {current_rate:.0%}"
    return None


@guard("meta_trigger_consumed")
def meta_trigger_consumed(ctx: Context) -> str | None:
    """True when _trigger_meta_research file has been consumed by the role."""
    trigger = ctx.results_dir / "_trigger_meta_research"
    if not trigger.exists():
        return "meta_research trigger consumed"
    return None


@guard("meta_trigger_stale")
def meta_trigger_stale(ctx: Context) -> str | None:
    """True when _trigger_meta_research has sat unconsumed for >15 minutes."""
    trigger = ctx.results_dir / "_trigger_meta_research"
    if not trigger.exists():
        return None
    try:
        age_min = (time.time() - trigger.stat().st_mtime) / 60
        if age_min > 15:
            return f"meta trigger stale ({age_min:.0f}min unconsumed)"
    except Exception:
        pass
    return None


@guard("research_stalled")
def research_stalled(ctx: Context) -> str | None:
    """True when research agent has produced zero new ideas for N consecutive cycles.

    Reads _research_logs/ to count recent cycles with zero output.
    """
    threshold = ctx.vars.get("stall_threshold", 10)
    log_dir = ctx.results_dir / "_research_logs"
    if not log_dir.exists():
        return None

    # Count recent consecutive zero-output cycles
    logs = sorted(log_dir.glob("cycle_*.log"), reverse=True)
    zero_count = 0
    for log_file in logs[:threshold + 5]:
        try:
            content = log_file.read_text(encoding="utf-8")
            # Check if the cycle produced any ideas
            if "new ideas" not in content.lower() and "ingested" not in content.lower():
                zero_count += 1
            else:
                break  # found a productive cycle, stop counting
        except Exception:
            continue

    ctx.vars["zero_output_count"] = zero_count
    if zero_count >= threshold:
        return f"research stalled: {zero_count} consecutive cycles with zero output"
    return None
