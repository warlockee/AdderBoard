"""Reusable actions for orze experiment pipelines."""

import json
import logging
import time
from pathlib import Path

from fsm.engine import action, Context

logger = logging.getLogger("fsm")

# ---------------------------------------------------------------------------
# Pause registry: multiple FSMs can hold independent pause locks.
# Research stays paused while ANY lock is held.
# The .pause_research sentinel is written/removed based on the registry.
# ---------------------------------------------------------------------------

PAUSE_REGISTRY = "_pause_registry.json"


def _load_registry(results_dir: Path) -> dict:
    path = results_dir / PAUSE_REGISTRY
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_registry(results_dir: Path, registry: dict):
    path = results_dir / PAUSE_REGISTRY
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _sync_sentinel(results_dir: Path, registry: dict):
    """Write or remove .pause_research based on registry state."""
    sentinel = results_dir / ".pause_research"
    if registry:
        reasons = [f"{k}: {v['reason']}" for k, v in registry.items()]
        sentinel.write_text(
            "Paused by FSM:\n" + "\n".join(reasons) + "\n",
            encoding="utf-8")
    else:
        if sentinel.exists():
            sentinel.unlink()


def _acquire_pause(results_dir: Path, owner: str, reason: str):
    """Acquire a named pause lock."""
    registry = _load_registry(results_dir)
    registry[owner] = {
        "reason": reason,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_registry(results_dir, registry)
    _sync_sentinel(results_dir, registry)
    logger.info("Pause acquired by '%s': %s (total locks: %d)",
                owner, reason, len(registry))


def _release_pause(results_dir: Path, owner: str):
    """Release a named pause lock. Sentinel removed only if no locks remain."""
    registry = _load_registry(results_dir)
    if owner in registry:
        del registry[owner]
        _save_registry(results_dir, registry)
        _sync_sentinel(results_dir, registry)
        logger.info("Pause released by '%s' (remaining locks: %d)",
                    owner, len(registry))
    else:
        logger.debug("Pause release by '%s' — no lock held", owner)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@action("trigger_evolution")
def trigger_evolution(ctx: Context):
    """Write trigger file for code_evolution role."""
    trigger_name = ctx.vars.get("trigger_file", "_trigger_code_evolution")
    trigger = ctx.results_dir / trigger_name
    attempt = ctx.vars.get("evolution_attempts", 0) + 1
    trigger.write_text(json.dumps({
        "signal": "plateau",
        "attempt": attempt,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "fsm",
    }, indent=2), encoding="utf-8")
    ctx.vars["evolution_attempts"] = attempt
    logger.info("Wrote %s (attempt %d)", trigger_name, attempt)


@action("pause_research")
def pause_research(ctx: Context):
    """Acquire a pause lock for the calling FSM."""
    # Derive owner from FSM name in state
    owner = ctx.state.get("_fsm_name", "unknown")
    # Fallback: try to get from the current state key
    if owner == "unknown":
        for key in ("current",):
            if key in ctx.state:
                owner = f"fsm_{ctx.state[key]}"
                break
    # During maintain ticks, ctx.reason is empty. Fall back to the reason
    # from the last transition that entered this state, so the pause registry
    # always has a meaningful explanation.
    reason = ctx.reason
    if not reason:
        history = ctx.state.get("history", [])
        if history:
            reason = history[-1].get("reason", "maintain (no reason recorded)")
        else:
            reason = f"maintain in state {ctx.state.get('current', '?')}"
    _acquire_pause(ctx.results_dir, owner, reason)


@action("unpause_research")
def unpause_research(ctx: Context):
    """Release the pause lock for the calling FSM."""
    owner = ctx.state.get("_fsm_name", "unknown")
    if owner == "unknown":
        for key in ("current",):
            if key in ctx.state:
                owner = f"fsm_{ctx.state[key]}"
                break
    _release_pause(ctx.results_dir, owner)


@action("reset_attempts")
def reset_attempts(ctx: Context):
    """Reset evolution attempt counter."""
    ctx.vars["evolution_attempts"] = 0


@action("update_best")
def update_best(ctx: Context):
    """Update best_accuracy and best_at_completion from current results."""
    from fsm.plugins.orze_guards import _get_best_accuracy
    best, completed = _get_best_accuracy(ctx.results_dir)
    ctx.vars["best_accuracy"] = best
    ctx.vars["best_at_completion"] = completed
    logger.info("Updated best: %.4f at %d completions", best, completed)


@action("write_marker")
def write_marker(ctx: Context):
    """Write a marker file for debugging/auditing."""
    marker = ctx.results_dir / f"_fsm_marker_{int(time.time())}.json"
    marker.write_text(json.dumps({
        "fsm": ctx.state.get("_fsm_name", "unknown"),
        "reason": ctx.reason,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "vars": ctx.vars,
    }, indent=2), encoding="utf-8")


@action("notify")
def notify_action(ctx: Context):
    """Log a notification (extend to Slack/Telegram via orze notifications)."""
    logger.warning("NOTIFICATION: [%s] %s", ctx.state.get("current", "?"), ctx.reason)


@action("trigger_meta_research")
def trigger_meta_research(ctx: Context):
    """Write trigger file for meta_research agent."""
    trigger = ctx.results_dir / "_trigger_meta_research"
    trigger.write_text(json.dumps({
        "signal": ctx.reason,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "fsm",
    }, indent=2), encoding="utf-8")
    ctx.vars["last_meta_time"] = time.time()
    logger.info("Triggered meta-research: %s", ctx.reason)


@action("record_pre_fix_state")
def record_pre_fix_state(ctx: Context):
    """Snapshot failure rate before bug fixer engages, for later comparison."""
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
    if recent:
        fail_rate = sum(1 for s in recent if s == "FAILED") / len(recent)
        ctx.vars["pre_fix_fail_rate"] = fail_rate
    ctx.vars["fixes_this_hour"] = ctx.vars.get("fixes_this_hour", 0) + 1
    logger.info("Recorded pre-fix state: fail_rate=%.0f%%",
                ctx.vars.get("pre_fix_fail_rate", 0) * 100)


@action("reset_fix_counter")
def reset_fix_counter(ctx: Context):
    """Reset the hourly fix counter."""
    ctx.vars["fixes_this_hour"] = 0
    ctx.vars.pop("pre_fix_fail_rate", None)
    logger.info("Bug fixer fix counter reset")


@action("clean_stale_trigger")
def clean_stale_trigger(ctx: Context):
    """Remove a stale trigger file that was never consumed."""
    trigger = ctx.results_dir / "_trigger_meta_research"
    if trigger.exists():
        trigger.unlink()
        logger.warning("Cleaned stale trigger: _trigger_meta_research")
