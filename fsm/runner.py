#!/usr/bin/env python3
"""FSM runner: loads all procedures from procedures/ and steps each one.

Run as an orze script role:
    roles:
      fsm:
        mode: script
        script: fsm/runner.py
        args: ["--results-dir", "{results_dir}"]
        cooldown: 120
        timeout: 30

Or standalone:
    python3 fsm/runner.py --results-dir results
"""

import importlib
import logging
import sys
from pathlib import Path

# Ensure project root is on path (fsm/ lives inside the project)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FSM] %(levelname)s %(message)s",
)
logger = logging.getLogger("fsm")


def main():
    results_dir = Path("results")
    procedures_dir = Path("procedures")

    for i, arg in enumerate(sys.argv):
        if arg == "--results-dir" and i + 1 < len(sys.argv):
            results_dir = Path(sys.argv[i + 1])
        if arg == "--procedures-dir" and i + 1 < len(sys.argv):
            procedures_dir = Path(sys.argv[i + 1])

    if not results_dir.exists():
        logger.error("Results dir not found: %s", results_dir)
        sys.exit(1)

    # Auto-discover and import all guard/action plugins from fsm/plugins/
    plugins_dir = Path(__file__).parent / "plugins"
    if plugins_dir.exists():
        for py_file in sorted(plugins_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_name = f"fsm.plugins.{py_file.stem}"
            try:
                importlib.import_module(module_name)
                logger.debug("Loaded plugin: %s", module_name)
            except Exception as e:
                logger.error("Failed to load plugin %s: %s", module_name, e)

    # Set up JSONL activity log
    from fsm.engine import FSM, set_activity_log
    set_activity_log(results_dir / "_fsm_activity.jsonl")
    if not procedures_dir.exists():
        logger.error("Procedures dir not found: %s", procedures_dir)
        sys.exit(1)

    fsms = []
    for yaml_file in sorted(procedures_dir.glob("*.yaml")):
        try:
            fsm = FSM.from_yaml(str(yaml_file), results_dir)
            fsms.append(fsm)
            logger.debug("Loaded procedure: %s from %s", fsm.name, yaml_file)
        except Exception as e:
            logger.error("Failed to load %s: %s", yaml_file, e)

    if not fsms:
        logger.warning("No procedures found in %s", procedures_dir)
        sys.exit(0)

    # Step each FSM
    for fsm in fsms:
        status = fsm.status()
        logger.info("[%s] state=%s, transitions=%d",
                    fsm.name, status["state"], status["transitions"])
        fsm.step()


if __name__ == "__main__":
    main()
