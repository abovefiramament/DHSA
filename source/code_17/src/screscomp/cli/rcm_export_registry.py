from __future__ import annotations

import argparse
from pathlib import Path

from screscomp.data import dump_json
from screscomp.rcm.defaults import (
    CLEAN_AXIS_ALPHAS,
    CLEAN_AXIS_TIMINGS,
    CLEAN_AXIS_WEIGHTS,
    CLEAN_COMPOSITE_OBJECTIVE,
    CLEAN_PAIRWISE_OBJECTIVE,
    CLEAN_SELECTOR_DEFAULTS,
)
from screscomp.rcm.events import default_axis_specs, default_event_specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the built-in RCM EventSpec and AxisSpec registry.")
    p.add_argument("--out_json", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = {
        "events": {name: spec.to_dict() for name, spec in default_event_specs().items()},
        "axes": {name: spec.to_dict() for name, spec in default_axis_specs().items()},
        "clean_protocol": {
            "axis_timings": CLEAN_AXIS_TIMINGS,
            "axis_alphas": CLEAN_AXIS_ALPHAS,
            "axis_weights": CLEAN_AXIS_WEIGHTS,
            "composite_objective": CLEAN_COMPOSITE_OBJECTIVE,
            "pairwise_objective": CLEAN_PAIRWISE_OBJECTIVE,
            "selector_defaults": CLEAN_SELECTOR_DEFAULTS,
        },
    }
    dump_json(args.out_json, data)
    print(f"[rcm-export-registry] events={len(data['events'])} axes={len(data['axes'])} out={args.out_json}")


if __name__ == "__main__":
    main()
