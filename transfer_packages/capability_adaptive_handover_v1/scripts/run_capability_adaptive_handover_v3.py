"""Run Adaptive-v3 using the frozen v2 environment, monitor and foundations."""

import copy

import run_capability_adaptive_handover_v2 as runner
from astar_d3qn.training.capability_adaptive_v3 import (
    METHOD, train_capability_adaptive_v3, validate_v3_settings,
)


def validate_config(config, manifest, source_protocol):
    validate_v3_settings(config)
    if config["experiment"] != METHOD or config["output_root"] != f"results/{METHOD}":
        raise SystemExit("Adaptive-v3 experiment/output namespace mismatch.")
    frozen = copy.deepcopy(config)
    frozen.pop("adaptive_v3")
    frozen["experiment"] = "capability_adaptive_handover_v2_independent_monitor"
    frozen["output_root"] = "results/capability_adaptive_handover_v2_independent_monitor"
    runner.validate_config(frozen, manifest, source_protocol)


if __name__ == "__main__":
    runner.main(config_default="configs/capability_adaptive_handover_v3.yaml",
                method=METHOD, config_validator=validate_config,
                trainer=train_capability_adaptive_v3)
