"""Use the established Office behavior visualizer with the v11 registration."""

import plot_office_behavior_v6_results as plotting


if __name__ == "__main__":
    plotting.DEFAULT_CONFIG = (
        "configs/dynamic_spatial_generalization_office_practical_v11.yaml"
    )
    plotting.main()
