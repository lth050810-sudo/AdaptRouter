"""Command dispatcher: schema/setup commands do not import a training model."""

import sys


def main():
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        print(
            "AdaptRouter Table 3 API trainer\n\nCommands:\n  validate-data  Validate offline feedback or unlabeled routing rows\n  prepare-refit  Prepare the fixed positive-cost global recipe\n  download-base  Explicitly download and verify pinned base assets\n  train          Calibrate or refit the full-parameter router\n  route          Select experts using a native checkpoint or HF weights\n  smoke          Explicit two-step CUDA integration check\n\nUse <command> --help for its arguments."
        )
        return
    if len(sys.argv) > 1 and sys.argv[1] in {
        "validate-data",
        "prepare-refit",
        "download-base",
    }:
        from .tools import main as tools_main

        return tools_main(sys.argv[1:])
    from .train import main as train_main

    return train_main()
