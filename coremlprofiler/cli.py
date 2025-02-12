from argparse import ArgumentParser
from coremlprofiler import CoreMLProfiler, ComputeUnitSetting
from huggingface_hub import snapshot_download
import os

def main():
    parser = ArgumentParser("coremlprofile", usage="coremlprofile <model_id>")
    parser.add_argument("file", help="Local path to mlpackage or mlmodelc, or path relative to the Hugging Face repo id specified in --hf_repo")
    parser.add_argument("--hf_repo", help="Hugging Face repository ID to download the mlpackage from", default=None)
    parser.add_argument("--detail", help="Report per-op device compatibility", action="store_true")
    parser.add_argument(
        "--compute_units",
        choices=[cu.value for cu in ComputeUnitSetting],
        default=ComputeUnitSetting.ALL.value,
        help="Compute units to use (default: all)"
    )

    args = parser.parse_args()
    if args.hf_repo:
        download_path = snapshot_download(
            repo_id=args.hf_repo,
            allow_patterns=[f"{args.file}/*"],
            local_dir=".",      # symlinks in ~/.cache don't work
        )
        print(download_path)
        model_path = os.path.join(download_path, args.file)
        print(model_path)
    else:
        model_path = args.file

    # Create the Profiler
    compute_units = ComputeUnitSetting(args.compute_units)
    profiler = CoreMLProfiler(model_path, compute_units=compute_units)

    # Print your device usage
    print(profiler.device_usage_summary_chart())

    if args.detail:
        print(profiler.operator_compatibility_report())