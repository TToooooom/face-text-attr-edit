import os
import json
import csv
import argparse
from pathlib import Path


def safe_get(d, key, default=None):
    return d[key] if key in d else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs")
    parser.add_argument("--prefix", type=str, default="eval_v2")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_v2_summary")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    all_metrics = {}

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        if not subdir.name.startswith(args.prefix):
            continue

        metrics_path = subdir / "metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        run_name = subdir.name
        all_metrics[run_name] = metrics

        record = {
            "run": run_name,
            "num_samples": safe_get(metrics, "num_samples"),
            "overall_attribute_accuracy": safe_get(metrics, "overall_attribute_accuracy"),
            "changed_attribute_accuracy": safe_get(metrics, "changed_attribute_accuracy"),
            "keep_attribute_accuracy": safe_get(metrics, "keep_attribute_accuracy"),
            "exact_match_accuracy": safe_get(metrics, "exact_match_accuracy"),
            "total_l1": safe_get(metrics, "total_l1"),
            "outside_mask_l1": safe_get(metrics, "outside_mask_l1"),
            "inside_mask_l1": safe_get(metrics, "inside_mask_l1"),
            "self_l1": safe_get(metrics, "self_l1"),
            "rec_l1": safe_get(metrics, "rec_l1"),
            "average_alpha": safe_get(metrics, "average_alpha"),
            "tv_alpha": safe_get(metrics, "tv_alpha"),
        }
        records.append(record)

    if len(records) == 0:
        print(f"No metrics.json found under {root} with prefix {args.prefix}")
        return

    csv_path = output_dir / "eval_v2_summary.csv"
    json_path = output_dir / "eval_v2_all_metrics.json"
    md_path = output_dir / "eval_v2_summary.md"

    fieldnames = list(records[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# V2 Evaluation Summary\n\n")
        f.write("| Run | Samples | Overall Acc | Changed Acc | Keep Acc | Exact Match | Outside L1 | Alpha | Self L1 | Rec L1 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

        for r in records:
            f.write(
                f"| {r['run']} "
                f"| {r['num_samples']} "
                f"| {r['overall_attribute_accuracy']:.4f} "
                f"| {r['changed_attribute_accuracy']:.4f} "
                f"| {r['keep_attribute_accuracy']:.4f} "
                f"| {r['exact_match_accuracy']:.4f} "
                f"| {r['outside_mask_l1']:.6f} "
                f"| {r['average_alpha']:.6f} "
                f"| {r['self_l1']:.6f} "
                f"| {r['rec_l1']:.6f} |\n"
            )

        f.write("\n## Notes\n\n")
        f.write("- `Changed Acc` measures whether edited target attributes match the target vector.\n")
        f.write("- `Keep Acc` measures whether non-edited attributes are preserved.\n")
        f.write("- `Outside L1` measures pixel changes outside the editing region; lower is better.\n")
        f.write("- `Alpha` reflects average edit strength / edit area.\n")

    print("=" * 80)
    print("Collected V2 evaluation results")
    print("=" * 80)
    print(f"Runs: {len(records)}")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()