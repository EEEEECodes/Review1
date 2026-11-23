"""Plot observed synthetic training metrics and preserve their source JSON.

Requires matplotlib. No training, package installation, or downloads occur.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil


def load_results(path: Path) -> tuple[dict, dict]:
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if result.get("status") != "completed":
        raise ValueError("Input must contain a completed training run")
    settings = result["settings"]
    for name in ("rounds", "clients", "samples", "test_samples"):
        if not isinstance(settings.get(name), int) or settings[name] < 1:
            raise ValueError(f"settings.{name} must be a positive integer")
    series = {}
    for method in ("fedavg", "fedbif"):
        observed = result["methods"][method]
        history = observed["history"]
        if [point["round"] for point in history] != list(range(1, settings["rounds"] + 1)):
            raise ValueError(f"{method} history must include every configured round in order")
        points = [observed["initial"], *history]
        loss = [point["loss"] for point in points]
        accuracy = [point["synthetic_accuracy"] for point in points]
        if not all(math.isfinite(value) and value >= 0 for value in loss):
            raise ValueError(f"{method} loss values must be finite and nonnegative")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in accuracy):
            raise ValueError(f"{method} accuracy values must be finite fractions from 0 to 1")
        series[method] = {"loss": loss, "accuracy": [100 * value for value in accuracy]}
    return result, series


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=here / "training-results.json")
    parser.add_argument("--output-dir", type=Path, default=here.parent / "results")
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    result, series = load_results(source)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = result["settings"]
    rounds = list(range(settings["rounds"] + 1))
    styles = {
        "fedavg": {"color": "#2368AD", "marker": "o", "markersize": 9,
                   "markerfacecolor": "none", "markeredgewidth": 1.6,
                   "linestyle": "-", "zorder": 3},
        "fedbif": {"color": "#C75627", "marker": "D", "markersize": 4.5,
                   "linestyle": "--", "zorder": 4},
    }
    names = {"fedavg": "FedAvg", "fedbif": "FedBiF"}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "semibold",
                         "savefig.facecolor": "white"}):
        figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12.4, 5.8))
        figure.subplots_adjust(left=0.075, right=0.98, top=0.77, bottom=0.20, wspace=0.24)
        for method in ("fedavg", "fedbif"):
            values = series[method]
            loss_axis.plot(rounds, values["loss"], linewidth=2,
                           label=f"{names[method]} (final {values['loss'][-1]:.5f})", **styles[method])
            accuracy_axis.plot(rounds, values["accuracy"], linewidth=2,
                               label=f"{names[method]} (final {values['accuracy'][-1]:.1f}%)", **styles[method])
        loss_axis.axhline(0, color="#757575", linestyle="--", linewidth=1.5,
                          label="Ideal target: 0", zorder=1)
        accuracy_axis.axhline(100, color="#757575", linestyle="--", linewidth=1.5,
                              label="Ideal target: 100%", zorder=1)
        loss_axis.set(title="Observed loss", ylabel="Cross-entropy loss")
        accuracy_axis.set(title="Observed synthetic accuracy", ylabel="Synthetic test accuracy (%)", ylim=(0, 106))
        maximum_loss = max(max(values["loss"]) for values in series.values())
        loss_axis.set_ylim(-0.035 * max(maximum_loss, 0.1), 1.08 * max(maximum_loss, 0.1))
        for axis in (loss_axis, accuracy_axis):
            axis.set_xlabel("Federated round (0 = initial)")
            axis.set_xticks(rounds)
            axis.grid(axis="y", color="#E1E5EB", linewidth=0.8)
            axis.set_axisbelow(True)
            axis.legend(loc="best", frameon=True, framealpha=0.95, fontsize=9)
        figure.suptitle("FedAvg and FedBiF: observed training results", fontsize=16, y=0.96)
        figure.text(0.5, 0.875,
                    f"CPU synthetic smoke test | {settings['samples']} training images | "
                    f"{settings['test_samples']} test images | {settings['clients']} clients | "
                    f"{settings['rounds']} rounds", ha="center", color="#495466", fontsize=10)
        seed = settings.get("seed", "unspecified")
        figure.text(0.5, 0.07,
                    f"Synthetic data; one seed ({seed}); no real-dataset or network measurements.",
                    ha="center", color="#5D6673", fontsize=9)
        destination = output_dir / "training-comparison.png"
        figure.savefig(destination, dpi=200)
        plt.close(figure)

    json_destination = output_dir / "training-results.json"
    if not (json_destination.exists() and source.samefile(json_destination)):
        shutil.copyfile(source, json_destination)
    print(f"Saved {destination}")
    print(f"Source JSON preserved at {json_destination}")


if __name__ == "__main__":
    main()
