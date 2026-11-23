"""Small CPU smoke benchmark using the checkout's trainer and FedBiF plugin.

Run the accompanying PowerShell setup first to prepare Python, PyTorch, and
the repaired working copy. This script uses generated images and no downloads.
The resulting accuracy is a synthetic smoke-test metric, not a paper result.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
from pathlib import Path
import sys
import time


def arguments() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=here.parent / "fedbif-tpds25")
    parser.add_argument("--output", type=Path, default=here / "training-results.json")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--samples", type=int, default=256, help="Total training images")
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    for key in ("rounds", "clients", "samples", "test_samples", "epochs", "batch_size", "threads"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not 2 <= args.bits <= 8:
        parser.error("--bits must be between 2 and 8")
    if args.samples < 2 * args.clients or args.samples % args.clients:
        parser.error("--samples must be divisible by --clients, with at least two per client")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    args.repo = args.repo.resolve()
    if not (args.repo / "plugin" / "bif.py").is_file():
        parser.error(f"Repository not found: {args.repo}")
    return args


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    sys.path.insert(0, str(args.repo))
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        bif = importlib.import_module("plugin.bif")
        trainer = importlib.import_module("train.trainer")
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency or unrepaired checkout: {exc}. "
            "Run the supplied PowerShell setup, then pass its repaired checkout with --repo."
        ) from exc

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    def require(condition, message):
        if not bool(condition):
            raise RuntimeError(message)

    # Exercise the actual quantizer, including the all-zero edge case repaired
    # by the setup helper, and the actual straight-through gradient estimator.
    zero = torch.zeros(16)
    frozen, virtual, alpha = bif.stochastic_quantize(zero, args.bits, 0)
    require(all(torch.isfinite(x).all() for x in (frozen, virtual, alpha)),
            "All-zero quantization produced non-finite values; apply the repair helper")
    require(torch.equal(frozen + bif.virtual_to_binary(virtual, alpha), zero),
            "All-zero quantization did not reconstruct zeros")
    probe = torch.tensor([-0.2, 0.3], requires_grad=True)
    binary = bif.virtual_to_binary(probe, torch.tensor(0.5))
    require(torch.equal(binary.detach(), torch.tensor([0.0, 0.5])), "Binary forward check failed")
    binary.sum().backward()
    require(torch.equal(probe.grad, torch.ones_like(probe)), "Straight-through gradient check failed")

    codes = torch.arange(2 ** args.bits, dtype=torch.int64)
    midpoint = 2 ** (args.bits - 1)
    weights = (codes.float() - midpoint) / midpoint
    for bit_pos in range(args.bits):
        mask = 1 << bit_pos
        frozen, virtual, alpha = bif.stochastic_quantize(weights, args.bits, bit_pos)
        require(torch.allclose(frozen + bif.virtual_to_binary(virtual, alpha), weights),
                f"Reconstruction failed for selected bit {bit_pos}")
        flipped_virtual = torch.where((codes & mask) != 0, -torch.ones_like(weights), torch.ones_like(weights))
        flipped = frozen + bif.virtual_to_binary(flipped_virtual, alpha)
        expected = ((codes ^ mask).float() - midpoint) / midpoint
        require(torch.allclose(flipped, expected), f"Selected-bit flip failed for bit {bit_pos}")

    def make_images(count: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(count) % 2
        images = 0.15 * torch.randn(count, 1, 8, 8, generator=generator)
        images[labels == 0, :, 1:7, 0:3] += 1.0
        images[labels == 1, :, 1:7, 5:8] += 1.0
        order = torch.randperm(count, generator=generator)
        return TensorDataset(images[order], labels[order])

    # Each client has both classes; test images use an independent random seed.
    per_client = args.samples // args.clients
    datasets = [make_images(per_client, args.seed + 100 + cid) for cid in range(args.clients)]
    test_loader = DataLoader(make_images(args.test_samples, args.seed + 500), batch_size=args.batch_size)
    base = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
        nn.ReLU(), nn.AvgPool2d(2), nn.Flatten(), nn.Linear(64, 2),
    )
    config = {"lr": args.lr, "momentum": 0.0, "l2": 0.0, "epochs": args.epochs}
    bit_checks = 0
    nonzero_virtual_gradients = 0

    def evaluate(model):
        loss, accuracy = trainer.test(model, test_loader, None, "cpu")
        require(math.isfinite(loss) and math.isfinite(accuracy), "Non-finite evaluation metric")
        return {"loss": float(loss), "synthetic_accuracy": float(accuracy)}

    def benchmark(method: str):
        nonlocal bit_checks, nonzero_virtual_gradients
        method_started = time.perf_counter()
        model = copy.deepcopy(base)
        if method == "fedbif":
            bif.bif_replace_modules(model)
            # The repository replacement initializes fresh conv weights. Copy
            # the baseline values explicitly to start both methods identically.
            with torch.no_grad():
                model[0].weight.copy_(base[0].weight)
        initial = evaluate(model)
        history = []
        for round_index in range(args.rounds):
            bit_pos = args.bits - 1 - (round_index % args.bits)
            if method == "fedbif":
                torch.manual_seed(args.seed + 10000 + round_index)
                bif.bif_before_server_send(model, args.bits, bit_pos)
            states = []
            for cid, dataset in enumerate(datasets):
                client = copy.deepcopy(model)
                if method == "fedbif":
                    frozen_before = client[0].weight.detach().clone()
                    alpha_before = client[0].alpha.detach().clone()
                    selected_before = (client[0].virtual.detach() > 0).to(torch.int64)
                generator = torch.Generator().manual_seed(args.seed + 20000 + round_index * args.clients + cid)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
                trainer.train(client, loader, test_loader, config, "cpu")
                require(all(torch.isfinite(v).all() for v in client.state_dict().values()),
                        f"Non-finite {method} client state")
                if method == "fedbif":
                    require(torch.equal(client[0].weight, frozen_before), "Frozen convolution weights changed")
                    require(torch.equal(client[0].alpha, alpha_before), "Frozen quantization scale changed")
                    gradient = client[0].virtual.grad
                    require(gradient is not None and torch.isfinite(gradient).all(), "Invalid virtual gradient")
                    nonzero_virtual_gradients += int(bool(torch.count_nonzero(gradient)))
                    selected_after = (client[0].virtual.detach() > 0).to(torch.int64)
                    expected = frozen_before + selected_after * alpha_before
                    bif.bif_before_client_send(client)
                    require(torch.allclose(client[0].weight, expected), "Client assembly changed frozen weights")
                    require(torch.count_nonzero(client[0].virtual) == 0 and client[0].alpha.item() == 0,
                            "Client assembly did not reset virtual state")
                    mask = 1 << bit_pos
                    step = float(alpha_before) / mask
                    if step > 0:
                        offset = step * midpoint
                        frozen_codes = torch.round((frozen_before + offset) / step).to(torch.int64)
                        before_codes = frozen_codes + selected_before * mask
                        after_codes = torch.round((client[0].weight + offset) / step).to(torch.int64)
                        require(torch.all(((before_codes ^ after_codes) & ~mask) == 0),
                                "A nonselected bit changed during local training")
                        require(torch.all((after_codes >= 0) & (after_codes < 2 ** args.bits)),
                                "Assembled integer weights are out of range")
                        bit_checks += after_codes.numel()
                states.append({key: value.detach().clone() for key, value in client.state_dict().items()})
            averaged = {}
            for key, value in states[0].items():
                if value.is_floating_point() or value.is_complex():
                    averaged[key] = torch.stack([state[key] for state in states]).mean(dim=0)
                else:
                    require(all(torch.equal(value, state[key]) for state in states),
                            f"Cannot average unequal non-floating state: {key}")
                    averaged[key] = value.clone()
            model.load_state_dict(averaged, strict=True)
            metrics = evaluate(model)
            history.append({"round": round_index + 1, **metrics,
                            **({"selected_bit": bit_pos} if method == "fedbif" else {})})
            print(f"{method:6s} round {round_index + 1}/{args.rounds}: "
                  f"loss={metrics['loss']:.4f}, synthetic accuracy={metrics['synthetic_accuracy']:.1%}", flush=True)
        return {"initial": initial, "final": history[-1], "history": history,
                "loss_decreased": history[-1]["loss"] < initial["loss"],
                "runtime_seconds": round(time.perf_counter() - method_started, 4)}

    methods = {method: benchmark(method) for method in ("fedavg", "fedbif")}
    require(nonzero_virtual_gradients > 0, "No nonzero virtual gradients observed in FedBiF training")
    require(methods["fedavg"]["initial"] == methods["fedbif"]["initial"], "Initial model outputs differed")
    return {
        "status": "completed", "benchmark": "CPU synthetic smoke test",
        "scope": "Generated left/right stripe images; accuracy is synthetic only. No real-dataset or network-savings measurements.",
        "python_version": sys.version.split()[0], "torch_version": str(torch.__version__),
        "repo": str(args.repo),
        "sources": {"bif": str(Path(bif.__file__).resolve()), "trainer": str(Path(trainer.__file__).resolve())},
        "settings": {key: value for key, value in vars(args).items() if key not in ("repo", "output")},
        "checks": {"zero_quantization": "passed", "straight_through_gradient": "passed",
                   "exact_reconstruction_cases": args.bits * 2 ** args.bits,
                   "single_bit_flip_cases": args.bits * 2 ** args.bits,
                   "local_nonselected_bits_preserved": "passed", "checked_client_weights": bit_checks,
                   "client_updates_with_nonzero_final_virtual_gradient": nonzero_virtual_gradients,
                   "identical_initial_outputs": True},
        "methods": methods, "runtime_seconds": round(time.perf_counter() - started, 4),
    }


def main() -> None:
    args = arguments()
    result = run(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
