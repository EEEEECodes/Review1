"""Repair the known FedBiF checkout blockers using only Python's standard library.

Run: python repair_imports.py --repo PATH_TO_FEDBIF
Add --check to validate and preview changes without writing files.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import uuid


def replace_known(text, old, new, count, label):
    """Accept either the original or fully repaired snippet; refuse drift."""
    if text.count(old) == count:
        return text.replace(old, new)
    if old not in text and text.count(new) == count:
        return text
    raise ValueError(f"Unexpected source for {label}; no files were changed.")


def repair_bif(text):
    missing_import = "from model.densenet import DenseBottleneck, Transition, DenseBlock\n"
    if text.count(missing_import) == 1:
        text = text.replace(missing_import, "")
    elif "model.densenet" in text:
        raise ValueError("Unexpected DenseNet import; no files were changed.")
    text = replace_known(
        text,
        "(Sequential, Bottleneck, BasicBlock, DenseBottleneck, Transition, DenseBlock)",
        "(Sequential, Bottleneck, BasicBlock)", 3, "supported module recursion",
    )
    text = replace_known(
        text,
        "    max_range = weight.abs().max()\n\n    alpha",
        "    max_range = weight.abs().max()\n"
        "    if max_range.item() == 0:\n"
        "        return torch.zeros_like(weight), torch.zeros_like(weight), torch.zeros_like(max_range)\n"
        "\n    alpha",
        1, "zero-weight quantization",
    )
    text = replace_known(
        text, "            self.alpha.data = torch.tensor(0.0)",
        "            self.alpha.zero_()", 2, "device-safe alpha reset",
    )
    return replace_known(
        text,
        '    weight_q = stochastic_quantize(weight, 4, 2, "max")\n'
        "    print(weight_q == weight)\n",
        "    remain, virtual, alpha = stochastic_quantize(weight, 4, 2)\n"
        "    print(remain + virtual_to_binary(virtual, alpha))\n",
        1, "module demonstration",
    )


def repair_models(text):
    text = replace_known(
        text, "from model.cnn import CNN_SVHN, CNN_CIFAR, CNN_MNIST",
        "from model.cnn import CNN_SVHN, CNN_MNIST", 1, "CNN imports",
    )
    helper = (
        "def _cifar_cnn(num_classes):\n"
        "    try:\n"
        "        from model.cnn import CNN_CIFAR\n"
        "    except ImportError as exc:\n"
        '        raise ValueError("CNN_CIFAR is absent from this checkout; use resnet18 for CIFAR.") from exc\n'
        "    return CNN_CIFAR(num_classes)\n\n\n"
    )
    if "def _cifar_cnn(" not in text:
        if text.count("def _weight_init(model):") != 1:
            raise ValueError("Unexpected model factory; no files were changed.")
        text = text.replace("def _weight_init(model):", helper + "def _weight_init(model):")
    elif helper not in text:
        raise ValueError("Unexpected CIFAR helper; no files were changed.")
    for classes in (10, 100):
        text = replace_known(
            text, f"model = CNN_CIFAR({classes})", f"model = _cifar_cnn({classes})",
            1, f"CIFAR-{classes} model factory",
        )
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate and preview without writing")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    changes = []
    for relative, transform in (("plugin/bif.py", repair_bif), ("plugin/models.py", repair_models)):
        path = (repo / relative).resolve(strict=True)
        if repo not in path.parents:
            raise ValueError(f"Source escapes the repository: {relative}")
        original = path.read_bytes()
        newline = "\r\n" if b"\r\n" in original else "\n"
        source = original.decode("utf-8").replace("\r\n", "\n")
        revised = transform(source)
        compile(revised, str(path), "exec")
        output = revised.replace("\n", newline).encode("utf-8")
        if output != original:
            changes.append((path, original, output))
    if not changes:
        print("Known repairs already applied; both source files compile.")
        return
    if args.check:
        for path, _, _ in changes:
            print(f"Would repair: {path}")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for path, original, _ in changes:
        backup = path.with_name(f"{path.name}.{stamp}.bak")
        with backup.open("xb") as handle:
            handle.write(original)
        print(f"Backup: {backup}")
    for path, _, output in changes:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(output)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Repaired: {path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, UnicodeError, ValueError, SyntaxError) as error:
        print(f"Repair failed: {error}", file=sys.stderr)
        sys.exit(1)
