# Quick concept run

## Fix setup and run actual Python training

From `C:\Users\User\E\Review1`, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\concept-results\setup-fedbif.ps1
```

This command installs Python 3.11 x64 if needed using WinGet, creates
`fedbif-tpds25\.venv`, installs the repository's pinned dependencies with CPU-only
PyTorch, and repairs the known source blockers. It backs up each changed source
file beside the original as `filename.TIMESTAMP.bak`. Existing virtual
environments must already use Python 3.11 x64; the script never deletes them.
Keep `setup-fedbif.ps1`, `repair_imports.py`, and `quick_train.py` together.

It then runs the repository's trainer and bit-freezing hooks on a tiny generated
image task: two simulated clients and four rounds each of FedAvg and FedBiF.
There is no Flower networking or real dataset download in this smoke test.
Loss and synthetic test accuracy are written to `training-results.json`.
These are smoke-test results, not the paper's benchmark results or measured
communication savings. Dependency installation may take several minutes.

To rerun without installing, or use an existing Python installation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\concept-results\setup-fedbif.ps1 -SkipInstall
powershell -NoProfile -ExecutionPolicy Bypass -File .\concept-results\setup-fedbif.ps1 -PythonExe C:\Python311\python.exe
```

`-SkipInstall` requires the existing project `.venv`; it does not create a new
environment or install missing dependencies. `-PythonExe` selects the base
interpreter when initially creating that environment.

`-ValidateOnly` checks PowerShell syntax and required file paths without installing
or running Python. That check passed during preparation; Python runtime tests
could not be run because the current environment has no usable interpreter.
`-SkipSmoke` prepares and checks the environment without running training.
For another checkout, pass `-RepoPath C:\path\to\fedbif-tpds25`.

The repair removes unused missing DenseNet references and makes the unavailable
CIFAR CNN optional. Selecting that missing CNN reports an explicit error;
`resnet18` remains supported for CIFAR. It also handles zero-weight quantization,
keeps alpha resets on the correct device, and corrects the plugin's demo call.
Unexpected edits to the patched snippets cause the helper to stop. Source
changes are syntax-checked before either source file is written.

If setup fails, the script stops and prints the failing command. Resolve the
reported issue and rerun the same command; installation and source repairs can
be repeated. If WinGet is unavailable or cannot execute, download and run the
Windows installer (64-bit) from the official [Python 3.11.9 release page](https://www.python.org/downloads/release/python-3119/).
Install for the current user, then run this from `C:\Users\User\E\Review1`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\concept-results\setup-fedbif.ps1 -PythonExe "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
```

If you selected a different installation folder, substitute its `python.exe`.
CPU package commands follow the official [PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/#v220).

## Initial PowerShell arithmetic demo

`quick-concept.ps1` ports the scalar arithmetic from `fedbif-tpds25/plugin/bif.py`:
stochastic 4-bit quantization, separation of one selected bit, and reconstruction.
It synthetically toggles the selected bit for half of 16 weights, then checks
that the other three bits remain unchanged. This does not execute the original
Python code, train a model, evaluate accuracy, or measure network compression.
The random generator differs from PyTorch's.

Run without installing dependencies, from `C:\Users\User\E\Review1`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\concept-results\quick-concept.ps1
```

The script writes `results.json` beside itself.

## Full experiment blockers found before setup

- The available `python.exe` is an inaccessible WindowsApps alias; no usable
  Python was found on PATH or in the usual local Python/Conda directories.
- `plugin/bif.py` imports `model.densenet`, which is absent from this checkout.
- `plugin/models.py` imports `CNN_CIFAR`, which `model/cnn.py` does not define.
- No datasets or generated client partitions are included. The README's
  `dataloader/datapartition.py` path should be `dataloaders/datapartition.py`.
- Flower clients return full NumPy state dictionaries; ideal bit-count savings
  must not be reported as measured network savings.

No software or data was installed or downloaded during preparation. The original
repository was not changed. To prepare an environment manually, run these commands from
`C:\Users\User\E\Review1\fedbif-tpds25` (the first installs Python if needed):

```powershell
winget install --id Python.Python.3.11 --exact --source winget
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The source import errors still need repair before a full experiment can run.
