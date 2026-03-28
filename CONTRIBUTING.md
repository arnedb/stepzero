# Contributing to stepzero

Thanks for your interest. Contributions are welcome — bug fixes, new tasks, better defaults, or documentation improvements.

## Branches

- `main` — stable, released code. Direct pushes are not allowed.
- `develop` — integration branch. All PRs should target `develop`, not `main`.

`main` is only updated by merging `develop` when cutting a release.

## Getting started

```bash
git clone https://github.com/arnedb/stepzero
cd stepzero
uv venv && uv pip install -e ".[dev]"
```

Run the tests:

```bash
pytest tests/ -v
```

## Workflow

1. Fork the repo and create a branch from `develop`:
   ```bash
   git checkout develop
   git checkout -b your-feature-name
   ```

2. Make your changes. Add or update tests if relevant.

3. Run the test suite before opening a PR.

4. Open a pull request targeting `develop`, not `main`.

## Adding a new task

Each task lives in `stepzero/tasks/<task_name>.py` and follows the same structure:

- Accept standard inputs (`X`, `y`, or `series`)
- Build pipelines or models to compare
- Run cross-validation via `_runner.run_cv()`
- Compute a headroom signal via `_headroom.compute_headroom()`
- Return a typed result dataclass defined in `_types.py`

Add the new result type to `_types.py`, the task function to `stepzero/tasks/`, wire it up in both `__init__.py` files, and add smoke tests.

## Reporting issues

Open an issue on GitHub. Include:

- Python version and OS
- stepzero version (`import stepzero; print(stepzero.__version__)`)
- A minimal reproducible example

## Releases

Releases are made from `main` by pushing a version tag:

```bash
git tag v0.2.0
git push origin v0.2.0
```

This triggers the publish workflow which builds the package, runs the full test matrix, and publishes to PyPI.

Before tagging, bump the version in `pyproject.toml` and `stepzero/__init__.py`.
