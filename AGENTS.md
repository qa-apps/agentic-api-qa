# Project commands

- Python: `python3.11`
- Install fast tests/dev: `.venv/bin/python -m pip install -e '.[test,dev]'`
- Install evaluation layer: `.venv/bin/python -m pip install -e '.[test,evals]'`
- Fast tests (no paid evals): `.venv/bin/python -m pytest -q`
- Deterministic decision evals: `.venv/bin/python -m evals.run_evals --suite decisions --no-deepeval --no-slack`
- Full nightly evals: `.venv/bin/python -m evals.run_evals --suite all --mutations 3 --seed 1337`
- LangGraph Studio: `.venv/bin/langgraph dev --no-browser`

Seeded bugs must run only against the localhost test copy created by `evals.seeded_site`, never against production. DeepEval tests are outside the default pytest `testpaths`; run them explicitly with `pytest evals` or `python -m evals.run_evals`.
