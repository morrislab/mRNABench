flake8 mrna_bench --ignore=D100,D104
mypy mrna_bench --ignore-missing-imports

# Require GPU
pytest -k "not omnigenome"
pytest -k "omnigenome"
