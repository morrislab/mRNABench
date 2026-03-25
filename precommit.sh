flake8 mrna_bench --ignore=D100,D104,E203
mypy mrna_bench --ignore-missing-imports

# Require GPU
pytest
