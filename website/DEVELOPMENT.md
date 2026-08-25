# Website development

The site is an Astro static build in `website/`. It is designed for GitHub
Pages, but it runs locally without a repository base path.

## Environment

Development uses the existing `mrnabench` environment with Node 22:

```bash
mamba install -n mrnabench -c conda-forge "nodejs>=22,<23" -y
```

Install the checked-in JavaScript dependencies:

```bash
mamba run -n mrnabench python -m pip install .
mamba run -n mrnabench npm --prefix website install
```

The Python install provides the dependencies used while generating website
data.

Start the local server:

```bash
mamba run -n mrnabench npm --prefix website run dev
```

Open `http://localhost:4321/` in a browser. Forward port `4321` when the server
runs on a remote machine.

Build for the repository's GitHub Pages base path:

```bash
SITE_BASE=/mRNABench \
  mamba run -n mrnabench npm --prefix website run build
```

The build copies the required web logo files from `assets/`.

## Generate website data

Every development and production build runs:

```bash
mamba run -n mrnabench npm --prefix website run prepare-data
```

This command validates and generates the leaderboard JSON and CSV, then builds
the Python API reference from package signatures and docstrings. Generated
files are written below ignored website data directories.
