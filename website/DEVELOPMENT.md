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
mamba run -n mrnabench python -m pip install -e .
mamba run -n mrnabench npm --prefix website install
```

The editable Python install is needed because the leaderboard builder reads the
current dataset registry when it selects default splits.

Start the local server:

```bash
mamba run -n mrnabench npm --prefix website run dev
```

To keep the preview inside VS Code, run **Simple Browser: Show** from the
Command Palette and open `http://localhost:4321/`. In a remote session, forward
port `4321` from the **Ports** panel first.

If VS Code and Copilot are running on different HPC compute nodes, start a
lightweight static preview from the VS Code terminal:

```bash
bash website/start-vscode-preview.sh
```

Run this before entering a separate Slurm allocation. The script builds once
and then uses Python's small static server on port `4321`; it does not keep the
Astro development server running on the VS Code node.

Build for the repository's GitHub Pages base path:

```bash
SITE_BASE=/mRNABench \
  mamba run -n mrnabench npm --prefix website run build
```

The build copies only the approved web logo files from `assets/`; it does not
modify the source design pack.

## Update the leaderboard

The source artifact is:

```text
paper/leaderboard_results.parquet
```

Every development or production build runs:

```bash
mamba run -n mrnabench python website/scripts/build_leaderboard.py
```

The script validates the parquet schema, keeps each dataset's declared default
split, normalizes legacy regression names, resolves duplicate legacy rows, and
writes browser data below `website/public/data/`. Those generated files are
ignored by Git because they are rebuilt from the parquet.

Generated metadata includes the source parquet SHA-256, the effective split map,
and the GitHub commit when built in Actions. The CSV uses
`result_release_id`; it does not claim that one release label is a dataset
version.

Before replacing the artifact, update `website/leaderboard.config.json` with
the release label and any datasets or model checkpoints that must be excluded.
The configuration labels the input as the May 2026 linear-probe archive and
excludes six datasets rebuilt after the recorded runs. Do not remove those
exclusions until the replacement parquet contains results from the rebuilt
datasets.

The config also selects one regression result task. Keep OLS and Ridge
leaderboards separate; the data builder rejects a task group that mixes
evaluation protocols. It also records an explicit source-task priority and
fails when overlapping legacy regression rows disagree beyond the configured
tolerance.

The consensus rank is available only to models with complete classification,
multilabel, and regression coverage. Zero-shot results require a separate
artifact with full scoring-protocol fields and must not be mixed into this
linear-probe ranking.
