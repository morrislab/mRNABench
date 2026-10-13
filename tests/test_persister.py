from types import SimpleNamespace

import pytest

from mrna_bench.persister import ResultsPersister, metrics_by_split


@pytest.fixture
def dataset(tmp_path) -> SimpleNamespace:
    """Dataset stub whose results.db lives in a temporary directory."""
    return SimpleNamespace(
        dataset_name="test",
        dataset_path=str(tmp_path),
        checkpoint_dir=str(tmp_path),
    )


def make_persister(dataset, probe_config=None) -> ResultsPersister:
    """Persister for one linear probe configuration."""
    return ResultsPersister(
        dataset,
        "model",
        "linear_probe",
        "regression",
        "target",
        "default",
        {"pooling": {"name": "MeanPool"}},
        probe_config or {"estimator": "RidgeCV()"},
    )


def test_persist_and_load_run(dataset):
    """Metrics and fit info round-trip through the database."""
    persister = make_persister(dataset)
    assert not persister.result_exists(0)

    persister.persist_run_results(
        {"test": {"r": 0.5}, "val": {"r": 0.4}}, 0, fit_info={"alpha": 10.0}
    )

    assert persister.result_exists(0)
    assert persister.load_run_results(0) == {
        "metrics": {"test": {"r": 0.5}, "val": {"r": 0.4}},
        "fit_info": {"alpha": 10.0},
    }


def test_rerun_replaces_metrics(dataset):
    """Persisting the same configuration again overwrites its metrics."""
    persister = make_persister(dataset)
    persister.persist_run_results({"test": {"r": 0.5, "mse": 1.0}}, 0)
    persister.persist_run_results({"test": {"r": 0.6}}, 0)

    assert persister.load_run_results(0)["metrics"] == {"test": {"r": 0.6}}
    assert len(ResultsPersister.load_all_results(dataset.dataset_path)) == 1


def test_probe_configs_store_separate_runs(dataset):
    """Runs differing only in probe config do not overwrite each other."""
    low = make_persister(dataset, {"lora_alpha": 8})
    high = make_persister(dataset, {"lora_alpha": 32})
    low.persist_run_results({"val": {"loss": 0.5}}, 0)
    high.persist_run_results({"val": {"loss": 0.3}}, 0)

    assert low.load_run_results(0)["metrics"]["val"]["loss"] == 0.5
    assert high.load_run_results(0)["metrics"]["val"]["loss"] == 0.3
    assert low.get_checkpoint_path(0) != high.get_checkpoint_path(0)

    results = ResultsPersister.load_all_results(dataset.dataset_path)
    assert sorted(r["probe_config"]["lora_alpha"] for r in results) == [8, 32]


def test_metrics_by_split():
    """Flat metric names are grouped by their split prefix."""
    assert metrics_by_split({"test_auprc_macro": 0.7, "val_r": 0.1}) == {
        "test": {"auprc_macro": 0.7},
        "val": {"r": 0.1},
    }
