import pytest

from imdb_mlops.data.faults import FAULTS, inject_fault
from imdb_mlops.data.quality import DataQualityError, validate


def test_clean_batch_passes(episodes_df, manifest, cfg):
    report = validate(episodes_df, manifest, cfg, declared_ep_count=len(episodes_df))
    assert report.success, report.summary()
    assert not report.failed


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_each_fault_is_blocked(episodes_df, manifest, cfg, fault):
    bad = inject_fault(episodes_df, fault)
    with pytest.raises(DataQualityError) as exc:
        validate(
            bad,
            manifest,
            cfg,
            previous_n_rows=len(episodes_df),
            declared_ep_count=len(episodes_df),
        )
    assert exc.value.report.failed, f"le défaut {fault} aurait dû être détecté"


def test_row_count_drift_vs_previous_snapshot(episodes_df, manifest, cfg):
    report = validate(
        episodes_df, manifest, cfg, previous_n_rows=len(episodes_df) * 2, raise_on_failure=False
    )
    failed = {c.name for c in report.failed}
    assert "row_count_change_vs_previous_snapshot" in failed


def test_warning_does_not_block(episodes_df, manifest, cfg):
    shifted = episodes_df.assign(rating=1.5)  # moyenne hors plage → warning seulement
    report = validate(shifted, manifest, cfg, raise_on_failure=False)
    assert report.success
    assert any("mean" in c.name for c in report.warnings)


def test_report_written(episodes_df, manifest, cfg, tmp_path):
    validate(episodes_df, manifest, cfg)
    assert (tmp_path / cfg.data.reports_dir / f"quality_{manifest.sha}.json").exists()


def test_unknown_fault_rejected(episodes_df):
    with pytest.raises(ValueError):
        inject_fault(episodes_df, "nope")
