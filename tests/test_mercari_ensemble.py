"""Selection-only model-count comparisons and guarded ensemble assembly."""

import json
import numpy as np
import pytest

from benchmarks.mercari_ensemble import compare_counts, combine, load_experts


def test_count_comparison_finds_complementary_pair_without_reading_audit_labels():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    candidates = {
        "left": y + [0.4, -0.4, 0.4, -0.4],
        "right": y + [-0.4, 0.4, -0.4, 0.4],
        "weak": y + 2.0,
    }
    rows = compare_counts(candidates, y, max_models=3)
    assert [r["count"] for r in rows] == [1, 2, 3]
    assert (
        rows[1]["selection_rmsle"] < 0.001
    )  # Small anchor penalty permits a tiny residual.
    assert set(rows[1]["names"]) == {"left", "right"}
    for row in rows:
        assert sum(row["weights"]) == pytest.approx(1.0)
        assert min(row["weights"]) >= 0.0
        assert row["selection_rmsle"] == pytest.approx(
            np.sqrt(np.mean((combine(candidates, row) - y) ** 2))
        )


@pytest.mark.parametrize(
    "candidates,y,maximum",
    [
        ({}, [1.0], 1),
        ({"x": [np.nan]}, [1.0], 1),
        ({"x": [1.0, 2.0]}, [1.0], 1),
        ({"x": [1.0]}, [np.inf], 1),
        ({"x": [1.0]}, [1.0], 0),
    ],
)
def test_invalid_study_inputs_fail(candidates, y, maximum):
    with pytest.raises(ValueError):
        compare_counts(candidates, np.array(y), max_models=maximum)


def test_named_prediction_assembly_rejects_missing_weights_and_shapes():
    for bad in [
        dict(names=["missing"], weights=[1.0]),
        dict(names=["x"], weights=[-1.0]),
        dict(names=["x"], weights=[0.5]),
        dict(names=["x", "x"], weights=[0.5, 0.5]),
    ]:
        with pytest.raises(ValueError):
            combine({"x": np.ones(3)}, bad)
    with pytest.raises(ValueError):
        combine(
            {"x": np.ones(2), "y": np.ones(3)},
            dict(names=["x", "y"], weights=[0.5, 0.5]),
        )


def test_loads_only_named_predictions_and_requires_identical_ordered_ids(tmp_path):
    ids = dict(
        calibration_ids=np.arange(2),
        selection_ids=np.arange(3),
        confirmation_ids=np.arange(4),
    )
    for name in ["a", "b"]:
        np.savez(
            tmp_path / (name + ".npz"),
            **ids,
            calibration=np.ones(2),
            selection=np.ones(3),
            confirmation=np.ones(4),
        )
    arrays, actual = load_experts(tmp_path, ["a", "b"])
    assert list(arrays) == ["a", "b"]
    np.testing.assert_array_equal(actual["selection"], ids["selection_ids"])
    np.savez(
        tmp_path / "b.npz",
        **dict(ids, selection_ids=np.arange(3)[::-1]),
        calibration=np.ones(2),
        selection=np.ones(3),
        confirmation=np.ones(4),
    )
    with pytest.raises(ValueError, match="IDs"):
        load_experts(tmp_path, ["a", "b"])
    with pytest.raises(ValueError):
        load_experts(tmp_path, ["a", "a"])
    with pytest.raises(ValueError):
        load_experts(tmp_path, ["evaluation"])


def release_inputs():
    from hashlib import sha256
    config=dict(outer=dict(names=['model'],weights=[1.]),pretrained_weight_sha256='weights')
    report=dict(smoke=False,source_archive_sha256='source',pretrained_weight_sha256='weights',
        protocol=dict(train_sha256='94b24fbecca842140b6276e9da43ce66be3b56b4b991375228802a74050fe471',no_selection_or_confirmation_labels=True,
                      splits=dict(base=dict(rows=1395326),calibration=dict(rows=87209))),
        stages={n+'-report':{} for n in ['prepare','baseline','lsa','sequence','separate','joint','stemmed','vectors','transformer','finish']},
        submission=dict(rows=3460725,no_final_parameter_selection=True,outer=config['outer'],sha256='a'*64,
                        frozen_config_sha256=sha256(json.dumps(config,indent=2).encode()).hexdigest()))
    return config,report


def test_release_gate_accepts_only_exact_frozen_complete_full_run():
    from benchmarks.mercari_ensemble import validate_hosted
    c,r=release_inputs()
    assert validate_hosted(r,c,'source')['rows']==3460725


@pytest.mark.parametrize('change',['smoke','rows','source','weights','pretrained','config','labels','training','stages','hash'])
def test_release_gate_rejects_wrong_or_partial_runs(change):
    from benchmarks.mercari_ensemble import validate_hosted
    c,r=release_inputs()
    if change=='smoke':r['smoke']=True
    elif change=='rows':r['submission']['rows']=7
    elif change=='source':r['source_archive_sha256']='wrong'
    elif change=='weights':r['submission']['outer']=dict(names=['inferior'],weights=[1.])
    elif change=='pretrained':r['pretrained_weight_sha256']='wrong'
    elif change=='config':r['submission']['frozen_config_sha256']='wrong'
    elif change=='labels':r['protocol']['no_selection_or_confirmation_labels']=False
    elif change=='training':r['protocol']['splits']['base']['rows']=100
    elif change=='stages':r['stages'].pop('finish-report')
    else:r['submission']['sha256']='missing'
    with pytest.raises(ValueError):validate_hosted(r,c,'source')
