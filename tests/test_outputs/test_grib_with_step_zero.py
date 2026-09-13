import datetime

import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.inference.testing.variables import z as VAR_Z

from anemoi_plugins_meteoswiss.outputs import GribWithStepZero

REFERENCE_DATE = datetime.datetime(2026, 8, 31, 0, 0)


class FakeContext:
    reference_date = REFERENCE_DATE
    write_initial_state = True
    output_frequency = None
    allow_nans = False
    typed_variables: dict = {}


class FakeMetadata:
    dataset_name = "test"
    typed_variables: dict = {}
    variables_metadata: dict = {}
    accumulations = ["2t"]
    number_of_grid_points = None
    grid = None
    area = None


@pytest.fixture
def output(data_dir, tmp_path):
    return GribWithStepZero(
        FakeContext(),
        FakeMetadata(),
        path=str(tmp_path / "out.grib"),
        step_zero_template=str(data_dir / "iaf2025010100"),
    )


def test_step_zero_template_index_keyed_by_param(output):
    assert set(output.step_zero_template_index) == {"2t", "t", "wz"}


def test_write_initial_state_emits_zero_field_from_template(output, data_dir):
    state = {"date": REFERENCE_DATE, "fields": {}, "step": datetime.timedelta(0)}
    output.write_initial_state(state)
    output.close()

    written = list(ekd.from_source("file", output.out))
    assert len(written) == 1

    field = written[0]
    assert field.metadata("shortName") == "2t"
    assert field.metadata("step") == 0
    assert field.metadata("dataDate") == 20260831
    assert field.metadata("dataTime") == 0

    values = field.to_numpy(flatten=True)
    assert values.shape == (output.step_zero_template_index["2t"].shape[0],)
    assert np.all(values == 0)

    # Metadata not explicitly overridden is inherited from the template.
    template = output.step_zero_template_index["2t"]
    assert field.metadata("gridType") == template.metadata("gridType")
    assert field.metadata("edition") == template.metadata("edition")


def test_write_initial_state_resolves_mars_param(data_dir, tmp_path):
    """When the anemoi variable name differs from the GRIB param (e.g. 'tp'
    vs 'TOT_PREC'), the template must be looked up by the resolved mars
    param, not the raw anemoi name."""

    class FakeMetadataWithMars(FakeMetadata):
        accumulations = ["total_precip"]
        variables_metadata = {"total_precip": {"mars": {"param": "t"}}}

    output = GribWithStepZero(
        FakeContext(),
        FakeMetadataWithMars(),
        path=str(tmp_path / "out.grib"),
        step_zero_template=str(data_dir / "iaf2025010100"),
    )
    state = {"date": REFERENCE_DATE, "fields": {}, "step": datetime.timedelta(0)}
    output.write_initial_state(state)
    output.close()

    written = list(ekd.from_source("file", output.out))
    assert len(written) == 1
    assert written[0].metadata("shortName") == "t"


def test_write_initial_state_emits_one_message_per_accumulation(data_dir, tmp_path):
    """Multiple accumulated variables each get their own zero-valued step=0
    message, resolved independently from the same template file(s)."""

    class FakeMetadataWithMultiple(FakeMetadata):
        accumulations = ["2t", "total_precip"]
        variables_metadata = {"total_precip": {"mars": {"param": "t"}}}

    output = GribWithStepZero(
        FakeContext(),
        FakeMetadataWithMultiple(),
        path=str(tmp_path / "out.grib"),
        step_zero_template=str(data_dir / "iaf2025010100"),
    )
    state = {"date": REFERENCE_DATE, "fields": {}, "step": datetime.timedelta(0)}
    output.write_initial_state(state)
    output.close()

    written = list(ekd.from_source("file", output.out))
    assert {f.metadata("shortName") for f in written} == {"2t", "t"}
    assert all(f.metadata("step") == 0 for f in written)


def test_write_initial_state_skips_field_already_present(output):
    state = {
        "date": REFERENCE_DATE,
        "fields": {"2t": np.zeros(1)},
        "step": datetime.timedelta(0),
    }
    # Scoped to the zero-step logic only: going through `write_initial_state`
    # here would also exercise GribFileOutput's own real-field writing path,
    # which needs a resolvable `typed_variables["2t"]` that isn't relevant to
    # what this test is checking.
    output._write_zero_step_messages(state)
    output.close()

    # Nothing was written, so the file was never even created.
    assert not output.out.exists()


def test_write_initial_state_writes_real_and_zero_step_fields_to_same_file(output, data_dir):
    """The whole point of this output: a real field (written by the
    inherited GribFileOutput logic) and the synthetic zero-step field both
    end up in the same GRIB file."""
    templates = {f.metadata("param"): f for f in ekd.from_source("file", data_dir / "iaf2025010100")}
    real_template = templates["t"]

    output.typed_variables = {"z": VAR_Z}
    state = {
        "date": REFERENCE_DATE,
        "fields": {"z": np.zeros(real_template.shape)},
        "step": datetime.timedelta(0),
        "_grib_templates_for_output": {"z": real_template},
    }
    output.write_initial_state(state)
    output.close()

    written_params = {f.metadata("shortName") for f in ekd.from_source("file", output.out)}
    assert written_params == {"z", "2t"}
