import datetime

import earthkit.data as ekd
import numpy as np
import pytest

from anemoi_plugins_meteoswiss.outputs import ZeroStepFromTemplate

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
    accumulations = ["2t"]


@pytest.fixture
def output(data_dir, tmp_path):
    return ZeroStepFromTemplate(
        FakeContext(),
        FakeMetadata(),
        path=str(tmp_path / "out.grib"),
        template_path=str(data_dir / "iaf2025010100"),
    )


def test_template_index_keyed_by_param(output):
    assert set(output.template_index) == {"2t", "t", "wz"}


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
    assert values.shape == (output.template_index["2t"].shape[0],)
    assert np.all(values == 0)

    # Metadata not explicitly overridden is inherited from the template.
    template = output.template_index["2t"]
    assert field.metadata("gridType") == template.metadata("gridType")
    assert field.metadata("edition") == template.metadata("edition")


def test_write_initial_state_skips_field_already_present(output):
    state = {"date": REFERENCE_DATE, "fields": {"2t": np.zeros(1)}, "step": datetime.timedelta(0)}
    output.write_initial_state(state)
    output.close()

    # Nothing was written, so the file was never even created.
    assert not output.out.exists()


def test_write_step_is_a_noop(output):
    state = {"date": REFERENCE_DATE, "fields": {"2t": np.zeros(1)}, "step": datetime.timedelta(hours=1)}
    output.write_step(state)
    output.close()

    assert not output.out.exists()
