import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.sources.array_list import ArrayField

from anemoi_plugins_meteoswiss.transform.filters import HeightAboveGround
from anemoi_plugins_meteoswiss.transform.filters import SoilDepthLevels


def _field(values, **metadata):
    metadata = {"valid_datetime": "2020-01-01T00:00:00", **metadata}
    metadata.setdefault("param", metadata["shortName"])
    return ArrayField(np.asarray(values, dtype=float), RawMetadata(metadata))


def test_height_above_ground():
    fieldlist = new_fieldlist_from_list(
        [
            _field([1000.0, 2000.0, 3000.0], shortName="HHL", level=79),
            _field([110.0, 510.0, 1010.0], shortName="HHL", level=80),
            _field([100.0, 500.0, 1000.0], shortName="HSURF"),
            _field([1.0, 2.0, 3.0], shortName="T", level=80),
        ]
    )

    out = HeightAboveGround(param=["HHL"]).forward(fieldlist)

    names = [
        (
            f.metadata("shortName"),
            f.metadata("param"),
            f.metadata("level", default=None),
        )
        for f in out
    ]
    # HSURF is consumed, the heights above sea level are kept, T passes through
    assert sorted(names, key=str) == sorted(
        [
            ("HHL", "HHL", 79),
            ("HHL", "HHL", 80),
            ("T", "T", 80),
            ("HHL_AGL", "HHL_AGL", 79),
            ("HHL_AGL", "HHL_AGL", 80),
        ],
        key=str,
    )
    agl = {
        f.metadata("level"): f.values
        for f in out
        if f.metadata("shortName") == "HHL_AGL"
    }
    np.testing.assert_array_equal(agl[79], [900.0, 1500.0, 2000.0])
    np.testing.assert_array_equal(agl[80], [10.0, 10.0, 10.0])


def test_height_above_ground_replace_and_errors():
    fieldlist = new_fieldlist_from_list(
        [
            _field([110.0, 510.0], shortName="HHL", level=80),
            _field([100.0, 500.0], shortName="HSURF"),
        ]
    )
    out = HeightAboveGround(param="HHL", name="{param}", keep=False).forward(fieldlist)
    assert [f.metadata("shortName") for f in out] == ["HHL"]
    np.testing.assert_array_equal(out[0].values, [10.0, 10.0])

    with pytest.raises(ValueError, match="different from the input"):
        HeightAboveGround(param="HHL", name="{param}", keep=True)

    no_surface = new_fieldlist_from_list(
        [_field([110.0, 510.0], shortName="HHL", level=80)]
    )
    with pytest.raises(ValueError, match="No HSURF field"):
        HeightAboveGround(param="HHL").forward(no_surface)


def _soil_field(name, value, factor, type_of_level="depthBelowLand"):
    return _field(
        [1.0, 2.0],
        shortName=name,
        typeOfLevel=type_of_level,
        level=int(
            value * 10.0**-factor
        ),  # what eccodes reports: truncated to whole metres
        scaledValueOfFirstFixedSurface=value,
        scaleFactorOfFirstFixedSurface=factor,
    )


def test_soil_depth_levels():
    fieldlist = new_fieldlist_from_list(
        [
            _soil_field("T_SO", 0, 2),
            _soil_field("T_SO", 5, 3),
            _soil_field("T_SO", 2, 2),
            _soil_field("T_SO", 486, 2),
            _soil_field("W_SO", 1, 2, "depthBelowLandLayer"),
            _soil_field("W_SO", 729, 2, "depthBelowLandLayer"),
            _field(
                [1.0, 2.0], shortName="T_2M", typeOfLevel="heightAboveGround", level=2
            ),
        ]
    )

    out = SoilDepthLevels().forward(fieldlist)

    assert [
        (f.metadata("shortName"), f.metadata("levelist", default=None)) for f in out
    ] == [
        ("T_SO", 0),
        ("T_SO", 5),
        ("T_SO", 20),
        ("T_SO", 4860),
        ("W_SO", 10),
        ("W_SO", 7290),
        ("T_2M", None),
    ]
    assert [f.metadata("level") for f in out] == [0, 5, 20, 4860, 10, 7290, 2]
    assert all(f.metadata("levtype") == "dp" for f in out[:6])
    np.testing.assert_array_equal(out[1].values, [1.0, 2.0])


def test_soil_depth_levels_mars_namespace():
    field = SoilDepthLevels().forward(
        new_fieldlist_from_list([_soil_field("T_SO", 5, 3)])
    )[0]
    mars = field.metadata(namespace="mars")
    assert mars["levtype"] == "dp"
    assert mars["levelist"] == 5


def test_soil_depth_levels_selected_param_and_bad_depth():
    fieldlist = new_fieldlist_from_list(
        [_soil_field("T_SO", 5, 3), _soil_field("W_SO", 1, 2, "depthBelowLandLayer")]
    )
    out = SoilDepthLevels(param="W_SO").forward(fieldlist)
    assert [f.metadata("level") for f in out] == [0, 10]

    with pytest.raises(ValueError, match="whole number of millimetres"):
        SoilDepthLevels().forward(new_fieldlist_from_list([_soil_field("T_SO", 5, 4)]))


def test_filters_registered():
    from anemoi.transform.filters import filter_registry

    assert isinstance(
        filter_registry.create("height-above-ground", param=["HHL"]), HeightAboveGround
    )
    assert isinstance(filter_registry.create("soil-depth-levels"), SoilDepthLevels)
    assert ekd  # imported for the fieldlist types
