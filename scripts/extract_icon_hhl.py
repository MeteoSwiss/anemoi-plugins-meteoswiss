"""Write the ICON interface heights (HHL) of a GRIB file to the NetCDF file read
by ``DeriveDiagnostic``'s ``vertical_grid_file`` (``cape_cin_method: icon``).

HHL is the geometric height (GRIB2 discipline 0, category 3, number 6) on the
generalized vertical coordinate, e.g. in an ICON analysis (``laf*``) or a
constants file. The output variable ``HHL`` has shape (interfaces, cells),
top-down, in the cell order of the GRIB file.

Usage::

    python scripts/extract_icon_hhl.py laf2022033012 icon-1_HHL.nc
"""

import argparse
from pathlib import Path

import eccodes
import numpy as np
import xarray as xr

HHL_KEYS = {"discipline": 0, "parameterCategory": 3, "parameterNumber": 6, "typeOfFirstFixedSurface": 150}


def read_hhl(path: Path) -> np.ndarray:
    levels = {}
    with path.open("rb") as stream:
        while (handle := eccodes.codes_grib_new_from_file(stream)) is not None:
            try:
                if eccodes.codes_get(handle, "edition") == 2 and all(
                    eccodes.codes_get(handle, key, int) == value for key, value in HHL_KEYS.items()
                ):
                    levels[eccodes.codes_get(handle, "level")] = eccodes.codes_get_values(handle)
            finally:
                eccodes.codes_release(handle)
    if not levels:
        raise ValueError(f"No HHL field in {path}.")
    return np.stack([levels[level] for level in sorted(levels)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grib", type=Path, help="GRIB file containing HHL")
    parser.add_argument("output", type=Path, help="NetCDF file to write")
    args = parser.parse_args()

    hhl = read_hhl(args.grib)
    xr.Dataset(
        {"HHL": (("interface", "cell"), hhl, {"units": "m", "long_name": "geometric height of the half levels"})},
        attrs={"source": str(args.grib.resolve())},
    ).to_netcdf(args.output)
    print(f"{args.output}: HHL with {hhl.shape[0]} interfaces, {hhl.shape[1]} cells")


if __name__ == "__main__":
    main()
