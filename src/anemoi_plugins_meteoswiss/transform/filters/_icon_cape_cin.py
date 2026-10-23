# Copyright (C) 2004-2026, DWD, MPI-M, DKRZ, KIT, ETH, MeteoSwiss
# SPDX-License-Identifier: BSD-3-Clause
"""NumPy port of ICON's optional NWP CAPE/CIN diagnostics, ML and MU parcels.

Source: src/atm_phy_nwp/mo_opt_nwp_diagnostics.f90, routines cal_cape_cin,
cal_cape_cin_mu, ascent and their thermodynamic helpers, extracted from
revision f2baf8ea04e6ac42bf1c1ea47d7632f4e6f338af. Copied from the
standalone port ``icon_cape_cin.py`` (tested against the compiled Fortran)
without its COSMO MU parcel, input checks and command-line interface; the
code is otherwise unchanged apart from ruff formatting.

Inputs have shape (column, k), ordered TOP TO BOTTOM: temperature (K),
specific humidity (kg/kg), pressure (Pa), and interface height (m) with one
more level, the last one being the surface. ``kmoist`` is the ZERO-BASED
uppermost integration level. CIN is positive, ``MISSING`` where undefined.
"""

import numpy as np

# mo_granules_phys, mo_physical_constants, mo_lookup_tables_constants.
RD = 287.04
RV = 461.51
CPD = 1004.64
ALV = 2.5008e6
GRAV = 9.80665
TMELT = 273.15
C1ES = 610.78
C3LES = 17.269
C4LES = 35.86
RDV = RD / RV
O_M_RDV = 1.0 - RD / RV
VTMPC1 = RV / RD - 1.0
RD_O_CPD = RD / CPD
ALVDCP = ALV / CPD
MISSING = -999.9
# Source declares REAL(wp), PARAMETER :: eps=0.03 (default-real literal).
_EPS = float(np.float32(0.03))
_DWD = (C3LES, C4LES)
_OUTPUTS = ("cape", "cin", "cape_3km", "lcl", "lfc")


def _esat_water(t, satpres=_DWD):
    # Evaluate only the safe branch, as in the elemental Fortran function.
    c3, c4 = satpres
    safe = t > c4 + 1.0e-6
    out = np.zeros_like(t)
    out[safe] = C1ES * np.exp(c3 * (t[safe] - TMELT) / (t[safe] - c4))
    return out


def _fqvs(e, p):
    e = np.minimum(e, p)
    return RDV * e / (p - O_M_RDV * e)


def _fthetae(t, p, q):
    return (1.0e5 / p) ** RD_O_CPD * t * np.exp(ALVDCP * q / (t * (1.0 - q)))


def _dewpoint_water(q, p, satpres=_DWD):
    c3, c4 = satpres
    out = np.full_like(q, c4)
    wet = q > 1.0e-16
    e = p[wet] / (RDV / q[wet] + O_M_RDV)
    z = np.log(e / C1ES)
    out[wet] = (z * c4 - c3 * TMELT) / (z - c3)
    return out


def _ascent(te, qve, prs, hhl, kmoist, kstart, qstart, tstart, *, three_km, satpres=_DWD):
    """Vectorized horizontal columns; sequential vertical/Newton iterations."""
    ncol, nlev = te.shape
    rows = np.arange(ncol)
    theta = tstart * (1.0e5 / prs[rows, kstart]) ** RD_O_CPD
    cape = np.zeros(ncol)
    cape3 = np.zeros(ncol)
    cin = np.zeros(ncol)
    cin_help = np.zeros(ncol)
    tp = np.zeros(ncol)
    qp = np.zeros(ncol)
    thp = np.zeros(ncol)
    buo = np.zeros(ncol)
    # Keep Fortran's one-based sentinel convention internally.
    lcl = np.zeros(ncol, dtype=np.intp)
    lfc = np.zeros(ncol, dtype=np.intp)
    lfcfound = np.zeros(ncol, dtype=bool)
    k3000 = np.full(ncol, -1, dtype=np.intp)

    if three_km:
        for k in range(nlev - 2, kmoist, -1):
            lower = 0.5 * (hhl[:, k] + hhl[:, k + 1]) - hhl[:, -1]
            upper = 0.5 * (hhl[:, k - 1] + hhl[:, k]) - hhl[:, -1]
            cross = (lower <= 3000.0) & (upper >= 3000.0)
            k3000[cross] = np.where(
                np.abs(lower[cross] - 3000.0) <= np.abs(upper[cross] - 3000.0),
                k,
                k - 1,
            )

    # Columns do not participate until their individual parcel start level.
    for k in range(int(kstart.max()), kmoist - 1, -1):
        active = k <= kstart
        dry = active & (k + 1 > lcl)
        ii = rows[dry]
        if ii.size:
            tp[ii] = theta[ii] * (prs[ii, k] / 1.0e5) ** RD_O_CPD
            qp[ii] = qstart[ii]
            saturated = qp[ii] >= _fqvs(_esat_water(tp[ii], satpres), prs[ii, k])
            jj = ii[saturated & (lcl[ii] == 0)]
            lcl[jj] = k + 1
            thp[jj] = _fthetae(tp[jj], prs[jj, k], qp[jj])

        ii = rows[active & (k + 1 <= lcl)]
        if ii.size:
            guess = np.where(k == kstart[ii], te[ii, k], tp[ii])
            # A column stops independently. ICON updates the guess BEFORE
            # testing the old theta-e residual, and permits 21 iterations.
            pending = np.arange(ii.size)
            for iteration in range(1, 22):
                jj = ii[pending]
                t1 = guess[pending]
                p = prs[jj, k]
                theta1 = _fthetae(t1, p, _fqvs(_esat_water(t1, satpres), p))
                t2 = t1 - 1.0
                theta2 = _fthetae(t2, p, _fqvs(_esat_water(t2, satpres), p))
                guess[pending] = t1 + (theta1 - thp[jj]) / (theta2 - theta1)
                done = (np.abs(theta1 - thp[jj]) < _EPS) | (iteration > 20)
                pending = pending[~done]
                if pending.size == 0:
                    break
            tp[ii] = guess
            qp[ii] = _fqvs(_esat_water(tp[ii], satpres), prs[ii, k])

        ii = rows[active]
        tvp = tp[ii] * (1.0 + VTMPC1 * qp[ii] / (1.0 - qp[ii]))
        qe = qve[ii, k]
        tve = te[ii, k] * (1.0 + VTMPC1 * qe / (1.0 - qe))
        previous = buo[ii].copy()
        buo[ii] = tvp - tve
        crossing = (buo[ii] > 0.0) & (previous <= 0.0) & ~lfcfound[ii]
        first = ii[crossing & (cape[ii] <= 0.0)]
        reset = ii[crossing & (cape[ii] > 0.0) & (2.0 * np.abs(cin_help[ii]) > cape[ii])]
        lfc[first] = k + 1
        cin_help[first] = 0.0
        lfc[reset] = k + 1
        cape[reset] = 0.0
        cape3[reset] = 0.0
        cin_help[reset] = 0.0
        lfcfound[reset] = True

        energy = (buo[ii] / tve) * GRAV * (hhl[ii, k] - hhl[ii, k + 1])
        positive = (buo[ii] >= 0.0) & (k + 1 <= lfc[ii])
        cape[ii[positive]] += energy[positive]
        if three_km:
            below3 = positive & (k3000[ii] >= 0) & (k >= k3000[ii])
            cape3[ii[below3]] += energy[below3]
        negative = ~positive & (buo[ii] < 0.0) & (k < kstart[ii])
        cin[ii[negative]] += energy[negative]
        cin_help[ii[negative]] += energy[negative]

    cin = np.abs(cin - cin_help)
    cin[(lfc == 0) | (np.abs(cape) < 1.0e-8)] = MISSING

    def height(level):
        out = np.full(ncol, MISSING)
        valid = level > 0
        jj = rows[valid]
        kk = level[valid] - 1
        out[valid] = 0.5 * (hhl[jj, kk] + hhl[jj, kk + 1]) - hhl[jj, -1]
        return out

    return dict(cape=cape, cin=cin, cape_3km=cape3, lcl=height(lcl), lfc=height(lfc))


def _compute(te, qve, prs, hhl, kmoist, parcel, z_limit, satpres):
    ncol, nlev = te.shape
    rows = np.arange(ncol)
    if parcel == "ml":
        qstart = np.zeros(ncol)
        theta = np.zeros(ncol)
        kstart = np.full(ncol, nlev - 1, dtype=np.intp)
        k_ml = np.full(ncol, nlev - 1, dtype=np.intp)
        for k in range(nlev - 1, kmoist - 1, -1):
            mixed = prs[:, k] > prs[:, -1] - 5000.0
            if not np.any(mixed):
                break
            qstart[mixed] += qve[mixed, k]
            theta[mixed] += te[mixed, k] * (1.0e5 / prs[mixed, k]) ** RD_O_CPD
            kstart[prs[:, k] > prs[:, -1] - 2500.0] = k
            k_ml[mixed] = k - 1
        count = nlev - 1 - k_ml
        qstart /= count
        theta /= count
        tstart = theta * (prs[rows, kstart] / 1.0e5) ** RD_O_CPD
        return _ascent(
            te,
            qve,
            prs,
            hhl,
            kmoist,
            kstart,
            qstart,
            tstart,
            three_km=True,
            satpres=satpres,
        )

    if parcel == "mu":
        # Preserve the actual adjacent-level comparison in cal_cape_cin_mu;
        # replacing it by argmax would change ICON results in this revision.
        previous = np.full(ncol, -np.finfo(np.float64).max)
        kstart = np.full(ncol, kmoist, dtype=np.intp)
        for k in range(kmoist, nlev):
            z = 0.5 * (hhl[:, k] + hhl[:, k + 1]) - hhl[:, -1]
            ii = rows[z <= z_limit]
            equiv = np.full(ncol, -np.finfo(np.float64).max)
            td = np.maximum(_dewpoint_water(qve[ii, k], prs[ii, k], satpres), 57.0)
            tlcl = 56.0 + 1.0 / (1.0 / (td - 56.0) + np.log(np.maximum(te[ii, k] / td, 1.0)) / 800.0)
            plcl = prs[ii, k] * (tlcl / te[ii, k]) ** (1.0 / RD_O_CPD)
            equiv[ii] = _fthetae(tlcl, plcl, qve[ii, k])
            if k > kmoist:
                kstart[equiv > previous] = k
            previous = equiv
        return _ascent(
            te,
            qve,
            prs,
            hhl,
            kmoist,
            kstart,
            qve[rows, kstart],
            te[rows, kstart],
            three_km=False,
            satpres=satpres,
        )

    raise ValueError(f"parcel must be ml or mu, got {parcel!r}")


def cape_cin(te, qve, prs, hhl, *, kmoist, parcel, z_limit=3000.0):
    """CAPE and CIN (J/kg) of the ``ml`` or ``mu`` parcel, with ICON's default
    (DWD) saturation coefficients; CIN is ``MISSING`` where undefined."""
    result = _compute(te, qve, prs, hhl, kmoist, parcel, z_limit, _DWD)
    return result["cape"], result["cin"]
