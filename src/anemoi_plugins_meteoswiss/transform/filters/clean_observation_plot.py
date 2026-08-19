import logging
import re as _re

import numpy as np

LOG = logging.getLogger(__name__)

_EXTENT = (0.0, 17.5, 40.5, 53.0)   # full obs domain: lon_min, lon_max, lat_min, lat_max
_EXTENT_CH = (5.9, 10.5, 45.8, 47.8)  # Switzerland


def make_qc_map(para, station_names, lats, lons, flagged_set, values, out_path,
                isolation_set=None, extent=None):
    """Save a PNG map for *para* using cartopy for the geographic background.

    Stations are drawn as circles colored by observed value (RdYlBu_r colormap).
    The numerical value is printed inside each circle.
    Flagged stations carry a thick red outline and a bold name label.

    Parameters
    ----------
    para : str
        QC parameter name used in the map title.
    station_names : array-like of str
        Station identifiers, same order as *lats*/*lons*/*values*.
    lats, lons : np.ndarray
        Station latitudes/longitudes (WGS-84 degrees).
    flagged_set : set of str
        Station names flagged by QC for this parameter.
    values : np.ndarray
        Observed value per station in QC-space units (NaN if unavailable).
    out_path : Path
        Destination PNG file.
    """
    import matplotlib
    import sys as _sys
    if "matplotlib.pyplot" not in _sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm
    from matplotlib.lines import Line2D

    ext = extent if extent is not None else _EXTENT  # (lon_min, lon_max, lat_min, lat_max)

    station_names = np.asarray(station_names)
    lats   = np.asarray(lats,   dtype=float)
    lons   = np.asarray(lons,   dtype=float)
    values = np.asarray(values, dtype=float)

    isolation_set = isolation_set or set()

    in_extent   = ((lons >= ext[0]) & (lons <= ext[1]) &
                   (lats >= ext[2]) & (lats <= ext[3]))
    is_isolated = np.array([n in isolation_set for n in station_names])
    is_flagged  = np.array([n in flagged_set and n not in isolation_set
                            for n in station_names])
    ok_mask   = in_extent & ~is_isolated & ~is_flagged
    iso_mask  = in_extent & is_isolated
    flag_mask = in_extent & is_flagged

    # Colormap: robust percentile range so outliers don't dominate the scale
    valid = values[in_extent & ~np.isnan(values)]
    if len(valid) >= 2:
        vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
    else:
        vmin, vmax = (valid[0] - 0.5, valid[0] + 0.5) if len(valid) == 1 else (0, 1)
    cmap = plt.get_cmap("RdYlBu_r")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        _has_cartopy = True
    except ImportError:
        LOG.warning("cartopy not available; maps will have a plain background")
        _has_cartopy = False

    if _has_cartopy:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(14, 10), facecolor="#f5f5f0",
                               subplot_kw={"projection": proj})
        fig.subplots_adjust(left=0.02, right=0.93, top=0.93, bottom=0.02)
        ax.set_extent([ext[0], ext[1], ext[2], ext[3]], crs=proj)
        ax.add_feature(cfeature.LAND.with_scale("10m"),   facecolor="#ededed",  zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"),  facecolor="#dceef5",  zorder=0)
        ax.add_feature(cfeature.LAKES.with_scale("10m"),  facecolor="#aad4ea",
                       edgecolor="#4a90c4", linewidth=0.6, zorder=1)
        ax.add_feature(cfeature.RIVERS.with_scale("10m"), edgecolor="#4a90c4",
                       linewidth=0.6, zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="#777777",
                       linewidth=1.0, zorder=2)
    else:
        proj = None
        fig, ax = plt.subplots(figsize=(14, 10), facecolor="#f5f5f0")
        fig.subplots_adjust(left=0.02, right=0.93, top=0.93, bottom=0.02)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.set_facecolor("#dceef5")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    _transform_kw = {"transform": proj} if _has_cartopy else {}

    import matplotlib.transforms as _mtr
    # Offset label anchor by marker radius (≈7 pt for s=150) upward in display space,
    # so the label bottom always touches the top of the circle regardless of zoom.
    import math as _math
    _marker_r_pt = _math.sqrt(60 / _math.pi) * 0.4
    _label_transform = _mtr.offset_copy(ax.transData, fig=fig, x=0,
                                        y=_marker_r_pt, units="points")

    # OK stations — colored circle + value inside + faint name above
    ax.scatter(lons[ok_mask], lats[ok_mask], s=60,
               c=values[ok_mask], cmap=cmap, norm=norm,
               edgecolors="#444444", linewidths=0.4, alpha=0.90,
               zorder=5, **_transform_kw)
    for name, lat, lon in zip(station_names[ok_mask], lats[ok_mask], lons[ok_mask]):
        ax.text(lon, lat, name, fontsize=5, color="#555555", alpha=0.75,
                ha="left", va="bottom", fontfamily="monospace",
                clip_on=True, zorder=6, transform=_label_transform)

    # Isolated stations — colored circle with thick green outline + bold name label
    if iso_mask.any():
        iso_lons  = lons[iso_mask]
        iso_lats  = lats[iso_mask]
        iso_names = station_names[iso_mask]
        iso_vals  = values[iso_mask]
        ax.scatter(iso_lons, iso_lats, s=60,
                   c=iso_vals, cmap=cmap, norm=norm,
                   edgecolors="#7b2d8b", linewidths=2.0,
                   zorder=7, alpha=0.95, **_transform_kw)
        for name, lon, lat, val in zip(iso_names, iso_lons, iso_lats, iso_vals):
            if not np.isnan(val):
                ax.text(lon, lat, f"{val:.1f}", fontsize=6, color="#111111",
                        fontweight="bold", ha="center", va="center",
                        fontfamily="monospace", clip_on=True, zorder=8, **_transform_kw)
        iso_texts = []
        for name, lon, lat in zip(iso_names, iso_lons, iso_lats):
            t = ax.text(lon, lat, name, fontsize=6.5, fontweight="bold",
                        color="#7b2d8b", ha="center", va="bottom",
                        fontfamily="monospace", clip_on=True, zorder=9,
                        transform=_label_transform)
            iso_texts.append(t)
        try:
            from adjustText import adjust_text
            adjust_text(iso_texts, x=list(iso_lons), y=list(iso_lats), ax=ax,
                        expand=(1.15, 1.4), force_text=(0.10, 0.15),
                        force_points=(0.06, 0.10),
                        arrowprops=dict(arrowstyle="-", color="#7b2d8b", lw=0.6),
                        time_lim=6)
        except Exception:
            pass

    # Flagged stations — colored circle with thick red outline + bold name label
    if flag_mask.any():
        fl_lons  = lons[flag_mask]
        fl_lats  = lats[flag_mask]
        fl_names = station_names[flag_mask]
        fl_vals  = values[flag_mask]
        ax.scatter(fl_lons, fl_lats, s=60,
                   c=fl_vals, cmap=cmap, norm=norm,
                   edgecolors="#8B4513", linewidths=2.0,
                   zorder=7, alpha=0.95, **_transform_kw)
        for name, lon, lat, val in zip(fl_names, fl_lons, fl_lats, fl_vals):
            if not np.isnan(val):
                ax.text(lon, lat, f"{val:.1f}", fontsize=6, color="#111111",
                        fontweight="bold", ha="center", va="center",
                        fontfamily="monospace", clip_on=True, zorder=8, **_transform_kw)
        texts = []
        for name, lon, lat in zip(fl_names, fl_lons, fl_lats):
            t = ax.text(lon, lat, name, fontsize=6.5, fontweight="bold",
                        color="#8B4513", ha="center", va="bottom",
                        fontfamily="monospace", clip_on=True, zorder=9,
                        transform=_label_transform)
            texts.append(t)
        try:
            from adjustText import adjust_text
            adjust_text(texts, x=list(fl_lons), y=list(fl_lats), ax=ax,
                        expand=(1.15, 1.4), force_text=(0.10, 0.15),
                        force_points=(0.06, 0.10),
                        arrowprops=dict(arrowstyle="-", color="#8B4513", lw=0.6),
                        time_lim=6)
        except Exception:
            pass

    # Colorbar
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.025, pad=0.01)
    cb.ax.tick_params(labelsize=9)
    cb.set_label(para, fontsize=10)

    n_ok = int(ok_mask.sum())
    n_isolated = int(iso_mask.sum())
    n_flagged = int(flag_mask.sum())
    ax.set_title(
        f"{para} QC map  ·  {n_ok} ok  ·  {n_isolated} isolated  ·  {n_flagged} flagged",
        fontsize=13, fontweight="bold", pad=8, color="#111111")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               markeredgecolor="#444444", markeredgewidth=0.4, markersize=8, label="OK"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               markeredgecolor="#7b2d8b", markeredgewidth=2.0, markersize=8,
               label=f"Isolated ({n_isolated})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               markeredgecolor="#8B4513", markeredgewidth=2.0, markersize=8,
               label=f"Flagged ({n_flagged})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    LOG.info("QC map saved: %s", out_path)


def plot_station_maps(flagged, par2check, df, obs_path_out, para_values=None, isolation_sets=None):
    """Save one PNG per parameter to the same directory as the output parquet.

    Parameters
    ----------
    flagged : list of dict
        ``self._flagged`` list from :class:`CleanObservation`.
    par2check : list of str
        Parameters to produce maps for (from ``clean_observation_config``).
    df : pd.DataFrame
        Observation DataFrame with ``latitude`` and ``longitude`` columns and
        station names as index.
    obs_path_out : Path
        Output parquet path — maps are written to ``obs_path_out.parent``.
    para_values : dict or None
        ``{para: {station: float}}`` of original observed values in QC-space units.
        Built by :meth:`CleanObservation._plot_station_maps`.
    """
    try:
        import matplotlib
        import sys as _sys
        if "matplotlib.pyplot" not in _sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:
        LOG.warning("matplotlib not available (%s); skipping QC station maps", exc)
        return

    para_values   = para_values   or {}
    isolation_sets = isolation_sets or {}

    flagged_by_para: dict = {}
    for entry in flagged:
        flagged_by_para.setdefault(entry["qc_parameter"], set()).add(entry["station"])

    ts_match = _re.search(r'\d{12}', obs_path_out.stem)
    ts_prefix = ts_match.group() + "_" if ts_match else ""

    lats = df["latitude"].to_numpy(dtype=float)
    lons = df["longitude"].to_numpy(dtype=float)
    station_names = np.array(df.index.to_list())
    plot_dir = obs_path_out.parent

    for para in par2check:
        flagged_set = flagged_by_para.get(para, set())
        iso_set = isolation_sets.get(para, set())
        vals_dict = para_values.get(para, {})
        values = np.array([vals_dict.get(n, np.nan) for n in station_names])

        out_full = plot_dir / f"{ts_prefix}{para}_qc_map.png"
        make_qc_map(para, station_names, lats, lons, flagged_set, values, out_full,
                    isolation_set=iso_set, extent=_EXTENT)

        out_ch = plot_dir / f"{ts_prefix}{para}_qc_map_ch.png"
        make_qc_map(para, station_names, lats, lons, flagged_set, values, out_ch,
                    isolation_set=iso_set, extent=_EXTENT_CH)
