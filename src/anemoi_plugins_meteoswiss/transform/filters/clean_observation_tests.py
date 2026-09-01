import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
import titanlib

#configuration variables
import clean_observation_config as c

LOG = logging.getLogger(__name__)

#----------------------------------
# isolation check
def isolation_check(stations, lats, lons, elevs, para, date, num_min=2, radius=50000):
    """Flag stations with too few neighbours within radius (titanlib isolation_check).

    Args:
        stations (ndarray): Station names.
        lats (ndarray): Station latitudes.
        lons (ndarray): Station longitudes.
        elevs (ndarray): Station elevations (m).
        para (str): QC parameter name.
        date (str): Observation timestamp label.
        num_min (int): Minimum number of neighbours required to pass. Default 2.
        radius (float): Search radius (m). Default 150000.

    Returns:
        dict: Keys 'Station', 'Time', 'Parameter' listing isolated stations,
            or empty dict if no stations were flagged.
    """
    points = titanlib.Points(lats, lons, elevs)
    flags = titanlib.isolation_check(points, num_min, radius)
    my_dict = {"Station": [], "Time": [], "Parameter": []}
    for idx in np.nonzero(np.asarray(flags))[0]:
        my_dict["Station"].append(stations[idx])
        my_dict["Time"].append(date)
        my_dict["Parameter"].append(para)
        LOG.info('isolation_check %s isolated station: %s', para, stations[idx])
    return my_dict if my_dict["Station"] else {}

#----------------------------------
# DWH quality flag test
def DWH_flag(current_f, para, stationss, plausibility):
    """Plausibility test using pre-fetched DWH plausibility values.

    Args:
        current_f (str): Current timestamp in the format "%Y%m%d%H%M".
        para (str): QC parameter name.
        stationss (list): Station names to evaluate.
        plausibility (pd.Series): Plausibility values indexed by station name
            (the ``*_pi`` column from the observation parquet, range 0–1).

    Returns:
        tuple: (blacklist_dict, freq) where blacklist_dict has keys
            "Station", "Time", "Parameter" listing flagged stations, and
            freq is a value_counts Series of the plausibility distribution.
            blacklist_dict is an empty dict when no stations are flagged.
    """
    threshold = float(c.DWH_flag[para]['dwh_plausibility_thr'])
    LOG.info('DWH plausibility test for %s, threshold=%.2f', para, threshold)

    # Restrict to the stations passed to this test and drop missing values
    pla = plausibility.reindex(stationss).dropna()
    freq = pla.value_counts()
    flagged = pla[pla < threshold]

    if len(flagged) > 0:
        my_dict = {"Station": [], "Time": [], "Parameter": []}
        for station in flagged.index:
            my_dict["Station"].append(station)
            my_dict["Time"].append(current_f)
            my_dict["Parameter"].append(para)
            LOG.info('DWH flag %s %s (pi=%.3f)', para, station, flagged[station])
        LOG.info('DWH flag %s blacklisted stations: %d', para, len(flagged))
    else:
        my_dict = {}

    return my_dict, freq

#-------------------------------------------------------------------------------
#buddy check
def buddy_check(stations,lats,lons,elevs,values,para,date,threshold=4,max_elev_diff=200,elev_gradient=-0.0065,
                          min_std=1,num_iterations=5,num_min=3,radius=30000):
    """Run titanlib buddy_check with separate thresholds for SMN and other stations.

    The test is run twice on the full station set:
      - Round 1 uses threshold[0] (lenient, for SMN); only flags from 3-letter stations are kept.
      - Round 2 uses threshold[1] (stricter, for other networks); only flags from longer-name
        stations are kept.
    Running both rounds on the full set preserves spatial context for both groups.

    Args:
        stations (list[str]): Station short names.
        lats (ndarray): Station latitudes.
        lons (ndarray): Station longitudes.
        elevs (ndarray): Station elevations (m).
        values (ndarray): Observed values.
        para (str): QC parameter name.
        date (str): Observation timestamp label (stored in the returned dict).
        threshold (float | list[float, float]): Buddy-check deviation threshold.
            Pass a 2-element list [SMN_thr, other_thr] to apply different thresholds per network.
            A scalar is broadcast to both rounds.
        max_elev_diff (float): Max elevation difference between buddies (m). Default 200.
        elev_gradient (float): Value lapse rate with elevation (unit/m). Default -0.0065.
        min_std (float): Minimum standard deviation of buddy values. Default 1.
        num_iterations (int): Number of titanlib buddy_check iterations. Default 5.
        num_min (int): Minimum number of neighbours required. Default 3.
        radius (float): Search radius (m). Default 30000.

    Returns:
        dict: Keys 'Station', 'Time', 'Parameter' listing flagged stations,
            or empty dict if no stations were flagged.
    """
    thresholds = threshold if isinstance(threshold, list) else [threshold, threshold]
    thr_short = thresholds[0]
    thr_long = thresholds[1] if len(thresholds) > 1 else thresholds[0]
    snames = np.asarray(stations)
    short_mask = np.array([len(s) == 3 for s in snames])
    points = titanlib.Points(lats, lons, elevs)
    radius_arr  = np.full(points.size(), radius)
    num_min_arr = np.full(points.size(), num_min)
    flagged_indices = set()
    # round 1: all stations, keep only SMN (3-letter) flags
    LOG.info('titanlib buddy_check %s threshold=%.2f (SMN)', para, thr_short)
    flags = titanlib.buddy_check(points, values, radius_arr, num_min_arr, thr_short,
                                 max_elev_diff, elev_gradient, min_std, num_iterations)
    for idx in np.nonzero(flags)[0]:
        if short_mask[idx]:
            flagged_indices.add(int(idx))
    # round 2: all stations, keep only other (longer-name) flags
    LOG.info('titanlib buddy_check %s threshold=%.2f (other)', para, thr_long)
    flags = titanlib.buddy_check(points, values, radius_arr, num_min_arr, thr_long,
                                 max_elev_diff, elev_gradient, min_std, num_iterations)
    for idx in np.nonzero(flags)[0]:
        if not short_mask[idx]:
            flagged_indices.add(int(idx))
    my_dict = {"Station": [], "Time": [], "Parameter": []}
    n = 0
    for idx in sorted(flagged_indices):
        if np.isfinite(values[idx]):
            my_dict["Station"].append(snames[idx])
            my_dict["Time"].append(date)
            my_dict["Parameter"].append(para)
            LOG.info('titanlib buddy_check %s %s %.4f', para, snames[idx], values[idx])
            n += 1
    LOG.info('titanlib buddy_check %s blacklisted stations: %d', para, n)
    if len(my_dict["Station"]) == 0:
        my_dict = {}
    return my_dict

#first guess test
def first_guess_test(stations,lats,lons,elevs,values,background_values,para,date,background_elab_type=1,num_min_outer=3,num_max_outer=10,
                    inner_radius=50000,outer_radius=100000,num_iterations=10,num_min_prof=1,min_elev_diff=250,
                    min_horizontal_scale=250,max_horizontal_scale=100000,kth_closest_obs_horizontal_scale=2,
                    bdebug=True,bbasic=True,tpostneg=5):
    """Run titanlib fgt (first-guess test) with separate thresholds for SMN and other stations.

    The test is run twice on the full station set:
      - Round 1 uses tpostneg[0] (lenient, for SMN); only flags from 3-letter stations are kept.
      - Round 2 uses tpostneg[1] (stricter, for other networks); only flags from longer-name
        stations are kept.
    Running both rounds on the full set preserves the spatial background for both groups.

    Args:
        stations (list[str]): Station short names.
        lats (ndarray): Station latitudes.
        lons (ndarray): Station longitudes.
        elevs (ndarray): Station elevations (m).
        values (ndarray): Observed values.
        background_values (ndarray): Model background values at station locations.
        para (str): QC parameter name.
        date (str): Observation timestamp label (stored in the returned dict).
        background_elab_type (int): Background elaboration type passed to fgt. Default 1.
        num_min_outer (int): Minimum outer neighbours for background estimation. Default 3.
        num_max_outer (int): Maximum outer neighbours. Default 10.
        inner_radius (float): Inner search radius (m). Default 50000.
        outer_radius (float): Outer search radius (m). Default 100000.
        num_iterations (int): Number of titanlib fgt iterations. Default 10.
        num_min_prof (int): Minimum neighbours for vertical profile. Default 1.
        min_elev_diff (float): Minimum elevation difference for vertical profile. Default 250.
        min_horizontal_scale (float): Minimum horizontal decorrelation length (m). Default 250.
        max_horizontal_scale (float): Maximum horizontal decorrelation length (m). Default 100000.
        kth_closest_obs_horizontal_scale (int): k-th closest obs used for horizontal scale. Default 2.
        bdebug (bool): Enable titanlib debug output. Default True.
        bbasic (bool): Enable titanlib basic output. Default True.
        tpostneg (float | list[float, float]): Obs-minus-background acceptance threshold.
            Pass a 2-element list [SMN_thr, other_thr] for per-network thresholds.
            A scalar is broadcast to both rounds.

    Returns:
        dict: Keys 'Station', 'Time', 'Parameter' listing flagged stations,
            or empty dict if no stations were flagged.
    """
    thresholds = tpostneg if isinstance(tpostneg, list) else [tpostneg, tpostneg]
    thr_short = thresholds[0]
    thr_long = thresholds[1] if len(thresholds) > 1 else thresholds[0]
    snames = np.asarray(stations)
    short_mask = np.array([len(s) == 3 for s in snames])
    N = len(lats)
    points = titanlib.Points(lats, lons, elevs)
    obs_to_check = np.repeat(1, N)
    background_uncertainties = np.repeat(np.nanstd(background_values), N)
    values_mina = values - 25
    values_maxa = values + 25
    values_minv = values - 1
    values_maxv = values + 1
    flagged_indices = set()
    # round 1: all stations, keep only SMN (3-letter) flags
    tpos = np.repeat(1, N) * thr_short
    tneg = np.repeat(1, N) * thr_short
    LOG.info('titanlib first_guess_test %s threshold=%.2f (SMN)', para, thr_short)
    try:
        flags, scores = titanlib.fgt(points, values, obs_to_check, background_values, background_uncertainties,
                                     background_elab_type, num_min_outer, num_max_outer, inner_radius,
                                     outer_radius, num_iterations, num_min_prof, min_elev_diff,
                                     values_mina, values_maxa, values_minv, values_maxv, tpos, tneg,
                                     bdebug, bbasic)
    except Exception:
        flags = []
        scores = []
        LOG.warning('exception in fgt test (SMN)')
    if len(flags) > 0:
        for idx in np.where((flags > 0) & (scores > 0))[0]:
            if short_mask[idx]:
                flagged_indices.add(int(idx))
    # round 2: all stations, keep only other (longer-name) flags
    tpos = np.repeat(1, N) * thr_long
    tneg = np.repeat(1, N) * thr_long
    LOG.info('titanlib first_guess_test %s threshold=%.2f (other)', para, thr_long)
    try:
        flags, scores = titanlib.fgt(points, values, obs_to_check, background_values, background_uncertainties,
                                     background_elab_type, num_min_outer, num_max_outer, inner_radius,
                                     outer_radius, num_iterations, num_min_prof, min_elev_diff,
                                     values_mina, values_maxa, values_minv, values_maxv, tpos, tneg,
                                     bdebug, bbasic)
    except Exception:
        flags = []
        scores = []
        LOG.warning('exception in fgt test (other)')
    if len(flags) > 0:
        for idx in np.where((flags > 0) & (scores > 0))[0]:
            if not short_mask[idx]:
                flagged_indices.add(int(idx))
    my_dict = {"Station": [], "Time": [], "Parameter": []}
    n = 0
    for idx in sorted(flagged_indices):
        if np.isfinite(values[idx]):
            my_dict["Station"].append(snames[idx])
            my_dict["Time"].append(date)
            my_dict["Parameter"].append(para)
            LOG.info('titanlib first_guess_test %s %s %.4f', para, snames[idx], values[idx])
            n += 1
    LOG.info('titanlib first_guess_test %s blacklisted stations: %d', para, n)
    return my_dict

#spacial concistency check resistant 
def spacial_ct_resistant(stations,lats,lons,elevs,values,para,date,background_elab_type=1,num_min_outer=3,num_max_outer=10,
                    inner_radius=50000,outer_radius=100000,num_iterations=10,num_min_prof=1,min_elev_diff=250,
                    min_horizontal_scale=250,max_horizontal_scale=100000,kth_closest_obs_horizontal_scale=2,
                    vertical_scale=200,bdebug=True,bbasic=True):
    """Run titanlib sct_resistant (spatial consistency test, resistant variant).

    Args:
        stations (list[str]): Station short names.
        lats (ndarray): Station latitudes.
        lons (ndarray): Station longitudes.
        elevs (ndarray): Station elevations (m).
        values (ndarray): Observed values.
        para (str): QC parameter name.
        date (str): Observation timestamp label.
        background_elab_type (int): Background elaboration type. Default 1.
        num_min_outer (int): Minimum outer neighbours. Default 3.
        num_max_outer (int): Maximum outer neighbours. Default 10.
        inner_radius (float): Inner search radius (m). Default 50000.
        outer_radius (float): Outer search radius (m). Default 100000.
        num_iterations (int): Number of iterations. Default 10.
        num_min_prof (int): Minimum neighbours for vertical profile. Default 1.
        min_elev_diff (float): Minimum elevation difference (m). Default 250.
        min_horizontal_scale (float): Minimum horizontal decorrelation length (m). Default 250.
        max_horizontal_scale (float): Maximum horizontal decorrelation length (m). Default 100000.
        kth_closest_obs_horizontal_scale (int): k-th closest obs for horizontal scale. Default 2.
        vertical_scale (float): Vertical decorrelation length (m). Default 200.
        bdebug (bool): Enable titanlib debug output. Default True.
        bbasic (bool): Enable titanlib basic output. Default True.

    Returns:
        dict: Keys 'Station', 'Time', 'Parameter' listing flagged stations,
            or empty dict if no stations were flagged.
    """
    points = titanlib.Points(lats, lons, elevs)
    npoints = len(lats)
    N=npoints
    obs_to_check = np.repeat(1, npoints)
    background_values = np.repeat(0, npoints)
    tpos = np.repeat(1,N) * 16
    tneg = np.repeat(1,N) * 16
    eps2 = np.repeat(1,N) * 0.5
    values_mina = values - 15
    values_maxa = values + 15
    values_minv = values - 1
    values_maxv = values + 1
    LOG.info('spacial_ct_resistant %s', para)
    try:
        flags,scores = titanlib.sct_resistant(points, values, obs_to_check, background_values, 
                        background_elab_type, num_min_outer, num_max_outer, 
                        inner_radius, outer_radius, num_iterations, num_min_prof, 
                        min_elev_diff, min_horizontal_scale, max_horizontal_scale, 
                        kth_closest_obs_horizontal_scale, vertical_scale, 
                        values_mina, values_maxa, values_minv, values_maxv,
                        eps2, tpos, tneg, bdebug, bbasic)
    except Exception:
        flags=[]
        scores=[]
        LOG.warning('Spacial_ct_resistant '+'exeption in test')
    if len(flags)>0:
        indices=np.where((flags>0) &  (scores>15))
        indices=indices[:][0]
        nblack=len(indices)
        my_dict = {"Station":[],"Time":[],"Parameter":[]}
        n=0
        for b in range(0,nblack):
            if np.isfinite(values[indices[b]]):
                my_dict["Station"].append(stations[indices[b]])
                my_dict["Time"].append(date)
                my_dict["Parameter"].append(para)
                LOG.info('titanlib sct_resistant %s %s %.4f score=%.4f', para, stations[indices[b]], values[indices[b]], scores[indices[b]])
                n=n+1
        LOG.info('spacial_ct_resistant %s blacklisted stations: %d', para, n)
    else:
        my_dict={}
    return my_dict
#-------------------------------------------------------------------------------------------
#spacial concistency check dual
def spacial_ct_dual(stations,lats,lons,elevs,values,para,date,num_min_outer=3,num_max_outer=10,
                    inner_radius=50000,outer_radius=100000,num_iterations=10,
                    min_horizontal_scale=250,max_horizontal_scale=100000,kth_closest_obs_horizontal_scale=2,
                    vertical_scale=200,debug=True,condition = 1, event_thresholds = 0.1,test_thresholds = 0.8):
    
    """Run titanlib sct_dual (spatial consistency test, dual-threshold variant).

    Args:
        stations (list[str]): Station short names.
        lats (ndarray): Station latitudes.
        lons (ndarray): Station longitudes.
        elevs (ndarray): Station elevations (m).
        values (ndarray): Observed values.
        para (str): QC parameter name.
        date (str): Observation timestamp label.
        num_min_outer (int): Minimum outer neighbours. Default 3.
        num_max_outer (int): Maximum outer neighbours. Default 10.
        inner_radius (float): Inner search radius (m). Default 50000.
        outer_radius (float): Outer search radius (m). Default 100000.
        num_iterations (int): Number of iterations. Default 10.
        min_horizontal_scale (float): Minimum horizontal decorrelation length (m). Default 250.
        max_horizontal_scale (float): Maximum horizontal decorrelation length (m). Default 100000.
        kth_closest_obs_horizontal_scale (int): k-th closest obs for horizontal scale. Default 2.
        vertical_scale (float): Vertical decorrelation length (m). Default 200.
        debug (bool): Enable titanlib debug output. Default True.
        condition (int): Event condition type passed to sct_dual. Default 1.
        event_thresholds (float): Threshold defining an 'event'. Default 0.1.
        test_thresholds (float): Threshold for flagging (broadcast to all stations). Default 0.8.

    Returns:
        dict: Keys 'Station', 'Time', 'Parameter' listing flagged stations,
            or empty dict if no stations were flagged.
    """
    points = titanlib.Points(lats, lons, elevs)
    npoints = len(lats)
    obs_to_check = np.repeat(1, npoints)
    test_thresholds = np.repeat(test_thresholds, npoints)
    event_thresholds = np.repeat(event_thresholds, npoints)
    #print(locals())
    LOG.info('titanlib spacial_ct_dual %s', para)
    try:
        values=np.asarray(values, dtype=np.float64)
        flags= titanlib.sct_dual(points, values, obs_to_check, event_thresholds, condition,
                        num_min_outer, num_max_outer, inner_radius, outer_radius,
                        num_iterations, min_horizontal_scale, max_horizontal_scale,
                        kth_closest_obs_horizontal_scale, vertical_scale,
                        test_thresholds, debug)
    except Exception:
        flags=[]
        LOG.warning('spacial_ct_dual '+'exeption in sct_resistant test')
    if len(flags)>0:
        indices=np.where((flags>0))
        indices=indices[:][0]
        nblack=len(indices)
        my_dict = {"Station":[],"Time":[],"Parameter":[]}
        n=0
        for b in range(0,nblack):
            if np.isfinite(values[indices[b]]):
                my_dict["Station"].append(stations[indices[b]])
                my_dict["Time"].append(date)
                my_dict["Parameter"].append(para)
                LOG.info('titanlib sct_dual %s %s %.4f', para, stations[indices[b]], values[indices[b]])
                n=n+1
        LOG.info('titanlib spacial_ct_dual %s blacklisted stations: %d', para, n)
    else:
        my_dict={}
    return my_dict
#--------------------------------------------------------------
def find_indices(list_to_check, item_to_find):
    """Return all indices where list_to_check equals item_to_find."""
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]

#--------------------------------------------------------------
def tests_summary (blacklist,weights,tests):
    """Aggregate per-test flags into a weighted score for each (time, station) pair.

    For each station the score is:

        score = sum(weight_i for each test_i that flagged the station)
                / len(blacklist['tests'])

    where ``blacklist['tests']`` is the list of tests that were actually run
    (not the full configured set).  A station is considered suspicious when
    ``score > threshold_summary`` (0.2 by default, checked by the caller).

    Args:
        blacklist (dict): Accumulator produced by make_tests.  Must have key
            'tests' (list of test names that ran) and one sub-dict per test
            with keys 'Station' and 'Time'.
        weights (list[float]): Per-test weights in the same order as ``tests``.
        tests (list[str]): Ordered list of all configured test names
            (used to align weights; subset of blacklist['tests']).

    Returns:
        pandas.DataFrame: Shape (n_times, n_stations).  Each cell is the
            weighted flag score for that (time, station) pair.
    """
    stations=[]
    times=[]
    for t in blacklist['tests']:
        n=0
        for sta in blacklist[t]['Station']:
            stations.append(blacklist[t]['Station'][n])
            times.append(blacklist[t]['Time'][n])
            n=n+1       
    times=np.unique(times)
    stations=np.unique(stations)
    qc=np.zeros((len(times),len(stations)))
    ns=0
   
    for sta in stations:
        k=0
        for time in times:
            n=0
            tt=0
            for t in tests:
                if sta in blacklist[t]['Station']:
                    id=find_indices(blacklist[t]['Station'], sta)
                    for i in id:
                        if time in blacklist[t]['Time'][i]:
                            n=n+1*weights[tt]
                tt=tt+1
            qc[k,ns]=n/len(blacklist['tests'])
            k=k+1
        ns=ns+1
    df=pd.DataFrame(data=qc,index=times,columns=stations)
   
    return df
def hard_test (obs_all,datamin,datamax,var,time):
    """Flag stations whose value falls outside absolute physical limits.

    Args:
        obs_all (DataFrame): Observation DataFrame (must contain 'sta_name' and var columns).
        datamin (float): Lower hard limit.
        datamax (float): Upper hard limit.
        var (str): Column name of the parameter to test.
        time (str): Timestamp label stored in the returned dict.

    Returns:
        dict: Keys 'Station', 'Time', 'Parameter' listing flagged stations.
            Always non-empty structure (never empty dict like other tests).
    """

    LOG.info('hard test %s min=%.2f max=%.2f', var, datamin, datamax)
    my_dict = {"Station":[],"Time":[],"Parameter":[]}
    n=0
    stations=obs_all['sta_name']
    for s in stations:
        sta_data=obs_all[obs_all['sta_name']==s][var]
        if not sta_data.isna().item():
            #check if value is outside the hard limits and insert case into the dictionary
            if (sta_data.iloc[0] < datamin) or (sta_data.iloc[0] > datamax):
                my_dict["Station"].append(stations[n])
                my_dict["Time"].append(time)
                my_dict["Parameter"].append(var)
                LOG.info('hard test %s %s %.4f', var, stations[n], sta_data.iloc[0])
        n=n+1
    return my_dict

#--------------------------------------------------------------
def plateau_test(data, window, std_lim, var, time, obs_path_in=None, gran_minutes=60):
    """
    Plateau test: detect frozen/stuck sensors by checking that observations vary
    over a past time window.

    Args:
        data (pandas.DataFrame): Current observation DataFrame (must contain 'sta_name' and var columns).
        window (int): Look-back window in hours.
        std_lim (float): Max std dev below which a station is flagged as stuck (0 = exact plateau).
        var (str): QC parameter column name.
        time (str): Current timestamp in '%Y%m%d%H%M' format.
        obs_path_in (str | Path, optional): Path to the current parquet file.  Historical files
            are discovered by replacing the timestamp in the filename.  If None or no historical
            files are found, the test is skipped.
        gran_minutes (int): Step between historical parquet files in minutes (default 60).

    Returns:
        dict | None: {'Station': [...], 'Time': [...], 'Parameter': [...]} for flagged stations,
            empty dict when run but nothing flagged, or None when the test could not run
            (no obs_path_in, no timestamp in filename, or zero historical files available).
    """
    LOG.info('plateau test for %s', var)
    my_dict = {"Station": [], "Time": [], "Parameter": []}

    if obs_path_in is None:
        LOG.warning('plateau_test %s: obs_path_in not provided, cannot run test', var)
        return None

    import re as _re
    obs_path_in = Path(obs_path_in)
    n_steps = int(window * 60 / gran_minutes)

    # Extract the observation timestamp from the filename (12-digit YYYYMMDDhhmm)
    ts_match = _re.search(r'\d{12}', obs_path_in.stem)
    if not ts_match:
        LOG.warning(
            'plateau_test %s: no 12-digit timestamp found in filename %s, '
            'cannot build historical file list',
            var, obs_path_in.name,
        )
        return None
    file_ts = ts_match.group()
    current = datetime.strptime(file_ts, '%Y%m%d%H%M')
    stem_pattern = obs_path_in.stem.replace(file_ts, '{}')

    def _normalize_hist(hist_df):
        """Return a two-column DataFrame {sta_name, var} from a raw obs parquet.

        Raw parquets use station nat_abbr as index and parquet column names
        (e.g. '2t') rather than the QC names (e.g. 'T_2M') used in df_qc.
        """
        norm = pd.DataFrame({'sta_name': hist_df.index.to_list()})
        # Map parquet column → QC column for this var
        for pc, (qp, conv) in c.parquet_to_qc.items():
            if qp == var and pc in hist_df.columns:
                norm[var] = conv(hist_df[pc].to_numpy())
                return norm
        # FF_10M is derived from U/V components
        if var == 'FF_10M' and '10u' in hist_df.columns and '10v' in hist_df.columns:
            norm[var] = np.sqrt(hist_df['10u'].to_numpy() ** 2 +
                                hist_df['10v'].to_numpy() ** 2)
        else:
            norm[var] = np.nan
        return norm

    historical_dfs = []
    missing = []
    for step in range(1, n_steps + 1):
        t = current - timedelta(minutes=step * gran_minutes)
        fn = obs_path_in.parent / (stem_pattern.format(t.strftime('%Y%m%d%H%M')) + obs_path_in.suffix)
        if fn.exists():
            try:
                historical_dfs.append(_normalize_hist(pd.read_parquet(fn)))
            except Exception as e:
                LOG.warning('plateau_test: could not read %s: %s', fn, e)
                missing.append(str(fn))
        else:
            missing.append(str(fn))

    if missing:
        for mf in missing:
            LOG.warning('plateau_test %s: missing file: %s', var, mf)

    n_found = len(historical_dfs)
    if n_found < n_steps / 2:
        LOG.warning(
            'plateau_test %s: cannot run — only %d/%d historical files available (window=%dh, gran=%dmin)',
            var, n_found, n_steps, window, gran_minutes,
        )
        return None
    if missing:
        LOG.warning(
            'plateau_test %s: running on partial history (%d/%d files available)',
            var, n_found, n_steps,
        )

    # Only test stations present in every historical file to avoid spurious
    # plateau flags caused by a station first appearing mid-window.
    always_available = set(data['sta_name'])
    for hdf in historical_dfs:
        always_available &= set(hdf['sta_name'].dropna())
    skipped = len(set(data['sta_name'])) - len(always_available)
    if skipped:
        LOG.info('plateau_test %s: skipping %d station(s) not present in all historical files', var, skipped)

    obs_all = pd.concat([data[[var, 'sta_name']]] + historical_dfs, ignore_index=True)
    for s in data['sta_name']:
        if s not in always_available:
            continue
        cur_val = data.loc[data['sta_name'] == s, var]
        if len(cur_val) == 0 or pd.isna(cur_val.iloc[0]):
            continue
        series = obs_all.loc[obs_all['sta_name'] == s, var]
        sd = series.std()
        nn = series.count()
        if not pd.isna(sd) and sd <= std_lim and nn > n_steps / 2:
            LOG.warning('plateau_test %s %s std=%.4f', var, s, sd)
            my_dict["Station"].append(s)
            my_dict["Time"].append(time)
            my_dict["Parameter"].append(var)
    LOG.info('plateau_test %s: %d stations flagged', var, len(my_dict["Station"]))
    return my_dict
