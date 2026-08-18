import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
import titanlib

#configuration variables
import clean_observation_config as c

LOG = logging.getLogger(__name__)


def make_tests(current_f, df_obs, df_diff, df_mod, para, stations, lats, lons, elevs, ii, jj, my_dict, tests_to_do, df_pi=None, obs_path_in=None):
    """
    Performs multiple tests on meteorological stations to detect outliers and generates a dictionary of blacklisted stations.

    Args:
        current_f (str): Current timestamp in the format "%Y%m%d%H%M".
        df_obs (DataFrame): DataFrame containing observed values.
        df_diff (DataFrame): DataFrame containing differences between model and observations.
        df_mod (DataFrame): DataFrame containing model values.
        para (str): Parameter name.
        stations (list): List of station names.
        lats (list): List of latitudes for each station.
        lons (list): List of longitudes for each station.
        elevs (list): List of elevations for each station.
        ii (int): Index value.
        jj (int): Index value.
        my_dict (dict): Dictionary to store blacklisted stations.

    Returns:
        dict: A dictionary containing the blacklisted stations and their corresponding information.
              The dictionary has keys for each test, such as 'hard', 'buddy_obs', 'buddy_diff', 'fgt', etc.
              Each test key contains a sub-dictionary with keys 'ID', 'Station', 'Time', and 'Parameter'.
              The 'ID' key provides a unique identifier for each blacklisted station.
    """
    #obs values
    values=df_obs[para].iloc[:].to_numpy()
    freq=pd.DataFrame()
    
    #remove stations with nan for titanlib tests
    ind=np.argwhere(np.isnan(values))
    values = np.delete(values,ind)
    stations = np.delete(stations,ind)
    lats = np.delete(lats,ind)
    lons = np.delete(lons,ind)
    elevs = np.delete(elevs,ind)
    mods = np.delete(df_mod[para].iloc[:].to_numpy(),ind)
    diffs= np.delete(df_diff[para].iloc[:].to_numpy(),ind)

    #0. hard test using hard max und min limits for observations
    if 'hard' in tests_to_do:
        ind_var=c.obs_variables.index(para)
        blacklist=hard_test(df_obs,c.plausibility_thresholds['pch_min'][ind_var],c.plausibility_thresholds['pch_max'][ind_var],para,current_f)
        nb0=len(my_dict['hard']['ID'])
        if blacklist:
            nblack=len(blacklist["Station"])
            for k in range(0,nblack):
                nb0=nb0+1
                my_d = {"ID":nb0,"Station":blacklist["Station"][k],"Time":blacklist["Time"][k],"Parameter":blacklist["Parameter"][k]}
                my_dict['hard']["ID"].append(my_d["ID"])
                my_dict['hard']["Station"].append(my_d["Station"])
                my_dict['hard']["Time"].append(my_d["Time"])
                my_dict['hard']["Parameter"].append(my_d["Parameter"])
        LOG.info('hard test         blacklisted stations: %d', len(blacklist["Station"]) if blacklist else 0)

    #1. buddy check using only observations
    if 'buddy_obs' in tests_to_do:
        blacklist=buddy_check(stations,lats,lons,elevs,values,para,current_f,c.buddy[para]["threshold"],
                            c.buddy[para]["max_elev_diff"],c.buddy[para]["elev_gradient"],
                            c.buddy[para]["min_std"],c.buddy[para]["num_iterations"],c.buddy[para]["num_min"],c.buddy[para]["radius"])
        nb1=len(my_dict['buddy_obs']['ID'])
        if blacklist:
            nblack=len(blacklist["Station"])
            for k in range(0,nblack):
                nb1=nb1+1
                my_d = {"ID":nb1,"Station":blacklist["Station"][k],"Time":blacklist["Time"][k],"Parameter":blacklist["Parameter"][k]}
                my_dict['buddy_obs']["ID"].append(my_d["ID"])
                my_dict['buddy_obs']["Station"].append(my_d["Station"])
                my_dict['buddy_obs']["Time"].append(my_d["Time"])
                my_dict['buddy_obs']["Parameter"].append(my_d["Parameter"])
        LOG.info('buddy_obs test    blacklisted stations: %d', len(blacklist["Station"]) if blacklist else 0)

    #2. buddy check with differences between model and observations (titanlib)
    if 'buddy_diff' in tests_to_do:
        blacklist2=buddy_check(stations,lats,lons,elevs,diffs,para,current_f,c.buddy_diff[para]["threshold"],
                            c.buddy_diff[para]["max_elev_diff"],c.buddy_diff[para]["elev_gradient"],
                            c.buddy_diff[para]["min_std"],c.buddy_diff[para]["num_iterations"],c.buddy_diff[para]["num_min"],c.buddy_diff[para]["radius"])
        nb2=len(my_dict['buddy_diff']['ID'])
        if blacklist2:
            nblack=len(blacklist2["Station"])
            for k in range(0,nblack):
                nb2=nb2+1
                my_d = {"ID":nb2,"Station":blacklist2["Station"][k],"Time":blacklist2["Time"][k],"Parameter":blacklist2["Parameter"][k]}
                my_dict['buddy_diff']["ID"].append(my_d["ID"])
                my_dict['buddy_diff']["Station"].append(my_d["Station"])
                my_dict['buddy_diff']["Time"].append(my_d["Time"])
                my_dict['buddy_diff']["Parameter"].append(my_d["Parameter"])
        LOG.info('buddy_diff test   blacklisted stations: %d', len(blacklist2["Station"]) if blacklist2 else 0)

    #3.first guess test (titanlib)
    if 'fgt' in tests_to_do:
        blacklist3=first_guess_test(stations,lats,lons,elevs,values,mods,para,current_f, c.fgt[para]['background_elab_type'], c.fgt[para]['num_min_outer'],
                                    c.fgt[para]['num_max_outer'], c.fgt[para]['inner_radius'], c.fgt[para]['outer_radius'], c.fgt[para]['num_iterations'],
                                    c.fgt[para]['num_min_prof'],c.fgt[para]['min_elev_diff'],
                                    c.fgt[para]['min_horizontal_scale'], c.fgt[para]['max_horizontal_scale'], c.fgt[para]['kth_closest_obs_horizontal_scale'],
                                    bool(c.fgt[para]['debug']), bool(c.fgt[para]['basic']),c.fgt[para]['tpostneg'])
        nb3=len(my_dict['fgt']['ID'])
        if blacklist3:
            nblack=len(blacklist3["Station"])
            for k in range(0,nblack):
                nb3=nb3+1
                my_d = {"ID":nb3,"Station":blacklist3["Station"][k],"Time":blacklist3["Time"][k],"Parameter": blacklist3["Parameter"][k]}
                my_dict['fgt']["ID"].append(my_d["ID"])
                my_dict['fgt']["Station"].append(my_d["Station"])
                my_dict['fgt']["Time"].append(my_d["Time"])
                my_dict['fgt']["Parameter"].append(my_d["Parameter"])
        LOG.info('fgt test          blacklisted stations: %d', len(blacklist3["Station"]) if blacklist3 else 0)

    #5 SCT resistant test
    if 'spt_resistant' in tests_to_do:
        blacklist4 = spacial_ct_resistant(stations,lats,lons,elevs,values,para,current_f,
                            c.spt_resistant[para]['background_elab_type'], c.spt_resistant[para]['num_min_outer'], c.spt_resistant[para]['num_max_outer'],
                            c.spt_resistant[para]['inner_radius'], c.spt_resistant[para]['outer_radius'], c.spt_resistant[para]['num_iterations'], c.spt_resistant[para]['num_min_prof'],
                            c.spt_resistant[para]['min_elev_diff'], c.spt_resistant[para]['min_horizontal_scale'], c.spt_resistant[para]['max_horizontal_scale'],
                            c.spt_resistant[para]['kth_closest_obs_horizontal_scale'], c.spt_resistant[para]['vertical_scale'],
                            c.spt_resistant[para]['debug'], c.spt_resistant[para]['basic'])
        nb4=len(my_dict['spt_resistant']['ID'])
        if blacklist4:
            nblack=len(blacklist4["Station"])
            for k in range(0,nblack):
                nb4=nb4+1
                my_d = {"ID":nb4,"Station":blacklist4["Station"][k],"Time":blacklist4["Time"][k],"Parameter":blacklist4["Parameter"][k]}
                my_dict['spt_resistant']["ID"].append(my_d["ID"])
                my_dict['spt_resistant']["Station"].append(my_d["Station"])
                my_dict['spt_resistant']["Time"].append(my_d["Time"])
                my_dict['spt_resistant']["Parameter"].append(my_d["Parameter"])
        LOG.info('spt_resistant test blacklisted stations: %d', len(blacklist4["Station"]) if blacklist4 else 0)

    #5. SCT dual test
    if 'spt_dual' in tests_to_do:
        blacklist5=spacial_ct_dual(stations,lats,lons,elevs,values,para,current_f,c.sct_dual[para]['num_min_outer'],c.sct_dual[para]['num_max_outer'],
                    c.sct_dual[para]['inner_radius'],c.sct_dual[para]['outer_radius'],c.sct_dual[para]['num_iterations'],
                    c.sct_dual[para]['min_horizontal_scale'], c.sct_dual[para]['max_horizontal_scale'],c.sct_dual[para]['kth_closest_obs_horizontal_scale'],
                    c.sct_dual[para]['vertical_scale'],bool(c.sct_dual[para]['debug']),c.sct_dual[para]['condition'],float(c.sct_dual[para]['event_thresholds']),
                    float(c.sct_dual[para]['test_thresholds']))
        nb5=len(my_dict['spt_dual']['ID'])
        if blacklist5:
            nblack=len(blacklist5["Station"])
            for k in range(0,nblack):
                nb5=nb5+1
                my_d = {"ID":nb5,"Station":blacklist5["Station"][k],"Time":blacklist5["Time"][k],"Parameter":blacklist5["Parameter"][k]}
                my_dict['spt_dual']["ID"].append(my_d["ID"])
                my_dict['spt_dual']["Station"].append(my_d["Station"])
                my_dict['spt_dual']["Time"].append(my_d["Time"])
                my_dict['spt_dual']["Parameter"].append(my_d["Parameter"])
        LOG.info('spt_dual test     blacklisted stations: %d', len(blacklist5["Station"]) if blacklist5 else 0)

    #6. quality flag of DWH
    if 'DWH_flag' in tests_to_do:
        pi_col = c.par2pi.get(para, "")
        if df_pi is not None and pi_col and pi_col in df_pi.columns:
            pla_series = df_pi[pi_col]
        else:
            LOG.warning('DWH_flag: no pi column for %s, skipping', para)
            pla_series = pd.Series(dtype=float)
        blacklist6,freq=DWH_flag(current_f,para,stations,pla_series)
        nb6=len(my_dict['DWH_flag']['ID'])
        if blacklist6:
            nblack=len(blacklist6["Station"])
            for k in range(0,nblack):
                nb6=nb6+1
                my_d = {"ID":nb6,"Station":blacklist6["Station"][k],"Time":blacklist6["Time"][k],"Parameter":blacklist6["Parameter"][k]}
                my_dict['DWH_flag']["ID"].append(my_d["ID"])
                my_dict['DWH_flag']["Station"].append(my_d["Station"])
                my_dict['DWH_flag']["Time"].append(my_d["Time"])
                my_dict['DWH_flag']["Parameter"].append(my_d["Parameter"])
        LOG.info('DWH_flag test     blacklisted stations: %d', len(blacklist6["Station"]) if blacklist6 else 0)

    #8. plateau test
    if 'plateau_test' in tests_to_do:
        blacklist8=plateau_test(df_obs,c.plateau_test[para]['window'],c.plateau_test[para]['sd'],para,current_f,obs_path_in=obs_path_in,gran_minutes=c.plateau_test[para]['gran'])
        nb8=len(my_dict['plateau_test']['ID'])
        if blacklist8:
            nblack=len(blacklist8["Station"])
            for k in range(0,nblack):
                nb8=nb8+1
                my_d = {"ID":nb8,"Station":blacklist8["Station"][k],"Time":blacklist8["Time"][k],"Parameter":blacklist8["Parameter"][k]}
                my_dict['plateau_test']["ID"].append(my_d["ID"])
                my_dict['plateau_test']["Station"].append(my_d["Station"])
                my_dict['plateau_test']["Time"].append(my_d["Time"])
                my_dict['plateau_test']["Parameter"].append(my_d["Parameter"])
        LOG.info('plateau test blacklisted stations: %d', len(blacklist8["Station"]) if blacklist8 else 0)
    return my_dict,freq

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
            LOG.info('DWH flag %s (pi=%.3f)', station, flagged[station])
        LOG.info('DWH flag blacklisted stations: %d', len(flagged))
    else:
        my_dict = {}

    return my_dict, freq

#-------------------------------------------------------------------------------
#buddy check
def buddy_check(stations,lats,lons,elevs,values,para,date,threshold=4,max_elev_diff=200,elev_gradient=-0.0065,
                          min_std=1,num_iterations=5,num_min=3,radius=30000):
    """
    Performs a buddy check on meteorological stations.

    Args:
        stations (list): List of station names.
        lats (list): List of latitudes for each station.
        lons (list): List of longitudes for each station.
        elevs (list): List of elevations for each station.
        values (list): List of observed values for each station.
        para (str): Parameter name.
        date (str): Date of observation.
        threshold (float, optional): Threshold for the buddy check. Default is 4.
        max_elev_diff (float, optional): Maximum elevation difference for the buddy check. Default is 200.
        elev_gradient (float, optional): Elevation gradient for the buddy check. Default is -0.0065.
        min_std (float, optional): Minimum standard deviation for the buddy check. Default is 1.
        num_iterations (int, optional): Number of iterations for the buddy check. Default is 5.
        num_min (int, optional): Number of minimum neighbors for the buddy check. Default is 3.
        radius (float, optional): Radius for the buddy check. Default is 30000.

    Returns:
        dict: A dictionary containing the blacklisted stations and their corresponding information.
              The dictionary has the following keys: "Station", "Time", "Parameter".
              If no stations are blacklisted, an empty dictionary is returned.
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
    LOG.info('titanlib buddy_check threshold=%.2f (SMN)', thr_short)
    flags = titanlib.buddy_check(points, values, radius_arr, num_min_arr, thr_short,
                                 max_elev_diff, elev_gradient, min_std, num_iterations)
    for idx in np.nonzero(flags)[0]:
        if short_mask[idx]:
            flagged_indices.add(int(idx))
    # round 2: all stations, keep only other (longer-name) flags
    LOG.info('titanlib buddy_check threshold=%.2f (other)', thr_long)
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
            LOG.info('%s %s', snames[idx], values[idx])
            n += 1
    LOG.info('titanlib buddy_check blacklisted stations: %d', n)
    if len(my_dict["Station"]) == 0:
        my_dict = {}
    return my_dict

#first guess test
def first_guess_test(stations,lats,lons,elevs,values,background_values,para,date,background_elab_type=1,num_min_outer=3,num_max_outer=10,
                    inner_radius=50000,outer_radius=100000,num_iterations=10,num_min_prof=1,min_elev_diff=250,
                    min_horizontal_scale=250,max_horizontal_scale=100000,kth_closest_obs_horizontal_scale=2,
                    bdebug=True,bbasic=True,tpostneg=5):
    """
    Performs a first guess test on meteorological stations.

    Args:
        stations (list): List of station names.
        lats (list): List of latitudes for each station.
        lons (list): List of longitudes for each station.
        elevs (list): List of elevations for each station.
        values (list): List of observed values for each station.
        background_values (list): List of background values for each station.
        para (str): Parameter name.
        date (str): Date of observation.
        background_elab_type (int, optional): Background elaboration type. Default is 1.
        num_min_outer (int, optional): Number of minimum outer neighbors. Default is 3.
        num_max_outer (int, optional): Number of maximum outer neighbors. Default is 10.
        inner_radius (float, optional): Inner radius for the test. Default is 50000.
        outer_radius (float, optional): Outer radius for the test. Default is 100000.
        num_iterations (int, optional): Number of iterations for the test. Default is 10.
        num_min_prof (int, optional): Number of minimum vertical profile neighbors. Default is 1.
        min_elev_diff (float, optional): Minimum elevation difference for the test. Default is 250.
        min_horizontal_scale (float, optional): Minimum horizontal scale for the test. Default is 250.
        max_horizontal_scale (float, optional): Maximum horizontal scale for the test. Default is 100000.
        kth_closest_obs_horizontal_scale (int, optional): Kth closest observation horizontal scale. Default is 2.
        bdebug (bool, optional): Enable debugging output. Default is True.
        bbasic (bool, optional): Enable basic output. Default is True.
        tpostneg (int, optional): Threshold for positive/negative differences. Default is 5.

    Returns:
        dict: A dictionary containing the blacklisted stations and their corresponding information.
              The dictionary has the following keys: "Station", "Time", "Parameter".
              If no stations are blacklisted, an empty dictionary is returned.
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
    LOG.info('titanlib first_guess_test threshold=%.2f (SMN)', thr_short)
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
    LOG.info('titanlib first_guess_test threshold=%.2f (other)', thr_long)
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
            LOG.info('titanlib first_guess_test %s %s %.4f', snames[idx], para, values[idx])
            n += 1
    LOG.info('titanlib first_guess_test %s blacklisted stations: %d', para, n)
    return my_dict

#spacial concistency check resistant 
def spacial_ct_resistant(stations,lats,lons,elevs,values,para,date,background_elab_type=1,num_min_outer=3,num_max_outer=10,
                    inner_radius=50000,outer_radius=100000,num_iterations=10,num_min_prof=1,min_elev_diff=250,
                    min_horizontal_scale=250,max_horizontal_scale=100000,kth_closest_obs_horizontal_scale=2,
                    vertical_scale=200,bdebug=True,bbasic=True):
    """
    Performs a spacial consistency test resistant using the titanlib library on meteorological stations.

    Args:
        stations (list): List of station names.
        lats (list): List of latitudes for each station.
        lons (list): List of longitudes for each station.
        elevs (list): List of elevations for each station.
        values (array): Array of observed values.
        para (str): Parameter name.
        date (str): Date of the observation.
        background_elab_type (int): Background elaboration type.
        num_min_outer (int): Minimum number of outer stations.
        num_max_outer (int): Maximum number of outer stations.
        inner_radius (int): Inner radius for the spacial consistency test.
        outer_radius (int): Outer radius for the spacial consistency test.
        num_iterations (int): Number of iterations.
        num_min_prof (int): Minimum number of profiles.
        min_elev_diff (int): Minimum elevation difference.
        min_horizontal_scale (int): Minimum horizontal scale.
        max_horizontal_scale (int): Maximum horizontal scale.
        kth_closest_obs_horizontal_scale (int): Kth closest observation for horizontal scale.
        vertical_scale (int): Vertical scale.
        bdebug (bool): Debug flag.
        bbasic (bool): Basic flag.

    Returns:
        dict: A dictionary containing the blacklisted stations and their corresponding information.
              The dictionary has the following keys: "Station", "Time", "Parameter".
              If no stations are blacklisted, an empty dictionary is returned.
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
    LOG.info('spacial_ct_resistant')
    try:
        flags,scores = titanlib.sct_resistant(points, values, obs_to_check, background_values, 
                        background_elab_type, num_min_outer, num_max_outer, 
                        inner_radius, outer_radius, num_iterations, num_min_prof, 
                        min_elev_diff, min_horizontal_scale, max_horizontal_scale, 
                        kth_closest_obs_horizontal_scale, vertical_scale, 
                        values_mina, values_maxa, values_minv, values_maxv, 
                        eps2, tpos, tneg, bdebug, bbasic)
    except:
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
                LOG.info('titanlib sct_resistant'+' '+stations[indices[b]]+' '+str(values[indices[b]])+' '+str(scores[indices[b]]))
                n=n+1
        LOG.info('spacial_ct_resistant '+'blacklisted stations: '+str(n))
    else:
        my_dict={}
    return my_dict
#-------------------------------------------------------------------------------------------
#spacial concistency check dual
def spacial_ct_dual(stations,lats,lons,elevs,values,para,date,num_min_outer=3,num_max_outer=10,
                    inner_radius=50000,outer_radius=100000,num_iterations=10,
                    min_horizontal_scale=250,max_horizontal_scale=100000,kth_closest_obs_horizontal_scale=2,
                    vertical_scale=200,debug=True,condition = 1, event_thresholds = 0.1,test_thresholds = 0.8):
    
    """
    Performs a spacial consistency test using the titanlib library on meteorological stations.

    Args:
        stations (list): List of station names.
        lats (list): List of latitudes for each station.
        lons (list): List of longitudes for each station.
        elevs (list): List of elevations for each station.
        values (array): Array of observed values.
        para (str): Parameter name.
        date (str): Date of the observation.
        num_min_outer (int): Minimum number of outer stations.
        num_max_outer (int): Maximum number of outer stations.
        inner_radius (int): Inner radius for the spacial consistency test.
        outer_radius (int): Outer radius for the spacial consistency test.
        num_iterations (int): Number of iterations.
        min_horizontal_scale (int): Minimum horizontal scale.
        max_horizontal_scale (int): Maximum horizontal scale.
        kth_closest_obs_horizontal_scale (int): Kth closest observation for horizontal scale.
        vertical_scale (int): Vertical scale.
        debug (bool): Debug flag.
        condition (int): Condition for the spacial consistency test.
        event_thresholds (float): Event thresholds.
        test_thresholds (float): Test thresholds.

    Returns:
        dict: A dictionary containing the blacklisted stations and their corresponding information.
              The dictionary has the following keys: "Station", "Time", "Parameter".
              If no stations are blacklisted, an empty dictionary is returned.
    """
    points = titanlib.Points(lats, lons, elevs)
    npoints = len(lats)
    N=npoints
    obs_to_check = np.repeat(1, npoints)
    test_thresholds = np.repeat(test_thresholds, npoints)
    event_thresholds = np.repeat(event_thresholds, npoints)
    #print(locals())
    LOG.info('titanlib spacial_ct_dual')
    try:
        values=np.asarray(values, dtype=np.float64)
        flags= titanlib.sct_dual(points, values, obs_to_check, event_thresholds, condition,
                        num_min_outer, num_max_outer, inner_radius, outer_radius,
                        num_iterations, min_horizontal_scale, max_horizontal_scale,
                        kth_closest_obs_horizontal_scale, vertical_scale,
                        test_thresholds, debug)
    except:
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
                LOG.info('titanlib sct_dual'+' '+stations[indices[b]]+' '+str(values[indices[b]]))
                n=n+1
        LOG.info('titanlib spacial_ct_dual '+'blacklisted stations: '+str(n))
    else:
        my_dict={}
    return my_dict
#--------------------------------------------------------------
def find_indices(list_to_check, item_to_find):
    """
    Find the indices of occurrences of 'item_to_find' in the given list 'list_to_check'.

    This function searches for all occurrences of 'item_to_find' in the 'list_to_check' and returns
    a list containing the indices of those occurrences.

    Parameters:
        list_to_check (list): The list in which to search for the 'item_to_find'.
        item_to_find: The item to find occurrences of in the 'list_to_check'.

    Returns:
        list: A list of integers representing the indices of occurrences of 'item_to_find' in 'list_to_check'.

    Note:
        If 'item_to_find' is not present in 'list_to_check', an empty list will be returned.
    """
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]

#--------------------------------------------------------------
def tests_summary (blacklist,weights,tests):
    """
    Calculate the summary of tests for each station and time from a given blacklist dictionary.

    This function takes a JSON-like dictionary 'mydict' containing information about blacklisted tests
    for different stations and times. It extracts the relevant data, processes it, and returns a DataFrame
    containing the quality control (QC) summary for each station and time.

    Parameters:
        blacklist (dict): A JSON-like dictionary containing blacklisted tests information.

    Returns:
        pandas.DataFrame: A DataFrame containing the quality control (QC) summary for each station and time.
                          The rows represent the different time values, and the columns represent the unique
                          station names. The value at df.loc[time, station] represents the ratio of blacklisted
                          tests for the station at the specified time to the total number of tests. This ratio
                          is calculated by counting the number of occurrences of a station-time pair in the
                          blacklisted tests and dividing it by the total number of tests.

    Note:
        This function relies on the 'find_indices' function, which should be defined before calling
        'tests_summary' or imported from another module.
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
    """
    Perform a hard test on the data within the given limits and record the observations that fall outside.

    Parameters:
    obs_all (DataFrame): The DataFrame containing all observations.
    datamin (float): The minimum threshold for the data.
    datamax (float): The maximum threshold for the data.
    var (str): The parameter being tested.
    time (str): The time of the observation.

    Returns:
    dict: A dictionary containing the stations, times, and parameters for the observations that fall outside the given limits.

    The function performs a hard test on the 'obs_all' DataFrame, checking the 'var' parameter against the 'datamin' and 'datamax' limits.
    For each station, if the 'var' parameter is outside the given limits, the station name, time, and parameter details are logged.
    The function then returns a dictionary 'my_dict' containing the recorded information for the observations that failed the test.

    """

    LOG.info('Hard test '+ var + ' '+time +' '+str(datamin) +'/'+ str(datamax))
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
                LOG.info('Hard test '+var+' '+stations[n]+' '+str(sta_data.iloc[0]))
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
        dict: {'Station': [...], 'Time': [...], 'Parameter': [...]} for flagged stations.
            Empty dict when nothing is flagged or the test is skipped.
    """
    LOG.info('plateau test for %s', var)
    my_dict = {"Station": [], "Time": [], "Parameter": []}

    if obs_path_in is None:
        LOG.warning('plateau_test %s: obs_path_in not provided, cannot run test', var)
        return my_dict

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
        return my_dict
    file_ts = ts_match.group()
    current = datetime.strptime(file_ts, '%Y%m%d%H%M')
    stem_pattern = obs_path_in.stem.replace(file_ts, '{}')

    historical_dfs = []
    missing = []
    for step in range(1, n_steps + 1):
        t = current - timedelta(minutes=step * gran_minutes)
        fn = obs_path_in.parent / (stem_pattern.format(t.strftime('%Y%m%d%H%M')) + obs_path_in.suffix)
        if fn.exists():
            try:
                historical_dfs.append(pd.read_parquet(fn))
            except Exception as e:
                LOG.warning('plateau_test: could not read %s: %s', fn, e)
                missing.append(str(fn))
        else:
            missing.append(str(fn))

    if missing:
        for mf in missing:
            LOG.warning('plateau_test %s: missing file: %s', var, mf)

    if not historical_dfs:
        LOG.warning(
            'plateau_test %s: cannot run — 0/%d historical files available (window=%dh, gran=%dmin)',
            var, n_steps, window, gran_minutes,
        )
        return my_dict
    if len(missing) > 0:
        LOG.warning(
            'plateau_test %s: running on partial history (%d/%d files available)',
            var, n_steps - len(missing), n_steps,
        )

    obs_all = pd.concat([data] + historical_dfs)
    stations = data['sta_name']
    for s in stations:
        sta_data = data[data['sta_name'] == s][var]
        if not sta_data.isna().item():
            sd = obs_all[obs_all['sta_name'] == s][var].std()
            nn = obs_all[obs_all['sta_name'] == s][var].shape[0]
            if sd <= std_lim and nn > n_steps / 2:
                LOG.warning('plateau_test %s %s std=%.4f', var, s, sd)
                my_dict["Station"].append(s)
                my_dict["Time"].append(time)
                my_dict["Parameter"].append(var)
    LOG.info('plateau_test %s: %d stations flagged', var, len(my_dict["Station"]))
    return my_dict
