#configurations titanlib and QC tests to be done
#test to be applied ('buddy_obs','buddy_diff','fgt','spt_resistant','spt_dual','DWH_flag','hard_test','plateau_test')

#TITANLIB e other tests
par2check=['T_2M','TD_2M',"RH_2M","FF_10M","VMAX10M","T_G"]

# QC parameter -> parquet plausibility column (written by RetrieveObservation as {param}_pi)
par2pi = {
    'T_2M':    '2t_pi',
    'TD_2M':   '2d_pi',
    'FF_10M':  'ff_pi',
    'VMAX10M': 'vmax_pi',
}

#DWH plausibility test configuration
dwh_force='True'
DWH_flag={
   "T_2M" : {
        'dwh_plausibility_thr' : 0.7,
        'time_window' : 60
   },
   "TD_2M" : {
        'dwh_plausibility_thr' : 0.7,
        'time_window' : 60
   },
   "T_G" : {
        'dwh_plausibility_thr' : 0.7,
        'time_window' : 60
   },
    "FF_10M" : {
        'dwh_plausibility_thr' : 0.7,
        'time_window' : 60
   },
   "RH_2M" : {
        'dwh_plausibility_thr' : 0.7,
        'time_window' : 60
   },
    "VMAX10M" : {
        'dwh_plausibility_thr' : 0.7,
        'time_window' : 60
   },
   
}
#TITALIB and other tests configuration
#available test: 'hard','buddy_obs','buddy_diff','fgt','spt_resistant','spt_dual','plateau_test', 'cosmo_test'
#tests_QC list (tests) has to have the same length as tests_QC_w (weights)
titan_ntests_threshold={
   "T_2M" : {
        'threshold_summary': 0.2, #npos/ntot positive tests over tot tests
        'tests_QC': ['hard','buddy_obs','buddy_diff','fgt','plateau_test','DWH_flag'],
        'tests_QC_w': [2,1,1,1,1,1], 
        'tests_QC2': ['DWH_flag']
   },
   "TD_2M" : {
        'threshold_summary': 0.2,
        'tests_QC': ['hard','buddy_obs','buddy_diff','fgt','plateau_test','DWH_flag'],
        'tests_QC_w': [2,1,1,1,1,1],
        'tests_QC2': ['DWH_flag']
   },
   "T_G" : {
        'threshold_summary': 0.2,
        'tests_QC': ['hard','buddy_obs','buddy_diff','fgt','plateau_test','DWH_flag'],
        'tests_QC_w': [2,1,1,1,1,1],
        'tests_QC2': ['DWH_flag']
   },
    "FF_10M" : {
        'threshold_summary': 0.2,
        'tests_QC': ['hard','buddy_obs','buddy_diff','fgt','plateau_test','DWH_flag'],
        'tests_QC_w': [2,1,1,1,2,1],
        'tests_QC2': ['DWH_flag']
   },
   "VMAX10M" : {
        'threshold_summary': 0.2,
        'tests_QC': ['hard','buddy_obs','buddy_diff','fgt','plateau_test','DWH_flag'],
        'tests_QC_w': [2,1,1,1,2,1],
        'tests_QC2': ['DWH_flag']
   },
   "RH_2M" : {
        'threshold_summary': 0.2,
        'tests_QC': ['hard','buddy_obs','buddy_diff','fgt','plateau_test','DWH_flag'],
        'tests_QC_w': [2,1,1,1,1,1],
        'tests_QC2': ['DWH_flag']
   },
}

# configuration of hard test
obs_variables = ['T_2M', 'TD_2M', 'RH_2M', 'FF_10M', 'VMAX10M', 'T_G']
plausibility_thresholds = {'variable': obs_variables,
                            #          T_2M   TD_2M  RH_2M  FF_10M  VMAX10M   T_G
                            'pch_min': [223.15, 203.15,   2,     0,       0, 223.15],
                            'pch_max': [323.15, 323.15, 100,    80,      80, 343.15]}
                            
#configuation of plateau test in order to identify frozen instruments
plateau_test={
    "T_2M" : {
        "window" : 24, #hours in the past
        "gran" : 360,  #minutes (modulo 10),granularity
        "sd" : 0    #max standard deviation to identify plateaus
    },
    "TD_2M" : {
        "window" : 24,
        "gran" : 360,
        "sd" : 0
    },
    "RH_2M" : {
        "window" : 48,
        "gran" : 360,
        "sd" : 0
    },
    "T_G" : {
        "window" : 24,
        "gran" : 360,
        "sd" : 0
    },
    "FF_10M" : {
        "window" : 24,
        "gran" :30,
        "sd" : 0
    },
    "VMAX10M" : {
        "window" : 24,
        "gran" :30,
        "sd" : 0
    }
}

#buddy check configuration
buddy= {
    "T_2M" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [4.5,3.5],  # [SMN, other stations]
        "max_elev_diff" : 500,
        "elev_gradient" : -0.0065,
        "min_std" : 2,
        "num_iterations" : 6
    },
    "TD_2M" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [6.0,5.0],  # [SMN, other stations]
        "max_elev_diff" : 700,
        "elev_gradient" : -0.0065,
        "min_std" : 2,
        "num_iterations" : 6
    },
    "T_G" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [6.0,5.0],  # [SMN, other stations]
        "max_elev_diff" : 500,
        "elev_gradient" : -0.0065,
        "min_std" : 2,
        "num_iterations" : 6
    },
    "FF_10M" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [9.5,8.5],  # [SMN, other stations]
        "max_elev_diff" : 500,
        "elev_gradient" : 0.0015,
        "min_std" : 2,
        "num_iterations" : 6
    },
    "RH_2M" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [15,10],    # [SMN, other stations]
        "max_elev_diff" : 700,
        "elev_gradient" : 0,
        "min_std" : 1.5,
        "num_iterations" : 6
    },
    "VMAX10M" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [9.5,8.5],  # [SMN, other stations]
        "max_elev_diff" : 500,
        "elev_gradient" : 0.0015,
        "min_std" : 2,
        "num_iterations" : 6
    },
}
buddy_diff= {
    "T_2M" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [5.0, 4.0],  # [SMN, other stations]
        "max_elev_diff" : 500,
        "elev_gradient" : 0,
        "min_std" : 2,
        "num_iterations" : 6
    },
    "TD_2M" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [6.5, 5.5],  # [SMN, other stations]
        "max_elev_diff" : 700,
        "elev_gradient" : 0,
        "min_std" : 2,
        "num_iterations" : 6
    },
    "T_G" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [6.0, 5.0],  # [SMN, other stations]
        "max_elev_diff" : 500,
        "elev_gradient" : 0,
        "min_std" : 2,
        "num_iterations" : 6
    },
    "FF_10M" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [9.0, 8.0],  # [SMN, other stations]
        "max_elev_diff" : 500,
        "elev_gradient" : 0,
        "min_std" : 2,
        "num_iterations" : 6
    },
    "RH_2M" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [15, 10],    # [SMN, other stations]
        "max_elev_diff" : 700,
        "elev_gradient" : 0,
        "min_std" : 1.5,
        "num_iterations" : 6
    },
    "VMAX10M" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [9.5, 8.5],  # [SMN, other stations]
        "max_elev_diff" : 500,
        "elev_gradient" : 0,
        "min_std" : 2,
        "num_iterations" : 6
    },
}
#first guess test configuration
fgt= {
    "T_2M" : {
        'background_elab_type' : 3, #1"VerticalProfileTheilSen",3 external data
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 100000,
        'outer_radius' : 200000,
        'num_iterations' : 10,
        'num_min_prof' : 0,
        'min_elev_diff' : 500,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 300000,
        'kth_closest_obs_horizontal_scale' : 2,
        'debug' : False,
        'basic' : True,
        'tpostneg' : [5.5, 4.5],  # [SMN, other stations]
    },
    "TD_2M" : {
        'background_elab_type' : 3, #1"VerticalProfileTheilSen",3 external data
        'num_min_outer' : 2,
        'num_max_outer' : 15,
        'inner_radius' : 100000,
        'outer_radius' : 200000,
        'num_iterations' : 10,
        'num_min_prof' : 0,
        'min_elev_diff' : 700,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 300000,
        'kth_closest_obs_horizontal_scale' : 2,
        'debug' : False,
        'basic' : True,
        'tpostneg' : [7.0, 6.0],  # [SMN, other stations]
    },
    "T_G" : {
        'background_elab_type' : 3, #1"VerticalProfileTheilSen",3 external data
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 100000,
        'outer_radius' : 200000,
        'num_iterations' : 10,
        'num_min_prof' : 0,
        'min_elev_diff' : 500,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 300000,
        'kth_closest_obs_horizontal_scale' : 2,
        'debug' : False,
        'basic' : True,
        'tpostneg' : [6.5, 5.5],  # [SMN, other stations]
    },
    "FF_10M" : {
        'background_elab_type' : 3, #1"VerticalProfileTheilSen",3 external data
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 100000,
        'outer_radius' : 200000,
        'num_iterations' : 10,
        'num_min_prof' : 0,
        'min_elev_diff' : 500,
        'min_horizontal_scale' : 400,
        'max_horizontal_scale' : 300000,
        'kth_closest_obs_horizontal_scale' : 2,
        'debug' : False,
        'basic' : True,
        'tpostneg' : [10, 9],     # [SMN, other stations]
    },
     "RH_2M" : {
        'background_elab_type' : 3, #1"VerticalProfileTheilSen",3 external data
        'num_min_outer' : 2,
        'num_max_outer' : 15,
        'inner_radius' : 100000,
        'outer_radius' : 200000,
        'num_iterations' : 10,
        'num_min_prof' : 0,
        'min_elev_diff' : 700,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 300000,
        'kth_closest_obs_horizontal_scale' : 2,
        'debug' : False,
        'basic' : True,
        'tpostneg' : [15, 12],    # [SMN, other stations]
    },
    "VMAX10M" : {
        'background_elab_type' : 3, #1"VerticalProfileTheilSen",3 external data
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 100000,
        'outer_radius' : 200000,
        'num_iterations' : 10,
        'num_min_prof' : 0,
        'min_elev_diff' : 500,
        'min_horizontal_scale' : 400,
        'max_horizontal_scale' : 300000,
        'kth_closest_obs_horizontal_scale' : 2,
        'debug' : False,
        'basic' : True,
        'tpostneg' : [10.5, 9.5], # [SMN, other stations]
    },
}
#Spacial consistency check resistant configuration
spt_resistant = {
    "T_2M" : {
        'background_elab_type' : 1,  #"VerticalProfileTheilSen"
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 100000,
        'outer_radius' : 200000,
        'num_iterations' : 10,
        'num_min_prof' : 1,
        'min_elev_diff' : 500,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 300000,
        'kth_closest_obs_horizontal_scale' : 2,
        'vertical_scale' : 300,
        'debug' : False,
        'basic' : True
    },
    "TD_2M" : {
        'background_elab_type' : 1,  #"VerticalProfileTheilSen"
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 50000,
        'outer_radius' : 100000,
        'num_iterations' : 10,
        'num_min_prof' : 1,
        'min_elev_diff' : 700,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 150000,
        'kth_closest_obs_horizontal_scale' : 2,
        'vertical_scale' : 300,
        'debug' : False,
        'basic' : False
    },
    "T_G" : {
        'background_elab_type' : 1,  #"VerticalProfileTheilSen"
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 50000,
        'outer_radius' : 100000,
        'num_iterations' : 10,
        'num_min_prof' : 1,
        'min_elev_diff' : 500,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 150000,
        'kth_closest_obs_horizontal_scale' : 2,
        'vertical_scale' : 300,
        'debug' : False,
        'basic' : False
    },
    "FF_10M" : {
        'background_elab_type' : 1,  #"VerticalProfileTheilSen"
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 50000,
        'outer_radius' : 100000,
        'num_iterations' : 10,
        'num_min_prof' : 1,
        'min_elev_diff' : 500,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 150000,
        'kth_closest_obs_horizontal_scale' : 2,
        'vertical_scale' : 400,
        'debug' : False,
        'basic' : False
    },
    "RH_2M" : {
        'background_elab_type' : 1,  #"VerticalProfileTheilSen"
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 50000,
        'outer_radius' : 100000,
        'num_iterations' : 10,
        'num_min_prof' : 1,
        'min_elev_diff' : 700,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 150000,
        'kth_closest_obs_horizontal_scale' : 2,
        'vertical_scale' : 300,
        'debug' : False,
        'basic' : False
    },  
    "VMAX10M" : {
        'background_elab_type' : 1,  #"VerticalProfileTheilSen"
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 50000,
        'outer_radius' : 100000,
        'num_iterations' : 10,
        'num_min_prof' : 1,
        'min_elev_diff' : 500,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 150000,
        'kth_closest_obs_horizontal_scale' : 2,
        'vertical_scale' : 400,
        'debug' : False,
        'basic' : False
    },
}
#Spacial consistency check dual configuration
sct_dual = {
    "T_2M" : {
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 100000,
        'outer_radius' : 200000,
        'num_iterations' : 10,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 300000,
        'kth_closest_obs_horizontal_scale' : 2,
        'vertical_scale' : 1000,
        'debug' : False,
        'condition' : 0,
        'event_thresholds' : 0.05,
        'test_thresholds' : 0.95
    },
    "TD_2M" : {
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 50000,
        'outer_radius' : 100000,
        'num_iterations' : 10,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 100000,
        'kth_closest_obs_horizontal_scale' : 2,
        'vertical_scale' : 300,
        'debug' : False,
        'condition' : 0,
        'event_thresholds' : 0.05,
        'test_thresholds' : 0.95
    },
    "T_G" : {
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 50000,
        'outer_radius' : 100000,
        'num_iterations' : 10,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 100000,
        'kth_closest_obs_horizontal_scale' : 2,
        'vertical_scale' : 300,
        'debug' : False,
        'condition' : 0,
        'event_thresholds' : 0.05,
        'test_thresholds' : 0.95
    },
    "FF_10M" : {
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 50000,
        'outer_radius' : 100000,
        'num_iterations' : 10,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 100000,
        'kth_closest_obs_horizontal_scale' : 2,
        'vertical_scale' : 400,
        'debug' : False,
        'condition' : 0,
        'event_thresholds' : 0.05,
        'test_thresholds' : 0.95
    },
    "RH_2M" : {
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 50000,
        'outer_radius' : 100000,
        'num_iterations' : 10,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 100000,
        'kth_closest_obs_horizontal_scale' : 2,
        'vertical_scale' : 300,
        'debug' : False,
        'condition' : 0,
        'event_thresholds' : 0.05,
        'test_thresholds' : 0.95
    },
    "VMAX10M" : {
        'num_min_outer' : 2,
        'num_max_outer' : 10,
        'inner_radius' : 50000,
        'outer_radius' : 100000,
        'num_iterations' : 10,
        'min_horizontal_scale' : 250,
        'max_horizontal_scale' : 100000,
        'kth_closest_obs_horizontal_scale' : 2,
        'vertical_scale' : 400,
        'debug' : False,
        'condition' : 0,
        'event_thresholds' : 0.05,
        'test_thresholds' : 0.95
    }
}

stations_excluded = { #list of stations to be excluded from the blacklisting of values e.g  ['BRL','COM']
    "T_2M" : {'stations':['BRL','VSSOR']},
    "TD_2M" : {'stations':['BRL']},
    "T_G" : {'stations':[]},
    "RH_2M" : {'stations':[]},
    "FF_10M" : {'stations':[]},
    "VMAX10M" : {'stations':[]}
}

#stations to be blacklisted anyway example: '1':  {'station': 'WNSDOR', 'paras': ['FF_10M','VMAX10M']}
hard_blacklist = {
    1:  {'station': 'WNSSAL', 'paras': ['FF_10M','VMAX10M','DD']},
    2:  {'station': 'IYABT', 'paras': ['T_2M', 'TD_2M']},
    3:  {'station': 'IYANT', 'paras': ['T_2M', 'TD_2M']},
    4:  {'station': 'IYAUE', 'paras': ['T_2M', 'TD_2M']},
    5:  {'station': 'IYBAR', 'paras': ['T_2M', 'TD_2M']},
    6:  {'station': 'IYBOZ', 'paras': ['T_2M', 'TD_2M']},
    7:  {'station': 'IYBRU', 'paras': ['T_2M', 'TD_2M']},
    8:  {'station': 'IYDEU', 'paras': ['T_2M', 'TD_2M']},
    9:  {'station': 'IYFRA', 'paras': ['T_2M', 'TD_2M']},
    10: {'station': 'IYGAR', 'paras': ['T_2M', 'TD_2M']},
    11: {'station': 'IYGRE', 'paras': ['T_2M', 'TD_2M']},
    12: {'station': 'IYHIN', 'paras': ['T_2M', 'TD_2M']},
    13: {'station': 'IYJAU', 'paras': ['T_2M', 'TD_2M']},
    14: {'station': 'IYKAL', 'paras': ['T_2M', 'TD_2M']},
    15: {'station': 'IYLAA', 'paras': ['T_2M', 'TD_2M']},
    16: {'station': 'IYLAD', 'paras': ['T_2M', 'TD_2M']},
    17: {'station': 'IYLVA', 'paras': ['T_2M', 'TD_2M']},
    18: {'station': 'IYMEL', 'paras': ['T_2M', 'TD_2M']},
    19: {'station': 'IYMER', 'paras': ['T_2M', 'TD_2M']},
    20: {'station': 'IYMMR', 'paras': ['T_2M', 'TD_2M']},
    21: {'station': 'IYMUW', 'paras': ['T_2M', 'TD_2M']},
    22: {'station': 'IYNAT', 'paras': ['T_2M', 'TD_2M']},
    23: {'station': 'IYOBV', 'paras': ['T_2M', 'TD_2M']},
    24: {'station': 'IYPEN', 'paras': ['T_2M', 'TD_2M']},
    25: {'station': 'IYPFD', 'paras': ['T_2M', 'TD_2M']},
    26: {'station': 'IYPFE', 'paras': ['T_2M', 'TD_2M']},
    27: {'station': 'IYPFG', 'paras': ['T_2M', 'TD_2M']},
    28: {'station': 'IYPFI', 'paras': ['T_2M', 'TD_2M']},
    29: {'station': 'IYPFR', 'paras': ['T_2M', 'TD_2M']},
    30: {'station': 'IYPRL', 'paras': ['T_2M', 'TD_2M']},
    31: {'station': 'IYPRR', 'paras': ['T_2M', 'TD_2M']},
    32: {'station': 'IYRAT', 'paras': ['T_2M', 'TD_2M']},
    33: {'station': 'IYRBW', 'paras': ['T_2M', 'TD_2M']},
    34: {'station': 'IYRIT', 'paras': ['T_2M', 'TD_2M']},
    35: {'station': 'IYROA', 'paras': ['T_2M', 'TD_2M']},
    36: {'station': 'IYSAN', 'paras': ['T_2M', 'TD_2M']},
    37: {'station': 'IYSAR', 'paras': ['T_2M', 'TD_2M']},
    38: {'station': 'IYSAU', 'paras': ['T_2M', 'TD_2M']},
    39: {'station': 'IYSCH', 'paras': ['T_2M', 'TD_2M']},
    40: {'station': 'IYSEI', 'paras': ['T_2M', 'TD_2M']},
    41: {'station': 'IYSFI', 'paras': ['T_2M', 'TD_2M']},
    42: {'station': 'IYSGR', 'paras': ['T_2M', 'TD_2M']},
    43: {'station': 'IYSIG', 'paras': ['T_2M', 'TD_2M']},
    44: {'station': 'IYSKL', 'paras': ['T_2M', 'TD_2M']},
    45: {'station': 'IYSMP', 'paras': ['T_2M', 'TD_2M']},
    46: {'station': 'IYSMT', 'paras': ['T_2M', 'TD_2M']},
    47: {'station': 'IYSTG', 'paras': ['T_2M', 'TD_2M']},
    48: {'station': 'IYSTZ', 'paras': ['T_2M', 'TD_2M']},
    49: {'station': 'IYSUL', 'paras': ['T_2M', 'TD_2M']},
    50: {'station': 'IYSUM', 'paras': ['T_2M', 'TD_2M']},
    51: {'station': 'IYSUS', 'paras': ['T_2M', 'TD_2M']},
    52: {'station': 'IYSVE', 'paras': ['T_2M', 'TD_2M']},
    53: {'station': 'IYSVP', 'paras': ['T_2M', 'TD_2M']},
    54: {'station': 'IYSWA', 'paras': ['T_2M', 'TD_2M']},
    55: {'station': 'IYTAU', 'paras': ['T_2M', 'TD_2M']},
    56: {'station': 'IYTER', 'paras': ['T_2M', 'TD_2M']},
    57: {'station': 'IYTRZ', 'paras': ['T_2M', 'TD_2M']},
    58: {'station': 'IYULT', 'paras': ['T_2M', 'TD_2M']},
    59: {'station': 'IYULW', 'paras': ['T_2M', 'TD_2M']},
    60: {'station': 'IYVAA', 'paras': ['T_2M', 'TD_2M']},
    61: {'station': 'IYVAL', 'paras': ['T_2M', 'TD_2M']},
    62: {'station': 'IYVOL', 'paras': ['T_2M', 'TD_2M']},
    63: {'station': 'IYWEI', 'paras': ['T_2M', 'TD_2M']},
    64: {'station': 'IYWEN', 'paras': ['T_2M', 'TD_2M']},
    65: {'station': 'IYWOL', 'paras': ['T_2M', 'TD_2M']},
}
hard_blacklist = {}


