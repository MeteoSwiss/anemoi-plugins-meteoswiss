#configurations titanlib and QC tests to be done
#test to be applied ('buddy_obs','buddy_diff','fgt','spt_resistant','spt_dual','DWH_flag','hard_test','plateau_test')

#TITANLIB and other tests
par2check=['T_2M','TD_2M',"RH_2M","FF_10M","VMAX10M","T_G","PS","PMSL"]

# QC parameter -> parquet plausibility column (written by RetrieveObservation as {param}_pi)
par2pi = {
    'T_2M':    '2t_pi',
    'TD_2M':   '2d_pi',
    'FF_10M':  'ff_pi',
    'VMAX10M': 'vmax_pi',
    'PS':      'sp_pi',
    'PMSL':    'msl_pi',
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
   "PS" : {
        'dwh_plausibility_thr' : 0.7,
        'time_window' : 60
   },
   "PMSL" : {
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
   "PS" : {
        'threshold_summary': 0.2,
        'tests_QC': ['hard','buddy_obs','buddy_diff','fgt','plateau_test','DWH_flag'],
        'tests_QC_w': [2,1,1,1,1,1],
        'tests_QC2': ['DWH_flag']
   },
   "PMSL" : {
        'threshold_summary': 0.2,
        'tests_QC': ['hard','buddy_obs','buddy_diff','fgt','plateau_test','DWH_flag'],
        'tests_QC_w': [2,1,1,1,1,1],
        'tests_QC2': ['DWH_flag']
   },
}

# configuration of hard test
obs_variables = ['T_2M', 'TD_2M', 'RH_2M', 'FF_10M', 'VMAX10M', 'T_G', 'PS', 'PMSL']
plausibility_thresholds = {'variable': obs_variables,
                            #          T_2M   TD_2M  RH_2M  FF_10M  VMAX10M   T_G      PS      PMSL   (Pa for pressure)
                            'pch_min': [223.15, 203.15,   2,     0,       0, 223.15,  50000,  87000],
                            'pch_max': [323.15, 323.15, 100,    80,      80, 343.15, 108000, 108500]}
                            
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
        "gran" :60,
        "sd" : 0
    },
    "VMAX10M" : {
        "window" : 24,
        "gran" :60,
        "sd" : 0
    },
    "PS" : {
        "window" : 24,
        "gran" : 360,
        "sd" : 0
    },
    "PMSL" : {
        "window" : 24,
        "gran" : 360,
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
    "PS" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [1000, 800],  # [SMN, other stations]  (Pa)
        "max_elev_diff" : 500,
        "elev_gradient" : -12.0,  # Pa/m standard atmosphere (dp/dz ≈ -ρg)
        "min_std" : 200,
        "num_iterations" : 6
    },
    "PMSL" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [500, 400],   # [SMN, other stations]  (Pa)
        "max_elev_diff" : 500,
        "elev_gradient" : 0,
        "min_std" : 100,
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
    "PS" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [1000, 800],  # [SMN, other stations]  (Pa)
        "max_elev_diff" : 500,
        "elev_gradient" : 0,
        "min_std" : 200,
        "num_iterations" : 6
    },
    "PMSL" : {
        "radius" : 150000,
        "num_min" :  2,
        "threshold" : [500, 400],   # [SMN, other stations]  (Pa)
        "max_elev_diff" : 500,
        "elev_gradient" : 0,
        "min_std" : 100,
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
    "PS" : {
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
        'tpostneg' : [1500, 1200],  # [SMN, other stations]  (Pa)
    },
    "PMSL" : {
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
        'tpostneg' : [800, 600],    # [SMN, other stations]  (Pa)
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
    "PS" : {
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
    "PMSL" : {
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
    },
    "PS" : {
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
    "PMSL" : {
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
    }
}

stations_excluded = { #list of stations to be excluded from the blacklisting of values e.g  ['BRL','COM']
    "T_2M" : {'stations':['BRL','VSSOR']},
    "TD_2M" : {'stations':['BRL']},
    "T_G" : {'stations':[]},
    "RH_2M" : {'stations':[]},
    "FF_10M" : {'stations':[]},
    "VMAX10M" : {'stations':[]},
    "PS" : {'stations':[]},
    "PMSL" : {'stations':[]}
}

#stations to be blacklisted anyway example: '1':  {'station': 'WNSDOR', 'paras': ['FF_10M','VMAX10M']}
hard_blacklist = {
    1:  {'station': 'WNSSAL', 'paras': ['FF_10M','VMAX10M','DD']},
}


