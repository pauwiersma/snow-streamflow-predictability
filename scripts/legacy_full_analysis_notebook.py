"""LEGACY reference script (machine-local Prior workflow).

This is a lightly path-sanitized copy of the original analysis notebook used to
produce the WRR Paper 2 figures. It still depends on the private ewc model-run
stack (Evaluation, Prior outputs, GIS layers, etc.) and is NOT the recommended
entry point for reproducing published figures.

Preferred public workflow:
  1. Download Zenodo analysis tables into data/tables/
  2. python scripts/02_make_figures.py

Set SSP_DATA_ROOT / DATA_ROOT to your data directory if running this legacy script.
"""


import os
os.environ["NUMBA_THREADING_LAYER"] = "tbb"
os.chdir(os.environ.get("SSP_CODE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Standard library imports
import concurrent.futures
import glob
import math
import pickle
import warnings
from functools import partial
from os.path import join
from typing import Tuple, Union

# Third-party imports
import geopandas as gpd
import hydrosignatures as hs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from flexitext import flexitext
from highlight_text import fig_text
from hydrosignatures import HydroSignatures
from matplotlib import patches
from matplotlib.colors import LightSource
from matplotlib.dates import DateFormatter, date2num
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from pypalettes import load_cmap
import colorcet as cc
from rasterio.mask import mask
from rasterio.plot import show
from scipy.stats import linregress, pearsonr, variation, zscore
from spotpy import analyser as spa

# Local imports
from Evaluation import *
from Postruns import *
from SnowClass import *
from Synthetic_obs import Synthetic_obs
from spotpy_analysis import spotpy_analysis
from spotpy_calib import SoilCalib
from swe_metrics import *
from wflow_spot_setup import *
from posterior_SWE_analysis import *
from q_metrics import calc_snowmeltseason_sum, calc_snowmeltseason_cv, calc_hfd_meltseason
import HydroErr as he

config_dir = os.environ.get("SSP_CONFIG_DIR", os.path.join(os.environ.get("SSP_DATA_ROOT", os.environ.get("DATA_ROOT", "./data")), "experiments", "config_files"))
# # config_dir="/work/FAC/FGSE/IDYST/gmariet1/gaia/pwiersma/ewatercycle/experiments/config_files"
ROOTDIR = os.environ.get("SSP_DATA_ROOT", os.environ.get("DATA_ROOT", "./data"))
# Flexible experiment configuration
# Define experiments and their display names in the order you want them processed
# Colors will be automatically assigned based on the order defined here

##################
"""WUS"""
#################



EXPERIMENT_CONFIG = {
    'AAAA_14': 'D0a',
    'AEAA_14': 'D0b',
    'ABAA_14': 'D0c',
    'ADAA_14': 'D0d',
    'AADA_14': 'D0.5a',
    'AEDA_14': 'D0.5b',    
    'ABDA_14': 'D0.5c',
    'ADDA_14': 'D0.5d',
    'AABA_14': 'D1a',
    'AEBA_14': 'D1b',
    'ABBA_14': 'D1c',
    'ADBA_14': 'D1d',
    'AACA_14': 'D2a',
    'AECA_14': 'D2b',
    'ABCA_14': 'D2c',
    'ADCA_14': 'D2d',
    # 'AAAA_14': 'D0a_new',
    # 'ABBA_14': 'D1c_new'
    # 'ABBA_23': 'D23',
    # 'ABBA_14_WUS_Sauk': 'Sauk',
    # 'ABBA_14_WUS_Payette': 'Payette',
    # 'ABBA_14_WUS_Andrews': 'Andrews',
    # 'ABBA_14_WUS_American_River_Near_Nile': 'American_River_Near_Nile',
    # 'ABBA_14_WUS_Cole': 'Cole',
    # 'ABBA_14_WUS_Pitman': 'Pitman',
    # 'ABBA_14_WUS_Encampment': 'Encampment',
    # 'ABBA_14_WUS_Vallecito': 'Vallecito',
    # 'ABBA_14_WUS_Cedar': 'Cedar',
    # 'ABBA_63_WUS_Merced': 'Merced',
    # 'ABBA_14_WUS_Merced_HI': 'Merced_HI_14',
    # 'ABBA_194_WUS_Merced_HI': 'Merced_HI_194',
    # 'ABBA_214_WUS_Merced_HI': 'Merced_HI',
    # 'ABBA_214_WUS_North_Brush': 'North_Brush',
    # 'ABBA_214_WUS_Encampment': 'Encampment',
    # 'ABBA_214_WUS_King_Canyon': 'King_Canyon',
    # 'ABBA_214_WUS_Halfmoon': 'Halfmoon',
    # 'ABBA_214_WUS_Black_Gore': 'Black_Gore',
    # 'ABBA_214_WUS_Vallecito': 'Vallecito',
    # 'ABBA_214_WUS_Beaver': 'Beaver',
    # 'ABBA_214_WUS_Pitman': 'Pitman',
    # 'ABBA_214_WUS_Cole': 'Cole',
    # 'ABBA_214_WUS_Cedar': 'Cedar',
    # 'ABBA_214_WUS_Sauk': 'Sauk',
    # 'ABBA_214_WUS_Arlee': 'Arlee',
    # 'ABBA_214_WUS_Andrews': 'Andrews',
    # 'ABBA_214_WUS_American_River_Near_Nile': 'American_River_Near_Nile',
    # 'ABBA_214_WUS_Cache': 'Cache',
    # 'ABBA_214_WUS_Payette': 'Payette',
    # 'ABBA_214_WUS_Umatilla': 'Umatilla',
    # 'ABBA_214_WUS_Shitike': 'Shitike',
    # 'ABBA_214_WUS_Blazed_Alder': 'Blazed_Alder',
    # 'ABBA_214_WUS_McKenzie': 'McKenzie',
    # 'ABBA_214_WUS_Smith': 'Smith',
    # 'ABBA_214_WUS_Bear': 'Bear',
    # 'ABBA_214_WUS_Skokomish': 'Skokomish',
    # 'ABBA_214_WUS_Fraser': 'Fraser',
    # 'ABBA_214_WUS_Williams': 'Williams',
    # # 'ABBA_214_WUS_Randolph': 'Randolph',
    # # 'ABBA_214_WUS_Salina': 'Salina',
    # # 'ABBA_214_WUS_Lookout': 'Lookout',
    # # 'ABBA_214_WUS_Little_North_Santiam': 'Little_North_Santiam',
    # # 'ABBA_214_WUS_East_Fork_Lewis': 'East_Fork_Lewis',
    'ABBA_214_WUS_Payette': 'Payette_ID',
    'ABBA_214_WUS_Pitman': 'Pitman_CA',
    'ABBA_214_WUS_Sauk': 'Sauk_WA',
    'ABBA_214_WUS_Vallecito': 'Vallecito_CO',
    'ABBA_214_WUS_Cole': 'Cole_CA',
    'ABBA_214_WUS_Fraser': 'Fraser_CO',
    'ABBA_214_WUS_Bear': 'Bear_CA',
    'ABBA_214_WUS_American_River_Near_Nile': 'American_WA',
    'ABBA_214_WUS_Merced_HI': 'Merced_CA',
    'ABBA_214_WUS_Cedar': 'Cedar_WA',
    'ABBA_214_WUS_Andrews': 'Andrews_WA',
    'ABBA_214_WUS_Skokomish': 'Skokomish_WA',
    'ABBA_214_WUS_Blazed_Alder': 'BlazedAlder_OR',
    'ABBA_214_WUS_Williams': 'SFWilliams_CO',
    'ABBA_214_WUS_North_Brush': 'NorthBrush_WY',  # Missing: needs state
    'ABBA_214_WUS_Encampment': 'Encampment_WY',    # Missing: needs state
    'ABBA_214_WUS_Smith': 'Smith_OR',              # Missing: needs state
    'ABBA_214_WUS_Black_Gore': 'BlackGore_CO',    # Missing: needs state
    'ABBA_214_WUS_Shitike': 'Shitike_OR',
    'ABBA_214_WUS_Umatilla': 'Umatilla_OR',
    'ABBA_214_WUS_Halfmoon': 'Halfmoon_CO',        # Missing: needs state
    'ABBA_214_WUS_King_Canyon': 'Rock_WY', # 'KingCanyon_UT'
    'ABBA_214_WUS_Beaver': 'Beaver_UT',            # Missing: needs state
    'ABBA_214_WUS_Arlee': 'SFJocko_MT',
    'ABBA_234_WUS_McKenzie': 'McKenzie_OR',
    'ABBA_214_WUS_Cache': 'Cache_WY',              # Missing: needs state


    # 'ABBA_14_WUS_King_Canyon': 'King_Canyon',
    # 'ABBA_14_WUS_Halfmoon': 'Halfmoon',
    # 'ABBA_14_WUS_Smith': 'Smith',
    # 'ABBA_14_WUS_Shitike': 'Shitike',
    # 'ABBA_14_WUS_Umatilla': 'Umatilla',
    # # 'ABBA_14_WUS_Beaver': 'Beaver',
    # 'ABBA_14_WUS_South_Santiam': 'South_Santiam',
    # 'ABBA_14_WUS_Arlee': 'Arlee',
    # 'ABBA_14_WUS_Blazed_Alder': 'Blazed_Alder',
    # 'ABBA_14_WUS_McKenzie': 'McKenzie',
    # 'ABBA_14_WUS_Mission': 'Mission',
    # 'ABBA_14_WUS_Steptoe': 'Steptoe',
    # 'ABBA_14_WUS_Mill': 'Mill',
    # 'ABBA_14_WUS_Rex': 'Rex',
    # 'ABBA_14_WUS_Cache': 'Cache',
    # # 'ABBA_14_WUS_Big_Lost': 'Big_Lost',
    # 'ABBA_14_WUS_Little_Sandy': 'Little_Sandy',
    # 'ABBA_14_WUS_Lookout': 'Lookout_old',
    # 'ABBA_14_WUS_Little_North_Santiam': 'Little_North_Santiam_old',
    # 'ABBA_14_WUS_East_Fork_Lewis': 'East_Fork_Lewis',
    # 'ABBA_23_WUS_Smith': 'Smith'
}
EXPERIMENT_SETTINGS = { 
    
    'AAAA_14': 'FS',
    'ABAA_14': 'FS',
    'ADAA_14': 'FS',
    'AEAA_14': 'FS',
    'AABA_14': 'FS',
    'ABBA_14': 'FS',
    'ADBA_14': 'FS',
    'AEBA_14': 'FS',
    'ABCA_14': 'FS',
    'BBBA_14': 'FS',
    'CBBA_14': 'FS',
    'DBBA_14': 'FS',
    'EBBA_14': 'FS',
    'ABBC_14': 'FS',
    'AACA_14': 'FS',
    'ADCA_14': 'FS',
    'AECA_14': 'FS',
    'AADA_14': 'FS',
    'ADDA_14': 'FS',
    'ABDA_14': 'FS',
    'AEDA_14': 'FS',
    'AAAA_14': 'FS',
    'ABBA_14': 'FS',

    'ABBA_23': 'FS',
    'ABBA_14_WUS_Encampment': 'FS',
    'ABBA_14_WUS_King_Canyon': 'FS',
    'ABBA_14_WUS_Halfmoon': 'FS',
    'ABBA_14_WUS_Vallecito': 'FS',
    'ABBA_14_WUS_Beaver': 'FS',
    'ABBA_14_WUS_Steptoe': 'FS',
    'ABBA_14_WUS_Pitman': 'FS',
    'ABBA_14_WUS_Merced': 'FS',
    'ABBA_14_WUS_Merced_HI': 'FS',
    'ABBA_194_WUS_Merced_HI': 'FS',
    'ABBA_214_WUS_Merced_HI': 'FS',
    'ABBA_14_WUS_Cole': 'FS',
    'ABBA_14_WUS_Mill': 'FS',
    'ABBA_14_WUS_Cedar': 'FS',
    'ABBA_14_WUS_Rex': 'FS',
    'ABBA_14_WUS_Sauk': 'FS',
    'ABBA_14_WUS_Mission': 'FS',
    'ABBA_14_WUS_Arlee': 'FS',
    'ABBA_14_WUS_Andrews': 'FS',
    'ABBA_14_WUS_American_River_Near_Nile': 'FS',
    'ABBA_14_WUS_Cache': 'FS',
    'ABBA_14_WUS_Big_Lost': 'FS',
    'ABBA_14_WUS_Payette': 'FS',
    'ABBA_14_WUS_Umatilla': 'FS',
    'ABBA_14_WUS_Shitike': 'FS',
    'ABBA_14_WUS_Blazed_Alder': 'FS',
    'ABBA_14_WUS_Little_Sandy': 'FS',
    'ABBA_14_WUS_McKenzie': 'FS',
    'ABBA_14_WUS_Smith': 'FS',
    'ABBA_14_WUS_Lookout': 'FS',
    'ABBA_14_WUS_Little_North_Santiam': 'FS',
    'ABBA_14_WUS_South_Santiam': 'FS',
    'ABBA_14_WUS_East_Fork_Lewis': 'FS',
    'ABBA_14_WUS_Bear': 'FS',
    'ABBA_14_WUS_Skokomish': 'FS',
    'ABBA_14_WUS_Fraser': 'FS',
    'ABBA_14_WUS_Williams': 'FS',
    'ABBA_14_WUS_Randolph': 'FS',
    'ABBA_14_WUS_Salina': 'FS',
    'ABBA_23_WUS_Smith': 'FS',
    'ABBA_214_WUS_North_Brush': 'FS',
    'ABBA_214_WUS_Encampment': 'FS',
    'ABBA_214_WUS_King_Canyon': 'FS',
    'ABBA_214_WUS_Halfmoon': 'FS',
    'ABBA_214_WUS_Black_Gore': 'FS',
    'ABBA_214_WUS_Vallecito': 'FS',
    'ABBA_214_WUS_Beaver': 'FS',
    'ABBA_214_WUS_Pitman': 'FS',
    'ABBA_214_WUS_Cole': 'FS',
    'ABBA_214_WUS_Cedar': 'FS',
    'ABBA_214_WUS_Sauk': 'FS',
    'ABBA_214_WUS_Arlee': 'FS',
    'ABBA_214_WUS_Andrews': 'FS',
    'ABBA_214_WUS_American_River_Near_Nile': 'FS',
    'ABBA_214_WUS_Cache': 'FS',
    'ABBA_214_WUS_Payette': 'FS',
    'ABBA_214_WUS_Umatilla': 'FS',
    'ABBA_214_WUS_Shitike': 'FS',
    'ABBA_214_WUS_Blazed_Alder': 'FS',
    'ABBA_214_WUS_McKenzie': 'FS',
    'ABBA_214_WUS_Smith': 'FS',
    'ABBA_214_WUS_Lookout': 'FS',
    'ABBA_214_WUS_Little_North_Santiam': 'FS',
    'ABBA_214_WUS_East_Fork_Lewis': 'FS',
    'ABBA_214_WUS_Merced_HI': 'FS',
    'ABBA_214_WUS_Bear': 'FS',
    'ABBA_214_WUS_Skokomish': 'FS',
    'ABBA_214_WUS_Fraser': 'FS',
    'ABBA_214_WUS_Williams': 'FS',
    'ABBA_214_WUS_Randolph': 'FS',
    'ABBA_214_WUS_Salina': 'FS',
    'ABBA_234_WUS_McKenzie': 'FS',
}

EXP_WUS_REGION = {
    'ABBA_214_WUS_Payette': 'Interior',
    'ABBA_214_WUS_Pitman': 'Sierra',#SierrA NEvada
    'ABBA_214_WUS_Sauk': 'PNW',
    'ABBA_214_WUS_Vallecito': 'Interior',
    'ABBA_214_WUS_Cole': 'Sierra',
    'ABBA_214_WUS_Fraser': 'Interior',
    'ABBA_214_WUS_Bear': 'Sierra',
    'ABBA_214_WUS_American_River_Near_Nile': 'PNW',
    'ABBA_214_WUS_Merced_HI': 'Sierra',
    'ABBA_214_WUS_Cedar': 'PNW',
    'ABBA_214_WUS_Andrews': 'PNW',
    'ABBA_214_WUS_Skokomish': 'PNW',
    'ABBA_214_WUS_Blazed_Alder': 'PNW',
    'ABBA_214_WUS_Williams': 'Interior',
    'ABBA_214_WUS_North_Brush': 'Interior',  # Missing: needs state
    'ABBA_214_WUS_Encampment': 'Interior',    # Missing: needs state
    'ABBA_214_WUS_Smith': 'PNW',              # Missing: needs state
    'ABBA_214_WUS_Black_Gore': 'Interior',    # Missing: needs state
    'ABBA_214_WUS_Halfmoon': 'Interior',        # Missing: needs state
    'ABBA_214_WUS_Shitike': 'PNW',
    'ABBA_214_WUS_Umatilla': 'PNW',
    'ABBA_214_WUS_King_Canyon': 'Interior', # 'KingCanyon_UT'
    'ABBA_214_WUS_Beaver': 'Interior',            # Missing: needs state
    'ABBA_214_WUS_Arlee': 'Interior',
    'ABBA_234_WUS_McKenzie': 'PNW',
    'ABBA_214_WUS_Cache': 'Interior', 
}


# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description='Analyze posterior SWE data')
parser.add_argument('exp_id', nargs='?', help='First experiment ID')
parser.add_argument('setting', nargs='?', help='Second experiment ID')

try:
    args, unknown = parser.parse_known_args()
except SystemExit:
    args = argparse.Namespace(exp_id=None, setting=None)

# If two experiment IDs provided via command line, override EXPERIMENT_CONFIG
if args.exp_id and args.setting:
    EXPERIMENT_CONFIG = {
        args.exp_id: args.exp_id,
    }
    EXPERIMENT_SETTINGS = {
        args.exp_id: args.setting,
    }


# Extract experiment IDs and names
EXPS = list(EXPERIMENT_CONFIG.keys())
EXPERIMENT_NAMES = list(EXPERIMENT_CONFIG.values())

SHARED_FOLDER_NAME = '_'.join(EXPS[:4])
SHARED_PLOTS_DIR = join(ROOTDIR, 'Analysis', SHARED_FOLDER_NAME)
Path(SHARED_PLOTS_DIR).mkdir(parents=True, exist_ok=True)

# Create color palette for experiments (will be assigned in order)
EXPERIMENT_COLORS = sns.color_palette('tab10', n_colors=len(EXPERIMENT_NAMES))
EXPERIMENT_COLOR_MAP = {name: color for name, color in zip(EXPERIMENT_NAMES, EXPERIMENT_COLORS)}


meltseason_test_name = 'Disch_NSEmeltseason'
plotting = False
LOA_objects = {}
for EXP_ID in EXPS:
    print(EXP_ID)
    ORIG_ID = EXP_ID

    cfg_file = join(config_dir, f"{EXP_ID}_config.json")
    with open(cfg_file, 'r') as f:
        config = json.load(f)
    # config['BASIN'] = 'Dischma'
    for key,value in config.items():
        #change the paths to the correct paths
        if type(value)==str and '/work/FAC' in value:
            config[key] = value.replace('/work/FAC/FGSE/IDYST/gmariet1/gaia/pwiersma/ewatercycle',
            '/home/pwiersma/scratch/Data/ewatercycle')
    
    if config['BASIN'] in EXP_ID:
        config['HYDROMT_POSTFIX'] = 'sep2025'
        config['FORCING'] = 'ERA5Land'
    else:
        config['HYDROMT_POSTFIX'] = 'feb2024'
        config['FORCING'] = 'MeteoSwiss'

    self = posterior_analysis(config)

    #ask user for input on runid_limit
    # self.runid_limit = int(input(f"Enter runid_limit for {EXP_ID}: "))
    self.runid_limit = 1000
    print(f"Runid_limit = {self.runid_limit}")

    self.ORIG_ID = ORIG_ID
    # Ensure EXP_ID case matches ORIG_ID case
    # if self.ORIG_ID.isupper():
    #     self.EXP_ID = EXP_ID.upper()
    #     # Make OUTDIR folder name match case of ORIG_ID
    #     self.OUTDIR = self.OUTDIR[:-len(self.EXP_ID)] + self.EXP_ID
    # else:
    #     # Make first letter lowercase
    #     self.EXP_ID = EXP_ID[0].lower() + EXP_ID[1:]
    #     self.OUTDIR = self.OUTDIR[:-len(self.EXP_ID)] + self.EXP_ID   

    self.ANA_DIR = join(self.OUTDIR, 'Analysis')
    Path.mkdir(Path(self.ANA_DIR), exist_ok=True)
    self.FIGDIR = join(self.ANA_DIR, 'Figures')
    Path(self.FIGDIR).mkdir(parents=True, exist_ok=True)

    self.load_prior_data(load_SWE=False) #this takes quite long
    self.load_metrics()
    self.load_Q_obs()
    self.load_SWE_obs()
    self.load_meteo()
    self.calc_elev_bands()

    # Use flexible experiment configuration
    self.EXP_ID_translation = EXPERIMENT_CONFIG
    self.experiment_name = EXPERIMENT_CONFIG[EXP_ID]
    self.experiment_color = EXPERIMENT_COLOR_MAP[self.experiment_name]
    self.experiment_setting = EXPERIMENT_SETTINGS[EXP_ID]

    self.q_metrics_to_use =  [
        # 'KGE_meltseason',
    # 'logKGE_meltseason',# 'KGE_meltseason_r','KGE_meltseason_alpha','KGE_meltseason_beta',
                    'NSE_meltseason',
                    'KGE_meltseason2',
                    # 'logNSE_meltseason',
                    # 'NSE_PeakFilter',
                    # 'KGE_PeakFilter',
                    # 'KGE_BaseFlow',
                    'NSE_weekly',
                    'NSE',
                    # 'KGE',
                    # 'R2',
                    'PFE',
                    # 't_Qend_ME',
                    't_melt_onset_ME',
                    'melt_onset_sharpness_ME',
                    # 'Viney_bias',
                    'NSE_bias',
                    
                    # 'NSE_BaseFlow',
                    # 'NSE_PeakFilter',
                    #   't_hfd_ME',
                        'Qmax_ME',
                        # 'Qmean_ME',
                    # 't_Qinflection_ME','Qmean_meltseason_ME','Qcv_meltseason_ME',
                    #  'BaseFlow_sum_ME',#'Qamp_ME',
                    # 'bfi_ME',
                    't_hfd_meltseason_ME',
                    't_Qinflection_ME',
                    # 't_Qmax_ME',
                    't_Qstart_ME',
                    'Qmean_APE', 
                    # 'Qcv_APE', 
                    'Q95_APE', 
                    'Q5_APE',
                        'Qamp_APE', 
                        'Qmean_meltseason_APE', 
                        'Qcv_meltseason_APE',
                        'Qcv_meltseason2_APE',
                        'Qmean_meltseason2_APE',
                        't_hfd_meltseason2_ME',
                        'Qmean_meltseason_beta_APE',
                        'Qcv_meltseason_beta_APE',
                        't_hfd_meltseason_beta_ME',
                        'NSE_meltseason_beta',
                        'NSE_meltseason2'
                    ]#,'t_Qrise_ME']'NSE','R2','PBIAS','KGE','KGE_PeakFilter','KGE_BaseFlow','Qcv_ME',
    self.swe_metrics_to_use = [
                                'SWE_7daymelt_ME',
                                # 't_SWE_start_ME',
                                # 't_SWE_end_ME',
                                # 'melt_sum_elev_ME',
                                # 'SWE_SWS_elev_ME',
                                # 't_SWE_max_elev_ME',
                                # 'SWE_7daymelt_elev_ME',
                                # 't_SWE_start_elev_ME',
                                'full_SC_end_ME',
                                'SC_depletion_duration_ME',
                                # 't_SWE_start_ME',
                                # 't_SWE_start_grid_ME',
                                't_SWE_end_ME',
                                # 't_SWE_end_elev_ME',
                                # 't_SWE_end_grid_ME',
                                # 't_SWE_max_elev_ME',
                                't_SWE_max_ME',
                                # 't_SWE_max_grid_ME',
                                'SWE_melt_NSE',
                                # 'SWE_melt_NSE_elev',
                                'SWE_melt_NSE_grid',
                                'SWE_melt_NSE_weekly',
                                # 'SWE_melt_NSE_monthly',
                                # 'SWE_melt_NSE_3days',
                                'SWE_melt_NSE_weekly_grid',
                                # 'SWE_snowfall_NSE',
                                # 'SWE_snowfall_NSE_elev',
                                # 'SWE_snowfall_NSE_grid',
                                # 'SWE_SPAEF',
                                # 'SWE_melt_KGE_elev',
                                # 'SWE_melt_KGE',
                                # 'SWE_NSE_elev',
                                'SWE_NSE',
                                # 'SWE_NSE_grid',
                                'melt_sum_grid_MAPE',
                                # 'melt_sum_elev_MAPE',
                                'melt_sum_APE',
                                'SWE_max_APE',
                                # 'SWE_max_elev_MAPE',
                                # 'SWE_max_grid_MAPE',
                                'SWE_SWS_APE',
                                # 'SWE_SWS_elev_MAPE',
                                # 'SWE_SWS_grid_MAPE',
                                't_melt_hfd_ME'
                                ]

    # Create a copy of the list to iterate over while modifying the original
    swe_metrics_to_check = self.swe_metrics_to_use.copy()
    
    for met in swe_metrics_to_check:
        # print(met)
        if not met in self.metrics[self.START_YEAR].columns:
            print(f"{met} not in metrics[self.START_YEAR].columns")
            self.swe_metrics_to_use.remove(met)
    q_metrics_to_check = self.q_metrics_to_use.copy()
    for met in q_metrics_to_check:
        if not met in self.metrics[self.START_YEAR].columns:
            print(f"{met} not in metrics[self.START_YEAR].columns")
            self.q_metrics_to_use.remove(met)

    SWE_target_metric = 'melt_sum_grid_MAPE'
    
    # # Calculate meltseason_beta periods and metrics
    # # Beta meltseason: from t_SWE_max to t_SWE_end + 1 month
    # print("Calculating meltseason_beta periods and metrics...")
    # Q_obs_mmd = self.Qobs * 86400 * 1000 / (self.E.dem_area * 1e6)  # convert to mm/day
    # Q_obs_mmd = Q_obs_mmd.squeeze()
    
    # for year in range(self.START_YEAR, self.END_YEAR + 1):
    #     # Get t_SWE_max and t_SWE_end (in DOY since Oct 1 of year-1)
    #     t_SWE_max = self.swe_signatures[year]['obs']['t_SWE_max']
    #     t_SWE_end = self.swe_signatures[year]['obs']['t_SWE_end']
        
    #     # Convert DOY to dates
    #     year_start = pd.Timestamp(f'{year-1}-10-01')
    #     meltseason_beta_start = year_start + pd.Timedelta(days=int(t_SWE_max))
    #     meltseason_beta_end = year_start + pd.Timedelta(days=int(t_SWE_end)) + pd.DateOffset(months=1)
        
    #     # Ensure end date doesn't exceed year end
    #     year_end = pd.Timestamp(f'{year}-09-30')
    #     if meltseason_beta_end > year_end:
    #         meltseason_beta_end = year_end
        
    #     # Create date range for meltseason_beta
    #     meltseason_beta_period = pd.date_range(start=meltseason_beta_start, end=meltseason_beta_end, freq='D')
        
    #     # Get Q data for the year
    #     mask = pd.date_range(start=f'{year-1}-10-01', end=f'{year}-09-30', freq='D')
    #     Q_obs_year = Q_obs_mmd.loc[mask]
        
    #     # Initialize columns in metrics dataframe if they don't exist
    #     if 'Qmean_meltseason_beta_APE' not in self.metrics[year].columns:
    #         self.metrics[year]['Qmean_meltseason_beta_APE'] = np.nan
    #     if 'Qcv_meltseason_beta_APE' not in self.metrics[year].columns:
    #         self.metrics[year]['Qcv_meltseason_beta_APE'] = np.nan
    #     if 't_hfd_meltseason_beta_ME' not in self.metrics[year].columns:
    #         self.metrics[year]['t_hfd_meltseason_beta_ME'] = np.nan
    #     if 'NSE_meltseason_beta' not in self.metrics[year].columns:
    #         self.metrics[year]['NSE_meltseason_beta'] = np.nan
        
    #     # Calculate metrics for each simulation
    #     for run_id in self.metrics[year].index:
    #         if run_id == 'obs':
    #             continue
            
    #         try:
    #             # Get simulated Q for this run_id
    #             Q_sim_year_raw = self.Q.loc[mask, run_id]
    #             Q_sim_year = Q_sim_year_raw * 86400 * 1000 / (self.E.dem_area * 1e6)  # convert to mm/day
                
    #             # Calculate Qmean_meltseason_beta
    #             Qmean_meltseason_beta_obs = calc_snowmeltseason_sum(Q_obs_year, melt_period=meltseason_beta_period) / len(meltseason_beta_period)
    #             Qmean_meltseason_beta_sim = calc_snowmeltseason_sum(Q_sim_year, melt_period=meltseason_beta_period) / len(meltseason_beta_period)
                
    #             # Calculate Qmean_meltseason_beta_APE
    #             if Qmean_meltseason_beta_obs != 0:
    #                 Qmean_meltseason_beta_APE = np.abs((Qmean_meltseason_beta_obs - Qmean_meltseason_beta_sim) / Qmean_meltseason_beta_obs) * 100
    #             else:
    #                 Qmean_meltseason_beta_APE = np.nan
                
    #             # Calculate Qcv_meltseason_beta
    #             Qcv_meltseason_beta_obs = calc_snowmeltseason_cv(Q_obs_year, melt_period=meltseason_beta_period)
    #             Qcv_meltseason_beta_sim = calc_snowmeltseason_cv(Q_sim_year, melt_period=meltseason_beta_period)
                
    #             # Calculate Qcv_meltseason_beta_APE
    #             if Qcv_meltseason_beta_obs != 0:
    #                 Qcv_meltseason_beta_APE = np.abs((Qcv_meltseason_beta_obs - Qcv_meltseason_beta_sim) / Qcv_meltseason_beta_obs) * 100
    #             else:
    #                 Qcv_meltseason_beta_APE = np.nan
                
    #             # Calculate t_hfd_meltseason_beta
    #             t_hfd_meltseason_beta_obs = calc_hfd_meltseason(Q_obs_year, melt_period=meltseason_beta_period)
    #             t_hfd_meltseason_beta_sim = calc_hfd_meltseason(Q_sim_year, melt_period=meltseason_beta_period)
                
    #             # Calculate t_hfd_meltseason_beta_ME
    #             if not np.isnan(t_hfd_meltseason_beta_obs) and not np.isnan(t_hfd_meltseason_beta_sim):
    #                 t_hfd_meltseason_beta_ME = np.abs(t_hfd_meltseason_beta_obs - t_hfd_meltseason_beta_sim)
    #             else:
    #                 t_hfd_meltseason_beta_ME = np.nan
                
    #             # Calculate NSE_meltseason_beta
    #             Q_obs_meltseason_beta = Q_obs_year.loc[meltseason_beta_period]
    #             Q_sim_meltseason_beta = Q_sim_year.loc[meltseason_beta_period]
    #             if len(Q_obs_meltseason_beta) > 0 and len(Q_sim_meltseason_beta) > 0:
    #                 NSE_meltseason_beta = he.nse(Q_sim_meltseason_beta.values, Q_obs_meltseason_beta.values)
    #             else:
    #                 NSE_meltseason_beta = np.nan
                
    #             # Store in metrics dataframe
    #             self.metrics[year].loc[run_id, 'Qmean_meltseason_beta_APE'] = Qmean_meltseason_beta_APE
    #             self.metrics[year].loc[run_id, 'Qcv_meltseason_beta_APE'] = Qcv_meltseason_beta_APE
    #             self.metrics[year].loc[run_id, 't_hfd_meltseason_beta_ME'] = t_hfd_meltseason_beta_ME
    #             self.metrics[year].loc[run_id, 'NSE_meltseason_beta'] = NSE_meltseason_beta
    #         except Exception as e:
    #             print(f"Error calculating beta metrics for {run_id} in year {year}: {e}")
    #             continue
    
    self.swe_metrics = {year: metrics[self.swe_metrics_to_use] for year, metrics in self.metrics.items()}
    self.Q_metrics = {year: metrics[self.q_metrics_to_use] for year, metrics in self.metrics.items()}

    # compute correlations within Q and SWE metrics and drop the ones we don't need 
    self.compute_QSWE_corr()
    self.QSWE_corr['level_0'] = [translate_SWE_metric_name(ent,keep_suffix=True) for ent in self.QSWE_corr['level_0']]
    self.QSWE_corr['level_1'] = [translate_Q_metric_name(ent) for ent in self.QSWE_corr['level_1']]

    if plotting:
    # for self in LOA_objects.values():
        # palette = sns.color_palette('colorblind',n_colors=len(self.swe_metrics_to_use))
        # self.swe_palette = {metric: palette[i] for i, metric in enumerate(self.swe_metrics_to_use)}
        all_swe_metrics = self.QSWE_corr['level_0'].unique()
        palette = sns.color_palette('colorblind',n_colors=len(all_swe_metrics))
        self.swe_palette = {metric: palette[i] for i, metric in enumerate(all_swe_metrics)}
        
        all_q_metrics = self.QSWE_corr['level_1'].unique()
        palette = sns.color_palette('colorblind',n_colors=len(all_q_metrics))
        self.Q_palette = {metric: palette[i] for i, metric in enumerate(all_q_metrics)}
        
        # self.compute_metric_corr(self.Q_metrics,self.q_metrics_to_use)
        # self.compute_metric_corr(self.swe_metrics,self.swe_metrics_to_use)
        # self.compute_Qsig_corr()
        # self.compute_swesig_elev_corr()

        #here you can plot either just a selection or all metrics 
        # self.plot_QSWE_signatures(swe_metrics_to_use=self.swe_metrics_to_use, 
        #     q_metrics_to_use=self.q_metrics_to_use)
        # self.plot_QSWE_signatures()

        # self.plot_QSWE_signatures_stripplot()
        # self.plot_best_QSWE_corrs()

    # choose a target SWE metric and check which combos of Q metrics work well 
    # q_selection = ['Qamp_ME','Qmean_meltseason_ME',
    #                'Qcv_meltseason_ME','bfi_ME','Qcv_ME',
    #                'Qmean_ME','KGE_meltseason','KGE_meltseason_r']
    q_selection = self.q_metrics_to_use
    swe_selection = self.swe_metrics_to_use
    self.make_Qsig_obs()
    self.calc_corr_single()
    # self.calc_corr_combos(Nmax=1)

    # self.dotty_plot('NSE','SWE_melt_NSE')
                
    LOA_objects[ORIG_ID] = self

#%% Load Safeeq data and do analysis 
safeeq_basins = pd.read_csv(join(ROOTDIR,'aux_data','Safeeq2013_catchments_USGS_metric_plusnames_shortnames.csv'),
index_col = 0,).rename(columns = {'k':'k_class'})
safeeq_k = pd.read_excel(join(ROOTDIR,'aux_data',
'Data_for_Matt_Safeeq2012_k_data.xlsx'),sheet_name = 'HP_paper').rename(columns = {'Basin':'ID'})
safeeq = pd.merge(safeeq_basins, safeeq_k, on = 'ID')

#%% Make overview plots and map 
# Define zeta (catchment characteristic) file paths and calculation functions
# To add a new zeta: add entry to zeta_configs dict with 'filepath' and 'calc_func' keys

# Get START_YEAR and END_YEAR from first LOA object
if LOA_objects:
    first_obj = next(iter(LOA_objects.values()))
    START_YEAR = first_obj.START_YEAR
    END_YEAR = first_obj.END_YEAR
else:
    raise ValueError("LOA_objects is empty, cannot determine START_YEAR and END_YEAR")

def calc_rainfall_mixing_ratio(rainfall_overlap_year, snowmelt_year):
    """Calculate rainfall mixing ratio"""
    return rainfall_overlap_year.sum() / snowmelt_year.sum() if snowmelt_year.sum() > 0 else np.nan

def calc_spring_rf_fraction(melt_period, rainfall_year, rfmelt_year):
    """Calculate spring snowfall fraction"""
    return rainfall_year[melt_period].sum() / rfmelt_year[melt_period].sum() if rfmelt_year[melt_period].sum() > 0 else np.nan

def calc_spring_et_fraction(melt_period, et_year, rfmelt_year):
    """Calculate spring ET fraction: total ET over meltseason / total rfmelt over meltseason"""
    return et_year[melt_period].sum() / rfmelt_year[melt_period].sum() if rfmelt_year[melt_period].sum() > 0 else np.nan

def calc_et_mixing_ratio(et_overlap_year, snowmelt_year):
    """Calculate ET mixing ratio"""
    return et_overlap_year.sum() / snowmelt_year.sum() if snowmelt_year.sum() > 0 else np.nan

def calc_sf_ratio(snowmelt_year, rfmelt_year):
    """Calculate snowfall fraction"""
    return snowmelt_year.sum() / rfmelt_year.sum() if rfmelt_year.sum() > 0 else np.nan

def calc_delay(rfmelt_year, Q_year, et_year):
    """Calculate delay as sum of cumulative deficit between rfmelt and Q+ET"""
    rfmelt_norm = rfmelt_year / rfmelt_year.sum() if rfmelt_year.sum() > 0 else rfmelt_year
    Q_ET_year = Q_year + et_year
    Q_ET_norm = Q_ET_year / Q_ET_year.sum() if Q_ET_year.sum() > 0 else Q_ET_year
    deficit = (rfmelt_norm - Q_ET_norm).cumsum()
    return np.abs(deficit).sum() / rfmelt_norm.sum() if rfmelt_norm.sum() > 0 else np.nan

def calc_S_fraction(total_storage, Q_year):
    """Calculate storage fraction as max storage / Q sum."""
    max_storage = total_storage.max()
    return max_storage / Q_year.sum() if Q_year.sum() > 0 else np.nan

def calc_S_fraction_spring(total_storage, melt_period, Q_year):
    """Calculate storage fraction as max storage / Q sum over melt period"""
    max_storage = total_storage[melt_period].max()
    return max_storage / Q_year[melt_period].sum() if Q_year[melt_period].sum() > 0 else np.nan

def calc_S_dynamic_fraction(total_storage, Q_year):
    """Calculate dynamic storage fraction as (max-min) storage / Q sum."""
    dynamic_storage = total_storage.max() - total_storage.min()
    return dynamic_storage / Q_year.sum() if Q_year.sum() > 0 else np.nan

def calc_S_dynamic_fraction_spring(total_storage, melt_period, Q_year):
    """Calculate spring dynamic storage fraction as (max-min) storage / Q sum."""
    dynamic_storage = total_storage[melt_period].max() - total_storage[melt_period].min()
    return dynamic_storage / Q_year[melt_period].sum() if Q_year[melt_period].sum() > 0 else np.nan

def calc_temp_storage(total_storage, snowmelt_year, rainfall_year):
    """Calculate temporary storage variation over whole year"""
    total_water = snowmelt_year.sum() + rainfall_year.sum()
    if total_water == 0:
        return np.nan
    norm_storage = total_storage / total_water
    mean_norm_storage = norm_storage.mean()
    return (np.abs(norm_storage - mean_norm_storage)).sum()

def calc_temp_storage_spring(total_storage, melt_period, snowmelt, rainfall):
    """Calculate temporary storage variation over melt period"""
    total_water_spring = snowmelt[melt_period].sum() + rainfall[melt_period].sum()
    if total_water_spring == 0:
        return np.nan
    norm_storage_spring = total_storage[melt_period] / total_water_spring
    mean_norm_storage_spring = norm_storage_spring.mean()
    return (np.abs(norm_storage_spring - mean_norm_storage_spring)).sum()

def calc_temp_storage_spring_2(total_storage, melt_period, snowmelt, rainfall):
    """Calculate temporary storage variation over melt period 
    but instead of mean_norm_storage, use start_norm_storage"""
    total_water_spring = snowmelt[melt_period].sum() + rainfall[melt_period].sum()
    if total_water_spring == 0:
        return np.nan
    norm_storage_spring = total_storage[melt_period] / total_water_spring
    start_norm_storage_spring = norm_storage_spring.iloc[0]
    return (np.abs(norm_storage_spring - start_norm_storage_spring)).sum()
def calc_antecedent_Q(Q_year, melt_period):
    """Calculate antecedent Q as Q_start/Qmean_meltseason"""
    Q_start = Q_year[melt_period].iloc[0]
    Q_mean_meltseason = Q_year[melt_period].mean()
    return Q_start / Q_mean_meltseason
def calc_DJF_rainfall_fraction(rainfall_year):
    """Calculate DJF rainfall fraction"""
    return rainfall_year[rainfall_year.index.month.isin([12,1,2])].sum() / rainfall_year.sum()
def calc_MAM_rainfall_fraction(rainfall_year):
    """Calculate MAM rainfall fraction"""
    return rainfall_year[rainfall_year.index.month.isin([3,4,5])].sum() / rainfall_year.sum()
def calc_DJF_temp(temp_year):
    """Calculate DJF temperature fraction"""
    return temp_year[temp_year.index.month.isin([12,1,2])].mean()
def calc_MAM_temp(temp_year):
    """Calculate MAM temperature fraction"""
    return temp_year[temp_year.index.month.isin([3,4,5])].mean()
def calc_DJF_pet(pet_year):
    """Calculate DJF pet fraction"""
    return pet_year[pet_year.index.month.isin([12,1,2])].mean()
def calc_MAM_pet(pet_year):
    """Calculate MAM pet fraction"""
    return pet_year[pet_year.index.month.isin([3,4,5])].mean()
def calc_spring_pet(melt_period,pet_year):
    """Calculate meltseason pet fraction"""
    return pet_year[melt_period].mean()

def calc_annual_q_sum(Q_year):
    """Calculate annual discharge sum over the hydrological year."""
    return Q_year.sum()

def calc_mean_dem_slope(dem):
    """Calculate mean catchment slope (degrees) from DEM."""
    try:
        z = np.asarray(dem.values, dtype=float)
        valid = np.isfinite(z)
        if valid.sum() < 4:
            return np.nan

        lat_vals = np.asarray(dem["lat"].values, dtype=float)
        lon_vals = np.asarray(dem["lon"].values, dtype=float)
        if len(lat_vals) < 2 or len(lon_vals) < 2:
            return np.nan

        dy_deg = np.abs(np.nanmean(np.diff(lat_vals)))
        dx_deg = np.abs(np.nanmean(np.diff(lon_vals)))
        if dy_deg == 0 or dx_deg == 0:
            return np.nan

        lat_mean = np.nanmean(lat_vals)
        dy_m = dy_deg * 111320.0
        dx_m = dx_deg * 111320.0 * np.cos(np.deg2rad(lat_mean))
        if dx_m == 0 or np.isnan(dx_m) or np.isnan(dy_m):
            return np.nan

        z_filled = np.where(valid, z, np.nanmean(z))
        grad_y, grad_x = np.gradient(z_filled, dy_m, dx_m)
        slope_rad = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
        return np.nanmean(np.degrees(slope_rad)[valid])
    except Exception:
        return np.nan

def calc_proxy_slope_from_relief(dem, dem_area_km2):
    """Estimate slope as (zmax - zmin) / sqrt(A), where A is basin area in m^2."""
    if dem_area_km2 is None or dem_area_km2 <= 0:
        return np.nan
    z = np.asarray(dem.values, dtype=float)
    zmax = np.nanmax(z)
    zmin = np.nanmin(z)
    if np.isnan(zmax) or np.isnan(zmin):
        return np.nan
    leff_m = np.sqrt(dem_area_km2 * 1e6)
    if leff_m <= 0:
        return np.nan
    return (zmax - zmin) / leff_m

def calc_mean_catchment_elevation(dem):
    """Calculate mean catchment elevation from DEM."""
    return np.nanmean(np.asarray(dem.values, dtype=float))

# Define zeta configurations: each entry needs 'filepath' and 'calc_func'
# To add a new zeta variable:
#   1. Create a calculation function (e.g., calc_my_new_zeta) that takes necessary inputs
#   2. Add an entry to zeta_configs with:
#      - 'filepath': path to CSV file where results will be saved
#      - 'calc_func': the calculation function
#   3. In the main loop, call the function and store result: 
#      zetas['my_new_zeta'].loc[year, self.experiment_name] = calc_my_new_zeta(...)
#   4. The zeta will be automatically loaded/saved and accessible via zetas['my_new_zeta']
zeta_configs = {
    'rainfall_mixing_ratios': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'rainfall_mixing_ALL.csv'),
        'calc_func': calc_rainfall_mixing_ratio
    },
    'spring_rf_fraction': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'spring_snowfall_fraction_ALL.csv'),
        'calc_func': calc_spring_rf_fraction
    },
    'et_mixing_ratios': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'et_mixing_ALL.csv'),
        'calc_func': calc_et_mixing_ratio
    },
    'sf_ratio': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'snowfall_fraction_ALL.csv'),
        'calc_func': calc_sf_ratio
    },
    'delays': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'delays_ALL.csv'),
        'calc_func': calc_delay
    },
    'S_fraction': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'Storage_fraction_ALL.csv'),
        'calc_func': calc_S_fraction
    },
    'S_fraction_spring': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'Storage_fraction_spring_ALL.csv'),
        'calc_func': calc_S_fraction_spring
    },
    'S_dynamic_fraction': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'Storage_dynamic_fraction_ALL.csv'),
        'calc_func': calc_S_dynamic_fraction
    },
    'S_dynamic_fraction_spring': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'Storage_dynamic_fraction_spring_ALL.csv'),
        'calc_func': calc_S_dynamic_fraction_spring
    },
    'temp_storage': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'Temp_storage_ALL.csv'),
        'calc_func': calc_temp_storage
    },
    'temp_storage_spring': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'Temp_storage_spring_ALL.csv'),
        'calc_func': calc_temp_storage_spring
    },
    'temp_storage_spring_2': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'Temp_storage_spring_2_ALL.csv'),
        'calc_func': calc_temp_storage_spring_2
    },
    'temp_storage_spring_beta': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'Temp_storage_spring_beta_ALL.csv'),
        'calc_func': calc_temp_storage_spring
    },
    'temp_storage_spring_2_beta': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'Temp_storage_spring_2_beta_ALL.csv'),
        'calc_func': calc_temp_storage_spring_2
    },
    'spring_rf_fraction_beta': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'spring_snowfall_fraction_beta_ALL.csv'),
        'calc_func': calc_spring_rf_fraction
    },
    'spring_et_fraction': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'spring_et_fraction_ALL.csv'),
        'calc_func': calc_spring_et_fraction
    },
    'spring_et_fraction_beta': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'spring_et_fraction_beta_ALL.csv'),
        'calc_func': calc_spring_et_fraction
    },
    'melt_period_start':{
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'melt_period_start_ALL.csv'),

    },
    'melt_period_end':{
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'melt_period_end_ALL.csv'),

    },
    'DJF_rainfall_fraction': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'DJF_rainfall_fraction_ALL.csv'),
        'calc_func': calc_DJF_rainfall_fraction
    },
    'MAM_rainfall_fraction': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'MAM_rainfall_fraction_ALL.csv'),
        'calc_func': calc_MAM_rainfall_fraction
    },
    # 'DJF_temp': {
    #     'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'DJF_temp_ALL.csv'),
    #     'calc_func': calc_DJF_temp
    # },
    # 'MAM_temp': {
    #     'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'MAM_temp_ALL.csv'),
    #     'calc_func': calc_MAM_temp
    # },
    'DJF_pet': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'DJF_pet_ALL.csv'),
        'calc_func': calc_DJF_pet
    },
    'MAM_pet': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'MAM_pet_ALL.csv'),
        'calc_func': calc_MAM_pet
    },
    'spring_pet': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'spring_pet_ALL.csv'),
        'calc_func': calc_spring_pet
    },
    'annual_q_sum': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'annual_q_sum_ALL.csv'),
        'calc_func': calc_annual_q_sum
    },
    'mean_slope_dem': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'mean_slope_dem_ALL.csv'),
        'calc_func': calc_mean_dem_slope
    },
    'slope_proxy_relief': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'slope_proxy_relief_ALL.csv'),
        'calc_func': calc_proxy_slope_from_relief
    },
    'mean_elevation_dem': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'mean_elevation_dem_ALL.csv'),
        'calc_func': calc_mean_catchment_elevation
    },
    'meltseason_length': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'meltseason_length_ALL.csv')
    },
    'melt_hfi': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'melt_hfi_ALL.csv')
    },
    'antecedent_Q': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'antecedent_Q_ALL.csv'),
            'calc_func': calc_antecedent_Q
    },
    'ERRA_center_of_mass': {
        'filepath': join(ROOTDIR, 'aux_data', 'catchment_characteristics', 'ERRA_center_of_mass_ALL.csv'),
        
    },
}

# Load or initialize dataframes for zetas
zetas = {}
for name, config in zeta_configs.items():
    filepath = config['filepath']
    if os.path.exists(filepath):
        zetas[name] = pd.read_csv(filepath, index_col=0)
    else:
        zetas[name] = pd.DataFrame(index=range(START_YEAR, END_YEAR + 1))

ERRA_center_of_mass = pd.read_csv(join(ROOTDIR, 'aux_data', 'ERRA_21', 'centers_of_mass.csv'), index_col=0)
plotting = True
# basin_list = [val.BASIN for val in LOA_objects.values()]
basin_list = [val.experiment_name for val in LOA_objects.values()]
#if there's a number in the basin_list entry, change the entry to 'Dischma'
for i in range(len(basin_list)):
    if any(char.isdigit() for char in basin_list[i]):
        basin_list[i] = 'Dischma'
for self in LOA_objects.values():
    Q = self.Qobs * 86400 * 1000 / (self.E.dem_area * 1e6)  # convert to mm/day
    Q = Q.squeeze()
    mean_slope_dem = calc_mean_dem_slope(self.E.dem)
    slope_proxy_relief = calc_proxy_slope_from_relief(self.E.dem, self.E.dem_area)
    mean_elevation_dem = calc_mean_catchment_elevation(self.E.dem)
    S = SwissStation(self.BASIN)
    S.read_station(startyear=self.START_YEAR, endyear=self.END_YEAR,
                   discharge_dir=join(ROOTDIR, "Discharge_data"))
    actual_Qobs = S.obs * 86400 * 1000 / (self.E.dem_area * 1e6)  # convert to mm/day

    SWE = self.SWEobs.mean(dim=['lat', 'lon']).to_pandas()
    snowmelt = -SWE.diff()  # Negative change in SWE = snowmelt
    snowmelt = snowmelt.where(snowmelt > 0, 0)

    scalars_file = glob.glob(join(self.SYNDIR, '*Synthetic_obs*.csv'))[0]
    scalars = pd.read_csv(scalars_file, index_col=0, parse_dates=True).loc[Q.index]
    rfmelt = scalars['rainfallplusmelt']
    rainfall = rfmelt - snowmelt
    if (self.experiment_name[1] =='0') & (self.experiment_name[2]!= '.'):
        rainfall *= 0
    rainfall_overlap = rainfall.combine(snowmelt, min)
    et = scalars['et']
    et_overlap = et.combine(snowmelt, min)
    # temperature = scalars['pet']
    pet = scalars['pet']

    plot_dir = join(SHARED_PLOTS_DIR, f"{self.experiment_name}_overview_plots")
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    ERRA_prep = pd.DataFrame(columns = ['Q','input','mask'],
        index = pd.date_range(start=f'{self.START_YEAR-1}-10-01', end=f'{self.END_YEAR}-09-30', freq='D'))

    for year in range(self.START_YEAR, self.END_YEAR + 1):
        mask = pd.date_range(start=f'{year-1}-10-01', end=f'{year}-09-30', freq='D')
        rainfall_overlap_year = rainfall_overlap[mask]
        snowmelt_year = snowmelt[mask]
        et_overlap_year = et_overlap[mask]
        rfmelt_year = rfmelt[mask]
        Q_year = Q[mask]
        et_year = et[mask]
        rainfall_year = rainfall[mask]
        total_storage = scalars['satwaterdepth'].loc[mask] + scalars['ustoredepth'].loc[mask]
        pet_year = pet[mask]
        # Calculate all zetas using the configured functions
        zetas['rainfall_mixing_ratios'].loc[year, self.experiment_name] = \
            calc_rainfall_mixing_ratio(rainfall_overlap_year, snowmelt_year)
        zetas['et_mixing_ratios'].loc[year, self.experiment_name] = \
            calc_et_mixing_ratio(et_overlap_year, snowmelt_year)
        zetas['sf_ratio'].loc[year, self.experiment_name] = \
            calc_sf_ratio(snowmelt_year, rfmelt_year)
        zetas['delays'].loc[year, self.experiment_name] = \
            calc_delay(rfmelt_year, Q_year, et_year)
        zetas['S_fraction'].loc[year, self.experiment_name] = \
            calc_S_fraction(total_storage, Q_year)
        zetas['S_dynamic_fraction'].loc[year, self.experiment_name] = \
            calc_S_dynamic_fraction(total_storage, Q_year)
        zetas['temp_storage'].loc[year, self.experiment_name] = \
            calc_temp_storage(total_storage, snowmelt_year, rainfall_year)
        zetas['annual_q_sum'].loc[year, self.experiment_name] = \
            calc_annual_q_sum(Q_year)
        zetas['mean_slope_dem'].loc[year, self.experiment_name] = mean_slope_dem
        zetas['slope_proxy_relief'].loc[year, self.experiment_name] = slope_proxy_relief
        zetas['mean_elevation_dem'].loc[year, self.experiment_name] = mean_elevation_dem
        try: 
            zetas['ERRA_center_of_mass'].loc[year, self.experiment_name] = \
            ERRA_center_of_mass.loc[self.experiment_name, "center_of_mass"]
        except:
            zetas['ERRA_center_of_mass'].loc[year, self.experiment_name] = np.nan
        # Calculate melt period for temp_storage_spring
        snowmelt_reduced = snowmelt[mask][90:-30]  # Ignore first 90 and last 30 days to avoid initial/fall melt
        if len(snowmelt_reduced) > 0 and snowmelt_reduced.sum() > 0:
            q10_timing = snowmelt_reduced[snowmelt_reduced.cumsum() >= 0.1 * snowmelt_reduced.sum()].index[0]
            q90_timing = snowmelt_reduced[snowmelt_reduced.cumsum() >= 0.9 * snowmelt_reduced.sum()].index[0]
            q25_timing = snowmelt_reduced[snowmelt_reduced.cumsum() >= 0.25 * snowmelt_reduced.sum()].index[0]
            q75_timing = snowmelt_reduced[snowmelt_reduced.cumsum() >= 0.75 * snowmelt_reduced.sum()].index[0]
            start_of_melt = q10_timing - pd.Timedelta(days=7)  # Add 7 days buffer before Q10
            end_of_melt = q90_timing + pd.Timedelta(days=30)  # Add 30 days buffer after Q90
            if end_of_melt > mask[-1]:
                end_of_melt = mask[-1]
            melt_period = pd.date_range(start=start_of_melt, end=end_of_melt, freq='D')
            zetas['melt_period_start'].loc[year, self.experiment_name] = start_of_melt
            zetas['melt_period_end'].loc[year, self.experiment_name] = end_of_melt
            zetas['meltseason_length'].loc[year, self.experiment_name] = (end_of_melt - start_of_melt).days
            zetas['melt_hfi'].loc[year, self.experiment_name] = (q75_timing - q25_timing).days
            zetas['temp_storage_spring'].loc[year, self.experiment_name] = \
                calc_temp_storage_spring(total_storage, melt_period, snowmelt, rainfall)
            zetas['temp_storage_spring_2'].loc[year, self.experiment_name] = \
                calc_temp_storage_spring_2(total_storage, melt_period, snowmelt, rainfall)
            zetas['S_fraction_spring'].loc[year, self.experiment_name] = \
            calc_S_fraction_spring(total_storage, melt_period, Q_year)
            zetas['S_dynamic_fraction_spring'].loc[year, self.experiment_name] = \
            calc_S_dynamic_fraction_spring(total_storage, melt_period, Q_year)
            zetas['antecedent_Q'].loc[year, self.experiment_name] = \
            calc_antecedent_Q(Q_year, melt_period)
            zetas['spring_rf_fraction'].loc[year, self.experiment_name] = \
            calc_spring_rf_fraction(melt_period, rainfall_year, rfmelt_year)
            zetas['spring_et_fraction'].loc[year, self.experiment_name] = \
            calc_spring_et_fraction(melt_period, et_year, rfmelt_year)
            zetas['DJF_rainfall_fraction'].loc[year, self.experiment_name] = \
            calc_DJF_rainfall_fraction(rainfall_year)
            zetas['MAM_rainfall_fraction'].loc[year, self.experiment_name] = \
            calc_MAM_rainfall_fraction(rainfall_year)

            # zetas['DJF_temp'].loc[year, self.experiment_name] = \
            # calc_DJF_temp(temp_year)
            # zetas['MAM_temp'].loc[year, self.experiment_name] = \
            # calc_MAM_temp(temp_year)
            zetas['DJF_pet'].loc[year, self.experiment_name] = \
            calc_DJF_pet(pet_year)
            zetas['MAM_pet'].loc[year, self.experiment_name] = \
            calc_MAM_pet(pet_year)
            zetas['spring_pet'].loc[year, self.experiment_name] = \
            calc_spring_pet(melt_period, pet_year)
            # Calculate meltseason_beta period: from t_SWE_max to t_SWE_end + 1 month
            try:
                t_SWE_max = self.swe_signatures[year]['obs']['t_SWE_max']
                t_SWE_end = self.swe_signatures[year]['obs']['t_SWE_end']
                
                # Check if values are valid (not NaN)
                if np.isnan(t_SWE_max) or np.isnan(t_SWE_end):
                    raise ValueError("t_SWE_max or t_SWE_end is NaN")
                
                # Convert DOY to dates
                year_start = pd.Timestamp(f'{year-1}-10-01')
                meltseason_beta_start = year_start + pd.Timedelta(days=int(t_SWE_max))
                meltseason_beta_end = year_start + pd.Timedelta(days=int(t_SWE_end)) + pd.DateOffset(months=1)
                
                # Ensure end date doesn't exceed year end
                year_end = pd.Timestamp(f'{year}-09-30')
                if meltseason_beta_end > year_end:
                    meltseason_beta_end = year_end
                
                # Ensure start date is not after end date
                if meltseason_beta_start >= meltseason_beta_end:
                    raise ValueError("meltseason_beta_start >= meltseason_beta_end")
                
                # Create date range for meltseason_beta
                meltseason_beta_period = pd.date_range(start=meltseason_beta_start, end=meltseason_beta_end, freq='D')
                
                # Calculate beta zetas using meltseason_beta_period
                zetas['temp_storage_spring_beta'].loc[year, self.experiment_name] = \
                    calc_temp_storage_spring(total_storage, meltseason_beta_period, snowmelt, rainfall)
                zetas['temp_storage_spring_2_beta'].loc[year, self.experiment_name] = \
                    calc_temp_storage_spring_2(total_storage, meltseason_beta_period, snowmelt, rainfall)
                zetas['spring_rf_fraction_beta'].loc[year, self.experiment_name] = \
                    calc_spring_rf_fraction(meltseason_beta_period, rainfall_year, rfmelt_year)
                zetas['spring_et_fraction_beta'].loc[year, self.experiment_name] = \
                    calc_spring_et_fraction(meltseason_beta_period, et_year, rfmelt_year)
            except (KeyError, ValueError, TypeError) as e:
                # If t_SWE_max or t_SWE_end are not available or invalid, set beta zetas to NaN
                print(f"Warning: Could not calculate beta zetas for {self.experiment_name} year {year}: {e}")
                zetas['temp_storage_spring_beta'].loc[year, self.experiment_name] = np.nan
                zetas['temp_storage_spring_2_beta'].loc[year, self.experiment_name] = np.nan
                zetas['spring_rf_fraction_beta'].loc[year, self.experiment_name] = np.nan
                zetas['spring_et_fraction_beta'].loc[year, self.experiment_name] = np.nan
            
            ERRA_prep.loc[mask, 'Q'] = Q_year
            ERRA_prep.loc[mask, 'input'] = rfmelt_year
            ERRA_prep.loc[mask, 'mask'] = 0
            ERRA_prep.loc[melt_period, 'mask'] = 1
            

        else:
            zetas['temp_storage_spring'].loc[year, self.experiment_name] = np.nan
            zetas['temp_storage_spring_2'].loc[year, self.experiment_name] = np.nan
            zetas['spring_rf_fraction'].loc[year, self.experiment_name] = np.nan
            zetas['spring_et_fraction'].loc[year, self.experiment_name] = np.nan
            zetas['temp_storage_spring_beta'].loc[year, self.experiment_name] = np.nan
            zetas['temp_storage_spring_2_beta'].loc[year, self.experiment_name] = np.nan
            zetas['spring_rf_fraction_beta'].loc[year, self.experiment_name] = np.nan
            zetas['spring_et_fraction_beta'].loc[year, self.experiment_name] = np.nan
        
        
        
        
        
        
        # Overview plot
        if plotting== True:
            # f1, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            # plt.subplots_adjust(hspace=0.05)
            
            # SWE[mask].plot(ax=axes[0], color='black', label='_noLegend')
            # melt_out = self.swe_signatures[year]['obs']['t_SWE_end']
            # axes[0].axvline(SWE[mask].index[melt_out], color='black', linestyle='--', label='Melt out date')
            # axes[0].set_ylabel('SWE (mm)')
            # axes[0].grid(alpha=0.5)
            # axes[0].legend(loc='upper left')

            # snowmelt[mask].plot(ax=axes[1], color='black', label='Snowmelt')
            # rainfall[mask].plot(ax=axes[1], color='tab:blue', alpha=0.6, label='Rainfall')
            # et[mask].plot(ax=axes[1], color='green', alpha=0.6, label='ET')
            # axes[1].set_ylabel('Snowmelt (mm/day)')
            # axes[1].grid(alpha=0.5)
            # axes[1].legend(loc='upper left')

            # # Plot normalized storage for melt period
            # if len(snowmelt_reduced) > 0 and snowmelt_reduced.sum() > 0:
            #     total_water_spring = snowmelt[melt_period].sum() + rainfall[melt_period].sum()
            #     if total_water_spring > 0:
            #         norm_storage_spring = total_storage[melt_period] / total_water_spring
            #         mean_norm_storage_spring = norm_storage_spring.mean()
                    
            #         norm_storage_spring.plot(ax=axes[2], color='black', label='Normalized storage')
            #         axes[2].fill_between(melt_period, norm_storage_spring[melt_period], mean_norm_storage_spring,
            #                             facecolor='white', hatch='/', alpha=0.5, label='Storage variation area')
            #         axes[2].axhline(mean_norm_storage_spring, color='black', linestyle='--',
            #                     label='Mean normalized storage')
            #         axes[2].axvline(q10_timing, color='black', linestyle='--', label='Q05 timing')
            #         axes[2].axvline(q90_timing, color='black', linestyle='--', label='Q95 timing')
            #         axes[2].axvline(end_of_melt, color='tab:blue', linestyle='--', label='End of melt period')
            #         axes[2].axvline(start_of_melt, color='tab:blue', linestyle='--', label='Start of melt period')
            
            # axes[2].set_ylabel('Normalized storage')
            # axes[2].grid(alpha=0.5)
            # axes[2].legend(loc='upper right')

            # Q.loc[mask].squeeze().plot(ax=axes[3], color='black', label='Synthetic obs')
            # COM = int(self.Qsig_obs.loc[year]['t_hfd_meltseason'])
            # try:
            #     actual_Qobs.loc[mask].squeeze().plot(ax=axes[3], color='black', linestyle='dotted',
            #                                         label='Real-world obs')
            # except Exception as e:
            #     print(f"No real-world obs for {self.experiment_name} year {year}: {e}")
            # axes[3].axvline(Q.loc[mask].index[COM + 180], color='black',
            #                 linestyle='--', label='Meltseason center of mass')
            # axes[3].set_ylabel('Discharge (mm/day)')
            # axes[3].grid(alpha=0.5)
            # axes[3].legend(loc='upper left')

            # # Get zeta values for plot title
            # rf_ratio = zetas['rainfall_mixing_ratios'].loc[year, self.experiment_name]
            # et_ratio = zetas['et_mixing_ratios'].loc[year, self.experiment_name]
            # sf_ratio_value = zetas['sf_ratio'].loc[year, self.experiment_name]
            # delay = zetas['temp_storage'].loc[year, self.experiment_name]
            # area = self.E.dem_area
            
            # text1 = f"{self.experiment_name} {year}        area = {area:.0f} km²\n"
            # text2 = f"delay={delay:.1f} [-]       "
            # text3 = f"Snowfall fraction={sf_ratio_value:.2f}\n "
            # text4 = f"Rain mixing ratio={rf_ratio:.2f}           ET mixing ratio={et_ratio:.2f}"
            # plt.suptitle(f"{text1}{text2}{text3}{text4}", fontsize=16)

            # plt.savefig(join(plot_dir, f"Overview_wStorage_{self.experiment_name}_{year}.png"),
            #             dpi=300, bbox_inches='tight')
            # if not year == 2005:
            #     plt.close()

            if year == 2003:
                melt_period_mask = pd.date_range(start=f'{year}-01-01', end=f'{year}-08-30', freq='D')
                f1, axes = plt.subplots(3, 1, figsize=(6,5), sharex=True)
                plt.subplots_adjust(hspace=0.05)
                ax0_twin = axes[0].twinx()
                SWE[melt_period_mask].plot(ax=ax0_twin, color='black', label='SWE', alpha=0.5)
                melt_out = self.swe_signatures[year]['obs']['t_SWE_end']
                # axes[0].axvline(SWE[melt_period_mask].index[melt_out], 
                # color='black', linestyle='--', label='Melt out date')
                ax0_twin.set_ylabel('SWE (mm)')
                axes[0].grid(alpha=0.5)

                snowmelt[melt_period_mask].plot(ax=axes[0], color='black', label='Snowmelt')
                rainfall[melt_period_mask].plot(ax=axes[0], color='tab:blue', alpha=0.6, label='Rainfall')
                # et[mask].plot(ax=axes[0], color='green', alpha=0.6, label='ET')
                
                melt_period_start = zetas['melt_period_start'].loc[year, self.experiment_name]
                melt_period_end = zetas['melt_period_end'].loc[year, self.experiment_name]
                axes[0].axvline(melt_period_start, color='black', linestyle='--')
                axes[0].axvline(melt_period_end, color='black', linestyle='--')

                #add meltseason_beta
                # meltseason_beta_start = zetas['meltseason_beta_start'].loc[year, self.experiment_name]
                # meltseason_beta_end = zetas['meltseason_beta_end'].loc[year, self.experiment_name]
                # axes[0].axvline(meltseason_beta_start, color='grey', linestyle='--')
                # axes[0].axvline(meltseason_beta_end, color='grey', linestyle='--')

                melt_com = self.swe_signatures[year]['obs']['t_melt_hfd']
                melt_com_date = pd.to_datetime(f'{year-1}-10-01') + pd.Timedelta(days=melt_com)
                axes[0].axvline(melt_com_date, color='black', linestyle='-.')
                
                axes[0].set_ylabel('Flux \n (mm/day)')
                axes[0].grid(alpha=0.5)
                # Combine legends from both axes
                lines1, labels1 = axes[0].get_legend_handles_labels()
                lines2, labels2 = ax0_twin.get_legend_handles_labels()
                axes[0].legend(lines1 + lines2, labels1 + labels2, loc='center left')

                # Plot normalized storage for melt period
                total_water_spring = snowmelt[melt_period].sum()+rainfall[melt_period].sum()
                norm_storage_spring = total_storage/total_water_spring
                mean_norm_storage_spring = norm_storage_spring[melt_period].mean()
                start_norm_storage_spring = norm_storage_spring[melt_period].iloc[0]
                # temp_storage_value_spring = (np.abs(norm_storage_spring-mean_norm_storage_spring)).sum()

                norm_storage_spring[melt_period_mask].plot(ax = axes[1],color = 'black')
                # axes[1].fill_between(melt_period, norm_storage_spring[melt_period], mean_norm_storage_spring,
                #                 facecolor = 'white', hatch = '/', alpha = 0.5)
                # axes[1].axhline(mean_norm_storage_spring, color = 'black',linestyle = '--', label = 'Mean normalized storage')
                axes[1].fill_between(melt_period, norm_storage_spring[melt_period], start_norm_storage_spring,
                                facecolor = 'white', hatch = '//', alpha = 0.5)
                axes[1].axhline(start_norm_storage_spring, color = 'black',linestyle = 'dotted', label = 'Initial \n storage')
                axes[1].set_ylabel('Normalized \n storage [-]')
                axes[1].axvline(melt_period_start, color='black', linestyle='--')
                axes[1].axvline(melt_period_end, color='black', linestyle='--')
                axes[1].grid(alpha = 0.5)
                axes[1].legend(loc = 'center left')


                Q.loc[melt_period_mask].squeeze().plot(ax=axes[2], color='black', label='Synthetic obs')
                #old COM 
                # COM = int(self.Qsig_obs.loc[year]['t_hfd_meltseason'])
                # #the actual COM date is COM days after october 1st, write it
                # COM_date = pd.to_datetime(f'{year}-{self.DOUBLE_MONTHS[0]}-01') + pd.Timedelta(days=COM)
                
                #new COM
                COM = int(self.Qsig_obs.loc[year]['t_hfd_meltseason2'])
                COM_date = melt_period_start + pd.Timedelta(days=COM)
                
                # Q_onset = self.Qsig_obs.loc[year]['t_Qstart']
                # Q_onset_date = pd.to_datetime(f'{year-1}-10-01') + pd.Timedelta(days=Q_onset)
                # print(self.experiment_name, year, Q_onset_date)



                Q_year = Q.loc[melt_period_mask]
                #calculate cayan onset date: date when Q first exceeds 2x mean winter Q
                # mean_winter_Q = Q_year[Q_year.index.month.isin([12,1,2])].mean()
                # Q_2x_mean_winter_Q = 2 * mean_winter_Q
                # Q_exceeds_2x_mean_winter_Q = Q_year > Q_2x_mean_winter_Q
                # Q_onset_cayan_date = Q_exceeds_2x_mean_winter_Q.idxmax()
                if self.BASIN != 'Dischma':
                    try:
                        actual_Qobs.loc[melt_period_mask].squeeze().plot(ax=axes[2], color='black', linestyle='dotted',
                                                            label='Real-world obs')
                    except Exception as e:
                        print(f"No real-world obs for {self.experiment_name} year {year}: {e}")
                axes[2].axvline(COM_date, color='black',
                                linestyle='-.')
                # axes[2].axvline(Q_onset_cayan_date, color='tab:orange', linestyle='dotted')
                # axes[2].axvline(Q_onset_date, color='tab:red', linestyle='dotted')
                axes[2].axvline(melt_period_end, color='black', linestyle='--')
                axes[2].axvline(melt_period_start, color='black', linestyle='--')
                axes[2].set_ylabel('Q (mm/day)')
                axes[2].grid(alpha=0.5)
                axes[2].legend(loc='upper left')

                # Get zeta values for plot title
                rf_ratio = zetas['rainfall_mixing_ratios'].loc[year, self.experiment_name]
                et_ratio = zetas['et_mixing_ratios'].loc[year, self.experiment_name]
                sf_ratio_value = zetas['sf_ratio'].loc[year, self.experiment_name]
                spring_rf_fraction = zetas['spring_rf_fraction'].loc[year, self.experiment_name]
                delay = zetas['temp_storage_spring'].loc[year, self.experiment_name]
                area = self.E.dem_area
                
                text1 = f"{self.experiment_name} {year} \n"
                text2 = f"Characteristic snowmelt repsonse time={delay:.1f} days \n"
                text3 = f"Meltseason rainfall fraction={spring_rf_fraction:.2f}\n "
                # text4 = f"Rain mixing ratio={rf_ratio:.2f}           ET mixing ratio={et_ratio:.2f}"
                plt.suptitle(f"{text1}{text2}{text3}", fontsize=14,y = 1.02)

                plt.savefig(join(plot_dir, f"Overview_compact_wnewtiming_{self.experiment_name}_{year}.png"),
                            dpi=300, bbox_inches='tight')
            # plt.show()
            # if not year == 2005:
            #     plt.close()
    
    ERRA_prep.to_csv(join(self.ROOTDIR,"aux_data", "ERRA_prep", f"ERRA_prep_{self.experiment_name}.csv"))
exp_name_list = [self.experiment_name for self in LOA_objects.values()]
np.savetxt(join(self.ROOTDIR,"aux_data", "ERRA_prep", f"basin.txt"), exp_name_list, delimiter='\n', fmt='%s')

# Save all zetas to CSV files
for name, config in zeta_configs.items():
    zetas[name].to_csv(config['filepath'])
#%%
# Create melted dataframes for analysis (using zetas)
# Create melted dataframes for analysis (using zetas)
melted_dfs = {}
for name, df in zetas.items():
    melted_dfs[name] = df.melt(ignore_index=False, value_name=name).reset_index(names='Year')

# Merge all melted dataframes
mixing_df = melted_dfs['rainfall_mixing_ratios']
for name, df in melted_dfs.items():
    if name != 'rainfall_mixing_ratios':
        mixing_df = mixing_df.merge(df, on=['Year', 'variable'])
        print(f"Merged {name} dataframe with shape {df.shape}")
        print(f"Mixing dataframe shape: {mixing_df.shape}")

# Filter to only include experiments in EXPS
mixing_df = mixing_df[mixing_df['variable'].isin(list(EXPERIMENT_CONFIG.values()))]

#count number of dischma catchments by counting the number of experiment_names with a number in it
try:
    dischma_count = mixing_df[mixing_df['variable'].str.contains(r'\d')].groupby('Year')['variable'].nunique()[2016]
except:
    dischma_count = 0
WUS_count = len(EXPS) - dischma_count
palette1 = sns.color_palette('tab20b', n_colors=dischma_count)
palette2 = sns.color_palette('colorblind',n_colors =WUS_count)
palette_x = palette1[::-1] + palette2[::-1]

sns.pairplot(mixing_df,
             hue='variable',
             palette=palette_x)


sns.pairplot(mixing_df,
             vars = ['ERRA_center_of_mass','S_fraction','delays','temp_storage'],
             hue = 'variable',
             palette = palette1,
             )
#%% plot k, rainfall mixing and et 

#make a map
# safeeq_filtered = safeeq[safeeq['short_name'].isin(basin_list)]
# safeeq_gdf = gpd.GeoDataFrame(safeeq_filtered, geometry = gpd.points_from_xy(
#     safeeq_filtered['LON'], safeeq_filtered['LAT']),
#     crs = 'EPSG:4326')
# safeeq_gdf['sf_ratio'] = np.nan
# for basin in basin_list:
#     safeeq_gdf.loc[safeeq_gdf['short_name']==basin,'sf_ratio'] = sf_ratio[basin].mean()

if not np.all(np.array(basin_list) =='Dischma'):
    basin_gdf = gpd.GeoDataFrame(index = basin_list)


    for basin in basin_list:
        if basin == 'Dischma':
            continue
        S =SwissStation(basin)
        basin_gdf.loc[basin,'LAT'] = S.lat
        basin_gdf.loc[basin,'LON'] = S.lon
        basin_gdf.loc[basin,'AREA [sqkm]'] = S.area
        basin_gdf.loc[basin, 'sf_ratio'] = zetas['sf_ratio'][basin].mean()
    basin_gdf.geometry = gpd.points_from_xy(basin_gdf['LON'],basin_gdf['LAT'],crs = 'EPSG:4326')


    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    import cartopy.io.img_tiles as cimgt
    import matplotlib.patheffects as patches
    import cartopy.feature as cfeature
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree(),
                        frameon=False)
    ax.patch.set_visible(False)

    # ax.set_extent([-125, -100, 35, 40], ccrs.Geodetic())
    extent = [-125, -103.5, 36.5, 50]
    ax.set_extent(extent, ccrs.Geodetic())
    ax.set_facecolor("#dceefc")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2efe8", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dceefc", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4, edgecolor="0.3", zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=1, edgecolor="0.4", alpha=1, zorder=1)
    states = cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "50m", facecolor="none"
    )

    ax.add_feature(states, edgecolor="0.35", linewidth=0.35, zorder=2)
    hydrography_path = "/home/pwiersma/scratch/Data/GIS/Hydrography/HydroRIVERS_v10_na.gdb/HydroRIVERS_v10_na.gdb"
    hydrography_gdf = gpd.read_file(hydrography_path, layer='HydroRIVERS_v10_na')
    #filter any entries with ORD_STRA<5
    hydrography_gdf = hydrography_gdf[hydrography_gdf['ORD_STRA']>=4]
    hydrography_gdf = hydrography_gdf.to_crs(epsg=4326)
    hydrography_gdf.plot(ax = ax, color = 'tab:blue', linewidth = 0.2, alpha = 0.5)

    gtopo_file = join("/home/pwiersma/scratch/Data/DEM/GTOPO/",f'WUS_143_gtopo.nc')
    gtopo_clipped = xr.open_dataarray(gtopo_file).squeeze()

    gtopo_clipped.plot(ax = ax,cmap = 'Greys', alpha = 0.5,vmax = 3500,
    add_colorbar = False)


    # Use adjustText to prevent text overlap
    from adjustText import adjust_text
    # area = safeeq_gdf['AREA [sqkm]'].astype(float)
    area = basin_gdf['AREA [sqkm]'].astype(float)
    # sf = safeeq_gdf['sf_ratio'].astype(float)

    # Scale marker sizes (s is marker area in points^2). Tweak s_min/s_max for visual result.
    s_min, s_max = 50, 800
    sizes = (area - area.min()) / (area.max() - area.min() + 1e-12) * (s_max - s_min) + s_min

    scatter = ax.scatter(
        basin_gdf['LON'], basin_gdf['LAT'],
        s=sizes*0.5,
        c=basin_gdf['sf_ratio'],
        cmap='Blues_r',
        edgecolor='black', linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
        alpha=0.8,
        vmin =0.3, vmax = 0.9
    )

    # Colorbar for sf_ratio (snowfall ratio)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.3, pad=0.02,
                        anchor = (-0.9,0.68),location ='right')
    cbar.set_label('Snowfall ratio')

    from matplotlib.lines import Line2D
    # Size legend: pick representative AREA values (km^2)
    size_legend_vals = [50, 200, 500]  # choose values meaningful for your data
    handles = []
    labels = []
    for val in size_legend_vals:
        # compute corresponding marker size used in the scatter
        s_val = (val - area.min()) / (area.max() - area.min() + 1e-12) * (s_max - s_min) + s_min
        circle = Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                        markeredgecolor='black', markersize=np.sqrt(s_val)*0.5,
                        linewidth=0)
        handles.append(circle)
        labels.append(f'{val} km²')
    leg = ax.legend(handles, labels, 
                    title='Catchment area',
                    loc=(0.65,0.7), frameon=False)
    ax.add_artist(leg)

    texts = []
    for i, row in basin_gdf.iterrows():
        if row.name =='Dischma':
            continue
        text = ax.text(row['LON'], row['LAT'], row.name,
                    fontsize=6, weight='bold', ha='center', va='center')
        texts.append(text)

    # Adjust text positions to avoid overlap
    adjust_text(texts, ax=ax,
        arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
    avoid_self =True,expand=(2, 2))

    plt.savefig(join(SHARED_PLOTS_DIR,'WUS_catchments_map_April.png'),
                dpi = 300, bbox_inches = 'tight')

#%% Discharge & climate per WUS region (Sierra / PNW / Interior)
# Plot every catchment-year as an independent line, coloured by WUS region,
# for Q, P, ET and T. Skip all Dischma variants (only keep WUS catchments).

WUS_REGION_COLORS = {
    'Sierra': 'tab:orange',
    'PNW': 'tab:green',
    'Interior': 'tab:purple',
}

# Day-of-water-year ticks (non-leap year reference: Oct 1 = DoWY 1)
_month_names_wy = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar',
                   'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
_month_starts_dowy = [1, 32, 62, 93, 124, 152,
                      183, 213, 244, 274, 305, 336]

_wus_var_info = {
    'Q':  {'ylabel': 'Q [mm/day]',  'title': 'Discharge'},
    'P':  {'ylabel': 'P [mm/day]',  'title': 'Precipitation'},
    'ET': {'ylabel': 'ET [mm/day]', 'title': 'Evapotranspiration'},
    'T':  {'ylabel': 'T [\u00b0C]', 'title': 'Temperature'},
}

wus_region_data = {var: {region: [] for region in WUS_REGION_COLORS}
                   for var in _wus_var_info}
wus_melt_period_dowy = {
    'start': {region: [] for region in WUS_REGION_COLORS},
    'end': {region: [] for region in WUS_REGION_COLORS},
}

def _date_to_water_year_day(date_value, water_year):
    if pd.isna(date_value):
        return np.nan
    water_year_start = pd.Timestamp(f'{water_year-1}-10-01')
    return (pd.to_datetime(date_value) - water_year_start).days + 1

for self in LOA_objects.values():
    if self.ORIG_ID not in EXP_WUS_REGION:
        continue
    region = EXP_WUS_REGION[self.ORIG_ID]

    Q = (self.Qobs * 86400 * 1000 / (self.E.dem_area * 1e6)).squeeze()

    pr = self.meteo_base['pr'].mean(dim=['lat', 'lon']).to_pandas()
    tas = self.meteo_base['tas'].mean(dim=['lat', 'lon']).to_pandas()
    if np.nanmax(tas.values) > 100:
        tas = tas - 273.15

    scalars_file = glob.glob(join(self.SYNDIR, '*Synthetic_obs*.csv'))[0]
    scalars = pd.read_csv(scalars_file, index_col=0, parse_dates=True)
    et = scalars['et']

    for year in range(self.START_YEAR, self.END_YEAR + 1):
        wy_index = pd.date_range(start=f'{year-1}-10-01',
                                 end=f'{year}-09-30', freq='D')
        dowy = np.arange(1, len(wy_index) + 1)

        wus_region_data['Q'][region].append((dowy, Q.reindex(wy_index).values))
        wus_region_data['P'][region].append((dowy, pr.reindex(wy_index).values))
        wus_region_data['ET'][region].append((dowy, et.reindex(wy_index).values))
        wus_region_data['T'][region].append((dowy, tas.reindex(wy_index).values))

        melt_start = zetas['melt_period_start'].loc[year, self.experiment_name]
        melt_end = zetas['melt_period_end'].loc[year, self.experiment_name]
        wus_melt_period_dowy['start'][region].append(
            _date_to_water_year_day(melt_start, year)
        )
        wus_melt_period_dowy['end'][region].append(
            _date_to_water_year_day(melt_end, year)
        )

def _stack_catchment_years(series_list, max_dowy=366):
    """Stack catchment-year series onto a common day-of-water-year grid."""
    cols = {}
    for i, (dowy, vals) in enumerate(series_list):
        cols[i] = pd.Series(vals, index=dowy)
    if not cols:
        return pd.DataFrame(index=np.arange(1, max_dowy + 1))
    df = pd.DataFrame(cols)
    df = df.reindex(np.arange(1, max_dowy + 1))
    return df

def _plot_melt_period_quantiles(ax, melt_period_dowy, region, color):
    period_styles = {
        'start': '--',
        'end': '-.',
    }
    for period, linestyle in period_styles.items():
        vals = pd.Series(melt_period_dowy[period][region]).dropna()
        if vals.empty:
            continue
        x = vals.quantile(0.50)
        ax.axvline(x, color=color, linestyle=linestyle,
                   alpha=0.9, linewidth=1.7)

for var, info in _wus_var_info.items():
    f_wus, ax_wus = plt.subplots(figsize=(8, 4))
    q90_max_global = -np.inf
    q10_min_global = np.inf
    for region, color in WUS_REGION_COLORS.items():
        series_list = wus_region_data[var][region]
        n_lines = len(series_list)
        for i, (dowy, vals) in enumerate(series_list):
            ax_wus.plot(dowy, vals, color=color, alpha=0.02,
                        linewidth=0.6)

        if n_lines == 0:
            continue
        df_region = _stack_catchment_years(series_list)
        q10 = df_region.quantile(0.10, axis=1)
        q50 = df_region.quantile(0.50, axis=1)
        q90 = df_region.quantile(0.90, axis=1)
        ax_wus.plot(q50.index, q50.values, color=color,
                    linewidth=2.0, linestyle='-')
        ax_wus.plot(q10.index, q10.values, color=color,
                    linewidth=1.2, linestyle='--')
        ax_wus.plot(q90.index, q90.values, color=color,
                    linewidth=1.2, linestyle=':')
        q90_max_global = max(q90_max_global, np.nanmax(q90.values))
        q10_min_global = min(q10_min_global, np.nanmin(q10.values))
        if var == 'Q':
            _plot_melt_period_quantiles(ax_wus, wus_melt_period_dowy,
                                        region, color)

    ax_wus.set_xticks(_month_starts_dowy)
    ax_wus.set_xticklabels(_month_names_wy)
    ax_wus.set_xlim(0, 367)
    if np.isfinite(q90_max_global):
        span = q90_max_global - q10_min_global if np.isfinite(q10_min_global) else 0
        top = q90_max_global + 0.05 * max(span, abs(q90_max_global))
        if np.isfinite(q10_min_global):
            bottom = q10_min_global - 0.05 * max(span, abs(q10_min_global))
            if var != 'T':
                bottom = min(bottom, 0)
            ax_wus.set_ylim(bottom=bottom, top=top)
        else:
            ax_wus.set_ylim(top=top)
    ax_wus.set_xlabel('Water year month')
    ax_wus.set_ylabel(info['ylabel'])
    ax_wus.set_title(f"{info['title']} \u2014 WUS regions (catchment-years)")
    ax_wus.grid(alpha=0.3)

    # Region colours (separate from linestyle meaning)
    region_handles = [
        Line2D([0], [0], color=color, linewidth=2.0, linestyle='-',
               label=region)
        for region, color in WUS_REGION_COLORS.items()
    ]
    leg_region = ax_wus.legend(handles=region_handles,
                               loc='upper left', frameon=True, fontsize=8)
    ax_wus.add_artist(leg_region)

    # Grey proxy lines for quantile / melt linestyles
    style_handles = [
        Line2D([0], [0], color='0.4', linewidth=2.0, linestyle='-',
               label='Q50'),
        Line2D([0], [0], color='0.4', linewidth=1.2, linestyle='--',
               label='Q10'),
        Line2D([0], [0], color='0.4', linewidth=1.2, linestyle=':',
               label='Q90'),
    ]
    if var == 'Q':
        style_handles.extend([
            Line2D([0], [0], color='0.4', linewidth=1.7, linestyle='--',
                   label='Melt start Q50'),
            Line2D([0], [0], color='0.4', linewidth=1.7, linestyle='-.',
                   label='Melt end Q50'),
        ])
    ax_wus.legend(handles=style_handles,
                  loc='upper right', frameon=True, fontsize=8)

    plt.tight_layout()
    plt.savefig(join(SHARED_PLOTS_DIR,
                     f"WUS_regions_{var}_catchment_years.png"),
                dpi=300, bbox_inches='tight')
    
    plt.show(f_wus)


#%%
# if len(EXPS)>1:
allcorrs_list = []

for self in LOA_objects.values():
    df = self.QSWE_corr_pairs.copy()
    df['EXP_ID'] = self.ORIG_ID
    df['BASIN'] = self.experiment_name
    allcorrs_list.append(df)    
allcorrs = pd.concat(allcorrs_list, axis=0).reset_index(drop=True)

bestcorrs = allcorrs.groupby(['level_0', 'level_1'])['correlation'].mean().reset_index().sort_values('correlation', ascending=False)

#keep the common string among EXPS list

allcorrs.to_csv(join(SHARED_PLOTS_DIR,'all_correlations.csv'))
bestcorrs.to_csv(join(SHARED_PLOTS_DIR,'best_correlations.csv'))

# SWE_target_metric
# q_metrics = q_selection + q_combos
# mask = QSWE_corr['level_1'].isin(q_metrics)
# QSWE_corr = QSWE_corr[mask]
# self.SWE_target_metric = 'SWE_SWS_elev_ME'

swe_metrics = [SWE_target_metric]
mask = bestcorrs['level_0'].isin(swe_metrics)
bestcorrs_selection = bestcorrs[mask]
# 
#print the 10 best in a nice way, or make a nice figure showing the top 10
# print(bestcorrs_selection.head(10))
# print(bestcorrs_selection.tail(10))
top_q = bestcorrs_selection['level_1'].head(5).values
# top_q = np.append(top_q,['KGE_meltseason'])
selection_corrs = allcorrs[(allcorrs['level_0'].isin(swe_metrics))& (allcorrs['level_1'].isin(top_q))]
#sort selection_corrs based on top_q 
selection_corrs['level_1'] = pd.Categorical(selection_corrs['level_1'],
                                    categories = top_q, ordered = True)
selection_corrs.sort_values('level_1',inplace = True)

f1,ax1 = plt.subplots(figsize = (5,0.5*len(top_q)))
sns.stripplot(selection_corrs, hue = "EXP_ID", y = 'level_1',
            x = 'correlation', orient = 'h',
                dodge = True, ax = ax1,palette = 'colorblind',
                legend = False)
# sns.boxenplot(selection_corrs, hue = "EXP_ID", y = 'level_1', x = 'correlation', orient = 'h',
#                     dodge = True, ax = ax1,palette = 'Blues',
#                     )
ax1.set_xlabel(r'$\rho$ [-]')
ax1.set_ylabel('Q metric')
ax1.grid()
ax1.axvline(0, color = 'black',  linestyle = '--')
ax1.set_title(f"Top 5 Q metric correlating \n with {SWE_target_metric} ")
# ax1.legend(loc = (1.05,0.5))
# ax1.legend(None)
# plt.suptitle(f"{self.EXP_ID_translation[common]}",y = 1.02, fontsize = 16)
plt.savefig(join(SHARED_PLOTS_DIR, f"top10_Q_combos_{SWE_target_metric}.png"),
            dpi = 300, bbox_inches = 'tight')

#%% Plot best combos for multipel experiments
# Paper-facing metric names for compact legends.
PAPER_METRIC_NAME_MAP = {
    'SWE_melt_NSE': 'Melt-NSE',
    'NSE_meltseason2': 'Q-NSE',
    'KGE_meltseason2': 'Q-KGE',
    'melt_sum_APE': 'Melt-bias',#r'$\Sigma$Melt bias',
    'melt_sum_grid_MAPE':'Melt-Spatial bias',
    'Qmean_meltseason2_APE': 'Q-bias',#r'$\Sigma$Q bias',
    'SWE_NSE': 'SWE-NSE',
    'SWE_SWS_APE': 'SWS-bias',
    't_SWE_end_grid_ME':'Melt-out timing-spatial error',
    't_melt_hfd_ME':'Melt-COM', 
    't_hfd_meltseason2_ME':'Q-COM',
    'Q5_APE':'Q5-bias',
    'logNSE_meltseason':'Q-logNSE',
    'SWE_melt_NSE_grid': 'Melt-spatial NSE',
    't_SWE_max_ME':'Peak SWE timing error',
    't_Qstart_ME':'Q start timing error',
    't_melt_onset_ME':'Melt onset timing error',
    


    
}
                    # 't_melt_hfd_ME__t_hfd_meltseason2_ME',

# Color design for compact pair plots:
# - base hue reflects SWE metric family
# - lightness reflects Q metric family
PAPER_SWE_BASE_COLORS = {
    'SWE_melt_NSE': '#1b9e77',
    'melt_sum_APE': '#d95f02',
    'SWE_NSE': '#7570b3',
}
PAPER_Q_LIGHTEN = {
    'Qmean_meltseason2_APE': 0.8,  
    'NSE_meltseason2': 0,       
    'KGE_meltseason2': 0.4,        
}

# Create figure and plot
def QSWE_pair_across_settings(pairs_selection_df,pair_selection,
palette = 'colorblind',figsize = None,dodge = 0.75,fontsize = 12,legend_loc = (1.11,0.3),legend_fontsize = 12):
    """ This function plots corr for multiple experiments 
    for a selection of QSWE metric pairs
    """
    setting_count = pairs_selection_df['Experiment'].nunique()
    if figsize is None:
        fig = plt.figure(figsize = (5,2*len(pair_selection)))
    else:
        fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    sns.stripplot(pairs_selection_df, hue="Experiment", y='QSWE_pair',
                x='correlation', orient='h',
                dodge=True if dodge is None else dodge, palette=palette,alpha =0.3,
                legend=False)
    # pairs_selection_means = pairs_selection_df.groupby(['QSWE_pair','EXP_ID'])['correlation'].mean().reset_index()
    sns.pointplot(pairs_selection_df, y='QSWE_pair', hue = 'Experiment', x='correlation', orient='h',
            palette=palette,errorbar = 'ci',dodge = dodge, #dodge gives a division by zero error? 
            markersize=7,marker = 'D',linestyle='',markeredgecolor = 'black',markeredgewidth = 0.5,
                legend=True, zorder = 10000)
    # Create twin axis for right side labels
    if len(pair_selection) == 3:
        #######Ticks just one the left 
        left_labels = [r'$\mathrm{MP}_{fit}$',r'$\mathrm{MP}_{volume}$',r'$\mathrm{MP}_{timing}$'        ]
        # left_labels = ['fit','Volume','Timing']
        ax.set_yticklabels(left_labels, fontsize=fontsize)
        ax.set_ylabel('Q-SWE performance metric pair', fontsize=fontsize)
        ax.legend(loc = legend_loc, fontsize=legend_fontsize)
    elif len(pair_selection) == 2:
        yticklabels = [label.get_text() for label in ax.get_yticklabels()]
        # left_labels = [r'$\rho_\mathrm{daily}$',r'$\rho_\mathrm{seasonal}$']
        left_labels = ['Melt-NSE/Q-NSE','Melt-bias/Q-bias']
        # left_labels = ['Daily','Seasonal']
        # left_labels = 
        ax.set_yticklabels(left_labels, fontsize=fontsize)
        # ax.set_title('Snow-Streamflow skill transfer', fontsize=fontsize+1)
        ax.legend(loc = legend_loc, fontsize=legend_fontsize,ncols = 2,
        title = 'Catchment',title_fontsize = fontsize)
        ax.set_ylabel('')

    ########Ticks on both sides  
    else:           
    # Create twin axis for right side labels
        ax2 = ax.twinx()

        # Get current tick labels and positions
        yticks = ax.get_yticks()
        yticklabels = [label.get_text() for label in ax.get_yticklabels()]

        # Split labels
        left_labels = [translate_SWE_metric_name(label.split('__')[0]) for label in yticklabels]
        right_labels = [translate_Q_metric_name(label.split('__')[1]) for label in yticklabels]

        # Set labels
        ax.set_yticklabels(left_labels, fontsize=fontsize)
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(right_labels, fontsize=fontsize)
        ax.set_ylabel('SWE metric',loc = 'bottom', fontsize=fontsize)
        ax2.set_ylabel('Q metric',loc = 'bottom', fontsize=fontsize)
        ax2.set_ylim(ax.get_ylim())
        ax.legend(loc = legend_loc, fontsize=legend_fontsize,)

    ax.axvline(0,linestyle ='dashed',color = 'black')
    ax.grid(alpha = 0.3)
    ax.set_xlabel(r'$\rho$ [-]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    if len(pair_selection) >3:
        ax2.tick_params(axis='y', labelsize=fontsize)
    ax.set_xlim(right = 1)   
    plt.savefig(join(SHARED_PLOTS_DIR, f"rho_values_QSWE_pair_selection_{len(pair_selection)}.png"),
                dpi = 300, bbox_inches = 'tight')
    plt.savefig(join(SHARED_PLOTS_DIR, f"rho_values_QSWE_pair_selection_{len(pair_selection)}.svg"),
                 bbox_inches = 'tight')    
    plt.show()


# Native lengths of common qualitative seaborn/matplotlib palettes.
# Requesting more colors than these via sns.color_palette interpolates and muddies hues.
_QUAL_PALETTE_NATIVE = {
    'colorblind': 10, 'deep': 10, 'muted': 10, 'pastel': 10,
    'bright': 10, 'dark': 10, 'Set1': 9, 'Set2': 8, 'Set3': 12,
    'tab10': 10, 'tab20': 20, 'tab20b': 20, 'tab20c': 20,
    'Paired': 12, 'Accent': 8, 'Dark2': 8,
}


def _combined_discrete_palette(
    n_colors,
    sources=('colorblind', 'bright', 'Set2', 'Set3', 'tab20'),
):
    """Concatenate full discrete qualitative palettes (no interpolation).

    Useful when many categories are needed (e.g. ~26 WUS catchments): a single
    palette like ``colorblind`` only has ~10 distinct colors. Avoids ``dark``
    (too similar at small marker size) in favor of mid-bright Set2/Set3.
    """
    n = max(1, int(n_colors))
    colors = []
    for name in sources:
        k = _QUAL_PALETTE_NATIVE.get(name)
        chunk = list(sns.color_palette(name, n_colors=k) if k else sns.color_palette(name))
        colors.extend(chunk)
    if not colors:
        colors = list(sns.color_palette('tab20', n_colors=20))
    if len(colors) < n:
        colors = (colors * ((n // len(colors)) + 1))[:n]
    else:
        colors = colors[:n]
    return colors


def _glasbey_palette(n_colors):
    """First ``n_colors`` Glasbey hues from colorcet (maximally distinct)."""
    n = max(1, int(n_colors))
    return [mcolors.to_rgb(h) for h in cc.glasbey_light[:n]]


def _palette_colors(palette, n_colors):
    """Resolve a palette name, ``'combined'``, ``'glasbey'``, or an explicit color sequence.

    If a named palette is shorter than ``n_colors``, fall back to
    ``_combined_discrete_palette`` with that palette first, so seaborn does not
    interpolate muddy near-duplicates.
    """
    n = max(1, int(n_colors))
    if palette is None or palette == 'combined':
        return _combined_discrete_palette(n)
    if palette == 'glasbey':
        return _glasbey_palette(n)
    if isinstance(palette, (list, tuple)):
        seq = list(palette)
        if not seq:
            return _combined_discrete_palette(n)
        if len(seq) < n:
            seq = (seq * ((n // len(seq)) + 1))[:n]
        else:
            seq = seq[:n]
        return seq
    native = _QUAL_PALETTE_NATIVE.get(palette)
    if native is not None and n > native:
        others = [
            s for s in ('colorblind', 'bright', 'Set2', 'Set3', 'tab20', 'tab20b', 'Paired')
            if s != palette
        ]
        return _combined_discrete_palette(n, sources=(palette, *others))
    return list(sns.color_palette(palette, n_colors=n))


def _qswe_pair_across_region_panel(
    ax,
    df_pair,
    title,
    *,
    color_map,
    dodge,
    strip_jitter,
    strip_alpha,
    fontsize,
    marker_dischma,
    marker_wus,
    show_ylabel=True,
):
    """One y=region strip + point panel (Dischma vs WUS); used by v2 and dual variants."""
    dm = df_pair[df_pair['region'] == 'Dischma']
    wus = df_pair[df_pair['region'] == 'WUS']
    dm_o = dm['Experiment'].drop_duplicates().tolist()
    w_o = wus['Experiment'].drop_duplicates().tolist()

    if not dm.empty and dm_o:
        sns.stripplot(
            data=dm,
            y='region', x='correlation', hue='Experiment',
            hue_order=dm_o, orient='h',
            dodge=dodge, jitter=strip_jitter,
            palette={e: color_map[e] for e in dm_o},
            alpha=strip_alpha, size=3, linewidth=0,
            ax=ax, legend=False, zorder=2,
        )
        sns.pointplot(
            data=dm,
            y='region', x='correlation', hue='Experiment',
            hue_order=dm_o, orient='h',
            palette=[color_map[e] for e in dm_o],
            errorbar='ci', dodge=dodge,
            markersize=7, marker=marker_dischma, linestyle='',
            markeredgecolor='black', markeredgewidth=0.5,
            ax=ax, legend=False, zorder=15,
        )
    if not wus.empty and w_o:
        sns.stripplot(
            data=wus,
            y='region', x='correlation', hue='Experiment',
            hue_order=w_o, orient='h',
            dodge=dodge, jitter=strip_jitter,
            palette={e: color_map[e] for e in w_o},
            alpha=strip_alpha, size=3, linewidth=0,
            ax=ax, legend=False, zorder=2,
        )
        sns.pointplot(
            data=wus,
            y='region', x='correlation', hue='Experiment',
            hue_order=w_o, orient='h',
            palette=[color_map[e] for e in w_o],
            errorbar='ci', dodge=dodge,
            markersize=7, marker=marker_wus, linestyle='',
            markeredgecolor='black', markeredgewidth=0.5,
            ax=ax, legend=False, zorder=15,
        )

    ax.axvline(0, linestyle='dashed', color='black')
    ax.grid(alpha=0.3)
    ax.set_xlim(right=1)
    ax.set_xlabel(r'$\rho$ [-]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel('Region', fontsize=fontsize)


def _catchment_handles_for_experiments(experiment_names, color_map):
    return [
        Line2D(
            [0], [0], marker='D', linestyle='',
            markerfacecolor=color_map.get(exp, '#7f7f7f'),
            markeredgecolor='black', markeredgewidth=0.35,
            markersize=6, label=exp
        )
        for exp in experiment_names
    ]


def _add_dischma_wus_figure_legends(
    fig,
    *,
    dischma_exps,
    wus_exps,
    color_map,
    legend_anchor,
    legend_fontsize,
    title_fontsize,
    ncols_dischma=4,
    ncols_wus=2,
):
    """Stack two figure legends (Dischma above, WUS below); skip empty groups."""
    lx, ly = legend_anchor
    ly_dm = min(1, ly + 0.24)
    ly_wus = max(0.06, ly - 0.3)
    lx_wus = lx + 0.025

    h_dm = _catchment_handles_for_experiments(dischma_exps, color_map)
    h_wus = _catchment_handles_for_experiments(wus_exps, color_map)

    if h_dm:
        leg_dm = fig.legend(
            handles=h_dm,
            loc='upper left',
            bbox_to_anchor=(lx, ly_dm),
            fontsize=legend_fontsize,
            ncols=ncols_dischma,
            title='Synthetic Dischma variants',
            title_fontsize=title_fontsize,
            frameon=True,
        )
        fig.add_artist(leg_dm)

    if h_wus:
        fig.legend(
            handles=h_wus,
            loc='upper left',
            bbox_to_anchor=(lx_wus, ly_wus),
            fontsize=legend_fontsize,
            ncols=ncols_wus,
            title='Western US catchments',
            title_fontsize=title_fontsize,
            frameon=True,
        )


# Marker shape encodes WUS subregion (same convention as the bubble plot).
_WUS_SUBREGION_MARKERS = {
    'PNW': 'X',
    'Sierra': '^',
    'Interior': 'P',
    'Unmapped WUS': 's',
}


def _experiment_to_exp_id(experiment_name):
    """Reverse ``EXPERIMENT_CONFIG`` lookup (experiment display name → EXP_ID)."""
    for eid, name in EXPERIMENT_CONFIG.items():
        if name == experiment_name:
            return eid
    return None


def _wus_subregion_for_experiment(experiment_name, exp_id=None):
    eid = exp_id if exp_id is not None else _experiment_to_exp_id(experiment_name)
    if eid is None:
        return 'Unmapped WUS'
    return EXP_WUS_REGION.get(eid, 'Unmapped WUS')


def _wus_marker_for_experiment(experiment_name, exp_id=None):
    return _WUS_SUBREGION_MARKERS.get(
        _wus_subregion_for_experiment(experiment_name, exp_id=exp_id),
        's',
    )


def _catchment_handles_for_experiments_v3(experiment_names, color_map, *, wus_only=False):
    """Legend handles; WUS catchments use PNW/Sierra/Interior markers."""
    handles = []
    for exp in experiment_names:
        if wus_only:
            marker = _wus_marker_for_experiment(exp)
        else:
            marker = 'D'
        handles.append(
            Line2D(
                [0], [0], marker=marker, linestyle='',
                markerfacecolor=color_map.get(exp, '#7f7f7f'),
                markeredgecolor='black', markeredgewidth=0.35,
                markersize=6, label=exp,
            )
        )
    return handles


def _add_dischma_wus_figure_legends_v3(
    fig,
    *,
    dischma_exps,
    wus_exps,
    color_map,
    legend_anchor,
    legend_fontsize,
    title_fontsize,
    ncols_dischma=4,
    ncols_wus=2,
):
    """Like ``_add_dischma_wus_figure_legends``; WUS legend markers encode subregion."""
    lx, ly = legend_anchor
    ly_dm = min(1, ly + 0.24)
    ly_wus = max(0.06, ly - 0.3)
    lx_wus = lx + 0.025

    h_dm = _catchment_handles_for_experiments_v3(dischma_exps, color_map, wus_only=False)
    h_wus = _catchment_handles_for_experiments_v3(wus_exps, color_map, wus_only=True)

    if h_dm:
        leg_dm = fig.legend(
            handles=h_dm,
            loc='upper left',
            bbox_to_anchor=(lx, ly_dm),
            fontsize=legend_fontsize,
            ncols=ncols_dischma,
            title='Synthetic Dischma variants',
            title_fontsize=title_fontsize,
            frameon=True,
        )
        fig.add_artist(leg_dm)

    if h_wus:
        fig.legend(
            handles=h_wus,
            loc='upper left',
            bbox_to_anchor=(lx_wus, ly_wus),
            fontsize=legend_fontsize,
            ncols=ncols_wus,
            title='Western US catchments',
            title_fontsize=title_fontsize,
            frameon=True,
        )


def _qswe_pair_across_region_panel_v3(
    ax,
    df_pair,
    title,
    *,
    color_map,
    dodge,
    strip_jitter,
    strip_alpha,
    fontsize,
    marker_dischma,
    show_ylabel=True,
):
    """Like ``_qswe_pair_across_region_panel``, but WUS mean markers encode subregion.

    Dischma: diamond (``marker_dischma``). WUS: PNW=X, Sierra=^, Interior=P.
    CIs are drawn first; mean markers are overlaid afterward at higher zorder so
    they sit above neighboring error bars.
    """
    dm = df_pair[df_pair['region'] == 'Dischma']
    wus = df_pair[df_pair['region'] == 'WUS']
    dm_o = dm['Experiment'].drop_duplicates().tolist()
    w_o = wus['Experiment'].drop_duplicates().tolist()

    if not dm.empty and dm_o:
        sns.stripplot(
            data=dm,
            y='region', x='correlation', hue='Experiment',
            hue_order=dm_o, orient='h',
            dodge=dodge, jitter=strip_jitter,
            palette={e: color_map[e] for e in dm_o},
            alpha=strip_alpha, size=3, linewidth=0,
            ax=ax, legend=False, zorder=2,
        )
        sns.pointplot(
            data=dm,
            y='region', x='correlation', hue='Experiment',
            hue_order=dm_o, orient='h',
            palette=[color_map[e] for e in dm_o],
            errorbar='ci', dodge=dodge,
            markersize=7, marker=marker_dischma, linestyle='',
            markeredgecolor='black', markeredgewidth=0.5,
            ax=ax, legend=False, zorder=15,
        )
    if not wus.empty and w_o:
        sns.stripplot(
            data=wus,
            y='region', x='correlation', hue='Experiment',
            hue_order=w_o, orient='h',
            dodge=dodge, jitter=strip_jitter,
            palette={e: color_map[e] for e in w_o},
            alpha=strip_alpha, size=3, linewidth=0,
            ax=ax, legend=False, zorder=2,
        )
        # Full hue_order keeps dodge slots aligned with the strip.
        # Pass 1: CI whiskers only (no markers) so later markers are never covered.
        for exp in w_o:
            sub = wus[wus['Experiment'] == exp]
            if sub.empty:
                continue
            sns.pointplot(
                data=sub,
                y='region', x='correlation', hue='Experiment',
                hue_order=w_o, orient='h',
                palette=[color_map[e] for e in w_o],
                errorbar='ci', dodge=dodge,
                markersize=0, marker='o', linestyle='',
                ax=ax, legend=False, zorder=12,
            )
        # Pass 2: subregion markers on top of all error bars.
        for exp in w_o:
            sub = wus[wus['Experiment'] == exp]
            if sub.empty:
                continue
            sns.pointplot(
                data=sub,
                y='region', x='correlation', hue='Experiment',
                hue_order=w_o, orient='h',
                palette=[color_map[e] for e in w_o],
                errorbar=None, dodge=dodge,
                markersize=7, marker=_wus_marker_for_experiment(exp), linestyle='',
                markeredgecolor='black', markeredgewidth=0.5,
                ax=ax, legend=False, zorder=40,
            )

    # Ensure mean markers stay above CI line artists (seaborn copies zorder to both).
    for line in ax.lines:
        mk = line.get_marker()
        if mk not in (None, 'None', '') and (line.get_markersize() or 0) > 0:
            line.set_zorder(40)

    ax.axvline(0, linestyle='dashed', color='black')
    ax.grid(alpha=0.3)
    ax.set_xlim(right=1)
    ax.set_xlabel(r'$\rho$ [-]', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel('Region', fontsize=fontsize)
    else:
        ax.set_ylabel('')


def QSWE_pair_across_settings_v2_dual_v3(
    pairs_selection_df,
    pair_left,
    pair_right,
    palette='glasbey',
    figsize=None,
    dodge=0,
    fontsize=12,
    legend_loc=(1.11, 0.3),
    legend_fontsize=12,
    *,
    dischma_palette='tab20b',
    wus_palette=None,
    strip_jitter=False,
    strip_alpha=0.28,
    marker_dischma='D',
):
    """
    Duplicate of ``QSWE_pair_across_settings_v2_dual`` with WUS mean markers by subregion.

    Marker map (same as bubble plot): PNW=``X``, Sierra=``^``, Interior=``P``.
    WUS colors default to Glasbey (``colorcet``). Saves under a ``_v3_`` basename.
    """
    wus_palette_eff = palette if wus_palette is None else wus_palette
    pair_keys = [pair_left, pair_right]
    plot_df = pairs_selection_df[pairs_selection_df['QSWE_pair'].isin(pair_keys)].copy()
    if plot_df.empty:
        return

    if 'region' not in plot_df.columns:
        plot_df['region'] = (
            plot_df['BASIN'].astype(str).str.contains(r'\d').map({True: 'Dischma', False: 'WUS'})
        )
    plot_df = plot_df.dropna(subset=['correlation', 'Experiment', 'region'])
    plot_df['Experiment'] = plot_df['Experiment'].astype(str)

    present_pairs = set(plot_df['QSWE_pair'].astype(str))
    if pair_left not in present_pairs or pair_right not in present_pairs:
        return

    plot_df['QSWE_pair'] = pd.Categorical(
        plot_df['QSWE_pair'].astype(str),
        categories=[pair_left, pair_right],
        ordered=True,
    )
    hue_order_global = plot_df['Experiment'].drop_duplicates().tolist()

    dischma_exps = (
        plot_df.loc[plot_df['region'] == 'Dischma', 'Experiment']
        .drop_duplicates().tolist()
    )
    wus_exps = (
        plot_df.loc[plot_df['region'] == 'WUS', 'Experiment']
        .drop_duplicates().tolist()
    )

    dm_colors = sns.color_palette(dischma_palette, n_colors=max(1, len(dischma_exps)))
    if dischma_palette == 'tab20b' and len(dischma_exps) > 0:
        dm_colors = dm_colors[::-1]
    w_colors = _palette_colors(wus_palette_eff, max(1, len(wus_exps)))
    color_map = {}
    color_map.update({e: dm_colors[i] for i, e in enumerate(dischma_exps)})
    color_map.update({e: w_colors[i] for i, e in enumerate(wus_exps)})

    region_order = [r for r in ['Dischma', 'WUS'] if r in plot_df['region'].unique()]
    plot_df['region'] = pd.Categorical(plot_df['region'], categories=region_order, ordered=True)

    if figsize is None:
        figsize = (9.2, max(1.45, 0.52 * len(region_order)))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    axes_flat = axes.flatten()

    for pi, pair in enumerate(pair_keys):
        ax = axes_flat[pi]
        sub = plot_df[plot_df['QSWE_pair'] == pair]
        if sub.empty:
            continue
        if '__' in pair:
            swe_m, q_m = pair.split('__', 1)
            t = (
                f"{PAPER_METRIC_NAME_MAP.get(swe_m, swe_m)}/{PAPER_METRIC_NAME_MAP.get(q_m, q_m)}"
            )
        else:
            t = PAPER_METRIC_NAME_MAP.get(pair, pair)
        _qswe_pair_across_region_panel_v3(
            ax,
            sub,
            t,
            color_map=color_map,
            dodge=dodge,
            strip_jitter=strip_jitter,
            strip_alpha=strip_alpha,
            fontsize=fontsize,
            marker_dischma=marker_dischma,
            show_ylabel=(pi == 0),
        )
        if pi > 0:
            ax.set_ylabel('')
            ax.tick_params(axis='y', left=False, labelleft=False)

    _add_dischma_wus_figure_legends_v3(
        fig,
        dischma_exps=dischma_exps,
        wus_exps=wus_exps,
        color_map=color_map,
        legend_anchor=(legend_loc[0], legend_loc[1]) if legend_loc is not None else (1.02, 0.5),
        legend_fontsize=legend_fontsize,
        title_fontsize=fontsize,
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    n_ex = len(hue_order_global)
    plt.savefig(
        join(SHARED_PLOTS_DIR, f"rho_values_QSWE_across_settings_v2_dual_v3_{n_ex}exps.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.savefig(
        join(SHARED_PLOTS_DIR, f"rho_values_QSWE_across_settings_v2_dual_v3_{n_ex}exps.svg"),
        bbox_inches='tight'
    )
    plt.show()


def QSWE_pair_across_settings_v2(
    pairs_selection_df,
    pair_selection,
    palette='combined',
    figsize=None,
    dodge=0,
    fontsize=12,
    legend_loc=(1.11, 0.3),
    legend_fontsize=12,
    *,
    dischma_palette='tab20b',
    wus_palette=None,
    strip_jitter=False,
    strip_alpha=0.28,
    marker_dischma='D',
    marker_wus='s',
):
    """
    Like ``QSWE_pair_across_settings`` but **y = region** (Dischma vs WUS).

    Strip + mean estimates are drawn **per region** with that region's ``hue_order`` and
    colors, so seaborn does not reserve empty hue slots across regions.

    - Dischma experiments: ``dischma_palette`` (default tab20b, reversed).
    - WUS experiments: ``wus_palette`` or ``palette`` (default ``'combined'``:
      colorblind + bright + Set2 + Set3 + tab20 for ~26 distinct hues).

    Mean markers are set by ``marker_dischma`` and ``marker_wus`` (defaults: diamond / square).

    Legends: two figure legends stacked outside the axes (Dischma above, WUS below);
    Dischma defaults to 4 columns (short labels), WUS to 2 — no separate mean-marker legend.

    Saves under a distinct basename from v1.

    Defaults: short figure height, ``strip_jitter=False``, and ``dodge=0`` on pointplots so means sit on the strips.
    """
    wus_palette_eff = palette if wus_palette is None else wus_palette

    plot_df = pairs_selection_df[pairs_selection_df['QSWE_pair'].isin(pair_selection)].copy()
    if plot_df.empty:
        return

    if 'region' not in plot_df.columns:
        plot_df['region'] = (
            plot_df['BASIN'].astype(str).str.contains(r'\d').map({True: 'Dischma', False: 'WUS'})
        )
    plot_df = plot_df.dropna(subset=['correlation', 'Experiment', 'region'])
    plot_df['Experiment'] = plot_df['Experiment'].astype(str)

    pairs_in_data = [p for p in pair_selection if p in plot_df['QSWE_pair'].values]
    if not pairs_in_data:
        return
    plot_df['QSWE_pair'] = pd.Categorical(plot_df['QSWE_pair'], categories=pairs_in_data, ordered=True)

    hue_order_global = plot_df['Experiment'].drop_duplicates().tolist()

    dischma_exps = (
        plot_df.loc[plot_df['region'] == 'Dischma', 'Experiment']
        .drop_duplicates().tolist()
    )
    wus_exps = (
        plot_df.loc[plot_df['region'] == 'WUS', 'Experiment']
        .drop_duplicates().tolist()
    )

    dm_colors = sns.color_palette(dischma_palette, n_colors=max(1, len(dischma_exps)))
    if dischma_palette == 'tab20b' and len(dischma_exps) > 0:
        dm_colors = dm_colors[::-1]
    w_colors = _palette_colors(wus_palette_eff, max(1, len(wus_exps)))
    color_map = {}
    color_map.update({e: dm_colors[i] for i, e in enumerate(dischma_exps)})
    color_map.update({e: w_colors[i] for i, e in enumerate(wus_exps)})

    region_order = [r for r in ['Dischma', 'WUS'] if r in plot_df['region'].unique()]
    plot_df['region'] = pd.Categorical(plot_df['region'], categories=region_order, ordered=True)

    n_panel = len(pairs_in_data)
    if figsize is None:
        figsize = (5.0, max(1.45, 0.52 * n_panel * len(region_order)))

    if n_panel == 1:
        fig, axes = plt.subplots(figsize=figsize)
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, n_panel, figsize=(figsize[0] * 0.85 * n_panel, figsize[1]), sharey=True)

    for pi, pair in enumerate(pairs_in_data):
        ax = axes[pi]
        sub = plot_df[plot_df['QSWE_pair'] == pair]
        if '__' in pair:
            swe_m, q_m = pair.split('__', 1)
            t = (
                f"{PAPER_METRIC_NAME_MAP.get(swe_m, swe_m)}\n"
                f"{PAPER_METRIC_NAME_MAP.get(q_m, q_m)}"
            )
        else:
            t = PAPER_METRIC_NAME_MAP.get(pair, pair)
        _qswe_pair_across_region_panel(
            ax, sub, t if n_panel > 1 else '',
            color_map=color_map,
            dodge=dodge,
            strip_jitter=strip_jitter,
            strip_alpha=strip_alpha,
            fontsize=fontsize,
            marker_dischma=marker_dischma,
            marker_wus=marker_wus,
            show_ylabel=(pi == 0),
        )
        if pi > 0:
            ax.set_ylabel('')
            ax.tick_params(axis='y', left=False, labelleft=False)

    _add_dischma_wus_figure_legends(
        fig,
        dischma_exps=dischma_exps,
        wus_exps=wus_exps,
        color_map=color_map,
        legend_anchor=(legend_loc[0], legend_loc[1]) if legend_loc is not None else (1.02, 0.5),
        legend_fontsize=legend_fontsize,
        title_fontsize=fontsize,
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    tag = f"{n_panel}pairs_in_{len(hue_order_global)}exps"
    plt.savefig(
        join(SHARED_PLOTS_DIR, f"rho_values_QSWE_across_settings_v2_region_{tag}.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.savefig(
        join(SHARED_PLOTS_DIR, f"rho_values_QSWE_across_settings_v2_region_{tag}.svg"),
        bbox_inches='tight'
    )
    plt.show()


def QSWE_pair_across_settings_v2_dual(
    pairs_selection_df,
    pair_left,
    pair_right,
    palette='combined',
    figsize=None,
    dodge=0,
    fontsize=12,
    legend_loc=(1.11, 0.3),
    legend_fontsize=12,
    *,
    dischma_palette='tab20b',
    wus_palette=None,
    strip_jitter=False,
    strip_alpha=0.28,
    marker_dischma='D',
    marker_wus='D',
):
    """
    Two side-by-side region panels (**y** = Dischma / WUS), one per ``pair_left`` and ``pair_right``.

    Shares one color map; two stacked catchment legends (Dischma then WUS): 4 cols for Dischma, 2 for WUS by default.
    Default mean markers are both diamonds (``marker_dischma`` / ``marker_wus``); pass e.g.
    ``marker_wus='s'`` to distinguish regions.

    WUS colors default to ``'combined'`` (colorblind + bright + Set2 + Set3 + tab20).
    """
    wus_palette_eff = palette if wus_palette is None else wus_palette
    pair_keys = [pair_left, pair_right]
    plot_df = pairs_selection_df[pairs_selection_df['QSWE_pair'].isin(pair_keys)].copy()
    if plot_df.empty:
        return

    if 'region' not in plot_df.columns:
        plot_df['region'] = (
            plot_df['BASIN'].astype(str).str.contains(r'\d').map({True: 'Dischma', False: 'WUS'})
        )
    plot_df = plot_df.dropna(subset=['correlation', 'Experiment', 'region'])
    plot_df['Experiment'] = plot_df['Experiment'].astype(str)

    present_pairs = set(plot_df['QSWE_pair'].astype(str))
    if pair_left not in present_pairs or pair_right not in present_pairs:
        return

    plot_df['QSWE_pair'] = pd.Categorical(
        plot_df['QSWE_pair'].astype(str),
        categories=[pair_left, pair_right],
        ordered=True,
    )
    hue_order_global = plot_df['Experiment'].drop_duplicates().tolist()

    dischma_exps = (
        plot_df.loc[plot_df['region'] == 'Dischma', 'Experiment']
        .drop_duplicates().tolist()
    )
    wus_exps = (
        plot_df.loc[plot_df['region'] == 'WUS', 'Experiment']
        .drop_duplicates().tolist()
    )

    dm_colors = sns.color_palette(dischma_palette, n_colors=max(1, len(dischma_exps)))
    if dischma_palette == 'tab20b' and len(dischma_exps) > 0:
        dm_colors = dm_colors[::-1]
    w_colors = _palette_colors(wus_palette_eff, max(1, len(wus_exps)))
    color_map = {}
    color_map.update({e: dm_colors[i] for i, e in enumerate(dischma_exps)})
    color_map.update({e: w_colors[i] for i, e in enumerate(wus_exps)})

    region_order = [r for r in ['Dischma', 'WUS'] if r in plot_df['region'].unique()]
    plot_df['region'] = pd.Categorical(plot_df['region'], categories=region_order, ordered=True)

    if figsize is None:
        figsize = (9.2, max(1.45, 0.52 * len(region_order)))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    axes_flat = axes.flatten()

    for pi, pair in enumerate(pair_keys):
        ax = axes_flat[pi]
        sub = plot_df[plot_df['QSWE_pair'] == pair]
        if sub.empty:
            continue
        if '__' in pair:
            swe_m, q_m = pair.split('__', 1)
            t = (
                f"{PAPER_METRIC_NAME_MAP.get(swe_m, swe_m)}/{PAPER_METRIC_NAME_MAP.get(q_m, q_m)}"
            )
        else:
            t = PAPER_METRIC_NAME_MAP.get(pair, pair)
        _qswe_pair_across_region_panel(
            ax,
            sub,
            t,
            color_map=color_map,
            dodge=dodge,
            strip_jitter=strip_jitter,
            strip_alpha=strip_alpha,
            fontsize=fontsize,
            marker_dischma=marker_dischma,
            marker_wus=marker_wus,
            show_ylabel=(pi == 0),
        )
        if pi > 0:
            ax.set_ylabel('')
            ax.tick_params(axis='y', left=False, labelleft=False)

    _add_dischma_wus_figure_legends(
        fig,
        dischma_exps=dischma_exps,
        wus_exps=wus_exps,
        color_map=color_map,
        legend_anchor=(legend_loc[0], legend_loc[1]) if legend_loc is not None else (1.02, 0.5),
        legend_fontsize=legend_fontsize,
        title_fontsize=fontsize,
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    n_ex = len(hue_order_global)
    plt.savefig(
        join(SHARED_PLOTS_DIR, f"rho_values_QSWE_across_settings_v2_dual_{n_ex}exps.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.savefig(
        join(SHARED_PLOTS_DIR, f"rho_values_QSWE_across_settings_v2_dual_{n_ex}exps.svg"),
        bbox_inches='tight'
    )
    plt.show()


def QSWE_pair_boxenplot_compact(
    pairs_selection_df,
    pair_selection,
    palette='colorblind',
    figsize=None,
    fontsize=10,
    legend_loc=(1.01, 0.5),
    legend_fontsize=9
):
    """Compact summary: one boxenplot per QSWE pair over all rows."""
    if figsize is None:
        figsize = (max(4.5, 0.7 * len(pair_selection)), 2.8)

    # Ensure deterministic order for plotting and legend.
    plot_df = pairs_selection_df.copy()
    plot_df['QSWE_pair'] = pd.Categorical(
        plot_df['QSWE_pair'], categories=pair_selection, ordered=True
    )
    plot_df = plot_df.sort_values('QSWE_pair')

    def _lighten_color(color, amount):
        base_rgb = np.array(mcolors.to_rgb(color))
        white = np.array([1.0, 1.0, 1.0])
        return tuple((1 - amount) * base_rgb + amount * white)

    pair_color_map = {}
    for pair in pair_selection:
        if '__' in pair:
            swe_m, q_m = pair.split('__', 1)
            base_color = PAPER_SWE_BASE_COLORS.get(swe_m, '#4c4c4c')
            lighten = PAPER_Q_LIGHTEN.get(q_m, 0.22)
            pair_color_map[pair] = _lighten_color(base_color, lighten)
        else:
            pair_color_map[pair] = '#808080'

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxenplot(
        data=plot_df,
        y='QSWE_pair',
        x='correlation',
        hue='QSWE_pair',
        palette=pair_color_map,
        dodge=False,
        linewidth=0.8,
        k_depth='proportion',
        ax=ax
    )
    auto_legend = ax.get_legend()
    if auto_legend is not None:
        auto_legend.remove()

    ax.axvline(0, linestyle='dashed', color='black', linewidth=0.9)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlabel(r'$\rho$ [-]', fontsize=fontsize)
    ax.set_ylabel('')
    ax.set_xlim(-0.1, 1)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.tick_params(axis='x', labelsize=fontsize)
    # ax.set_title('QSWE Pair Correlation Distributions', fontsize=fontsize + 1)

    clean_labels = []
    bold_flags = []
    for pair in pair_selection:
        lbl = pair
        label = ""
        bold = False
        if '__' in lbl:
            swe_m, q_m = lbl.split('__', 1)
            if lbl == 'SWE_melt_NSE__NSE_meltseason2':
                bold = True
            elif lbl == 'melt_sum_APE__Qmean_meltseason2_APE':
                bold = True
            suffix = ''
            label = (
                f"{PAPER_METRIC_NAME_MAP.get(swe_m, swe_m)}/"
                f"{PAPER_METRIC_NAME_MAP.get(q_m, q_m)} {suffix}"
            )
        else:
            label = PAPER_METRIC_NAME_MAP.get(lbl, lbl)
        clean_labels.append(label)
        bold_flags.append(bold)

    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(ax.get_yticks())
    ax_right.set_yticklabels(clean_labels, fontsize=legend_fontsize)
    for tick_lbl, want_bold in zip(ax_right.get_yticklabels(), bold_flags):
        if want_bold:
            tick_lbl.set_fontweight('bold')
    ax_right.tick_params(axis='y', right=True, labelright=True, length=0, pad=4)
    ax_right.set_ylabel('')
    ax_right.grid(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.savefig(join(SHARED_PLOTS_DIR,
                f"rho_boxen_QSWE_pair_selection_{len(pair_selection)}.png"),
                dpi=300, bbox_inches='tight')
    plt.savefig(join(SHARED_PLOTS_DIR,
                f"rho_boxen_QSWE_pair_selection_{len(pair_selection)}.svg"),
                bbox_inches='tight')
    plt.show()


def QSWE_pair_reduced_horizontal_panels(
    pairs_selection_df,
    pair_selection,
    palette='combined',
    figsize=None,
    fontsize=12,
    dodge=0.75,
    basin_order=None
):
    """Plot reduced QSWE pairs by region with region-specific mean markers."""
    plot_df = pairs_selection_df[pairs_selection_df['QSWE_pair'].isin(pair_selection)].copy()
    if plot_df.empty:
        return

    # Dischma variants include digits in the basin name; WUS catchments do not.
    plot_df['region'] = (
        plot_df['BASIN'].astype(str).str.contains(r'\d').map({True: 'Dischma', False: 'WUS'})
    )
    region_order = [r for r in ['Dischma', 'WUS'] if r in plot_df['region'].unique()]
    plot_df['region'] = pd.Categorical(plot_df['region'], categories=region_order, ordered=True)
    plot_df['QSWE_pair'] = pd.Categorical(plot_df['QSWE_pair'], categories=pair_selection, ordered=True)
    plot_df = plot_df.sort_values(['QSWE_pair', 'region', 'Experiment'])

    plot_df = plot_df.dropna(subset=['correlation', 'Experiment', 'region'])
    # Match QSWE_pair_across_settings behavior: keep first-appearance order.
    hue_order = plot_df['Experiment'].astype(str).drop_duplicates().tolist()
    if not hue_order or not region_order:
        return
    # Split palette by region: Dischma in tab20b, WUS in combined discrete palettes.
    dischma_exps = (
        plot_df.loc[plot_df['region'] == 'Dischma', 'Experiment']
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    wus_exps = (
        plot_df.loc[plot_df['region'] == 'WUS', 'Experiment']
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    dischma_colors = sns.color_palette('tab20b', n_colors=max(1, len(dischma_exps)))
    wus_colors = _palette_colors(palette, max(1, len(wus_exps)))
    color_map = {}
    color_map.update({exp: color for exp, color in zip(dischma_exps, dischma_colors)})
    color_map.update({exp: color for exp, color in zip(wus_exps, wus_colors)})

    n_pairs = len(pair_selection)
    if figsize is None:
        # Taller and narrower by default for readability with side-by-side panels.
        figsize = (2.8 * n_pairs, 6.8)

    fig, axes = plt.subplots(1, n_pairs, figsize=figsize, sharey=True)
    if n_pairs == 1:
        axes = [axes]

    for idx, pair in enumerate(pair_selection):
        ax = axes[idx]
        pair_df = plot_df[plot_df['QSWE_pair'] == pair].copy()
        pair_df = pair_df.dropna(subset=['correlation', 'Experiment', 'region'])
        if pair_df.empty:
            continue
        pair_df['Experiment'] = pair_df['Experiment'].astype(str)

        sns.stripplot(
            data=pair_df,
            y='region',
            x='correlation',
            hue='Experiment',
            hue_order=hue_order,
            orient='h',
            dodge=False,
            jitter=0.17,
            palette=color_map,
            alpha=0.25,
            size=3,
            ax=ax,
            legend=False
        )

        mean_df = (
            pair_df.groupby(['region', 'Experiment'], as_index=False)['correlation']
            .mean()
        )
        mean_df = mean_df.dropna(subset=['correlation'])
        y_map = {region: float(i) for i, region in enumerate(region_order)}
        offset_values = np.linspace(-0.23, 0.23, len(hue_order)) if len(hue_order) > 1 else np.array([0.0])
        offset_map = dict(zip(hue_order, offset_values))
        marker_map = {'Dischma': 'D', 'WUS': 's'}

        for _, row in mean_df.iterrows():
            ax.scatter(
                row['correlation'],
                y_map[row['region']] + offset_map[row['Experiment']],
                marker=marker_map.get(row['region'], 'o'),
                s=62,
                color=color_map[row['Experiment']],
                edgecolor='black',
                linewidth=0.45,
                zorder=20
            )

        ax.axvline(0, linestyle='dashed', color='black', linewidth=0.9)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(right=1)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.set_xlabel(r'$\rho$ [-]', fontsize=fontsize)

        if '__' in pair:
            swe_m, q_m = pair.split('__', 1)
            pair_title = (
                f"{PAPER_METRIC_NAME_MAP.get(swe_m, swe_m)}\n"
                f"{PAPER_METRIC_NAME_MAP.get(q_m, q_m)}"
            )
        else:
            pair_title = PAPER_METRIC_NAME_MAP.get(pair, pair)
        ax.set_title(pair_title, fontsize=fontsize)

        if idx == 0:
            ax.set_ylabel('Region', fontsize=fontsize)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', left=False, labelleft=False)

    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='',
            markerfacecolor=color_map[exp],
            markeredgecolor='black',
            markeredgewidth=0.45,
            markersize=7,
            label=exp
        )
        for exp in hue_order
    ]
    legend_handles.extend([
        Line2D(
            [0], [0],
            marker='D',
            linestyle='',
            color='black',
            markerfacecolor='white',
            markeredgecolor='black',
            markersize=7,
            label='Dischma mean'
        ),
        Line2D(
            [0], [0],
            marker='s',
            linestyle='',
            color='black',
            markerfacecolor='white',
            markeredgecolor='black',
            markersize=7,
            label='WUS mean'
        ),
    ])
    fig.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        ncols=1,
        fontsize=fontsize - 1,
        title='Catchment / mean marker',
        title_fontsize=fontsize
    )

    plt.tight_layout(rect=[0, 0, 0.84, 1])
    plt.savefig(
        join(SHARED_PLOTS_DIR, f"rho_values_QSWE_reduced_panels_{len(pair_selection)}.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.savefig(
        join(SHARED_PLOTS_DIR, f"rho_values_QSWE_reduced_panels_{len(pair_selection)}.svg"),
        bbox_inches='tight'
    )
    plt.show()
    # plt.show()


def QSWE_pair_reduced_horizontal_panels_v2(
    pairs_selection_df,
    pair_selection,
    palette='combined',
    figsize=None,
    fontsize=12,
    dodge=0.75,
    basin_order=None,
    *,
    dischma_palette='tab20b',
    wus_palette=None,
    legend_ncols=2,
    strip_jitter=0.12,
    mean_y_span=0.22,
):
    """
    Alternative reduced-pair plot using ``sns.stripplot`` with ``y='region'``.

    Raw points are strip-jittered along ``rho`` (`dodge=False` avoids hue spacing gaps).
    Region means: Dischma = diamond, WUS = square; experiments offset slightly on y.

    Dischma colors from ``dischma_palette`` (tab20b reversed like elsewhere in script);
    WUS from ``wus_palette`` or ``palette`` (default ``'combined'``).

    Saves to ``rho_values_QSWE_reduced_panels_v2_*`` so the original panels plot is untouched.
    """
    del dodge, basin_order  # kept for drop-in parity with v1 signature

    wus_palette_eff = palette if wus_palette is None else wus_palette

    plot_df = pairs_selection_df[pairs_selection_df['QSWE_pair'].isin(pair_selection)].copy()
    if plot_df.empty:
        return

    plot_df['region'] = (
        plot_df['BASIN'].astype(str).str.contains(r'\d').map({True: 'Dischma', False: 'WUS'})
    )
    region_order = [r for r in ['Dischma', 'WUS'] if r in plot_df['region'].unique()]

    plot_df = plot_df.dropna(subset=['correlation', 'Experiment', 'region'])
    plot_df['QSWE_pair'] = pd.Categorical(plot_df['QSWE_pair'], categories=pair_selection, ordered=True)
    plot_df = plot_df.sort_values(['QSWE_pair', 'region', 'Experiment'])

    hue_order = plot_df['Experiment'].astype(str).drop_duplicates().tolist()
    if not hue_order or not region_order:
        return

    dischma_exps = (
        plot_df.loc[plot_df['region'] == 'Dischma', 'Experiment']
        .astype(str).drop_duplicates().tolist()
    )
    wus_exps = (
        plot_df.loc[plot_df['region'] == 'WUS', 'Experiment']
        .astype(str).drop_duplicates().tolist()
    )

    dm_colors = sns.color_palette(dischma_palette, n_colors=max(1, len(dischma_exps)))
    if dischma_palette == 'tab20b' and len(dischma_exps) > 0:
        dm_colors = dm_colors[::-1]
    w_colors = _palette_colors(wus_palette_eff, max(1, len(wus_exps)))
    color_map = {}
    color_map.update({e: dm_colors[i] for i, e in enumerate(dischma_exps)})
    color_map.update({e: w_colors[i] for i, e in enumerate(wus_exps)})

    n_exp = len(hue_order)
    if n_exp > 1:
        mean_offsets = np.linspace(-mean_y_span, mean_y_span, n_exp)
    else:
        mean_offsets = np.array([0.0])
    exp_to_mean_dy = dict(zip(hue_order, mean_offsets))

    n_pairs = len(pair_selection)
    if figsize is None:
        figsize = (2.6 * n_pairs, 5.2)

    fig, axes = plt.subplots(1, n_pairs, figsize=figsize, sharey=True)
    if n_pairs == 1:
        axes = [axes]

    for idx, pair in enumerate(pair_selection):
        ax = axes[idx]
        pair_df = plot_df[plot_df['QSWE_pair'] == pair].copy()
        pair_df['Experiment'] = pair_df['Experiment'].astype(str)
        pair_df = pair_df.dropna(subset=['correlation'])
        if pair_df.empty:
            continue
        pair_df['region'] = pd.Categorical(
            pair_df['region'], categories=region_order, ordered=True
        )

        sns.stripplot(
            data=pair_df,
            y='region',
            x='correlation',
            hue='Experiment',
            hue_order=hue_order,
            orient='h',
            dodge=False,
            jitter=strip_jitter,
            palette=color_map,
            alpha=0.28,
            size=3,
            linewidth=0,
            ax=ax,
            legend=False,
        )

        mean_df = (
            pair_df.groupby(['region', 'Experiment'], as_index=False)['correlation']
            .mean()
        )
        mean_df = mean_df.dropna(subset=['correlation'])
        marker_map = {'Dischma': 'D', 'WUS': 's'}
        y_map = {r: float(k) for k, r in enumerate(region_order)}
        for _, row in mean_df.iterrows():
            y0 = y_map[row['region']]
            y = y0 + exp_to_mean_dy[row['Experiment']]
            ax.scatter(
                row['correlation'], y,
                marker=marker_map.get(row['region'], 'o'),
                s=55,
                color=color_map.get(row['Experiment'], '#7f7f7f'),
                edgecolors='black',
                linewidths=0.45,
                zorder=10
            )

        ax.axvline(0, linestyle='dashed', color='black', linewidth=0.9)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(right=1)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.set_xlabel(r'$\rho$ [-]', fontsize=fontsize)

        if '__' in pair:
            swe_m, q_m = pair.split('__', 1)
            pair_title = (
                f"{PAPER_METRIC_NAME_MAP.get(swe_m, swe_m)}\n"
                f"{PAPER_METRIC_NAME_MAP.get(q_m, q_m)}"
            )
        else:
            pair_title = PAPER_METRIC_NAME_MAP.get(pair, pair)
        ax.set_title(pair_title, fontsize=fontsize)

        if idx == 0:
            ax.set_ylabel('Region', fontsize=fontsize)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', left=False, labelleft=False)

    exp_handles = [
        Line2D(
            [0], [0], marker='o', linestyle='',
            markerfacecolor=color_map.get(exp, '#7f7f7f'),
            markeredgecolor='black', markeredgewidth=0.35,
            markersize=6, label=exp
        )
        for exp in hue_order
    ]
    shape_handles = [
        Line2D([0], [0], marker='D', linestyle='', color='black',
               markerfacecolor='white', markeredgecolor='black', markersize=7,
               label='Dischma mean'),
        Line2D([0], [0], marker='s', linestyle='', color='black',
               markerfacecolor='white', markeredgecolor='black', markersize=7,
               label='WUS mean'),
    ]
    fig.legend(
        handles=shape_handles + exp_handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        ncols=legend_ncols,
        fontsize=fontsize - 1,
        title='Mean marker / catchment',
        title_fontsize=fontsize,
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    base = join(SHARED_PLOTS_DIR, f"rho_values_QSWE_reduced_panels_v2_{len(pair_selection)}")
    plt.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base}.svg', bbox_inches='tight')
    plt.show()

#%%
allcorrs['QSWE_pair'] = allcorrs['level_0'] + '__' + allcorrs['level_1']

pair_selection = [    
    #  'SWE_melt_NSE__NSE',
    #  'SWE_melt_NSE__NSE_meltseason_beta',
     'SWE_melt_NSE__NSE_meltseason2',
     'SWE_melt_NSE__KGE_meltseason2',
     'SWE_melt_NSE__Qmean_meltseason2_APE',
    #  'SWE_melt_NSE_weekly__NSE_meltseason2',
#  'SWE_melt_NSE_weekly__NSE_weekly',  
#    'SWE_melt_NSE_weekly__NSE_bias',
#    'melt_sum_APE__NSE_bias',
# 'melt_sum_APE__Qmean_APE',      
# 'melt_sum_APE__Qmean_meltseason_APE',  
# 'melt_sum_APE__Qmean_meltseason_beta_APE',
#   'melt_sum_APE__Qamp_APE',
    # 'melt_sum_APE__KGE_meltseason',
    'melt_sum_APE__NSE_meltseason2',    
     'melt_sum_APE__KGE_meltseason2',
'melt_sum_APE__Qmean_meltseason2_APE',
    #  'melt_sum_APE__KGE_meltseason2_r',
    #  'melt_sum_APE__KGE_meltseason2_alpha',
    #  'melt_sum_APE__KGE_meltseason2_beta',

    # 'melt_sum_APE__NSE_weekly',  
    # 'melt_sum_grid_MAPE__KGE_meltseason',
                    # 'melt_sum_grid_MAPE__Qmean_meltseason_APE',

                    # 'SWE_NSE__KGE_BaseFlow',
                    # 'SWE_NSE__NSE',
                    'SWE_NSE__NSE_meltseason2', 
                    'SWE_NSE__KGE_meltseason2',
                    'SWE_NSE__Qmean_meltseason2_APE',
                    # 'SWE_NSE__KGE',
                    # 'SWE_NSE__Qmean_meltseason_APE',
                        # 'SWE_max_grid_MAPE__KGE_meltseason'   ,
                    # 'SWE_7daymelt_ME__Qmax_ME',
                    # 't_SWE_max_ME__t_melt_onset_ME',
                    # 't_SWE_max_grid_ME__NSE_meltseason'
                # "melt_sum_grid_MAPE__Qmean_meltseason2_APE", #30-7
                    # 'melt_sum_grid_MAPE__Qamp_APE',
                    # 't_SWE_end_grid_ME__t_Qstart_ME',
                    # 't_SWE_end_grid_ME__Q5_APE',
                    # 't_melt_hfd_ME__t_hfd_meltseason_ME',
                    # 't_melt_hfd_ME__t_hfd_meltseason2_ME', #30-7
                    # 't_melt_hfd_ME__t_hfd_meltseason_beta_ME',
                    # 't_SWE_end_grid_ME__logNSE_meltseason',
                    # 't_SWE_end_ME__t_hfd_ME',
                    # 't_SWE_end_ME__logNSE_meltseason',
                    # 't_SWE_end_ME__KGE_meltseason_r',
                    # 't_SWE_end_ME__t_hfd_meltseason_ME',
                    # 't_SWE_end_ME__t_hfd_meltseason2_ME', #30-7
                    # 't_SWE_end_ME__t_hfd_meltseason_beta_ME',
                    # 'SWE_SWS_APE__NSE_weekly',
                    # "SWE_SWS_APE__KGE_meltseason2", #30-7
                    # 't_Qstart'
]
MP_label_dic = {
        # 'SWE_melt_NSE__NSE': 'Melt-NSE/Q-NSE',
    # 'SWE_melt_NSE__NSE_meltseason_beta': 'Melt-NSE/Q-NSE',
    'SWE_melt_NSE__NSE_meltseason2': 'Melt-NSE/Q-NSE',
    'SWE_melt_NSE_weekly__NSE_meltseason2': 'Melt-NSEweekly/Q-NSE',
    # 'melt_sum_APE__Qmean_meltseason_APE': 'Melt-bias/Q-bias',
    'melt_sum_APE__Qmean_meltseason2_APE': 'Melt-bias/Q-bias',
    # 't_melt_hfd_ME__t_hfd_meltseason_ME': 'Melt-HFD/Q-HFD',
    't_melt_hfd_ME__t_hfd_meltseason2_ME': 'Melt-HFD/Q-HFD',
    # 'melt_sum_APE__Qmean_meltseason_beta_APE': 'Melt-bias/Q-bias',
    # 't_melt_hfd_ME__t_hfd_meltseason_beta_ME': 'Melt-bias/Q-bias',
 't_SWE_max_ME__t_melt_onset_ME': 'Peak SWE date error/Meltseason start error',
 'SWE_NSE__NSE_meltseason2': 'SWE-NSE/Q-NSE',
 'SWE_NSE__KGE_meltseason2': 'SWE-NSE/Q-KGE',
 'melt_sum_APE__KGE_meltseason2': 'Melt-bias/Q-KGE',
"melt_sum_grid_MAPE__Qmean_meltseason2_APE": "Melt-spatial bias/Q-bias",
"t_SWE_end_grid_ME__Q5_APE": "Melt-out timing error/Q5-bias",
"t_SWE_end_grid_ME__logNSE_meltseason": "Melt-out timing error/Q-logNSE",
"t_SWE_end_ME__t_hfd_meltseason2_ME": "Melt-out timing error/Q-HFD",
"SWE_SWS_APE__KGE_meltseason2": "SWS-bias/Q-KGE",
}
 


# MP_label_dic = {
#         'SWE_melt_NSE__NSE': r'$\rho_\mathrm{daily}$',
#     'SWE_melt_NSE__NSE_meltseason_beta': r'$\rho_\mathrm{daily,\beta}$',
#     'SWE_melt_NSE__NSE_meltseason2': r'$\rho_\mathrm{daily}$',
#     'SWE_melt_NSE_weekly__NSE_meltseason2': r'$\rho_\mathrm{daily}$',
#     'melt_sum_APE__Qmean_meltseason_APE': r'$\rho_\mathrm{seasonal}$',
#     'melt_sum_APE__Qmean_meltseason2_APE': r'$\rho_\mathrm{seasonal}$',
#     't_melt_hfd_ME__t_hfd_meltseason_ME': r'$\rho_\mathrm{event}$',
#     't_melt_hfd_ME__t_hfd_meltseason2_ME': r'$\rho_\mathrm{event}$',
#     'melt_sum_APE__Qmean_meltseason_beta_APE': r'$\rho_\mathrm{seasonal,\beta}$',
#     't_melt_hfd_ME__t_hfd_meltseason_beta_ME': r'$\rho_\mathrm{event,\beta}$',
#  't_SWE_max_ME__t_melt_onset_ME': r'$\rho_\mathrm{event}$',
#  'SWE_NSE__NSE_meltseason2': r'$\rho_\mathrm{default1}$',
#  'SWE_NSE__KGE_meltseason2': r'$\rho_\mathrm{default2}$',
#  'melt_sum_APE__KGE_meltseason2': r'$\rho_\mathrm{seasonal}$',
# }

# MP_label_dic = {
#         'SWE_melt_NSE__NSE': 'Melt-NSE/Q-NSE',
#     'SWE_melt_NSE__NSE_meltseason_beta': 'Melt-NSE/Q-NSE',
#     'SWE_melt_NSE__NSE_meltseason2': 'Melt-NSE/Q-NSE',
#     'SWE_melt_NSE_weekly__NSE_meltseason2': 'Melt-NSE/Q-NSE',
#     'melt_sum_APE__Qmean_meltseason_APE': 'Melt-bias/Q-bias',
#     'melt_sum_APE__Qmean_meltseason2_APE': 'Melt-bias/Q-bias',
#     't_melt_hfd_ME__t_hfd_meltseason_ME': 'Melt-bias/Q-bias',
#     't_melt_hfd_ME__t_hfd_meltseason2_ME': 'Melt-bias/Q-bias',
#     'melt_sum_APE__Qmean_meltseason_beta_APE': 'Melt-bias/Q-bias',
#     't_melt_hfd_ME__t_hfd_meltseason_beta_ME': 'Melt-bias/Q-bias',
#  't_SWE_max_ME__t_melt_onset_ME': 'Melt-bias/Q-bias',
#  'SWE_NSE__NSE_meltseason2': 'Melt-NSE/Q-NSE',
#  'SWE_NSE__KGE_meltseason2': 'Melt-KGE/Q-KGE',
#  'melt_sum_APE__KGE_meltseason2': 'Melt-bias/Q-bias',
 
# }
all_pairs = allcorrs[allcorrs['QSWE_pair'].isin(pair_selection)].drop(['level_0', 'level_1'], axis=1).reset_index(drop=True)

# Helper function to safely get values with default fallback
def safe_get(func, default=np.nan):
    """Safely execute a function and return default if it fails."""
    try:
        return func()
    except:
        return default

for i in range(len(all_pairs)):
    all_pairs.loc[i,'Uncertainty'] = all_pairs.loc[i,'EXP_ID'][0]
    all_pairs.loc[i,'Catchment_behavior'] = all_pairs.loc[i,'EXP_ID'][1]
    all_pairs.loc[i,'Rainfall regime'] = all_pairs.loc[i,'EXP_ID'][2]
    all_pairs.loc[i,'Param choice'] = all_pairs.loc[i,'EXP_ID'][3]
    
    # Get year and basin for lookups
    year = all_pairs.loc[i, 'Year']
    basin = all_pairs.loc[i, 'BASIN']
    
    # Define lookup mappings: (column_name, lookup_function, default_value)
    lookups = [
        ('k_value', lambda: safeeq.loc[safeeq['short_name'] == basin, 'k'].values[0], 0.03),
        ('rainfall_mixing', lambda: zetas['rainfall_mixing_ratios'].loc[year, basin], np.nan),
        ('et_mixing', lambda: zetas['et_mixing_ratios'].loc[year, basin], np.nan),
        ('sf_ratio', lambda: zetas['sf_ratio'].loc[year, basin], np.nan),
        ('spring_rf_fraction', lambda: zetas['spring_rf_fraction'].loc[year, basin], np.nan),
        ('delay', lambda: zetas['delays'].loc[year, basin], np.nan),
        ('area_km2', lambda: SwissStation(basin).area, np.nan),
        ('S_fraction', lambda: zetas['S_fraction'].loc[year, basin], np.nan),
        ('S_fraction_spring', lambda: zetas['S_fraction_spring'].loc[year, basin], np.nan),
        ('S_dynamic_fraction', lambda: zetas['S_dynamic_fraction'].loc[year, basin], np.nan),
        ('S_dynamic_fraction_spring', lambda: zetas['S_dynamic_fraction_spring'].loc[year, basin], np.nan),
        ('temp_storage', lambda: zetas['temp_storage'].loc[year, basin], np.nan),
        ('temp_storage_spring', lambda: zetas['temp_storage_spring'].loc[year, basin], np.nan),
        ('temp_storage_spring_2', lambda: zetas['temp_storage_spring_2'].loc[year, basin], np.nan),
        ('temp_storage_spring_beta', lambda: zetas['temp_storage_spring_beta'].loc[year, basin], np.nan),
        ('temp_storage_spring_2_beta', lambda: zetas['temp_storage_spring_2_beta'].loc[year, basin], np.nan),
        ('spring_rf_fraction_beta', lambda: zetas['spring_rf_fraction_beta'].loc[year, basin], np.nan),
        ('spring_et_fraction', lambda: zetas['spring_et_fraction'].loc[year, basin], np.nan),
        ('spring_et_fraction_beta', lambda: zetas['spring_et_fraction_beta'].loc[year, basin], np.nan),
        ('DJF_pet', lambda: zetas['DJF_pet'].loc[year, basin], np.nan),
        ('MAM_pet', lambda: zetas['MAM_pet'].loc[year, basin], np.nan),
        ('spring_pet', lambda: zetas['spring_pet'].loc[year, basin], np.nan),
        ('annual_q_sum', lambda: zetas['annual_q_sum'].loc[year, basin], np.nan),
        ('mean_slope_dem', lambda: zetas['mean_slope_dem'].loc[year, basin], np.nan),
        ('slope_proxy_relief', lambda: zetas['slope_proxy_relief'].loc[year, basin], np.nan),
        ('mean_elevation_dem', lambda: zetas['mean_elevation_dem'].loc[year, basin], np.nan),
        ('DJF_rainfall_fraction', lambda: zetas['DJF_rainfall_fraction'].loc[year, basin], np.nan),
        ('MAM_rainfall_fraction', lambda: zetas['MAM_rainfall_fraction'].loc[year, basin], np.nan),
        ('antecedent_Q', lambda: zetas['antecedent_Q'].loc[year, basin], np.nan),
        ('meltseason_length', lambda: zetas['meltseason_length'].loc[year, basin], np.nan),
        ('melt_hfi', lambda: zetas['melt_hfi'].loc[year, basin], np.nan),
        ('ERRA_center_of_mass', lambda: zetas['ERRA_center_of_mass'].loc[year, basin], np.nan),
    ]
    
    # Apply all lookups
    for col_name, lookup_func, default_val in lookups:
        all_pairs.loc[i, col_name] = safe_get(lookup_func, default_val)

# Add experiment names and make categorical variables
all_pairs['Experiment'] = all_pairs['EXP_ID'].map(EXPERIMENT_CONFIG)
all_pairs['QSWE_pair'] = pd.Categorical(all_pairs['QSWE_pair'], categories=pair_selection, ordered=True) 
all_pairs['Experiment'] = pd.Categorical(all_pairs['Experiment'], categories=EXPERIMENT_NAMES, ordered=True)
all_pairs['region'] = all_pairs['BASIN'].str.contains(r'\d').map({True: 'Dischma', False: 'WUS'})



all_pairs.sort_values(['QSWE_pair', 'Experiment'], inplace=True)

# QSWE_pair_across_settings(all_pairs,pair_selection)
QSWE_pair_boxenplot_compact(all_pairs, pair_selection,figsize = (5,2.5))
# Original (uniform WUS marker): QSWE_pair_across_settings_v2_dual(...)
QSWE_pair_across_settings_v2_dual_v3(
    all_pairs,
    'SWE_melt_NSE__NSE_meltseason2',
    'melt_sum_APE__Qmean_meltseason2_APE',
    figsize=(6, max(5, 0.52 * 2)),
    dodge=0.75,
    legend_loc = (0.8,1.05),
    legend_fontsize = 10
)
# Single-pair variant: QSWE_pair_across_settings_v2(df, pair_selection, …, marker_dischma='D', marker_wus='s')

#Print the pairs sorted on their overall average correlation
print(all_pairs.groupby('QSWE_pair')['correlation'].mean().sort_values(ascending=False))

#%% compare results of the different meltseason defintiions over Q_mean_meltseason and t_hfd_meltseason
ms1 = all_pairs[all_pairs['QSWE_pair'] == 'melt_sum_APE__Qmean_meltseason_APE']
ms2 = all_pairs[all_pairs['QSWE_pair'] == 'melt_sum_APE__Qmean_meltseason2_APE']
ms3 = all_pairs[all_pairs['QSWE_pair'] == 'melt_sum_APE__Qmean_meltseason_beta_APE']
print('meltseason1 bias correatlion =',ms1['correlation'].mean())
print('meltseason2 bias correatlion =', ms2['correlation'].mean())
print('meltseason beta bias correatlion =', ms3['correlation'].mean())

t1 = all_pairs[all_pairs['QSWE_pair'] == 't_melt_hfd_ME__t_hfd_meltseason_ME']
t2 = all_pairs[all_pairs['QSWE_pair'] == 't_melt_hfd_ME__t_hfd_meltseason2_ME']
t3 = all_pairs[all_pairs['QSWE_pair'] == 't_melt_hfd_ME__t_hfd_meltseason_beta_ME']
print('meltseason 1 QCOM correatlion =',t1['correlation'].mean())
print('meltseason 2 QCOM correatlion =', t2['correlation'].mean())
print('meltseason beta QCOM correatlion =', t3['correlation'].mean())


#compare NSE and NSE_meltseason
# nse1 = all_pairs[all_pairs['QSWE_pair'] == 'SWE_melt_NSE__NSE']
# nse2 = all_pairs[all_pairs['QSWE_pair'] == 'SWE_melt_NSE__NSE_meltseason_beta']
# print('NSE correatlion =',nse1['correlation'].mean())
# print('NSE beta correatlion =', nse2['correlation'].mean())

#%% REduced pairs of QSWE metrics
reduced_pair_selection = [
    # 'SWE_melt_NSE_weekly__NSE_meltseason2',
'SWE_melt_NSE__NSE_meltseason2',
'melt_sum_APE__Qmean_meltseason2_APE',
# 't_melt_hfd_ME__t_hfd_meltseason2_ME',
# 't_SWE_max_ME__t_melt_onset_ME',
# 't_SWE_end_ME__t_hfd_meltseason_ME'
# 'melt_sum_APE__KGE_meltseason2',
# 'SWE_NSE__NSE_meltseason2',

]

all_pairs_reduced = all_pairs[all_pairs['QSWE_pair'].isin(reduced_pair_selection)].reset_index(drop=True)

# Define experiment groups to plot
# experiment_groups = {
#     'Catchment_rainfall': ['Rain-free', 'Cracked concrete', 'Parking lot', 'Sponge', 'Default', 'Double rain'],
#     'Uncertainty': ['Default', 'Spatial', 'Temporal'], 
#     'Misc': ['Default', 'Tcorr', 'Runoff parameters','correction'],
#     'Minimum':['Default','Rain-free','Double rain']
# }
# palette1 = sns.color_palette('tab20c',n_colors=13)
# palette1 = sns.color_palette('tab20b',n_colors=8) + [sns.color_palette('tab20b',n_colors=18)[16]] + sns.color_palette('tab20b',n_colors=18)[8:17] + [sns.color_palette('tab20b',n_colors=18)[17]]
# palette1 = sns.color_palette('tab20c',n_colors=12)

# Plot Catchment_rainfall experiments
all_pairs_reduced['Experiment'] = pd.Categorical(
    all_pairs_reduced['Experiment'], 
    categories=all_pairs_reduced['Experiment'].astype('str').unique(),
    ordered=True
)
all_pairs_reduced['QSWE_pair'] = pd.Categorical(all_pairs_reduced['QSWE_pair'], categories=reduced_pair_selection, ordered=True)
# palette1 = sns.color_palette('tab20b',n_colors=8) + [sns.color_palette('tab20b',n_colors=17)[16]] + sns.color_palette('tab20b',n_colors=17)[8:17]
#make a palette of Reds that scales with the values of k in safeeq per basin
# palette1 = sns.color_palette('Reds',n_colors=len(safeeq))
if np.all(np.array(basin_list) =='Dischma'):
    palette1 = sns.color_palette('tab20b',n_colors=16)
    palette1 = palette1[::-1]
else:
    palette1= 'colorblind'

sorted_on_daily = all_pairs_reduced[all_pairs_reduced['QSWE_pair']=='SWE_melt_NSE__NSE_meltseason2'].groupby('BASIN')['correlation'].mean().sort_values(ascending=False)
print(sorted_on_daily)

# QSWE_pair_across_settings(all_pairs_reduced, 
# reduced_pair_selection, palette=palette_x,figsize = (5,10),fontsize = 18,
# legend_loc = (1.05,0.1),legend_fontsize = 14,
# dodge = 0.788)
# QSWE_pair_reduced_horizontal_panels(
#     all_pairs_reduced,
#     reduced_pair_selection,
#     palette='colorblind',
#     figsize=None,
#     fontsize=12,
#     dodge=0.75,
#     basin_order=None
# )
# QSWE_pair_reduced_horizontal_panels_v2(
#     all_pairs_reduced, reduced_pair_selection, palette='colorblind', fontsize=12)

#%%
from matplotlib.lines import Line2D

def plot_corr_grid(df, pairs, plot_vars, mp_labels=None,
                   palette='colorblind', figsize=None, savepath=None,
                   vmin=None, vmax=None):
    """
    df: dataframe containing columns ['QSWE_pair','correlation','BASIN', <yvars>...]
    pairs: list of length 3 (columns) with QSWE_pair values (reduced_pair_selection)
    plot_vars: list of tuples [(ycol, ylabel), ...] order = rows
    mp_labels: list of column titles (same length as pairs)
    """
    nrows = len(plot_vars)
    ncols = len(pairs)
    if figsize is None:
        figsize = (3 * ncols, 2.6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey='row',
                             sharex = True)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # axes = np.atleast_2d(axes).reshape(nrows, ncols)


    # palette mapping for BASIN
    basins = sorted(df['BASIN'].unique())
    colors = dict(zip(basins, sns.color_palette(palette, n_colors=len(basins))))

    for col, pair in enumerate(pairs):
        pairs_selection_df = df[df['QSWE_pair'] == pair]
        for row, (ycol, ylabel) in enumerate(plot_vars):
            ax = axes[row, col]

            if pairs_selection_df.empty or ycol not in pairs_selection_df.columns:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel(ylabel if col == 0 else '')
                # if row == nrows - 1:
                #     ax.set_xlabel(r'$\rho [-])
                # continue

            # Draw the scatter; set legend only once (top-right position)
            sns.scatterplot(
                data=pairs_selection_df, x='correlation', y=ycol, hue='BASIN',
                palette=colors, alpha=0.6, ax=ax, legend=False,
                 linewidth=0.4, edgecolor='black'
            )

            # median overlay (group by ycol and show median correlation)
            # try:
            #     medians = pairs_selection_df.groupby([ycol])['correlation'].median().reset_index()
            #     sns.scatterplot(data=medians, x='correlation', y=ycol,
            #                     color='black', marker='x', s=50, ax=ax, legend=False)
            # except Exception:
            #     pass

            ax.grid(alpha=0.3)
            ax.axvline(0, color='black', linestyle='--')
            ax.set_xlim(-0.3,1)
            if col == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel('')
            if row == nrows - 1:
                ax.set_xlabel(r'$\rho$ [-]')

            if row == 0:
                title = (mp_labels[col] if mp_labels is not None else pair)
                ax.set_title(title, fontsize=10)

    # Create a single legend for BASIN on the right
    proxy_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[b], markersize=7,
               markeredgecolor='black', label=b) for b in basins
    ]
    fig.legend(handles=proxy_handles, 
               loc='center right', bbox_to_anchor=(1.12, 0.5),
                 title='Catchment')
    # plt.tight_layout(rect=[0, 0, 0.90, 1.0])

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()


# Example usage: choose rows dynamically
plot_vars = [
    # ('k_value', r'$\zeta_{Safeeq_{k}}$ [-]'),
    # ('DJF_rainfall_fraction', r'DJF rainfall fraction [-]'),
    # ('MAM_rainfall_fraction', r'MAM rainfall fraction [-]'),
    # ('DJF_temp', r'DJF temperature [-]'),
    # ('MAM_temp', r'MAM temperature [-]'),
    ('DJF_pet', r'DJF PET [-]'),
    ('MAM_pet', r'MAM PET [-]'),
    ('spring_pet', r'Meltseason PET [-]'),
    ('rainfall_mixing', r'$\zeta_{rain}$ [-]'),
    # ('et_mixing', r'$\zeta_{ET}$ [-]'),
    # ('S_fraction', r'Characteristic catchment storage [-]'),
        ('S_fraction', r'Storage turnover time [-]'),
    ('S_dynamic_fraction', r'Dynamic storage turnover time [-]'),
    ('et_mixing', r'Meltseason ET fraction [-]'),
    ('S_fraction_spring', r'Characteristic spring storage [-]'),
    ('S_dynamic_fraction_spring', r'Spring dynamic storage turnover time (max-min) [days]'),
    ('delay', r'$\zeta_{delay}$ [-]'),
    ('temp_storage', r'Temp storage '),
    ('temp_storage_spring', r'Characteristic meltseason response time [days]'),
    ('temp_storage_spring_2', 'S2 \n'+ r'Characteristic meltseason response time [days]'),
    ('temp_storage_spring_beta', r'Characteristic meltseason response time $\beta$ [days]'),
    ('temp_storage_spring_2_beta', r'Characteristic meltseason response time $\beta$ [days]'),
    # ('spring_rf_fraction', r'Meltseason rainfall fraction [-]'),# of total water input [-]'),
    ('spring_rf_fraction', 'Meltseason rainfall fraction [-]'),
    ('spring_rf_fraction_beta', r'Meltseason rainfall fraction $\beta$ [-]'),
    ('area_km2', r'Catchment area [km$^2$]'),
    ('annual_q_sum', r'Annual Q sum [mm]'),
    # ('mean_slope_dem', r'Mean catchment slope from DEM [$^\circ$]'),
    ('mean_slope_dem', r'Slope [-]'),
    ('slope_proxy_relief', r'Slope [-]'),
    ('mean_elevation_dem', r'Mean catchment elevation [m]'),
    ('spring_et_fraction', r'Meltseason ET fraction [-]'),
    ('antecedent_Q', r'Antecedent Q [-]'),
    ('meltseason_length', r'Meltseason length [days]'),
    ('melt_hfi', r'Melt half-flow interval [days]'),
    ('ERRA_center_of_mass', r'ERRA center of mass [days]'),
]
plot_vars_dic= dict(plot_vars)
MP_labels = [r'$\mathrm{MP}_\mathrm{fit}$',r'$\mathrm{MP}_\mathrm{volume}$',r'$\mathrm{MP}_\mathrm{timing}$']
# MP_label_dic = {
#     'SWE_melt_NSE__NSE': r'$\mathrm{MP}_\mathrm{fit}$',
#     'SWE_melt_NSE__NSE_meltseason_beta': r'$\mathrm{MP}_\mathrm{fit,\beta}$',
#     'SWE_melt_NSE__NSE_meltseason2': r'$\mathrm{MP}_\mathrm{fit}$',
#     'SWE_melt_NSE_weekly__NSE_meltseason2': r'$\mathrm{MP}_\mathrm{fit}$',
#     'melt_sum_APE__Qmean_meltseason_APE': r'$\mathrm{MP}_\mathrm{volume}$',
#     'melt_sum_APE__Qmean_meltseason2_APE': r'$\mathrm{MP}_\mathrm{volume}$',
#     't_melt_hfd_ME__t_hfd_meltseason_ME': r'$\mathrm{MP}_\mathrm{timing}$',
#     't_melt_hfd_ME__t_hfd_meltseason2_ME': r'$\mathrm{MP}_\mathrm{timing}$',
#     'melt_sum_APE__Qmean_meltseason_beta_APE': r'$\mathrm{MP}_\mathrm{volume,\beta}$',
#     't_melt_hfd_ME__t_hfd_meltseason_beta_ME': r'$\mathrm{MP}_\mathrm{timing,\beta}$',
#  't_SWE_max_ME__t_melt_onset_ME': r'$\mathrm{MP}_\mathrm{timing}$',
#  'SWE_NSE__NSE_meltseason2': r'$\mathrm{MP}_\mathrm{Mainstream}$',
# }

# plot_corr_grid(
#     all_pairs_reduced,
#     reduced_pair_selection,
#     plot_vars,
#     mp_labels=MP_labels,
#     palette='colorblind',
#     figsize=(2*len(reduced_pair_selection), 2*len(plot_vars)),
#     savepath=join(SHARED_PLOTS_DIR, f"Correlation_grid_all_pairs.png")
# )

#%% Bubble plot 
#first average over all years 
# delay_zeta = 'delay'
# delay_zeta = 'temp_storage_spring'
# delay_zeta = 'temp_storage_spring_2'
delay_zeta = 'S_dynamic_fraction'
meteo_zeta = 'spring_rf_fraction'
# meteo_zeta = 'et_mixing'

# Assign wus_region to the original data first (before grouping)
all_pairs_reduced['wus_region'] = all_pairs_reduced['EXP_ID'].map(EXP_WUS_REGION).fillna('Unmapped WUS')

# Group and aggregate all relevant columns, including wus_region, using a representative value (e.g., first)
all_pairs_reduced_avg = all_pairs_reduced.groupby(
    ['QSWE_pair', 'BASIN']
).agg({
    'correlation': 'mean',
    meteo_zeta: 'mean',
    delay_zeta: 'mean',
    'wus_region': 'first'  # This will propagate the correct WUS region for each group
}).reset_index()

# Add a column specifying the region based on whether a basin has a number in its name or not
all_pairs_reduced_avg['region'] = all_pairs_reduced_avg['BASIN'].str.contains(r'\d').map({True: 'Dischma', False: 'WUS'})

# for MP in reduced_pair_selection:
# # MP = reduced_pair_selection[0]
#     all_pairs_reduced_MP = all_pairs_reduced_avg[all_pairs_reduced_avg['QSWE_pair'] == MP]
#     plt.figure(figsize=(10,10))
#     sns.scatterplot(
#         data=all_pairs_reduced_MP,
#         x=delay_zeta,
#         # y='correlation',
#         # size=meteo_zeta,
#         y = meteo_zeta,
#         size ='correlation',
#         sizes = (20,2000),
#         size_norm=(0.8,1.0),
#         hue='BASIN',
#         palette='colorblind',
#         alpha=0.6,
#         edgecolor='black',
#         linewidth=0.4,
#         style = 'Dischma',
#     )
#     plt.legend(loc = (1.05,0.0), title = 'Catchment')
#     plt.title(MP_label_dic[MP])

    #Just make a plot of the points with text annotations
all_pairs_reduced_MP = all_pairs_reduced_avg[all_pairs_reduced_avg['QSWE_pair'] == reduced_pair_selection[0]]
all_pairs_reduced_MP = all_pairs_reduced_MP.copy()

# Toggle colorful mode:
# - Dischma points get basin-specific colors.
# - WUS points get region-specific colors.
bubble_plot_colorful = True

dischma_basins = sorted(
    all_pairs_reduced_MP.loc[all_pairs_reduced_MP['region'] == 'Dischma', 'BASIN'].unique()
)
dischma_palette = sns.color_palette('tab20b', n_colors=max(1, len(dischma_basins)))[::-1]
dischma_color_map = {b: c for b, c in zip(dischma_basins, dischma_palette)}

# Use a dedicated WUS region palette distinct from the Dischma palette.
wus_region_order = ['PNW', 'Sierra', 'Interior', 'Unmapped WUS']
wus_region_colors = {
    'PNW': '#7f7f7f',
    'Sierra': '#7f7f7f',
    'Interior': '#7f7f7f',
    'Unmapped WUS': '#7f7f7f'
}

# WUS regions share one grey color and are differentiated by marker shape.
wus_region_markers = {
    'PNW': 'X',
    'Sierra': '^',
    'Interior': 'P',
    'Unmapped WUS': 'D'
}

plt.figure(figsize=(5,5))
ax = plt.gca()
if bubble_plot_colorful:
    all_pairs_reduced_MP['bubble_color_key'] = np.where(
        all_pairs_reduced_MP['region'] == 'Dischma',
        all_pairs_reduced_MP['BASIN'],
        all_pairs_reduced_MP['wus_region']
    )
    all_pairs_reduced_MP['marker_key'] = np.where(
        all_pairs_reduced_MP['region'] == 'Dischma',
        'Dischma',
        all_pairs_reduced_MP['wus_region']
    )
    hue_order = dischma_basins + [r for r in wus_region_order if r in all_pairs_reduced_MP['wus_region'].unique()]
    bubble_color_map = dict(dischma_color_map)
    bubble_color_map.update(wus_region_colors)
    marker_map = {'Dischma': 'o'}
    marker_map.update(wus_region_markers)
    sns.scatterplot(
        data=all_pairs_reduced_MP,
        x=delay_zeta,
        y=meteo_zeta,
        style='marker_key',
        hue='bubble_color_key',
        hue_order=hue_order,
        markers=marker_map,
        palette=bubble_color_map,
        edgecolor='black',
        linewidth=0.4,
        s=150,
        legend=False
    )
else:
    sns.scatterplot(
        data=all_pairs_reduced_MP,
        x=delay_zeta,
        y=meteo_zeta,
        hue='region',
        edgecolor='black',
        linewidth=0.4,
        marker='X',
        s=150,
        legend=False
    )

# Use adjustText to prevent text overlap
from adjustText import adjust_text

texts = []
for i, row in all_pairs_reduced_MP.iterrows():
    text = ax.text(row[delay_zeta], row[meteo_zeta], row['BASIN'],
                   fontsize=6.5, weight='bold', ha='center', va='center')
    texts.append(text)

# Adjust text positions to avoid overlap
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
avoid_self =True,expand=(1.2, 2))

# plt.legend(loc = (1.05,0.0), title = 'Region')
if bubble_plot_colorful:
    from matplotlib.lines import Line2D

    # Custom legend: one hollow Dischma circle + three colored WUS region crosses.
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor='none',
            markeredgecolor='black',
            markeredgewidth=1.2,
            markersize=8,
            label='Dischma'
        )
    ]

    for wus_region in ['PNW', 'Sierra', 'Interior']:
        legend_handles.append(
            Line2D(
                [0], [0],
                marker=wus_region_markers[wus_region],
                linestyle='None',
                markerfacecolor=wus_region_colors[wus_region],
                markeredgecolor='black',
                markeredgewidth=1.0,
                markersize=8,
                label=f'WUS - {wus_region}'
            )
        )

    ax.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(0.6, 0.855),
        frameon=True,
        title='Catchments'
    )
else:
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title='Region')
plt.xlabel(plot_vars_dic[delay_zeta],fontsize = 12)
plt.ylabel(plot_vars_dic[meteo_zeta],fontsize = 12)
plt.savefig(join(SHARED_PLOTS_DIR, f"Bubble_plot_{meteo_zeta}_{delay_zeta}_{meltseason_test_name}.png"),
                dpi=300, bbox_inches='tight')
plt.savefig(join(SHARED_PLOTS_DIR, 
f"Bubble_plot_{meteo_zeta}_{delay_zeta}_{meltseason_test_name}.svg"),
                dpi=300, bbox_inches='tight')
# plt.title(MP_label_dic[reduced_pair_selection[0]])



#%%
vmin = 0.0
vmax = 1.0
vmean = (vmin + vmax)/2
for MP in reduced_pair_selection:
    all_pairs_reduced_avg = all_pairs_reduced#[all_pairs_reduced['Year']>=2015].copy()

    #add a column specifiying whehter a basin has a number in its name or not
    all_pairs_reduced_avg['region'] = all_pairs_reduced_avg['BASIN'].str.contains(r'\d').map({True: 'Dischma', False: 'WUS'})

    # for MP in reduced_pair_selection:
    # MP = reduced_pair_selection[0]
    all_pairs_reduced_MP = all_pairs_reduced_avg[all_pairs_reduced_avg['QSWE_pair'] == MP]
   

    fig, ax = plt.subplots(figsize=(8,6))
    palette = 'Spectral'

    # Hexbin plot colored by mean correlation in each hex
    hexbin = ax.hexbin(all_pairs_reduced_MP[delay_zeta], 
                    all_pairs_reduced_MP[meteo_zeta],
                    C=all_pairs_reduced_MP['correlation'],
                    gridsize=12, cmap=palette, 
                    vmin=vmin, vmax=vmax,
                    alpha=1, edgecolors='none')

    # Plot the actual points on top
    scatter = sns.scatterplot(
        data=all_pairs_reduced_MP,
        x=delay_zeta,
        y = meteo_zeta,
        hue='correlation',
        s= 80,
        palette=palette,
        alpha=1,
        edgecolor='black',
        linewidth=0.4,
        style = 'region',
        hue_norm = (vmin,1.0),
        ax=ax,
        legend=False
    )

    # Create colorbar with fixed intervals
    sm = plt.cm.ScalarMappable(cmap=palette, norm=plt.Normalize(vmin=vmin, vmax=1.0))
    sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, ticks=[0, 0.25, 0.5, 0.75, 1.0],
    #       location ='right', shrink = 0.5, anchor = (-1.45,0.9))
    cbar = plt.colorbar(sm, ax=ax, ticks=[vmin, vmean, vmax],
          location ='right', shrink = 0.5, anchor = (-1.45,0.9))
    cbar.set_label(r'$\rho$ [-]', rotation=270, labelpad=20,fontsize = 14)
    
    plt.title(MP_label_dic[MP],fontsize = 16)
    ax.set_xlabel(plot_vars_dic[delay_zeta],fontsize = 14)
    ax.set_ylabel(plot_vars_dic[meteo_zeta],fontsize = 14)
    plt.savefig(join(SHARED_PLOTS_DIR, f"Hexbin_plot_{meteo_zeta}_{delay_zeta}_{MP}_{meltseason_test_name}.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

# Duplicate of the hexbin plot with marker styles split by WUS region.
for MP in reduced_pair_selection:
    all_pairs_reduced_avg = all_pairs_reduced#[all_pairs_reduced['Year']>=2015].copy()

    # Add region columns for marker assignment.
    all_pairs_reduced_avg['region'] = all_pairs_reduced_avg['BASIN'].str.contains(r'\d').map({True: 'Dischma', False: 'WUS'})
    all_pairs_reduced_avg['wus_region'] = all_pairs_reduced_avg['EXP_ID'].map(EXP_WUS_REGION).fillna('Unmapped WUS')

    all_pairs_reduced_MP = all_pairs_reduced_avg[all_pairs_reduced_avg['QSWE_pair'] == MP].copy()
    all_pairs_reduced_MP['marker_key'] = np.where(
        all_pairs_reduced_MP['region'] == 'Dischma',
        'Dischma',
        all_pairs_reduced_MP['wus_region']
    )

    palette = 'Spectral'
    marker_map = {
        'Dischma': 'o',
        'PNW': 'X',
        'Sierra': '^',
        'Interior': 'P',
        'Unmapped WUS': 'D'
    }
    marker_order = ['Dischma', 'PNW', 'Sierra', 'Interior', 'Unmapped WUS']
    marker_order = [m for m in marker_order if m in all_pairs_reduced_MP['marker_key'].unique()]

    # Custom marker legend for regions.
    from matplotlib.lines import Line2D
    legend_entries = [
        ('Dischma', 'o'),
        ('WUS - PNW', 'X'),
        ('WUS - Sierra', '^'),
        ('WUS - Interior', 'P')
    ]
    present_wus_regions = set(all_pairs_reduced_MP.loc[all_pairs_reduced_MP['region'] == 'WUS', 'wus_region'])
    handles = []
    for label, marker in legend_entries:
        if label == 'Dischma' or label.replace('WUS - ', '') in present_wus_regions:
            handles.append(
                Line2D(
                    [0], [0],
                    marker=marker,
                    linestyle='None',
                    markerfacecolor='none' ,#if label == 'Dischma' else '#7f7f7f',
                    markeredgecolor='black',
                    markeredgewidth=1.1,
                    markersize=8,
                    label=label
                )
            )

    # Create one figure per MP:
    # - NSE pair: marker legend only (top-right)
    # - Bias pair: colorbar only (top-right)
    mp_title = MP_label_dic[MP]
    mp_title_lower = mp_title.lower()
    show_legend = 'nse' in mp_title_lower
    show_colorbar = 'bias' in mp_title_lower

    fig, ax = plt.subplots(figsize=(7,6))
    hexbin = ax.hexbin(all_pairs_reduced_MP[delay_zeta], 
                    all_pairs_reduced_MP[meteo_zeta],
                    C=all_pairs_reduced_MP['correlation'],
                    gridsize=12, cmap=palette, 
                    vmin=vmin, vmax=vmax,
                    alpha=1, edgecolors='none')
    scatter = sns.scatterplot(
        data=all_pairs_reduced_MP,
        x=delay_zeta,
        y = meteo_zeta,
        hue='correlation',
        s= 80,
        palette=palette,
        alpha=1,
        edgecolor='black',
        linewidth=0.4,
        style='marker_key',
        style_order=marker_order,
        markers=marker_map,
        hue_norm = (vmin,1.0),
        ax=ax,
        legend=False
    )

    if show_legend:
        ax.legend(handles=handles, title='Catchments', loc='upper right', frameon=True)
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=palette, norm=plt.Normalize(vmin=vmin, vmax=1.0))
        sm.set_array([])
        # Axes fractions must stay within [0, 1] so the colorbar is not clipped outside the figure.
        cax = ax.inset_axes([0.84, 0.55, 0.035, 0.38])
        cbar = plt.colorbar(sm, cax=cax, ticks=[vmin, vmean, vmax])
        cbar.set_label(r'$\rho$ [-]', rotation=270, labelpad=20,fontsize = 14)

    plt.title(mp_title,fontsize = 16)
    ax.set_xlabel(plot_vars_dic[delay_zeta],fontsize = 14)
    ax.set_ylabel(plot_vars_dic[meteo_zeta],fontsize = 14)
    plt.savefig(join(SHARED_PLOTS_DIR, f"Hexbin_plot_markers_{meteo_zeta}_{delay_zeta}_{MP}_{meltseason_test_name}.png"),
                dpi=300, bbox_inches='tight')
    plt.show()
#%%
# if not np.all(np.array(basin_list) =='Dischma'):
#     for MP in reduced_pair_selection:
#         all_pairs_reduced_avg = all_pairs_reduced#[all_pairs_reduced['Year']>=2015].copy()

#         #add a column specifiying whehter a basin has a number in its name or not
#         all_pairs_reduced_avg['region'] = all_pairs_reduced_avg['BASIN'].str.contains(r'\d').map({True: 'Dischma', False: 'WUS'})

#         # for MP in reduced_pair_selection:
#         # MP = reduced_pair_selection[0]
#         all_pairs_reduced_MP = all_pairs_reduced_avg[all_pairs_reduced_avg['QSWE_pair'] == MP]
    

#         fig, axes = plt.subplots(1, 2, figsize=(16,8))
#         palette = 'viridis'

#         # Filter data for Dischma and non-Dischma
#         dischma_data = all_pairs_reduced_MP[all_pairs_reduced_MP['region'] == 'Dischma']
#         non_dischma_data = all_pairs_reduced_MP[all_pairs_reduced_MP['region'] == 'WUS']
        
#         # Plot for Dischma catchments
#         ax = axes[0]
#         if len(dischma_data) > 0:
#             hexbin = ax.hexbin(dischma_data[delay_zeta], 
#                             dischma_data[meteo_zeta],
#                             C=dischma_data['correlation'],
#                             gridsize=10, cmap=palette, 
#                             vmin=0, vmax=1.0,
#                             alpha=1, edgecolors='none')

#             sns.scatterplot(
#                 data=dischma_data,
#                 x=delay_zeta,
#                 y = meteo_zeta,
#                 hue='correlation',
#                 s= 80,
#                 palette=palette,
#                 alpha=1,
#                 edgecolor='black',
#                 linewidth=0.4,
#                 hue_norm = (0,1.0),
#                 ax=ax,
#                 legend=False
#             )
#         ax.set_title(f'{MP_label_dic[MP]} - Region: Dischma')
#         ax.grid(alpha=0.3)
        
#         # Plot for non-Dischma catchments
#         ax = axes[1]
#         if len(non_dischma_data) > 0:
#             hexbin = ax.hexbin(non_dischma_data[delay_zeta], 
#                             non_dischma_data[meteo_zeta],
#                             C=non_dischma_data['correlation'],
#                             gridsize=10, cmap=palette, 
#                             vmin=0, vmax=1.0,
#                             alpha=1, edgecolors='none')

#             scatter = sns.scatterplot(
#                 data=non_dischma_data,
#                 x=delay_zeta,
#                 y = meteo_zeta,
#                 hue='correlation',
#                 s= 80,
#                 palette=palette,
#                 alpha=1,
#                 edgecolor='black',
#                 linewidth=0.4,
#                 hue_norm = (0,1.0),
#                 ax=ax,
#                 legend=False
#             )
            
#             # Create colorbar with fixed intervals
#             sm = plt.cm.ScalarMappable(cmap=palette, norm=plt.Normalize(vmin=0, vmax=1.0))
#             sm.set_array([])
#             cbar = plt.colorbar(sm, ax=ax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
#             cbar.set_label('Metric correlation [-]', rotation=270, labelpad=20)
            
#         ax.set_title(f'{MP_label_dic[MP]    } - Region: WUS')
#         ax.grid(alpha=0.3)
#         ax.set_xlabel(plot_vars_dic[delay_zeta])
#         ax.set_ylabel(plot_vars_dic[meteo_zeta])
        
#         plt.tight_layout()
    

#%%
# Systematic correlation analysis: individual and joint zeta correlations
import statsmodels.api as sm

# Define zeta groups
# meteorological_zetas = ['rainfall_mixing', 'et_mixing', 'sf_ratio','spring_rf_fraction','spring_rf_fraction_beta']
# catchment_zetas = ['S_fraction', 'delay', 'temp_storage','temp_storage_spring','temp_storage_spring_2','temp_storage_spring_beta','temp_storage_spring_2_beta']
# meteorological_zetas = ['spring_rf_fraction',
# 'spring_et_fraction',
# # 'spring_pet',
# 'melt_hfi'
# ]
# 'meltseason_length']
catchment_zetas = [
    'S_fraction',
    # 'S_dynamic_fraction',
    # "temp_storage",
    # "temp_storage_spring",
    # "S_fraction_spring",
    # "S_dynamic_fraction_spring",
]#'ERRA_center_of_mass',



#Extended set of zetas 
meteorological_zetas = [
    # 'rainfall_mixing',
 
 'sf_ratio',
'spring_rf_fraction', 'et_mixing','melt_hfi']
# 'DJF_rainfall_fraction']

catchment_zetas = [
    # 'S_fraction',
    'S_dynamic_fraction',
    'area_km2',
    # 'annual_q_sum',
    'mean_slope_dem',
    # 'slope_proxy_relief',
    # 'mean_elevation_dem',
]
# 'antecedent_Q']'delay',

# 'temp_storage_spring_2']#area_km2
# Keep plotting / analysis order explicit and stable.
# Requested order includes "catchment_area", which corresponds to "area_km2" in the dataframe.
all_zetas = [
    'S_dynamic_fraction',
    'spring_rf_fraction',
    'et_mixing',
    'area_km2',
    'melt_hfi',
    # 'slope_proxy_relief',
    'mean_slope_dem',
    # 'sf_ratio',
]

# Initialize results storage
results = []

# For each QSWE metric pair
for MP in reduced_pair_selection:
    all_pairs_reduced_MP = all_pairs_reduced_avg[all_pairs_reduced_avg['QSWE_pair'] == MP]
    
    # Remove rows with missing correlation values
    data_clean = all_pairs_reduced_MP[['correlation'] + all_zetas].dropna(subset=['correlation'])
    
    if len(data_clean) < 3:
        print(f"\n{MP}: Insufficient data (n={len(data_clean)})")
        continue
    
    # 1. Individual correlations for each zeta
    for zeta in all_zetas:
        if zeta not in data_clean.columns:
            print(zeta, 'not in data_clean')
            continue
        
        zeta_data = data_clean[[zeta, 'correlation']].dropna()
        if len(zeta_data) >= 3:
            # Use pearsonr to get correlation and p-value
            corr_coef, p_value = pearsonr(zeta_data[zeta], zeta_data['correlation'])
            
            results.append({
                'QSWE_pair': MP,
                'zeta_1': zeta,
                'zeta_2': None,
                'zeta_type': 'meteorological' if zeta in meteorological_zetas else 'catchment',
                'correlation_type': 'individual',
                'R': corr_coef,
                'R_squared': corr_coef**2 if not np.isnan(corr_coef) else np.nan,
                'p_value': p_value,
                'n': len(zeta_data)
            })
    
    # 2. Joint correlations for pairs (meteorological × catchment)
    for zeta_m in meteorological_zetas:
        for catch_zeta in catchment_zetas:
            if zeta_m not in data_clean.columns or catch_zeta not in data_clean.columns:
                continue
            
            pair_data = data_clean[[zeta_m, catch_zeta, 'correlation']].dropna()
            if len(pair_data) >= 3:
                X = pair_data[[zeta_m, catch_zeta]]
                y = pair_data['correlation']
                
                # Add constant for intercept
                X_with_const = sm.add_constant(X)
                
                # Fit OLS model
                try:
                    model = sm.OLS(y, X_with_const).fit()
                    R_squared = model.rsquared
                    R_multiple = np.sqrt(R_squared) if R_squared >= 0 else np.nan
                    f_pvalue = model.f_pvalue
                    
                    results.append({
                        'QSWE_pair': MP,
                        'zeta_1': zeta_m,
                        'zeta_2': catch_zeta,
                        'zeta_type': 'meteorological × catchment',
                        'correlation_type': 'joint',
                        'R': R_multiple,
                        'R_squared': R_squared,
                        'p_value': f_pvalue,
                        'n': len(pair_data),
                        'coef_zeta1': model.params[zeta_m],
                        'coef_zeta2': model.params[catch_zeta],
                        'p_zeta1': model.pvalues[zeta_m],
                        'p_zeta2': model.pvalues[catch_zeta]
                    })
                except Exception as e:
                    print(f"Error fitting model for {MP}, {zeta_m} × {catch_zeta}: {e}")

# Convert results to DataFrame
correlation_results_df = pd.DataFrame(results)

# Display summary
print("\n" + "="*80)
print("SYSTEMATIC ZETA CORRELATION ANALYSIS")
print("="*80)

# Summary by correlation type
print("\n--- INDIVIDUAL CORRELATIONS ---")
individual = correlation_results_df[correlation_results_df['correlation_type'] == 'individual'].copy()
if len(individual) > 0:
    print(f"\nTop individual correlations (by absolute R):")
    top_individual = individual.reindex(individual['R'].abs().sort_values(ascending=False).index)
    print(top_individual[['QSWE_pair', 'zeta_1', 'R', 'R_squared', 'p_value', 'n']].head(20).to_string(index=False))
    
    # Summary statistics
    print(f"\nSummary by zeta:")
    zeta_summary = individual.groupby('zeta_1').agg({
        'R': ['mean', 'std', 'count'],
        'R_squared': 'mean',
        'p_value': lambda x: (x < 0.05).sum()
    }).round(4)
    print(zeta_summary)

print("\n--- JOINT CORRELATIONS (Meteorological × Catchment) ---")
joint = correlation_results_df[correlation_results_df['correlation_type'] == 'joint'].copy()
if len(joint) > 0:
    print(f"\nTop joint correlations (by R²):")
    top_joint = joint.reindex(joint['R_squared'].sort_values(ascending=False).index)
    print(top_joint[['QSWE_pair', 'zeta_1', 'zeta_2', 'R', 'R_squared', 'p_value', 'n']].head(20).to_string(index=False))
    
    # Summary by zeta pair
    print(f"\nSummary by zeta pair:")
    joint['zeta_pair'] = joint['zeta_1'] + ' × ' + joint['zeta_2']
    pair_summary = joint.groupby('zeta_pair').agg({
        'R_squared': ['mean', 'std', 'count'],
        'p_value': lambda x: (x < 0.05).sum()
    }).round(4)
    print(pair_summary)

print("\n" + "="*80)
print(f"Total results: {len(correlation_results_df)}")
print(f"  - Individual: {len(individual)}")
print(f"  - Joint: {len(joint)}")
print("="*80)

# Create pivot tables for easier viewing
print("\n--- PIVOT TABLE: Individual Correlations by QSWE Pair ---")
if len(individual) > 0:
    pivot_individual = individual.pivot_table(
        index='zeta_1', 
        columns='QSWE_pair', 
        values='R', 
        aggfunc='mean'
    )
    print(pivot_individual.round(3).to_string())

print("\n--- PIVOT TABLE: Joint Correlations (R²) by QSWE Pair ---")
if len(joint) > 0:
    joint['zeta_pair'] = joint['zeta_1'] + ' × ' + joint['zeta_2']
    pivot_joint = joint.pivot_table(
        index='zeta_pair',
        columns='QSWE_pair',
        values='R_squared',
        aggfunc='mean'
    )
    print(pivot_joint.round(3).to_string())

# Save to CSV
output_file = join(ROOTDIR, 'aux_data', f"zeta_correlation_analysis_{meltseason_test_name}.csv")
correlation_results_df.to_csv(output_file, index=False)
print(f"\nFull results saved to: {output_file}")

#%%
# Prepare data for heatmap
heatmap_data = []

for MP in reduced_pair_selection:
    mp_data = correlation_results_df[correlation_results_df['QSWE_pair'] == MP].copy()
    
    # 1. Mean R: actual mean correlation of the metric pair across all experiments and years
    # mp_correlations = all_pairs_reduced[all_pairs_reduced['QSWE_pair'] == MP]['correlation']
    # mean_R = mp_correlations.mean() if len(mp_correlations) > 0 else np.nan
    
    # 2. Correlation with meteo_zeta (individual)
    individual_mp = mp_data[mp_data['correlation_type'] == 'individual']
    meteo_individual = individual_mp[individual_mp['zeta_1'] == meteo_zeta]
    R_meteo = meteo_individual['R'].values[0] if len(meteo_individual) > 0 else np.nan
    R_meteo = R_meteo**2
    
    # 3. Correlation with delay_zeta (individual)
    delay_individual = individual_mp[individual_mp['zeta_1'] == delay_zeta]
    R_delay = delay_individual['R'].values[0] if len(delay_individual) > 0 else np.nan
    R_delay = R_delay**2
    # 4. Joint correlation (meteo_zeta × delay_zeta)
    joint_mp = mp_data[mp_data['correlation_type'] == 'joint']
    joint_specific = joint_mp[
        ((joint_mp['zeta_1'] == meteo_zeta) & (joint_mp['zeta_2'] == delay_zeta)) |
        ((joint_mp['zeta_1'] == delay_zeta) & (joint_mp['zeta_2'] == meteo_zeta))
    ]
    R_joint = joint_specific['R'].values[0] if len(joint_specific) > 0 else np.nan
    R_joint = R_joint**2
    # heatmap_data.append({
    # 'MP': MP,
    # # 'Mean Q-SWE correlation': mean_R,  # The actual metric correlation
    # f'Signature corr: {meteo_zeta}': R_meteo,  # How meteo signature explains Q-SWE corr
    # f'Signature corr: {delay_zeta}': R_delay,  # How delay signature explains Q-SWE corr
    # f'Joint signature corr ({meteo_zeta} × {delay_zeta})': R_joint  # Combined explanation
    #         })

    # heatmap_data.append({
    #     'MP': MP,
    #     fr'$\rho (\rho_\mathrm{{MP}}  |  \zeta_\mathrm{{meteo}})$': R_meteo,
    #     fr'$\rho (\rho_\mathrm{{MP}}  |  \zeta_\mathrm{{delay}})$': R_delay,
    #     fr'$R^2(\rho_\mathrm{{MP}} | \zeta_\mathrm{{meteo}}, \zeta_\mathrm{{delay}})$': R_joint
        
    #     # f'Joint signature correlation': R_joint
    # })

    # heatmap_data.append({
    #     'MP': MP,
    #     fr'$(R^2_\mathrm{{MP}}  |  \zeta_\mathrm{{rain}})$': R_meteo,
    #     fr'$(R^2_\mathrm{{MP}}  |  \zeta_\mathrm{{\tau}})$': R_delay,
    #     fr'$(R^2_\mathrm{{MP}} | \zeta_\mathrm{{rain}}, \zeta_\mathrm{{\tau}})$': R_joint
        
    #     # f'Joint signature correlation': R_joint
    # })

    heatmap_data.append({
        'MP': MP,
        'S1': R_meteo,
        'S2': R_delay,
        'Joint \n S1+S2': R_joint
        # f'Joint signature correlation': R_joint
    })


# Create DataFrame for heatmap
heatmap_df = pd.DataFrame(heatmap_data)
heatmap_df.set_index('MP', inplace=True)

# Transpose so MPs are columns and correlation types are rows
heatmap_df = heatmap_df.T

# Use shorter labels from MP_label_dic if available
column_labels = [MP_label_dic.get(mp, mp) for mp in heatmap_df.columns]

# Create heatmap
fig, ax = plt.subplots(figsize=(max(3.5, len(reduced_pair_selection) * 0.8), 3))
sns.heatmap(
    heatmap_df,
    annot=True,
    fmt='.2f',
    cmap='Reds',
    # center=0,
    vmin=0,
    vmax=1,
    # cbar_kws={'label': r'Predictor correlation (R²)'},
    cbar_kws = {'label': r'Explained variance of \rho (R²)'},
    ax=ax,
    linewidths=0.5,
    linecolor='gray',
    xticklabels=column_labels
)

ax.set_xlabel('Metric Pair', fontsize=10)
# ax.set_ylabel('Correlation Type', fontsize=12)

# Move x-axis labels to top
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
# plt.xticks(rotation=45, ha='left')
plt.yticks(rotation=0)
plt.axhline(2,color = 'black', alpha  =0.5)

plt.tight_layout()

# Save heatmap
heatmap_file = join(SHARED_PLOTS_DIR, f"zeta_correlation_heatmap_{meteo_zeta}_{delay_zeta}_{meltseason_test_name}.png")
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
print(f"\nHeatmap saved to: {heatmap_file}")
plt.show()
print(heatmap_df)

#%%
# Heatmap: Average metric pair correlation for each MP
print("\n" + "="*80)
print("HEATMAP: Average Metric Pair Correlation")
print("="*80)

# Prepare data for average correlation heatmap
avg_corr_data = {}
for MP in reduced_pair_selection:
    mp_correlations = all_pairs_reduced[all_pairs_reduced['QSWE_pair'] == MP]['correlation']
    avg_corr_data[MP] = mp_correlations.median() if len(mp_correlations) > 0 else np.nan

# Create DataFrame with one row
avg_corr_df = pd.DataFrame([avg_corr_data])
avg_corr_df.index = ['Mean \n ' + r'\rho']

# Use shorter labels from MP_label_dic if available
column_labels = [MP_label_dic.get(mp, mp) for mp in avg_corr_df.columns]

# Create heatmap
fig, ax = plt.subplots(figsize=(max(3.5, len(reduced_pair_selection) * 0.8), 1.5))
sns.heatmap(
    avg_corr_df,
    annot=True,
    fmt='.2f',
    cmap='viridis',
    vmin=0.5,
    vmax=1,
    cbar_kws={'label': r'Correlation (R²)'},
    ax=ax,
    linewidths=0.5,
    linecolor='gray',
    xticklabels=column_labels
)

ax.set_xlabel('Metric Pair', fontsize=10)

# Move x-axis labels to top
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.yticks(rotation=0)

plt.tight_layout()

# Save heatmap
avg_corr_heatmap_file = join(SHARED_PLOTS_DIR, f"avg_metric_correlation_heatmap_{meltseason_test_name}.png")
plt.savefig(avg_corr_heatmap_file, dpi=300, bbox_inches='tight')
print(f"\nAverage correlation heatmap saved to: {avg_corr_heatmap_file}")
plt.show()
print(avg_corr_df)

#%%

#%%
# ============================================================================
# Hierarchical regression analysis: Does predictor B explain additional variance 
# once predictor A is already known?
# ============================================================================

from scipy.stats import f

print("\n" + "="*80)
print("HIERARCHICAL REGRESSION ANALYSIS")
print("Testing if each predictor adds significant variance beyond the other")
print("="*80)

# Initialize results storage for hierarchical analysis
hierarchical_results = []

# For each QSWE metric pair
for MP in reduced_pair_selection:
    all_pairs_reduced_MP = all_pairs_reduced_avg[all_pairs_reduced_avg['QSWE_pair'] == MP]
    
    # Remove rows with missing correlation values
    data_clean = all_pairs_reduced_MP[['correlation'] + all_zetas].dropna(subset=['correlation'])
    
    if len(data_clean) < 3:
        continue
    
    # For each pair (meteorological × catchment)
    for m_zeta in meteorological_zetas:
        for catch_zeta in catchment_zetas:
            if m_zeta not in data_clean.columns or catch_zeta not in data_clean.columns:
                continue
            
            pair_data = data_clean[[m_zeta, catch_zeta, 'correlation']].dropna()
            if len(pair_data) < 3:
                continue
            
            y = pair_data['correlation']
            
            # Test 1: Does catch_zeta add variance beyond m_zeta?
            try:
                # Model 1: Only m_zeta
                X1 = sm.add_constant(pair_data[[m_zeta]])
                model1 = sm.OLS(y, X1).fit()
                R2_model1 = model1.rsquared
                
                # Model 2: m_zeta + catch_zeta
                X2 = sm.add_constant(pair_data[[m_zeta, catch_zeta]])
                model2 = sm.OLS(y, X2).fit()
                R2_model2 = model2.rsquared
                
                # Calculate ΔR²
                delta_R2 = R2_model2 - R2_model1
                
                # F-test for nested models
                # F = ((R2_full - R2_reduced) / (df_full - df_reduced)) / ((1 - R2_full) / df_residual_full)
                n = len(pair_data)
                df_reduced = model1.df_resid
                df_full = model2.df_resid
                df_diff = df_reduced - df_full  # Should be 1
                
                if df_full > 0 and delta_R2 > 0:
                    F_stat = (delta_R2 / df_diff) / ((1 - R2_model2) / df_full)
                    # p-value from F-distribution
                    p_value_delta = 1 - f.cdf(F_stat, df_diff, df_full)
                else:
                    F_stat = np.nan
                    p_value_delta = np.nan
                
                hierarchical_results.append({
                    'QSWE_pair': MP,
                    'zeta_base': m_zeta,
                    'zeta_added': catch_zeta,
                    'R2_base_only': R2_model1,
                    'R2_base_plus_added': R2_model2,
                    'delta_R2': delta_R2,
                    'F_statistic': F_stat,
                    'p_value_delta': p_value_delta,
                    'n': n
                })
            except Exception as e:
                print(f"Error in hierarchical test 1 for {MP}, {m_zeta} + {catch_zeta}: {e}")
            
            # Test 2: Does m_zeta add variance beyond catch_zeta?
            try:
                # Model 1: Only catch_zeta
                X1 = sm.add_constant(pair_data[[catch_zeta]])
                model1 = sm.OLS(y, X1).fit()
                R2_model1 = model1.rsquared
                
                # Model 2: catch_zeta + m_zeta
                X2 = sm.add_constant(pair_data[[catch_zeta, m_zeta]])
                model2 = sm.OLS(y, X2).fit()
                R2_model2 = model2.rsquared
                
                # Calculate ΔR²
                delta_R2 = R2_model2 - R2_model1
                
                # F-test for nested models
                n = len(pair_data)
                df_reduced = model1.df_resid
                df_full = model2.df_resid
                df_diff = df_reduced - df_full  # Should be 1
                
                if df_full > 0 and delta_R2 > 0:
                    F_stat = (delta_R2 / df_diff) / ((1 - R2_model2) / df_full)
                    p_value_delta = 1 - f.cdf(F_stat, df_diff, df_full)
                else:
                    F_stat = np.nan
                    p_value_delta = np.nan
                
                hierarchical_results.append({
                    'QSWE_pair': MP,
                    'zeta_base': catch_zeta,
                    'zeta_added': m_zeta,
                    'R2_base_only': R2_model1,
                    'R2_base_plus_added': R2_model2,
                    'delta_R2': delta_R2,
                    'F_statistic': F_stat,
                    'p_value_delta': p_value_delta,
                    'n': n
                })
            except Exception as e:
                print(f"Error in hierarchical test 2 for {MP}, {catch_zeta} + {m_zeta}: {e}")

print("\n" + "="*80)
print("HEATMAP: Correlations for m_zeta × delay_zeta combination")
print(f"m_zeta: {m_zeta}, delay_zeta: {delay_zeta}")
print("="*80)

#%%
# Convert hierarchical results to DataFrame
hierarchical_df = pd.DataFrame(hierarchical_results)

if len(hierarchical_df) > 0:
    # Display summary
    print("\n--- HIERARCHICAL REGRESSION RESULTS ---")
    print("\nTop cases where added predictor significantly improves model (p < 0.05):")
    significant = hierarchical_df[hierarchical_df['p_value_delta'] < 0.05].copy()
    if len(significant) > 0:
        significant_sorted = significant.reindex(significant['delta_R2'].abs().sort_values(ascending=False).index)
        print(significant_sorted[['QSWE_pair', 'zeta_base', 'zeta_added', 'R2_base_only', 
                                  'R2_base_plus_added', 'delta_R2', 'p_value_delta', 'n']].head(30).to_string(index=False))
    else:
        print("No significant improvements found (p < 0.05)")
    
    print("\n--- Summary by added predictor ---")
    summary_by_added = hierarchical_df.groupby('zeta_added').agg({
        'delta_R2': ['mean', 'std', 'count'],
        'p_value_delta': lambda x: (x < 0.05).sum()
    }).round(4)
    print(summary_by_added)
    
    print("\n--- Summary by base predictor ---")
    summary_by_base = hierarchical_df.groupby('zeta_base').agg({
        'delta_R2': ['mean', 'std', 'count'],
        'p_value_delta': lambda x: (x < 0.05).sum()
    }).round(4)
    print(summary_by_base)
    
    print("\n--- Cases where added predictor explains substantial additional variance (ΔR² > 0.1) ---")
    substantial = hierarchical_df[hierarchical_df['delta_R2'] > 0.1].copy()
    if len(substantial) > 0:
        substantial_sorted = substantial.reindex(substantial['delta_R2'].sort_values(ascending=False).index)
        print(substantial_sorted[['QSWE_pair', 'zeta_base', 'zeta_added', 'R2_base_only', 
                                  'R2_base_plus_added', 'delta_R2', 'p_value_delta', 'n']].head(20).to_string(index=False))
    else:
        print("No cases with ΔR² > 0.1")
    
    # Create pivot table for ΔR²
    print("\n--- PIVOT TABLE: ΔR² by QSWE Pair (when adding second predictor) ---")
    hierarchical_df['zeta_combo'] = hierarchical_df['zeta_base'] + ' + ' + hierarchical_df['zeta_added']
    pivot_delta = hierarchical_df.pivot_table(
        index='zeta_combo',
        columns='QSWE_pair',
        values='delta_R2',
        aggfunc='mean'
    )
    print(pivot_delta.round(3).to_string())
    
    # Save hierarchical results
    hierarchical_output_file = join(ROOTDIR, 'aux_data', f"zeta_hierarchical_regression_analysis_{meltseason_test_name}.csv")
    hierarchical_df.to_csv(hierarchical_output_file, index=False)
    print(f"\nHierarchical regression results saved to: {hierarchical_output_file}")
    
    print("\n" + "="*80)
    print(f"Total hierarchical tests: {len(hierarchical_df)}")
    print(f"Significant improvements (p < 0.05): {len(hierarchical_df[hierarchical_df['p_value_delta'] < 0.05])}")
    print(f"Substantial improvements (ΔR² > 0.1): {len(hierarchical_df[hierarchical_df['delta_R2'] > 0.1])}")
    print("="*80)
else:
    print("\nNo hierarchical regression results generated.")

# %%
# Create zeta_combo column first
hierarchical_df['zeta_combo'] = hierarchical_df['zeta_base'] + ' + ' + hierarchical_df['zeta_added']

# Prepare data for grouped bar plot
plot_data = hierarchical_df.copy()
plot_data['R2_base_only'] = plot_data['R2_base_only'].clip(0, 1)
plot_data['R2_base_plus_added'] = plot_data['R2_base_plus_added'].clip(0, 1)

# Create faceted plot by QSWE pair
g = sns.FacetGrid(plot_data, col='QSWE_pair', col_wrap=3, 
                  height=6, aspect=0.8, sharex=True)  # sharex instead of sharey

# Plot both bars horizontally - order matters! Plot base first (left), then full (right)
def plot_bars(data, **kwargs):
    ax = plt.gca()
    
    # Get unique zeta_combos for this facet
    zeta_combos = data['zeta_combo'].unique()
    y = np.arange(len(zeta_combos))
    height = 0.35
    
    # Plot base only (left)
    base_values = [data[data['zeta_combo'] == zc]['R2_base_only'].values[0] 
                   for zc in zeta_combos]
    bars1 = ax.barh(y - height/2, base_values, height, 
                    label='Base only', color='orange', alpha=0.7)
    
    # Plot base + added (right)
    full_values = [data[data['zeta_combo'] == zc]['R2_base_plus_added'].values[0] 
                   for zc in zeta_combos]
    bars2 = ax.barh(y + height/2, full_values, height, 
                    label='Base + Added', color='darkblue', alpha=0.7)
    
    ax.set_yticks(y)
    ax.set_yticklabels(zeta_combos)
    ax.set_xlabel('R²')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

g.map_dataframe(plot_bars)
g.fig.suptitle('R² Comparison: Base Model vs Full Model', 
               fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

#%%
# ============================================================================
# COMPREHENSIVE VARIANCE PARTITIONING ANALYSIS
# ============================================================================
# Framework (following hierarchical / variance decomposition approach):
#   1. Multicollinearity check (pairwise correlations + VIF)
#   2. Marginal R² (single-predictor regressions)
#   3. Full model R² (all predictors together)
#   4. Unique ΔR² (each predictor beyond all others)
#   5. Variance decomposition (unique + shared + unexplained)
#   6. Cross-validated R² (LOGO by catchment / BASIN; k-fold by row)
#   7. LASSO robustness check
# Response variable is Fisher z-transformed rank correlation.
# ============================================================================

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f as f_dist
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    LeaveOneGroupOut,
    GroupKFold,
    KFold,
    cross_val_predict,
)
from sklearn.metrics import r2_score

print("\n" + "="*80)
print("COMPREHENSIVE VARIANCE PARTITIONING ANALYSIS")
print("="*80)

# --- Data preparation ---------------------------------------------------------
# Each catchment-year is one data point (no averaging over years)
zeta_cols = [z for z in all_zetas if z in all_pairs_reduced.columns]
vp_data = all_pairs_reduced[['QSWE_pair', 'BASIN', 'correlation'] + zeta_cols]
vp_data = vp_data.dropna(subset=['correlation'] + zeta_cols).copy()

# Fisher z-transform: unbounds correlation from [-1,1] → (-∞,∞)
vp_data['correlation_z'] = np.arctanh(
    np.clip(vp_data['correlation'], -0.999, 0.999))
resp = 'correlation_z'

print(f"Data points per MP (catchment-years):")
for MP in reduced_pair_selection:
    n_mp = len(vp_data[vp_data['QSWE_pair'] == MP])
    print(f"  {MP_label_dic.get(MP, MP)}: n={n_mp}")

# Readable predictor labels
zeta_short = {z: plot_vars_dic.get(z, z) for z in zeta_cols}

# =============================================================================
# STEP 1 — Multicollinearity (pooled across all MPs)
# =============================================================================
print("\n--- STEP 1: Multicollinearity ---")
pool = vp_data[zeta_cols].dropna()
corr_mat = pool.corr()
print("Pairwise correlations:\n", corr_mat.round(3))

pool_c = sm.add_constant(pool)
vif_df = pd.DataFrame({
    'Predictor': zeta_cols,
    'VIF': [variance_inflation_factor(pool_c.values, i + 1)
            for i in range(len(zeta_cols))]
})
print("\nVariance Inflation Factors:\n", vif_df.to_string(index=False))
print("  (VIF > 5 = moderate, VIF > 10 = severe multicollinearity)")

# Plot: correlation matrix + VIF bars
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5),
                         gridspec_kw={'width_ratios': [1.2, 0.8]})
tl = [zeta_short.get(z, z) for z in corr_mat.columns]
sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=axes[0], linewidths=.5, square=True,
            xticklabels=tl, yticklabels=tl)
axes[0].set_title('Predictor correlations')

c_vif = ['indianred' if v > 5 else 'steelblue' for v in vif_df['VIF']]
axes[1].barh(range(len(vif_df)), vif_df['VIF'], color=c_vif)
axes[1].set_yticks(range(len(vif_df)))
axes[1].set_yticklabels([zeta_short.get(z, z) for z in vif_df['Predictor']])
axes[1].axvline(5, color='red', ls='--', alpha=.5, label='VIF = 5')
axes[1].axvline(10, color='darkred', ls='--', alpha=.5, label='VIF = 10')
axes[1].set_xlabel('VIF')
axes[1].legend(fontsize=7)
axes[1].grid(axis='x', alpha=.3)
axes[1].set_title('Variance Inflation Factors')
# plt.tight_layout()
plt.savefig(join(SHARED_PLOTS_DIR,
                 f"vp_multicollinearity_{meltseason_test_name}.png"),
            dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# STEPS 2–7 — Per metric pair
# =============================================================================
vp_results = {}

for MP in reduced_pair_selection:
    lab = MP_label_dic.get(MP, MP)
    df_mp = vp_data[vp_data['QSWE_pair'] == MP][['BASIN', resp] + zeta_cols].dropna()
    groups = df_mp['BASIN'].to_numpy()
    n_basins = len(np.unique(groups))
    n = len(df_mp)
    print(f"\n{'='*60}\n  {lab}  (n={n} catchment-years, {n_basins} catchments)\n{'='*60}")
    if n < len(zeta_cols) + 2:
        print("  Insufficient data — skipping")
        continue

    y = df_mp[resp].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df_mp[zeta_cols].values)
    Xdf = pd.DataFrame(Xs, columns=zeta_cols)

    # STEP 2 — Marginal R² (individual regressions)
    marg = {}
    for z in zeta_cols:
        m = sm.OLS(y, sm.add_constant(Xdf[[z]])).fit()
        marg[z] = {'R2': m.rsquared, 'p': m.pvalues[z], 'beta': m.params[z]}
    print("  Step 2 — Marginal R²:")
    for z in zeta_cols:
        s = '*' if marg[z]['p'] < .05 else ''
        print(f"    {z:30s}  R²={marg[z]['R2']:.3f}  p={marg[z]['p']:.3f} {s}")

    # STEP 3 — Full model
    mf = sm.OLS(y, sm.add_constant(Xdf)).fit()
    R2f = mf.rsquared
    R2a = mf.rsquared_adj
    print(f"  Step 3 — Full model:  R²={R2f:.3f}  Adj R²={R2a:.3f}  "
          f"F-p={mf.f_pvalue:.4f}")
    for z in zeta_cols:
        print(f"    {z:30s}  β={mf.params[z]:+.3f}  p={mf.pvalues[z]:.3f}")

    # STEP 4 — Unique ΔR² (drop-one-predictor approach)
    uniq = {}
    for z in zeta_cols:
        others = [zz for zz in zeta_cols if zz != z]
        mr = sm.OLS(y, sm.add_constant(Xdf[others])).fit()
        dR2 = R2f - mr.rsquared
        dfn, dfd = 1, mf.df_resid
        if dfd > 0 and dR2 > 0:
            F = (dR2 / dfn) / ((1 - R2f) / dfd)
            p = 1 - f_dist.cdf(F, dfn, dfd)
        else:
            F, p = 0, 1.0
        uniq[z] = {'dR2': dR2, 'p': p}
    print("  Step 4 — Unique ΔR²:")
    for z in zeta_cols:
        s = '*' if uniq[z]['p'] < .05 else ''
        print(f"    {z:30s}  ΔR²={uniq[z]['dR2']:.3f}  p={uniq[z]['p']:.3f} {s}")

    # STEP 5 — Variance decomposition
    sum_uniq = sum(v['dR2'] for v in uniq.values())
    shared = R2f - sum_uniq
    unexp = 1 - R2f
    print(f"  Step 5 — Decomposition:  Σunique={sum_uniq:.3f}  "
          f"shared={shared:.3f}  unexplained={unexp:.3f}")
    if shared < 0:
        print(f"  ⚠ SUPPRESSOR EFFECT detected (shared < 0):")
        print(f"    Correlated predictors act cooperatively — together they")
        print(f"    explain more than the sum of their individual effects.")
        print(f"    Unique ΔR² values sum to {sum_uniq:.3f} > R²_full={R2f:.3f}.")

    # STEP 6 — Cross-validated R²
    # LOGO: leave one catchment (BASIN) out — all years from that basin held out together
    logo = LeaveOneGroupOut()
    lr = LinearRegression()
    if n_basins >= 2:
        y_pred_logo = cross_val_predict(lr, Xs, y, cv=logo, groups=groups)
        cv_logo = r2_score(y, y_pred_logo)
    else:
        cv_logo = np.nan
        print("  Step 6 — LOGO-CV skipped (need ≥2 distinct catchments)")
    k = min(5, n)
    if k >= 2:
        y_pred_kf = cross_val_predict(
            lr, Xs, y, cv=KFold(k, shuffle=True, random_state=42))
        cv_kf = r2_score(y, y_pred_kf)
    else:
        cv_kf = np.nan
    if np.isfinite(cv_logo):
        print(f"  Step 6 — CV R²:  LOGO(catchment)={cv_logo:.3f}  "
              f"{k}-fold(row)={cv_kf:.3f}")
    else:
        print(f"  Step 6 — CV R²:  LOGO=nan  {k}-fold(row)={cv_kf:.3f}")

    # STEP 7 — LASSO robustness (grouped CV via explicit splits; sklearn
    # LassoCV.fit(..., groups=...) requires enable_metadata_routing in recent versions)
    try:
        if n_basins >= 2:
            n_splits_las = max(2, min(5, n_basins))
            las_cv_splits = list(
                GroupKFold(n_splits=n_splits_las).split(Xs, y, groups))
            las = LassoCV(cv=las_cv_splits, random_state=42, max_iter=10000).fit(
                Xs, y)
        else:
            k_las = min(5, max(2, n - 1))
            las = LassoCV(cv=k_las, random_state=42, max_iter=10000).fit(Xs, y)
        las_r2 = las.score(Xs, y)
        las_c = dict(zip(zeta_cols, las.coef_))
        print(f"  Step 7 — LASSO R²={las_r2:.3f}  (α={las.alpha_:.4f})")
        for z in zeta_cols:
            tag = 'selected' if abs(las_c[z]) > 1e-6 else 'dropped'
            print(f"    {z:30s}  β={las_c[z]:+.3f}  ({tag})")
    except Exception as e:
        print(f"  LASSO failed: {e}")
        las_r2 = np.nan
        las_c = {z: np.nan for z in zeta_cols}

    vp_results[MP] = dict(label=lab, n=n, n_basins=n_basins, marginal=marg,
                          R2_full=R2f, R2_adj=R2a,
                          unique=uniq, shared=shared,
                          cv_logo=cv_logo, cv_kf=cv_kf,
                          lasso_r2=las_r2, lasso_coefs=las_c)
#%%
# =============================================================================
# SUMMARY PLOTS
# =============================================================================
if vp_results:
    mps = list(vp_results.keys())
    labs = [vp_results[m]['label'] for m in mps]
    nm = len(mps)

    # ---- Plot A: Marginal vs Unique R² per MP ----
    fig, axes = plt.subplots(1, nm, figsize=(4.5 * nm, 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for idx, (MP, ax) in enumerate(zip(mps, axes)):
        r = vp_results[MP]
        xpos = np.arange(len(zeta_cols))
        w = 0.35
        m_vals = [r['marginal'][z]['R2'] for z in zeta_cols]
        u_vals = [r['unique'][z]['dR2'] for z in zeta_cols]
        ax.bar(xpos - w/2, m_vals, w, label='Marginal R²',
               color='steelblue', alpha=.8, ec='k', lw=.5)
        ax.bar(xpos + w/2, u_vals, w, label='Unique ΔR²',
               color='coral', alpha=.8, ec='k', lw=.5)
        # significance markers
        for i, z in enumerate(zeta_cols):
            if r['marginal'][z]['p'] < .05:
                ax.text(i - w/2, m_vals[i] + .01, '*',
                        ha='center', fontsize=13, fontweight='bold')
            if r['unique'][z]['p'] < .05:
                ax.text(i + w/2, u_vals[i] + .01, '*',
                        ha='center', fontsize=13, fontweight='bold')
        ax.axhline(r['R2_full'], c='k', ls='-', lw=1.2,
                   label=f"Full R²={r['R2_full']:.2f}")
        if np.isfinite(r['cv_logo']):
            ax.axhline(r['cv_logo'], c='grey', ls='--', lw=1.2,
                       label=f"LOO R²={r['cv_logo']:.2f}")
        ax.set_xticks(xpos)
        ax.set_xticklabels([zeta_short.get(z, z) for z in zeta_cols],
                           rotation=45, ha='right', fontsize=7)
        ax.set_title(r['label'], fontsize=12)
        ax.legend(fontsize=7, loc='best')
        ax.grid(axis='y', alpha=.3)
        if idx == 0:
            ax.set_ylabel('R²')
    fig.suptitle('Marginal vs Unique Predictor Contributions',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(join(SHARED_PLOTS_DIR,
                     f"vp_marginal_unique_{meltseason_test_name}.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # ---- Plot A2: Marginal vs Unique R² per MP (horizontal) ----
    fig, axes = plt.subplots(1, nm, figsize=(3.8 * nm, max(3.5, 0.25 * len(zeta_cols) + 1.8)),
                            sharey=True, squeeze=False)
    
    axes = axes[0]

    for idx, (MP, ax) in enumerate(zip(mps, axes)):
        r = vp_results[MP]
        ypos = np.arange(len(zeta_cols))
        h = 0.35

        m_vals = [r['marginal'][z]['R2'] for z in zeta_cols]
        u_vals = [r['unique'][z]['dR2'] for z in zeta_cols]

        if idx == 0:
            ax.barh(ypos - h/2, m_vals, h, label='Marginal R²',
                    color='steelblue', alpha=.8, ec='k', lw=.5)
            ax.barh(ypos + h/2, u_vals, h, label='Unique ΔR²',
                    color='coral', alpha=.8, ec='k', lw=.5)
        else:
            ax.barh(ypos - h/2, m_vals, h,
                    color='steelblue', alpha=.8, ec='k', lw=.5)
            ax.barh(ypos + h/2, u_vals, h,
                    color='coral', alpha=.8, ec='k', lw=.5)
           

        # # significance markers
        # for i, z in enumerate(zeta_cols):
        #     if r['marginal'][z]['p'] < .05:
        #         ax.text(m_vals[i] + .01, i - h/2, '*',
        #                 va='center', fontsize=13, fontweight='bold')
        #     if r['unique'][z]['p'] < .05:
        #         ax.text(u_vals[i] + .01, i + h/2, '*',
        #                 va='center', fontsize=13, fontweight='bold')

        # full-model and CV references (vertical lines for horizontal bars)
        ax.axvline(r['R2_full'], c='k', ls='-', lw=1.2,
                label=f"Full R²={r['R2_full']:.2f}")
        if np.isfinite(r['cv_logo']):
            ax.axvline(r['cv_logo'], c='grey', ls='--', lw=1.2,
                       label=f"LOO R²={r['cv_logo']:.2f}")

        ax.set_yticks(ypos)
        ax.set_yticklabels(
            [
                zeta_short.get(z, z).rsplit('[', 1)[0].rstrip(' ') 
                if '[' in zeta_short.get(z, z) else zeta_short.get(z, z)
                for z in zeta_cols
            ],
            fontsize=9
        )
   
        # With shared y-axis, invert only once so first zeta appears on top.
        if idx == 0:
            ax.invert_yaxis()
        ax.set_title(r['label'], fontsize=12)
        # if idx == 0:
        ax.legend(fontsize=9, loc='lower center')
        ax.grid(axis='x', alpha=.3)
        ax.set_ylabel('')
        # if idx == 0:
        #     ax.set_ylabel('Catchment descriptor')
        ax.set_xlabel('Variance explained (R²)')

    plt.tight_layout()
    plt.savefig(join(SHARED_PLOTS_DIR,
                    f"vp_marginal_unique_horizontal_{meltseason_test_name}.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # ---- All VP predictors (ζ) vs ρ (catchment-years; same rows as VP) ----
    nz = len(zeta_cols)
    if nz > 0:
        fig, axes = plt.subplots(nz, nm, figsize=(3.8 * nm, 2.65 * nz),
                                 sharey=True, squeeze=False)
        axes = np.atleast_2d(axes)
        for i, z in enumerate(zeta_cols):
            for j, MP in enumerate(mps):
                ax = axes[i, j]
                sub = all_pairs_reduced.loc[
                    all_pairs_reduced['QSWE_pair'] == MP, [z, 'correlation']
                ].dropna()
                if len(sub) >= 2:
                    ax.scatter(
                        sub[z], sub['correlation'],
                        alpha=.55, s=22, c='steelblue', ec='k', lw=.35)
                    xn = sub[z].to_numpy()
                    yn = sub['correlation'].to_numpy()
                    if np.std(xn) > 1e-12:
                        coef = np.polyfit(xn, yn, 1)
                        xs = np.linspace(xn.min(), xn.max(), 40)
                        ax.plot(xs, np.poly1d(coef)(xs), c='crimson',
                                ls='--', lw=1.1)
                if i == 0:
                    ax.set_title(vp_results[MP]['label'], fontsize=11)
                # if i == nz - 1:
                xl = zeta_short.get(z, z)
                if '[' in xl:
                    xl = xl.rsplit('[', 1)[0].rstrip(' ')
                ax.set_xlabel(xl, fontsize=9)
                if j == 0:
                    ax.set_ylabel(r'$\rho$ [-]', fontsize=9)
                ax.grid(alpha=.3)
        fig.suptitle(
            r'Predictors of predictability (dashed: OLS)',
            fontsize=12, y=0.995)
        plt.tight_layout(rect=(0, 0, 1, 0.98))
        plt.savefig(
            join(SHARED_PLOTS_DIR,
                 f"zeta_vs_rho_predictability_{meltseason_test_name}.png"),
            dpi=300, bbox_inches='tight')
        plt.show()

    # ---- Plot B: Stacked variance decomposition ----
    # When suppressor effects exist (shared < 0), unique ΔR² values sum
    # to more than R²_full. In that case we rescale unique contributions
    # proportionally so the stacked bar sums to exactly 1, and annotate.
    fig, ax = plt.subplots(figsize=(max(3.5, nm * 2), 4))
    xpos = np.arange(nm)
    w = 0.5
    bot = np.zeros(nm)
    # Colorblind-friendly palette: brown, blue, green
    colors_cb = ['#8C510A', '#2166AC', '#5AAE61',
    'tab:orange','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive']  # brown, blue, green, orange, purple, brown, pink, gray, olive
    cmap_u = colors_cb[:len(zeta_cols)]
    # colors_cb = sns.color_palette('colorblind',n_colors=len(zeta_cols))
    # cmap_u = colors_cb[:len(zeta_cols)]
    has_suppressor = []

    for j, m in enumerate(mps):
        r = vp_results[m]
        raw_uniq = {z: max(0, r['unique'][z]['dR2']) for z in zeta_cols}
        s_uniq = sum(raw_uniq.values())
        suppressor = r['shared'] < 0

        if suppressor and s_uniq > 0:
            # Rescale unique contributions so they sum to R²_full
            scale = r['R2_full'] / s_uniq
        else:
            scale = 1.0
        has_suppressor.append(suppressor)

        for i, z in enumerate(zeta_cols):
            val = raw_uniq[z] * scale
            ax.bar(xpos[j], val, w, bottom=bot[j],
                   color=cmap_u[i], ec='k', lw=.5,
                   label=f'{zeta_short.get(z, z)}'
                   if j == 0 else '')
            bot[j] += val

        # Shared (only if positive)
        if not suppressor:
            sh_val = max(0, r['shared'])
            ax.bar(xpos[j], sh_val, w, bottom=bot[j],
                   color='lightgray', ec='k', lw=.5, hatch='//',
                   label='Shared' if j == 0 else '')
            bot[j] += sh_val

        # Unexplained
        ux_val = 1 - r['R2_full']
        ax.bar(xpos[j], ux_val, w, bottom=bot[j],
               color='white', ec='k', lw=.5,
               label='Unexplained' if j == 0 else '')

    # Mark suppressor MPs
    # for j, (m, supp) in enumerate(zip(mps, has_suppressor)):
    #     if supp:
    #         ax.annotate('suppressor\n(rescaled)',
    #                     xy=(xpos[j], vp_results[m]['R2_full']),
    #                     xytext=(xpos[j], vp_results[m]['R2_full'] + 0.08),
    #                     ha='center', fontsize=7, fontstyle='italic',
    #                     color='red',
    #                     arrowprops=dict(arrowstyle='->', color='red',
    #                                     lw=0.8))

    ax.set_xticks(xpos)
    ax.set_xticklabels(labs, fontsize=11)
    ax.set_ylabel('Proportion of variance')
    ax.set_ylim(0, 1.12) 
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.set_title('Variance Decomposition', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=.3)
    # plt.tight_layout()
    plt.savefig(join(SHARED_PLOTS_DIR,
                     f"vp_decomposition_{meltseason_test_name}.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # ---- Plot B2: Pie variance decomposition (same data as Plot B) ----
    fig, axes = plt.subplots(1, nm, figsize=(max(5.5, 4.8 * nm), 4.8),
                             squeeze=False)
    axes = axes[0]
    legend_map = {}
    for i, z in enumerate(zeta_cols):
        legend_map[zeta_short.get(z, z)] = cmap_u[i]
    legend_map['Shared'] = 'lightgray'
    legend_map['Unexplained'] = 'white'

    for ax, m in zip(axes, mps):
        r = vp_results[m]
        raw_uniq = {z: max(0, r['unique'][z]['dR2']) for z in zeta_cols}
        s_uniq = sum(raw_uniq.values())
        suppressor = r['shared'] < 0
        scale = (r['R2_full'] / s_uniq) if (suppressor and s_uniq > 0) else 1.0

        vals = []
        pie_keys = []
        for i, z in enumerate(zeta_cols):
            v = raw_uniq[z] * scale
            if v > 0:
                vals.append(v)
                pie_keys.append(zeta_short.get(z, z))

        if not suppressor:
            sh_val = max(0, r['shared'])
            if sh_val > 0:
                vals.append(sh_val)
                pie_keys.append('Shared')

        ux_val = max(0, 1 - r['R2_full'])
        if ux_val > 0:
            vals.append(ux_val)
            pie_keys.append('Unexplained')

        pie_colors = [legend_map[k] for k in pie_keys]

        if sum(vals) > 0:
            ax.pie(
                vals,
                colors=pie_colors,
                startangle=90,
                counterclock=False,
                autopct=lambda p: f'{p:.1f}%' if p >= 4 else '',
                pctdistance=0.72,
                wedgeprops=dict(edgecolor='k', linewidth=0.5),
                textprops=dict(fontsize=8)
            )
        else:
            ax.text(0.5, 0.5, 'No variance\ndecomposition',
                    ha='center', va='center', fontsize=9)
        title = r['label'] + ('\n(suppressor: unique rescaled)' if suppressor else '')
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=legend_map[k],
                      edgecolor='k', linewidth=0.5)
        for k in legend_map.keys()
    ]
    fig.legend(legend_handles, list(legend_map.keys()),
               loc='center left', bbox_to_anchor=(1.01, 0.5),
               fontsize=8, frameon=True, title='Components')

    fig.suptitle('Variance Decomposition (Pie View)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(join(SHARED_PLOTS_DIR,
                     f"vp_decomposition_pie_{meltseason_test_name}.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # ---- Plot C: LASSO coefficients ----
    fig, ax = plt.subplots(figsize=(max(3.5, nm * 2), 3.5))
    xpos = np.arange(len(zeta_cols))
    w = .8 / nm
    for i, MP in enumerate(mps):
        coefs = [vp_results[MP]['lasso_coefs'].get(z, 0) for z in zeta_cols]
        ax.bar(xpos + (i - nm/2 + .5) * w, coefs, w,
               label=vp_results[MP]['label'], alpha=.8, ec='k', lw=.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([zeta_short.get(z, z) for z in zeta_cols],
                       rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Standardized LASSO coef.')
    ax.axhline(0, c='k', lw=.5)
    ax.legend(fontsize=8)
    ax.set_title('LASSO Coefficients (standardized predictors)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(join(SHARED_PLOTS_DIR,
                     f"vp_lasso_{meltseason_test_name}.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # ---- Summary table ----
    rows = []
    for MP in mps:
        r = vp_results[MP]
        row = {'MP': r['label']}
        for z in zeta_cols:
            row[f'Marg R² {z}'] = r['marginal'][z]['R2']
            row[f'Uniq ΔR² {z}'] = r['unique'][z]['dR2']
        row.update({'Full R²': r['R2_full'], 'Adj R²': r['R2_adj'],
                    'LOGO-CV R² (catchment)': r['cv_logo'],
                    'LASSO R²': r['lasso_r2'],
                    'Shared': r['shared']})
        rows.append(row)
    vp_summary_df = pd.DataFrame(rows)
    print("\n" + "="*80 + "\nVARIANCE PARTITIONING SUMMARY\n" + "="*80)
    print(vp_summary_df.round(3).to_string(index=False))
    vp_summary_df.to_csv(
        join(ROOTDIR, 'aux_data',
             f"variance_partitioning_summary_{meltseason_test_name}.csv"),
        index=False)
    print(f"\nSummary saved.")

print("\n" + "="*80)
print("END OF VARIANCE PARTITIONING ANALYSIS")
print("="*80)

#%%

asymmetry_results = []

for self in LOA_objects.values():
    for pair in pair_selection:
        if 'beta' in pair:
            continue
        metric1, metric2 = pair.split('__')
        print(metric1, metric2)
        

        # Loop through all years and all self instances
        for year in range(self.START_YEAR, self.END_YEAR + 1):
            # Get ranks for both metrics (lower rank = better performance)
            if 'KGE' in metric1 or 'NSE' in metric1 or 'R2' in metric1 or 'PFE' in metric1:
                df1 = self.metrics[year][metric1].rank(ascending=False)
            else:
                df1 = self.metrics[year][metric1].rank(ascending=True)
            
            if 'KGE' in metric2 or 'NSE' in metric2 or 'R2' in metric2 or 'PFE' in metric2:
                df2 = self.metrics[year][metric2].rank(ascending=False)
            else:
                df2 = self.metrics[year][metric2].rank(ascending=True)
            
            # Mean ranks among best 10%; A = log2(mean1|best2 / mean2|best1)
            # Centered at 0; reciprocal mean-rank ratios → +/- equal |A|
            n_best = int(0.1 * len(df1))
            
            best_df2_idx = df2.nsmallest(n_best).index
            mean_df1_given_best_df2 = df1.loc[best_df2_idx].mean()
            
            best_df1_idx = df1.nsmallest(n_best).index
            mean_df2_given_best_df1 = df2.loc[best_df1_idx].mean()
            
            asymmetry_ratio = np.log2(
                mean_df1_given_best_df2 / mean_df2_given_best_df1)
            
            # Store results
            asymmetry_results.append({
                'QSWE_pair': pair,
                'metric1': metric1,
                'metric2': metric2,
                'year': year,
                'basin': self.BASIN,
                'EXP_ID': self.experiment_name,
                'asymmetry_ratio': asymmetry_ratio,
                'mean_metric1_given_best_metric2': mean_df1_given_best_df2,
                'mean_metric2_given_best_metric1': mean_df2_given_best_df1
            })


            print(f"Asymmetry A (log2): {asymmetry_ratio:.3f}")
            print(f"Mean({metric1}|best {metric2}): {mean_df1_given_best_df2:.3f}")
            print(f"Mean({metric2}|best {metric1}): {mean_df2_given_best_df1:.3f}")
            
            # if self.experiment_name == 'D0d' and pair == 'SWE_melt_NSE__NSE':
            if self.experiment_name == 'D0b' and pair == 'melt_sum_APE__Qmean_meltseason2_APE':

                if year ==2014:
                    metric1_name= translate_SWE_metric_name(metric1)
                    metric2_name= translate_Q_metric_name(metric2)
                    metric1_name = 'Melt volume bias'
                    metric2_name = 'Q volume bias'
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,4))
                    # Plot actual values
                    actual_metric1 = self.metrics[year][metric1]
                    actual_metric2 = self.metrics[year][metric2]
                    sns.scatterplot(x = actual_metric1, y = actual_metric2, alpha = 0.5, color = 'black', ax=ax1)
                    #plot squares for the best 10% of df2 and df1
                    # sns.scatterplot(x = actual_metric1.loc[best_df2_idx], y = actual_metric2.loc[best_df2_idx], alpha = 0.5, color = 'red',marker = 's', ax=ax1)
                    # sns.scatterplot(x = actual_metric1.loc[best_df1_idx], y = actual_metric2.loc[best_df1_idx], alpha = 0.5, color = 'blue',marker = 's', ax=ax1)
                    # ax1.set_title(f"{metric1_name} vs {metric2_name} (Values) \n Asymmetry ratio: {asymmetry_ratio:.3f}")
                    ax1.set_xlabel(f"{metric1_name}")
                    ax1.set_ylabel(f"{metric2_name}")
                    # ax1.set_xlim(0.5,1)
                    # ax1.set_ylim(0.5,1) 
                    # ax1.grid(alpha = 0.5)
                    
                    # Plot ranks
                    sns.scatterplot(x = df1, y = df2, alpha = 0.5, color = 'black', ax=ax2)
                    #plot squares for the best 10% of df2 and df1
                    # sns.scatterplot(x = df1.loc[best_df2_idx], y = df2.loc[best_df2_idx], alpha = 0.5, color = 'red',marker = 's', ax=ax2)
                    # sns.scatterplot(x = df1.loc[best_df1_idx], y = df2.loc[best_df1_idx], alpha = 0.5, color = 'blue',marker = 's', ax=ax2)
                    # ax2.set_title(f"{metric1_name} vs {metric2_name} (Ranks) \n Asymmetry ratio: {asymmetry_ratio:.3f}")
                    ax2.set_xlabel(f"{metric1_name} (rank)")
                    ax2.set_ylabel(f"{metric2_name} (rank)")
                    # ax2.grid(alpha = 0.5)

                    labels = ['All simulations', f'Best 10% {metric2_name}', f'Best 10% {metric1_name}']
                    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=7),
                            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=7),
                            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=7)]
                    fig.legend(handles, labels, loc='upper right')
                    correlation = self.metrics[year][metric1].corr(self.metrics[year][metric2], method='spearman')
                    plt.suptitle(f"{self.experiment_name} {year} \n A = {asymmetry_ratio:.3f} \n Correlation: {correlation:.3f}",y=1.01)
                    plt.tight_layout()
                    
                    plt.savefig(join(SHARED_PLOTS_DIR, f"asymmetry_scatter_{self.experiment_name}_{year}_{pair}.png"),
                    dpi = 300, bbox_inches = 'tight')

                    #make a seperate plot showing just the second subplot scatter without the blue and red points 
                    fontsize = 14
                    fig2, ax2 = plt.subplots(1,1,figsize=(4,4))
                    sns.scatterplot(x = df2, y = df1, alpha = 0.5, 
                    color = 'black', ax=ax2,label = 'Ensemble members')
                    ax2.plot([0,500],[0,500],linestyle = 'dashed',color = 'black')
                    ax2.set_xlabel(f"Streamflow skill metric rank",fontsize = fontsize)
                    ax2.set_ylabel(f"Snow skill metric rank",fontsize = fontsize)
                    # ax2.grid(alpha = 0.5)
                    ax2.legend(loc = 'lower right',fontsize = fontsize)
                    plt.savefig(join(SHARED_PLOTS_DIR, f"asymmetry_scatter_rank_{self.experiment_name}_{year}_{pair}.png"),
                    dpi = 300, bbox_inches = 'tight')

                # plt.savefig(join(SHARED_PLOTS_DIR, f"asymmetry_scatter_rank_{self.experiment_name}_{year}_{pair}.png"),

                # dpi = 300, bbox_inches = 'tight')


# Convert to DataFrame
asymmetry_df = pd.DataFrame(asymmetry_results)

asymmetry_df['Experiment'] = asymmetry_df['EXP_ID']
asymmetry_df['QSWE_pair'] = pd.Categorical(asymmetry_df['QSWE_pair'], categories=pair_selection, ordered=True) 
asymmetry_df['Experiment'] = pd.Categorical(asymmetry_df['Experiment'], categories=EXPERIMENT_NAMES, ordered=True)

asymmetry_df.sort_values(['QSWE_pair', 'Experiment'], inplace=True)

#%%
def asymmetry_pair_across_settings(asymmetry_df, pair_selection,
palette = 'colorblind', figsize = None, dodge = 0.75,fontsize = 12,legend_loc = (1.11,0.3),legend_fontsize = 12):
    """ This function plots asymmetry ratio for multiple experiments 
    for a selection of QSWE metric pairs
    """
    setting_count = asymmetry_df['EXP_ID'].nunique()
    if figsize is None:
        fig = plt.figure(figsize = (5,2*len(pair_selection)))
    else:
        fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    sns.stripplot(asymmetry_df, hue="Experiment", y='QSWE_pair',
                x='asymmetry_ratio', orient='h',
                dodge=True if dodge is None else dodge, palette=palette,alpha =0.3,
                legend=False)
    # asymmetry_selection_means = asymmetry_df.groupby(['pair','exp_id'])['asymmetry_ratio'].mean().reset_index()
    sns.pointplot(asymmetry_df, y='QSWE_pair', hue = 'Experiment', x='asymmetry_ratio', orient='h',
            palette=palette,errorbar = 'ci',dodge = dodge, #dodge gives a division by zero error? 
            markersize=7,marker = 'D',linestyle='',
                legend=True)
    print(len(pair_selection)==3)
    if len(pair_selection) == 3:
        #######Ticks just one the left 
        left_labels = [r'$MP_{fit}$',r'$MP_{volume}$',r'$MP_{timing}$'        ]
        # left_labels = ['fit','Volume','Timing']
        ax.set_yticklabels(left_labels, fontsize=fontsize)
        # ax.set_ylabel('SWE-Q metric pair')
        ax.legend(loc = legend_loc, fontsize=legend_fontsize)
    elif len(pair_selection) == 2:
        yticklabels = [label.get_text() for label in ax.get_yticklabels()]
        left_labels = ['Daily','Seasonal']
        ax.set_yticklabels(left_labels, fontsize=fontsize)
        ax.legend(loc = legend_loc, fontsize=legend_fontsize,ncols = 2,
        title = 'Catchment',title_fontsize = fontsize)
        ax.set_ylabel('')


    ########Ticks on both sides  
    else:           
    # Create twin axis for right side labels
        ax2 = ax.twinx()

        # Get current tick labels and positions
        yticks = ax.get_yticks()
        yticklabels = [label.get_text() for label in ax.get_yticklabels()]

        # Split labels
        left_labels = [translate_SWE_metric_name(label.split('__')[0]) for label in yticklabels]
        right_labels = [translate_Q_metric_name(label.split('__')[1]) for label in yticklabels]

        # Set labels
        ax.set_yticklabels(left_labels, fontsize=fontsize)
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(right_labels, fontsize=fontsize)
        ax.set_ylabel('SWE metric',loc = 'bottom', fontsize=fontsize)
        ax2.set_ylabel('Q metric',loc = 'bottom', fontsize=fontsize)
        ax2.set_ylim(ax.get_ylim())
        ax.legend(loc = legend_loc, fontsize=legend_fontsize)

    ax.axvline(0,linestyle ='dashed',color = 'black')
    ax.grid(alpha = 0.3)
    ax.set_xlabel(r'A [-]', fontsize=fontsize)
    ax.set_title(f'Skill transfer \n asymmetry',fontsize = fontsize+1)
    # ax.set_ylabel('Q-SWE performance metric pair', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    if len(pair_selection) > 3:
        ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_xlim(-1, 1)  
    plt.savefig(join(SHARED_PLOTS_DIR, f"asymmetry_ratios_across_experiments_{len(pair_selection)}.png"),
                dpi = 300, bbox_inches = 'tight')
    plt.savefig(join(SHARED_PLOTS_DIR, f"asymmetry_ratios_across_experiments_{len(pair_selection)}.svg"),
                bbox_inches = 'tight')
    plt.show()


def asymmetry_pair_boxenplot_compact(
    asymmetry_df,
    pair_selection,
    figsize=None,
    fontsize=10,
    legend_loc=(1.01, 0.5),
    legend_fontsize=9
):
    """Compact summary: one asymmetry boxenplot per QSWE pair over all rows."""
    if figsize is None:
        figsize = (max(4.5, 0.7 * len(pair_selection)), 2.8)

    plot_df = asymmetry_df.copy()
    plot_df['QSWE_pair'] = pd.Categorical(
        plot_df['QSWE_pair'], categories=pair_selection, ordered=True
    )
    plot_df = plot_df.sort_values('QSWE_pair')

    def _lighten_color(color, amount):
        base_rgb = np.array(mcolors.to_rgb(color))
        white = np.array([1.0, 1.0, 1.0])
        return tuple((1 - amount) * base_rgb + amount * white)

    pair_color_map = {}
    for pair in pair_selection:
        if '__' in pair:
            swe_m, q_m = pair.split('__', 1)
            base_color = PAPER_SWE_BASE_COLORS.get(swe_m, '#4c4c4c')
            lighten = PAPER_Q_LIGHTEN.get(q_m, 0.22)
            pair_color_map[pair] = _lighten_color(base_color, lighten)
        else:
            pair_color_map[pair] = '#808080'

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxenplot(
        data=plot_df,
        y='QSWE_pair',
        x='asymmetry_ratio',
        hue='QSWE_pair',
        palette=pair_color_map,
        dodge=False,
        linewidth=0.8,
        k_depth='proportion',
        ax=ax
    )
    auto_legend = ax.get_legend()
    if auto_legend is not None:
        auto_legend.remove()

    ax.axvline(0, linestyle='dashed', color='black', linewidth=0.9)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlabel(r'A [-]', fontsize=fontsize)
    ax.set_ylabel('')
    ax.set_xlim(-1, 1)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.tick_params(axis='x', labelsize=fontsize)
    # ax.set_title('Skill Transfer Asymmetry Ratio', fontsize=fontsize + 1)

    clean_labels = []
    bold_flags = []
    for pair in pair_selection:
        bold = pair in (
            'SWE_melt_NSE__NSE_meltseason2',
            'melt_sum_APE__Qmean_meltseason2_APE',
        )
        if '__' in pair:
            swe_m, q_m = pair.split('__', 1)
            # if pair == 'SWE_melt_NSE__NSE_meltseason2':
            #     suffix = r'($\rho_\mathrm{daily})$'
            # elif pair == 'melt_sum_APE__Qmean_meltseason2_APE':
            #     suffix = r'($\rho_\mathrm{seasonal})$'
            # else:
            suffix = ''
            clean_labels.append(
                f"{PAPER_METRIC_NAME_MAP.get(swe_m, swe_m)}/"
                f"{PAPER_METRIC_NAME_MAP.get(q_m, q_m)} {suffix}"
            )
        else:
            clean_labels.append(PAPER_METRIC_NAME_MAP.get(pair, pair))
        bold_flags.append(bold)

    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(ax.get_yticks())
    ax_right.set_yticklabels(clean_labels, fontsize=legend_fontsize)
    for tick_lbl, want_bold in zip(ax_right.get_yticklabels(), bold_flags):
        if want_bold:
            tick_lbl.set_fontweight('bold')
    ax_right.tick_params(axis='y', right=True, labelright=True, length=0, pad=4)
    ax_right.set_ylabel('')
    ax_right.grid(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.savefig(join(SHARED_PLOTS_DIR,
                f"asymmetry_boxen_QSWE_pair_selection_{len(pair_selection)}.png"),
                dpi=300, bbox_inches='tight')
    plt.savefig(join(SHARED_PLOTS_DIR,
                f"asymmetry_boxen_QSWE_pair_selection_{len(pair_selection)}.svg"),
                bbox_inches='tight')
    plt.show()
# asymmetry_pair_across_settings(asymmetry_df, pair_selection)

# Define experiment groups to plot
experiment_groups = {
    'Catchment_rainfall': ['Rain-free', 'Shallow soil', 'No soil', 'Deep soil', 'Default', 'Double rain'],
    'Uncertainty': ['Default', 'Spatial', 'Temporal'], 
    'Misc': ['Default', 'Tcorr', 'Runoff parameters','correction'],
    'Minimum':['Default','Rain-free','Double rain']
}

asymmetry_df_reduced = asymmetry_df[asymmetry_df['QSWE_pair'].isin(reduced_pair_selection)].reset_index(drop=True)

# Plot Catchment_rainfall experiments
asymmetry_subset = asymmetry_df_reduced#[asymmetry_df_reduced['Experiment'].str.contains('|'.join(experiment_groups['Catchment_rainfall']), na=False)]
asymmetry_subset['Experiment'] = pd.Categorical(
    asymmetry_subset['Experiment'], 
    categories=asymmetry_subset['Experiment'].astype('str').unique(),
    ordered=True
)
asymmetry_subset['QSWE_pair'] = pd.Categorical(asymmetry_subset['QSWE_pair'], 
categories=reduced_pair_selection, ordered=True)
palette1 = sns.color_palette('tab20b',n_colors=12)[::-1]

asymmetry_pair_across_settings(asymmetry_subset, reduced_pair_selection, 
palette=palette_x, figsize=(5,10),legend_loc = (1.05,0.2),fontsize = 18,legend_fontsize = 14)
asymmetry_pair_boxenplot_compact(asymmetry_df,pair_selection,figsize = (5,2.5))



#%% HEre i want
def plot_example_discordant_qswe_metrics(
    loa_obj,
    metric1,
    metric2,
    year=None,
    save_plots=True,
    outdir=None,
    swe_months=(2, 9),
    q_months=(2, 9),
    max_good_rank=None,
):
    """
    Plot discordant SWE-Q metric behavior for one station-year.

    metric1 should be a SWE metric and metric2 a Q metric. The function creates:
      - SWE ensemble + reference (top-left)
      - Q ensemble + reference (bottom-left)
      - Scatter of metric1 vs metric2 (top-right)
      - Scatter of rank(metric1) vs rank(metric2) (bottom-right)

    Highlighted members:
      - Best metric1
      - Best metric2
      - Good metric1 / poor metric2
      - Good metric2 / poor metric1

    Parameters
    ----------
    max_good_rank : int or None
        Optional upper rank threshold for selecting discordant "good" members.
        Example: max_good_rank=250 means only members with rank <= 250 can be
        considered "good" on the corresponding metric.
    """
    def _is_higher_better(metric_name):
        name = str(metric_name).lower()
        better_high = ['nse', 'kge', 'r2', 'correlation', 'pfe']
        return any(tag in name for tag in better_high)

    def _pick_discordant_and_best(df, metric1_name, metric2_name, ascending1, ascending2, good_rank_max=None):
        rank1 = df[metric1_name].rank(ascending=ascending1, method='min')
        rank2 = df[metric2_name].rank(ascending=ascending2, method='min')
        rank_diff = rank2 - rank1

        idx_best_metric1 = df[metric1_name].idxmax() if not ascending1 else df[metric1_name].idxmin()
        idx_best_metric2 = df[metric2_name].idxmax() if not ascending2 else df[metric2_name].idxmin()

        if good_rank_max is None:
            idx_good_metric1_bad_metric2 = rank_diff.idxmax()
            idx_good_metric2_bad_metric1 = rank_diff.idxmin()
        else:
            good1_pool = rank_diff[rank1 <= good_rank_max]
            good2_pool = rank_diff[rank2 <= good_rank_max]

            # Fallback to unconstrained selection when threshold is too strict.
            idx_good_metric1_bad_metric2 = good1_pool.idxmax() if not good1_pool.empty else rank_diff.idxmax()
            idx_good_metric2_bad_metric1 = good2_pool.idxmin() if not good2_pool.empty else rank_diff.idxmin()

        return rank1, rank2, rank_diff, idx_good_metric1_bad_metric2, idx_good_metric2_bad_metric1, idx_best_metric1, idx_best_metric2

    metric1_label = translate_Q_metric_name(metric1)
    metric2_label = translate_SWE_metric_name(metric2, keep_suffix=True)

    #iff AGG in the SWE metric name, remove it and its hyhpen behind it
    if 'AGG' in metric2_label:
        metric2_label = metric2_label.replace('AGG-', '')
        metric2_label = metric2_label.replace('_', ' ')
    if metric2_label == "Melt":
        metric2_label = "Melt-NSE"
    if metric1_label == "Meltseason NSE":
        metric1_label = "Q-NSE"    # ------------------------------------------------------------------
    # Step 1: pick year (lowest abs Spearman) if not provided
    # ------------------------------------------------------------------
    if year is None:
        records = []
        for yr in sorted(loa_obj.metrics.keys()):
            if metric1 not in loa_obj.metrics[yr] or metric2 not in loa_obj.metrics[yr]:
                continue
            pair = loa_obj.metrics[yr][[metric1, metric2]].dropna()
            if len(pair) < 5:
                continue
            r = pair[metric1].corr(pair[metric2], method='spearman')
            records.append({'year': int(yr), 'spearman_r': r, 'n': len(pair)})

        if not records:
            raise ValueError("No valid years found (need >=5 non-null metric pairs).")
        corr_df = pd.DataFrame(records)
        corr_df['abs_r'] = corr_df['spearman_r'].abs()
        best_row = corr_df.loc[corr_df['abs_r'].idxmin()]
        sel_year = int(best_row['year'])
        print(f"Selected year: {sel_year} (Spearman r={best_row['spearman_r']:.3f})")
    else:
        sel_year = int(year)
        if sel_year not in loa_obj.metrics:
            raise ValueError(f"Year {sel_year} not found in loa_obj.metrics.")

    # ------------------------------------------------------------------
    # Step 2: identify best + discordant runs for selected year
    # ------------------------------------------------------------------
    if metric1 not in loa_obj.metrics[sel_year] or metric2 not in loa_obj.metrics[sel_year]:
        raise KeyError(f"metric1='{metric1}' or metric2='{metric2}' not found for year {sel_year}.")
    sy_metrics = loa_obj.metrics[sel_year][[metric1, metric2]].dropna().copy()
    if sy_metrics.empty:
        raise ValueError("No non-null metric rows in selected year.")

    asc1 = not _is_higher_better(metric1)
    asc2 = not _is_higher_better(metric2)

    (
        sy_metrics['rank1'],
        sy_metrics['rank2'],
        sy_metrics['rank_diff'],
        idx_good1_bad2,
        idx_good2_bad1,
        idx_best1,
        idx_best2,
    ) = _pick_discordant_and_best(sy_metrics, metric1, metric2, asc1, asc2, max_good_rank)

    # ------------------------------------------------------------------
    # Step 3: pull SWE/Q ensembles and references from LOA object
    # ------------------------------------------------------------------
    if sel_year not in loa_obj.SWE:
        raise ValueError(f"Year {sel_year} is not available in loa_obj.SWE.")

    swe_obs = loa_obj.SWEobs.mean(dim=['lat', 'lon']).to_pandas()
    swe_sims = loa_obj.SWE[sel_year].mean(dim=['lat', 'lon']).to_pandas()
    if isinstance(swe_sims, pd.DataFrame) and 'spatial_ref' in swe_sims.columns:
        swe_sims = swe_sims.drop(columns='spatial_ref')

    q_sims = loa_obj.Q * 86400 * 1000 / (loa_obj.E.dem_area * 1e6)
    q_obs = loa_obj.Qobs * 86400 * 1000 / (loa_obj.E.dem_area * 1e6)
    q_obs = q_obs.squeeze()

    swe_start = pd.Timestamp(sel_year, swe_months[0], 1)
    swe_end = pd.Timestamp(sel_year, swe_months[1], 1) + pd.offsets.MonthEnd(1)
    q_start = pd.Timestamp(sel_year, q_months[0], 1)
    q_end = pd.Timestamp(sel_year, q_months[1], 1) + pd.offsets.MonthEnd(1)
    swe_timeslice = slice(swe_start, swe_end)
    q_timeslice = slice(q_start, q_end)

    swe_obs = swe_obs.loc[swe_timeslice]
    swe_sims = swe_sims.loc[swe_timeslice]
    q_sims = q_sims.loc[q_timeslice]
    q_obs = q_obs.loc[q_timeslice]

    # Daily rainfall (liquid) for inverted twin axis on Q panel
    rainfall = None
    try:
        scalars_files = glob.glob(join(loa_obj.SYNDIR, '*Synthetic_obs*.csv'))
        if not scalars_files:
            raise FileNotFoundError(f"No Synthetic_obs csv in {loa_obj.SYNDIR}")
        swe_full = loa_obj.SWEobs.mean(dim=['lat', 'lon']).to_pandas()
        scalars = pd.read_csv(scalars_files[0], index_col=0, parse_dates=True).loc[swe_full.index]
        snowmelt = (-swe_full.diff()).clip(lower=0)
        rainfall = (scalars['rainfallplusmelt'] - snowmelt).loc[q_timeslice].astype(float)
        rainfall = rainfall.clip(lower=0)
    except Exception as exc:
        print(f"Skipping rainfall twin axis ({loa_obj.experiment_name}, {sel_year}): {exc}")
        rainfall = None

    # Melt season boundaries (10th percentile - 1 week, 90th percentile + 1 month)
    melt_period_start = None
    melt_period_end = None
    try:
        melt_period_start = pd.to_datetime(
            zetas['melt_period_start'].loc[sel_year, loa_obj.experiment_name]
        )
        melt_period_end = pd.to_datetime(
            zetas['melt_period_end'].loc[sel_year, loa_obj.experiment_name]
        )
    except Exception:
        # Keep plotting even if melt period metadata is unavailable.
        melt_period_start = None
        melt_period_end = None

    # Keep only members present in both metric table and simulation columns.
    run_ids = [rid for rid in sy_metrics.index if rid in swe_sims.columns and rid in q_sims.columns]
    if len(run_ids) < 5:
        raise ValueError("Too few common run IDs between metrics and SWE/Q simulations.")
    sy_metrics = sy_metrics.loc[run_ids]
    swe_sims = swe_sims.loc[:, run_ids]
    q_sims = q_sims.loc[:, run_ids]

    # Recompute after filtering to shared runs
    (
        sy_metrics['rank1'],
        sy_metrics['rank2'],
        sy_metrics['rank_diff'],
        idx_good1_bad2,
        idx_good2_bad1,
        idx_best1,
        idx_best2,
    ) = _pick_discordant_and_best(sy_metrics, metric1, metric2, asc1, asc2, max_good_rank)

    # ------------------------------------------------------------------
    # Step 4: figure layout (2x2, with square scatter panels on right)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1.0], wspace=0.24, hspace=0.14)
    ax_swe = fig.add_subplot(gs[0, 0])
    ax_q = fig.add_subplot(gs[1, 0], sharex=ax_swe)
    ax_scatter_vals = fig.add_subplot(gs[0, 1])
    ax_scatter_ranks = fig.add_subplot(gs[1, 1])
    # Keep scatter panels square and centered in their cells.
    ax_scatter_vals.set_box_aspect(1)
    ax_scatter_vals.set_anchor('C')
    ax_scatter_ranks.set_box_aspect(1)
    ax_scatter_ranks.set_anchor('C')

    for col in swe_sims.columns:
        ax_swe.plot(swe_sims.index, swe_sims[col], color='gray', alpha=0.15, linewidth=0.6)
    for col in q_sims.columns:
        ax_q.plot(q_sims.index, q_sims[col], color='gray', alpha=0.15, linewidth=0.6)

    styles = {
        'good1_bad2': (f'Good {metric1_label}, poor {metric2_label}', 'tab:blue'),
        'good2_bad1': (f'Good {metric2_label}, poor {metric1_label}', 'tab:red'),
        'best1': (f'Best {metric1_label}', 'tab:green'),
        'best2': (f'Best {metric2_label}', 'tab:orange'),
    }
    selected_runs = {
        'good1_bad2': idx_good1_bad2,
        'good2_bad1': idx_good2_bad1,
        'best1': idx_best1,
        'best2': idx_best2,
    }

    for key, run_id in selected_runs.items():
        label, color = styles[key]
        if run_id in swe_sims.columns:
            ax_swe.plot(swe_sims.index, swe_sims[run_id], color=color, linewidth=2.4, label=label)
        if run_id in q_sims.columns:
            ax_q.plot(q_sims.index, q_sims[run_id], color=color, linewidth=2.4, label=label)

    ax_swe.plot(swe_obs.index, swe_obs.values, color='black', linestyle='--', linewidth=2.2, label='Reference')
    ax_q.plot(q_obs.index, q_obs.values, color='black', linestyle='--', linewidth=2.2, label='Reference')

    if pd.notna(melt_period_start) and pd.notna(melt_period_end):
        for ax in (ax_swe, ax_q):
            ax.axvline(melt_period_start, color='black', linestyle='--', linewidth=1.2, alpha=0.9)
            ax.axvline(melt_period_end, color='black', linestyle='--', linewidth=1.2, alpha=0.9)
        ax_q.text(
            melt_period_start, 0.98, 'Meltseason start',
            transform=ax_q.get_xaxis_transform(), rotation=90,
            va='top', ha='right', fontsize=12, color='black'
        )
        ax_q.text(
            melt_period_end+pd.Timedelta(days=2), 0.98, 'Meltseason end',
            transform=ax_q.get_xaxis_transform(), rotation=90,
            va='top', ha='left', fontsize=12, color='black'
        )

    ax_swe.set_ylabel('SWE (mm)')
    ax_swe.set_title(f"{loa_obj.experiment_name} SWE")
    ax_swe.grid(True, alpha=0.3)
    ax_swe.tick_params(labelbottom=False)
    ax_swe.legend(fontsize=10, loc='upper left', ncols=2)

    ax_q.set_ylabel('Q (mm/day)')
    ax_q.set_title(f"{loa_obj.experiment_name} Streamflow")
    ax_q.grid(True, alpha=0.3)
    plt.setp(ax_q.get_xticklabels(), rotation=35, ha='right')

    # Tight Feb–Nov window (no date-axis padding on either side)
    ax_swe.set_xlim(swe_start, swe_end)
    ax_q.set_xlim(q_start, q_end)
    ax_swe.margins(x=0)
    ax_q.margins(x=0)

    # Inverted twin axis: daily rainfall bars hanging from the top
    handles_q, labels_q = ax_q.get_legend_handles_labels()
    if rainfall is not None and rainfall.notna().any():
        ax_pr = ax_q.twinx()
        ax_pr.invert_yaxis()
        ax_pr.bar(
            rainfall.index.values,
            rainfall.values,
            width=1.0,
            align='center',
            color='tab:blue',
            alpha=0.35,
            edgecolor='none',
            label='Rainfall',
            zorder=1,
        )
        ax_pr.set_ylabel('Rainfall (mm/day)')
        ax_pr.set_ylim(ax_pr.get_ylim()[0] * (7 / 2), 0)
        handles_pr, labels_pr = ax_pr.get_legend_handles_labels()
        handles_q = handles_q + handles_pr
        labels_q = labels_q + labels_pr
    # ax_q.legend(handles_q, labels_q, fontsize=10, loc='best', ncols=1)

    # Top-right: actual metric values (SWE skill on x, Q skill on y)
    ax_scatter_vals.scatter(
        sy_metrics[metric2], sy_metrics[metric1], color='gray', alpha=0.45, s=30, label='Ensemble members'
    )
    for key, run_id in selected_runs.items():
        label, color = styles[key]
        ax_scatter_vals.scatter(
            sy_metrics.loc[run_id, metric2],
            sy_metrics.loc[run_id, metric1],
            color=color,
            s=145,
            zorder=5,
            edgecolors='black',
            linewidths=1.0,
            label=label
        )

    r_val = sy_metrics[metric1].corr(sy_metrics[metric2], method='spearman')
    # Asymmetry A = log2(mean(SWE ranks | best Q) / mean(Q ranks | best SWE))
    # with metric1=Q, metric2=SWE (matches SWE__Q asymmetry definition elsewhere).
    n_best = max(1, int(0.1 * len(sy_metrics)))
    best_metric1_idx = sy_metrics['rank1'].nsmallest(n_best).index
    best_metric2_idx = sy_metrics['rank2'].nsmallest(n_best).index
    mean_metric2_given_best_metric1 = sy_metrics.loc[best_metric1_idx, 'rank2'].mean()
    mean_metric1_given_best_metric2 = sy_metrics.loc[best_metric2_idx, 'rank1'].mean()
    asymmetry_ratio = np.log2(
        mean_metric2_given_best_metric1 / mean_metric1_given_best_metric2)

    ax_scatter_vals.set_xlabel(metric2_label)
    ax_scatter_vals.set_ylabel(metric1_label)
    ax_scatter_vals.set_title(f"{metric2_label}/{metric1_label}")
    ax_scatter_vals.grid(True, alpha=0.3)
    ax_scatter_vals.legend(fontsize=10, loc='best')
    if 'nse' in metric2.lower():
        ax_scatter_vals.set_xlim(0, 1)
    if 'nse' in metric1.lower():
        ax_scatter_vals.set_ylim(0, 1)

    # Bottom-right: metric ranks (SWE rank on x, Q rank on y)
    ax_scatter_ranks.scatter(
        sy_metrics['rank2'], sy_metrics['rank1'], color='gray', alpha=0.45, s=30
    )
    for key, run_id in selected_runs.items():
        _, color = styles[key]
        ax_scatter_ranks.scatter(
            sy_metrics.loc[run_id, 'rank2'],
            sy_metrics.loc[run_id, 'rank1'],
            color=color,
            s=145,
            zorder=5,
            edgecolors='black',
            linewidths=1.0,
        )

    ax_scatter_ranks.set_xlabel(f"Rank ({metric2_label})")
    ax_scatter_ranks.set_ylabel(f"Rank ({metric1_label})")
    ax_scatter_ranks.set_xlim(0, 500)
    ax_scatter_ranks.set_ylim(0, 500)
    ax_scatter_ranks.grid(True, alpha=0.3)
    ax_scatter_ranks.text(
        0.98, 0.02,
f'$\\boldsymbol{{\\rho={r_val:.3f}}}$\n'
f'$\\boldsymbol{{A={asymmetry_ratio:.3f}}}$',
        transform=ax_scatter_ranks.transAxes,
        ha='right', va='bottom', fontsize=13,
        linespacing=1.3,
    )

    # Panel labels: a/b top-right, c/d top-left
    label_kw = dict(fontsize=14, zorder=10)
    ax_swe.text(0.98, 0.98, 'a)', transform=ax_swe.transAxes, ha='right', va='top', **label_kw)
    ax_q.text(0.98, 0.98, 'b)', transform=ax_q.transAxes, ha='right', va='top', **label_kw)
    ax_scatter_vals.text(0.02, 0.98, 'c)', transform=ax_scatter_vals.transAxes, ha='left', va='top', **label_kw)
    ax_scatter_ranks.text(0.02, 0.98, 'd)', transform=ax_scatter_ranks.transAxes, ha='left', va='top', **label_kw)

    plt.tight_layout()
    # Re-apply spacing after tight_layout (it can reset gaps).
    # Extra wspace leaves room for the rainfall twin-axis ylabel.
    gs.update(wspace=0.24, hspace=0.14)
    ax_swe.set_xlim(swe_start, swe_end)
    ax_q.set_xlim(q_start, q_end)

    plt.show()

    if save_plots:
        if outdir is None:
            outdir = join(SHARED_PLOTS_DIR)
        os.makedirs(outdir, exist_ok=True)
        fn = f"discordant_qswe_{metric1}_vs_{metric2}_{loa_obj.BASIN}_{sel_year}.png"
        outpath = join(outdir, fn)
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {outpath}")

    plt.close(fig)
    return loa_obj.BASIN, sel_year

LOA_objects['ABBA_14'].load_prior_data(load_SWE=True)

plot_example_discordant_qswe_metrics(
        loa_obj=LOA_objects['ABBA_14'],
        metric1='Qmean_meltseason_APE',     # Q metric
        metric2='melt_sum_APE',          # SWE metric
        year=2010,                          # or None for auto year
        save_plots=True,
    )
plot_example_discordant_qswe_metrics(
        loa_obj=LOA_objects['ABBA_14'],
        metric1='NSE_meltseason2',     # Q metric
        metric2='SWE_melt_NSE',          # SWE metric
        year=2010,                          # or None for auto year
        save_plots=True,
        max_good_rank=100,
    )
# for year in range(2001,2022):
#     plot_example_discordant_qswe_metrics(
#             loa_obj=LOA_objects['ABBA_14'],
#             metric1='NSE_meltseason2',     # Q metric
#             metric2='SWE_melt_NSE',          # SWE metric
#             year=year,                          # or None for auto year
#             save_plots=True,
#             max_good_rank=100,
#     )

# LOA_objects['ABBA_234_WUS_McKenzie'].load_prior_data(load_SWE=True)
# plot_example_discordant_qswe_metrics(
#         loa_obj=LOA_objects['ABBA_234_WUS_McKenzie'],
#         metric1='NSE_meltseason2',     # Q metric
#         metric2='SWE_melt_NSE',          # SWE metric
#         year=2010,                          # or None for auto year
#         save_plots=True,
#     )

# LOA_objects['ABBA_214_WUS_Merced_HI'].load_prior_data(load_SWE=True)
# plot_example_discordant_qswe_metrics(
#         loa_obj=LOA_objects['ABBA_214_WUS_Merced_HI'],
#         metric1='NSE_meltseason2',     # Q metric
#         metric2='SWE_melt_NSE',          # SWE metric
#         year=2010,                          # or None for auto year
#         save_plots=True,
#     )
#%%
# LOA_objects['ABBA_234_WUS_McKenzie'].load_prior_data(load_SWE=True)
# for year in range(2001,2022):
#     plot_example_discordant_qswe_metrics(
#         loa_obj=LOA_objects['ABBA_234_WUS_McKenzie'],
#         metric1='Qmean_meltseason_APE',     # Q metric
#         metric2='melt_sum_APE',          # SWE metric
#         year=year,                          # or None for auto year
#         save_plots=True,
#         max_good_rank=100,
#         swe_months=(1, 10),
#         q_months=(1, 10),
#     )
#%% Basic Q and SWE plots 

for self in LOA_objects.values():
    break
    # for y in range(2001,2022):
    #     print(self.experiment_name, len(self.Q.columns))   
        
        
         # if not self.BASIN == 'Bear':
    #     continue
    # for year in [YEAR]:
    if 'D0.5c' in self.experiment_name:
        self.load_prior_data(load_SWE=True)
        for y in range(2001,2022):
            # try: 
            print(self.EXP_ID, len(self.SWE[y].data_vars))
            # except:
            #     print(y)
    else:
        continue

    for year in [2010]:
        # Get all simulation IDs from available data
        all_ids = list(self.SWE[year].data_vars)

        #get the top 1% performing runs 
        top_1 = self.metrics[year]['NSE_meltseason2'].nlargest(int(0.1*len(self.metrics[year][SWE_target_metric]))).index
        
        #SWE plot
        SWEobs = self.SWEobs.sel(time = slice(f'{year-1}-10-01',f'{year}-09-30'))
        SWE_sims_all = self.SWE[year]

        SWEobs_bands = self.E.swe2bands(SWEobs,bands = list(self.elev_bands))
        SWE_sims_all_bands = self.E.swe2bands(SWE_sims_all,bands = list(self.elev_bands))
        bands  = list(SWEobs_bands.keys())

        axi, axj = 2,int(np.ceil(len(bands)/2))
        f1, axes = plt.subplots(axi,axj, figsize=(2 * len(bands), 8))
        axes = axes.flatten()
        for i, band in enumerate(bands):
            ax1 = axes[i]
            obsplot = SWEobs_bands[band].plot(ax=ax1, color='black', 
                                                label='SWEobs', zorder=100,
                                                linestyle = '--')
            SWE_all_t = SWE_sims_all_bands[band].drop(columns='spatial_ref', errors='ignore')
            simplot = SWE_all_t.plot(ax=ax1, color='tab:blue', label='_noLegend', legend=False,
                                    alpha=0.1, linestyle='-')
            ax1.legend([])
            ax1.set_title(f"{band}")
            ax1.grid()

        # handles = [Line2D([0], [0], color='black', linestyle='--')]
        handles = [Line2D([0], [0], color='black', linestyle='--'), 
                   Line2D([0], [0], color='tab:blue', alpha=0.5)]
        labels = ['Observed', 'Simulations']
        ax1.legend(handles, labels)
        ax1.set_title(f"{band}")
        f1.suptitle(f'{self.experiment_name}')
        plt.show()

        #make catchment-wide plot 
        SWEobs = self.SWEobs.sel(time = slice(f'{year-1}-10-01',f'{year}-09-30'))
        SWE_sims_all = self.SWE[year]
        SM = SWEMetrics(dem = self.E.dem, elev_bands = self.elev_bands)
        SWEobs_catchment = SM.calc_sum2d(SWEobs).to_pandas()
        SWE_sims_all_catchment = SM.calc_sum2d(SWE_sims_all).to_pandas()
        
        f1,ax1 = plt.subplots(1,1,figsize=(10,6))
        obsplot = SWEobs_catchment.plot(ax=ax1, color='black', 
                                        label='SWEobs', zorder=100,
                                        linestyle = '--')
        SWE_all_t = SWE_sims_all_catchment.drop(columns='spatial_ref', errors='ignore')
        simplot = SWE_all_t.plot(ax=ax1, color='tab:blue', label='_noLegend', legend=False,
                                alpha=0.1, linestyle='-')
        # SWE_all_t_top1 = SWE_all_t.loc[:,top_1]
        # SWE_all_t_top1.plot(ax=ax1, color='tab:red', label='_noLegend', legend=False,
        #                         alpha=0.3, linestyle='-')
        ax1.fill_between(SWE_all_t.index, SWE_all_t.min(axis=1), SWE_all_t.max(axis=1),
                        color='tab:blue', alpha=0.2, zorder=0)
        ax1.legend(['Observed', 'Simulations'])
        ax1.set_title(f'{self.experiment_name} \n Catchment-wide SWE')
        ax1.grid()
        plt.show()

        #make melt plot
        def swe3d_to_melt(swe3d):
                swe1d = swe3d.sum(dim = ['lat','lon'])
                swe3d_melt = swe1d.diff('time')
                swe3d_melt = xr.where(swe3d_melt > 0, 0, swe3d_melt)*-1
                melt1d = swe3d_melt.to_pandas()
                if isinstance(melt1d, pd.DataFrame) and 'spatial_ref' in melt1d.columns:
                    melt1d = melt1d.drop(columns = 'spatial_ref')
                return melt1d
        def swe3d_to_snowfall(swe3d):
            swe1d = swe3d.sum(dim = ['lat','lon'])
            swe3d_snowfall = swe1d.diff('time')
            swe3d_snowfall = xr.where(swe3d_snowfall < 0, 0, swe3d_snowfall)
            snowfall1d = swe3d_snowfall.to_pandas()
            if isinstance(snowfall1d, pd.DataFrame) and 'spatial_ref' in snowfall1d.columns:
                snowfall1d = snowfall1d.drop(columns = 'spatial_ref')
            return snowfall1d
        
        SWEobs_melt = swe3d_to_melt(SWEobs)
        SWE_sims_all_melt = swe3d_to_melt(SWE_sims_all)

        f1,ax1 = plt.subplots(1,1,figsize=(10,6))
        # Plot total melt
        SWEobs_melt.plot(ax=ax1, color='black', label='SWEobs', linestyle='--',zorder=102)
        SWE_sims_all_melt.plot(ax=ax1, color='tab:blue', 
                                    label='_noLegend', legend=None,
                                alpha = 0.3)
        SWE_sims_all_melt_top1 = SWE_sims_all_melt.loc[:,top_1]
        SWE_sims_all_melt_top1.plot(ax=ax1, color='tab:red', label='_noLegend', legend=False,
                                alpha=0.3, linestyle='-')
        ax1.fill_between(SWE_sims_all_melt.index, 
                            SWE_sims_all_melt.min(axis=1), 
                            SWE_sims_all_melt.max(axis=1), 
                            color='tab:blue', alpha=0.2)
        ax1.set_xlim(f"{year}-03-01", f"{year}-07-30")
        ax1.set_title(f'{self.experiment_name} \n Total Meltwater Production')
        ax1.set_ylabel('SWE dz')
        ax1.grid()
        ax1.legend(['Observed', 'Simulations'])

        #plot snow melt per grid cell 

        def swe1d_to_melt(swe1d):
            swe1d_melt = swe1d.diff()
            # swe1d_melt = np.where(swe1d_melt > 0, 0, swe1d_melt)*-1
            swe1d_melt = swe1d_melt.mask(swe1d_melt > 0, 0) * -1
            return swe1d_melt
        
        # Plot melt for each grid cell (only show a few to avoid too many plots)
        for ilat in range(min(3, len(SWEobs.lat))):  # Limit to first 3 lat indices
            for ilon in range(min(3, len(SWEobs.lon))):  # Limit to first 3 lon indices
                SWEobs_melt_cell =  swe1d_to_melt(SWEobs.isel(lat = ilat, lon = ilon).to_pandas())
                if np.all(np.isnan(SWEobs_melt_cell)):
                    continue
                SWE_sims_all_melt_cell = swe1d_to_melt(SWE_sims_all.isel(lat = ilat, lon = ilon).to_pandas().drop(columns = ['spatial_ref','lat','lon']))

                f1,ax1 = plt.subplots(1,1,figsize=(10,6))
                SWEobs_melt_cell.plot(ax=ax1, color='black', label='SWEobs', linestyle='--',zorder=102)
                SWE_sims_all_melt_cell.plot(ax=ax1, color='tab:blue', alpha = 0.3,
                                            label='Simulations', linestyle='-')
                ax1.fill_between(SWE_sims_all_melt_cell.index, SWE_sims_all_melt_cell.min(axis = 1), SWE_sims_all_melt_cell.max(axis = 1), color='tab:blue', alpha=0.2)
                ax1.set_xlim(f"{year}-03-01", f"{year}-07-30")
                ax1.set_title(f'{self.experiment_name} \n Meltwater Production at {ilat}, {ilon}')
                ax1.set_ylabel('SWE dz')
                ax1.grid()
                ax1.legend(['Observed', 'Simulations'])
                plt.show()

        #snowfall plot
        # SWEobs_snowfall = swe3d_to_snowfall(SWEobs)
        # SWE_sims_all_snowfall = swe3d_to_snowfall(SWE_sims_all)

        # f1,ax1 = plt.subplots(1,1,figsize=(10,6))
        # # Plot total snowfall
        # SWEobs_snowfall.plot(ax=ax1, color='black', label='SWEobs', linestyle='--',zorder=102)
        # SWE_sims_all_snowfall.plot(ax=ax1, color='tab:blue',
        #                             label='_noLegend', legend=None,
        #                             alpha = 0.3)
        # ax1.fill_between(SWE_sims_all_snowfall.index,
        #                 SWE_sims_all_snowfall.min(axis=1),
        #                 SWE_sims_all_snowfall.max(axis=1),
        #                 color='tab:blue', alpha=0.2)
        # ax1.set_xlim(f"{year}-03-01", f"{year}-07-30")
        # ax1.set_title('Total Snowfall Production')
        # ax1.set_ylabel('SWE dz')
        # ax1.grid()
        # ax1.legend(['Observed', 'Simulations'])
        plt.show()

        #2D plots 
        # Calculate melt NSE for all simulations
        melt_NSE_2d = [SM.calc_melt_nse_grid(SWEobs, self.SWE[year][id]) for id in all_ids]
        melt_NSE_2d_joint = xr.concat(melt_NSE_2d, dim = 'id')
        melt_NSE_2d_median = melt_NSE_2d_joint.median(dim = 'id')
        
        # f1,ax1 = plt.subplots(1,1,figsize=(8,6))
        # melt_NSE_2d_median.plot(ax=ax1,cmap = 'RdBu')
        # ax1.set_title(f'{self.experiment_name} \n Median Melt NSE across all simulations')

        # Streamflow plot
        timeslice = slice(f'{year}-02-01',f'{year}-09-30')
        QQall = self.Q.loc[timeslice]
        QQmin = QQall.min(axis = 1)
        QQmax = QQall.max(axis = 1)
        QQobs = self.Qobs.loc[timeslice]

        f1,ax1 = plt.subplots(figsize = (7,4))
        QQall.plot(ax = ax1, color = 'tab:blue',
                    alpha = 0.3, legend = None,
                    zorder = 50,
                    linewidth = 0.5)
        QQobs.plot(ax = ax1,color= 'black', linestyle = '--', legend = None,zorder = 100)
        ax1.fill_between(QQall.index, QQmin, QQmax, color = 'tab:blue', alpha = 0.2)
        QQall_top1 = QQall.loc[:,top_1]
        QQall_top1.plot(ax=ax1, color='tab:red', label='_noLegend', legend=False,
                        alpha=0.3, linestyle='-',zorder =100)
        handles = [Line2D([0], [0], color='black', linestyle='--'),
                   Line2D([0], [0], color='tab:blue', alpha=0.3)]
        labels = ['Observed', 'Simulations']
        ax1.set_title(f'{self.experiment_name} Streamflow')
        ax1.legend(handles, labels)
        ax1.grid()
        ax1.set_ylabel('Q [m3/s]')
        ax1.set_ylim(bottom = 0, top = ax1.get_ylim()[1]/2)
        ax1.set_xlabel(None)
        plt.savefig(join(self.FIGDIR, f'QQ_SWE_2D_{year}.png'),
                    bbox_inches='tight', dpi=300)

        # Poster plot: Catchment-wide melt and Q
        timeslice = slice(f'{year}-03-01', f'{year}-09-30')
        
        # Calculate catchment-wide melt in mm/day
        SWEobs_melt = swe1d_to_melt(SWEobs.mean(dim = ['lat','lon']).to_pandas())
        SWE_sims_all_melt = swe1d_to_melt(SWE_sims_all.mean(dim = ['lat','lon']).to_pandas())
        
        # Convert Q to mm/day
        Q_all_mmd = self.Q.loc[timeslice] * 86400 * 1000 / (self.E.dem_area * 1e6)
        Q_obs_mmd = self.Qobs.loc[timeslice] * 86400 * 1000 / (self.E.dem_area * 1e6)
        Q_obs_mmd = Q_obs_mmd.squeeze()
        
        # Filter melt to same time slice
        melt_obs = SWEobs_melt.loc[timeslice]
        melt_sims = SWE_sims_all_melt.loc[timeslice]
        
        # Create figure with two subplots
        # Set adjustable font size
        fontsize = 16
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4,6), sharex=True)
        
        # Top panel: Melt
        melt_obs.plot(ax=ax1, color='black', linestyle='--', 
        linewidth=3, label='Observed', zorder=100)
        # melt_sims.plot(ax=ax1, color='tab:blue', alpha=0.2,
        #  linewidth=0.5, legend=False, label='_noLegend')
        for i,col in enumerate(melt_sims.columns):
            if i == 0:
                melt_sims[col].plot(ax=ax1, color='tab:blue', alpha=0.2,
                 linewidth=0.5, legend=False, label='Simulations')
            else:
                melt_sims[col].plot(ax=ax1, color='tab:blue', alpha=0.2,
                 linewidth=0.5, legend=False, label='_nolegend_')
        # ax1.fill_between(melt_sims.index, melt_sims.min(axis=1), melt_sims.max(axis=1), 
        #                 color='tab:blue', alpha=0.15, zorder=0)
        ax1.set_ylabel('Melt (mm/day)', fontsize=fontsize)
        ax1.tick_params(labelsize=0)  # Remove tick labels
        # ax1.grid(True, alpha=0.3, linestyle='--')
        handles = [Line2D([0], [0], color='black', linestyle='--'), 
                   Line2D([0], [0], color='tab:blue', alpha=0.8)]
        labels = ['Reference', 'Ensemble']
        ax1.legend(handles, labels, fontsize=16, frameon=False)
        # ax1.legend(fontsize=16, frameon=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Bottom panel: Q
        # Q_all_mmd.plot(ax=ax2, color='tab:blue',
        #  alpha=0.2, linewidth=0.5, legend=False, label='_noLegend')
        for i,col in enumerate(Q_all_mmd.columns):
            if i == 0:
                Q_all_mmd[col].plot(ax=ax2, color='tab:blue',
                 alpha=0.2, linewidth=0.5, legend=False, label='Simulations')
            else:
                Q_all_mmd[col].plot(ax=ax2, color='tab:blue',
                 alpha=0.2, linewidth=0.5, legend=False, label='_nolegend_')
        # ax2.fill_between(Q_all_mmd.index, Q_all_mmd.min(axis=1), Q_all_mmd.max(axis=1), 
        #                  color='tab:blue', alpha=0.15, zorder=0)
        Q_obs_mmd.plot(ax=ax2, color='black', linestyle='--', linewidth=3, label='Observed', zorder=100)
        ax2.set_ylabel('Q (mm/day)', fontsize=fontsize)
        # ax2.set_xlabel('Date', fontsize=20, fontweight='bold')
        ax2.set_xlabel(None)
        ax2.tick_params(labelsize=0)  # Remove tick labels
        # ax2.grid(True, alpha=0.3, linestyle='--')
        handles = [Line2D([0], [0], color='black', linestyle='--'), 
                   Line2D([0], [0], color='tab:blue', alpha=0.8)]
        labels = ['Reference', 'Ensemble']
        # ax2.legend(handles, labels, fontsize=fontsize-1)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(DateFormatter('%b'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')
        
        plt.tight_layout()
        plt.savefig(join(self.FIGDIR, f'melt_Q_poster_{year}.png'),
                    bbox_inches='tight', dpi=300)
        plt.show()

        # SWE plot: Catchment-wide SWE
        timeslice_swe = slice(f'{year}-01-01', f'{year}-06-30')
        
        # Filter SWE to same time slice
        swe_obs = SWEobs.mean(dim = ['lat','lon']).to_pandas().loc[timeslice_swe]
        swe_sims = SWE_sims_all.mean(dim = ['lat','lon']).to_pandas().loc[timeslice_swe]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4,6), sharex=True)
        
        # Top panel: SWE
        swe_obs.plot(ax=ax1, color='black', linestyle='--', 
        linewidth=3, label='Observed', zorder=100)
        for i,col in enumerate(swe_sims.columns):
            if i == 0:
                swe_sims[col].plot(ax=ax1, color='tab:blue', alpha=0.2,
                 linewidth=0.5, legend=False, label='Simulations')
            else:
                swe_sims[col].plot(ax=ax1, color='tab:blue', alpha=0.2,
                 linewidth=0.5, legend=False, label='_nolegend_')
        ax1.set_ylabel('SWE (mm)', fontsize=fontsize)
        ax1.tick_params(labelsize=0)  # Remove tick labels
        handles = [Line2D([0], [0], color='black', linestyle='--'), 
                   Line2D([0], [0], color='tab:blue', alpha=0.8)]
        labels = ['Reference', 'Ensemble']
        ax1.legend(handles, labels, fontsize=16, frameon=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Bottom panel: Q
        Q_all_mmd_swe = self.Q.loc[timeslice_swe] * 86400 * 1000 / (self.E.dem_area * 1e6)
        Q_obs_mmd_swe = self.Qobs.loc[timeslice_swe] * 86400 * 1000 / (self.E.dem_area * 1e6)
        Q_obs_mmd_swe = Q_obs_mmd_swe.squeeze()
        
        for i,col in enumerate(Q_all_mmd_swe.columns):
            if i == 0:
                Q_all_mmd_swe[col].plot(ax=ax2, color='tab:blue',
                 alpha=0.2, linewidth=0.5, legend=False, label='Simulations')
            else:
                Q_all_mmd_swe[col].plot(ax=ax2, color='tab:blue',
                 alpha=0.2, linewidth=0.5, legend=False, label='_nolegend_')
        Q_obs_mmd_swe.plot(ax=ax2, color='black', linestyle='--', linewidth=3, label='Observed', zorder=100)
        ax2.set_ylabel('Q (mm/day)', fontsize=fontsize)
        ax2.set_xlabel(None)
        ax2.tick_params(labelsize=0)  # Remove tick labels
        handles = [Line2D([0], [0], color='black', linestyle='--'), 
                   Line2D([0], [0], color='tab:blue', alpha=0.8)]
        labels = ['Reference', 'Ensemble']
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(DateFormatter('%b'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')
        
        plt.tight_layout()
        plt.savefig(join(self.FIGDIR, f'SWE_Q_poster_{year}.png'),
                    bbox_inches='tight', dpi=300)
        plt.show()

        


# %%
#for one pair of QSWE metrics,

# %%

#collect all streamflow 
Q_list = []
for self in LOA_objects.values():
    # if not self.BASIN == 'Dischma':
    #     continue
    if not self.experiment_name in ['D0a','D0d','D2a','D1c','D2d']:
        continue
    #check if any of the entries of experiment_groups['Catchment_rainfall'] is in self.experiment_name
    Q_list.append(self.Q.median(axis=1).to_frame(name = self.experiment_name))

    # if any(group in self.experiment_name for group in experiment_groups['Catchment_rainfall']):
    #     Q_list.append(self.Q.median(axis=1).to_frame(name = self.experiment_name))

Q_df = pd.concat(Q_list, axis=1)
#convert to mm/day
Q_df = Q_df * 86400 * 1000 / (LOA_objects['ABBA_14'].E.dem_area*1e6)

S = SwissStation('Dischma')
S.read_station(startyear=self.START_YEAR, endyear=self.END_YEAR,
discharge_dir = join(ROOTDIR,"Discharge_data"))
Q_obs = S.obs['obs']
Q_obs = Q_obs * 86400 * 1000 / (LOA_objects['ABBA_14'].E.dem_area*1e6)

#%%
year = 2003
daterange = pd.date_range(start = f"{year}-04-01", end = f"{year}-08-30")
# Streamflow colors: same scheme as elsewhere — tab20b reversed, one color per
# experiment in EXPERIMENT_CONFIG insertion order (D0a, D0d, …).
_n_exp = len(EXPERIMENT_CONFIG)
_tab20b_rev = sns.color_palette('tab20b', n_colors=16)[::-1]
_color_by_experiment = {'D0a':_tab20b_rev[0], 
'D0d':_tab20b_rev[3],
 'D2a':_tab20b_rev[12], 
 'D2d':_tab20b_rev[15],
 'D1c':_tab20b_rev[10]}
# _color_by_experiment = dict(zip(EXPERIMENT_CONFIG.values(), _tab20b_rev))
# linestyles = ['solid', 'solid', 'dashed', 'dashed','dashed']
linestyles = ['solid','solid','solid','solid','solid']
alphas = [1, 1, 1,1,1]
sizes = [2.5,  2.5,2, 2, 2]
labels = {'D1c':'D1c (Default)',
          'D0d': 'D0d',
          'D2a': 'D2a',
          'D2d': 'D2d',
          'D0a': 'D0a',
          }
cols_order = ['D1c', 'D0d', 'D2a', 'D2d', 'D0a']
cols_order = ['D0a','D0d','D1c','D2a','D2d']
colors = [_color_by_experiment[c] for c in cols_order]
# Upper panel: melt vs rain — similar earth tones, distinct from tab20b streamflow colors
COLOR_SNOWMELT_INPUT = 'tab:red'#'#6d4c3d'
COLOR_RAINFALL_INPUT = 'tab:blue'#'#b08968'


def _rainfall_daily_bars(ax, rainfall_series, daterange, color, label, alpha=0.85):
    """One bar per day; draw before snowmelt line so the line stays on top."""
    s = rainfall_series.loc[daterange].astype(float)
    x = date2num(s.index.to_pydatetime())
    ax.bar(x, s.values, width=1.0, align='center', color=color, alpha=alpha,
           label=label, edgecolor='none', zorder=2)


# Get D0a object for the upper subplot
D1a_obj = None
for obj in LOA_objects.values():
    if obj.experiment_name == 'D1c':
        D1a_obj = obj
        break

precip = self.meteo_base['pr'].mean(dim = ['lat','lon']).to_pandas()
#%%
# Create figure with two subplots
f1, (ax_top, ax1) = plt.subplots(2, 1, figsize=(7,5), sharex=True, 
                                 gridspec_kw={'height_ratios': [1, 1]})
plt.subplots_adjust(hspace = 0.05)
# Upper subplot: D0a melt, rain and SWE time series
if D1a_obj is not None:
    # Get SWE data for D0a (median across simulations)
    swe_d0a = D1a_obj.SWEobs.mean(dim=['lat','lon']).to_pandas()
    
    # Calculate snowmelt (negative change in SWE = snowmelt)
    snowmelt_d0a = -swe_d0a.diff()
    snowmelt_d0a = snowmelt_d0a.where(snowmelt_d0a > 0, 0)
    
    # Get rfmelt from scalars file
    scalars_file = glob.glob(join(D1a_obj.SYNDIR, '*Synthetic_obs*.csv'))[0]
    scalars = pd.read_csv(scalars_file, index_col=0, parse_dates=True).loc[swe_d0a.index]
    rfmelt_d0a = scalars['rainfallplusmelt']
    
    # Calculate rainfall
    rainfall_d0a = rfmelt_d0a - snowmelt_d0a
    total_precip = scalars['Pmean']
    total_precip_d0a = total_precip.loc[daterange]
    # Plot SWE on twin axis (right)
    ax0_twin = ax_top.twinx()
    swe_d0a.loc[daterange].plot(ax=ax0_twin, 
    color='grey', label='SWE', alpha=1)
    ax0_twin.set_ylabel('SWE (mm)')
    ax0_twin.set_ylim(bottom = 0)
    ax_top.grid(alpha=0.5)
    # ax_top.plot(total_precip_d0a, color='black', label='Total Precipitation', alpha=1)
    
    # Rainfall: daily bars; snowmelt: line on top
    _rainfall_daily_bars(ax_top, rainfall_d0a, daterange, COLOR_RAINFALL_INPUT, 'Rainfall', alpha=0.9)
    snowmelt_d0a.loc[daterange].plot(ax=ax_top, color=COLOR_SNOWMELT_INPUT, label='Snowmelt', zorder=3)
    
    ax_top.set_ylabel('Catchment inputs \n (mm/day)')
    ax_top.grid(alpha=0.5)
    
    # Combine legends from both axes
    lines1, labels1 = ax_top.get_legend_handles_labels()
    lines2, labels2 = ax0_twin.get_legend_handles_labels()
    ax_top.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # ax_top.set_title('D0a: SWE, Melt, and Rainfall')

# Lower subplot: Streamflow comparison (same order and styling as frame plots)
for i, col in enumerate(cols_order):
    Q_df.loc[daterange, col].plot(ax=ax1,
        color=colors[i], alpha=alphas[i], label=labels[col],
        linestyle=linestyles[i], linewidth=sizes[i])
Q_obs.loc[daterange].plot(ax = ax1, linestyle = 'solid',
color = 'black',linewidth = 2,label = 'Observed')

ax_top.set_title(f"Synthetic Dischma variants")
# ax1.set_ylabel('Q (m3/s)')
ax1.set_ylabel('Streamflow \n (mm/day)')
ax1.grid(alpha = 0.5)
ax1.set_ylim(bottom = 0,top = 50)
ax1.legend(ncols = 3)

plt.tight_layout()
plt.savefig(join(SHARED_PLOTS_DIR, f"Q_timeseries_catchment_rainfall_variations_{self.BASIN}_{year}_forpaper.png"),
 dpi = 300, bbox_inches = 'tight')












#%%
# for frame in range(1, 5):
#     cols_frame = cols_order[:frame]
#     f1, (ax_top, ax1) = plt.subplots(2, 1, figsize=(7,4), sharex=True,
#                                      gridspec_kw={'height_ratios': [1, 1]})
#     plt.subplots_adjust(hspace=0.05)
#     # Upper subplot: D1a melt, rain and SWE time series
#     if D1a_obj is not None:
#         swe_d0a = D1a_obj.SWEobs.mean(dim=['lat','lon']).to_pandas()
#         snowmelt_d0a = -swe_d0a.diff()
#         snowmelt_d0a = snowmelt_d0a.where(snowmelt_d0a > 0, 0)
#         scalars_file = glob.glob(join(D1a_obj.SYNDIR, '*Synthetic_obs*.csv'))[0]
#         scalars = pd.read_csv(scalars_file, index_col=0, parse_dates=True).loc[swe_d0a.index]
#         rfmelt_d0a = scalars['rainfallplusmelt']
#         rainfall_d0a = rfmelt_d0a - snowmelt_d0a
#         ax0_twin = ax_top.twinx()
#         swe_d0a.loc[daterange].plot(ax=ax0_twin, color='black', label='SWE', alpha=0.5)
#         ax0_twin.set_ylabel('SWE (mm)')
#         _rainfall_daily_bars(ax_top, rainfall_d0a, daterange, COLOR_RAINFALL_INPUT, 'Rainfall', alpha=0.9)
#         snowmelt_d0a.loc[daterange].plot(ax=ax_top, color=COLOR_SNOWMELT_INPUT, label='Snowmelt', zorder=3)
#         ax_top.set_ylabel('Catchment inputs \n (mm/day)')
#         lines1, labels1 = ax_top.get_legend_handles_labels()
#         lines2, labels2 = ax0_twin.get_legend_handles_labels()
#         ax_top.legend(lines1 + lines2, labels1 + labels2, loc='center left')
#     # Lower subplot: plot in legend order (D1c, D0c, D1a, D0a), hide lines not in this frame (same legend every time)
#     for i, col in enumerate(cols_order):
#         Q_df.loc[daterange, col].plot(ax=ax1,
#             color=colors[i], alpha=alphas[i], label=labels[col],
#             linestyle=linestyles[i], linewidth=sizes[i])
#     # Hide lines that are not in this frame; set their legend text to white so they don't show
#     for j in range(frame, 4):
#         ax1.lines[j].set_visible(False)
#     ax1.set_ylabel('Streamflow \n Q (mm/day)')
#     ax1.set_ylim(bottom=0, top=50)
#     leg = ax1.legend(ncols=2)
#     for j in range(frame, 4):
#         leg.get_texts()[j].set_color('white')
#     plt.tight_layout()
#     plt.savefig(join(SHARED_PLOTS_DIR, f"Q_timeseries_catchment_rainfall_variations_{self.BASIN}_{year}_frame{frame}.png"),
#                 dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close(f1)

# # Same loop but without SWE/twin axis and only 3 Q lines (no fourth)
# cols_order_3 = cols_order[:3]
# for frame in range(1, 4):
#     f1, (ax_top, ax1) = plt.subplots(2, 1, figsize=(7,4), sharex=True,
#                                      gridspec_kw={'height_ratios': [1, 1]})
#     plt.subplots_adjust(hspace=0.05)
#     # Upper subplot: melt and rain only (no SWE, no twin axis)
#     if D1a_obj is not None:
#         swe_d0a = D1a_obj.SWEobs.mean(dim=['lat','lon']).to_pandas()
#         snowmelt_d0a = -swe_d0a.diff()
#         snowmelt_d0a = snowmelt_d0a.where(snowmelt_d0a > 0, 0)
#         scalars_file = glob.glob(join(D1a_obj.SYNDIR, '*Synthetic_obs*.csv'))[0]
#         scalars = pd.read_csv(scalars_file, index_col=0, parse_dates=True).loc[swe_d0a.index]
#         rfmelt_d0a = scalars['rainfallplusmelt']
#         rainfall_d0a = rfmelt_d0a - snowmelt_d0a
#         _rainfall_daily_bars(ax_top, rainfall_d0a, daterange, COLOR_RAINFALL_INPUT, 'Rainfall', alpha=0.9)
#         snowmelt_d0a.loc[daterange].plot(ax=ax_top, color=COLOR_SNOWMELT_INPUT, label='Snowmelt', zorder=3)
#         ax_top.set_ylabel('Catchment inputs \n (mm/day)')
#         ax_top.legend(loc='center left')
#     # Lower subplot: only first 3 Q lines (D1c, D0c, D1a), hide lines not in this frame
#     for i, col in enumerate(cols_order_3):
#         Q_df.loc[daterange, col].plot(ax=ax1,
#             color=colors[i], alpha=alphas[i], label=labels[col],
#             linestyle=linestyles[i], linewidth=sizes[i])
#     for j in range(frame, 3):
#         ax1.lines[j].set_visible(False)
#     ax1.set_ylabel('Streamflow \n Q (mm/day)')
#     ax1.set_ylim(bottom=0, top=50)
#     leg = ax1.legend(ncols=2)
#     for j in range(frame, 3):
#         leg.get_texts()[j].set_color('white')
#     plt.tight_layout()
#     plt.savefig(join(SHARED_PLOTS_DIR, f"Q_timeseries_catchment_rainfall_variations_{self.BASIN}_{year}_noSWE_frame{frame}.png"),
#                 dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close(f1)


# %%
