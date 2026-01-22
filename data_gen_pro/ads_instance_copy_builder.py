# This script is only for 
# "Optimizing_ADS_Bidi_Chenyan_v1" suitable

# ======================================================================================
# TODO ATTENTION: 
# This script must be runned in ADS internal python console, not in external py env !
# ======================================================================================

from keysight.ads import de
from keysight.ads.de import db_uu as db
from keysight.edatoolbox import ads
import keysight.ads.dataset as dataset
from keysight.edatoolbox import util
from pathlib import Path
from IPython.core import getipython

# packages to import data analysis and save
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random 
import h5py



workspace_path = "E:\\personal_Data\\Document of School\\Uni Stuttgart\\Masterarbeit\\Simulation\\ADS\\GaN4EMoBiL_BiDi_GaN_wrk"
library_name  = "GaN4EMoBiL_BiDi_GaN_lib"
design_name   = "Optimizing_ADS_Bidi_Chenyan_mp"    
instance_name_var_mp = "sweep_param_mp"       
param_range   = {                         
    'VOFF': (-4.7, -4),
    'U0': (0.1, 2.2),
    'NS0ACCS': (1e15, 1e20),
    'NFACTOR': (0, 10),
    'ETA0': (0, 1),
    'VSAT': (5e4, 1e7),
    'VDSCALE': (0.5, 1e6),
    'CDSCD': (0, 0.75),
    'LAMBDA': (0, 0.2),
    'MEXPACCD': (0, 12),
    'DELTA': (2, 100),
    'UA': (0, 1e-8),
    'UB': (0, 1e-16),
    'U0ACCS': (0.01, 0.4)
    # ...
}
vgs_bias_param_sweep_name = "Sweep_vgs" 
vds_bias_param_sweep_name = "Sweep_vds"
vgs_bias_simulation_name  = "Output"   
vds_bias_simulation_name  = "Transfer"

num_circuits = 30

# ----------------------------------------------

if de.workspace_is_open():
    de.close_workspace()
        
# Open the workspace
# if (not self.workspace) or (self.cur_workspace_path != workspace_path):
workspace = de.open_workspace(workspace_path)
cur_workspace_path = workspace_path
# Open the library
# if (not self.library) or (self.cur_library_name != library_name):
library = workspace.open_library(lib_name=library_name, mode=de.LibraryMode.SHARED)
cur_library_name = library_name
# Open the design
# if (not self.design) or (self.cur_design_name != design_name):
design = db.open_design((library_name, design_name, "schematic"), db.DesignMode.APPEND)
cur_design_name = design_name

v_mp = design.get_instance(inst_name=instance_name_var_mp)


# ----------------------------------------------

# an init param dict
param_init = {key: str(max(value)) for key, value in param_range.items()}

# 1. create a new VAR to save all sweep parameters
for index in range(1, num_circuits + 1):
    var_mp_iterator_dict = dict(zip([f"{key}_{index}" for key in param_init.keys()], param_init.values()))
    v_mp.vars.update(var_mp_iterator_dict)

# 2. mapping these parameters to the switch instances
# mapping rule:
    # SW1, SW2 -> param group 1
    # SW3, SW4 -> param group 2
    # ...


# for Bidi-directional GaN HEMT
additional_params = ['NS0ACCD', 'MEXPACCS', 'U0ACCD']
additional_params_mapping = ['NS0ACCS', 'MEXPACCD', 'U0ACCS']
for index in range(1, num_circuits + 1):
    sw_name_odd = f'SW{2 * index - 1}'
    sw_name_even = f'SW{2 * index}'
    sw_odd = design.get_instance(inst_name=sw_name_odd)
    sw_even = design.get_instance(inst_name=sw_name_even)
    for key in param_range.keys():
        sw_odd.parameters[key.lower()].value = f"{key}_{index}" 
        sw_even.parameters[key.lower()].value = f"{key}_{index}" 
    for i, key in enumerate(additional_params):
        sw_odd.parameters[key.lower()].value = f"{additional_params_mapping[i]}_{index}" 
        sw_even.parameters[key.lower()].value = f"{additional_params_mapping[i]}_{index}" 


# # for uni-directional GaN HEMT
# additional_params = ['NS0ACCD', 'MEXPACCS', 'U0ACCD']
# additional_params_mapping = ['NS0ACCS', 'MEXPACCD', 'U0ACCS']
# for index in range(1, num_circuits + 1):
#     sw_name = f'SW{index}'
#     sw = design.get_instance(inst_name=sw_name)
#     for key in param_range.keys():
#         sw.parameters[key.lower()].value = f"{key}_{index}" 
#     for i, key in enumerate(additional_params):
#         sw.parameters[key.lower()].value = f"{additional_params_mapping[i]}_{index}" 


design.save_design()

