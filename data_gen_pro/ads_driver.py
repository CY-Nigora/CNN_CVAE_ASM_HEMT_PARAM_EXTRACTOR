# This file is part of the ADS Parameter Fitting project.
# must be used under ADS integrated Python env (A)
# namely, ..\ADS_install_path\tools\python\python.exe --> Python 3.13.2
# TODO: whole script is run in Jupyter because ADS python ADI only
# supports IPython kernel !!!

# packages to build DIR env
import os, json
# set ads dict: HPEESOF_DIR and home director : HOME
os.environ['HPEESOF_DIR'] = 'D:/ADS/install'
os.environ['HOME'] = 'D:/ADS/dir'

# packages to import ADS
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

# packages to multiprocessing
# add current working directory
import sys
cur_path = "E:/personal_Data/Document of School/Uni Stuttgart/Masterarbeit/Code/param_regression/ADS_Parameter_Fitting/IV_param_regression/data_gen_pro"
sys.path.append(cur_path)

# more meaningful data generator based on log-uniform distribution
from log_data_gen import param_random_generator as param_random_generator_log

# necessary class and functions

class PyADS():
    def __init__(self):
        self.cur_workspace_path = None
        self.workspace = None
        self.cur_library_name = None
        self.library = None
        self.cur_design_name = None
        self.design = None

    def close(self):
        if de.workspace_is_open():
            de.close_workspace()

    def create_and_open_an_empty_workspace(self, workspace_path: str):
    # example : workspace_path = "C:/ADS_Python_Tutorials/tutorial1_wrk"
    # Ensure there isn't already a workspace open
        if de.workspace_is_open():
            de.close_workspace()
    
        # Cannot create a workspace if the directory already exists
        if os.path.exists(workspace_path):
            raise RuntimeError(f"Workspace directory already exists: {workspace_path}")
    
        # Create the workspace
        workspace = de.create_workspace(workspace_path)
        # Open the workspace
        workspace.open()
        # Return the open workspace and close when it finished
        return workspace
    
    def create_a_library_and_add_it_to_the_workspace(self, workspace: de.Workspace, library_name: str) -> None:
        #assert workspace.path is not None
        # Libraries can only be added to an open workspace
        assert workspace.is_open
        # We'll create a library in the directory of the workspace
        library_path = workspace.path / library_name
        # Create the library
        de.create_new_library(library_name, library_path)
        # And add it to the workspace (update lib.defs)
        workspace.add_library(library_name, library_path, de.LibraryMode.SHARED)
        lib=workspace.open_library(library_name,library_path,de.LibraryMode.SHARED)
        return lib

    # @timeout_thread_soft(20) # 20s timeout
    def schematic_simulation(self, workspace_path: str, library_name: str, design_name: str, instance_name: str, var_dict: dict, vgs_bias_param_sweep_name: str, vds_bias_param_sweep_name: str, vgs_bias_simulation_name: str, vds_bias_simulation_name: str, verilog_a_dir:str = None) -> None:
        ''' Load Path and files, Edit the design variables, Simulate the design, and return the dataset '''

        # test timeout
        # random_number = random.random()
        # time.sleep(13) if random_number >= 0.5 else time.sleep(0.5)

        # >> Load Path and files
        if not os.path.exists(workspace_path):
            raise RuntimeError(f"Workspace directory doesn't exist: {workspace_path}")
        if de.workspace_is_open():
            de.close_workspace()
        
        # Open the workspace
        # if (not self.workspace) or (self.cur_workspace_path != workspace_path):
        self.workspace = de.open_workspace(workspace_path)
        self.cur_workspace_path = workspace_path
        # Open the library
        # if (not self.library) or (self.cur_library_name != library_name):
        self.library = self.workspace.open_library(lib_name=library_name, mode=de.LibraryMode.SHARED)
        self.cur_library_name = library_name
        # Open the design
        # if (not self.design) or (self.cur_design_name != design_name):
        self.design = db.open_design((library_name, design_name, "schematic"), db.DesignMode.APPEND)
        self.cur_design_name = design_name

        # >> Edit the design variables
        # edit VAR
        v = self.design.get_instance(inst_name=instance_name)
        assert v.is_var_instance
        for var_name in var_dict:
            v.vars[var_name] = var_dict[var_name]
        # Save the design
        self.design.save_design()
        # Simulate the design
        output_dir = os.path.join(self.workspace.path, "output")
        netlist_file = os.path.join(output_dir, "data_gen.ckt")
        output_file =  os.path.join(output_dir, "data_gen.ckt.out")
        # create the simulation output directory
        util.safe_makedirs(output_dir)

        # >> Simulate and return the dataset
        # ipython = getipython.get_ipython()
        # if ipython is None:
        #     print("The remaining portion of the script must be run in an IPython environment. Exiting.")
        #     return
        # capture the netlist in a string
        netlist = self.design.generate_netlist()
        # access to the simulator object to run netlists
        simulator = ads.CircuitSimulator()
        # run the netlist, this will block output
        simulator.run_netlist(netlist, output_dir=output_dir, netlist_file=netlist_file, output_file=output_file, verilog_dir=verilog_a_dir)
        output_data = dataset.open(Path(os.path.join(output_dir, f"{design_name}.ds")))
        
        # >> return data in pandas DataFrame format
        # <class 'pandas.core.frame.DataFrame'>
        data_ids_vds = output_data[f'{vgs_bias_param_sweep_name}.{vgs_bias_simulation_name}.DC'].to_dataframe().reset_index()
        data_ids_vgs = output_data[f'{vds_bias_param_sweep_name}.{vds_bias_simulation_name}.DC'].to_dataframe().reset_index()

        return data_ids_vds, data_ids_vgs
    
    

    def dataset_reshape(self, pd_data_IV: pd.DataFrame, pd_data_gm: pd.DataFrame, IV_dimension: list, gm_dimension: list, var_dict: dict, mp_num: int = 1):
        ''' reshape the dataset into desired input matrix and output vector '''
        IV_row_count = IV_dimension[0] # Vgs
        IV_col_count = IV_dimension[1] # Vds
        gm_row_count = gm_dimension[0] # Vds
        gm_col_count = gm_dimension[1] # Vgs
        y_row = int(len(var_dict)/mp_num)

        output_x_IV = np.empty((mp_num, IV_row_count, IV_col_count),dtype=np.float64)
        output_x_gm = np.empty((mp_num, gm_row_count, gm_col_count),dtype=np.float64)
        output_y = np.empty((mp_num, y_row, 1),dtype=np.float64)
        var_dict2list = np.array(list(var_dict.values())).reshape((-1,1))

        IV_row = pd_data_IV["VGS"].drop_duplicates().sort_values(ascending=False).tolist() # attention here: must be descending order
        gm_row = pd_data_gm["VDS"].drop_duplicates().sort_values(ascending=False).tolist()

        for outer_index in range(mp_num):
            if mp_num > 1:
                ids_index = outer_index + 1
            else:
                ids_index = ''
            for index,row_value in enumerate(IV_row):
                output_x_IV[outer_index, index, :] = pd_data_IV.loc[pd_data_IV['VGS'] == row_value, f"IDS{ids_index}.i"].to_numpy()
            for index,row_value in enumerate(gm_row):
                output_x_gm[outer_index, index, :] = pd_data_gm.loc[pd_data_gm['VDS'] == row_value, f"IDS{ids_index}.i"].to_numpy()
            output_y[outer_index, :, :] = var_dict2list[outer_index * y_row : (outer_index + 1) * y_row]

        # return form (mp_num, goal_row, goal_col)
        return output_x_IV, output_x_gm, output_y
    

def param_random_generator(param_range: dict):
    ''' generate a random parameter set for the HEMT model '''
    # define the parameter range
    # param_range = {
    #     'VOFF': (-1.2, 2.6),
    #     'U0': (0, 2.2),
    #     'NS0ACCS': (1e15, 1e20),
    #     'NFACTOR': (0.1, 5),
    #     'ETA0': (0, 1),
    #     'VSAT': (5e4, 1e7),
    #     'VDSCALE': (0.5, 1e6),
    #     'CDSCD': (1e-5, 0.75),
    #     'LAMBDA': (0, 0.2),
    #     'MEXPACCD': (0.05, 12),
    #     'DELTA': (2, 100)
    # }
    # generate random parameters
    var_dict = {key: str(np.random.uniform(low=val[0], high=val[1])) for key, val in param_range.items()}
    return var_dict

def init_h5_file(h5_path, x_iv_shape, x_gm_shape, y_shape,
                 dtype_x=np.float64, dtype_y=np.float64):
    with h5py.File(h5_path, 'w') as f:
        # X: [num_samples, m, n]
        f.create_dataset(
            'X_iv',
            shape=(0, x_iv_shape[0], x_iv_shape[1]),
            maxshape=(None, x_iv_shape[0], x_iv_shape[1]),
            dtype=dtype_x
        )
        f.create_dataset(
            'X_gm',
            shape=(0, x_gm_shape[0], x_gm_shape[1]),
            maxshape=(None, x_gm_shape[0], x_gm_shape[1]),
            dtype=dtype_x
        )
        # Y: [num_samples, y_len]
        f.create_dataset(
            'Y',
            shape=(0, y_shape[0], 1),
            maxshape=(None, y_shape[0], 1),
            dtype=dtype_y
        )

def append_to_h5(h5_path, x_iv_new, x_gm_new, y_new):
    x_iv_new = np.asarray(x_iv_new, dtype=np.float64)
    x_gm_new = np.asarray(x_gm_new, dtype=np.float64)
    y_new = np.asarray(y_new, dtype=np.float64)

    # make sure all of them are 3-dim : (mp_num, goal_row, goal_col)
    if (y_new.ndim == 3) and (x_iv_new.ndim == 3) and (x_gm_new.ndim == 3):
        pass
    else:
        raise ValueError(f"Output/Transfer/Lable must be 3-dim, but got shape respectively: {x_iv_new.shape, x_gm_new.shape, y_new.shape}")

    with h5py.File(h5_path, 'a') as f:
        ds_x_iv = f['X_iv']
        ds_x_gm = f['X_gm']
        ds_y = f['Y']

        cur_len = ds_x_iv.shape[0]
        new_len = cur_len + x_iv_new.shape[0]

        # 扩展
        ds_x_iv.resize(new_len, axis=0)
        ds_x_gm.resize(new_len, axis=0)
        ds_y.resize(new_len, axis=0)

        # 赋值
        ds_x_iv[cur_len:new_len, :, :] = x_iv_new
        ds_x_gm[cur_len:new_len, :, :] = x_gm_new
        ds_y[cur_len:new_len, :, :] = y_new


# for mp ADS project file with multiple identical circuits
def singel_process_iteration_data_gen2h5(workspace_path: str, 
                                         validate_dict: dict, 
                                         library_name: str, 
                                         design_name: str, 
                                         instance_name: str, 
                                         param_range: dict, 
                                         vgs_bias_param_sweep_name: str, 
                                         vds_bias_param_sweep_name: str, 
                                         vgs_bias_simulation_name: str, 
                                         vds_bias_simulation_name: str, 
                                         data_shape: list, 
                                         iteration_num: int, 
                                         mp_num: int = 1, # number of multi-processing circuits
                                         process_id: int = 1, 
                                         save_path: str = None, 
                                         verilog_a_dir: str = None, 
                                         new_start: bool = True, 
                                         old_stop_index: int = 0, 
                                         dtype_x=np.float64, 
                                         dtype_y=np.float64):
    ''' generate dataset in single process iteration '''
    # create an instance of the PyADS class
    ads_ctrl = PyADS()
    X_iv_shape = data_shape[0]
    X_gm_shape = data_shape[1]
    Y_shape = data_shape[2]

    # init h5 file
    if new_start:
        init_h5_file(f"{save_path}\\dataset_process_{process_id}.h5", X_iv_shape, X_gm_shape, Y_shape)

    for i in range(iteration_num) if new_start else range(old_stop_index, iteration_num):
        while True:
            try:
                start_time = time.time()
                if validate_dict:
                    var_dict = validate_dict
                else:
                    # var_dict = param_random_generator(param_range)
                    var_dict = param_random_generator_log(param_range, mp_num) # use log-uniform distribution based generator
                # if mp_num > 1, which corresponds multiple identical circuits in one project file
                # the returned:
                #        pd_data_vgs_bias & pd_data_gm
                # in which IDS has a index from [1] to [mp_num]
                pd_data_vgs_bias, pd_data_gm = ads_ctrl.schematic_simulation(
                    workspace_path,
                    library_name,
                    design_name,
                    instance_name,
                    var_dict,
                    vgs_bias_param_sweep_name,
                    vds_bias_param_sweep_name,
                    vgs_bias_simulation_name,
                    vds_bias_simulation_name,
                    verilog_a_dir
                ) 
                # return form of dataset_reshape : (mp_num, goal_row, goal_col)
                X_iv, X_gm, y = ads_ctrl.dataset_reshape(pd_data_vgs_bias, pd_data_gm, X_iv_shape, X_gm_shape, var_dict, mp_num)
                end_time = time.time()
                print(f" >> Process {process_id} :: Loop {i+1}/{iteration_num} :: used time: {round(end_time - start_time, 2)} s", file=sys.stderr, flush=True)
                try:
                    append_to_h5(f"{save_path}\\dataset_process_{process_id}.h5", X_iv, X_gm, y)
                except:
                    print(f"[ERROR] Error while appending data in process {process_id} at iteration {i + 1}", file=sys.stderr, flush=True)
                break
            # except TimeoutError:
            #     end_time = time.time()
            #     print(f' >> Process {process_id} :: Loop {i + 1}/{iteration_num} [Failed because Timeout] :: used time:', round(end_time - start_time, 2), 's', file=sys.stderr, flush=True)
            #     time.sleep(3)
            #     ads_ctrl.close()
            #     time.sleep(10)
            #     continue
            except:
                end_time = time.time()
                print(f' >> Process {process_id} :: Loop {i + 1}/{iteration_num} [Failed because cannot converge] :: used time:', round(end_time - start_time, 2), 's', file=sys.stderr, flush=True)
                time.sleep(1)
                ads_ctrl.close()
                time.sleep(2)
                continue