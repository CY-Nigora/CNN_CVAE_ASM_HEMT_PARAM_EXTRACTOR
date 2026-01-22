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
import h5py

# packages to multiprocessing
import multiprocessing as mp
import data_gen_multi_processing as gen
import sys




# necessary class and functions

class PyADS():
    def __init__(self):
        self.HPEESOF_DIR = 'D:/ADS/install'
        self.HOME = 'D:/ADS/dir'
        self.cur_workspace_path = None
        self.workspace = None
        self.cur_library_name = None
        self.library = None
        self.cur_design_name = None
        self.design = None

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

    def schematic_simulation(self, workspace_path: str, library_name: str, design_name: str, instance_name: str, var_dict: dict, param_sweep_name: str, simulation_name: str) -> None:
        ''' Load Path and files, Edit the design variables, Simulate the design, and return the dataset '''

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
        # create the simulation output directory
        util.safe_makedirs(output_dir)


        # >> Simulate and return the dataset
        ipython = getipython.get_ipython()
        if ipython is None:
            print("The remaining portion of the script must be run in an IPython environment. Exiting.")
            return
        # capture the netlist in a string
        netlist = self.design.generate_netlist()
        # access to the simulator object to run netlists
        simulator = ads.CircuitSimulator()
        # run the netlist, this will block output
        simulator.run_netlist(netlist, output_dir=output_dir)
        output_data = dataset.open(Path(os.path.join(output_dir, f"{design_name}.ds")))
        
        # >> return data in pandas DataFrame format
        # <class 'pandas.core.frame.DataFrame'>
        return output_data[f'{param_sweep_name}.{simulation_name}.DC'].to_dataframe().reset_index()
    


    def dataset_reshape(self, pd_data: pd.DataFrame, input_dimension: list, var_dict: dict):
        ''' reshape the dataset into desired input matrix and output vector '''
        row_count = input_dimension[0]
        col_count = input_dimension[1]
        row_name = 'VGS'
        col_name = 'VDS'
        item_name = 'IDS.i'
        output_x = np.empty((row_count, col_count),dtype=np.float64)
        output_y = np.empty((len(var_dict), 1),dtype=np.float64)

        for row in range(row_count):
            output_x[row, :] = pd_data.loc[pd_data[row_name] == (row + 1), item_name].to_numpy()
        for index, item in enumerate(var_dict):
            output_y[index, 0] = var_dict[item]

        return output_x, output_y
    

def param_random_generator(param_range: dict):
    ''' generate a random parameter set for the HEMT model '''
    # define the parameter range
    # param_range = {
    #     'VOFF': (1, 3.5),
    #     'U0': (150e-3, 800e-3),
    #     'NS0ACCS': (5e15, 5e20),
    #     'NFACTOR': (0.2, 1),
    #     'ETA0': (0, 1),
    #     'VSAT': (50e3, 250e3),
    #     'VDSCALE': (0, 10),
    #     'CDSCD': (0, 5),
    #     'LAMBDA': (100e-6, 0.5),
    #     'MEXPACCS': (1, 5)
    # }
    
    # generate random parameters
    var_dict = {key: str(np.random.uniform(low=val[0], high=val[1])) for key, val in param_range.items()}
    return var_dict

def init_h5_file(h5_path, m, n, y_len,
                 dtype_x=np.float64, dtype_y=np.float64):
    with h5py.File(h5_path, 'w') as f:
        # X: [num_samples, m, n]
        f.create_dataset(
            'X',
            shape=(0, m, n),
            maxshape=(None, m, n),
            dtype=dtype_x
        )
        # Y: [num_samples, y_len]
        f.create_dataset(
            'Y',
            shape=(0, y_len, 1),
            maxshape=(None, y_len, 1),
            dtype=dtype_y
        )

def append_to_h5(h5_path, x_new, y_new):
    x_new = np.asarray(x_new, dtype=np.float64)
    y_new = np.asarray(y_new, dtype=np.float64)

    # 确保 y_new 是二维 (batch_size, y_len)
    if y_new.ndim == 2:
        y_new = y_new.reshape(-1, 1)
    else:
        raise RuntimeError(f"y_new must be vector, but got shape {y_new.shape}")

    with h5py.File(h5_path, 'a') as f:
        ds_x = f['X']
        ds_y = f['Y']

        cur_len = ds_x.shape[0]
        new_len = cur_len + 1

        # 扩展
        ds_x.resize(new_len, axis=0)
        ds_y.resize(new_len, axis=0)

        # 赋值
        ds_x[cur_len:new_len, :, :] = x_new
        ds_y[cur_len:new_len, :] = y_new


def singel_process_iteration_data_gen2h5(workspace_path: str, library_name: str, design_name: str, instance_name: str, param_range: dict, param_sweep_name: str, simulation_name: str, iteration_num: int, process_id: int, save_path: str, dtype_x=np.float64, dtype_y=np.float64):
    ''' generate dataset in single process iteration '''
    # create an instance of the PyADS class
    ads_ctrl = PyADS()
    X_shape = [7,236]
    Y_shape = [10,1]

    # init h5 file
    init_h5_file(f"{save_path}\\dataset_process_{process_id}.h5", X_shape[0], X_shape[1], Y_shape[0])

    for i in range(iteration_num):
        start_time = time.time()
        var_dict = param_random_generator(param_range)
        pd_data = ads_ctrl.schematic_simulation(
            workspace_path,
            library_name,
            design_name,
            instance_name,
            var_dict,
            param_sweep_name,
            simulation_name
        )
        X, y = ads_ctrl.dataset_reshape(pd_data, X_shape, var_dict)
        end_time = time.time()
        print(f' >> Process {process_id} :: Loop {i + 1}/{iteration_num} :: used time:', round(end_time - start_time, 2), 's')

        try:
            append_to_h5(f"{save_path}\\dataset_process_{process_id}.h5", X, y)
        except:
            print(f"【ERROR】Error appending data in process {process_id} at iteration {i + 1}.")
            continue