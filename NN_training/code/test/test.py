# packages to import data analysis and save
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import h5py

# for NN model inference
from typing import Literal
import subprocess 
import time
import os
def model_inference(python_path:str, 
                     script_dir:str, 
                     code_file_name:str, 
                     model_path:str, 
                     inference_data_path:str, 
                     inference_data_index:str,
                     output_csv_path:str, 
                     single_input_mode:bool,
                     cvae_ena:bool = False,
                     cvae_mode:Literal['rand', 'mean'] = 'mean',
                     num_sampling:int = 1) -> dict:
    
    if cvae_ena: # CVAE model (prability based) inference
        if inference_data_index == 'None':
            # single input with single a or multiple input
            cmd = [
                python_path, code_file_name,
                "--infer-run", model_path,
                "--input-h5",  inference_data_path,
                "--save-csv",  output_csv_path,
                "--sample_mode", cvae_mode,
                "--num-samples", str(num_sampling) if cvae_mode=='rand' else '1'
        ]
        else:
            cmd = [
            python_path, code_file_name,
            "--infer-run", model_path,
            "--input-h5", inference_data_path,
            "--index", inference_data_index,
            "--save-csv", output_csv_path,
            "--sample_mode", cvae_mode,
            "--num-samples", str(num_sampling) if cvae_mode=='rand' else '1'
            ]

    else: # 2 stage DNN (value based) model inference
        if inference_data_index == 'None':
            # single input with single a or multiple input
            cmd = [
                python_path, code_file_name,
                "--data", inference_data_path,
                "--infer-run", model_path,
                "--input-h5",  inference_data_path,
                "--save-csv",  output_csv_path
        ]
        else:
            cmd = [
            python_path, code_file_name,
            "--data", inference_data_path,
            "--infer-run", model_path,
            "--input-h5", inference_data_path,
            "--index", inference_data_index,
            "--save-csv", output_csv_path
            ]

    result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
    print(result.stdout)

    # transform output from str to var dict
    import pandas as pd

    df = pd.read_csv(output_csv_path)
    if bool(inference_data_index) & single_input_mode:
    # single input
        df_dict = df.to_dict(orient='list')
        var_dict = {key:str(value) for (key, value) in zip(df_dict['param'], df_dict['value'])}
    else:
    # Batch input
        df_dict = df[df['index'] == 1].to_dict(orient='list')
        df_dict.pop('index')
        var_dict = {key:str(df_dict[key][0]) for key in df_dict}
        
    
    return var_dict

# For ADS
# DEFINE VARIABLES

workspace_path = "E:\\personal_Data\\Document of School\\Uni Stuttgart\\Masterarbeit\\Simulation\\ADS\\ASM_HEMT1_wrk_1_Jia"
validate_dict = None
library_name = "IAF_pGaN_lib"
design_name = "gs66508bv1_Pytest_simple_paramset"
instance_name = "IV"
var_dict_default = {'VOFF':'1.785', 'U0':'0.424', 'NS0ACCS':'2e+17', 'NFACTOR':'1', 'ETA0':'0.06', 'VSAT':'8e+4', 'VDSCALE':'5', 'CDSCD':'0.1', 'LAMBDA':'0.01', 'MEXPACCD':'1.5', 'DELTA':'3'}
param_range = {
        'VOFF': (-1.2, 2.6),
        'U0': (0, 2.2),
        'NS0ACCS': (1e15, 1e20),
        'NFACTOR': (0.1, 5),
        'ETA0': (0, 1),
        'VSAT': (5e4, 1e7),
        'VDSCALE': (0.5, 1e6),
        'CDSCD': (1e-5, 0.75),
        'LAMBDA': (0, 0.2),
        'MEXPACCD': (0.05, 12),
        'DELTA': (2, 100)
    }
vgs_bias_param_sweep_name = 'Sweep_vgs'
vds_bias_param_sweep_name = 'Sweep_vds'
vgs_bias_simulation_name = 'DC1'
vds_bias_simulation_name = 'DC2'
iteration_num = 1000
process_id = 1
data_save_path = "E:\\personal_Data\\Document of School\\Uni Stuttgart\\Masterarbeit\\Code\\param_regression\\ADS_Parameter_Fitting\\IV_param_regression\\NN_training\\dataset\\temp_generated"


# ------------------------------------------------------------------
# For NN model inference

python_path = r"D:/Miniconda/envs/DL/python.exe"  # model based python path
script_dir  = r"E:\\personal_Data\\Document of School\\Uni Stuttgart\\Masterarbeit\\Code\\param_regression\\ADS_Parameter_Fitting\\IV_param_regression\\NN_training"

training_data_path   = r"E:\\personal_Data\\Document of School\\Uni Stuttgart\\Masterarbeit\\Code\\param_regression\\ADS_Parameter_Fitting\\IV_param_regression\\NN_training\\dataset\\training\\stage_x_ml_full_data.h5"  # path of training dataset

meas_like_val_set = r"E:/personal_Data/Document of School/Uni Stuttgart/Masterarbeit/Code/param_regression/ADS_Parameter_Fitting/IV_param_regression/NN_training/dataset/training/stage_2_ml_data.h5" 

meas_path = r"E:\\personal_Data\\Document of School\\Uni Stuttgart\\Masterarbeit\\Code\\param_regression\\ADS_Parameter_Fitting\\IV_param_regression\\NN_training\\dataset\\training\\meas_data.h5"

output_csv_path = r"E:\\personal_Data\\Document of School\\Uni Stuttgart\\Masterarbeit\\Code\\param_regression\\ADS_Parameter_Fitting\\IV_param_regression\\NN_training\\dataset\\temp_generated\\pred_7row.csv"

# model_path = r"E:\\personal_Data\\Document of School\\Uni Stuttgart\\Masterarbeit\\Code\\param_regression\\ADS_Parameter_Fitting\\IV_param_regression\\NN_training\\runs_test\\log_dataset_test\\2_4_0_1_normal_log"

model_path = r"E:\\personal_Data\\Document of School\\Uni Stuttgart\\Masterarbeit\\Code\\param_regression\\ADS_Parameter_Fitting\\IV_param_regression\\NN_training\\temp_2\\version_2_1"


# basic params
inference_data_path = training_data_path
inference_data_index = "7" #  from 0 to 9999
# "None"    : for 1. single input with shape (2,6,121)
#                 2. multile input with shaple (N,2,6,121)
# str(int)  : for single input with shape (N,2,6,121)
single_input_mode = bool(True)
cvae_ena = True
cvae_mode = 'mean'
num_sampling = 20 # only used when cvae_mode = 'rand'

if __name__ == "__main__":
    # Model inference
    print(" >> Start model inference...")
#     var_dict = model_inference(
#         python_path=python_path,
#         script_dir=script_dir,
#         code_file_name=script_dir+"\\asm_hemt_2stage_dnn.py",
#         model_path=model_path,
#         inference_data_path=inference_data_path,
#         inference_data_index=inference_data_index,
#         output_csv_path=output_csv_path,
#         single_input_mode=single_input_mode,
#         cvae_ena=cvae_ena,
#         cvae_mode=cvae_mode,
#         num_sampling=num_sampling
# )
    
    cmd = [
            python_path, "asm_hemt_cvae.py",
            "--infer-run", model_path,
            "--input-h5", inference_data_path,
            "--index", inference_data_index,
            "--save-csv", output_csv_path,
            "--sample-mode", cvae_mode,
            "--num-samples", str(num_sampling) if cvae_mode=='rand' else '1'
            ]
    result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True, check=True)
    print(result.stdout)