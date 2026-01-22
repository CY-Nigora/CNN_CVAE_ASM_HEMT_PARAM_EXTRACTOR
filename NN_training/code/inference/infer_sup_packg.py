# packages to import data analysis and save
from matplotlib.pylab import norm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import h5py
import os
# for NN model inference
from typing import Literal
import subprocess 
import time
from scipy.stats import gaussian_kde
from collections import deque

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder created: {path}")
    else:
        print(f"Folder already exists: {path}")

# Model inference : call func in other Python env

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
                     num_sampling:int = 1,
                     dropout_enable:bool = False) -> dict:
    
    if cvae_ena: # CVAE model (prability based) inference
        if inference_data_index == 'None':
            # single input with single a or multiple input
            cmd = [
                python_path, code_file_name,
                "--infer-run", model_path,
                "--input-h5",  inference_data_path,
                "--save-csv",  output_csv_path,
                "--sample-mode", cvae_mode,
                "--num-samples", str(num_sampling) if cvae_mode=='rand' else '1'
                ]
            if dropout_enable:
                cmd += ["--dropout-infer"]
            # print('DEBUG CMD:', cmd)
        else:
            cmd = [
            python_path, code_file_name,
            "--infer-run", model_path,
            "--input-h5", inference_data_path,
            "--index", inference_data_index,
            "--save-csv", output_csv_path,
            "--sample-mode", cvae_mode,
            "--num-samples", str(num_sampling) if cvae_mode=='rand' else '1'
            ]
            if dropout_enable:
                cmd += ["--dropout-infer"]


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
    df = pd.read_csv(output_csv_path)
    var_dict = {}
    if cvae_ena: # cvae model
        num_col = df["sample_idx"].max() + 1
        for name in df.columns:
            if name not in ['input_idx', 'sample_idx']:
                var_dict[name] = df[name].astype(str).to_list() if num_col > 1 else str(df[name][0])
    else: # not cvae
        if bool(inference_data_index) & single_input_mode:
        # single input
            df_dict = df.to_dict(orient='list')
            for name in df.columns:
                if name not in ['input_idx', 'sample_idx']:
                    var_dict[name] = df[name].astype(str).to_list()*2
        else:
        # Batch input
            df_dict = df[df['index'] == 1].to_dict(orient='list')
            df_dict.pop('index')
            var_dict = {key:str(df_dict[key][0]) for key in df_dict}
            
    
    return var_dict


def TTO_infer(  python_path:str, 
                script_dir:str, 
                code_file_name:str,
                model_path:str, 
                proxy_path:str, 
                infer_data_path:str,
                infer_data_index:str,
                steps:int,
                lr:float):
    
    # check if target folder exist
    ensure_dir(model_path + "/tto_infer")

    output_csv_path = model_path + '/tto_infer/tto_strict_result.csv'

    cmd = [
        python_path, script_dir + '/' + code_file_name,
        "--cvae-run", model_path,
        "--proxy-run", proxy_path,
        "--meas-h5", infer_data_path,
        "--save-to", output_csv_path,
        "--steps", str(steps),
        "--lr", str(lr)
        ]

    result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
    print(result.stdout)
    print(f"succesfully save TTO infer result in {output_csv_path} !")

    # transform output from str to var dict
    df = pd.read_csv(output_csv_path)
    col_num = len(df.columns) - 1 # remove the final col of loss
    var_dict = {}
    for index, name in enumerate(df.columns):
        if index <= col_num-1:
            var_dict[name] = df[name].astype(str).to_list()*2
    
    return var_dict


def  proxy_model_inference(python_path:str, 
                     script_dir:str, 
                     code_file_name:str, 
                     model_path:str, 
                     inference_data_path:str, 
                     inference_data_index:str,
                     output_npy_path:str) -> np.ndarray:
    
    if inference_data_index == 'None':
        # single input with single a or multiple input
        cmd = [
            python_path, code_file_name,
            "--data", inference_data_path,
            "--infer-proxy-run", model_path,
            "--proxy-input-h5",  inference_data_path,
            "--save-xhat-npy",  output_npy_path
    ]
    else:
        cmd = [
        python_path, code_file_name,
        "--data", inference_data_path,
        "--infer-proxy-run", model_path,
        "--proxy-input-h5", inference_data_path,
        "--proxy-index", inference_data_index,
        "--save-xhat-npy", output_npy_path
        ]

    result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
    print(result.stdout)



def plot_proxy_error(inference_data_path:str, x_hat_path:str, inference_data_index:str) -> None:
    if inference_data_index != 'None':
        x_hat = np.load(x_hat_path)[0,:]
        x_original = h5py.File(inference_data_path, 'r')
        x_original = x_original['X'][int(inference_data_index), :, :]
        RMSE = np.sqrt(np.mean((x_hat - x_original) ** 2))
        MAE = np.mean(np.abs(x_hat - x_original))
        NMAE = MAE / np.mean(x_original) *100

        Vds_range = np.arange(-3.5, 8.5 + 0.1, 0.1)

        # plot validation result
        # legend_values = np.linspace(1.5, 6.5, 6)  
        legend_values = np.linspace(1, 7, 7)  
        # create colormap
        cmap = plt.cm.winter
        norm = plt.Normalize(vmin=legend_values.min(), vmax=legend_values.max())

        fig, ax = plt.subplots(1, 3, figsize=(15, 8)) 
        for i in range(7):
            color = cmap(norm(legend_values[i]))
            ax[0].plot( Vds_range, x_original[i, :], label=f"VGS={i+1} V", color = color)
            ax[1].plot( Vds_range, x_hat[i, :], label=f"VGS={i+1} V", color = color)
            ax[2].plot( Vds_range, x_hat[i, :] - x_original[i, :], label=f"VGS={i+1} V", color = color)

        ax[0].legend()
        ax[0].grid(True)
        ax[0].set_title('original I-V (generated, with noise)')
        ax[0].set_xlabel("VDS (V)")
        ax[0].set_ylabel("IDS (A)")
        ax[1].grid(True)
        ax[1].legend()
        ax[1].set_title('predicted I-V (proxy g, DNN based)')
        ax[1].set_xlabel("VDS (V)")
        ax[1].set_ylabel("IDS (A)")
        ax[2].grid(True)
        ax[2].set_title(f'error of both I-V \n average MAE = {MAE:.3f} \n nMAE = {NMAE:.3f}%')
        ax[2].set_xlabel("VDS (V)")
        ax[2].set_ylabel("IDS (A)")

        # add color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("VGS (V)")


    else:
        assert ValueError("No index provided, cannot plot proxy error for multiple samples.")
    



def plot_error(inference_data_path:str, data_save_path:str, inference_data_index:str, infer_param_dict: dict, check_matrix: np.array, goal_size_dict: dict, extra_csv_path:str=None):
    # extract data from both pred and validation
    X_orginal = h5py.File(inference_data_path, 'r')
    X_pred = h5py.File(f"{data_save_path}\\validate.h5", 'r')
    if inference_data_index != 'None':
        index = int(inference_data_index) 
        X_orginal_goal = X_orginal['X'][index, :, :]
        X_orginal_y = X_orginal['Y'][index, :, :]
    else:
        if len(X_orginal['X'].shape) == 2:
            X_orginal_goal = X_orginal['X'][:]
        else:
            X_orginal_goal = X_orginal['X'][0,:]
    X_pred_goal = X_pred[goal_size_dict['goal']][:]

    # print error in percentage
    if (inference_data_index != 'None'): # not for real meas-data
        print('Relative errors of predicted paramters:')
        if check_matrix.shape[0] == 1: # single output, for 2 stage or single sampling cvae
            check_list = check_matrix[:]
            for index,key in enumerate(infer_param_dict):
                val = float(infer_param_dict[key])  
                ref = float(X_orginal_y[index][0])    
                if check_list[0,index] != 'OK':
                    print(f'{key:<10}: {(val-ref)/ref*100:>10.2f}% ({check_list[index]})')
                else:   
                    print(f'{key:<10}: {(val-ref)/ref*100:>10.2f}%')
        elif check_matrix.shape[0] > 1 and extra_csv_path is not None:
            std_csv_path = extra_csv_path[:-4] + '_mean_std.csv'
            if os.path.exists(std_csv_path):
                df = pd.read_csv(std_csv_path)
                for index,key in enumerate(df['name']):
                    val = float(df.loc[index, 'mean'])  
                    ref = float(X_orginal_y[index][0])    
                    print(f'{key:<10}: {(val-ref)/ref*100:>10.2f}%')

    V_col_range = np.linspace(min(goal_size_dict['col_range']), max(goal_size_dict['col_range']), goal_size_dict['col_num'])

    # plot validation result
    # legend_values = np.linspace(1.5, 6.5, 6)  
    legend_values = np.linspace(min(goal_size_dict['row_range']), max(goal_size_dict['row_range']), goal_size_dict['row_num'])
    # create colormap
    cmap = plt.cm.winter
    norm = plt.Normalize(vmin=legend_values.min(), vmax=legend_values.max())
    fig, ax = plt.subplots(1, 4, figsize=(20, 8)) 

    samp_num = None
    if check_matrix.shape[0] == 1:
        RMSE = np.mean((X_pred_goal - X_orginal_goal) ** 2)
        MAE = np.mean(np.abs(X_pred_goal - X_orginal_goal))
        NMAE = MAE / np.mean(X_orginal_goal) *100
    else: # cvae + rand + multi-sampling
        samp_num = check_matrix.shape[0]
        RMSE, MAE, NMAE = np.zeros((samp_num,1)), np.zeros((samp_num,1)), np.zeros((samp_num,1))
        for i in range(samp_num):
            RMSE[i] = np.mean((X_pred_goal[i,:,:] - X_orginal_goal) ** 2)
            MAE[i] = np.mean(np.abs(X_pred_goal[i,:,:] - X_orginal_goal))
            NMAE[i] = MAE[i] / np.mean(X_orginal_goal) *100

    min_val = MAE.min()
    min_idx = MAE.argmin()
    origin_color = 'pink'
    predict_color = 'green'

    if samp_num is None:
        for i in range(goal_size_dict['row_num']):
            colormap = cmap(norm(legend_values[i]))
            ax[0].plot( V_col_range, X_orginal_goal[i, :], label=f"V_row={i+1} V", color = colormap)
            ax[1].plot( V_col_range, X_pred_goal[0, i, :], label=f"V_row={i+1} V", color = colormap)
            ax[2].plot( V_col_range, X_orginal_goal[i, :], label=f"original" if i==0 else None, color = origin_color)
            ax[2].plot( V_col_range, X_pred_goal[0, i, :], label=f"prediction" if i==0 else None, color = predict_color)
            ax[3].plot( V_col_range, X_pred_goal[0, i, :] - X_orginal_goal[i, :], label=f"V_row={i+1} V", color = colormap)
    else:
        for j in range(goal_size_dict['row_num']):
            colormap = cmap(norm(legend_values[j]))
            # left
            ax[0].plot(V_col_range, X_orginal_goal[j, :], color=colormap, label=f"V_row={j+1} V")
            # mitte
            q5  = np.percentile(X_pred_goal[:, j, :], 5, axis=0)
            q25 = np.percentile(X_pred_goal[:, j, :], 25, axis=0)
            q50 = np.percentile(X_pred_goal[:, j, :], 50, axis=0)
            q75 = np.percentile(X_pred_goal[:, j, :], 75, axis=0)
            q95 = np.percentile(X_pred_goal[:, j, :], 95, axis=0)
            ax[1].fill_between(V_col_range, q5, q95,  color="lightgray", alpha=0.4, zorder=1)
            ax[1].fill_between(V_col_range, q25, q75, color="gray", alpha=0.3, zorder=2)
            ax[1].plot(V_col_range, X_pred_goal[min_idx, j, :], color=colormap, label = f"V_row={j+1} V", zorder=3)
            # right
            ax[2].plot(V_col_range, X_orginal_goal[j, :], color=origin_color, label=f"original" if j==0 else None)
            ax[2].plot(V_col_range, X_pred_goal[min_idx, j, :], color=predict_color, label=f"prediction" if j==0 else None)
            ax[3].plot(V_col_range, X_pred_goal[min_idx, j, :] - X_orginal_goal[j, :], color=colormap)

    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_title('original I-V (generated, with noise)')
    ax[0].set_xlabel("V_col (V)")
    ax[0].set_ylabel("IDS (A)")
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_title('predicted I-V (NN based)')
    ax[1].set_xlabel("V_col (V)")
    ax[1].set_ylabel("IDS (A)")
    ax[2].grid(True)
    ax[2].legend()
    ax[2].set_title('predicted v.s. original')
    ax[2].set_xlabel("V_col (V)")
    ax[2].set_ylabel("IDS (A)")
    ax[3].grid(True)
    if samp_num is None:
        ax[3].set_title(f'error of both I-V \n average MAE = {MAE:.3f} \n nMAE = {NMAE:.3f}%')
    else:
        ax[3].set_title(f'error of both I-V \n average MAE = {min_val:.3f} \n nMAE = {NMAE[min_idx].item():.3f}% \n (best case from {samp_num} samplings)')
    ax[3].set_xlabel("V_col (V)")
    ax[3].set_ylabel("IDS (A)")

    # add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("V_row (V)")

    return min_idx


def plot_error_2channel(inference_data_path:str, data_save_path:str, inference_data_index:str, infer_param_dict: dict, check_matrix: np.array, goal_size_dict: dict, extra_csv_path:str=None, shrink:bool=False, size=(35, 40)):
    # extract data from both pred and validation
    X_orginal = h5py.File(inference_data_path, 'r')
    X_pred = h5py.File(f"{data_save_path}\\validate.h5", 'r')
    if inference_data_index != 'None':
        index = int(inference_data_index) 
        X_orginal_iv = X_orginal['X_iv'][index, :, :]
        X_orginal_gm = X_orginal['X_gm'][index, :, :]
        X_orginal_y = X_orginal['Y'][index, :, :]
    else:
        if len(X_orginal['X_iv'].shape) == 2:
            X_orginal_iv = X_orginal['X_iv'][:]
            X_orginal_gm = X_orginal['X_gm'][:]
        else:
            X_orginal_iv = X_orginal['X_iv'][0,:]
            X_orginal_gm = X_orginal['X_gm'][0,:]
    X_pred_iv = X_pred['X_iv'][:]
    X_pred_gm = X_pred['X_gm'][:]

    # print error in percentage
    if (inference_data_index != 'None'): # not for real meas-data
        print('Relative errors of predicted paramters:')
        if check_matrix.shape[0] == 1: # single output, for 2 stage or single sampling cvae
            check_list = check_matrix[:]
            for index,key in enumerate(infer_param_dict):
                val = float(infer_param_dict[key])  
                ref = float(X_orginal_y[index][0])    
                if check_list[0,index] != 'OK':
                    print(f'{key:<10}: {(val-ref)/ref*100:>10.2f}% ({check_list[index]})')
                else:   
                    print(f'{key:<10}: {(val-ref)/ref*100:>10.2f}%')
        elif check_matrix.shape[0] > 1 and extra_csv_path is not None:
            std_csv_path = extra_csv_path[:-4] + '_mean_std.csv'
            if os.path.exists(std_csv_path):
                df = pd.read_csv(std_csv_path)
                for index,key in enumerate(df['name']):
                    val = float(df.loc[index, 'mean'])  
                    ref = float(X_orginal_y[index][0])    
                    print(f'{key:<10}: {(val-ref)/ref*100:>10.2f}%')

    V_col_range_iv = np.linspace(min(goal_size_dict['X_iv']['col_range']), max(goal_size_dict['X_iv']['col_range']), goal_size_dict['X_iv']['col_num'])
    V_col_range_gm = np.linspace(min(goal_size_dict['X_gm']['col_range']), max(goal_size_dict['X_gm']['col_range']), goal_size_dict['X_gm']['col_num'])

    # plot validation result
    # legend_values = np.linspace(1.5, 6.5, 6)  
    legend_values_iv = np.linspace(min(goal_size_dict['X_iv']['row_range']), max(goal_size_dict['X_iv']['row_range']), goal_size_dict['X_iv']['row_num'])
    legend_values_gm = np.linspace(min(goal_size_dict['X_gm']['row_range']), max(goal_size_dict['X_gm']['row_range']), goal_size_dict['X_gm']['row_num'])
    # create colormap
    cmap_iv = plt.cm.winter
    cmap_gm = plt.cm.winter
    norm_iv = plt.Normalize(vmin=legend_values_iv.min(), vmax=legend_values_iv.max())
    norm_gm = plt.Normalize(vmin=legend_values_gm.min(), vmax=legend_values_gm.max())

    samp_num = None
    # 预先计算 Ground Truth 的均值，用于 R2 分母计算
    iv_mean = np.mean(X_orginal_iv)
    gm_mean = np.mean(X_orginal_gm)
    ss_tot_iv = np.sum((X_orginal_iv - iv_mean) ** 2)
    ss_tot_gm = np.sum((X_orginal_gm - gm_mean) ** 2)

    if check_matrix.shape[0] == 1: # single sampling
        # --- IV ---
        mse_iv = np.mean((X_pred_iv - X_orginal_iv) ** 2)
        RMSE_iv = np.sqrt(mse_iv)  
        NRMSE_iv = np.sqrt(np.mean(((X_pred_iv - X_orginal_iv)/np.max(X_orginal_iv)) ** 2)) *100
        MAE_iv = np.mean(np.abs(X_pred_iv - X_orginal_iv))
        NMAE_iv = MAE_iv / np.abs(iv_mean) * 100 
        ss_res_iv = np.sum((X_pred_iv - X_orginal_iv) ** 2)
        R2_iv = 1 - (ss_res_iv / (ss_tot_iv + 1e-8)) 

        # --- GM ---
        mse_gm = np.mean((X_pred_gm - X_orginal_gm) ** 2)
        RMSE_gm = np.sqrt(mse_gm)  
        NRMSE_gm = np.sqrt(np.mean(((X_pred_gm - X_orginal_gm)/np.max(X_orginal_gm)) ** 2)) *100
        MAE_gm = np.mean(np.abs(X_pred_gm - X_orginal_gm))
        NMAE_gm = MAE_gm / np.abs(gm_mean) * 100
        ss_res_gm = np.sum((X_pred_gm - X_orginal_gm) ** 2)
        R2_gm = 1 - (ss_res_gm / (ss_tot_gm + 1e-8)) 

    else: # cvae + rand + multi-sampling
        samp_num = check_matrix.shape[0]
        # init
        RMSE_iv, NRMSE_iv, MAE_iv, NMAE_iv, R2_iv = [np.zeros((samp_num, 1)) for _ in range(5)]
        RMSE_gm, NRMSE_gm, MAE_gm, NMAE_gm, R2_gm = [np.zeros((samp_num, 1)) for _ in range(5)]

        for i in range(samp_num):
            # --- IV ---
            curr_pred_iv = X_pred_iv[i, :, :]
            mse_iv = np.mean((curr_pred_iv - X_orginal_iv) ** 2)
            RMSE_iv[i] = np.sqrt(mse_iv) 
            NRMSE_iv[i] = np.sqrt(np.mean(((curr_pred_iv - X_orginal_iv)/np.max(X_orginal_iv)) ** 2)) *100
            MAE_iv[i] = np.mean(np.abs(curr_pred_iv - X_orginal_iv))
            NMAE_iv[i] = MAE_iv[i] / np.abs(iv_mean) * 100
            
            ss_res_iv = np.sum((curr_pred_iv - X_orginal_iv) ** 2)
            R2_iv[i] = 1 - (ss_res_iv / (ss_tot_iv + 1e-8))

            # --- GM ---
            curr_pred_gm = X_pred_gm[i, :, :]
            
            mse_gm = np.mean((curr_pred_gm - X_orginal_gm) ** 2)
            RMSE_gm[i] = np.sqrt(mse_gm) 
            NRMSE_gm[i] = np.sqrt(np.mean(((curr_pred_gm - X_orginal_gm)/np.max(X_orginal_gm)) ** 2)) *100
            MAE_gm[i] = np.mean(np.abs(curr_pred_gm - X_orginal_gm))
            NMAE_gm[i] = MAE_gm[i] / np.abs(gm_mean) * 100
            
            ss_res_gm = np.sum((curr_pred_gm - X_orginal_gm) ** 2)
            R2_gm[i] = 1 - (ss_res_gm / (ss_tot_gm + 1e-8))

    # Best IV result
    target_iv = NRMSE_iv
    best_val_iv = target_iv.min()    
    best_idx_iv = target_iv.argmin()

    # Best GM result
    target_gm = NRMSE_gm
    best_val_gm = target_gm.min()
    best_idx_gm = target_gm.argmin()

    # Best Total result
    target_total = target_iv + target_gm
    best_val_total = target_total.min()
    best_idx_total = target_total.argmin()


    origin_color = 'pink'
    predict_color = 'green'

    if samp_num is None: # single sampling
        fig, ax = plt.subplots(2, 4, figsize=(20, 8)) 
        for i in range(goal_size_dict['X_iv']['row_num']):
            colormap = cmap_iv(norm_iv(legend_values_iv[i]))
            ax[0][0].plot( V_col_range_iv, X_orginal_iv[i, :], label=f"V_row={i+1} V", color = colormap)
            ax[0][1].plot( V_col_range_iv, X_pred_iv[0, i, :], label=f"V_row={i+1} V", color = colormap)
            ax[0][2].plot( V_col_range_iv, X_orginal_iv[i, :], label=f"original" if i==0 else None, color = origin_color)
            ax[0][2].plot( V_col_range_iv, X_pred_iv[0, i, :], label=f"prediction" if i==0 else None, color = predict_color)
            ax[0][3].plot( V_col_range_iv, X_pred_iv[0, i, :] - X_orginal_iv[i, :], label=f"V_row={i+1} V", color = colormap)
        for i in range(goal_size_dict['X_gm']['row_num']):
            colormap = cmap_gm(norm_gm(legend_values_gm[i]))
            ax[1][0].plot( V_col_range_gm, X_orginal_gm[i, :], label=f"V_row={i+1} V", color = colormap)
            ax[1][1].plot( V_col_range_gm, X_pred_gm[0, i, :], label=f"V_row={i+1} V", color = colormap)
            ax[1][2].plot( V_col_range_gm, X_orginal_gm[i, :], label=f"original" if i==0 else None, color = origin_color)
            ax[1][2].plot( V_col_range_gm, X_pred_gm[0, i, :], label=f"prediction" if i==0 else None, color = predict_color)
            ax[1][3].plot( V_col_range_gm, X_pred_gm[0, i, :] - X_orginal_gm[i, :], label=f"V_row={i+1} V", color = colormap)

    else:
        if not shrink:
            fig, ax = plt.subplots(3, 8, figsize=size) 
            # best output result
            best_goal_name = ['Best output (min NRMSE output)', 'Best transfer (min NRMSE transfer)', 'Best total (min NRMSE output + transfer)']
            for index, best_goal in enumerate([best_idx_iv, best_idx_gm, best_idx_total]):
                for j in range(goal_size_dict['X_iv']['row_num']):
                    colormap = cmap_iv(norm_iv(legend_values_iv[j]))
                    # left
                    ax[index][0].plot(V_col_range_iv, X_orginal_iv[j, :], color=colormap, label=f"V_row={j+1} V")
                    # mitte
                    q5  = np.percentile(X_pred_iv[:, j, :], 5, axis=0)
                    q25 = np.percentile(X_pred_iv[:, j, :], 25, axis=0)
                    q50 = np.percentile(X_pred_iv[:, j, :], 50, axis=0)
                    q75 = np.percentile(X_pred_iv[:, j, :], 75, axis=0)
                    q95 = np.percentile(X_pred_iv[:, j, :], 95, axis=0)
                    ax[index][1].fill_between(V_col_range_iv, q5, q95,  color="lightgray", alpha=0.4, zorder=1)
                    ax[index][1].fill_between(V_col_range_iv, q25, q75, color="gray", alpha=0.3, zorder=2)
                    ax[index][1].plot(V_col_range_iv, X_pred_iv[best_goal, j, :], color=colormap, label = f"V_row={j+1} V", zorder=3)
                    # right
                    ax[index][2].plot(V_col_range_iv, X_orginal_iv[j, :], color=origin_color, label=f"original" if j==0 else None)
                    ax[index][2].plot(V_col_range_iv, X_pred_iv[best_goal, j, :], color=predict_color, label=f"prediction" if j==0 else None)
                    ax[index][3].plot(V_col_range_iv, X_pred_iv[best_goal, j, :] - X_orginal_iv[j, :], color=colormap)
                for j in range(goal_size_dict['X_gm']['row_num']):
                    colormap = cmap_gm(norm_gm(legend_values_gm[j]))
                    # left
                    ax[index][4].plot(V_col_range_gm, X_orginal_gm[j, :], color=colormap, label=f"V_row={j+1} V")
                    # mitte
                    q5  = np.percentile(X_pred_gm[:, j, :], 5, axis=0)
                    q25 = np.percentile(X_pred_gm[:, j, :], 25, axis=0)
                    q50 = np.percentile(X_pred_gm[:, j, :], 50, axis=0)
                    q75 = np.percentile(X_pred_gm[:, j, :], 75, axis=0)
                    q95 = np.percentile(X_pred_gm[:, j, :], 95, axis=0)
                    ax[index][5].fill_between(V_col_range_gm, q5, q95,  color="lightgray", alpha=0.4, zorder=1)
                    ax[index][5].fill_between(V_col_range_gm, q25, q75, color="gray", alpha=0.3, zorder=2)
                    ax[index][5].plot(V_col_range_gm, X_pred_gm[best_goal, j, :], color=colormap, label = f"V_row={j+1} V", zorder=3)
                    # right
                    ax[index][6].plot(V_col_range_gm, X_orginal_gm[j, :], color=origin_color, label=f"original" if j==0 else None)
                    ax[index][6].plot(V_col_range_gm, X_pred_gm[best_goal, j, :], color=predict_color, label=f"prediction" if j==0 else None)
                    ax[index][7].plot(V_col_range_gm, X_pred_gm[best_goal, j, :] - X_orginal_gm[j, :], color=colormap)
                for fig_col in range(8):
                    ax[index][fig_col].legend()
                    ax[index][fig_col].grid(True)
                    ax[index][fig_col].set_xlabel("V_col (V)")
                    ax[index][fig_col].set_ylabel("IDS (A)")
                ax[index][0].set_title('original I-V (generated, with noise)')
                ax[index][4].set_title('original I-V (generated, with noise)')
                ax[index][1].set_title('predicted I-V (NN based)')
                ax[index][5].set_title('predicted I-V (NN based)')
                ax[index][2].set_title('predicted v.s. original')
                ax[index][6].set_title('predicted v.s. original')
                if samp_num is None:
                    ax[index][3].set_title(f'error of both I-V \n NRMSE = {NRMSE_iv[best_goal].item():.3f}% \n RMSE = {RMSE_iv:.3f}')
                    ax[index][7].set_title(f'error of both I-V \n NRMSE = {NRMSE_gm[best_goal].item():.3f}% \n RMSE = {RMSE_gm:.3f}')
                else:
                    ax[index][3].set_title(f'{best_goal_name[index]}\nerror of both I-V \n NRMSE = {NRMSE_iv[best_goal].item():.3f}% \n RMSE = {RMSE_iv[best_goal].item():.3f} \n (best case from {samp_num} samplings)')
                    ax[index][7].set_title(f'{best_goal_name[index]}\nerror of both I-V \n NRMSE = {NRMSE_gm[best_goal].item():.3f}% \n RMSE = {RMSE_gm[best_goal].item():.3f} \n (best case from {samp_num} samplings)')
            
            fig.savefig("C:\\Users\\97427\\Desktop\\figure_IV_full.svg", bbox_inches='tight', format='svg')
            print('saved figure under path : C:\\Users\\97427\\Desktop\\figure_IV_full.svg ')
  
        else:
            fig, ax = plt.subplots(1, 2, figsize=size) 
            # best output result
            best_goal_name = ['Best output (min NRMSE output)', 'Best transfer (min NRMSE transfer)', 'Best total (min NRMSE output + transfer)']
            best_goal = best_idx_total  # only plot best total
            index = 2
            for j in range(goal_size_dict['X_iv']['row_num']):
                colormap = cmap_iv(norm_iv(legend_values_iv[j]))
                # left
                q5  = np.percentile(X_pred_iv[:, j, :], 5, axis=0)
                q25 = np.percentile(X_pred_iv[:, j, :], 25, axis=0)
                q50 = np.percentile(X_pred_iv[:, j, :], 50, axis=0)
                q75 = np.percentile(X_pred_iv[:, j, :], 75, axis=0)
                q95 = np.percentile(X_pred_iv[:, j, :], 95, axis=0)
                ax[0].fill_between(V_col_range_iv, q5, q95,  color="lightgray", alpha=0.4, zorder=1)
                ax[0].fill_between(V_col_range_iv, q25, q75, color="gray", alpha=0.3, zorder=2)
                ax[0].plot(V_col_range_iv, X_pred_iv[best_goal, j, :], color=colormap, label = f"VGS={j+1} V", zorder=3)
            for j in range(goal_size_dict['X_gm']['row_num']):
                colormap = cmap_gm(norm_gm(legend_values_gm[j]))
                # left
                q5  = np.percentile(X_pred_gm[:, j, :], 5, axis=0)
                q25 = np.percentile(X_pred_gm[:, j, :], 25, axis=0)
                q50 = np.percentile(X_pred_gm[:, j, :], 50, axis=0)
                q75 = np.percentile(X_pred_gm[:, j, :], 75, axis=0)
                q95 = np.percentile(X_pred_gm[:, j, :], 95, axis=0)
                ax[1].fill_between(V_col_range_gm, q5, q95,  color="lightgray", alpha=0.4, zorder=1)
                ax[1].fill_between(V_col_range_gm, q25, q75, color="gray", alpha=0.3, zorder=2)
                ax[1].plot(V_col_range_gm, X_pred_gm[best_goal, j, :], color=colormap, label = f"VDS={j+1} V", zorder=3)
            for fig_col in range(1):
                ax[fig_col].legend()
                ax[fig_col].grid(True)
                ax[fig_col].set_xlabel("VDS (V)")
                ax[fig_col].set_ylabel("IDS (A)")
            for fig_col in range(1,2):
                ax[fig_col].legend()
                ax[fig_col].grid(True)
                ax[fig_col].set_xlabel("VGS (V)")
                ax[fig_col].set_ylabel("IDS (A)")           
            ax[0].set_title('predicted output characteristic')
            ax[1].set_title('predicted transfer characteristic')
            # if samp_num is None:
            #     ax[1].set_title(f'error of both I-V \n NRMSE = {NRMSE_iv[best_goal].item():.3f}% \n RMSE = {RMSE_iv:.3f}')
            #     ax[3].set_title(f'error of both I-V \n NRMSE = {NRMSE_gm[best_goal].item():.3f}% \n RMSE = {RMSE_gm:.3f}')
            # else:
            #     ax[1].set_title(f'NRMSE = {NRMSE_iv[best_goal].item():.3f}% \n RMSE = {RMSE_iv[best_goal].item():.3f} \n (best case from {samp_num} samplings)')
            #     ax[3].set_title(f'NRMSE = {NRMSE_gm[best_goal].item():.3f}% \n RMSE = {RMSE_gm[best_goal].item():.3f} \n (best case from {samp_num} samplings)')

            fig.savefig("C:\\Users\\97427\\Desktop\\figure_IV_shrink.svg", bbox_inches='tight', format='svg')
            print('saved figure under path : C:\\Users\\97427\\Desktop\\figure_IV_shrink.svg ')
    return best_idx_iv, best_idx_gm, best_idx_total


def four_fig_plot(X_origin_o, X_origin_t, X_pred, vgs_bias_x_axis, vds_bias_x_axis, size = (16,9)):

    # vgs_bias_x_axis = np.linspace(-1, 5, 61)
    # vds_bias_x_axis = np.linspace(-10, 0, 101)

    print('size of measured Output feature: ', X_origin_o['X'].shape)
    print('size of measured Transfer feature: ', X_origin_t['X'].shape)
    print('size of predicted Output feature: ', X_pred['X_iv'].shape)
    print('size of predicted Transfer feature: ', X_pred['X_gm'].shape)

    Gds_origin = np.gradient(X_origin_o['X'][0], vgs_bias_x_axis, axis=1)
    Gds_pred = np.gradient(X_pred['X_iv'][0], vgs_bias_x_axis, axis=1)
    Gm_origin = np.gradient(X_origin_t['X'][0], vds_bias_x_axis, axis=1)
    Gm_pred = np.gradient(X_pred['X_gm'][0], vds_bias_x_axis, axis=1)


    # Sub-Fig 1 : Transfer
    # Sub-Fig 2 : Output
    # Sub-Fig 3 : Gm  (Transfer based)
    # Sub-Fig 4 : Gds (Output based)

    fig, ax = plt.subplots(2, 2, figsize=size) 

    for i in range(X_pred['X_gm'].shape[1]):
        ax[0,0].plot(vds_bias_x_axis, X_origin_t['X'][0,i,:], 'o', color = 'red', label = None if (i+1) != X_pred['X_gm'].shape[1] else "Measured")
        ax[0,0].plot(vds_bias_x_axis, X_pred['X_gm'][0,i,:], color = 'blue', label = None if (i+1) != X_pred['X_gm'].shape[1] else 'DL based Prediction', linewidth = 3.0)
        ax[1,0].plot(vds_bias_x_axis, Gm_origin[i,:], 'o-', color = 'red', label = None if (i+1) != X_pred['X_gm'].shape[1] else "Measured")
        ax[1,0].plot(vds_bias_x_axis, Gm_pred[i,:], color = 'blue', label = None if (i+1) != X_pred['X_gm'].shape[1] else 'DL based Prediction', linewidth = 3.0)

    for i in range(X_pred['X_iv'].shape[1]):
        ax[0,1].plot(vgs_bias_x_axis, X_origin_o['X'][0,i,:], 'o', color = 'red', label = None if (i+1) != X_pred['X_iv'].shape[1] else "Measured")
        ax[0,1].plot(vgs_bias_x_axis, X_pred['X_iv'][0,i,:], color = 'blue', label = None if (i+1) != X_pred['X_iv'].shape[1] else 'DL based Prediction', linewidth = 3.0)
        ax[1,1].plot(vgs_bias_x_axis, Gds_origin[i,:], 'o-', color = 'red', label = None if (i+1) != X_pred['X_iv'].shape[1] else "Measured")
        ax[1,1].plot(vgs_bias_x_axis, Gds_pred[i,:], color = 'blue', label = None if (i+1) != X_pred['X_iv'].shape[1] else 'DL based Prediction', linewidth = 3.0)

    y_name = deque(['Drain Current (A)', 'Drain Current (A)', 'Transconductance (S)', 'Gds (S)'])
    x_name = deque(['Gate Voltage (V)\n(a)','Drain Voltage (V)\n(b)','Gate Voltage (V)\n(c)','Drain Voltage (V)\n(d)'])
    for row in range(2):
        for col in range(2):
            ax[row,col].legend()
            ax[row,col].grid(True)
            ax[row,col].set_xlabel(x_name.popleft())
            ax[row,col].set_ylabel(y_name.popleft())
    
    fig.savefig("C:\\Users\\97427\\Desktop\\figure.svg", bbox_inches='tight', format='svg')
    print('saved figure under path : C:\\Users\\97427\\Desktop\\figure.svg ')




def four_fig_plus_mlp_plot(X_origin_o, X_origin_t, X_pred, X_pred_mlp, vgs_bias_x_axis, vds_bias_x_axis, size = (16,9), fontsize=None):


    # print('size of measured Output feature: ', X_origin_o['X'].shape)
    # print('size of measured Transfer feature: ', X_origin_t['X'].shape)
    # print('size of predicted Output feature: ', X_pred['X_iv'].shape)
    # print('size of predicted Transfer feature: ', X_pred['X_gm'].shape)

    Gds_origin = np.gradient(X_origin_o['X'][0], vgs_bias_x_axis, axis=1)
    Gds_pred = np.gradient(X_pred['X_iv'][0], vgs_bias_x_axis, axis=1)
    Gds_pred_mlp = np.gradient(X_pred_mlp['X_iv'][0], vgs_bias_x_axis, axis=1)
    Gm_origin = np.gradient(X_origin_t['X'][0], vds_bias_x_axis, axis=1)
    Gm_pred = np.gradient(X_pred['X_gm'][0], vds_bias_x_axis, axis=1)
    Gm_pred_mlp = np.gradient(X_pred_mlp['X_gm'][0], vds_bias_x_axis, axis=1)


    # Sub-Fig 1 : Transfer
    # Sub-Fig 2 : Output
    # Sub-Fig 3 : Gm  (Transfer based)
    # Sub-Fig 4 : Gds (Output based)

    fig, ax = plt.subplots(2, 2, figsize=size) 

    for i in range(X_pred['X_gm'].shape[1]):
        ax[0,0].plot(vds_bias_x_axis, X_origin_t['X'][0,i,:], 'o', color = "black", 
                 zorder=10, label = None if (i+1) != X_pred['X_gm'].shape[1] else "Measured",alpha=0.3)
        ax[0,0].plot(vds_bias_x_axis, X_pred['X_gm'][0,i,:], color = "blue", label = None if (i+1) != X_pred['X_gm'].shape[1] else 'CVAE based Prediction', linewidth = 3.0)
        ax[0,0].plot(vds_bias_x_axis, X_pred_mlp['X_gm'][0,i,:], color = "red", label = None if (i+1) != X_pred['X_gm'].shape[1] else 'MLP based Prediction', linewidth = 3.0)
        ax[1,0].plot(vds_bias_x_axis, Gm_origin[i,:], 'o', color = "black", 
                 zorder=10, label = None if (i+1) != X_pred['X_gm'].shape[1] else "Measured",alpha=0.3)
        ax[1,0].plot(vds_bias_x_axis, Gm_pred[i,:], color = "blue", label = None if (i+1) != X_pred['X_gm'].shape[1] else 'CVAE based Prediction', linewidth = 3.0)
        ax[1,0].plot(vds_bias_x_axis, Gm_pred_mlp[i,:], color = "red", label = None if (i+1) != X_pred['X_gm'].shape[1] else 'MLP based Prediction', linewidth = 3.0)

    for i in range(X_pred['X_iv'].shape[1]):
        ax[0,1].plot(vgs_bias_x_axis, X_origin_o['X'][0,i,:], 'o', color = "black", 
                 zorder=10, label = None if (i+1) != X_pred['X_iv'].shape[1] else "Measured",alpha=0.3)
        ax[0,1].plot(vgs_bias_x_axis, X_pred['X_iv'][0,i,:], color = "blue", label = None if (i+1) != X_pred['X_iv'].shape[1] else 'CVAE based Prediction', linewidth = 3.0)
        ax[0,1].plot(vgs_bias_x_axis, X_pred_mlp['X_iv'][0,i,:], color = "red", label = None if (i+1) != X_pred['X_iv'].shape[1] else 'MLP based Prediction', linewidth = 3.0)
        ax[1,1].plot(vgs_bias_x_axis, Gds_origin[i,:], 'o', color = "black", 
                 zorder=10, label = None if (i+1) != X_pred['X_iv'].shape[1] else "Measured",alpha=0.3)
        ax[1,1].plot(vgs_bias_x_axis, Gds_pred[i,:], color = "blue", label = None if (i+1) != X_pred['X_iv'].shape[1] else 'CVAE based Prediction', linewidth = 3.0)
        ax[1,1].plot(vgs_bias_x_axis, Gds_pred_mlp[i,:], color = "red", label = None if (i+1) != X_pred['X_iv'].shape[1] else 'MLP based Prediction', linewidth = 3.0)

    y_name = deque(['Drain Current (A)', 'Drain Current (A)', 'Transconductance (S)', 'Gds (S)'])
    x_name = deque(['Gate Voltage (V)\n(a)','Drain Voltage (V)\n(b)','Gate Voltage (V)\n(c)','Drain Voltage (V)\n(d)'])
    for row in range(2):
        for col in range(2):
            ax[row,col].legend()
            ax[row,col].grid(True)
            ax[row,col].set_xlabel(x_name.popleft())
            ax[row,col].set_ylabel(y_name.popleft())
    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})
    fig.savefig("C:\\Users\\97427\\Desktop\\figure.svg", bbox_inches='tight', format='svg')
    print('saved figure under path : C:\\Users\\97427\\Desktop\\figure.svg ')





def two_fig_plot(X_origin_o, X_origin_t, X_pred, vgs_bias_x_axis, vds_bias_x_axis, size = (16,9), fontsize=None):

    # vgs_bias_x_axis = np.linspace(-1, 5, 61)
    # vds_bias_x_axis = np.linspace(-10, 0, 101)

    print('size of measured Output feature: ', X_origin_o['X'].shape)
    print('size of measured Transfer feature: ', X_origin_t['X'].shape)
    print('size of predicted Output feature: ', X_pred['X_iv'].shape)
    print('size of predicted Transfer feature: ', X_pred['X_gm'].shape)

    # Sub-Fig 1 : Transfer
    # Sub-Fig 2 : Output
    # Sub-Fig 3 : Gm  (Transfer based)
    # Sub-Fig 4 : Gds (Output based)

    fig, ax = plt.subplots(1, 2, figsize=size) 

    for i in range(X_pred['X_gm'].shape[1]):
        ax[0].plot(vds_bias_x_axis, X_origin_t['X'][0,i,:], 'o', color = 'red', label = None if (i+1) != X_pred['X_gm'].shape[1] else "Measured")
        ax[0].plot(vds_bias_x_axis, X_pred['X_gm'][0,i,:], color = 'blue', label = None if (i+1) != X_pred['X_gm'].shape[1] else 'DL based Prediction', linewidth = 3.0)

    for i in range(X_pred['X_iv'].shape[1]):
        ax[1].plot(vgs_bias_x_axis, X_origin_o['X'][0,i,:], 'o', color = 'red', label = None if (i+1) != X_pred['X_iv'].shape[1] else "Measured")
        ax[1].plot(vgs_bias_x_axis, X_pred['X_iv'][0,i,:], color = 'blue', label = None if (i+1) != X_pred['X_iv'].shape[1] else 'DL based Prediction', linewidth = 3.0)

    y_name = deque(['Drain Current (A)', 'Drain Current (A)', 'Transconductance (S)', 'Gds (S)'])
    x_name = deque(['Gate Voltage (V)\n(a)','Drain Voltage (V)\n(b)','Gate Voltage (V)\n(c)','Drain Voltage (V)\n(d)'])
    for row in range(2):
        ax[row].legend()
        ax[row].grid(True)
        ax[row].set_xlabel(x_name.popleft())
        ax[row].set_ylabel(y_name.popleft())
        
    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})
    fig.savefig("C:\\Users\\97427\\Desktop\\figure.svg", bbox_inches='tight', format='svg')
    print('saved figure under path : C:\\Users\\97427\\Desktop\\figure.svg ')