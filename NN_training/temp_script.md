
proxy生成:

python asm_hemt_2stage_dnn.py --data dataset/training/Bidi_Transfer_log_3_merge.h5 ^
  --train-proxy-only  --outdir temp_Bidi_cvae --proxy-hidden 512,384,256 --proxy-lr 1.33e-4 --proxy-wd 6e-5




proxy 推理：

python asm_hemt_2stage_dnn.py ^
  --infer-proxy-run runs_g/hidden_2_net ^
  --proxy-input-h5 dataset/train_dataset_7row.h5 ^
  --proxy-index 12 ^
  --save-xhat-npy xhat.npy



onecycle自适应学习率 (训练proxy):

python asm_hemt_2stage_dnn.py --data dataset/training/stage_1_fpml_data.h5 ^
  --hidden 960,512,256,128 --batch-size 512 --lr 1e-3 ^
  --dropout 0.2 --weight-decay 3e-4 --max-epochs 120 --onecycle-epochs 120 --patience 12 ^
  --meas-h5 dataset/training/meas_data.h5 ^
  --lambda-cyc-sim 0.1 --lambda-cyc-meas 0.1 --cyc-warmup-epochs 15 ^
  --outdir runs_test --proxy-hidden 512,384,256 --proxy-lr 1.33e-4 --proxy-wd 6e-5


onecycle自适应学习率 (不训练proxy):
python asm_hemt_2stage_dnn.py --data dataset/training/stage_1_full_data.h5 ^
  --hidden 960,512,256,128 --batch-size 512 --lr 1e-3 ^
  --dropout 0.2 --weight-decay 3e-4 --max-epochs 150 --onecycle-epochs 120 --patience 15 ^
  --meas-h5 dataset/training/meas_data.h5 ^
  --lambda-cyc-sim 0.1 --lambda-cyc-meas 0.1 --cyc-warmup-epochs 15 ^
  --outdir runs_2stage/temp_check --proxy-run runs_2stage/temp_check/best_128batch_512layer_1e-4



固定学习率：

  python asm_hemt_2stage_dnn.py --data "E:\personal_Data\Document of School\Uni Stuttgart\Masterarbeit\Code\param_regression\ADS_Parameter_Fitting\IV_param_regression\data_gen\data\log_normal_dataset\noisy_data\train_log_dataset_7row_double.h5" ^
  --hidden 960,512,256,128 --batch-size 512 --lr 1e-3 ^
  --dropout 0.2 --weight-decay 3e-4 --max-epochs 150 --patience 12 ^
  --meas-h5 dataset/meas_data.h5 ^
  --lambda-cyc-sim 8 --lambda-cyc-meas 14 --cyc-warmup-epochs 33 ^
  --proxy-hidden 512,384,256 --proxy-lr 5e-4 --outdir runs_2stage



全量微调：

python asm_hemt_2stage_dnn.py --data dataset/training/stage_2_ml_data.h5 ^
  --finetune-from runs_2stage/ft_1st_stage --ft-use-prev-transforms ^
  --lr 5e-3 --dropout 0.2 --weight-decay 1e-5 --max-epochs 120 --onecycle-epochs 120 --patience 12 ^
  --meas-h5 dataset/training/meas_data.h5 ^
  --lambda-cyc-sim 5 --lambda-cyc-meas 5 --cyc-warmup-epochs 15 ^
  --outdir runs_2stage --proxy-run runs_2stage/best_g_fpml

只微调输出头（冻结干路）：

python asm_hemt_2stage_dnn.py --data dataset/training/stage_2_ml_data.h5 ^
  --finetune-from runs_2stage/ft_1st_stage --ft-use-prev-transforms --ft-freeze-trunk ^
  --lr 8e-3 --dropout 0.2 --weight-decay 1e-4 --max-epochs 120 --onecycle-epochs 120 --patience 10 ^
  --meas-h5 dataset/training/meas_data.h5 ^
  --lambda-cyc-sim 20 --lambda-cyc-meas 20 --cyc-warmup-epochs 15 ^
  --outdir runs_2stage --proxy-run runs_2stage/best_g_fpml





best case:

python asm_hemt_2stage_dnn.py --data dataset/training/stage_x_ml_full_data.h5 ^
  --train-proxy-only  --outdir model/runs_2stage_3rd_version/proxy_g --proxy-hidden 512,512,512,512 --proxy-batch-size 2048^
  --proxy-lr 2.0e-4 --proxy-wd 5e-5 --proxy-patience 25 --proxy-epochs 200 --proxy-seed 3943386831

stage-1:

python asm_hemt_2stage_dnn.py --data dataset/training/stage_x_ml_full_data.h5 ^
  --hidden 960,512,256,128 --batch-size 512 --lr 1.7e-4 ^
  --dropout 0.0 --weight-decay 2e-4 --max-epochs 350 --onecycle-epochs 300 --patience 25 ^
  --meas-h5 dataset/training/meas_data.h5 ^
  --lambda-cyc-sim 1.2 --lambda-cyc-meas 0.8 --cyc-warmup-epochs 110 ^
  --aug-noise-std 0.0 --aug-prob 0 ^
  --sup-weight 0.0 --prior-l2 1.0e-2 --prior-bound 3e-3 --prior-bound-margin 0.05 ^
  --es-metric val_cyc_sim --es-min-delta 5e-6 ^
  --outdir temp_1 --proxy-run model/runs_2stage_3rd_version/proxy_g/best_g_log_bigger ^
  --trust-alpha 0.18 --trust-alpha-meas 0.08 --trust-tau 1.00 ^
  --cyc-meas-knn-weight --cyc-meas-knn-gamma 0.8 ^
  --diag 

  stage-1: directly in meas-like dataset

python asm_hemt_2stage_dnn.py --data dataset/training/stage_2_ml_data.h5 ^
  --hidden 960,512,256,128 --batch-size 128 --lr 9e-4 ^
  --dropout 0.2 --weight-decay 3e-4 --max-epochs 200 --onecycle-epochs 200 --patience 20 ^
  --meas-h5 dataset/training/meas_data.h5 ^
  --lambda-cyc-sim 1.0 --lambda-cyc-meas 1.0 --cyc-warmup-epochs 60 ^
  --aug-noise-std 0.0 --aug-prob 0 ^
  --sup-weight 0.0 --prior-l2 8e-4 --prior-bound 2e-3 --prior-bound-margin 0.02 ^
  --es-metric val_cyc_meas --es-min-delta 5e-6 ^
  --outdir model/runs_2stage_3rd_version/stage_1 --proxy-run model/runs_2stage_3rd_version/proxy_g/proxy_run_20250916-112536



stage-2:

python asm_hemt_2stage_dnn.py --data dataset/training/stage_2_ml_data.h5 ^
  --finetune-from temppppp\asm_hemt_dnn_20250912-141329 --ft-use-prev-transforms ^
  --lr 8e-4 --dropout 0.2 --weight-decay 1e-5 --max-epochs 150 --onecycle-epochs 150 --patience 15 ^
  --meas-h5 dataset/training/meas_data.h5 ^
  --lambda-cyc-sim 0.4 --lambda-cyc-meas 0.1 --cyc-warmup-epochs 30 ^
  --aug-noise-std 0.0 --aug-prob 0 ^
  --sup-weight 0.0 --prior-l2 1e-3 --prior-bound 3e-4 ^
  --es-metric val_cyc_sim --es-min-delta 5e-6 ^
  --outdir temppppp_2 --proxy-run model/runs_2stage_2st_version/proxy_g/proxy_run_20250912-134117


 diag cmd:

 python ../data_viewer/plot_diag.py --csv temp_Bidi_cvae/version_1_9/diag_test.csv --outdir temp_Bidi_cvae/version_1_9/diag_plots ^
 --knn_thresh 2.0 --cyc_good 0.15 --cyc_warn 0.25


# symmetric
python asm_hemt_cvae.py --data dataset/training/stage_x_ml_symmetric.h5 ^
  --outdir temp_2 --proxy-run model/cvae_symmetric/proxy/proxy_v1 ^
  --meas-h5 dataset/training/meas_data.h5 ^
  --hidden 960,512,256 --batch-size 512 --lr 1.2e-4 ^
  --dropout 0.0 --weight-decay 2e-4 --max-epochs 300 --onecycle-epochs 300 --patience 40 ^
  --lambda-cyc-sim 1.2 --lambda-cyc-meas 0.8 --cyc-warmup-epochs 110 ^
  --aug-noise-std 0 --aug-prob 0 ^
  --sup-weight 1.0 --prior-l2 1.0e-2 --prior-bound 3e-3 --prior-bound-margin 0.05 ^
  --es-metric val_cyc_meas --es-min-delta 5e-6 ^
  --trust-alpha 0.18 --trust-alpha-meas 0.08 --trust-tau 1.00 ^
  --cyc-meas-knn-weight --cyc-meas-knn-gamma 0.8 ^
  --latent-dim 32 --kl-beta 0.1 ^
  --diag --z-sample-mode mean  

# unsymmetric diffusion
python asm_hemt_diffusion.py --data dataset/training/stage_x_ml_full_data.h5 ^
  --outdir temp_diffusion --proxy-run model/runs_2stage_3rd_version/proxy_g/best_g_log_bigger ^
  --meas-h5 dataset/training/meas_data.h5 ^
  --hidden 960,512,256 --batch-size 512 --lr 1.45e-4 ^
  --dropout 0.0 --weight-decay 2e-4 --max-epochs 300 --onecycle-epochs 300 --patience 130 ^
  --lambda-cyc-sim 1.2 --lambda-cyc-meas 0.8 --cyc-warmup-epochs 90 ^
  --aug-noise-std 0.0 --aug-prob 0 ^
  --sup-weight 0.1 --prior-l2 1.0e-2 --prior-bound 3e-3 --prior-bound-margin 0.05 ^
  --es-metric val_cyc_sim_prior --es-min-delta 5e-6 ^
  --trust-alpha 0.18 --trust-alpha-meas 0.1 --trust-tau 1.00 ^
  --cyc-meas-knn-weight --cyc-meas-knn-gamma 0.8 ^
  --diag 

  # Bidi CVAE
python code/training/asm_hemt_cvae_Bidi.py --data dataset/training/Bidi_Transfer_log_merge.h5 ^
  --outdir temp_Bidi_cvae --proxy-run temp_Bidi_cvae/proxy ^
  --meas-h5 dataset/training/meas_smoothed_Bidi_Transfer.h5 ^
  --hidden 960,512,256 --batch-size 512 --lr 1.55e-4 ^
  --dropout 0.0 --weight-decay 2e-4 --max-epochs 300 --onecycle-epochs 300 --patience 40 ^
  --lambda-cyc-sim 1.2 --lambda-cyc-meas 0.8 --cyc-warmup-epochs 110 ^
  --aug-noise-std 0 --aug-prob 0 ^
  --sup-weight 0.5 --prior-l2 1.0e-2 --prior-bound 3e-3 --prior-bound-margin 0.05 ^
  --es-metric val_cyc_meas --es-min-delta 5e-6 ^
  --trust-alpha 0.18 --trust-alpha-meas 0.08 --trust-tau 1.00 ^
  --cyc-meas-knn-weight --cyc-meas-knn-gamma 0.8 ^
  --latent-dim 32 --kl-beta 0.1 ^
  --diag --z-sample-mode mean  



# Bidi CVAE + noise
python code/training/asm_hemt_cvae_Bidi.py --data dataset/training/Bidi_Transfer_log_2_merge.h5 ^
  --outdir temp_Bidi_cvae --proxy-run temp_Bidi_cvae/proxy ^
  --meas-h5 dataset/training/meas_smoothed_Bidi_Transfer.h5 ^
  --hidden 1280,640,320 --batch-size 512 --lr 1.55e-4 ^
  --dropout 0.0 --weight-decay 2e-4 --max-epochs 300 --onecycle-epochs 300 --patience 40 ^
  --lambda-cyc-sim 1.2 --lambda-cyc-meas 0.8 --cyc-warmup-epochs 110 ^
  --sup-weight 0.9 --prior-l2 1.0e-2 --prior-bound 3e-3 --prior-bound-margin 0.05 ^
  --es-metric val_cyc_meas --es-min-delta 5e-6 ^
  --trust-alpha 0.18 --trust-alpha-meas 0.08 --trust-tau 1.50 ^
  --cyc-meas-knn-weight --cyc-meas-knn-gamma 0.8 ^
  --latent-dim 32 --kl-beta 0.2 ^
  --aug-prob 0.3 --aug-noise-std 0.02 ^
  --aug-gain-std 0.02 --aug-row-gain-std 0.015 --aug-smooth-window 7 --aug-schedule linear_decay --aug-final-scale 0.2 ^
  --best-of-k 0 --bok-warmup-epochs 40 --bok-target meas --bok-apply train ^
  --diag --z-sample-mode mean


python code/training/asm_hemt_cvae_Bidi.py --data dataset/training/Bidi_Transfer_log_2_merge.h5 ^
  --train-proxy-only  --outdir temp_Bidi_cvae --proxy-hidden 512,512,512,512 --proxy-batch-size 2048^
  --proxy-lr 2.0e-4 --proxy-wd 5e-5 --proxy-patience 25 --proxy-epochs 200 --proxy-seed 3943386831

# Unidi CVAE symmetric + noise
 python ../data_viewer/plot_diag.py --csv temp_Unidi_cvae_symmetric/version_1_4/diag_test.csv --outdir temp_Unidi_cvae_symmetric/version_1_4/diag_plots ^
 --knn_thresh 2.0 --cyc_good 0.15 --cyc_warn 0.25

python code/training/asm_hemt_cvae_Unidi.py --data dataset/training/Uni_Output_symmetric_merge_mask1.h5 ^
  --train-proxy-only  --outdir temp_Unidi_cvae_symmetric --proxy-hidden 512,512,512,512 --proxy-batch-size 2048^
  --proxy-lr 2.0e-4 --proxy-wd 5e-5 --proxy-patience 25 --proxy-epochs 200 --proxy-seed 3943386831

python code/training/asm_hemt_cvae_Unidi.py --data dataset/training/Uni_Output_symmetric_merge_mask1.h5 ^
  --outdir temp_Unidi_cvae_symmetric --proxy-run temp_Unidi_cvae_symmetric/proxy_mask1 ^
  --meas-h5 dataset/training/meas_data.h5 ^
  --hidden 1280,640,320 --batch-size 512 --lr 4e-4 ^
  --dropout 0.0 --weight-decay 2e-4 --max-epochs 300 --onecycle-epochs 300 --patience 40 ^
  --lambda-cyc-sim 1.2 --lambda-cyc-meas 0.8 --cyc-warmup-epochs 110 ^
  --sup-weight 0.9 --prior-l2 1.0e-2 --prior-bound 3e-3 --prior-bound-margin 0.05 ^
  --es-metric val_cyc_meas --es-min-delta 5e-6 ^
  --trust-alpha 0.18 --trust-alpha-meas 0.08 --trust-tau 2.0 ^
  --cyc-meas-knn-weight --cyc-meas-knn-gamma 0.8 ^
  --latent-dim 32 --kl-beta 0.2 ^
  --aug-prob 0.3 --aug-noise-std 0.02 ^
  --aug-gain-std 0.02 --aug-row-gain-std 0.015 --aug-smooth-window 7 --aug-schedule linear_decay --aug-final-scale 0.2 ^
  --best-of-k 0 ^
  --diag --z-sample-mode mean

## Unidi CVAE symmetric + 14params
python code/training/asm_hemt_cvae_Unidi.py --data dataset/training/Uni_Output_symmetric_mask1_14param.h5 ^
  --train-proxy-only  --outdir temp_Unidi_cvae_14param_1channel --proxy-hidden 512,512,512,512 --proxy-batch-size 2048^
  --proxy-lr 2.0e-4 --proxy-wd 5e-5 --proxy-patience 25 --proxy-epochs 200 --proxy-seed 3943386831

python code/training/asm_hemt_cvae_Unidi.py --data dataset/training/Uni_Output_symmetric_mask1_14param.h5 ^
  --outdir temp_Unidi_cvae_14param_1channel --proxy-run temp_Unidi_cvae_14param_1channel/proxy ^
  --meas-h5 dataset/training/meas_data.h5 ^
  --hidden 1280,640,320 --batch-size 512 --lr 4e-4 ^
  --dropout 0.0 --weight-decay 2e-4 --max-epochs 300 --onecycle-epochs 300 --patience 40 ^
  --lambda-cyc-sim 1.2 --lambda-cyc-meas 0.8 --cyc-warmup-epochs 110 ^
  --sup-weight 0.9 --prior-l2 1.0e-2 --prior-bound 3e-3 --prior-bound-margin 0.05 ^
  --es-metric val_cyc_meas --es-min-delta 5e-6 ^
  --trust-alpha 0.18 --trust-alpha-meas 0.08 --trust-tau 2.0 ^
  --cyc-meas-knn-weight --cyc-meas-knn-gamma 0.8 ^
  --latent-dim 32 --kl-beta 0.2 ^
  --aug-prob 0.3 --aug-noise-std 0.02 ^
  --aug-gain-std 0.02 --aug-row-gain-std 0.015 --aug-smooth-window 7 --aug-schedule linear_decay --aug-final-scale 0.2 ^
  --best-of-k 0 ^
  --diag --z-sample-mode mean


## Unidi CNN + CVAE symmetric + 14params
 python ../data_viewer/plot_diag.py --csv temp_Unidi_cvae_14param_2channel/version_1_4/diag_test.csv --outdir temp_Unidi_cvae_14param_2channel/version_1_4/diag_plots ^
 --knn_thresh 2.0 --cyc_good 0.15 --cyc_warn 0.25

python code/training/cvae_Unidi_CNN/main.py --data dataset/training/Uni_log_2Channel_mask1_2.h5 ^
  --train-proxy-only  --outdir temp_Unidi_cvae_14param_2channel --proxy-hidden 512,512,512,512 --proxy-batch-size 1024^
  --proxy-lr 2.0e-4 --proxy-wd 5e-5 --proxy-patience 25 --proxy-epochs 180 

### 1-stage strategy
python code/training/cvae_Unidi_CNN/main.py --data dataset/training/Uni_log_2Channel_mask1_2.h5 ^
  --outdir temp_Unidi_cvae_14param_2channel --proxy-run temp_Unidi_cvae_14param_2channel/proxy_dual_bigger ^
  --meas-h5 dataset/training/meas_smoothed_Uni_2Channel_wide.h5 ^
  --hidden 1280,640,320 --batch-size 256 --lr 2.5e-4 ^
  --feat-dim 256 ^
  --dropout 0.1 --weight-decay 1e-4 --max-epochs 300 --onecycle-epochs 300 --patience 100 ^
  --lambda-cyc-sim 10.0 --lambda-cyc-meas 0.1 --weight-iv 5.0 --weight-gm 1.0 --cyc-warmup-epochs 150 ^
  --sup-weight 1.0 --prior-l2 1.0e-2 --prior-bound 1e-2 --prior-bound-margin 0.05 ^
  --es-metric val_cyc_meas --es-min-delta 1e-5 ^
  --trust-alpha 1.0 --trust-alpha-meas 0.1 --trust-tau 1.0 ^
  --cyc-meas-knn-weight --cyc-meas-knn-gamma 0.8 ^
  --latent-dim 32 --kl-beta 0.5 ^
  --aug-prob 0.5 --aug-noise-std 0.015 ^
  --best-of-k 0 --cnn-dropout 0.1 ^
  --diag --z-sample-mode mean --phys-loss


### 2-stage strategy
python code/training/cvae_Unidi_CNN/main.py --data dataset/training/Uni_log_2Channel_mask1_mask2.h5 ^
  --outdir temp_Unidi_cvae_14param_2channel --proxy-run temp_Unidi_cvae_14param_2channel/proxy_dual_bigger ^
  --meas-h5 dataset/training/meas_smoothed_Uni_2Channel_wide.h5 ^
  --resume temp_Unidi_cvae_14param_2channel/version_2_12/best_model.pt ^
  --hidden 1280,640,320 --batch-size 256 --lr 1e-5 ^
  --feat-dim 256 ^
  --dropout 0.1 --weight-decay 1e-4 --max-epochs 100 --no-onecycle --patience 40 ^
  --lambda-cyc-sim 1.0 --lambda-cyc-meas 5.0 --weight-iv 6.0 --weight-gm 1.0 --cyc-warmup-epochs 150 ^
  --sup-weight 1.0 --prior-l2 1.0e-2 --prior-bound 1e-2 --prior-bound-margin 0.05 ^
  --es-metric val_cyc_meas --es-min-delta 1e-5 ^
  --trust-alpha 1.0 --trust-alpha-meas 0.1 --trust-tau 1.0 ^
  --cyc-meas-knn-weight --cyc-meas-knn-gamma 0.8 ^
  --latent-dim 32 --kl-beta 0.5 ^
  --aug-prob 0.0 --aug-noise-std 0.015 ^
  --best-of-k 0 --cnn-dropout 0.1 ^
  --diag --z-sample-mode mean




## Bidi CNN + CVAE symmetric + 11params
 python ../data_viewer/plot_diag.py --csv temp_Unidi_cvae_14param_2channel/version_1_4/diag_test.csv --outdir temp_Unidi_cvae_14param_2channel/version_1_4/diag_plots ^
 --knn_thresh 2.0 --cyc_good 0.15 --cyc_warn 0.25

python code/training/cvae_Bidi_CNN/main.py --data dataset/training/Bidi_merge_2Channel_mask1_11param.h5 ^
  --train-proxy-only  --outdir temp_Bidi_cvae_11param_2channel --proxy-hidden 512,512,512,512 --proxy-batch-size 1024^
  --proxy-lr 2.0e-4 --proxy-wd 5e-5 --proxy-patience 25 --proxy-epochs 180 

### 1-stage strategy
python code/training/cvae_Bidi_CNN/main.py --data dataset/training/Bidi_merge_2Channel_mask1_11param.h5 ^
  --outdir temp_Bidi_cvae_11param_2channel --proxy-run temp_Bidi_cvae_11param_2channel/proxy_dual ^
  --meas-h5 dataset/training/meas_smoothed_Bidi_2Channel.h5 ^
  --hidden 1280,640,320 --batch-size 256 --lr 2.0e-4 ^
  --feat-dim 256 ^
  --dropout 0.1 --weight-decay 1e-4 --max-epochs 300 --onecycle-epochs 300 --patience 100 ^
  --lambda-cyc-sim 10.0 --lambda-cyc-meas 0.1 --weight-iv 5.0 --weight-gm 1.0 --cyc-warmup-epochs 150 ^
  --sup-weight 1.0 --prior-l2 1.0e-2 --prior-bound 1e-2 --prior-bound-margin 0.05 ^
  --es-metric val_cyc_meas --es-min-delta 1e-5 ^
  --trust-alpha 1.0 --trust-alpha-meas 0.1 --trust-tau 1.0 ^
  --cyc-meas-knn-weight --cyc-meas-knn-gamma 0.8 ^
  --latent-dim 32 --kl-beta 0.5 ^
  --aug-prob 0.5 --aug-noise-std 0.015 ^
  --best-of-k 0 --cnn-dropout 0.1 ^
  --diag --z-sample-mode mean --phys-loss

### 2-stage strategy
python code/training/cvae_Bidi_CNN/main.py --data dataset/training/Bidi_merge_2Channel_mask1_11param.h5 ^
  --outdir temp_Bidi_cvae_11param_2channel --proxy-run temp_Bidi_cvae_11param_2channel/proxy_dual ^
  --meas-h5 dataset/training/meas_smoothed_Bidi_2Channel.h5 --resume temp_Bidi_cvae_11param_2channel/version_1_1/best_model.pt ^
  --hidden 1280,640,320 --batch-size 256 --lr 1.5e-5 ^
  --feat-dim 256 ^
  --dropout 0.0 --weight-decay 1e-4 --max-epochs 300 --onecycle-epochs 300 --patience 100 ^
  --lambda-cyc-sim 0.1 --lambda-cyc-meas 10.0 --weight-iv 5.0 --weight-gm 1.0 --cyc-warmup-epochs 150 ^
  --sup-weight 1.0 --prior-l2 1.0e-2 --prior-bound 1e-2 --prior-bound-margin 0.05 ^
  --es-metric val_cyc_meas --es-min-delta 1e-5 ^
  --trust-alpha 1.0 --trust-alpha-meas 0.1 --trust-tau 1.0 ^
  --cyc-meas-knn-weight --cyc-meas-knn-gamma 0.8 ^
  --latent-dim 32 --kl-beta 0.05 ^
  --aug-prob 0.0 --aug-noise-std 0.015 ^
  --best-of-k 0 --cnn-dropout 0.0 ^
  --diag --z-sample-mode mean --phys-loss

### TTO-inference
python code/training/cvae_Bidi_CNN/infer_tto_strict.py ^
  --cvae-run temp_Bidi_cvae_11param_2channel/version_1_1 ^
  --proxy-run temp_Bidi_cvae_11param_2channel/proxy_dual ^
  --meas-h5 dataset/training/meas_smoothed_Bidi_2Channel.h5 ^
  --save-to temp_Bidi_cvae_11param_2channel/version_1_1/tto_infer/tto_strict_result.csv ^
  --steps 1000 --lr 0.05


# pure MLP :: baseline Bidi
python code/training/pure_MLP_Bidi/main.py --data dataset/training/Bidi_merge_2Channel_mask1_11param.h5 ^
  --outdir temp_Bidi_pureMLP_2channel ^
  --hidden 1280,640,320 --batch-size 256 --lr 2.0e-4 ^
  --dropout 0.1 --weight-decay 1e-4 --max-epochs 300 --onecycle-epochs 300 --patience 100 


# pure MLP :: baseline Unidi
python code/training/pure_MLP_Unidi/main.py --data dataset/training/Uni_log_2Channel_mask1_2.h5 ^
  --outdir temp_Unidi_pureMLP_2channel ^
  --hidden 1280,640,320 --batch-size 256 --lr 2.0e-4 ^
  --dropout 0.1 --weight-decay 1e-4 --max-epochs 300 --onecycle-epochs 300 --patience 100 

















