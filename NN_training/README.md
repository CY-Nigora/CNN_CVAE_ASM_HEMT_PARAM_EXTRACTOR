# Basic cmd for model training/inference
---


**parameter options:**

```python
def parse_args():
    parser = argparse.ArgumentParser(description='ASM-HEMT DNN Regressor')
    # 训练/通用
    parser.add_argument('--data', type=str, default=None, help='Path to .h5 dataset with X,Y')
    parser.add_argument('--outdir', type=str, default='runs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-split', type=float, default=0.15)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden', type=str, default='960,512,256,128', help='Comma-separated hidden sizes')
    parser.add_argument('--trials', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--no-onecycle', action='store_true')
    # 新增特性开关
    parser.add_argument('--aug-noise-std', type=float, default=0.015)
    parser.add_argument('--aug-prob', type=float, default=0.5)
    parser.add_argument('--no-multihead', action='store_true', help='Disable multi-head; default is ON')
    parser.add_argument('--no-uncertainty', action='store_true', help='Disable uncertainty weighting')
    parser.add_argument('--no-bounds', action='store_true', help='Disable output range enforcement in metrics/inference')
    # 推理
    parser.add_argument('--infer-run', type=str, default=None, help='Run dir containing best_model.pt & transforms.json')
    parser.add_argument('--input-npy', type=str, default=None)
    parser.add_argument('--input-h5', type=str, default=None)
    parser.add_argument('--index', type=int, default=None, help='Index for --input-h5; if None and X is 4D, run full batch')
    parser.add_argument('--save-csv', type=str, default=None)

    args = parser.parse_args()
    hidden = tuple(int(x) for x in args.hidden.split(','))

    cfg = TrainConfig(
        data=args.data,
        outdir=args.outdir,
        seed=args.seed,
        test_split=args.test_split,
        val_split=args.val_split,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden=hidden,
        trials=args.trials,
        patience=args.patience,
        num_workers=args.num_workers,
        compile=args.compile,
        use_onecycle=not args.no_onecycle,
                aug_noise_std=args.aug_noise_std,
        aug_prob=args.aug_prob,
        multihead=(not args.no_multihead),
        uncertainty_weighting=(not args.no_uncertainty),
        enforce_bounds=(not args.no_bounds),
    )
    return cfg, args
```


## version 2.x - Main model

- **Training:** 
``` bash
python asm_hemt_dnn.py --data dataset/train_dataset_7row.h5 ^
  --hidden 1024,512,256,128 ^
  --batch-size 256 --lr 9e-4 --weight-decay 5e-5 ^
  --dropout 0.25 --weight-decay 6e-4
  --max-epochs 180 --patience 10 ^
  --aug-noise-std 0.015 --aug-prob 0.5
```

- **real-time tensorboard GUI:**
```bash
tensorboard --logdir runs_test/log_merge_dataset_trail_test
```

- **Inference:**
```bash
# multi-samples H5（for only ids，use channel 0 automatically）
python asm_hemt_dnn.py ^
  --infer-run runs/version_2_0 ^
  --input-h5 "E:/personal_Data/Document of School/Uni Stuttgart/Masterarbeit/Code/param_regression/ADS_Parameter_Fitting_local/IV_param_regression/data_pre_processing/data/meas_file_reshape_7row.h5" ^
  --save-csv dataset/preds.csv
```


## New version :: double stage Model 

### Main model

- **Training:** 
```bash
python asm_hemt_2stage_dnn.py --data ./data/train.h5 ^
  --hidden 960,512,256,128 --batch-size 512 --lr 1e-3 ^
  --dropout 0.2 --weight-decay 3e-4 --max-epochs 120 --patience 12 ^
  --meas-h5 ./data/meas_u.h5 ^
  --lambda-cyc-sim 0.1 --lambda-cyc-meas 0.3 --cyc-warmup-epochs 15 ^
  --proxy-hidden 512,512 --proxy-activation gelu --proxy-norm layernorm

```


### Proxy Model $g$

- **Training:** 
``` bash
python asm_hemt_2stage_dnn.py --data ./data/train.h5 --train-proxy-only ^
  --proxy-hidden 384,384,256 --proxy-activation silu --proxy-norm layernorm ^
  --proxy-epochs 120 --proxy-lr 1e-3 --proxy-wd 1e-4

```

- **Inference:**
```bash
# multi-samples H5（for only ids，use channel 0 automatically）
python asm_hemt_2stage_dnn.py ^
  --infer-proxy-run runs/asm_hemt_dnn_YYYYMMDD-HHMMSS ^
  --proxy-input-h5 ./data/any_set_with_Y.h5 ^ 
  --proxy-index 12 ^
  --save-xhat-npy xhat_12.npy  
```