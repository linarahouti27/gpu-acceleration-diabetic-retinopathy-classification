# Tables (auto-generated from results/)

## Table 2 - Per-stage timings, CPU vs GPU

| stage | device | mean_s | std_s | imgs_per_s | speedup |
|---|---|---|---|---|---|
| loading | cpu | 1.962 | 0.035 | 32.627 |  |
| preprocessing | cpu | 0.128 | 0.004 | 499.637 |  |
| filter_gaussian | cpu | 0.017 | 0.001 | 3868.678 |  |
| filter_median | cpu | 0.042 | 0.001 | 1537.789 |  |
| filter_green | cpu | 0.007 | 0.001 | 9835.229 |  |
| filter_contrast | cpu | 0.106 | 0.003 | 604.033 |  |
| augmentation | cpu | 0.020 | 0.001 | 3149.231 |  |
| train_step | cpu | 1.322 | 0.120 | 48.405 |  |
| inference | cpu | 0.410 | 0.039 | 156.070 |  |
| loading | cuda | 0.003 | 0.001 | 21502.486 | 659.046 |
| h2d_transfer | cuda | 0.001 | 0.000 | 55240.035 |  |
| preprocessing | cuda | 0.025 | 0.000 | 2610.150 | 5.224 |
| filter_gaussian | cuda | 0.003 | 0.000 | 23784.925 | 6.148 |
| filter_median | cuda | 0.320 | 0.005 | 200.267 | 0.130 |
| filter_green | cuda | 0.000 | 0.000 | 245964.643 | 25.009 |
| filter_contrast | cuda | 0.030 | 0.000 | 2154.277 | 3.566 |
| augmentation | cuda | 0.009 | 0.000 | 7342.784 | 2.332 |
| train_step | cuda | 0.141 | 0.000 | 452.987 | 9.358 |
| inference | cuda | 0.050 | 0.000 | 1286.844 | 8.245 |
| train_step_amp | cuda | 0.068 | 0.000 | 935.539 | 19.327 |
| inference_amp | cuda | 0.027 | 0.000 | 2389.308 | 15.309 |

## Table 3 - End-to-end per configuration

| run | device | precision | preprocessing | steady epoch (s) | speedup vs baseline | gpu mem (MB) | val macro-F1 |
|---|---|---|---|---|---|---|---|
| baseline | cpu | fp32 | cpu | 112.266 | 1.000 |  | 0.336 |
| optimized | cuda | fp16 (amp) | gpu | 3.574 | 31.413 | 476.000 | 0.305 |
| exp_amp_off | cuda | fp32 | gpu | 5.964 | 18.825 | 720.700 | 0.332 |
| exp_augment | cuda | fp32 | cpu | 64.860 | 1.731 | 711.600 | 0.319 |
| exp_baseline_seed43 | cuda | fp32 | cpu | 55.331 | 2.029 | 711.600 | 0.314 |
| exp_cache_off | cuda | fp16 (amp) | gpu | 74.200 | 1.513 | 476.000 | 0.305 |
| exp_channelslast_off | cuda | fp16 (amp) | gpu | 4.110 | 27.314 | 475.800 | 0.301 |
| exp_cosine | cuda | fp32 | cpu | 60.479 | 1.856 | 711.600 | 0.329 |
| exp_gpupre_off | cuda | fp16 (amp) | cpu | 5.841 | 19.221 | 504.000 | 0.312 |
| exp_no_clahe | cuda | fp32 | cpu | 56.402 | 1.990 | 711.600 | 0.333 |
| exp_prefetchgpu_off | cuda | fp16 (amp) | gpu | 3.555 | 31.578 | 466.800 | 0.305 |
| exp_res160 | cuda | fp32 | cpu | 61.468 | 1.826 | 381.000 | 0.320 |
| exp_resnet18 | cuda | fp32 | cpu | 56.535 | 1.986 | 885.300 | 0.519 |
| exp_resnet18_lr1e-4 | cuda | fp32 | cpu | 55.597 | 2.019 | 885.300 | 0.510 |
| exp_resnet18_lr1e-4_augment | cuda | fp32 | cpu | 57.151 | 1.964 | 885.300 | 0.536 |
| final_r1 | cuda | fp16 (amp) | gpu | 6.170 | 18.194 | 616.400 | 0.541 |
| final_r2 | cuda | fp16 (amp) | gpu | 6.206 | 18.090 | 616.400 | 0.534 |
| final_r3 | cuda | fp16 (amp) | gpu | 6.167 | 18.204 | 616.400 | 0.549 |
| final (mean of 3 repeats) | cuda | fp16 (amp) | gpu | 6.181 +/- 0.022 | 18.162 |  | 0.541 |

## Table 4 - Quality on the test set (baseline and final model only)

| run | n_test | accuracy | macro_precision | macro_recall | macro_f1 | inference_imgs_per_s |
|---|---|---|---|---|---|---|
| baseline | 458.000 | 0.332 | 0.275 | 0.346 | 0.288 | 2.901 |

### Confusion matrix: baseline

| 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 41 | 9 | 0 | 13 | 15 |
| 32 | 35 | 0 | 11 | 30 |
| 32 | 11 | 0 | 20 | 26 |
| 33 | 10 | 0 | 21 | 37 |
| 6 | 7 | 0 | 14 | 55 |
