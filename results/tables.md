# Tables (auto-generated from results/)

## Table 2 - Per-stage timings, CPU vs GPU

| stage | device | mean_s | std_s | imgs_per_s |
|---|---|---|---|---|
| loading | cpu | 2.072 | 0.014 | 30.884 |
| preprocessing | cpu | 0.134 | 0.003 | 478.386 |
| filter_gaussian | cpu | 0.018 | 0.001 | 3595.586 |
| filter_median | cpu | 0.041 | 0.001 | 1552.391 |
| filter_green | cpu | 0.007 | 0.000 | 9820.259 |
| filter_contrast | cpu | 0.113 | 0.004 | 564.365 |
| augmentation | cpu | 0.020 | 0.001 | 3135.724 |
| train_step | cpu | 1.621 | 0.067 | 39.491 |
| inference | cpu | 0.396 | 0.010 | 161.505 |

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

## Table 4 - Quality on the test set (baseline and final model only)

| run | amp | channels_last | checkpoint_epoch | val_acc | n_test | accuracy | macro_precision | macro_recall | macro_f1 | inference_time_s | inference_imgs_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.000 | 0.000 | 4.000 | 0.386 | 458.000 | 0.332 | 0.275 | 0.346 | 0.288 | 157.863 | 2.901 |

### Confusion matrix: baseline

| 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 41 | 9 | 0 | 13 | 15 |
| 32 | 35 | 0 | 11 | 30 |
| 32 | 11 | 0 | 20 | 26 |
| 33 | 10 | 0 | 21 | 37 |
| 6 | 7 | 0 | 14 | 55 |
