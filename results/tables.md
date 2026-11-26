# Tables (auto-generated from results/)

## Table 2 - Per-stage timings, CPU vs GPU

| stage | device | mean_s | std_s | imgs_per_s | speedup |
|---|---|---|---|---|---|
| loading | cpu | 2.047 | 0.146 | 31.271 |  |
| preprocessing | cpu | 0.135 | 0.003 | 473.231 |  |
| filter_gaussian | cpu | 0.017 | 0.000 | 3661.780 |  |
| filter_median | cpu | 0.041 | 0.000 | 1578.939 |  |
| filter_green | cpu | 0.006 | 0.000 | 10008.100 |  |
| filter_contrast | cpu | 0.111 | 0.003 | 574.234 |  |
| augmentation | cpu | 0.020 | 0.001 | 3246.971 |  |
| train_step | cpu | 1.438 | 0.089 | 44.498 |  |
| inference | cpu | 0.398 | 0.012 | 161.003 |  |
| loading | cuda | 0.000 | 0.000 | 384923.806 | 12309.245 |
| h2d_transfer | cuda | 0.001 | 0.000 | 51271.370 |  |
| preprocessing | cuda | 0.024 | 0.000 | 2645.376 | 5.590 |
| filter_gaussian | cuda | 0.003 | 0.000 | 21543.454 | 5.883 |
| filter_median | cuda | 0.319 | 0.004 | 200.603 | 0.127 |
| filter_green | cuda | 0.000 | 0.000 | 6926402.182 | 692.080 |
| filter_contrast | cuda | 0.030 | 0.000 | 2142.571 | 3.731 |
| augmentation | cuda | 0.009 | 0.000 | 7159.782 | 2.205 |
| train_step | cuda | 0.142 | 0.000 | 451.625 | 10.149 |
| inference | cuda | 0.050 | 0.000 | 1280.333 | 7.952 |
| train_step_amp | cuda | 0.072 | 0.001 | 894.562 | 20.103 |
| inference_amp | cuda | 0.027 | 0.000 | 2335.008 | 14.503 |

## Table 3 - End-to-end per configuration

| run | device | precision | preprocessing | steady epoch (s) | speedup vs baseline | gpu mem (MB) | val macro-F1 |
|---|---|---|---|---|---|---|---|
| baseline | cpu | fp32 | cpu | 112.266 | 1.000 |  | 0.336 |
| optimized | cuda | fp16 (amp) | gpu | 3.574 | 31.413 | 476.000 | 0.305 |
| final_r1 | cuda | fp16 (amp) | gpu | 6.170 | 18.194 | 616.400 | 0.541 |
| final_r2 | cuda | fp16 (amp) | gpu | 6.206 | 18.090 | 616.400 | 0.534 |
| final_r3 | cuda | fp16 (amp) | gpu | 6.167 | 18.204 | 616.400 | 0.549 |
| final (mean of 3 repeats) | cuda | fp16 (amp) | gpu | 6.181 +/- 0.022 | 18.162 |  | 0.541 |

## Table 4 - Quality on the test set (baseline and final model only)

| run | n_test | accuracy | macro_precision | macro_recall | macro_f1 | inference_imgs_per_s |
|---|---|---|---|---|---|---|
| baseline | 458.000 | 0.332 | 0.275 | 0.346 | 0.288 | 2.901 |
| final_r1 | 458.000 | 0.517 | 0.517 | 0.521 | 0.518 | 34.940 |

### Confusion matrix: baseline

| 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 41 | 9 | 0 | 13 | 15 |
| 32 | 35 | 0 | 11 | 30 |
| 32 | 11 | 0 | 20 | 26 |
| 33 | 10 | 0 | 21 | 37 |
| 6 | 7 | 0 | 14 | 55 |

### Confusion matrix: final_r1

| 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 38 | 21 | 14 | 5 | 0 |
| 29 | 54 | 13 | 9 | 3 |
| 14 | 19 | 28 | 26 | 2 |
| 5 | 7 | 21 | 53 | 15 |
| 0 | 0 | 2 | 16 | 64 |
