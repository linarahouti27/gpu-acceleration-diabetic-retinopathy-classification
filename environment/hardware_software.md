# Hardware and software (Table 1)

TODO(A1.2): run `python scripts/record_env.py`, then copy the values from
`results/environment.json` into this table. Add anything the script does not record
(power mode, laptop on battery or AC, background load, CPU/GPU temperature if you have it).

| Item | Value |
|---|---|
| CPU (model, cores, threads) |  13th Gen Intel Core i5-13420H,8,12 | 
| RAM | 8 GB | 
| GPU (model, VRAM, compute capability) | NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6.4 GB , 8.6 | 
| NVIDIA driver / CUDA runtime / cuDNN | 617.42/12.8/91900 | 
| OS | Windows 11 (10.0.26300) | 
| Python | 3.12.10 | 
| PyTorch / torchvision | 2.11.0+cu128/ 0.26.0+cu128 |  
| NumPy / OpenCV / scikit-learn | 2.5.4/5.0.0/1.9.1 |  
| Image size / batch size | [224, 224]/32 | 
| Data split (train/val/test, patient-level) |2116 / 425 / 458 images, split by patient (no patient appears in two splits) |
| Disk type of the dataset (SSD, HDD, network) |  SSD (KIOXIA KBG60ZNV512G, internal disk) |
| Power mode during measurements |  Best performance |
