import yaml
from src.data_loader import find_images, split_by_patient, patient_id
cfg = yaml.safe_load(open("configs/baseline.yaml"))
s = split_by_patient(find_images(cfg["data"]["root"]), cfg["data"]["train_frac"],
                     cfg["data"]["val_frac"], cfg["seed"])
for k, v in s.items():
    print(k, len(v), {c: sum(l == c for _, l in v) for c in range(5)}, len({patient_id(p) for p, _ in v}))
ids = {k: {patient_id(p) for p, _ in v} for k, v in s.items()}
names = list(ids)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(names[i], "&", names[j], "patients partagés :", len(ids[names[i]] & ids[names[j]]))