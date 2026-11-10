# Lab notebook

One entry per experiment. Write the prediction BEFORE you run anything. A wrong prediction with a good explanation is worth more than a right one you cannot explain.

## Entry template

```
### <assignment id> <short title>          date: <yyyy-mm-dd>
Config / commit:      <config file and git commit hash>
Question:             <what do you want to find out?>
Prediction:           <number or ranking, and the reason, written before the run>
Command:              <exact command line>
Result:               <measured numbers: mean +/- std, n repeats, cold vs steady>
Prediction vs result: <what was different?>
Explanation:          <why? what evidence supports it (profiler output, a second measurement)?>
Decision:             <keep / drop the change, and what you do next>
```

## A1.3 Baseline configuration review

TODO(A1.3): answers to the questions listed at the top of `configs/baseline.yaml`.

## Entries

(add yours below)


### A2.3 : Prédiction de la répartition du temps (Baseline CPU)
- **Prédiction :** Le prétraitement (CLAHE, crop, normalization) représente la part la plus importante du temps par époque, en raison des conversions d'espaces colorimétriques sur le CPU et de l'absence de cache initial.
- **Justification :** Les opérations OpenCV en séquentiel bloquent le pipeline avant même la passe avant (forward).

### A2.7 : Profilage fin avec line_profiler
- **Résultat :** Le profilage de `apply_clahe` a révélé que la ligne `cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)` consomme **91,0 % du temps total** de la fonction.
- **Comparaison :** Un profilage global classique (cProfile) aurait simplement indiqué que la fonction apply_clahe prend environ 0,1 seconde au total, sans plus de détails.
- **Conclusion :** Ce profilage prouve que si l'on veut accélérer ce pipeline sur GPU, il ne suffit pas de réécrire l'algorithme CLAHE : il faut repenser l'ensemble du traitement en évitant les allers-retours coûteux entre les espaces de couleurs sur le CPU, ou en exploitant un pipeline de tenseurs entièrement sur le GPU.
