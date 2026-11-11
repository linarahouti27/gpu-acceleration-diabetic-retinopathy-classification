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

- **Conclusion du profilage (Moyenne sur 7 itérations)**

Le temps total moyen d'exécution est de ~2.2 ms par image. L'analyse de la répartition du temps montre que l'algorithme d'égalisation d'histogramme n'est pas le véritable goulot d'étranglement :

* **Le coût des conversions (L'overhead) :** Les étapes de conversion d'espace colorimétrique consomment la majorité du temps de calcul. La conversion RGB vers LAB (ligne 26) prend 21,5 % du temps, et le retour de LAB vers RGB (ligne 29) représente l'opération la plus lourde avec 33,3 % du temps.
* **Le coût de l'algorithme :** En comparaison, l'application effective du filtre CLAHE (ligne 28) ne représente que 25,4 % du temps total.

**Conclusion :** 
Le ralentissement (bottleneck) de cette fonction CPU provient majoritairement (~54,8 %) des allers-retours entre les espaces de couleurs nécessaires pour isoler le canal de luminance. Pour optimiser ce pipeline sur le GPU (A4.3), il sera crucial de calculer le CLAHE directement sur la luminance reconstituée ou d'éviter ces conversions complexes de l'API OpenCV.
