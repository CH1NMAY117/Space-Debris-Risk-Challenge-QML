# Space Debris Collision-Risk Classification with a Quantum SVM

A Qiskit notebook that classifies the collision risk of space-debris encounters with a **quantum kernel support vector machine (QSVM)** and compares it with a classical RBF-kernel SVM.

This was the final project of an 8-week quantum machine learning course run by the Quantum Tech Community and 4i Labs at IIT Guwahati, completed within one week. [Open the notebook in Colab](https://colab.research.google.com/github/CH1NMAY117/Space-Debris-Risk-Challenge-QML/blob/main/QML_Chinmay_Qiskit.ipynb).

## Approach

1. **Data:** a labelled training table (5,000 rows, 40 features, target `risk_level`) and an unlabelled test table (300 rows).
2. **Features:** the first 12 feature columns, standardised. (The notebook comment calls this "top 12", but columns are taken in file order, not ranked by importance.)
3. **Classical baseline:** SVM with RBF kernel and balanced class weights, trained on an 80/20 split.
4. **Quantum model:** `QSVC` with a `ZZFeatureMap` (12 qubits, 1 repetition) evaluated on the Aer simulator with 512 shots. Because simulating the kernel is expensive, it uses a stratified 1,200-row subset, split 80/20, so it trains on **960 samples with 12 features**. Training took about 3.3 hours on Colab.
5. **Evaluation:** balanced accuracy and macro F1 on the validation split.
6. **Prediction:** the test set is scored in chunks of 100 with resume support, then merged into `predictions.csv`.

## Data is not included

The challenge dataset (`train.csv`, `test.csv`) belongs to the course organisers and is **not** in this repository, so the notebook cannot be run end to end as is. The notebook expects the files in `MyDrive/QMLChallenge/` on Google Drive. Edit the path in *STEP 4* to point to your own copy of a similar table with a `risk_level` column.

## Reproducing the environment

The notebook uses older Qiskit interfaces (`QuantumKernel`, `QuantumInstance`) that were deprecated and later removed, so the versions are pinned:

```bash
pip install qiskit==0.44.0 qiskit-aer==0.12.0 qiskit-machine-learning==0.6.1 scikit-learn pandas numpy tqdm
```

## Limitations

- A single validation split is used and no confidence intervals are reported, so the quantum and classical scores should not be treated as a rigorous comparison.
- The QSVM sees fewer samples than the classical model (960 against the full training split), so the two are not trained on equal data.
- Feature selection is by column order, not by importance.
- Simulator only; no quantum hardware was used.

## Licence

[MIT](LICENSE)
