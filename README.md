# FAL-Net: A Hybrid LoRA-Enhanced Architecture with Spatial Attention for Robust Fingerprint Liveness Detection

This repository provides the implementation and reproducibility resources for:

> **FAL-Net: A Hybrid LoRA-Enhanced Architecture with Spatial Attention for Robust Fingerprint Liveness Detection**

The repository is intended to facilitate the independent verification and reproduction of the experimental results reported in the manuscript.

The reproducibility package includes the complete training and evaluation code, ablation-study implementation, repeated-experiment scripts, pretrained checkpoints, experimental configurations, and saved evaluation results.

---

## 1. Reproducibility Package

The complete reproducibility package is distributed through the combination of this GitHub repository and a dedicated Google Drive folder.

### GitHub Repository

The GitHub repository contains:

- the complete model implementation
- training scripts
- testing and evaluation scripts
- dataset loading and preprocessing code
- ablation-study implementation
- repeated-experiment scripts
- metric computation
- DET/ROC plotting utilities
- configuration files and documentation

### Pretrained Models and Experimental Results

The pretrained checkpoints and associated experimental results are available in the following Google Drive folder:

**FAL-Net Reproducibility Package**
https://drive.google.com/drive/u/0/folders/1xD8beNvmwn-r5pVSiCu4RkZg2Tu_w5nZ

The Drive folder contains the pretrained checkpoints corresponding to the experiments reported in the paper, together with the associated experimental results and configuration files.

---

## 2. Reproducibility Overview

The current reproducibility package has been substantially expanded and updated with respect to the previous version.

For every experimental result reported in the paper, the corresponding pretrained checkpoint is provided in the reproducibility package.

The checkpoints are organized using descriptive dataset- and experiment-specific names. Each checkpoint can therefore be associated with:

- the corresponding dataset
- the experimental protocol
- the evaluation script
- the corresponding result directory
- the saved metric results

The package is designed so that a reported result can be independently checked using the corresponding pretrained checkpoint without requiring model retraining.

The package also provides the scripts required to reproduce the experiments from training when desired.

---

## 3. Paper-to-Checkpoint-to-Results Mapping

The following table provides a comprehensive and exhaustive mapping between every table in the manuscript, the evaluated dataset and sensors, the exact architectural configuration, the pretrained checkpoint file, the execution run type, and the corresponding result artifacts.

| Paper result | Dataset | Protocol / Experiment | Checkpoint | Results |
|---|---|---|---|---|
| Table III | LivDet 2011 | Intra-sensor | Corresponding sensor-specific `.h5` checkpoint | Corresponding `results_*` directory |
| Table IV | LivDet 2013 | Intra-sensor | Corresponding sensor-specific `.h5` checkpoint | Corresponding `results_*` directory |
| Table V | LivDet 2015 | Known/unknown material | Corresponding sensor-specific `.h5` checkpoints | Corresponding `results_*` directories |
| Table VI | LivDet 2015 | Intra-sensor | Corresponding sensor-specific `.h5` checkpoints | Corresponding `results_*` directories |
| Table VII | LivDet 2017 | Intra-sensor / unknown material | `best_model_with_attention_Greenbit.h5`, `best_model_with_attention_Orcanthus.h5` | `results_Greenbit/`, `results_Orcanthus/` |
| Table VIII | LivDet 2015 | Cross-sensor | Corresponding source-sensor `.h5` checkpoints | Corresponding source-target result directories |
| Table IX | -- | Model complexity | Corresponding model configuration | `training_info.json` |
| Table X | Reported datasets | Repeated experiments | Corresponding checkpoints / repeated runs | Mean ± standard deviation results |
| Table XI | Reported dataset | Ablation study | Corresponding ablation checkpoints | Corresponding ablation results |

---

## 4. Repository Structure

The main repository is organized as follows:

```
FAL-Net/
│
├── data_loader.py               # Dataset loading, preprocessing, and 15% validation splitting
├── model_ablation.py            # Model architecture and ablation variants (ResNet50, SSA, LoRA)
├── train_ablation.py            # End-to-end model training script with early stopping
├── test_ablation.py             # Evaluation and metric calculation script
├── ablation_runner.py           # Consolidated ablation study suite (Table XI)
├── repeat_experiments.py        # Repeated training experiments across multiple seeds (Table X)
├── analyze_model_complexity.py  # Portable complexity & inference latency profiler (Table IX / XII)
├── plot_det_curve.py            # ISO/IEC 30107-3 compliant single-sensor DET curve plotting
├── plot_det_evolution.py        # ISO/IEC 30107-3 compliant multi-sensor DET comparison plotting
├── README.md                    # Detailed documentation and reproduction guide
```

---

## 5. Main Implementation Components

### 5.1 `data_loader.py`

This file contains the dataset loading and preprocessing procedures, including:

- image loading and normalization (224x224x3)
- preprocessing and data augmentation (rotations $\pm10^\circ$, translations $\pm5$ px, horizontal flip)
- label assignment (0 = Live / bona fide, 1 = Fake / presentation attack)
- training/validation splitting (15% validation split reserved for checkpoint selection)

The official test partition is not used for model selection. The test set remains reserved exclusively for the final evaluation.


### 5.2 `model_ablation.py`

This file implements the complete FAL-Net architecture (ResNet50 encoder, hierarchical U-Net decoder, Soft Spatial Attention gates, and LoRA residual branch in the Dense(512) classification layer) as well as all modular ablation configurations.

### 5.3 `train_ablation.py`

This script is used for model training with Adam optimizer ($\eta=10^{-5}$), `ReduceLROnPlateau`, and early stopping.

Example:

```bash
python train_ablation.py \
  --real-dir-train "<TRAIN_LIVE_DIRECTORY>" \
  --fake-dir-train "<TRAIN_SPOOF_DIRECTORY>" \
  --real-dir-test "<TEST_LIVE_DIRECTORY>" \
  --fake-dir-test "<TEST_SPOOF_DIRECTORY>" \
  --rank 8 \
  --ssa \
  --multiscale \
  --epochs 100 \
  --output "<OUTPUT_DIRECTORY>"
```

The exact configuration used for an experiment is recorded in `training_info.json`.

### 5.4 `test_ablation.py`

This script evaluates a trained or pretrained model on an independent test set.

Example:

```bash
python test_ablation.py \
  --model "<PATH_TO_MODEL>.h5" \
  --output "<OUTPUT_DIRECTORY>" \
  --config-note "proposed_r8" \
  --real-dir-test "<TEST_LIVE_DIRECTORY>" \
  --fake-dir-test "<TEST_SPOOF_DIRECTORY>"
```

The evaluation produces `results_metrics.json` containing: APCER, BPCER, ACE, Accuracy, EER, BPCER@10, and BPCER@100.

### 5.5 `ablation_runner.py`

This script executes the configurations used in the ablation study. It allows the different architectural and training configurations to be evaluated using a consistent experimental procedure.

The results generated by the ablation experiments are used for the analysis reported in Table XI.

### 5.6 `repeat_experiments.py`

This script performs repeated training runs across multiple random seeds (default: 3 runs) to evaluate inter-run variance and produce the Mean $\pm$ Standard Deviation metrics reported in Table X.

### 5.7 `plot_det_curve.py` and `plot_det_evolution.py`
These scripts plot Detection Error Trade-off (DET) curves strictly adhering to ISO/IEC 30107-3 reporting standards:
- **Horizontal Axis (X):** APCER (%) [Attack Presentation Classification Error Rate]
- **Vertical Axis (Y):** BPCER (%) [Bona-fide Presentation Classification Error Rate]

## 5.8. Computational-Efficiency Benchmark

Computational-efficiency measurements are generated by the ablation workflow using `benchmark_efficiency.py`.
---

## 6. Dataset Organization

The experiments use the following LivDet datasets:

- LivDet 2011
- LivDet 2013
- LivDet 2015
- LivDet 2017

The datasets are not redistributed with this repository. Users should obtain the corresponding datasets from their official sources and organize them according to the directory structures expected by the scripts.

The exact directory names used by the experiments are specified in the corresponding commands and configuration files.

---

## 7. Label Convention and Evaluation Metrics

The implementation uses the following label convention:

```
0 = Live / bona fide
1 = Fake / presentation attack
```

This convention is consistently applied in the data loading and evaluation procedures.

**APCER** — Attack Presentation Classification Error Rate:

```
APCER = N(attack classified as bona fide) / N(attack)
```

**BPCER** — Bona Fide Presentation Classification Error Rate:

```
BPCER = N(bona fide classified as attack) / N(bona fide)
```

**ACE** — Average Classification Error:

```
ACE = (APCER + BPCER) / 2
```

**EER** — Equal Error Rate: the operating point at which APCER and BPCER are equal.

**BPCER at fixed APCER** — the evaluation also reports BPCER at fixed APCER operating points, including `BPCER@10` and `BPCER@100`, corresponding to the fixed APCER operating points defined in the experimental protocol.

---

## 8. Main Training Configuration

The main FAL-Net configuration used in the experiments is documented in the corresponding `training_info.json` files.

---

## 9. Reproducing Table III — LivDet 2011

Table III reports the results obtained on LivDet 2011 using the intra-sensor protocol.

The reported sensors are: Biometrika, Digital, Italdata, Sagem.

For each sensor, the reproducibility package provides the corresponding pretrained checkpoint. The corresponding result directory contains the evaluation outputs. For example:

```
LivDet_2011/
│
├── best_model_with_attention_Biometrika.h5
├── best_model_with_attention_Digital.h5
├── best_model_with_attention_Italdata.h5
├── best_model_with_attention_Sagem.h5
│
├── results_Biometrika/
├── results_Digital/
├── results_Italdata/
└── results_Sagem/
```

The results can be independently checked using:

```bash
python test_ablation.py \
  --model "<CHECKPOINT>.h5" \
  --output "<RESULT_DIRECTORY>" \
  --real-dir-test "<LIVE_TEST_DIRECTORY>" \
  --fake-dir-test "<SPOOF_TEST_DIRECTORY>"
```

The resulting `results_metrics.json` file contains the computed evaluation metrics.

---

## 10. Reproducing Table IV — LivDet 2013

Table IV reports the results obtained on LivDet 2013 using the intra-sensor protocol.

The reported sensors are: Biometrika, Italdata.

The corresponding pretrained checkpoints are provided in the reproducibility package. Example organization:

```
LivDet_2013/
│
├── best_model_with_attention_Biometrika.h5
├── best_model_with_attention_Italdata.h5
│
├── results_Biometrika/
└── results_Italdata/
```

The results are evaluated using:

```bash
python test_ablation.py \
  --model "<CHECKPOINT>.h5" \
  --output "<RESULT_DIRECTORY>" \
  --real-dir-test "<LIVE_TEST_DIRECTORY>" \
  --fake-dir-test "<SPOOF_TEST_DIRECTORY>"
```

The corresponding `results_metrics.json` files contain the metrics reported in Table IV.

---

## 11. Reproducing Tables V and VI — LivDet 2015

The LivDet 2015 experiments include intra-sensor evaluation and the material-based evaluation protocols reported in the manuscript.

The sensors include: CrossMatch, Digital Persona, GreenBit, Hi-Scan.

The corresponding pretrained checkpoints are provided in the reproducibility package.

### 11.1 Table V — Known and Unknown Materials

The known-material and unknown-material evaluation results are obtained using their corresponding test partitions and pretrained checkpoints.

The package contains separate result directories for the different experimental configurations. Example:

```
LivDet_2015/
│
├── known/
│   └── ...
│
├── unknown/
│   └── ...
│
└── results_*/
```

The corresponding evaluation can be performed using:

```bash
python test_ablation.py \
  --model "<CHECKPOINT>.h5" \
  --output "<RESULT_DIRECTORY>" \
  --real-dir-test "<LIVE_TEST_DIRECTORY>" \
  --fake-dir-test "<SPOOF_TEST_DIRECTORY>"
```

The resulting `results_metrics.json` files provide the metrics used for Table V.

### 11.2 Table VI — Intra-Sensor Accuracy

Table VI reports the intra-sensor accuracy comparison for LivDet 2015.

The corresponding checkpoints and evaluation results are provided for the reported sensors in Ablation Study directory. The metrics can be independently recomputed using `test_ablation.py`.

---

## 12. Reproducing Table VII — LivDet 2017

Table VII reports the LivDet 2017 evaluation.

The reported sensors are: GreenBit, Orcanthus.

The reproducibility package contains the corresponding checkpoints and result directories. Example:

```
LivDet_2017/
│
├── best_model_with_attention_Greenbit.h5
├── best_model_with_attention_Orcanthus.h5
│
├── results_Greenbit/
└── results_Orcanthus/
```

The Greenbit checkpoint corresponds to the GreenBit experiment reported in the paper. The Orcanthus checkpoint corresponds to the Orcanthus experiment reported in the paper. The corresponding result directories contain the evaluation outputs associated with these checkpoints.

To independently verify a result:

```bash
python test_ablation.py \
  --model "best_model_with_attention_Greenbit.h5" \
  --output "results_Greenbit/" \
  --real-dir-test "<GREENBIT_LIVE_TEST_DIRECTORY>" \
  --fake-dir-test "<GREENBIT_SPOOF_TEST_DIRECTORY>"
```

For Orcanthus:

```bash
python test_ablation.py \
  --model "best_model_with_attention_Orcanthus.h5" \
  --output "results_Orcanthus/" \
  --real-dir-test "<ORCANTHUS_LIVE_TEST_DIRECTORY>" \
  --fake-dir-test "<ORCANTHUS_SPOOF_TEST_DIRECTORY>"
```

The resulting metrics correspond to the evaluation reported in Table VII.

---

## 13. Reproducing Table VIII — Cross-Sensor Generalization

Cross-sensor evaluation is performed on LivDet 2015. In this protocol, the model is trained on one sensor and evaluated on another sensor.

The source and target sensors are explicitly indicated by the experiment directory names. For example:

```
Source: GreenBit
Target: Hi-Scan
```

The trained checkpoint is then evaluated on the target sensor:

```bash
python test_ablation.py \
  --model "best_model_<SOURCE_TARGET_DIRECTORY>.h5" \
  --output "<SOURCE_TARGET_DIRECTORY>" \
  --real-dir-test "<TARGET_LIVE_TEST_DIRECTORY>" \
  --fake-dir-test "<TARGET_SPOOF_TEST_DIRECTORY>"
```

The corresponding result directory contains the metrics reported in Table VIII. The source-target organization makes it possible to identify the training sensor and evaluation sensor for each cross-sensor experiment.

---

## 14. Reproducing Table IX & XII — Model Complexity and Profiling

Table IX and Table XII report the model complexity and computational profiling comparison.

The model configurations, trainable parameter counts, FLOPs, and inference latencies can be verified using:
```bash
python benchmark_efficiency.py
```

The resulting metrics are also summarized in `training_info.json` and `full_summary.json`.

---

## 15. Reproducing Table X — Repeated Experiments

Table X reports the biometric evaluation metrics using repeated experiments to estimate mean $\pm$ standard deviation across 3 independent training runs.

The repeated experiments are implemented in `repeat_experiments.py`:

```bash
python repeat_experiments.py \
  --real-dir-train "<CROSSMATCH_LIVE_TRAIN_DIRECTORY>" \
  --fake-dir-train "<CROSSMATCH_SPOOF_TRAIN_DIRECTORY>" \
  --real-dir-test "<CROSSMATCH_LIVE_TEST_DIRECTORY>" \
  --fake-dir-test "<CROSSMATCH_SPOOF_TEST_DIRECTORY>" \
  --sensor CrossMatch
```

The generated results contain the individual-run metrics and the aggregated mean and standard deviation.

The metrics considered include: APCER, BPCER, ACE, EER, BPCER@10, BPCER@100.

The resulting aggregated values are used for Table X.

---

## 16. Reproducing Table XI — Ablation Study

Table XI evaluates the contribution of the individual modules of FAL-Net on LivDet-2015.

The ablation configurations are executed via `ablation_runner.py`.

Example:

```bash
python ablation_runner.py \
  --config "<ABLATION_CONFIGURATION>" \
  --dataset "<DATASET>"
```

The corresponding model checkpoints and result files are stored in the experiment-specific directories.

The metric results are stored in `results_metrics.json`, and the configurations are documented in `training_info.json`.

---

## 17. Reproducing DET Curves

The raw operating point data are stored in `roc_data.npy`.

The DET curves can be generated using:

```bash
python plot_det_curve.py
```

For comparison of DET curves across experiments or sensors:

```bash
python plot_det_evolution.py
```

The plotting scripts use the stored evaluation data and do not require model retraining.

---

## 18. Result Files

Each experiment-specific result directory may contain the following files:

```
results_<experiment>/
│
├── results_metrics.json
├── training_info.json
└── roc_data.npy
```

**`results_metrics.json`** — This file stores the evaluation metrics generated by the corrected evaluation implementation. Depending on the experiment, it may include: APCER, BPCER, ACE, Accuracy, EER, BPCER@10, BPCER@100.

**`training_info.json`** — This file documents the experimental configuration, including relevant training and model parameters.

**`roc_data.npy`** — This file contains the data required to generate ROC/DET curves and evaluate operating points.

---

## 19. Independent Verification Without Retraining

One of the main objectives of the updated reproducibility package is to allow independent verification without requiring the reviewer or user to retrain the model.

The basic procedure is:

```
1. Download the corresponding pretrained checkpoint
                ↓
2. Obtain and prepare the corresponding dataset
                ↓
3. Run test_ablation.py
                ↓
4. Obtain results_metrics.json
                ↓
5. Compare the computed metrics with the paper
```

This procedure allows the reported evaluation metrics to be independently verified using the released checkpoint and the corrected evaluation implementation.

---

## 20. Pretrained Checkpoints

Pretrained checkpoints are provided for the experiments reported in the manuscript.

The naming convention is designed to make the correspondence between the checkpoint and the experimental setting explicit. Examples include:

```
best_model_with_attention_Greenbit.h5
best_model_with_attention_Orcanthus.h5
best_model_with_attention_Italdata.h5
best_model_with_attention_Biometrika.h5
```

The checkpoint files are organized together with experiment-specific result directories. For example:

```
LivDet_2017/
├── best_model_with_attention_Greenbit.h5
├── best_model_with_attention_Orcanthus.h5
├── results_Greenbit/
└── results_Orcanthus/
```

This organization allows each reported result to be directly traced to its corresponding checkpoint.

---

## 21. Recommended Reproduction Workflow

For users who want to reproduce a result reported in the paper, the recommended workflow is:

**Step 1 — Obtain the code**

Clone the GitHub repository:

```bash
git clone https://github.com/Sonda-Ammar/
cd FAL-Net
```

**Step 2 — Download the pretrained checkpoint**

Open the Google Drive reproducibility package:
https://drive.google.com/drive/u/0/folders/1xD8beNvmwn-r5pVSiCu4RkZg2Tu_w5nZ

Download the checkpoint corresponding to the desired experiment.

**Step 3 — Prepare the dataset**

Obtain the corresponding LivDet dataset and organize the test data according to the directory structure expected by the evaluation script.

**Step 4 — Run the evaluation**

```bash
python test_ablation.py \
  --model "<CHECKPOINT>.h5" \
  --output "<OUTPUT_DIRECTORY>" \
  --real-dir-test "<LIVE_TEST_DIRECTORY>" \
  --fake-dir-test "<SPOOF_TEST_DIRECTORY>"
```

**Step 5 — Inspect the results**

The evaluation produces `results_metrics.json`. The resulting metrics can then be compared with the corresponding table in the manuscript.

---

## 22. Contact

For questions concerning the implementation, datasets, or reproducibility package, please contact me : sonda.ammarisims.usf.tn

---

## 23. Citation

If you use FAL-Net or this implementation in your research, please cite the corresponding paper:

```bibtex
@article{FALNet,
  title={FAL-Net: A Hybrid LoRA-Enhanced Architecture with Spatial Attention for Robust Fingerprint Liveness Detection},
  author={Sonda Ammar Bouhamed, Mouna Medhioub, Manel Ayadi, Nesrine Masmoudi, Tagrid Abdullah N. Alshalali},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2026}
}
```
