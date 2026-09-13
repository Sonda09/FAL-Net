# FAL-Net

**FAL-Net: A Hybrid LoRA-Enhanced Architecture with Spatial Attention for Robust Fingerprint Liveness Detection**

This repository contains the implementation and reproducibility resources associated with the FAL-Net paper submitted to *IEEE Transactions on Biometrics, Behavior, and Identity Science*.

The implementation supports model training, testing, ablation experiments, repeated experiments, DET-curve generation, and computational-efficiency evaluation.

---

## 1. Repository Contents

The main scripts are:

```text
train_ablation.py
test_ablation.py
ablation_runner.py
repeat_experiments.py
plot_det_curve.py
plot_det_evolution.py
benchmark_efficiency.py
requirements.txt
```

The repository also contains the generated results and, where applicable, trained model checkpoints.

Additional large reproducibility files are provided in the supplementary repository:

[FAL-Net reproducibility resources](https://drive.google.com/drive/u/0/folders/1xD8beNvmwn-r5pVSiCu4RkZg2Tu_w5nZ?utm_source=chatgpt.com)

---

## 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone <FAL-NET-GITHUB-URL>
cd FAL-Net
pip install -r requirements.txt
```

The exact package versions used for the experiments are specified in `requirements.txt`.

---

## 3. Dataset Organization

The scripts expect separate directories for bona fide/live and spoof/attack samples.

```text
dataset/
├── train/
│   ├── live/
│   └── fake/
└── test/
    ├── live/
    └── fake/
```

The corresponding directories are provided through:

```text
--real-dir-train
--fake-dir-train
--real-dir-test
--fake-dir-test
```

The label convention used by the implementation is:

```text
0 = Live / Bona Fide
1 = Fake / Spoof / Presentation Attack
```

The experiments cover LivDet 2011, 2013, 2015, and 2017, including the sensors reported in the paper.

---

## 4. Main FAL-Net Configuration

The main FAL-Net configuration uses:

* ImageNet-pretrained ResNet50 backbone;
* multi-scale feature extraction;
* attention-guided decoder;
* Soft Spatial Attention (SSA);
* multi-scale feature fusion;
* LoRA with rank 8 in the classification head.

In the released implementation, the **entire ResNet50 backbone is fine-tuned jointly** with the decoder and classification head.

The principal configuration is:

```text
proposed_r8
```

with:

```text
LoRA rank = 8
SSA = enabled
Multi-scale fusion = enabled
Backbone = trainable
```

---

## 5. Reproducing the Ablation Study

The ablation configurations are predefined in `ablation_runner.py`.

Available configurations include:

```text
full_finetuning
frozen_backbone
baseline
lora_only
ssa_only
proposed_r4
proposed_r8
single_scale
```

The complete command-line interface can be checked with:

```bash
python ablation_runner.py --help
```

### Example

For LivDet 2015 CrossMatch:

```bash
python ablation_runner.py \
  --epochs 50 \
  --real-dir-train "<CROSSMATCH_LIVE_TRAIN_DIRECTORY>" \
  --fake-dir-train "<CROSSMATCH_SPOOF_TRAIN_DIRECTORY>" \
  --real-dir-test "<CROSSMATCH_LIVE_TEST_DIRECTORY>" \
  --fake-dir-test "<CROSSMATCH_SPOOF_TEST_DIRECTORY>" \
  --sensor CrossMatch
```

Results are saved under:

```text
results_CrossMatch/ablation/
```

Each configuration produces, where applicable:

```text
best_model.h5
training_info.json
results_metrics.json
roc_data.npy
benchmark_metrics.json
```

A consolidated summary is saved as:

```text
full_summary.json
```

---

## 6. Repeated Experiments

Repeated experiments are implemented in:

```text
repeat_experiments.py
```

The available arguments can be checked with:

```bash
python repeat_experiments.py --help
```

Example:

```bash
python repeat_experiments.py \
  --runs 3 \
  --real-dir-train "<CROSSMATCH_LIVE_TRAIN_DIRECTORY>" \
  --fake-dir-train "<CROSSMATCH_SPOOF_TRAIN_DIRECTORY>" \
  --real-dir-test "<CROSSMATCH_LIVE_TEST_DIRECTORY>" \
  --fake-dir-test "<CROSSMATCH_SPOOF_TEST_DIRECTORY>" \
  --sensor CrossMatch
```

The sensor argument is explicitly passed to `ablation_runner.py`.

The generated summaries are stored as:

```text
full_summary_run_1.json
full_summary_run_2.json
full_summary_run_3.json
```

The current aggregation reports mean and standard deviation for:

* APCER;
* BPCER;
* ACE;
* EER.

---

## 7. DET Curves

### DET curve for one sensor

The script:

```text
plot_det_curve.py
```

accepts the result directory through `--dir`.

Example:

```bash
python plot_det_curve.py \
  --dir "results_CrossMatch/ablation"
```

The resulting figure is saved as:

```text
det_curve.png
```

### Cross-sensor DET comparison

For the LivDet 2015 sensors used in the paper:

```bash
python plot_det_evolution.py \
  --dir2015CM "results_CrossMatch/ablation/proposed_r8" \
  --dir2015HS "results_Hi_Scan/ablation/proposed_r8" \
  --dir2015DP "results_Digital_Persona/ablation/proposed_r8" \
  --dir2015GB "results_GreenBit/ablation/proposed_r8"
```

The resulting figure is:

```text
det_evolution_livdet.png
```

---

## 8. Computational-Efficiency Benchmark

Computational-efficiency measurements are generated by the ablation workflow using `benchmark_efficiency.py`.

The benchmark reports:

```text
trainable parameters
total parameters
FLOPs
training peak memory
inference peak memory
inference latency
training-step latency
```

For the results reported in the paper, the benchmark was conducted using the **LivDet 2015 Hi-Scan setting** under the same hardware and software environment for all ablation configurations.

The benchmark uses a fixed input resolution of:

```text
224 × 224 × 3
```

and consistent measurement conditions.

This controlled setup is intended to provide a fair comparison of computational cost between the evaluated configurations.

The per-configuration benchmark results are stored in:

```text
benchmark_metrics.json
```

and consolidated in:

```text
full_summary.json
```

---

## 9. Reproducibility and CLI Verification

To verify the exact interfaces implemented in the released code, users can run:

```bash
python train_ablation.py --help
python test_ablation.py --help
python ablation_runner.py --help
python repeat_experiments.py --help
python plot_det_curve.py --help
python plot_det_evolution.py --help
```

The current implementation does **not** use undocumented arguments such as:

```text
--config
--dataset
```

The ablation configurations are predefined internally in `ablation_runner.py`.

---

## 10. Citation

If you use this implementation, please cite:

```bibtex
@article{FALNet,
  title={FAL-Net: A Hybrid LoRA-Enhanced Architecture with Spatial Attention for Robust Fingerprint Liveness Detection},
  author={Bouhamed, Sonda Ammar and Medhioub, Mouna and Ayadi, Manel and Masmoudi, Nesrine and Alshalali, Tagrid Abdullah N.},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2026}
}
```

---

## 11. Contact

**Sonda Ammar Bouhamed**
Associate Professor
Higher Institute of Computer Science and Multimedia of Sfax (ISIMS)
University of Sfax, Tunisia
SM@RTS Laboratory, Digital Research Center of Sfax (CRNS)
