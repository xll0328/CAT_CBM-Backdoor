# CAT: Concept-Level Backdoor Attacks on Concept Bottleneck Models

Official implementation for **"Multimodal Deception in Explainable AI: Concept-Level Backdoor Attacks on Concept Bottleneck Models"** (TMLR 2026).

- Project page: https://xll0328.github.io/cat/
- Paper: https://openreview.net/forum?id=bntZBG9fBY
- Code repository: https://github.com/xll0328/CAT_CBM-Backdoor

## Overview

Concept Bottleneck Models (CBMs) expose human-interpretable concepts, but this semantic interface also creates a new attack surface. This repository releases the code for:

- **CAT**: concept-level backdoor attacks with filtered trigger selection.
- **CAT+**: an enhanced variant that optimizes trigger-concept associations.
- **Random trigger baseline**: an ablation baseline for comparison.
- **Clean CBM training**: baseline training and evaluation pipelines.
- **Data preprocessing** for CUB-200-2011 and AwA2.

The public release focuses on the attack side of the project and supports the two stages used in the paper:

1. training and evaluating clean CBMs,
2. injecting concept-level triggers and measuring attack success.

## Repository Structure

```text
.
├── src/
│   ├── CAT/
│   │   ├── cat.py               # CAT trigger construction
│   │   └── cat_plus.py          # CAT+ trigger optimization
│   ├── data/
│   │   ├── dataset.py           # Base dataset and dataloader
│   │   └── poison_dataset.py    # Poisoned dataset construction
│   ├── experiments/
│   │   ├── baseline.py          # Clean CBM training
│   │   ├── attack.py            # CAT / CAT+ attack experiments
│   │   ├── attack_random.py     # Random-trigger attack baseline
│   │   └── template.py          # Shared training utilities
│   ├── models/
│   │   └── model.py             # CBM backbone
│   ├── processing/
│   │   ├── cub_data_processing.py
│   │   └── awa_data_processing.py
│   └── utils/
│       ├── config.py
│       ├── data_path.yml        # Edit dataset paths before running
│       ├── metrics.py
│       └── util.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Environment

Tested with Python 3.9+.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### 1. CUB-200-2011

Download [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and place it under a local source directory, for example:

```text
source_data/CUB_200_2011
```

Then preprocess it with:

```bash
python src/processing/cub_data_processing.py \
  -data_dir source_data/CUB_200_2011 \
  -save_dir processed_data/cub_processed_data
```

### 2. Animals with Attributes 2 (AwA2)

Download [AwA2](https://cvml.ista.ac.at/AwA2/) and place it under:

```text
source_data/Animals_with_Attributes2
```

Then preprocess it with:

```bash
python src/processing/awa_data_processing.py \
  -data_dir source_data/Animals_with_Attributes2 \
  -save_dir processed_data/awa_processed_data
```

### 3. Configure dataset paths

Before training or evaluation, edit `src/utils/data_path.yml`:

```yaml
cub:
  processed_dir: /path/to/processed_data/cub_processed_data
  source_dir: /path/to/source_data/CUB_200_2011
awa:
  processed_dir: /path/to/processed_data/awa_processed_data
  source_dir: /path/to/source_data/Animals_with_Attributes2
```

## Usage

Run all commands from the repository root.

### Train a clean CBM

```bash
python src/experiments/baseline.py \
  -dataset cub \
  -epoch 50 \
  -batch_size 128
```

### Run CAT

```bash
python src/experiments/attack.py \
  -dataset cub \
  -trigger_mode cat \
  -trigger_size 2 \
  -injection_rate 0.1 \
  -injection_mode mix_label
```

### Run CAT+

```bash
python src/experiments/attack.py \
  -dataset cub \
  -trigger_mode cat+ \
  -trigger_size 2 \
  -injection_rate 0.1 \
  -injection_mode mix_label
```

### Run the random-trigger baseline

```bash
python src/experiments/attack_random.py \
  -dataset cub \
  -trigger_mode random \
  -trigger_size 2 \
  -injection_rate 0.1
```

## Main Arguments

| Argument | Description | Default |
|---|---|---|
| `-dataset` | Dataset name: `cub` or `awa` | `cub` |
| `-epoch` | Number of training epochs | `50` |
| `-batch_size` | Batch size | `128` |
| `-learning_rate` | Learning rate | `1e-4` |
| `-weight_decay` | Weight decay | `5e-5` |
| `-gamma` | Exponential LR decay | `0.95` |
| `-concept_lambda` | Weight for concept prediction loss | `0.5` |
| `-v_backbone` | Vision backbone: `resnet` or `vit` | `resnet` |
| `-target_class` | Target class for attack | `0` |
| `-trigger_mode` | `cat`, `cat+`, or `random` | `cat` |
| `-trigger_size` | Number of trigger concepts | `2` |
| `-injection_rate` | Poisoning rate | `0.1` |
| `-injection_mode` | `clean_label` or `mix_label` | `mix_label` |
| `-saved_dir` | Directory for checkpoints and logs | `results` |

## Notes

- This repository is the public **CAT** release and does not include the separate ConceptGuard defense pipeline.
- Dataset paths are intentionally left as placeholders in `src/utils/data_path.yml`; please replace them with your local paths.
- Results may vary slightly across hardware and random seeds.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{lai2026cat,
  title={Multimodal Deception in Explainable AI: Concept-Level Backdoor Attacks on Concept Bottleneck Models},
  author={Lai, Songning and Yang, Jiayu and Huang, Yu and Hu, Lijie and Xue, Tianlang and Hu, Zhangyi and Li, Jiaxu and Liao, Haicheng and Liu, Zongyang and Yue, Yutao},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```
