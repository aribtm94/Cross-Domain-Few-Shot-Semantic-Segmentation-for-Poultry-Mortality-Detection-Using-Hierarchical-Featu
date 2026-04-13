# Cross-Domain Few-Shot Semantic Segmentation for Poultry Mortality Detection Using Hierarchical Feature-Enhanced Prototype Learning

This repository contains several implementations and experimental variants for Cross-Domain Few-Shot Semantic Segmentation (CD-FSS) and SAM2-based segmentation.

## Project Root Structure

The repository follows a consistent naming convention across the main project folders:

- Any folder name that includes `pure` indicates that training is performed only on PASCAL VOC as the source domain.
- Any folder name that does not include `pure` indicates that the source-domain training set is mixed with the Chicken dataset.
- In other words, Chicken is included in the training domain for all non-`pure` folders.

### Folder Overview

| Folder | Description |
| --- | --- |
| `patnet/` | PATNet variant trained with the mixed VOC + Chicken source domain. |
| `patnet-pure/` | PATNet variant trained only with VOC. |
| `sam2-cdfsspure/` | SAM2-CD-FSS variant trained only with VOC. |
| `SAM2CDFSS/` | SAM2-CD-FSS variant trained with the mixed VOC + Chicken source domain. |

Auxiliary directories such as `logs/`, `vis_out/`, and `data/` within each project are used for experiment outputs, visualizations, and supporting data files.

## Reproduction Procedure

The steps below are organized by root folder to support direct experimental replication.

### Common Preparation

1. Prepare the dataset root directory so that the required subfolders are available to the selected project.
2. Install the dependencies specified in the corresponding project folder before running training or testing.
3. Use the same random seed, fold, and checkpoint selection when comparing results across variants.

### 1. `patnet-pure/`

This variant is trained only on PASCAL VOC.

1. Change into the project directory.
2. Train the model on VOC only.

```bash
cd patnet-pure
python train.py --datapath ./dataset --benchmark_train pascal --benchmark_val pascal --fold 4 --val_fold 0 --backbone resnet50 --bsz 2 --bsz_val 2 --lr 3e-4 --niter 8 --nworker 0 --logpath patnet_pure_replication
```

3. Evaluate the trained checkpoint on the desired target benchmark.

```bash
python test.py --datapath ./dataset --benchmark chick --fold 0 --nshot 1 --backbone resnet50 --load path/to/best_model.pt
python test.py --datapath ./dataset --benchmark chick --fold 0 --nshot 5 --backbone resnet50 --load path/to/best_model.pt
```

### 2. `patnet/`

This variant uses a mixed source domain, where Chicken is included during training.

1. Change into the project directory.
2. Train the primary VOC branch and enable the auxiliary Chicken branch.

```bash
cd patnet
python train.py --datapath ./dataset --benchmark_train pascal --benchmark_train_aux chick --benchmark_val chick --fold 4 --val_fold 0 --backbone resnet50 --bsz 2 --bsz_aux 2 --bsz_val 2 --lr 3e-4 --niter 30 --nworker 0 --logpath patnet_mixed_replication
```

3. Evaluate the checkpoint on the target benchmark used in the experiment.

```bash
python test.py --datapath ./dataset --benchmark chick --fold 0 --nshot 1 --backbone resnet50 --load path/to/best_model.pt
python test.py --datapath ./dataset --benchmark chick --fold 0 --nshot 5 --backbone resnet50 --load path/to/best_model.pt
```

### 3. `sam2-cdfsspure/`

This variant is the SAM2-based pure setting and is trained only on PASCAL VOC.

1. Change into the project directory.
2. Train the model using the VOC source domain only.

```bash
cd sam2-cdfsspure
python train.py --sam2_cfg sam2_hiera_l.yaml --sam2_ckpt path/to/sam2_hiera_large.pt --datapath_src ./dataset --datapath_tgt ./dataset --fold 4 --benchmark_val pascal --split_val val --train_shot 1 --val_shot 1 --bsz 2 --bsz_val 2 --lr 3e-4 --niter 2000 --nworker 4 --logpath sam2_pure_replication
```

3. Test the resulting checkpoint on the selected benchmark.

```bash
python test.py --load path/to/best_model.pt --sam2_cfg sam2_hiera_l.yaml --sam2_ckpt path/to/sam2_hiera_large.pt --benchmark chick --split test --fold 0 --nshot 1 --datapath_src ./dataset --datapath_tgt ./dataset
python test.py --load path/to/best_model.pt --sam2_cfg sam2_hiera_l.yaml --sam2_ckpt path/to/sam2_hiera_large.pt --benchmark chick --split test --fold 0 --nshot 5 --datapath_src ./dataset --datapath_tgt ./dataset
```

### 4. `SAM2CDFSS/`

This variant is the SAM2-based mixed setting and includes Chicken in the training domain.

1. Change into the project directory.
2. Train the primary VOC branch and enable the auxiliary Chicken branch.

```bash
cd SAM2CDFSS
python train.py --sam2_cfg sam2_hiera_l.yaml --sam2_ckpt path/to/sam2_hiera_large.pt --datapath_src ./dataset --datapath_tgt ./dataset --benchmark_train pascal --benchmark_train_aux chick --benchmark_val chick --split_val val --fold 4 --val_fold 0 --train_shot 1 --aux_shot 5 --val_shot 1 --bsz 2 --bsz_aux 2 --bsz_val 2 --lr 3e-4 --niter 2000 --nworker 4 --logpath sam2_mixed_replication
```

3. Evaluate the checkpoint on the selected benchmark.

```bash
python test.py --load path/to/best_model.pt --sam2_cfg sam2_hiera_l.yaml --sam2_ckpt path/to/sam2_hiera_large.pt --benchmark chick --split test --fold 0 --nshot 1 --datapath_src ./dataset --datapath_tgt ./dataset
python test.py --load path/to/best_model.pt --sam2_cfg sam2_hiera_l.yaml --sam2_ckpt path/to/sam2_hiera_large.pt --benchmark chick --split test --fold 0 --nshot 5 --datapath_src ./dataset --datapath_tgt ./dataset
```

The exact checkpoint path, benchmark, and fold should be aligned with the experimental protocol you intend to reproduce.

## Dataset

The dataset configuration used for CD-FSS evaluation is summarized below.

### Source Domain

* **PASCAL VOC2012**:

    Download the PASCAL VOC2012 SDS extended mask annotations from [Google Drive](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view?usp=sharing).

    Download the Chicken dataset from [Google Drive](https://drive.google.com/drive/folders/10idb9WpEDYqqHSbIvjHoF4ll1Iq_KNla?usp=sharing).

## Note

For a fair comparison across experiments, the distinction between `pure` and non-`pure` variants must be maintained, as their training data compositions are not identical.
