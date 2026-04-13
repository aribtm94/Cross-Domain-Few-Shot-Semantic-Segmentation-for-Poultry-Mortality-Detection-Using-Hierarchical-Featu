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

## Dataset

The dataset configuration used for CD-FSS evaluation is summarized below.

### Source Domain

* **PASCAL VOC2012**:

    Download the PASCAL VOC2012 SDS extended mask annotations from [Google Drive](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view?usp=sharing).

    Download the Chicken dataset from [Google Drive](https://drive.google.com/drive/folders/10idb9WpEDYqqHSbIvjHoF4ll1Iq_KNla?usp=sharing).

## Note

For a fair comparison across experiments, the distinction between `pure` and non-`pure` variants must be maintained, as their training data compositions are not identical.
