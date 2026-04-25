---
title: "Albumentations in Medical Imaging: Who Actually Uses It"
date: 2026-04-25
author: vladimir-iglovikov
categories:
  - community
tags:
  - medical-imaging
  - biomedical-imaging
  - radiology
  - histopathology
  - microscopy
  - adoption
excerpt: "Reproducible audit of who uses Albumentations in the medical-imaging ecosystem: 452 medical / biomedical papers cite it, TIAToolbox declares it as a direct dependency, 12 public repos across MIC-DKFZ, bowang-lab, and TissueImageAnalytics import it, and 33 Hugging Face medical artifacts reference it."
image: images/hero.webp
featured: false
---

# Albumentations in Medical Imaging: Who Actually Uses It

Albumentations is infrastructure in the medical-imaging / biomedical-imaging ecosystem, not a research curio. This post is the receipts: which named organizations import it, which OSS medical library declares it as a direct dependency, how many papers cite it, and where it appears in public model cards.

All numbers below are reproducible from public APIs and public repository files: citation metadata, GitHub Code Search, the [Hugging Face Hub](https://huggingface.co/), and root-level packaging files (`requirements.txt`, `pyproject.toml`, etc.) in each OSS repo. The org-scoped grep is `org:<name> "import albumentations"`.

## Headline

- **452 medical / biomedical papers** cite Albumentations
- **1 OSS medical-imaging library** declares it as a direct dependency
- **12 public repositories** across **3 named medical-imaging organizations** import it
- **33 Hugging Face artifacts** in the medical-imaging / biomedical tag space reference it

"Albumentations" here means the project stewarded by Albumentations LLC: the legacy MIT `albumentations` package (archived June 2025) plus the maintained successor `albumentationsx` (AGPL-3.0 + Commercial), which preserves API compatibility. See the [dual-licensing post](/blog/2025/01-albumentationsx-dual-licensing) for context.

## Why Medical Imaging Pulls in an Augmentation Library at All

Medical imaging is not one data type. A medical training pipeline might ingest chest X-rays, CT slices, retinal fundus photos, endoscopy frames, histopathology tiles, phase-contrast microscopy videos, ultrasound frames, OCT scans, or multichannel fluorescence images. The common thread is not the sensor. The common thread is that the image and the labels have to be transformed together without corrupting the clinical or biological meaning.

Three details matter in practice:

1. **The labels are often spatial.** Segmentation masks for organs, lesions, nuclei, plaques, cysts, polyps, vessels, and tissue regions have to move exactly with the image. The same is true for bounding boxes and landmarks. Albumentations is built around `Compose` over `(image, mask, bboxes, keypoints)`, which is why it shows up in medical repositories that train segmentation and detection models.
2. **The valid invariances are modality-specific.** A 90-degree rotation may be fine for histopathology tiles, microscopy patches, or some cell-imaging tasks. It can be wrong for chest X-rays, retinal laterality, or workflows where orientation encodes acquisition protocol. Horizontal flips can silently create anatomically impossible examples. Medical augmentation is not "add randomness"; it is "encode the invariances the target task can actually tolerate."
3. **Throughput still matters.** Medical datasets are often tiled at training time: whole-slide pathology tiles, CT/MRI slices, endoscopy frames, microscopy crops. Augmentation usually runs CPU-side inside a data loader and has to feed the GPU. In the current [9-channel CPU benchmark](https://albumentations.ai/docs/benchmarks/multichannel-benchmarks/), AlbumentationsX is fastest on **58 of 68 transforms**, with a median **3.73x speedup vs Kornia** and **2.26x vs Torchvision** on the head-to-head subset. That benchmark is not "medical" by itself, but the arbitrary-channel constraint is directly relevant to CT slice stacks, fluorescence channels, and scientific-imaging data that do not look like ImageNet RGB.

Concretely, a conservative pathology / microscopy segmentation pipeline looks like this:

```python
import albumentations as A
import numpy as np

image = np.load("h_and_e_tile.npy")
mask = np.load("nuclei_mask.npy")

transform = A.Compose([
    A.RandomCrop(height=512, width=512),
    A.SquareSymmetry(p=1.0),
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.03, 0.03),
        rotate=(-10, 10),
        shear=(-3, 3),
        p=0.5,
    ),
    A.RandomBrightnessContrast(
        brightness_range=(-0.08, 0.08),
        contrast_range=(-0.08, 0.08),
        p=0.4,
    ),
    A.GaussNoise(std_range=(0.01, 0.04), p=0.2),
])

out = transform(image=image, mask=mask)
tile, label = out["image"], out["mask"]
```

In order, that pipeline is [`RandomCrop`](https://explore.albumentations.ai/transform/RandomCrop) -> [`SquareSymmetry`](https://explore.albumentations.ai/transform/SquareSymmetry) -> [`Affine`](https://explore.albumentations.ai/transform/Affine) -> [`RandomBrightnessContrast`](https://explore.albumentations.ai/transform/RandomBrightnessContrast) -> [`GaussNoise`](https://explore.albumentations.ai/transform/GaussNoise). For H&E patches or microscopy tiles, square symmetries can be a reasonable default because the tissue or cells usually do not have a canonical camera-up direction. For chest X-ray, retinal left/right classification, ECG-rendered images, or tasks where acquisition orientation matters, the same transform would be a bug. The library gives you the mechanism; the domain decides the invariance.

The same `Compose` pipeline would also accept `bboxes=...` and `keypoints=...` and keep them aligned.

## OSS Medical-Imaging Libraries That Depend on Albumentations

These are repository-rooted facts. The dependency is declared in packaging files, not inferred from a citation graph or README mention.

Of 16 verified medical OSS projects, **1 project declares `albumentations` as a direct dependency**:

| Library         | Org                         | Evidence file(s)                  | Repo                                                                  |
| --------------- | --------------------------- | --------------------------------- | --------------------------------------------------------------------- |
| **TIAToolbox**  | Tissue Image Analytics Centre | `requirements/requirements.txt` | [TissueImageAnalytics/tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox) |

TIAToolbox is the notable one here: it is a real pathology toolkit, not a one-off experiment repository. Direct dependency counts are conservative by design. They miss internal hospital code, private commercial pipelines, and research repos that use Albumentations in training scripts without packaging it as a reusable library.

## Named Medical-Imaging Organizations Using It

Org-scoped GitHub Code Search (`org:<name> "import albumentations"`) found `import albumentations` in **12 repositories across 3 organizations** from a hand-curated tier-1 list of medical AI toolkits, pathology and microscopy projects, research labs, and clinical-imaging OSS.

| Organization                                                  | Repos | Type         |
| ------------------------------------------------------------- | ----: | ------------ |
| [MIC-DKFZ](https://github.com/MIC-DKFZ)                       | 9     | Organization |
| [bowang-lab](https://github.com/bowang-lab)                   | 2     | Organization |
| [TissueImageAnalytics](https://github.com/TissueImageAnalytics) | 1   | Organization |

MIC-DKFZ is the largest public-code cluster in this audit. That matters because the German Cancer Research Center has been central to medical-imaging ML tooling and challenge code for years. The point is not that every repo below is a maintained library. The point is that public, named medical-imaging groups repeatedly reach for Albumentations as the augmentation layer in training code.

A representative path list from the search:

| Repo                                                                 | File                                                                  |
| -------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [MIC-DKFZ/AGGC2022](https://github.com/MIC-DKFZ/AGGC2022)             | `data/test_augs.py`                                                   |
| [MIC-DKFZ/BodyPartRegression](https://github.com/MIC-DKFZ/BodyPartRegression) | `bpreg/preprocessing/nrrd2npy.py`                              |
| [MIC-DKFZ/diabetes-xai](https://github.com/MIC-DKFZ/diabetes-xai)     | `feature_extraction/extract_features_fp_timm.py`                      |
| [MIC-DKFZ/generalized_yolov5](https://github.com/MIC-DKFZ/generalized_yolov5) | `utils/augmentations.py`                                       |
| [MIC-DKFZ/help_a_hematologist_out_challenge](https://github.com/MIC-DKFZ/help_a_hematologist_out_challenge) | `augmentation/policies/cifar.py` |
| [MIC-DKFZ/image_classification](https://github.com/MIC-DKFZ/image_classification) | `augmentation/policies/cifar.py`                              |
| [MIC-DKFZ/perovskite-xai](https://github.com/MIC-DKFZ/perovskite-xai) | `data/augmentations/perov_2d.py`                                      |
| [MIC-DKFZ/radioactive](https://github.com/MIC-DKFZ/radioactive)       | `src/radioa/model/SAMMed2D.py`                                        |
| [MIC-DKFZ/semantic_segmentation](https://github.com/MIC-DKFZ/semantic_segmentation) | `src/semantic_segmentation/datasets/base_dataset.py`          |
| [TissueImageAnalytics/tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox) | `tiatoolbox/tools/stainaugment.py`                         |
| [bowang-lab/EchoJEPA](https://github.com/bowang-lab/EchoJEPA)         | `data/batch_depth_attenuation.py`                                     |
| [bowang-lab/MedSAMSlicer](https://github.com/bowang-lab/MedSAMSlicer) | `MedSAMLite/Resources/server_essentials/medsam_interface/engines/src/data/medsam_datamodule.py` |

## Academic Citations

Albumentations is cited by **452 unique medical-imaging / biomedical-imaging papers**. The count is filtered from the citation audit by keeping papers whose title, abstract, venue, or metadata match medical and biomedical keywords: radiology, histopathology, pathology, microscopy, CT, MRI, ultrasound, X-ray, OCT, endoscopy, dermatology, ophthalmology, biomedical, clinical, lesion, tumor, cell, nuclei, and related terms.

The citation data is deduplicated by paper URL. That detail matters because the raw citation export contains one row per `(paper x author x affiliation)`, so counting rows would overstate adoption.

### Year-over-Year Growth

| Year | Medical papers citing Albumentations |
| ---- | ------------------------------------ |
| 2020 | 13                                   |
| 2021 | 31                                   |
| 2022 | 59                                   |
| 2023 | 75                                   |
| 2024 | 90                                   |
| 2025 | 130                                  |
| 2026 | 54 (YTD, April)                      |

The visible pattern is steady growth, with a large jump in 2025. The conservative interpretation is simple: medical-imaging ML papers increasingly publish code, increasingly use standard augmentation libraries instead of local one-off transforms, and increasingly cite the tooling that sits in the training pipeline.

### Top-Cited Medical Papers (Sample)

| Citations | Year | Paper | Matched keyword |
| --------: | ---- | ----- | --------------- |
| 6 | 2025 | [Rapid label-free identification of seven bacterial species using microfluidics, single-cell time-lapse phase-contrast mi](https://doi.org/10.1371/journal.pone.0330265) | microscopy |
| 6 | 2024 | [Rapid label-free identification of seven bacterial species using microfluidics, single-cell time-lapse phase-contrast mi](https://doi.org/10.1101/2024.10.15.618380) | microscopy |
| 5 | 2021 | [Semi-supervised training of deep convolutional neural networks with heterogeneous data and few local annotations: An exp](https://doi.org/10.1016/j.media.2021.102165) | histopathology |
| 5 | 2024 | [Multimodal representations of biomedical knowledge from limited training whole slide images and reports using deep learn](https://doi.org/10.1016/j.media.2024.103303) | whole slide |
| 5 | 2021 | [Impact of Lung Segmentation on the Diagnosis and Explanation of COVID-19 in Chest X-ray Images](https://doi.org/10.3390/s21217116) | chest x-ray |
| 5 | 2025 | [Automatic labels are as effective as manual labels in digital pathology images classification with deep learning](https://doi.org/10.1016/j.jpi.2025.100462) | digital pathology |
| 5 | 2025 | [Segmentation and quantification of atherosclerotic plaques in optical coherence tomography](https://doi.org/10.1016/j.compbiomed.2025.111061) | optical coherence tomography |
| 5 | 2026 | [A Transformer-Based Framework for OCT Cyst Segmentation](https://doi.org/10.1007/978-3-032-11381-8_49) | oct |
| 4 | 2023 | [AUTOMATIC POLYP SEMANTIC SEGMENTATION USING WIRELESS CAPSULE ENDOSCOPY IMAGES WITH VARIOUS CONVOLUTIONAL NEURAL NETWORK](https://doi.org/10.4015/s1016237223500266) | endoscopy |
| 4 | 2024 | [Design and development of artificial intelligence-based application programming interface for early detection and diagno](https://doi.org/10.1002/ima.23034) | endoscopy |

The truncated titles are exactly what the public citation export returned in this audit. The point of the table is not bibliographic polish; it is a reproducible sample of medical papers where Albumentations appears in the citation trail.

### Top Affiliations

Affiliations with at least three medical papers in the filtered citation set:

| Affiliation | Papers |
| ----------- | -----: |
| Radboud University Medical Center | 5 |
| Technical University of Munich | 4 |
| University of Oxford | 4 |
| University of Ulsan College of Medicine, Seoul | 4 |
| Affiliated Hospital of Hubei University of Arts and Science | 3 |
| Beihang University | 3 |
| Chinese Academy of Sciences, Shenzhen | 3 |
| Concordia University | 3 |
| First Affiliated Hospital of Jinan University | 3 |
| Fraunhofer Institute for Digital Medicine MEVIS, Bremen | 3 |
| Hangzhou Dianzi University | 3 |
| King Saud University | 3 |
| Mahidol University | 3 |
| McGill University | 3 |
| Memorial Sloan Kettering Cancer Center | 3 |

## Hugging Face Ecosystem

Across Hugging Face Hub artifacts tagged `medical` / `medical-imaging` / `radiology` / `histopathology` / `microscopy` / `healthcare`, **33 artifacts** reference Albumentations in their model or dataset card: **32 models** and **1 dataset**.

The absolute download counts are small for most of these cards, which is normal for specialized medical artifacts on Hugging Face. The useful signal is not popularity ranking. The useful signal is that Albumentations appears in public training recipes across radiology, histopathology, dermatology, endoscopy, pressure-sore classification, polyp segmentation, and related biomedical tasks.

| Kind | ID | Downloads | Likes | Tags |
| ---- | -- | --------: | ----: | ---- |
| model | [Snarcy/RedDino-large](https://huggingface.co/Snarcy/RedDino-large) | 915 | 1 | medical-imaging |
| dataset | [LosHuesitos9-9/Huesitos](https://huggingface.co/datasets/LosHuesitos9-9/Huesitos) | 15 | 1 | medical |
| model | [RuthvikBandari/DiaFootAI](https://huggingface.co/RuthvikBandari/DiaFootAI) | 10 | 0 | medical-imaging |
| model | [ibrahim313/ducknet-polyp-segmentation](https://huggingface.co/ibrahim313/ducknet-polyp-segmentation) | 3 | 1 | medical-imaging |
| model | [Lab-Rasool/PRIMER](https://huggingface.co/Lab-Rasool/PRIMER) | 3 | 1 | radiology |
| model | [Thiyaga158/Custom_CNN_For_Pneumonia_Detection_Using_Check_X-Ray](https://huggingface.co/Thiyaga158/Custom_CNN_For_Pneumonia_Detection_Using_Check_X-Ray) | 0 | 0 | healthcare; medical-imaging |
| model | [dheeren-tejani/DiabeticRetinpathyClassifier](https://huggingface.co/dheeren-tejani/DiabeticRetinpathyClassifier) | 0 | 0 | medical-imaging |
| model | [adelelsayed1991/chexpert-mae-densenet-fpn](https://huggingface.co/adelelsayed1991/chexpert-mae-densenet-fpn) | 0 | 0 | healthcare; medical-imaging |
| model | [ayanahmedkhan/VIT-gi-endoscopy-classifier](https://huggingface.co/ayanahmedkhan/VIT-gi-endoscopy-classifier) | 0 | 0 | medical-imaging |
| model | [RuthvikBandari/DiaFoot.AI-v2](https://huggingface.co/RuthvikBandari/DiaFoot.AI-v2) | 0 | 0 | medical-imaging |
| model | [tanishq74/retinasense-vit](https://huggingface.co/tanishq74/retinasense-vit) | 0 | 0 | medical-imaging |
| model | [MrCzaro/Pressure_sore_cascade_classifier_Torch](https://huggingface.co/MrCzaro/Pressure_sore_cascade_classifier_Torch) | 0 | 0 | medical-imaging |
| model | [csmp-hub/cellpose-histo-hgsc-nuc-v1](https://huggingface.co/csmp-hub/cellpose-histo-hgsc-nuc-v1) | 0 | 0 | histopathology |
| model | [csmp-hub/hovernet-histo-hgsc-nuc-v1](https://huggingface.co/csmp-hub/hovernet-histo-hgsc-nuc-v1) | 0 | 0 | histopathology |
| model | [csmp-hub/stardist-histo-hgsc-nuc-v1](https://huggingface.co/csmp-hub/stardist-histo-hgsc-nuc-v1) | 0 | 0 | histopathology |
| model | [csmp-hub/cellvit-histo-hgsc-nuc-v1](https://huggingface.co/csmp-hub/cellvit-histo-hgsc-nuc-v1) | 0 | 0 | histopathology |
| model | [csmp-hub/cppnet-histo-hgsc-nuc-v1](https://huggingface.co/csmp-hub/cppnet-histo-hgsc-nuc-v1) | 0 | 0 | histopathology |
| model | [histolytics-hub/hovernet-histo-hgsc-pan-v1](https://huggingface.co/histolytics-hub/hovernet-histo-hgsc-pan-v1) | 0 | 0 | histopathology |
| model | [histolytics-hub/cellpose-histo-hgsc-pan-v1](https://huggingface.co/histolytics-hub/cellpose-histo-hgsc-pan-v1) | 0 | 0 | histopathology |
| model | [histolytics-hub/stardist-histo-hgsc-pan-v1](https://huggingface.co/histolytics-hub/stardist-histo-hgsc-pan-v1) | 0 | 0 | histopathology |

## What This Means

Medical-imaging ML pipelines depend on label-preserving image transforms: CT and MRI slices, X-ray and ultrasound frames, histopathology tiles, microscopy channels, endoscopy frames, segmentation masks, boxes, and landmarks all have to move together. Funding maintenance of Albumentations keeps that shared augmentation layer fast, inspectable, and usable by the research and OSS projects listed above.

The most important caveat is that medical augmentation is less forgiving than generic computer vision. A transform can be technically correct and clinically wrong. [`HorizontalFlip`](https://explore.albumentations.ai/transform/HorizontalFlip) is harmless for many tissue patches and harmful for laterality-sensitive tasks. [`RandomBrightnessContrast`](https://explore.albumentations.ai/transform/RandomBrightnessContrast) is a reasonable nuisance model for camera or staining variation, but a poor substitute for scanner physics. [`ElasticTransform`](https://explore.albumentations.ai/transform/ElasticTransform) can help in some microscopy / histology segmentation settings and can destroy morphology in others. The right question is never "does this transform improve validation score once?" The right question is "does this transform encode a variation that can exist at deployment time without changing the label?"

Every named org in the table above is a current, public-code user. TIAToolbox ships Albumentations transitively to its users. The 452-paper citation count is a lower bound because it only counts papers whose metadata explicitly contains medical or biomedical keywords.

If you maintain a medical-imaging OSS project, foundation model, or training pipeline and want to be added to or removed from this evidence set, ping me. The methodology is scripted and the audit can be rerun.

---

*This brief is regenerated from public APIs and public repository files. All counts are reproducible. Last regenerated 2026-04-25.*

*Hero image: cropped and resized from [Lung cancer histology collection.png](https://commons.wikimedia.org/wiki/File:Lung_cancer_histology_collection.png) by "Atlas of Pulmonary Pathology" on Flickr (Yale Rosen), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*
