---
title: "Albumentations in Geospatial: Who Actually Uses It"
date: 2026-04-22
author: vladimir-iglovikov
categories:
  - community
tags:
  - geospatial
  - remote-sensing
  - satellite-imagery
  - earth-observation
  - multispectral
  - adoption
excerpt: "Reproducible audit of who uses Albumentations in the satellite / remote-sensing ecosystem: 5 OSS geo libraries (raster-vision, solaris, TerraTorch, NASA/IBM Prithvi, GeoSeg) depend on it, 44 public repos across 19 named geo organizations import it (NASA, IBM, Microsoft, AWS, DLR, JPL, Satellogic, Development Seed, Allen AI, Radiant Earth, Global Fishing Watch, World Resources Institute), and 382 geospatial papers cite it."
image: images/hero.jpg
featured: false
---

# Albumentations in Geospatial: Who Actually Uses It

Albumentations is infrastructure in the satellite / remote-sensing ecosystem, not a research curio. This post is the receipts: which named organizations import it, which OSS geo libraries declare it as a direct dependency, how many papers cite it, and how that adoption has grown year over year.

All numbers below are reproducible from public APIs: [OpenAlex](https://openalex.org/) (citations), [GitHub](https://github.com/) Code Search (org-scoped `import` queries), the [Hugging Face Hub](https://huggingface.co/) (tagged model cards), and root-level packaging files (`requirements.txt`, `pyproject.toml`, etc.) in each OSS repo. The headline org-scoped grep is `org:<name> "import albumentations"`.

## Headline

- **382 geospatial papers** cite Albumentations
- **5 OSS geospatial libraries** declare it as a direct dependency
- **44 public repositories** across **19 named geospatial organizations** import it
- **3 HuggingFace artifacts** in the geo / remote-sensing tag space reference it

"Albumentations" here means the project stewarded by Albumentations LLC: the legacy MIT `albumentations` package (archived June 2025) plus the maintained successor `albumentationsx` (AGPL-3.0 + Commercial), which preserves API compatibility — see the [dual-licensing post](/blog/2025/01-albumentationsx-dual-licensing) for context.

## Why Geospatial Pulls in an Augmentation Library at All

Three things make satellite / drone / aerial imagery harder than the consumer-photo case Albumentations was originally designed for, and all three are exactly what an augmentation library buys you:

1. **Multi-band, non-RGB rasters.** Sentinel-2 has 13 bands, Landsat-8 has 11, Planet has 4–8, hyperspectral sensors can have 200+. Most ImageNet-era augmentation code assumes 3 channels of `uint8`. Albumentations transforms operate on arbitrary `(H, W, C)` arrays in `uint8` or `float32`. Native EO rasters are usually `uint16` (Sentinel-2 L1C/L2A reflectance, Landsat surface reflectance) — the standard approach is to scale to `float32` once at load time (e.g. `arr.astype(np.float32) / 10000.0` for Sentinel-2 reflectance) and let the augmentation pipeline run on the float tensor. Chromatic-shift / spectral / atmospheric ops stay band-aware.
2. **Tight label co-transforms.** A geo training sample is typically the image plus a segmentation mask (land cover, building footprint, burn scar) plus optionally bounding boxes (vehicles, ships, planes) plus keypoints (tower bases, well heads). Geometric ops have to apply identically to all of them or the labels silently drift. Albumentations is built around `Compose` over `(image, mask, bboxes, keypoints)` — that's why every geo OSS library below ends up using it.
3. **Tile pipelines.** Geo training is rarely "load whole image, augment, train." It's "stream tiles from a COG / GeoTIFF / Zarr, augment per-tile, batch." Augmentation has to be CPU-side and fast enough to feed the GPU. Albumentations is OpenCV-backed and dominates on multi-channel inputs: in our [9-channel CPU benchmark](https://albumentations.ai/docs/benchmarks/multichannel-benchmarks/) it is fastest on **58 of 68 transforms**, with a median **3.7× speedup vs Kornia** and **2.3× vs Torchvision** on the head-to-head subset, and a long tail of transforms that the other libraries don't implement for arbitrary channel counts at all. Both of those facts matter for geo: the speed feeds the GPU, and the coverage means you don't silently lose half your augmentation toolbox the moment you go past 3 channels.

Concretely, the typical geospatial use looks like this — note the multi-channel input, the paired mask, and the chromatic ops chosen specifically to be safe across bands:

```python
import albumentations as A
import numpy as np

image = np.load("sentinel2_tile.npy")
mask = np.load("landcover_tile.npy")

transform = A.Compose([
    A.RandomCrop(height=256, width=256),
    A.SquareSymmetry(p=1.0),
    A.RandomBrightnessContrast(
        brightness_range=(-0.1, 0.1),
        contrast_range=(-0.1, 0.1),
        p=0.5,
    ),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
])

out = transform(image=image, mask=mask)
tile, label = out["image"], out["mask"]
```

`SquareSymmetry` samples one of the eight **D4** symmetries (four 90°-step rotations plus four reflections) in a single pass. Chaining `HorizontalFlip`, `VerticalFlip`, and `RandomRotate90` with independent `p` values does **not** yield a uniform distribution over those symmetries and costs three geometric ops per step instead of one.

Same `Compose` would also accept `bboxes=...` and `keypoints=...` and keep them aligned.

## OSS Geospatial Libraries That Depend on Albumentations

These are repository-rooted facts — the dependency is declared in `pyproject.toml` / `requirements.txt` / `setup.py` / `environment.yml`, not inferred from a citation graph.


| Library             | Org                         | Evidence file(s)                              | Repo                                                                        |
| ------------------- | --------------------------- | --------------------------------------------- | --------------------------------------------------------------------------- |
| **raster-vision**   | Azavea / Element 84         | `requirements.txt`                            | [azavea/raster-vision](https://github.com/azavea/raster-vision)             |
| **solaris**         | CosmiQ / IQT                | `setup.py; requirements.txt; environment.yml` | [CosmiQ/solaris](https://github.com/CosmiQ/solaris)                         |
| **TerraTorch**      | IBM Research                | `pyproject.toml`                              | [IBM/terratorch](https://github.com/IBM/terratorch)                       |
| **prithvi-pytorch** | NASA / IBM                  | `requirements.txt`                            | [NASA-IMPACT/Prithvi-EO-2.0](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) |
| **GeoSeg**          | Academic (Wuhan University) | `requirements.txt`                            | [WangLibo1995/GeoSeg](https://github.com/WangLibo1995/GeoSeg)               |


Notable: **Prithvi** is the NASA/IBM foundation model for Earth Observation. **TerraTorch** is IBM's geospatial fine-tuning toolkit built on top of Prithvi. **Raster Vision** is Azavea's (now Element 84's) production geospatial deep-learning framework. **Solaris** is the CosmiQ / IQT toolkit used for SpaceNet challenges. All four declare Albumentations as a direct, hard dependency — meaning anyone who `pip install`s these libraries pulls Albumentations transitively.

## Named Geospatial Organizations Using It

Org-scoped GitHub Code Search (`org:<name> "import albumentations"`) found `import albumentations` in **44 repositories across 19 organizations** from a hand-curated tier-1 list (commercial EO providers, space agencies, research labs, OSS geo ML projects).


| Organization                                                | Repos | Notes                                          |
| ----------------------------------------------------------- | ----- | ---------------------------------------------- |
| [aws-samples](https://github.com/aws-samples)               | 10    | AWS reference architectures (SageMaker, etc.)  |
| [IBM](https://github.com/IBM)                               | 6     | TerraTorch, TerraMind, ML4EO, peft-geofm       |
| [microsoft](https://github.com/microsoft)                   | 5     | Microsoft AI for Earth / planetary computer    |
| [satellogic](https://github.com/satellogic)                 | 3     | Commercial EO constellation operator           |
| [developmentseed](https://github.com/developmentseed)       | 3     | Geospatial ML consultancy (NASA, World Bank)   |
| [zhu-xlab](https://github.com/zhu-xlab)                     | 2     | TUM Prof. Zhu's lab — major SSL-for-EO group   |
| [nasa-jpl](https://github.com/nasa-jpl)                     | 2     | NASA Jet Propulsion Laboratory                 |
| [DLR-MF-DAS](https://github.com/DLR-MF-DAS)                 | 2     | German Aerospace Center (SSL4EO-S12, etc.)     |
| [allenai](https://github.com/allenai)                       | 1     | Allen Institute for AI                         |
| [radiantearth](https://github.com/radiantearth)             | 1     | Radiant Earth Foundation                       |
| [azavea](https://github.com/azavea)                         | 1     | Maker of raster-vision (now Element 84)        |
| [awslabs](https://github.com/awslabs)                       | 1     | AWS Labs                                       |
| [CosmiQ](https://github.com/CosmiQ)                         | 1     | CosmiQ Works / IQT (SpaceNet)                  |
| [NASA-IMPACT](https://github.com/NASA-IMPACT)               | 1     | NASA IMPACT (Prithvi, ESA-NASA workshops)      |
| [spaceml-org](https://github.com/spaceml-org)               | 1     | SpaceML / FDL (NASA Frontier Development Lab)  |
| [tudelft3d](https://github.com/tudelft3d)                   | 1     | TU Delft 3D geoinformation                     |
| [wri](https://github.com/wri)                               | 1     | World Resources Institute                      |
| [WildMeOrg](https://github.com/WildMeOrg)                   | 1     | Wildlife computer vision                       |
| [GlobalFishingWatch](https://github.com/GlobalFishingWatch) | 1     | Global Fishing Watch (industrial activity SAR) |


The interesting cluster here is the foundation-model orgs — IBM (TerraTorch / TerraMind / Prithvi tooling), NASA-IMPACT (Prithvi-EO-2.0, ESA-NASA workshop notebooks), DLR (SSL4EO-S12), zhu-xlab (TUM SSL-for-EO). All of them ship public training notebooks where the augmentation pipeline is `import albumentations as A`.

A few representative paths from the search (one per org, abridged):


| Repo                                                                                                                | File                                                                    |
| ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [CosmiQ/solaris](https://github.com/CosmiQ/solaris)                                                                 | `solaris/nets/transform.py`                                             |
| [NASA-IMPACT/ESA-NASA-workshop-2025](https://github.com/NASA-IMPACT/ESA-NASA-workshop-2025)                         | `Track 1 (EO)/TerraMind/notebooks/terramind_v1_base_sen1floods11.ipynb` |
| [IBM/terramind](https://github.com/IBM/terramind)                                                                   | `notebooks/terramind_v1_small_burnscars.ipynb`                          |
| [IBM/peft-geofm](https://github.com/IBM/peft-geofm)                                                                 | `src/peft_geofm/datamodules/utils.py`                                   |
| [DLR-MF-DAS/SSL4EO-S12-v1.1](https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1)                                         | `README.md`                                                             |
| [GlobalFishingWatch/paper-industrial-activity](https://github.com/GlobalFishingWatch/paper-industrial-activity)     | `nnets/fishing/dataset.py`                                              |
| [aws-samples/aws-vegetation-management-workshop](https://github.com/aws-samples/aws-vegetation-management-workshop) | `remars2022-workshop/dataset.py`                                        |
| [azavea/raster-vision](https://github.com/azavea/raster-vision)                                                     | `rastervision_pytorch_backend/.../semantic_segmentation/utils.py`       |


## Academic Citations

Filtered from 2,403 unique citing papers (12,015 author-paper-affiliation rows in OpenAlex), keeping only those whose title / abstract / venue contain geospatial keywords (`satellite`, `remote sensing`, `aerial`, `UAV`, `drone`, `multispectral`, `hyperspectral`, `land cover`, `crop`, `wildfire`, `canopy`, etc.) — **382 unique geospatial papers cite Albumentations**.

### Year-over-year growth


| Year | Geo papers citing Albumentations |
| ---- | -------------------------------- |
| 2020 | 6                                |
| 2021 | 28                               |
| 2022 | 56                               |
| 2023 | 76                               |
| 2024 | 64                               |
| 2025 | 132                              |
| 2026 | 20 (YTD, April)                  |


The 2024→2025 jump (64 → 132) tracks the rise of geospatial foundation models (Prithvi, TerraMind, SatMAE, Clay) — each one ships a downstream-task notebook, and almost all of them ship it with Albumentations as the augmentation layer.

### Top-cited geo papers (sample)


| Citations | Year | Paper                                                                                                                                                   | Matched keyword |
| --------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| 16        | 2024 | [Using Generative Models to Improve Fire Detection Efficiency](https://doi.org/10.1109/itnt60778.2024.10582386)                                         | fire detection  |
| 10        | 2025 | [Exploration of geo-spatial data and machine learning algorithms for robust wildfire occurrence prediction](https://doi.org/10.1038/s41598-025-94002-4) | wildfire        |
| 10        | 2022 | [Estimation of the Canopy Height Model From Multispectral Satellite Imagery With CNNs](https://doi.org/10.1109/access.2022.3161568)                     | canopy          |
| 10        | 2021 | [MixChannel: Advanced Augmentation for Multispectral Satellite Images](https://doi.org/10.3390/rs13112181)                                              | multispectral   |
| 6         | 2025 | [Improving Small Drone Detection Through Multi-Scale Processing and Data Augmentation](https://doi.org/10.1109/ijcnn64981.2025.11227421)                | drone           |
| 5         | 2022 | [The Self-Supervised Spectral–Spatial Vision Transformer Network for Wheat Nitrogen Status from UAV](https://doi.org/10.3390/rs14061400)                | uav             |
| 4         | 2022 | [GANs for image augmentation in agriculture: A systematic review](https://doi.org/10.1016/j.compag.2022.107208)                                         | agriculture     |
| 4         | 2024 | [Ticino: A multi-modal remote sensing dataset for semantic segmentation](https://doi.org/10.1016/j.eswa.2024.123600)                                    | remote sensing  |
| 4         | 2022 | [HAGDAVS: Height-Augmented Geo-Located Dataset for Drone Aerial Orthomosaics](https://doi.org/10.3390/data7040050)                                      | drone           |
| 4         | 2022 | [A GIS Pipeline to Produce GeoAI Datasets from Drone Overhead Imagery](https://doi.org/10.3390/ijgi11100508)                                            | gis             |


### Top affiliations (≥ 3 geo papers)


| Affiliation                                               | # papers |
| --------------------------------------------------------- | -------- |
| Michigan State University                                 | 7        |
| Wuhan University                                          | 7        |
| Chinese Academy of Sciences                               | 4        |
| Skolkovo Institute of Science and Technology              | 4        |
| Zhejiang University                                       | 4        |
| Central South University                                  | 3        |
| Facultad de Minas                                         | 3        |
| Institute of Intelligent Emergency Information Processing | 3        |
| Ocean University of China                                 | 3        |
| Silesian University of Technology                         | 3        |
| Technical University of Munich (TUM)                      | 3        |
| University of California, Davis                           | 3        |


## HuggingFace Ecosystem

Across HuggingFace Hub artifacts tagged `remote-sensing` / `satellite-imagery` / `earth-observation` / `aerial-imagery` / `geospatial` / `land-cover`, **3 model cards reference Albumentations** in their training recipe (0 datasets — datasets typically don't carry augmentation pipelines, only training notebooks do).


| Kind  | ID                                                                                                                      | Downloads | Likes |
| ----- | ----------------------------------------------------------------------------------------------------------------------- | --------- | ----- |
| model | [Pranilllllll/segformer-satellite-segementation](https://huggingface.co/Pranilllllll/segformer-satellite-segementation) | 259       | 0     |
| model | [IsmatS/crop_desease_detection](https://huggingface.co/IsmatS/crop_desease_detection)                                   | 1         | 0     |
| model | [zcash/DEM-SuperRes-Model](https://huggingface.co/zcash/DEM-SuperRes-Model)                                             | 0         | 1     |


## What This Means

The 3D-geospatial / Earth-observation ML ecosystem already runs on Albumentations for the imagery half of essentially every supervised pipeline that ingests sensor frames, orthophotos, drone tiles, or satellite rasters. Funding maintenance of the underlying augmentation primitives — chromatic-shift, atmospheric, geometric, multispectral-safe ops — directly reduces friction for every grantee, every academic group, and every commercial EO operator listed above.

Every named org in the table above is a current, public-code user. Every library in the dependency table ships Albumentations transitively to its own users. The 382-paper citation count is a lower bound — it only counts papers whose metadata explicitly contains a geospatial keyword.

If you maintain a geospatial OSS project, foundation model, or training pipeline and want to be added to (or removed from) this evidence set, ping me — the methodology is fully scripted and the audit is rerun on demand.

---

*This brief is regenerated from the public APIs above. All counts are reproducible. Last regenerated 2026-04-19.*

*Hero image: nine Albumentations 2.2.0 transforms applied to the same Sentinel-2 tile of Moorea (French Polynesia). Source tile: ESA / CNES, [Copernicus Sentinel-2 imagery, 21 June 2021](https://commons.wikimedia.org/wiki/File:Moorea_et_Tahiti_vues_par_Sentinel_2.jpg), CC BY-SA 3.0 IGO. Grid produced by [build_hero.py](images/build_hero.py).*