# AdaMixer: A Fast-Converging Query-Based Object Detector [arxiv](https://arxiv.org/abs/2203.16507)

> [**AdaMixer: A Fast-Converging Query-Based Object Detector**](https://arxiv.org/abs/2203.16507)<br>
> _accept to CVPR 2022 as an oral presentation_ <br>
> [Ziteng Gao](https://sebgao.github.io), [Limin Wang](http://wanglimin.github.io/), Bing Han, Sheng Guo<br>Nanjing University, MYbank Ant Group

[[slides]](adamixer_cvpr_2022_keynote.pdf)
[[arxiv]](https://arxiv.org/abs/2203.16507)

## 📰 News
[2022.7.3] Reproduced model checkpoints and logs are available.

[2022.4.4] The code is available now.

[2022.3.31] Code will be released in a few days (not too long). Pre-trained models will take some time to grant the permission of Ant Group to be available online. Please stay tuned or *watch this repo* for quick information.

## ✨ Highlights
### 🆕 MLP-Mixer for Object Detection
To our best knowledge, we are the first to introduce the MLP-Mixer for Object detection. The MLP-Mixer is used in the DETR-like decoder in an adaptive and query-wise manner to enrich the adaptibility to varying objects across images.

### ⚡️ Fast Converging DETR-like Architecture
AdaMixer enjoys fast convergence speed and reach up to 45.0 AP on COCO val within 12 epochs with only the architectural design improvement. Our method is compatible with other training improvements, like [multiple predictions from a query](https://github.com/megvii-research/AnchorDETR) and [denosing training](https://github.com/FengLi-ust/DN-DETR), which are expected to improve AdaMixer further (we have not tried yet). 

### 🧱 Simple Architecture, NO extra attentional encoders or FPNs required
Our AdaMixer does not hunger for extra attention encoders or explicit feature pyramid networks. Instead, we improve the query decoder in DETR-like detectors to keep the architecture as simple, efficient, and strong as possible.





## ➡️ Guide to Our Code
Our code structure follows the MMDetection framework. To get started, please refer to mmdetection doc [get_started.md](docs/get_started.md) for installation.

Our AdaMixer config file lies in [configs/adamixer](configs/adamixer) folder. You can start training our detectors with make targets in [Makefile](Makefile).

The code of a AdaMixer decoder stage is in
[mmdet/models/roi_heads/bbox_heads/adamixer_decoder_stage.py](mmdet/models/roi_heads/bbox_heads/adamixer_decoder_stage.py).
The code of the 3D feature space sampling is in [mmdet/models/roi_heads/bbox_heads/sampling_3d_operator.py](mmdet/models/roi_heads/bbox_heads/sampling_3d_operator.py).
The code of the adaptive mixing process is in [mmdet/models/roi_heads/bbox_heads/adaptive_mixing_operator.py](mmdet/models/roi_heads/bbox_heads/adaptive_mixing_operator.py).


__NOTE:__
1. Please use `mmcv_full==1.3.3` and `pytorch>=1.5.0` for correct reproduction ([#4](/../../issues/4), [#12](/../../issues/12)).~~Please make sure `init_weight` methods in `AdaptiveSamplingMixing` and `AdaptiveMixing`  are called for correct initializations *AND* the initialized weights are not overrided by other methods (some MMCV versions may incur repeated initializations).~~
2. We notice ~0.3 AP (42.7 AP reported in the paper) noise for AdaMixer w/ R50 with 1x training settings.

## 🧪 Main Results
Checkpoints and logs are available at [google drive](https://drive.google.com/drive/folders/1VPP-wJV6BzI_8MVy33RQZkV8ONgg5uLQ?usp=sharing).

| config |  detector | backbone  | APval | APtest  | APval (reprod.) | ckpt (reprod.) | log (reprod.) |
| :---: | :-------: | :------:  | :---: | :----:  | :---: | :---: | :---: | 
| [config](configs/adamixer/adamixer_r50_1x_coco.py) | AdaMixer (1x schedule, 100 queries)   |  R50     |  42.7  |   |  42.6 |[ckpt](https://drive.google.com/file/d/1v3rDczN2VXSgRYmSX0-XRy8racgpfed2/view?usp=sharing) | [log](https://drive.google.com/file/d/1nSjufpRiRYUn_gfJiG2RAv_MTWC2NiNr/view?usp=sharing) |
| [config](configs/adamixer/adamixer_r50_300_query_crop_mstrain_480-800_3x_coco.py) | AdaMixer (3x schedule, 300 queries)  |  R50     |  47.0  | 47.2   |  46.8 |[ckpt](https://drive.google.com/file/d/1Qj3sbcFF1zV8oeSfAWzZnYqv8mXWN8wJ/view?usp=sharing) | [log](https://drive.google.com/file/d/1EUAYtK-owJj1vlFG-NGaezw2uNWTjvzv/view?usp=sharing) |
| [config](configs/adamixer/adamixer_r101_300_query_crop_mstrain_480-800_3x_coco.py) | AdaMixer (3x schedule, 300 queries)  |  R101    |  48.0  | 48.1   |  48.1 | [ckpt](https://drive.google.com/file/d/1dx1-FZ20VDX7nCHtXh2_5AMrSysKEGK8/view?usp=sharing) | [log](https://drive.google.com/file/d/1MIocHsn9PHxWh5f1uua1OsEX_zbqiyX0/view?usp=sharing)
| [config](configs/adamixer/adamixer_dx101_300_query_crop_mstrain_480-800_3x_coco.py) | AdaMixer (3x schedule, 300 queries)  |  X101-DCN|  49.5  | 49.3   |  49.7 | [ckpt](https://drive.google.com/file/d/1vbIYuq8hvebP-DkqyFCMFVh5CSBjZ8cA/view?usp=sharing) | [log](https://drive.google.com/file/d/1nztwEaVSvNaM5os9NsecV97jilgHIuoO/view?usp=sharing)
| [config](configs/adamixer/adamixer_swin_s_300_query_crop_mstrain_480-800_3x_coco.py) | AdaMixer (3x schedule, 300 queries)   |  Swin-S  |  51.3  | 51.3   | on the way |

Special thanks to [Zhan Tong](https://github.com/yztongzhan) for these reproduced models.

## ✏️ Citation
If you find AdaMixer useful in your research, please cite us using the following entry:
```
@inproceedings{adamixer22cvpr,
  author    = {Ziteng Gao and
               Limin Wang and
               Bing Han and
               Sheng Guo},
  title     = {AdaMixer: A Fast-Converging Query-Based Object Detector},
  booktitle = {{CVPR}},
  year      = {2022}
}
```


## 👍 Acknowledgement
Thanks to [Zhan Tong](https://github.com/yztongzhan) and Zihua Xiong for their help.
























## Original MMDetection README.md
_The following begins the original mmdetection README.md file_
<div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>
</div>

**News**: We released the technical report on [ArXiv](https://arxiv.org/abs/1906.07155).

Documentation: https://mmdetection.readthedocs.io/

## Introduction

English | [简体中文](README_zh-CN.md)

MMDetection is an open source object detection toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.3+**.
The old v1.x branch works with PyTorch 1.1 to 1.4, but v2.0 is strongly recommended for faster speed, higher performance, better design and more friendly usage.

![demo image](resources/coco_test_12510.jpg)

### Major features

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **High efficiency**

  All basic bbox and mask operations run on GPUs. The training speed is faster than or comparable to other codebases, including [Detectron2](https://github.com/facebookresearch/detectron2), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](https://github.com/TuSimple/simpledet).

- **State of the art**

  The toolbox stems from the codebase developed by the *MMDet* team, who won [COCO Detection Challenge](http://cocodataset.org/#detection-leaderboard) in 2018, and we keep pushing it forward.

Apart from MMDetection, we also released a library [mmcv](https://github.com/open-mmlab/mmcv) for computer vision research, which is heavily depended on by this toolbox.

## License

The mmdetection project is released under the [Apache 2.0 license](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE).

## Changelog

v2.12.0 was released in 01/05/2021.
Please refer to [changelog.md](docs/changelog.md) for details and release history.
A comparison between v1.x and v2.0 codebases can be found in [compatibility.md](docs/compatibility.md).

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] VGG (ICLR'2015)
- [x] HRNet (CVPR'2019)
- [x] RegNet (CVPR'2020)
- [x] Res2Net (TPAMI'2020)
- [x] ResNeSt (ArXiv'2020)

Supported methods:

- [x] [RPN (NeurIPS'2015)](configs/rpn)
- [x] [Fast R-CNN (ICCV'2015)](configs/fast_rcnn)
- [x] [Faster R-CNN (NeurIPS'2015)](configs/faster_rcnn)
- [x] [Mask R-CNN (ICCV'2017)](configs/mask_rcnn)
- [x] [Cascade R-CNN (CVPR'2018)](configs/cascade_rcnn)
- [x] [Cascade Mask R-CNN (CVPR'2018)](configs/cascade_rcnn)
- [x] [SSD (ECCV'2016)](configs/ssd)
- [x] [RetinaNet (ICCV'2017)](configs/retinanet)
- [x] [GHM (AAAI'2019)](configs/ghm)
- [x] [Mask Scoring R-CNN (CVPR'2019)](configs/ms_rcnn)
- [x] [Double-Head R-CNN (CVPR'2020)](configs/double_heads)
- [x] [Hybrid Task Cascade (CVPR'2019)](configs/htc)
- [x] [Libra R-CNN (CVPR'2019)](configs/libra_rcnn)
- [x] [Guided Anchoring (CVPR'2019)](configs/guided_anchoring)
- [x] [FCOS (ICCV'2019)](configs/fcos)
- [x] [RepPoints (ICCV'2019)](configs/reppoints)
- [x] [Foveabox (TIP'2020)](configs/foveabox)
- [x] [FreeAnchor (NeurIPS'2019)](configs/free_anchor)
- [x] [NAS-FPN (CVPR'2019)](configs/nas_fpn)
- [x] [ATSS (CVPR'2020)](configs/atss)
- [x] [FSAF (CVPR'2019)](configs/fsaf)
- [x] [PAFPN (CVPR'2018)](configs/pafpn)
- [x] [Dynamic R-CNN (ECCV'2020)](configs/dynamic_rcnn)
- [x] [PointRend (CVPR'2020)](configs/point_rend)
- [x] [CARAFE (ICCV'2019)](configs/carafe/README.md)
- [x] [DCNv2 (CVPR'2019)](configs/dcn/README.md)
- [x] [Group Normalization (ECCV'2018)](configs/gn/README.md)
- [x] [Weight Standardization (ArXiv'2019)](configs/gn+ws/README.md)
- [x] [OHEM (CVPR'2016)](configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py)
- [x] [Soft-NMS (ICCV'2017)](configs/faster_rcnn/faster_rcnn_r50_fpn_soft_nms_1x_coco.py)
- [x] [Generalized Attention (ICCV'2019)](configs/empirical_attention/README.md)
- [x] [GCNet (ICCVW'2019)](configs/gcnet/README.md)
- [x] [Mixed Precision (FP16) Training (ArXiv'2017)](configs/fp16/README.md)
- [x] [InstaBoost (ICCV'2019)](configs/instaboost/README.md)
- [x] [GRoIE (ICPR'2020)](configs/groie/README.md)
- [x] [DetectoRS (ArXix'2020)](configs/detectors/README.md)
- [x] [Generalized Focal Loss (NeurIPS'2020)](configs/gfl/README.md)
- [x] [CornerNet (ECCV'2018)](configs/cornernet/README.md)
- [x] [Side-Aware Boundary Localization (ECCV'2020)](configs/sabl/README.md)
- [x] [YOLOv3 (ArXiv'2018)](configs/yolo/README.md)
- [x] [PAA (ECCV'2020)](configs/paa/README.md)
- [x] [YOLACT (ICCV'2019)](configs/yolact/README.md)
- [x] [CentripetalNet (CVPR'2020)](configs/centripetalnet/README.md)
- [x] [VFNet (ArXix'2020)](configs/vfnet/README.md)
- [x] [DETR (ECCV'2020)](configs/detr/README.md)
- [x] [Deformable DETR (ICLR'2021)](configs/deformable_detr/README.md)
- [x] [CascadeRPN (NeurIPS'2019)](configs/cascade_rpn/README.md)
- [x] [SCNet (AAAI'2021)](configs/scnet/README.md)
- [x] [AutoAssign (ArXix'2020)](configs/autoassign/README.md)
- [x] [YOLOF (CVPR'2021)](configs/yolof/README.md)


Some other methods are also supported in [projects using MMDetection](./docs/projects.md).

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection.
We provide [colab tutorial](demo/MMDet_Tutorial.ipynb), and full guidance for quick run [with existing dataset](docs/1_exist_data_model.md) and [with new dataset](docs/2_new_data_model.md) for beginners.
There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing models](docs/tutorials/customize_models.md), [customizing runtime settings](docs/tutorials/customize_runtime.md) and [useful tools](docs/useful_tools.md).

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMDetection. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
