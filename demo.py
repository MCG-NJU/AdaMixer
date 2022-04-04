from PIL import Image
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import sys
import os.path
name = sys.argv[1]
thr = sys.argv[2]
# config_file = './configs/fass/fass_r50_fpn_gn_1x.py'
# checkpoint_file = '../work_dirs/fass4/fass_r50_fpn_gn_1x-0125_1914-/latest.pth'

config_file = './configs/adamixer/adamixer_r50_1x_coco.py'
# config_file = './configs/featron/featron_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py'
checkpoint_file = name

if name == 'none':
    checkpoint_file = None

# build the model from a config file and a checkpoint file
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(test=dict(pipeline=test_pipeline))
# cfg_options = dict(data=data)
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img = 'data/coco/val2017/000000210299.jpg'  # person on bike
img = 'data/coco/val2017/000000057597.jpg'
# img = 'data/coco/val2017/000000577959.jpg'
# img = 'resources/corruptions_sev_3.png'
# img = 'demo/traffic.png'

# Image.open(img).save('demo_testin.jpg')
# assert False
print(img)
result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=float(thr),
                   out_file='demo/result.jpg')
