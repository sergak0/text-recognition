from mmcv import Config
from mmdet.apis import set_random_seed
from os.path import join

ROOT_DIR = './'

cfg = Config.fromfile(join(ROOT_DIR, 'mmdetection/configs/configs/detectors/cascade_rcnn_r50_sac_1x_coco.py'))

cfg.dataset_type = 'CocoDataset'
cfg.classes = join(ROOT_DIR, 'labels.txt')
cfg.data_root = join(ROOT_DIR, 'mmdetection/converter')
for i in range(len(cfg.model.roi_head.bbox_head)):  
  cfg.model.roi_head.bbox_head[i].num_classes = 1
cfg.data.train = cfg.data.train.dataset


cfg.data.test.type = 'CocoDataset'
cfg.data.test.classes = join(ROOT_DIR, 'labels.txt')
cfg.data.test.data_root = join(ROOT_DIR, 'mmdetection/convertor/val')
cfg.data.test.ann_file = 'annotations.json'
cfg.data.test.img_prefix = 'images'

cfg.data.train.type = 'CocoDataset'
cfg.data.train.data_root = join(ROOT_DIR, 'mmdetection/convertor/train')
cfg.data.train.ann_file = 'annotations.json'
cfg.data.train.img_prefix = 'images'
cfg.data.train.classes = join(ROOT_DIR, 'labels.txt')

cfg.data.val.type = 'CocoDataset'
cfg.data.val.classes = join(ROOT_DIR, 'labels.txt')
cfg.data.val.data_root = join(ROOT_DIR, 'mmdetection/convertor/val')
cfg.data.val.ann_file = 'annotations.json'
cfg.data.val.img_prefix = 'images'

cfg.load_from = '/content/drive/MyDrive/НТИ ИИ /team/sergey_models/mask-rcnn-resnet101/epoch_12.pth'

cfg.work_dir = './' #'/content/drive/MyDrive/НТИ ИИ /team/sergey_models/mask-rcnn-resnet101'

# The original learning rate (LR) is set for 8-GPU training.
# You divide it by 8 since you only use one GPU with Kaggle.
cfg.optimizer.lr = 0.01 / 8
cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

#cfg.lr_config.policy = 'step'
cfg.lr_config.step = 7
cfg.data.samples_per_gpu = 1
cfg.data.workers_per_gpu = 1
cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 12
cfg.checkpoint_config.interval = 6
cfg.log_config.interval = 100
cfg.runner.max_epochs = 18

cfg.model.test_cfg.max_per_img = 1000
cfg.model.test_cfg.rcnn.max_per_img = 1000
cfg.model.test_cfg.rcnn.mask_thr_binary = 0.3


cfg.model.train_cfg.max_per_img = 1000
cfg.model.train_cfg.rcnn.max_per_img = 1000

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
