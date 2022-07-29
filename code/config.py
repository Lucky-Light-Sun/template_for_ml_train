import torch

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# dataset
DATA_ROOT = '../../../data/format'
DATA_FOLDER = 'Atelectasis'
BATCH_SIZE = [16, 16, 8,  4, 4,   2,   1, 1, 1]
            # 5, 10, 20, 40, 80, 160, 320
NUM_WORKERS = 6
DATA_SET_MAX_NUM = 1000 ###############

# Optimizer
LR_G = 1e-5
LR_D = 1e-5
BETA1 = 0.5
BETA2 = 0.999
SCHEDULER_GAMMA = 0.99


# model
MAPPING_FMAPS = 512
DLATENT_SIZE = 512
RESOLUTION = 80
FMAP_BASE = 8192        ######### 8192
FMAP_MAX = 512          ######### 512
FMAP_DECAY = 1.0
NUM_CHANNELS = 1
HAS_SA = []   # 倒数第二，倒数第三个


# train
BASE_CONSTANT_IMAGE_SIZE = 5
START_TRAIN_AT_IMG_SIZE = 5   # TEST TEST TEST TEST TEST TEST
# PROGRESSIVE_EPOCHS = [50] * 100
PROGRESSIVE_EPOCHS = [300] * 100
LAMBDA_GP = 10
T_MAX = 20


# model save and load
SAVE_MODEL = True
LOAD_MODEL = False
MODEL_SAVE_DIR = '../model2'
SAVE_MODEL_INTERVAL = 100
SAVE_MODEL_INDEX = 0    # 'best' for the best


# log files
LOG_SAVE_DIR = '../logs2'


# save images dir
IMAGE_SAVE_DIR = '../images2'
SAMPLE_INTERVAL = 100
FIXED_BATCH_SIZE = 16
FIXED_NOISE = torch.randn(FIXED_BATCH_SIZE, DLATENT_SIZE).to(DEVICE)


