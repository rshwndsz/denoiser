from torch.cuda import is_available

SEED          = 42

LOG_FORMAT    = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE      = 'russel-denoising-log.log'

IN_CHANNELS   = 3
IMG_SIZE      = (128, 128)
SIGMA         = 30

BSDS_URL      = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
BSDS_ROOT     = "./BSDS300/images"

INIT_LR       = 3e-4
BATCH_SIZE    = 8
NUM_WORKERS   = 4
USE_GPU       = is_available()

DETERMINISTIC = True
BENCHMARK     = False
PROFILE       = False
MIN_EPOCHS    = 12
MAX_EPOCHS    = 50
VAL_FREQ      = 3
