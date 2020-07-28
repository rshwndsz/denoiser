import logging
import argparse
import pytorch_lightning as pl

from src.logger import logger
from src.architecture import MemNet
import src.constants as C


def setup():
    """Deterministic setup"""
    # See https://pytorch.org/docs/stable/notes/randomness.html
    pl.seed_everything(C.SEED)


def dev():
    """Sanity run

    Run a train, validation, testing epoch
    """
    memnet  = MemNet(in_channels=3, channels=64, num_memblock=6, num_resblock=6)
    trainer = pl.Trainer(fast_dev_run=True, gpus=1)
    logger.setLevel(logging.DEBUG)

    # 🐉🐉🐉
    trainer.fit(memnet)
    trainer.test()


def train():
    """Start training phase"""
    memnet  = MemNet(in_channels=3, channels=64, num_memblock=6, num_resblock=6)
    trainer = pl.Trainer(deterministic=C.DETERMINISTIC,      # Run slow, but same result every run
                         benchmark=C.BENCHMARK,              # Run fast
                         log_gpu_memory='all',               # Log GPU memory data
                         profiler=C.PROFILE,                 # Identify perf bottlenecks
                         min_epochs=C.MIN_EPOCHS,            # Run for atleast MIN_EPOCHS
                         max_epochs=C.MAX_EPOCHS,            # Run for a max of MAX_EPOCHS
                         check_val_every_n_epoch=C.VAL_FREQ, # Validate every VAL_FREQ
                         num_sanity_val_steps=5,             # 5 sanity checks before the real training
                         gpus=1,                             # Use the 1 GPU provided on Colab
                         )
    logger.setLevel(level=logging.INFO)
    # 🐉🐉🐉
    trainer.fit(memnet)


def test():
    """Inference"""
    trainer = pl.Trainer()
    trainer.test()


if __name__ == "__main__":
    # Parse arguments for sanity, trainval, test
    parser = argparse.ArgumentParser(description='Options for Image denoising')
    parser.add_argument('--run',
                        choices=['dev', 'train', 'test'],
                        help='Different modes of running the model')
    args = parser.parse_args()

    setup()

    if args.run == 'dev':
        dev()
    elif args.run == 'train':
        train()
    elif args.run == 'test':
        test()
    else:
        print("Invalid argument. Enter one of ['dev', 'train', 'test'].")
