import logging
import pytorch_lightning as pl

from architecture import MemNet

# See https://pytorch.org/docs/stable/notes/randomness.html
pl.seed_everything(SEED)

memnet  = MemNet(in_channels=3, channels=64, num_memblock=6, num_resblock=6)
trainer = pl.Trainer(fast_dev_run=True, gpus=1)
logger.setLevel(logging.DEBUG)

# 🐉🐉🐉
trainer.fit(memnet)
trainer.test()

memnet  = MemNet(in_channels=3, channels=64, num_memblock=6, num_resblock=6)
trainer = pl.Trainer(deterministic=DETERMINISTIC,         # Run slow, but same result every run
                     benchmark=BENCHMARK,                 # Run fast
                     log_gpu_memory='all',                # Log GPU memory data
                     profiler=PROFILE,                    # Identify perf bottlenecks
                     min_epochs=MIN_EPOCHS,               # Run for atleast MIN_EPOCHS
                     max_epochs=MAX_EPOCHS,               # Run for a max of MAX_EPOCHS
                     check_val_every_n_epoch=VAL_FREQ,    # Validate every VAL_FREQ
                     num_sanity_val_steps=5,              # 5 sanity checks before the real training
                     gpus=1,                              # Use the 1 GPU provided on Colab
)
logger.setLevel(level=logging.INFO)

# 🐉🐉🐉
trainer.fit(memnet)

