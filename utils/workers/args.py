import logging

model_nam = "barissayil/bert-sentiment-analysis-sst"

rank = 0

world_size = 1  # number of gpu

TIMEOUT = 1
WORKER_TIMEOUT = 20
SLEEP = 1e-2
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
