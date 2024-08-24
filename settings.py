DEBUG = False

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9

VIEW_DIM = 5
assert VIEW_DIM % 2 == 1, "View dimension must be odd"