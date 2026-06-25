# This includes most of the configurations you may be interested in tinkering with.
from pathlib import Path


SEED = 42
PATH_ROOT = Path('../')
PATH_DATA = PATH_ROOT / 'data'
PATH_MIMICIV = PATH_DATA / 'MIMIC-IV_2_2'
PATH_PROCESSED = PATH_DATA / 'precomputed' # Where to keep processed data. This includes processed tables out of MIMIC, generative content such as summaries, and tensors.