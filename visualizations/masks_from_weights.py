import toml
from helper_functions.utils import create_masks

fold = 0
config = toml.load(open('../configs/demolisher.toml'))
create_masks(fold, config, 0.5)
