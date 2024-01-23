from helper_functions.utils import cover_from_mask
import toml

fold = 0
config = toml.load(open('../configs/demolisher.toml'))
cover_from_mask(fold, config)