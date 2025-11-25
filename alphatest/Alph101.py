import pandas as pd
from alphatest.backtest_utilts_new import *
from alphatest.FactorCollection import FactorCollection

factor_collection: FactorCollection = FactorCollection()
# factor_df = factor_collection.load_factor_df(["alpha101"])
factor_collection.update_alphas()

