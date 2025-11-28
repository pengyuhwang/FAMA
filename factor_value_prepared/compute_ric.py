from factor_value_prepared.backtest_utilts_new import *
import pandas as pd
from factor_value_prepared.efficientCalculation import EfficientCalculator
from tqdm import tqdm
from pathlib import Path


def read_factor_file(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        return None
    # 读取存储的因子数据
    if file_path.suffix == ".parquet":
        factor_df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        factor_df = pd.read_csv(file_path, parse_dates=["time"])
    else:
        raise ValueError(f"不支持的文件格式{file_path}")
    return factor_df


def load_factor_long_csv(path: str) -> pd.DataFrame:
    """
    读取因子长表 CSV，要求列: time, unique_id, factor_tag, value
    返回 DataFrame，并保证 time 为 datetime、按 (unique_id, factor_tag, time) 排序。
    """
    df = pd.read_csv(path)
    required = {"time", "unique_id", "factor_tag", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"因子CSV缺少必要列: {sorted(missing)}")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["unique_id", "factor_tag", "time"]).reset_index(drop=True)
    return df


# 准备因子数据
factor_df = read_factor_file("/Users/hpy/PycharmProjects/FAMA/factor_value_prepared/data/factors/dsl_factors_new.parquet")

native_price, price_df, open_price_df, working_days = prepare_price_data(
    data_path="/Users/hpy/PycharmProjects/FAMA/data/fof_price_updating.parquet"
)

price_df = price_df.ffill().bfill()
open_price_df = open_price_df.ffill().bfill()
zz800 = price_df["000906.SH"]
tester = NMWBacktester(price_df, fee=0.0002)

pd.set_option("display.max_columns", None)
factor_df_500_1000 = factor_df[factor_df['unique_id'].isin(['000905.SH', '000852.SH'])]
# factor_df_500_1000 = factor_df

eff_calc = EfficientCalculator()

# 参数设置
need_assets = ['000905.SH', '000852.SH']  # 需要计算的资产
# need_assets = list(price_df.columns)
start_date = '2015-01-01'
end_date = '2020-12-31'

# 计算收益率（shift(-1)表示使用未来1日收益率）
asset_returns = price_df[need_assets].pct_change(1).shift(-1).dropna()

# 存储结果的列表
ric_results = []

print("开始计算RIC...")

# 筛选时间范围内的数据
factor_data = factor_df_500_1000.query(
    "time >= @start_date and time <= @end_date"
).copy()

# 按 unique_id 和 factor_tag 分组
grouped = factor_data.groupby(['unique_id', 'factor_tag'])

print(f"共有 {len(grouped)} 个 资产-因子 组合\n")

# 遍历每个组合
for (asset_id, factor_tag), group in tqdm(grouped, desc="计算RIC"):

    # 获取因子值（按时间排序）
    factor_series = group.set_index('time')['value'].sort_index()

    # 获取对应的收益率
    if asset_id not in asset_returns.columns:
        continue

    returns_series = asset_returns[asset_id]

    # 对齐数据
    common_dates = factor_series.index.intersection(returns_series.index)
    factor_aligned = factor_series.loc[common_dates]
    returns_aligned = returns_series.loc[common_dates]

    # 去除NaN
    valid_mask = ~(factor_aligned.isna() | returns_aligned.isna())
    factor_clean = factor_aligned[valid_mask]
    returns_clean = returns_aligned[valid_mask]

    # 检查样本数量
    if len(factor_clean) < 10:
        continue

    # 检查是否为常量
    if factor_clean.nunique() <= 1 or returns_clean.nunique() <= 1:
        continue

    # 计算RIC
    ric = eff_calc.efficent_cal_ric(factor_clean.values, returns_clean.values)

    if not pd.isna(ric):
        ric_results.append({
            'unique_id': asset_id,
            'factor_tag': factor_tag,
            'ric': ric,
            'sample_count': len(factor_clean),
            'start_date': factor_clean.index.min(),
            'end_date': factor_clean.index.max()
        })

# 转换为DataFrame
ric_df = pd.DataFrame(ric_results)

print(f"\n计算完成！共得到 {len(ric_df)} 个有效RIC结果")

# 按RIC绝对值排序
ric_df['abs_ric'] = ric_df['ric'].abs()
ric_df = ric_df.sort_values('abs_ric', ascending=False).reset_index(drop=True)
# 只看关键三列，去掉索引
ric_df.to_csv("/Users/hpy/PycharmProjects/FAMA/factor_value_prepared/data/factors/factor_ric.csv")
