from KunQuant.jit import cfake
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from KunQuant.predefined import Alpha101, Alpha158
from KunQuant.runner import KunRunner as kr
# from .predefined import Alpha360
import numpy as np
import pandas as pd


def build_alpha_lib(module_name):
    builder = Builder()
    with builder:
        if module_name == "alpha158":
            pack_158 = Alpha158.AllData(
                low=Input("low"),
                high=Input("high"),
                close=Input("close"),
                open=Input("open"),
                amount=Input("amount"),
                volume=Input("volume"),
            )
            alpha158, names = pack_158.build(
                {
                    "kbar": {},
                    "price": {
                        "windows": [0],
                        "feature": [
                            ("OPEN", pack_158.open),
                            ("HIGH", pack_158.high),
                            ("LOW", pack_158.low),
                            ("VWAP", pack_158.vwap),
                        ],
                    },
                    "rolling": {
                        "windows": [5, 10, 20, 30, 60],
                    },
                }
            )
            for v, k in zip(alpha158, names):
                Output(v, k)
        elif module_name == "alpha101":
            pack_101 = Alpha101.AllData(
                low=Input("low"),
                high=Input("high"),
                close=Input("close"),
                open=Input("open"),
                amount=Input("amount"),
                volume=Input("volume"),
            )
            for cal_alpha in Alpha101.all_alpha:
                Output(cal_alpha(pack_101), cal_alpha.__name__)
        elif module_name == "alpha360":
            pack_360 = Alpha360.AllData(
                low=Input("low"),
                high=Input("high"),
                close=Input("close"),
                open=Input("open"),
                amount=Input("amount"),
                volume=Input("volume"),
            )
            alpha360, names = pack_360.build()
            for v, k in zip(alpha360, names):
                Output(v, k)
        else:
            raise NotImplementedError("Unknown libname: ", module_name)

    f = Function(builder.ops)
    return (module_name, f, KunCompilerConfig(input_layout="TS", output_layout="TS"))


def get_run_lib(target_modules, libname):
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    assert isinstance(target_modules, list)

    target = [build_alpha_lib(module_name) for module_name in target_modules]
    lib = cfake.compileit(target, libname, cfake.CppCompilerConfig())
    return lib


def collect_input_dict(native_price, working_days):
    # assert native_price的列名至少包含date, open, high, low, close, volume, amount
    assert set(native_price.columns) >= set(
        ["date", "open", "high", "low", "close", "volume", "amount"]
    )

    print(f"collecting start from {working_days[0]} to {working_days[-1]}")

    unique_ids = sorted(native_price["unique_id"].unique().tolist())
    print("unique_ids", unique_ids)
    watch_list = unique_ids
    num_stocks = len(watch_list)

    df = []
    for unique_id in watch_list:
        d = (
            native_price[native_price["unique_id"] == unique_id]
            .set_index("date")
            .reindex(working_days).ffill()
        )
        d = d[["open", "high", "low", "close", "volume", "amount"]]
        assert len(d) == len(working_days)
        df.append(d)

    cols = df[0].columns.values
    col2idx = dict(zip(cols, range(len(cols))))
    print("columns to index", col2idx)
    num_time = len(df[0])
    print("dimension in time", num_time)

    # [features, stocks, time]
    collected = np.empty((len(col2idx), num_stocks, len(df[0])), dtype="float32")
    for stockidx, data in enumerate(df):
        for colname, colidx in col2idx.items():
            mat = data[colname].to_numpy()
            collected[colidx, stockidx, :] = mat

    # [features, stocks, time] => [features, time, stocks]
    transposed = collected.transpose((0, 2, 1))
    transposed = np.ascontiguousarray(transposed)

    input_dict = dict()
    for colname, colidx in col2idx.items():
        input_dict[colname] = transposed[colidx]
    other_info = {
        "transposed": transposed,
        "df": df,
        "col2idx": col2idx,
        "watch_list": watch_list,
        "unique_ids": unique_ids,
        "num_stocks": num_stocks,
        "num_time": num_time,
    }
    return input_dict, other_info


def run_lib_module(
    input_dict, lib, module_name, num_time, num_stocks, multi_thread_num=4
):
    # results are in "out" and "sharedbuf"
    modu = lib.getModule(module_name)
    outnames = modu.getOutputNames()
    out_dict = dict()
    # [Factors, Time, Stock]
    sharedbuf = np.empty((len(outnames), num_time, num_stocks), dtype="float32")
    for idx, name in enumerate(outnames):
        out_dict[name] = sharedbuf[idx]

    executor = kr.createMultiThreadExecutor(multi_thread_num)
    out = kr.runGraph(executor, modu, input_dict, 0, num_time, out_dict)
    return out


def parse_out_to_factor_df(out, other_info, working_days):
    import gc

    unique_ids = other_info["unique_ids"]
    factor_dfs = []
    for factor in out.keys():
        part_factor_df = (
            pd.DataFrame(out[factor], index=working_days, columns=unique_ids)
            .reset_index()
            .melt(var_name="unique_id", value_name="value", id_vars="time")
        )
        part_factor_df["factor"] = factor
        part_factor_df = part_factor_df[["time", "factor", "unique_id", "value"]]
        factor_dfs.append(part_factor_df)

    factor_df = pd.concat(factor_dfs)
    del factor_dfs
    gc.collect()
    return factor_df
