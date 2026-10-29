# -*- coding: utf-8 -*-
"""拉取 A 股前复权日线数据（来源：新浪财经，经 akshare 获取）

用法：
    python fetch_data.py                  # 默认贵州茅台 600519
    python fetch_data.py --code 000001    # 平安银行
    python fetch_data.py --code 300750    # 宁德时代
"""
import argparse
import os

import akshare as ak
import pandas as pd

START = "20190101"  # 起始日期


def sina_symbol(code):
    """6 位代码 -> 新浪格式：沪市 6/9 开头 sh，深市 0/3 开头 sz，北交所 8/4 开头 bj"""
    if code.startswith(("6", "9", "5")):
        return "sh" + code
    if code.startswith(("8", "4")):
        return "bj" + code
    return "sz" + code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="600519", help="6 位股票代码，如 600519（茅台）")
    ap.add_argument("--start", default=START, help="起始日期 YYYYMMDD")
    args = ap.parse_args()

    symbol = sina_symbol(args.code)
    df = ak.stock_zh_a_daily(symbol=symbol, start_date=args.start, adjust="qfq")
    df = df[["date", "open", "high", "low", "close", "volume"]]
    os.makedirs("data", exist_ok=True)
    out = os.path.join("data", f"{args.code}_daily.csv")
    df.to_csv(out, index=False)
    print(f"已保存 {len(df)} 条记录 -> {out}")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
