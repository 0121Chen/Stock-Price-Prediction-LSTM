# -*- coding: utf-8 -*-
"""
A 股股票价格预测：LSTM 与基线模型对比（多步预测版）

- 数据：贵州茅台（600519）前复权日线，`fetch_data.py` 一键更新
- 模型：持久化基线 / 线性回归 / LSTM，三者同窗口、同切分，公平对比
- 评估：RMSE / MAE / MAPE（按时间顺序切分，绝不随机打乱）
- 多步预测：用测试集末尾窗口递归预测未来 30 个交易日

用法：
    python fetch_data.py              # 先拉数据
    python predict.py                 # 训练 + 评估 + 出图
    python predict.py --epochs 50     # 调整 LSTM 训练轮数
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # 无窗口环境，图片只落盘
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

# 中文字体（否则图片里的中文会显示为方框）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

LOOKBACK = 10       # 用过去 10 个交易日预测下一天
FUTURE_DAYS = 30    # 递归预测未来天数
SPLIT_RATIO = 0.8   # 80% 训练 / 20% 测试（时间顺序）


def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def make_sequences(series, lookback):
    """滑窗构造 (X, y)：X[i] = series[i-lookback:i]，y[i] = series[i]"""
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def mape(y_true, y_pred):
    """平均绝对百分比误差（%）"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = np.abs(y_true) > 1e-8
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(name, y_true, y_pred, results):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mp = float(mape(y_true, y_pred))
    results[name] = {"rmse": rmse, "mae": mae, "mape": mp}
    print(f"{name:<10}  RMSE={rmse:>9.2f}   MAE={mae:>8.2f}   MAPE={mp:>6.2f}%")


def recursive_forecast(model_fn, seed_window, steps, scaler=None):
    """递归多步预测：每次把预测值接回窗口尾部，再预测下一天"""
    window = list(seed_window)
    preds = []
    for _ in range(steps):
        w = np.array(window[-LOOKBACK:], dtype=float)
        if scaler is not None:  # LSTM：先缩放 -> 预测 -> 反缩放
            w_s = scaler.transform(w.reshape(-1, 1)).reshape(1, LOOKBACK, 1)
            p = model_fn(w_s).ravel()[0]
            p = float(scaler.inverse_transform([[p]])[0, 0])
        else:                   # 线性回归：直接预测
            p = float(model_fn(w.reshape(1, -1)).ravel()[0])
        preds.append(p)
        window.append(p)
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "600519_daily.csv"))
    ap.add_argument("--name", default=None, help="股票名称（用于图片标题，默认取数据文件名）")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lookback", type=int, default=LOOKBACK)
    ap.add_argument("--future", type=int, default=FUTURE_DAYS)
    args = ap.parse_args()

    df = load_data(args.data)
    close = df["close"].values.astype(float)
    name = args.name or os.path.splitext(os.path.basename(args.data))[0].replace("_daily", "")
    print(f"数据：{args.data}，{len(df)} 个交易日，收盘价范围 "
          f"{close.min():.2f} ~ {close.max():.2f}\n")

    # ---- 时间顺序切分（不打乱）----
    split = int(len(close) * SPLIT_RATIO)
    train_raw, test_raw = close[:split], close[split:]
    test_true = test_raw[args.lookback:]          # 各模型统一评估区间
    test_windows = make_sequences(test_raw, args.lookback)[0]

    results = {}

    # ---- 基线 1：持久化（拿前一天价格当预测）----
    print("== 模型对比（测试集） ==")
    pred_persist = test_raw[args.lookback - 1:-1]
    evaluate("持久化基线", test_true, pred_persist, results)

    # ---- 基线 2：线性回归（同 lookback 窗口）----
    Xtr_lr, ytr_lr = make_sequences(train_raw, args.lookback)
    lr = LinearRegression().fit(Xtr_lr, ytr_lr)
    evaluate("线性回归", test_true, lr.predict(test_windows), results)

    # ---- LSTM ----
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1)).ravel()
    Xtr, ytr = make_sequences(train_scaled, args.lookback)
    Xtr = Xtr.reshape(Xtr.shape[0], args.lookback, 1)
    test_scaled = scaler.transform(test_raw.reshape(-1, 1)).ravel()
    Xte = make_sequences(test_scaled, args.lookback)[0]
    Xte = Xte.reshape(Xte.shape[0], args.lookback, 1)

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(args.lookback, 1)),
        LSTM(16),
        Dense(1, activation="linear"),
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(Xtr, ytr, epochs=args.epochs, batch_size=32, verbose=0)
    pred_lstm = scaler.inverse_transform(model.predict(Xte, verbose=0)).ravel()
    evaluate("LSTM", test_true, pred_lstm, results)
    print()

    # ---- 图 1：测试集真实 vs 三模型预测 ----
    plt.figure(figsize=(12, 6))
    plt.plot(test_true, "g-", label="真实收盘价", linewidth=1.2)
    plt.plot(pred_persist, "k--", label="持久化基线", alpha=0.6)
    plt.plot(lr.predict(test_windows), "y-.", label="线性回归", alpha=0.7)
    plt.plot(pred_lstm, "r-", label="LSTM", linewidth=1.2)
    plt.legend()
    plt.xlabel("测试集交易日序号")
    plt.ylabel("收盘价（元）")
    plt.title(f"测试集预测对比（{name}）")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join("output", f"test_predictions_{name}.png"), dpi=150, bbox_inches="tight")

    # ---- 图 2：未来 30 天递归多步预测 ----
    future_lstm = recursive_forecast(
        lambda w: model.predict(w, verbose=0),
        test_scaled[-args.lookback:], args.future, scaler=scaler)
    future_persist = np.full(args.future, test_raw[-1])  # 基线：未来保持最后一天价格
    x_axis = np.arange(-60, args.future)
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis[:60], close[-60:], "g-", label="历史收盘价", linewidth=1.2)
    plt.plot(np.arange(args.future), future_persist, "k--", label="持久化基线（未来）", alpha=0.6)
    plt.plot(np.arange(args.future), future_lstm, "r-", label="LSTM 递归预测（未来）", linewidth=1.2)
    plt.axvline(0, color="gray", linestyle=":", alpha=0.6)
    plt.legend()
    plt.xlabel("交易日（0 为今天）")
    plt.ylabel("收盘价（元）")
    plt.title(f"未来 {args.future} 个交易日递归预测")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join("output", f"future_forecast_{name}.png"), dpi=150, bbox_inches="tight")

    # ---- 汇总表 ----
    print("== 汇总 ==")
    tbl = pd.DataFrame(results).T.round(2)
    print(tbl.to_string())
    print("\n图片已保存到 output/ 目录")


if __name__ == "__main__":
    main()
