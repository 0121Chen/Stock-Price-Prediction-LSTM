# -*- coding: utf-8 -*-
"""双击 run.bat 时的交互式启动器：输入股票代码 -> 拉数据 -> 训练评估"""
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

code = input("请输入6位股票代码（直接回车默认贵州茅台 600519）: ").strip()
if not code:
    code = "600519"

print("=" * 52)
print(f"  A股股票价格预测 | 当前股票：{code}")
print("=" * 52)

print("\n[1/2] 拉取行情数据...")
r = subprocess.run([sys.executable, "fetch_data.py", "--code", code])
if r.returncode != 0:
    input("\n拉取数据失败！请检查：1) 代码是否正确  2) 网络/VPN 是否正常。按回车退出...")
    sys.exit(1)

print("\n[2/2] 训练模型并评估（首次约 1-2 分钟）...")
subprocess.run([sys.executable, "predict.py", "--data", f"data/{code}_daily.csv"])

print("\n完成！结果图在 output 文件夹：")
print(f"  - test_predictions_{code}.png   测试集预测对比图")
print(f"  - future_forecast_{code}.png    未来30天预测图")
input("\n按回车退出...")
