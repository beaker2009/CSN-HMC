
#!/usr/bin/env python3
"""
测试交易成本计算修复的验证脚本
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 创建模拟数据
dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='B')
assets = ['A', 'B', 'C']

# 创建模拟权重数据 - 类似月度调仓
w_mv = pd.DataFrame(index=dates, columns=assets, dtype=float)

# 1月权重
w_mv.loc['2024-01-01':'2024-01-31', :] = [0.5, 0.3, 0.2]
# 2月权重
w_mv.loc['2024-02-01':'2024-02-29', :] = [0.4, 0.4, 0.2]
# 3月权重
w_mv.loc['2024-03-01':'2024-03-31', :] = [0.3, 0.3, 0.4]

print("=== 模拟权重数据 ===")
print(w_mv.head())
print(w_mv.loc['2024-01-31':'2024-02-02', :])
print()

# 1. 原始方法（使用 diff）
print("=== 原始方法交易成本计算 ===")
tc_old = 0.003 * w_mv.diff().abs().sum(axis=1).fillna(0.0)
print(tc_old[tc_old > 0])
print(f"总交易成本: {tc_old.sum():.6f}")
print()

# 2. 修复方法（仅在调仓日计算）
rebalance_dates = pd.to_datetime(['2024-01-31', '2024-02-29', '2024-03-31'])

# 提取调仓日权重
rebalance_weight_mv = w_mv.loc[rebalance_dates].dropna(how='all')
print("=== 调仓日权重 ===")
print(rebalance_weight_mv)
print()

# 计算权重变化
weight_change_mv = rebalance_weight_mv.diff().abs().sum(axis=1).fillna(0.0)
print("=== 调仓日权重变化 ===")
print(weight_change_mv)
print()

# 模拟 _next_trading_position 函数
def _next_trading_position(date_series, target_date):
    return date_series.searchsorted(target_date, side='right')

# 构建新的交易成本序列
tc_new = pd.Series(0.0, index=dates)
for idx, date in enumerate(rebalance_dates):
    if idx == 0:
        continue
    next_pos = _next_trading_position(dates, date)
    if next_pos < len(dates):
        next_date = dates[next_pos]
        tc_new.loc[next_date] = weight_change_mv.get(date, 0.0)

tc_new = 0.003 * tc_new

print("=== 修复方法交易成本计算 ===")
print(tc_new[tc_new > 0])
print(f"总交易成本: {tc_new.sum():.6f}")
print()

# 对比两种方法
print("=== 对比 ===")
print(f"原始方法总交易成本: {tc_old.sum():.6f}")
print(f"修复方法总交易成本: {tc_new.sum():.6f}")
print(f"差异: {tc_old.sum() - tc_new.sum():.6f}")
