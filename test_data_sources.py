#!/usr/bin/env python3
"""
测试各种数据源接口
"""
import sys

print("=" * 60)
print("测试数据源接口")
print("=" * 60)

# 测试1: baostock
print("\n[1] 测试 baostock...")
try:
    import baostock as bs
    lg = bs.login()
    print(f"  登录结果: {lg.error_code}, {lg.error_msg}")
    
    # 获取上证50成分股
    rs = bs.query_zz500_stocks()
    print(f"  查询结果: {rs.error_code}, {rs.error_msg}")
    
    if rs.error_code == '0':
        print("  ✓ baostock 可用!")
        bs.logout()
    else:
        print("  ✗ baostock 查询失败")
except Exception as e:
    print(f"  ✗ baostock 错误: {e}")

# 测试2: akshare
print("\n[2] 测试 akshare...")
try:
    import akshare as ak
    print(f"  akshare 版本: {ak.__version__}")
    
    # 尝试获取ETF列表
    df = ak.stock_zh_a_spot_em()
    print(f"  获取到 {len(df)} 只股票")
    print("  ✓ akshare 可用!")
except Exception as e:
    print(f"  ✗ akshare 错误: {e}")

# 测试3: tushare (需要token)
print("\n[3] 测试 tushare...")
try:
    import tushare as ts
    print("  tushare 已安装，但需要 token")
    print("  ✗ tushare 需要注册获取token")
except Exception as e:
    print(f"  ✗ tushare 错误: {e}")

# 测试4: yfinance
print("\n[4] 测试 yfinance...")
try:
    import yfinance as yf
    data = yf.download("AAPL", start="2024-01-01", end="2024-12-31", progress=False, timeout=10)
    print(f"  获取到 {len(data)} 条数据")
    print("  ✓ yfinance 可用!")
except Exception as e:
    print(f"  ✗ yfinance 错误: {e}")

# 测试5: pandas_datareader
print("\n[5] 测试 pandas_datareader...")
try:
    import pandas_datareader as pdr
    print("  pandas_datareader 已安装")
    # 尝试从FRED获取数据
    import pandas as pd
    data = pdr.get_data_fred('GS10')
    print(f"  获取到 {len(data)} 条数据")
    print("  ✓ pandas_datareader 可用!")
except Exception as e:
    print(f"  ✗ pandas_datareader 错误: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
