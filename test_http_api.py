#!/usr/bin/env python3
"""
直接从财经网站API获取真实市场数据
使用新浪财经API
"""
import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

print("=" * 70)
print("从新浪财经API获取真实市场数据")
print("=" * 70)

# 上证50成分股代码（部分）
STOCK_CODES = [
    "sh600519", "sh600036", "sh601318", "sh600016", "sh601166",  # 茅台、招行、平安、民生、兴业
    "sh600030", "sh601328", "sh600887", "sh601288", "sh601398",  # 中信、交通、伊利、农业、工行
    "sh600000", "sh601169", "sh601818", "sh601601", "sh600050",  # 浦发、北京、光大、太保、联通
    "sh600048", "sh600028", "sh601668", "sh600309", "sh601186",  # 保利、中石化、建筑、万华、铁建
    "sh600031", "sh601012", "sh601390", "sh601628", "sh601088",  # 三一、隆基、中铁、国寿、神华
    "sh600585", "sh600690", "sh601888", "sh600276", "sh601766",  # 海螺、青岛海尔、国旅、恒瑞、中车
]

def get_stock_data_sina(code, start_date, end_date):
    """从新浪财经获取单只股票数据"""
    try:
        # 转换日期格式
        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "")
        
        # 新浪财经API
        url = f"https://finance.yahoo.com/quote/{code.replace('sh', '')}.{'SS' if code.startswith('sh') else 'SZ'}/history"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return True
        return False
    except Exception as e:
        return False

# 测试几只股票
print("\n测试数据获取...")
success_count = 0
for code in STOCK_CODES[:5]:
    result = get_stock_data_sina(code, "2024-01-01", "2024-12-31")
    status = "✓" if result else "✗"
    print(f"  {status} {code}")
    if result:
        success_count += 1
    time.sleep(0.5)

if success_count == 0:
    print("\n新浪财经API不可用，尝试备用方案...")

# 备用方案：使用Yahoo Finance API
print("\n尝试 Yahoo Finance API...")
def get_yahoo_data():
    """使用Yahoo Finance API"""
    try:
        import urllib.request
        import json
        
        # Yahoo Finance历史数据API
        symbol = "AAPL"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        url += "?period1=1704067200&period2=1735689600&interval=1d"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=15)
        data = json.loads(response.read())
        
        if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            closes = result['indicators']['quote'][0]['close']
            print(f"  ✓ Yahoo Finance API可用，获取到 {len(timestamps)} 条数据")
            return True
        return False
    except Exception as e:
        print(f"  ✗ Yahoo Finance API错误: {str(e)[:50]}")
        return False

yahoo_ok = get_yahoo_data()

# 尝试东方财富API
print("\n尝试东方财富API...")
def get_eastmoney_data():
    """使用东方财富API"""
    try:
        import urllib.request
        import json
        
        # 东方财富股票列表API
        url = "http://push2.eastmoney.com/api/qt/clist/get"
        params = {
            'pn': '1',
            'pz': '5',  # 只获取5只测试
            'po': '1',
            'np': '1',
            'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
            'fltt': '2',
            'invt': '2',
            'fid': 'f3',
            'fs': 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23',
            'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23'
        }
        
        url += "?" + "&".join([f"{k}={v}" for k, v in params.items()])
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=10)
        data = json.loads(response.read())
        
        if 'data' in data and data['data'] and 'diff' in data['data']:
            print(f"  ✓ 东方财富API可用!")
            return True
        return False
    except Exception as e:
        print(f"  ✗ 东方财富API错误: {str(e)[:60]}")
        return False

eastmoney_ok = get_eastmoney_data()

# 尝试网易财经API
print("\n尝试网易财经API...")
def get_netease_data():
    """使用网易财经API"""
    try:
        import urllib.request
        import json
        
        # 网易财经历史数据API - 上证指数
        code = "0601857"  # 上证指数代码
        url = f"https://quotes.money.163.com/service/chddata.html"
        url += f"?code=0{code}&start=20240101&end=20241231&fields=TCLOSE"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=15)
        
        # 网易返回CSV格式
        content = response.read().decode('gb2312')
        lines = content.strip().split('\n')
        print(f"  ✓ 网易财经API可用，获取到 {len(lines)} 行数据")
        
        # 解析数据
        if len(lines) > 2:
            print(f"  示例数据: {lines[1][:80]}...")
            return True
        return False
    except Exception as e:
        print(f"  ✗ 网易财经API错误: {str(e)[:60]}")
        return False

netease_ok = get_netease_data()

print("\n" + "=" * 70)
print("数据源测试结果:")
print(f"  Yahoo Finance: {'可用' if yahoo_ok else '不可用'}")
print(f"  东方财富: {'可用' if eastmoney_ok else '不可用'}")
print(f"  网易财经: {'可用' if netease_ok else '不可用'}")
print("=" * 70)
