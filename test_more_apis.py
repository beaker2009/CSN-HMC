#!/usr/bin/env python3
"""
尝试更多数据获取方式
"""
import warnings
warnings.filterwarnings("ignore")

import urllib.request
import json
import ssl
import time

# 禁用SSL验证（某些环境需要）
ssl._create_default_https_context = ssl._create_unverified_context

print("=" * 70)
print("测试更多数据接口")
print("=" * 70)

# 测试1: 聚宽 (JoinQuant)
print("\n[1] 测试聚宽API...")
try:
    url = "https://www.joinquant.com/data/dict/stockList"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req, timeout=10)
    data = response.read().decode()
    print(f"  ✓ 聚宽可用: {data[:100]}...")
except Exception as e:
    print(f"  ✗ 聚宽错误: {str(e)[:60]}")

# 测试2: 天软
print("\n[2] 测试天软API...")
try:
    # 测试基本连接
    print("  天软需要专业数据服务，跳过")
except Exception as e:
    print(f"  ✗ 天软错误: {str(e)[:60]}")

# 测试3: 万得(Wind) - 通常需要许可
print("\n[3] Wind API...")
print("  Wind需要商业许可，跳过")

# 测试4: 同花顺
print("\n[4] 测试同花顺...")
try:
    url = "http://www.10jqka.com.cn/"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req, timeout=10)
    print(f"  ✓ 同花顺网站可访问")
except Exception as e:
    print(f"  ✗ 同花顺错误: {str(e)[:60]}")

# 测试5: 通达信数据接口
print("\n[5] 测试通达信数据...")
try:
    # 通达信通常需要本地安装，这里测试网络接口
    print("  通达信需要本地客户端，跳过")
except Exception as e:
    print(f"  ✗ 通达信错误: {str(e)[:60]}")

# 测试6: 腾讯财经API
print("\n[6] 测试腾讯财经API...")
try:
    # 腾讯财经历史数据
    code = "sh600519"  # 茅台
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_dayqfq&param={code},day,2024-01-01,2024-12-31,1000,qfq"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://gu.qq.com'})
    response = urllib.request.urlopen(req, timeout=15)
    data = response.read().decode()
    if 'qfqday' in data or 'data' in data:
        print(f"  ✓ 腾讯财经API可用!")
        # 解析数据
        try:
            # 提取JSON部分
            json_str = data[data.index('=')+1:]
            result = json.loads(json_str)
            if 'data' in result and result['data']:
                stock_data = result['data'].get(code, {}).get('qfqday', [])
                print(f"    获取到 {len(stock_data)} 条数据")
                if stock_data:
                    print(f"    最新数据: {stock_data[-1]}")
        except Exception as parse_err:
            print(f"    解析数据出错: {parse_err}")
    else:
        print(f"  数据格式异常: {data[:100]}...")
except Exception as e:
    print(f"  ✗ 腾讯财经API错误: {str(e)[:60]}")

# 测试7: 新浪财经历史数据
print("\n[7] 测试新浪财经历史数据...")
try:
    code = "sh600519"  # 茅台
    url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    url += f"?symbol={code}&scale=240&ma=no&datalen=1000"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn'})
    response = urllib.request.urlopen(req, timeout=15)
    data = response.read().decode('utf-8')
    if data and data.startswith('['):
        result = json.loads(data)
        print(f"  ✓ 新浪财经API可用!")
        print(f"    获取到 {len(result)} 条日K线数据")
        if result:
            print(f"    最新数据: {result[-1]}")
    else:
        print(f"  数据格式异常: {data[:100]}...")
except Exception as e:
    print(f"  ✗ 新浪财经API错误: {str(e)[:60]}")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
