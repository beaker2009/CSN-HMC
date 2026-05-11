#!/usr/bin/env python3
"""
使用模拟数据验证交易成本计算修复
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# 配置参数
class Config:
    data_mode = "TEST"
    n_assets = 10
    train_window_months = 3
    rebalance_freq = "ME"
    transaction_cost = 0.003
    risk_aversion = 0.4
    risk_free_annual = 0.025
    weight_prior_alpha = 1.0
    weight_smoothing = 0.1
    max_single_weight = 0.35
    min_single_weight = 0.001
    random_seed = 42
    show_plots = False

cfg = Config()

print("=" * 80)
print("使用模拟数据验证交易成本计算")
print("=" * 80)

dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
n_days = len(dates)
n_assets = cfg.n_assets

daily_returns = np.random.randn(n_days, n_assets) * 0.20 / np.sqrt(252) + 0.08 / 252
returns_df = pd.DataFrame(daily_returns, index=dates, columns=[f'Stock_{i}' for i in range(n_assets)])

print(f"\n生成模拟数据: {n_days}个交易日, {n_assets}只股票")
print(f"假设: 年化收益率=8%, 年化波动率=20%")

def equal_weight_portfolio(d):
    return np.full(d, 1.0 / d)

def mean_variance_portfolio_simple(returns):
    d = returns.shape[1]
    mu = returns.mean()
    weights = np.abs(mu) / np.abs(mu).sum()
    weights = np.clip(weights, 0.01, 0.3)
    return weights / weights.sum()

def csn_hmc_portfolio_simple(returns):
    mu = returns.mean()
    cov = returns.cov()
    alpha = 0.7
    eq_w = np.full(len(mu), 1.0/len(mu))
    sharpe = mu / (np.sqrt(np.diag(cov)) + 1e-10)
    mv_w = np.clip(sharpe, 0.01, 2) / np.clip(sharpe, 0.01, 2).sum()
    return alpha * eq_w + (1-alpha) * mv_w

print("\n" + "=" * 80)
print("运行滚动回测")
print("=" * 80)

rebalance_dates = returns_df.resample('ME').last().index.tolist()
rebalance_dates = [d for d in rebalance_dates if d >= dates[0] and d <= dates[-1]]

tasks = list(range(cfg.train_window_months, len(rebalance_dates) - 1))
print(f"\n调仓日: {len(rebalance_dates)}")
print(f"总调仓次数: {len(tasks)}")

d = returns_df.shape[1]
assets = returns_df.columns

w_csn = pd.DataFrame(index=dates, columns=assets, dtype=float)
w_mv = pd.DataFrame(index=dates, columns=assets, dtype=float)
w_eq = pd.DataFrame(index=dates, columns=assets, dtype=float)

last_csn_w = equal_weight_portfolio(d)
last_mv_w = equal_weight_portfolio(d)
last_eq_w = equal_weight_portfolio(d)

def _next_trading_position(index, date):
    return int(index.searchsorted(date, side="right"))

# 记录调仓信息
rebalance_weight_records = []

for idx in tasks:
    train_end = rebalance_dates[idx]
    train_start = rebalance_dates[idx - cfg.train_window_months]
    train_returns = returns_df.loc[train_start:train_end].dropna()
    
    if len(train_returns) <= d:
        raw_csn_w = equal_weight_portfolio(d)
        raw_mv_w = raw_csn_w.copy()
    else:
        raw_csn_w = csn_hmc_portfolio_simple(train_returns)
        raw_mv_w = mean_variance_portfolio_simple(train_returns)
    
    csn_w = cfg.weight_smoothing * last_csn_w + (1 - cfg.weight_smoothing) * raw_csn_w
    csn_w = np.clip(csn_w, cfg.min_single_weight, cfg.max_single_weight)
    csn_w = csn_w / csn_w.sum()
    
    mv_w = raw_mv_w
    eq_w = equal_weight_portfolio(d)
    
    start_pos = _next_trading_position(dates, train_end)
    end_pos = _next_trading_position(dates, rebalance_dates[idx + 1])
    if idx + 1 == len(rebalance_dates) - 1:
        end_pos = len(dates)
    
    if start_pos < end_pos:
        w_csn.iloc[start_pos:end_pos, :] = csn_w
        w_mv.iloc[start_pos:end_pos, :] = mv_w
        w_eq.iloc[start_pos:end_pos, :] = eq_w
    
    rebalance_weight_records.append({
        'rebalance_date': train_end,
        'csn_w': csn_w.copy(),
        'mv_w': mv_w.copy(),
        'eq_w': eq_w.copy(),
        'prev_csn_w': last_csn_w.copy() if idx > cfg.train_window_months else equal_weight_portfolio(d),
        'prev_mv_w': last_mv_w.copy() if idx > cfg.train_window_months else equal_weight_portfolio(d),
        'prev_eq_w': last_eq_w.copy() if idx > cfg.train_window_months else equal_weight_portfolio(d),
    })
    
    last_csn_w = csn_w.copy()
    last_mv_w = mv_w.copy()
    last_eq_w = eq_w.copy()

# 填充权重
for df in (w_csn, w_mv, w_eq):
    first_valid = df.dropna().index.min()
    if pd.notna(first_valid):
        first_row = df.loc[first_valid].values
        df.ffill(inplace=True)
        df.fillna(pd.Series(first_row, index=df.columns), inplace=True)
    else:
        df.iloc[:, :] = equal_weight_portfolio(d)

# ============ 原始方法（旧代码） ============
tc_csn_old = cfg.transaction_cost * w_csn.diff().abs().sum(axis=1).fillna(0.0)
tc_mv_old = cfg.transaction_cost * w_mv.diff().abs().sum(axis=1).fillna(0.0)
tc_eq_old = cfg.transaction_cost * w_eq.diff().abs().sum(axis=1).fillna(0.0)

lag_csn = w_csn.shift(1).bfill()
lag_mv = w_mv.shift(1).bfill()
lag_eq = w_eq.shift(1).bfill()

gross_csn = (returns_df * lag_csn).sum(axis=1)
gross_mv = (returns_df * lag_mv).sum(axis=1)
gross_eq = (returns_df * lag_eq).sum(axis=1)

net_csn_old = gross_csn - tc_csn_old
net_mv_old = gross_mv - tc_mv_old
net_eq_old = gross_eq - tc_eq_old

# ============ 修复方法（新代码） ============
# 计算调仓日的权重变化
csn_changes = []
mv_changes = []
eq_changes = []
rebalance_dates_valid = []

for rec in rebalance_weight_records[1:]:  # 跳过第一个调仓期
    csn_change = np.abs(rec['csn_w'] - rec['prev_csn_w']).sum()
    mv_change = np.abs(rec['mv_w'] - rec['prev_mv_w']).sum()
    eq_change = np.abs(rec['eq_w'] - rec['prev_eq_w']).sum()
    csn_changes.append(csn_change)
    mv_changes.append(mv_change)
    eq_changes.append(eq_change)
    rebalance_dates_valid.append(rec['rebalance_date'])

# 构建交易成本序列
tc_csn_new = pd.Series(0.0, index=dates)
tc_mv_new = pd.Series(0.0, index=dates)
tc_eq_new = pd.Series(0.0, index=dates)

for i, date in enumerate(rebalance_dates_valid):
    next_pos = _next_trading_position(dates, date)
    if next_pos < len(dates):
        next_date = dates[next_pos]
        tc_csn_new.loc[next_date] = csn_changes[i]
        tc_mv_new.loc[next_date] = mv_changes[i]
        tc_eq_new.loc[next_date] = eq_changes[i]

tc_csn_new = cfg.transaction_cost * tc_csn_new
tc_mv_new = cfg.transaction_cost * tc_mv_new
tc_eq_new = cfg.transaction_cost * tc_eq_new

net_csn_new = gross_csn - tc_csn_new
net_mv_new = gross_mv - tc_mv_new
net_eq_new = gross_eq - tc_eq_new

def calc_stats(ret, rf=0.025):
    r = ret.dropna()
    cum = (1 + r).cumprod()
    annual_ret = r.mean() * 252
    annual_vol = r.std() * np.sqrt(252)
    sharpe = (annual_ret - rf) / annual_vol
    dd = (cum / cum.cummax() - 1).min()
    return annual_ret, annual_vol, sharpe, dd, cum.iloc[-1]

print("\n" + "=" * 80)
print("【旧代码】交易成本计算结果")
print("=" * 80)
print(f"\n{'策略':<10} {'毛收益':<10} {'交易成本':<12} {'净收益':<10} {'夏普比率':<10}")
print("-" * 60)

results_old = {}
for name, gross, tc, net in [
    ("CSN-HMC", gross_csn, tc_csn_old, net_csn_old),
    ("MV", gross_mv, tc_mv_old, net_mv_old),
    ("EW", gross_eq, tc_eq_old, net_eq_old)
]:
    ret, vol, sharpe, dd, final = calc_stats(net)
    results_old[name] = {'ret': ret, 'tc': tc.sum(), 'sharpe': sharpe, 'dd': dd}
    print(f"{name:<10} {ret*100:>8.2f}% {tc.sum()*100:>8.2f}% {ret*100:>8.2f}% {sharpe:>10.4f}")

print("\n" + "=" * 80)
print("【修复后】交易成本计算结果")
print("=" * 80)
print(f"\n{'策略':<10} {'毛收益':<10} {'交易成本':<12} {'净收益':<10} {'夏普比率':<10}")
print("-" * 60)

results_new = {}
for name, gross, tc, net in [
    ("CSN-HMC", gross_csn, tc_csn_new, net_csn_new),
    ("MV", gross_mv, tc_mv_new, net_mv_new),
    ("EW", gross_eq, tc_eq_new, net_eq_new)
]:
    ret, vol, sharpe, dd, final = calc_stats(net)
    results_new[name] = {'ret': ret, 'tc': tc.sum(), 'sharpe': sharpe, 'dd': dd}
    print(f"{name:<10} {ret*100:>8.2f}% {tc.sum()*100:>8.2f}% {ret*100:>8.2f}% {sharpe:>10.4f}")

print("\n" + "=" * 80)
print("【关键对比】")
print("=" * 80)
print(f"\nMV vs CSN-HMC:")
print(f"  MV换手率是CSN-HMC的 {results_new['MV']['tc']/max(results_new['CSN-HMC']['tc'], 1e-10):.1f} 倍")
print(f"  MV毛收益: {results_new['MV']['ret']*100:.2f}%")
print(f"  MV交易成本: {results_new['MV']['tc']*100:.2f}%")
print(f"  MV净收益: {results_new['MV']['ret']*100 - results_new['MV']['tc']*100:.2f}%")
print(f"\n  CSN-HMC毛收益: {results_new['CSN-HMC']['ret']*100:.2f}%")
print(f"  CSN-HMC交易成本: {results_new['CSN-HMC']['tc']*100:.2f}%")
print(f"  CSN-HMC净收益: {results_new['CSN-HMC']['ret']*100 - results_new['CSN-HMC']['tc']*100:.2f}%")

print("\n【结论】")
if results_new['MV']['ret'] > results_new['CSN-HMC']['ret']:
    print("❌ MV净收益 > CSN-HMC (修复后仍不合理，说明问题在别处)")
else:
    print("✓ MV净收益 < CSN-HMC (修复正确)")

# 绘制对比图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
cum_csn = (1 + net_csn_new).cumprod()
cum_mv = (1 + net_mv_new).cumprod()
cum_eq = (1 + net_eq_new).cumprod()
ax1.plot(cum_csn.index, cum_csn.values, label='CSN-HMC', linewidth=2)
ax1.plot(cum_mv.index, cum_mv.values, label='MV', linewidth=2)
ax1.plot(cum_eq.index, cum_eq.values, label='EW', linewidth=2)
ax1.set_title('Cumulative Return - Fixed Code')
ax1.set_ylabel('Net Value')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = axes[0, 1]
x = range(len(csn_changes))
width = 0.25
ax2.bar([i-width for i in x], np.array(csn_changes)*100, width, label='CSN-HMC', alpha=0.7)
ax2.bar(list(x), np.array(mv_changes)*100, width, label='MV', alpha=0.7)
ax2.bar([i+width for i in x], np.array(eq_changes)*100, width, label='EW', alpha=0.7)
ax2.set_title('Weight Change at Each Rebalance (%)')
ax2.set_ylabel('Weight Change (%)')
ax2.legend()
ax2.grid(alpha=0.3)

ax3 = axes[1, 0]
tc_csn_days = tc_csn_new[tc_csn_new > 0]
tc_mv_days = tc_mv_new[tc_mv_new > 0]
ax3.bar(range(len(tc_csn_days)), tc_csn_days.values * 100, alpha=0.7, label='CSN-HMC')
ax3.bar([i+0.4 for i in range(len(tc_mv_days))], tc_mv_days.values * 100, alpha=0.7, label='MV')
ax3.set_title('Transaction Cost per Rebalance Day (%)')
ax3.set_ylabel('Cost (%)')
ax3.legend()
ax3.grid(alpha=0.3)

ax4 = axes[1, 1]
metrics = ['Annual Ret\n(%)', 'Sharpe', 'Max DD\n(abs)']
csn_vals = [results_new['CSN-HMC']['ret']*100, results_new['CSN-HMC']['sharpe'], abs(results_new['CSN-HMC']['dd'])*100]
mv_vals = [results_new['MV']['ret']*100, results_new['MV']['sharpe'], abs(results_new['MV']['dd'])*100]
eq_vals = [results_new['EW']['ret']*100, results_new['EW']['sharpe'], abs(results_new['EW']['dd'])*100]
x = np.arange(len(metrics))
width = 0.25
ax4.bar(x - width, csn_vals, width, label='CSN-HMC', alpha=0.7)
ax4.bar(x, mv_vals, width, label='MV', alpha=0.7)
ax4.bar(x + width, eq_vals, width, label='EW', alpha=0.7)
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.legend()
ax4.set_title('Strategy Comparison (Fixed Code)')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/verify_fix.png', dpi=150, bbox_inches='tight')
print(f"\n验证图表已保存: /workspace/verify_fix.png")
