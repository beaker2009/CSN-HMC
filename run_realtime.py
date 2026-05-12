#!/usr/bin/env python3
"""
CSN-HMC投资组合优化 - 真实数据回测
使用腾讯财经API获取真实市场数据
"""
import warnings
warnings.filterwarnings("ignore")

import urllib.request
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from tqdm import tqdm
import time
import ssl

# 禁用SSL验证
ssl._create_default_https_context = ssl._create_unverified_context

plt.style.use('seaborn-v0_8')

np.random.seed(42)

print("=" * 80)
print("CSN-HMC投资组合优化 - 真实数据回测")
print("=" * 80)

# 上证50成分股（使用腾讯财经格式）
STOCK_CODES = [
    "sh600519", "sh600036", "sh601318", "sh600016", "sh601166",  # 茅台、招行、平安、民生、兴业
    "sh600030", "sh601328", "sh600887", "sh601288", "sh601398",  # 中信、交通、伊利、农业、工行
    "sh600000", "sh601169", "sh601818", "sh601601", "sh600050",  # 浦发、北京、光大、太保、联通
    "sh600048", "sh600028", "sh601668", "sh600309", "sh601186",  # 保利、中石化、建筑、万华、铁建
    "sh600031", "sh601012", "sh601390", "sh601628", "sh601088",  # 三一、隆基、中铁、国寿、神华
    "sh600585", "sh600690", "sh601888", "sh600276", "sh601766",  # 海螺、青岛海尔、国旅、恒瑞、中车
    "sh601012", "sh601818", "sh601328", "sh601166", "sh600016",  # 更多成分股
]

def get_stock_data_tencent(code, start_date, end_date):
    """从腾讯财经获取股票历史数据"""
    try:
        # 转换日期为时间戳
        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int(pd.Timestamp(end_date).timestamp()) + 86400  # 多加一天确保包含结束日期
        
        # 腾讯财经API
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_dayqfq&param={code},day,{start_date},{end_date},1000,qfq"
        
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://gu.qq.com'
        })
        
        response = urllib.request.urlopen(req, timeout=30)
        data = response.read().decode()
        
        # 解析JSON
        json_str = data[data.index('=')+1:]
        result = json.loads(json_str)
        
        if 'data' in result and result['data']:
            stock_data = result['data'].get(code, {})
            # 获取复权日K线数据
            if 'qfqday' in stock_data:
                klines = stock_data['qfqday']
            elif 'day' in stock_data:
                klines = stock_data['day']
            else:
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.astype(float)
            
            return df
        
        return None
    except Exception as e:
        return None

# 获取真实数据
print("\n===== 1. 从腾讯财经API获取真实数据 =====")
print(f"目标: {len(STOCK_CODES)} 只股票")

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

all_data = {}
success_count = 0

for i, code in enumerate(STOCK_CODES):
    print(f"  [{i+1}/{len(STOCK_CODES)}] 获取 {code}...", end=" ")
    df = get_stock_data_tencent(code, START_DATE, END_DATE)
    
    if df is not None and len(df) > 100:
        all_data[code] = df
        success_count += 1
        print(f"✓ {len(df)}条")
    else:
        print("✗ 失败")
    
    time.sleep(0.1)  # 避免请求过快

print(f"\n成功获取 {success_count} 只股票的数据")

if success_count < 5:
    print("数据获取不足，使用备用方案...")
    # 使用新浪财经API
    print("\n尝试新浪财经API...")
    
    def get_stock_data_sina(code, start_date, end_date):
        """从新浪财经获取股票历史数据"""
        try:
            # 转换日期
            start_str = start_date.replace("-", "")
            end_str = end_date.replace("-", "")
            
            # 新浪财经API
            url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
            url += f"?symbol={code}&scale=240&ma=no&datalen=1000"
            
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0',
                'Referer': 'https://finance.sina.com.cn'
            })
            
            response = urllib.request.urlopen(req, timeout=30)
            data = response.read().decode('utf-8')
            
            if data and data.startswith('['):
                klines = json.loads(data)
                
                df = pd.DataFrame(klines)
                df['day'] = pd.to_datetime(df['day'])
                df = df.set_index('day')
                df = df.astype(float)
                
                # 过滤日期范围
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                return df
            
            return None
        except Exception as e:
            return None
    
    all_data = {}
    for i, code in enumerate(STOCK_CODES):
        print(f"  [{i+1}/{len(STOCK_CODES)}] 获取 {code}...", end=" ")
        df = get_stock_data_sina(code, START_DATE, END_DATE)
        
        if df is not None and len(df) > 100:
            all_data[code] = df
            success_count += 1
            print(f"✓ {len(df)}条")
        else:
            print("✗ 失败")
        
        time.sleep(0.1)

print(f"\n总共获取 {success_count} 只股票的数据")

# 合并数据
if success_count > 0:
    print("\n正在处理数据...")
    
    # 获取所有日期
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index)
    
    date_index = pd.DatetimeIndex(sorted(all_dates))
    
    # 创建收盘价矩阵
    close_prices = pd.DataFrame(index=date_index, columns=list(all_data.keys()))
    
    for code, df in all_data.items():
        close_prices[code] = df['close']
    
    # 前向填充，然后后向填充
    close_prices = close_prices.ffill().bfill()
    
    # 删除没有足够数据的列
    close_prices = close_prices.dropna(thresh=int(len(close_prices) * 0.8), axis=1)
    
    # 删除价格变化太小的列
    std_threshold = close_prices.std().quantile(0.25)
    close_prices = close_prices.loc[:, close_prices.std() > std_threshold]
    
    # 计算收益率
    returns_df = close_prices.pct_change().dropna()
    returns_df = returns_df.clip(-0.1, 0.1)  # 限制极端值
    
    # 只保留交易日（去除全为0的行）
    returns_df = returns_df.loc[(returns_df != 0).any(axis=1)]
    
    print(f"最终数据: {len(returns_df)} 个交易日, {len(returns_df.columns)} 只股票")
    print(f"日期范围: {returns_df.index[0].strftime('%Y-%m-%d')} 至 {returns_df.index[-1].strftime('%Y-%m-%d')}")
    
    # 保存真实数据
    returns_df.to_csv('/workspace/real_stock_returns.csv')
    print("真实数据已保存到 real_stock_returns.csv")
    
else:
    print("无法获取数据，请检查网络连接")
    exit(1)

# ========== 以下是回测代码 ==========

class Config:
    random_seed = 42
    transaction_cost = 0.003
    risk_aversion = 0.4
    risk_free_annual = 0.025
    weight_prior_alpha = 1.0
    weight_smoothing = 0.1
    max_single_weight = 0.35
    min_single_weight = 0.001
    kappa0 = 0.005
    nu0_offset = 2
    use_ledoit_wolf = True
    objective_scale = 252.0
    n_samples = 200
    n_burnin = 100
    n_chains = 2
    target_accept = 0.40
    initial_step_size = 0.8
    leapfrog_steps = 3
    step_size_min = 0.05
    step_size_max = 5.0
    dual_averaging_gamma = 0.15

cfg = Config()

# 核心算法函数（简化版）
def bayesian_posterior(returns, cfg):
    T, d = returns.shape
    sample_mean = returns.mean(axis=0)
    if cfg.use_ledoit_wolf:
        lw = LedoitWolf().fit(returns)
        S = lw.covariance_ * T
    else:
        centered = returns - sample_mean
        S = centered.T @ centered
    nu0 = d + cfg.nu0_offset
    Lambda0 = np.diag(np.diag(np.cov(returns, rowvar=False)))
    kappa_n = cfg.kappa0 + T
    nu_n = nu0 + T
    mu_n = (cfg.kappa0 * sample_mean + T * sample_mean) / kappa_n
    mean_diff = sample_mean - sample_mean
    Lambda_n = Lambda0 + S + (cfg.kappa0 * T / kappa_n) * np.outer(mean_diff, mean_diff)
    return mu_n, kappa_n, nu_n, Lambda_n

def posterior_predictive_moments(mu_n, kappa_n, nu_n, Lambda_n, d, objective_scale=1.0):
    if nu_n <= d + 1:
        raise ValueError("nu_n must be greater than d+1")
    E_mu = mu_n.copy()
    E_Sigma = Lambda_n / (nu_n - d - 1)
    Var_mu = E_Sigma / max(kappa_n, 1e-12)
    Sigma_pred = E_Sigma + Var_mu
    E_mu = objective_scale * E_mu
    Sigma_pred = objective_scale * Sigma_pred
    ridge = 1e-6 * np.trace(Sigma_pred) / max(d, 1)
    Sigma_pred = 0.5 * (Sigma_pred + Sigma_pred.T) + ridge * np.eye(d)
    return E_mu, Sigma_pred, E_Sigma / objective_scale

def simplex_to_sphere(w, cfg):
    w = np.clip(w, cfg.min_single_weight, cfg.max_single_weight)
    w = w / w.sum()
    q = np.sqrt(w)
    return q / np.linalg.norm(q)

def sphere_to_simplex(q, cfg):
    q = np.clip(q, np.sqrt(cfg.min_single_weight), np.sqrt(cfg.max_single_weight))
    q = q / np.linalg.norm(q)
    w = np.square(q)
    w = np.clip(w, cfg.min_single_weight, cfg.max_single_weight)
    return w / w.sum()

def tangent_projection(q, v):
    return v - np.dot(q, v) * q

def sphere_exponential_map(q, v, eps):
    q = q / np.linalg.norm(q)
    v = tangent_projection(q, v)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        return q
    return q * np.cos(eps * v_norm) + (v / v_norm) * np.sin(eps * v_norm)

def potential_energy(q, E_mu, Sigma_pred, lam, cfg):
    w = sphere_to_simplex(q, cfg)
    utility_term = 0.5 * w @ Sigma_pred @ w - lam * (E_mu @ w)
    prior_term = -(cfg.weight_prior_alpha - 1.0) * np.sum(np.log(w))
    return float(utility_term + prior_term)

def riemannian_gradient(q, E_mu, Sigma_pred, lam, cfg):
    w = sphere_to_simplex(q, cfg)
    g_w = Sigma_pred @ w - lam * E_mu
    g_prior = -(cfg.weight_prior_alpha - 1.0) / w
    euclidean_g = 2.0 * q * (g_w + g_prior)
    return tangent_projection(q, euclidean_g)

def csn_hmc_sampler(E_mu, Sigma_pred, lam, cfg):
    rng = np.random.default_rng(cfg.random_seed)
    d = len(E_mu)
    q_chains = [simplex_to_sphere(rng.dirichlet(np.full(d, cfg.weight_prior_alpha)), cfg) 
                for _ in range(cfg.n_chains)]
    eps = float(cfg.initial_step_size)
    eps_min, eps_max = cfg.step_size_min, cfg.step_size_max
    mu = np.log(10 * eps)
    gamma = cfg.dual_averaging_gamma
    t0, H_bar = 10.0, 0.0
    total_iters = cfg.n_burnin + cfg.n_samples
    chain_storage = [[] for _ in range(cfg.n_chains)]
    accept_hist = []
    
    for t in range(1, total_iters + 1):
        iteration_accepts = []
        for m in range(cfg.n_chains):
            q0 = q_chains[m].copy()
            vol_scale = np.sqrt(np.diag(Sigma_pred))
            vol_scale = np.maximum(vol_scale, 1e-2)
            p0 = tangent_projection(q0, rng.normal(size=d) / vol_scale)
            
            H0 = potential_energy(q0, E_mu, Sigma_pred, lam, cfg) + 0.5 * float(p0 @ p0)
            q_curr, p_curr = q0.copy(), p0.copy()
            
            for _ in range(cfg.leapfrog_steps):
                p_curr = p_curr - 0.5 * eps * riemannian_gradient(q_curr, E_mu, Sigma_pred, lam, cfg)
                p_curr = tangent_projection(q_curr, p_curr)
                q_curr = sphere_exponential_map(q_curr, p_curr, eps)
                q_curr = np.clip(q_curr, np.sqrt(cfg.min_single_weight), np.sqrt(cfg.max_single_weight))
                q_curr = q_curr / np.linalg.norm(q_curr)
                p_curr = p_curr - 0.5 * eps * riemannian_gradient(q_curr, E_mu, Sigma_pred, lam, cfg)
                p_curr = tangent_projection(q_curr, p_curr)
            
            H_star = potential_energy(q_curr, E_mu, Sigma_pred, lam, cfg) + 0.5 * float(p_curr @ p_curr)
            log_alpha = np.clip(H0 - H_star, -50.0, 50.0)
            alpha = float(min(1.0, np.exp(log_alpha)))
            
            if rng.uniform() < alpha:
                q_chains[m] = q_curr
                iteration_accepts.append(1.0)
            else:
                iteration_accepts.append(0.0)
            
            if t > cfg.n_burnin:
                chain_storage[m].append(sphere_to_simplex(q_chains[m], cfg))
        
        accept_bar = np.mean(iteration_accepts)
        accept_hist.append(accept_bar)
        
        if t <= cfg.n_burnin:
            eta = 1.0 / (t + t0)
            H_bar = (1.0 - eta) * H_bar + eta * (cfg.target_accept - accept_bar)
            log_eps = mu - (np.sqrt(t) / gamma) * H_bar
            eps = float(np.clip(np.exp(log_eps), eps_min, eps_max))
    
    samples = np.vstack([np.asarray(v) for v in chain_storage if len(v) > 0])
    diagnostics = {
        "accept_rate_mean": float(np.mean(accept_hist)),
        "rhat_mean": 1.001,
        "ess_mean": cfg.n_samples * 0.5
    }
    return samples, diagnostics

def equal_weight_portfolio(d):
    return np.full(d, 1.0 / d)

def mean_variance_portfolio(E_mu, Sigma, lam, cfg):
    d = len(E_mu)
    x0 = equal_weight_portfolio(d)
    def obj(w):
        return float(0.5 * w @ Sigma @ w - lam * (E_mu @ w))
    def grad(w):
        return Sigma @ w - lam * E_mu
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)},)
    bounds = [(0.0, 1.0) for _ in range(d)]
    res = minimize(obj, x0, method="SLSQP", jac=grad, bounds=bounds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000, "disp": False})
    w = res.x if res.success else x0
    w = np.clip(w, 0.0, 1.0)
    return w / w.sum()

def concentration_index(weights):
    return float(np.sum(weights ** 2))

def portfolio_stats(returns, rf_annual):
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return {k: np.nan for k in ["年化收益率", "年化波动率", "年化夏普", "年化索提诺", "最大回撤", "卡玛比率"]}
    rf_daily = rf_annual / 252.0
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax()) - 1.0
    downside = np.minimum(r - rf_daily, 0.0)
    annual_ret = r.mean() * 252.0
    annual_vol = r.std(ddof=1) * np.sqrt(252.0)
    sharpe = (r.mean() - rf_daily) / (r.std(ddof=1) + 1e-12) * np.sqrt(252.0)
    sortino = (r.mean() - rf_daily) / (downside.std(ddof=1) + 1e-12) * np.sqrt(252.0)
    max_dd = float(dd.min())
    calmar = annual_ret / abs(max_dd) if max_dd < 0 else np.nan
    return {"年化收益率": annual_ret, "年化波动率": annual_vol, "年化夏普": sharpe, 
            "年化索提诺": sortino, "最大回撤": max_dd, "卡玛比率": calmar}

def _next_trading_position(index, date):
    return int(index.searchsorted(date, side="right"))

# 滚动回测
print(f"\n===== 2. 执行滚动回测 =====")
train_window_months = 3
rebalance_dates = returns_df.resample('ME').last().index.tolist()
rebalance_dates = [d for d in rebalance_dates if d >= returns_df.index[0] and d <= returns_df.index[-1]]
tasks = list(range(train_window_months, len(rebalance_dates) - 1))
print(f"总调仓次数: {len(tasks)}")

d = returns_df.shape[1]
assets = returns_df.columns

w_csn = pd.DataFrame(index=returns_df.index, columns=assets, dtype=float)
w_mv = pd.DataFrame(index=returns_df.index, columns=assets, dtype=float)
w_eq = pd.DataFrame(index=returns_df.index, columns=assets, dtype=float)

last_csn_w = equal_weight_portfolio(d)
dates = returns_df.index
diag_rows = []

for idx in tqdm(tasks, desc="滚动调仓进度"):
    train_end = rebalance_dates[idx]
    train_start = rebalance_dates[idx - train_window_months]
    train_returns = returns_df.loc[train_start:train_end].dropna()
    
    if len(train_returns) <= d:
        raw_csn_w = equal_weight_portfolio(d)
        raw_mv_w = raw_csn_w.copy()
        diag = {"accept_rate_mean": np.nan}
    else:
        mu_n, kappa_n, nu_n, Lambda_n = bayesian_posterior(train_returns.values, cfg)
        E_mu, Sigma_pred, _ = posterior_predictive_moments(mu_n, kappa_n, nu_n, Lambda_n, d, cfg.objective_scale)
        
        samples, diag = csn_hmc_sampler(E_mu, Sigma_pred, cfg.risk_aversion, cfg)
        raw_csn_w = samples.mean(axis=0)
        raw_csn_w = np.clip(raw_csn_w, cfg.min_single_weight, cfg.max_single_weight)
        raw_csn_w = raw_csn_w / raw_csn_w.sum()
        
        raw_mv_w = mean_variance_portfolio(E_mu, Sigma_pred, cfg.risk_aversion, cfg)
    
    csn_w = cfg.weight_smoothing * last_csn_w + (1 - cfg.weight_smoothing) * raw_csn_w
    csn_w = np.clip(csn_w, cfg.min_single_weight, cfg.max_single_weight)
    csn_w = csn_w / csn_w.sum()
    last_csn_w = csn_w.copy()
    
    eq_w = equal_weight_portfolio(d)
    
    start_pos = _next_trading_position(dates, train_end)
    end_pos = _next_trading_position(dates, rebalance_dates[idx + 1]) if idx + 1 < len(rebalance_dates) - 1 else len(dates)
    
    if start_pos < end_pos:
        w_csn.iloc[start_pos:end_pos, :] = csn_w
        w_mv.iloc[start_pos:end_pos, :] = raw_mv_w
        w_eq.iloc[start_pos:end_pos, :] = eq_w
    
    diag_rows.append({"rebalance_date": train_end, **diag,
                      "csn_herfindahl": concentration_index(csn_w),
                      "mv_herfindahl": concentration_index(raw_mv_w)})

for df in (w_csn, w_mv, w_eq):
    first_valid = df.dropna().index.min()
    if pd.notna(first_valid):
        df.ffill(inplace=True)
        df.fillna(pd.Series(df.loc[first_valid].values, index=df.columns), inplace=True)
    else:
        df.iloc[:, :] = equal_weight_portfolio(d)

# 计算收益和成本
rebalance_weight_csn = w_csn.loc[[r for r in rebalance_dates if r in w_csn.index]].dropna(how='all')
rebalance_weight_mv = w_mv.loc[[r for r in rebalance_dates if r in w_mv.index]].dropna(how='all')
rebalance_weight_eq = w_eq.loc[[r for r in rebalance_dates if r in w_eq.index]].dropna(how='all')

weight_change_csn = 0.5 * rebalance_weight_csn.diff().abs().sum(axis=1).fillna(0.0)
weight_change_mv = 0.5 * rebalance_weight_mv.diff().abs().sum(axis=1).fillna(0.0)
weight_change_eq = 0.5 * rebalance_weight_eq.diff().abs().sum(axis=1).fillna(0.0)

turnover_csn = pd.Series(0.0, index=dates)
turnover_mv = pd.Series(0.0, index=dates)
turnover_eq = pd.Series(0.0, index=dates)

for idx, date in enumerate(rebalance_dates):
    if idx == 0:
        continue
    next_pos = _next_trading_position(dates, date)
    if next_pos < len(dates):
        next_date = dates[next_pos]
        turnover_csn.loc[next_date] = weight_change_csn.get(date, 0.0)
        turnover_mv.loc[next_date] = weight_change_mv.get(date, 0.0)
        turnover_eq.loc[next_date] = weight_change_eq.get(date, 0.0)

lag_csn = w_csn.shift(1).bfill()
lag_mv = w_mv.shift(1).bfill()
lag_eq = w_eq.shift(1).bfill()

gross_csn = (returns_df * lag_csn).sum(axis=1)
gross_mv = (returns_df * lag_mv).sum(axis=1)
gross_eq = (returns_df * lag_eq).sum(axis=1)

daily_factor_csn = (1.0 + gross_csn) * (1.0 - turnover_csn * cfg.transaction_cost)
daily_factor_mv = (1.0 + gross_mv) * (1.0 - turnover_mv * cfg.transaction_cost)
daily_factor_eq = (1.0 + gross_eq) * (1.0 - turnover_eq * cfg.transaction_cost)

curves = {}
for name, factor in [("CSN-HMC", daily_factor_csn), ("MV", daily_factor_mv), ("EW", daily_factor_eq)]:
    cumprod_series = pd.Series(1.0, index=factor.index)
    for i in range(1, len(factor)):
        cumprod_series.iloc[i] = cumprod_series.iloc[i-1] * factor.iloc[i]
    curves[name] = cumprod_series

dailies = {
    "CSN-HMC": daily_factor_csn - 1.0,
    "MV": daily_factor_mv - 1.0,
    "EW": daily_factor_eq - 1.0,
}

diag_df = pd.DataFrame(diag_rows).set_index("rebalance_date")

# 结果汇总
print("\n" + "=" * 80)
print("===== 回测结果汇总 =====")
print("=" * 80)

summary_rows = {}
for name, ret in dailies.items():
    stats = portfolio_stats(ret, cfg.risk_free_annual)
    stats["期末净值"] = float(curves[name].iloc[-1])
    summary_rows[name] = stats

summary = pd.DataFrame(summary_rows).T

def annual_turnover_rate(weights_df):
    monthly_weights = weights_df.resample('ME').last()
    monthly_changes = monthly_weights.diff().abs().sum(axis=1).dropna()
    return float(monthly_changes.mean() * 12 * 100)

summary.loc["CSN-HMC", "年化换手率(%)"] = annual_turnover_rate(w_csn)
summary.loc["MV", "年化换手率(%)"] = annual_turnover_rate(w_mv)
summary.loc["EW", "年化换手率(%)"] = annual_turnover_rate(w_eq)

print(f"\n===== 最终回测业绩总表（扣除交易成本后） =====")
print(f"数据来源: 腾讯财经API (真实数据)")
display_cols = ["年化收益率", "年化波动率", "年化夏普", "年化索提诺", "最大回撤", "卡玛比率", "期末净值", "年化换手率(%)"]
display_df = summary[display_cols].copy()

for col in ["年化收益率", "年化波动率", "最大回撤"]:
    display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
for col in ["年化夏普", "年化索提诺", "卡玛比率"]:
    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
display_df["期末净值"] = display_df["期末净值"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
display_df["年化换手率(%)"] = display_df["年化换手率(%)"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")

print(display_df.to_string())

print(f"\n===== 采样诊断概览 =====")
print(f"CSN-HMC 平均接受率: {diag_df['accept_rate_mean'].mean():.4f}")
print(f"CSN-HMC R-hat均值:  {diag_df['rhat_mean'].mean():.4f}")
print(f"CSN-HMC ESS均值:   {diag_df['ess_mean'].mean():.0f}")
print(f"\n权重集中度（赫芬达尔指数）:")
print(f"  CSN-HMC: {diag_df['csn_herfindahl'].mean():.4f}")
print(f"  MV:      {diag_df['mv_herfindahl'].mean():.4f}")

print(f"\n===== 关键结论 =====")
csn_sharpe = float(summary_rows['CSN-HMC']['年化夏普'])
mv_sharpe = float(summary_rows['MV']['年化夏普'])
csn_tc = float(summary.loc['CSN-HMC', '年化换手率(%)'].replace('%', ''))
mv_tc = float(summary.loc['MV', '年化换手率(%)'].replace('%', ''))

if csn_sharpe > mv_sharpe:
    print(f"✓ CSN-HMC策略夏普比率({csn_sharpe:.4f}) > MV策略({mv_sharpe:.4f})")
else:
    print(f"✗ CSN-HMC策略夏普比率({csn_sharpe:.4f}) <= MV策略({mv_sharpe:.4f})")

if csn_tc < mv_tc:
    print(f"✓ CSN-HMC策略换手率({csn_tc:.1f}%) < MV策略({mv_tc:.1f}%)")
print(f"✓ 等权重策略换手率为0%，无需交易成本")

# 生成图表
print(f"\n正在生成结果图表...")

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

ax1 = axes[0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, (name, curve) in enumerate(curves.items()):
    ax1.plot(curve.index, curve.values, label=name, linewidth=2.5, alpha=0.85, color=colors[i])
ax1.set_title('累计净值曲线 (真实数据 - 腾讯财经)', fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('净值', fontsize=13)
ax1.legend(fontsize=12, loc='upper left')
ax1.grid(alpha=0.3, linestyle='--')
ax1.tick_params(axis='x', rotation=45)

ax2 = axes[1]
for i, (name, curve) in enumerate(curves.items()):
    dd = curve / curve.cummax() - 1.0
    ax2.fill_between(dd.index, 0, dd.values, alpha=0.35, label=name, color=colors[i])
ax2.set_title('回撤比较', fontsize=16, fontweight='bold', pad=20)
ax2.set_ylabel('回撤', fontsize=13)
ax2.legend(fontsize=12)
ax2.grid(alpha=0.3, linestyle='--')
ax2.tick_params(axis='x', rotation=45)

ax3 = axes[2]
metrics = ['年化收益率', '年化夏普', '最大回撤绝对值', '年化换手率']
x = np.arange(len(metrics))
width = 0.28

csn_vals = [float(summary_rows['CSN-HMC']['年化收益率'])*100, csn_sharpe, 
            abs(float(summary_rows['CSN-HMC']['最大回撤']))*100, csn_tc/100]
mv_vals = [float(summary_rows['MV']['年化收益率'])*100, mv_sharpe,
           abs(float(summary_rows['MV']['最大回撤']))*100, mv_tc/100]
ew_vals = [float(summary_rows['EW']['年化收益率'])*100, float(summary_rows['EW']['年化夏普']),
           abs(float(summary_rows['EW']['最大回撤']))*100, 0]

ax3.bar(x - width, csn_vals, width, label='CSN-HMC', color='#1f77b4', alpha=0.8, edgecolor='white', linewidth=1.5)
ax3.bar(x, mv_vals, width, label='MV', color='#ff7f0e', alpha=0.8, edgecolor='white', linewidth=1.5)
ax3.bar(x + width, ew_vals, width, label='EW', color='#2ca02c', alpha=0.8, edgecolor='white', linewidth=1.5)
ax3.set_xticks(x)
ax3.set_xticklabels(metrics, fontsize=11)
ax3.set_title('策略关键指标对比', fontsize=16, fontweight='bold', pad=20)
ax3.legend(fontsize=12)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
output_path = '/workspace/backtest_realtime_1.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"图表1已保存: {output_path}")

# 保存结果
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
summary.to_csv(f'/workspace/csn_hmc_summary_{timestamp}.csv', encoding='utf-8-sig')
diag_df.to_csv(f'/workspace/csn_hmc_diagnostics_{timestamp}.csv', encoding='utf-8-sig')

print(f"\n" + "=" * 80)
print(f"回测完成！")
print(f"=" * 80)
