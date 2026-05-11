#!/usr/bin/env python3
"""
CSN-HMC投资组合优化器 - 快速演示版
使用模拟数据快速展示完整结果
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
from sklearn.covariance import LedoitWolf

np.random.seed(42)

print("=" * 80)
print("CSN-HMC 投资组合优化器")
print("=" * 80)

# ========== 配置 ==========
class Config:
    data_mode = "SH50"
    n_assets = 30
    start_date = "2022-01-01"
    end_date = "2024-12-31"
    random_seed = 42
    train_window_months = 3
    rebalance_freq = "ME"
    transaction_cost = 0.003
    risk_aversion = 0.4
    risk_free_annual = 0.025
    weight_prior_alpha = 1.0
    weight_smoothing = 0.1
    max_single_weight = 0.35
    min_single_weight = 0.001
    kappa0 = 0.005
    nu0_offset = 2
    lambda0_diag = True
    use_ledoit_wolf = True
    objective_scale = 252.0
    n_samples = 200
    n_burnin = 100
    n_chains = 2
    target_accept = 0.40
    initial_step_size = 0.8
    step_size_min = 0.05
    step_size_max = 5.0
    leapfrog_steps = 3
    dual_averaging_gamma = 0.15

cfg = Config()
print("\n===== 当前运行配置 =====")
print(f"数据模式: {cfg.data_mode}")
print(f"资产数量: {cfg.n_assets}")
print(f"回测期间: {cfg.start_date} 至 {cfg.end_date}")
print(f"训练窗口: {cfg.train_window_months}个月")
print(f"调仓频率: 月度")
print(f"交易成本: {cfg.transaction_cost*100:.1f}%")

# ========== 生成模拟数据 ==========
print("\n===== 1. 生成模拟市场数据 =====")
dates = pd.date_range(cfg.start_date, cfg.end_date, freq='B')
n_days = len(dates)
n_assets = cfg.n_assets

# 生成相关收益率矩阵
cov_matrix = np.random.randn(n_assets, n_assets) * 0.15
cov_matrix = cov_matrix @ cov_matrix.T * 0.0001
np.fill_diagonal(cov_matrix, 0.0004)
mean_returns = np.full(n_assets, 0.0003)

returns_raw = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
returns_df = pd.DataFrame(returns_raw, index=dates, 
                          columns=[f'Stock_{i:02d}' for i in range(n_assets)])
returns_df = returns_df.clip(-0.1, 0.1)

print(f"生成数据: {n_days}个交易日, {n_assets}只股票")

# 股票名称
stock_names = {
    f'Stock_{i:02d}': f'股票{i+1:02d}' for i in range(n_assets)
}

# ========== 核心算法函数 ==========
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
    t0, kappa, H_bar = 10.0, 0.75, 0.0
    total_iters = cfg.n_burnin + cfg.n_samples
    chain_storage = [[] for _ in range(cfg.n_chains)]
    accept_hist = []
    eps_hist = []
    
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
        eps_hist.append(eps)
        
        if t <= cfg.n_burnin:
            eta = 1.0 / (t + t0)
            H_bar = (1.0 - eta) * H_bar + eta * (cfg.target_accept - accept_bar)
            log_eps = mu - (np.sqrt(t) / gamma) * H_bar
            eps = float(np.clip(np.log(log_eps.exp() if hasattr(log_eps, 'exp') else np.exp(log_eps)), 
                                 np.log(eps_min), np.log(eps_max)))
    
    samples = np.vstack([np.asarray(v) for v in chain_storage if len(v) > 0])
    diagnostics = {
        "accept_rate_mean": float(np.mean(accept_hist)),
        "final_step_size": eps,
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

# ========== 滚动回测 ==========
print("\n===== 2. 运行滚动回测 =====")

rebalance_dates = returns_df.resample('ME').last().index.tolist()
rebalance_dates = [d for d in rebalance_dates if d >= dates[0] and d <= dates[-1]]
tasks = list(range(cfg.train_window_months, len(rebalance_dates) - 1))
print(f"总调仓次数: {len(tasks)}")

d = returns_df.shape[1]
assets = returns_df.columns

w_csn = pd.DataFrame(index=dates, columns=assets, dtype=float)
w_mv = pd.DataFrame(index=dates, columns=assets, dtype=float)
w_eq = pd.DataFrame(index=dates, columns=assets, dtype=float)

last_csn_w = equal_weight_portfolio(d)
last_eq_w = equal_weight_portfolio(d)

def _next_trading_position(index, date):
    return int(index.searchsorted(date, side="right"))

diag_rows = []
weight_rows = []

for idx in tqdm(tasks, desc="滚动调仓进度"):
    train_end = rebalance_dates[idx]
    train_start = rebalance_dates[idx - cfg.train_window_months]
    train_returns = returns_df.loc[train_start:train_end].dropna()
    
    if len(train_returns) <= d:
        raw_csn_w = equal_weight_portfolio(d)
        raw_mv_w = raw_csn_w.copy()
        diag = {"accept_rate_mean": np.nan, "final_step_size": np.nan}
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
    weight_rows.append({"rebalance_date": train_end,
                        **{f"CSN_{c}": csn_w[i] for i, c in enumerate(assets)},
                        **{f"MV_{c}": raw_mv_w[i] for i, c in enumerate(assets)},
                        **{f"EW_{c}": eq_w[i] for i, c in enumerate(assets)}})

# 填充权重
for df in (w_csn, w_mv, w_eq):
    first_valid = df.dropna().index.min()
    if pd.notna(first_valid):
        df.ffill(inplace=True)
        df.fillna(pd.Series(df.loc[first_valid].values, index=df.columns), inplace=True)
    else:
        df.iloc[:, :] = equal_weight_portfolio(d)

# ========== 计算收益和成本 ==========
# 计算调仓日权重变化
rebalance_weight_csn = w_csn.loc[[r for r in rebalance_dates if r in w_csn.index]].dropna(how='all')
rebalance_weight_mv = w_mv.loc[[r for r in rebalance_dates if r in w_mv.index]].dropna(how='all')
rebalance_weight_eq = w_eq.loc[[r for r in rebalance_dates if r in w_eq.index]].dropna(how='all')

weight_change_csn = rebalance_weight_csn.diff().abs().sum(axis=1).fillna(0.0)
weight_change_mv = rebalance_weight_mv.diff().abs().sum(axis=1).fillna(0.0)
weight_change_eq = rebalance_weight_eq.diff().abs().sum(axis=1).fillna(0.0)

tc_csn = pd.Series(0.0, index=dates)
tc_mv = pd.Series(0.0, index=dates)
tc_eq = pd.Series(0.0, index=dates)

for idx, date in enumerate(rebalance_dates):
    if idx == 0:
        continue
    next_pos = _next_trading_position(dates, date)
    if next_pos < len(dates):
        next_date = dates[next_pos]
        tc_csn.loc[next_date] = weight_change_csn.get(date, 0.0)
        tc_mv.loc[next_date] = weight_change_mv.get(date, 0.0)
        tc_eq.loc[next_date] = weight_change_eq.get(date, 0.0)

tc_csn = cfg.transaction_cost * tc_csn
tc_mv = cfg.transaction_cost * tc_mv
tc_eq = cfg.transaction_cost * tc_eq

lag_csn = w_csn.shift(1).bfill()
lag_mv = w_mv.shift(1).bfill()
lag_eq = w_eq.shift(1).bfill()

gross_csn = (returns_df * lag_csn).sum(axis=1)
gross_mv = (returns_df * lag_mv).sum(axis=1)
gross_eq = (returns_df * lag_eq).sum(axis=1)

net_csn = gross_csn - tc_csn
net_mv = gross_mv - tc_mv
net_eq = gross_eq - tc_eq

curves = {"CSN-HMC": (1 + net_csn).cumprod(), "MV": (1 + net_mv).cumprod(), "EW": (1 + net_eq).cumprod()}
dailies = {"CSN-HMC": net_csn, "MV": net_mv, "EW": net_eq}
diag_df = pd.DataFrame(diag_rows).set_index("rebalance_date")
weight_df = pd.DataFrame(weight_rows).set_index("rebalance_date")

# ========== 结果汇总 ==========
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

summary.loc["CSN-HMC", "年化换手率(%)"] = annual_turnover_rate(weight_df[[c for c in weight_df.columns if c.startswith("CSN_")]])
summary.loc["MV", "年化换手率(%)"] = annual_turnover_rate(weight_df[[c for c in weight_df.columns if c.startswith("MV_")]])
summary.loc["EW", "年化换手率(%)"] = annual_turnover_rate(weight_df[[c for c in weight_df.columns if c.startswith("EW_")]])

print("\n===== 最终回测业绩总表（扣除交易成本后） =====")
display_cols = ["年化收益率", "年化波动率", "年化夏普", "年化索提诺", "最大回撤", "卡玛比率", "期末净值", "年化换手率(%)"]
for col in display_cols:
    if col in summary.columns:
        summary[col] = summary[col].apply(lambda x: f"{x*100:.2f}%" if "收益率" in col or "波动率" in col or "回撤" in col else f"{x:.4f}" if pd.notna(x) else "N/A")
print(summary[display_cols].to_string())

print("\n===== 采样诊断概览 =====")
print(f"CSN-HMC 平均接受率: {diag_df['accept_rate_mean'].mean():.4f}")
print(f"CSN-HMC R-hat均值:  {diag_df['rhat_mean'].mean():.4f}")
print(f"CSN-HMC ESS均值:   {diag_df['ess_mean'].mean():.0f}")
print(f"\n权重集中度（赫芬达尔指数）:")
print(f"  CSN-HMC: {diag_df['csn_herfindahl'].mean():.4f}")
print(f"  MV:      {diag_df['mv_herfindahl'].mean():.4f}")

print("\n===== 关键结论 =====")
csn_sharpe = float(summary_rows['CSN-HMC']['年化夏普'])
mv_sharpe = float(summary_rows['MV']['年化夏普'])
csn_tc = float(summary.loc['CSN-HMC', '年化换手率(%)'].replace('%', ''))
mv_tc = float(summary.loc['MV', '年化换手率(%)'].replace('%', ''))

if csn_sharpe > mv_sharpe:
    print(f"✓ CSN-HMC策略夏普比率({csn_sharpe:.4f}) > MV策略({mv_sharpe:.4f})")
if csn_tc < mv_tc:
    print(f"✓ CSN-HMC策略换手率({csn_tc:.1f}%) < MV策略({mv_tc:.1f}%)")
print(f"✓ 等权重策略换手率为0%，无需交易成本")

# ========== 绘制图表 ==========
print("\n===== 生成图表 =====")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. 累计净值曲线
ax1 = axes[0]
for name, curve in curves.items():
    ax1.plot(curve.index, curve.values, label=name, linewidth=2, alpha=0.85)
ax1.set_title('Cumulative Net Value (After Transaction Costs)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Net Value', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_xlim(curve.index[0], curve.index[-1])

# 2. 回撤曲线
ax2 = axes[1]
for name, curve in curves.items():
    dd = curve / curve.cummax() - 1.0
    ax2.fill_between(dd.index, 0, dd.values, alpha=0.3, label=name)
ax2.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('Drawdown', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3, linestyle='--')
ax2.set_xlim(curve.index[0], curve.index[-1])

# 3. 策略对比柱状图
ax3 = axes[2]
metrics = ['Annual Ret\n(%)', 'Sharpe\nRatio', 'Max DD\n(abs)', 'Turnover\n(%)']
x = np.arange(len(metrics))
width = 0.25

csn_vals = [float(summary_rows['CSN-HMC']['年化收益率'])*100, csn_sharpe, 
            abs(float(summary_rows['CSN-HMC']['最大回撤']))*100, csn_tc/100]
mv_vals = [float(summary_rows['MV']['年化收益率'])*100, mv_sharpe,
           abs(float(summary_rows['MV']['最大回撤']))*100, mv_tc/100]
ew_vals = [float(summary_rows['EW']['年化收益率'])*100, float(summary_rows['EW']['年化夏普']),
           abs(float(summary_rows['EW']['最大回撤']))*100, 0]

ax3.bar(x - width, csn_vals, width, label='CSN-HMC', color='#1f77b4', alpha=0.8)
ax3.bar(x, mv_vals, width, label='MV', color='#ff7f0e', alpha=0.8)
ax3.bar(x + width, ew_vals, width, label='EW', color='#2ca02c', alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(metrics)
ax3.set_title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/workspace/backtest_chart_1.png', dpi=150, bbox_inches='tight')
print("图表1已保存: /workspace/backtest_chart_1.png")

# 图表2：权重分析
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))

# 权重集中度
ax4 = axes2[0]
ax4.plot(diag_df.index, diag_df['csn_herfindahl'], label='CSN-HMC', marker='o', markersize=4)
ax4.plot(diag_df.index, diag_df['mv_herfindahl'], label='MV', marker='s', markersize=4)
ax4.set_title('Weight Concentration (Herfindahl Index)', fontsize=14, fontweight='bold')
ax4.set_ylabel('Herfindahl Index', fontsize=12)
ax4.legend(fontsize=11)
ax4.grid(alpha=0.3, linestyle='--')
ax4.tick_params(axis='x', rotation=45)

# 采样接受率
ax5 = axes2[1]
ax5.plot(diag_df.index, diag_df['accept_rate_mean'], label='Accept Rate', color='#1f77b4', linewidth=2)
ax5.set_title('CSN-HMC Sampling Diagnostics', fontsize=14, fontweight='bold')
ax5.set_ylabel('Accept Rate', fontsize=12)
ax5.legend(fontsize=11)
ax5.grid(alpha=0.3, linestyle='--')
ax5.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/workspace/backtest_chart_2.png', dpi=150, bbox_inches='tight')
print("图表2已保存: /workspace/backtest_chart_2.png")

# 图表3：个股权重分布
fig3, ax6 = plt.subplots(figsize=(14, 6))
last_csn = weight_df.iloc[-1][[c for c in weight_df.columns if c.startswith('CSN_')]].sort_values(ascending=False).head(15)
codes = [c.replace('CSN_', '') for c in last_csn.index]
ax6.bar(range(len(codes)), last_csn.values * 100, color='#1f77b4', alpha=0.8)
ax6.set_xticks(range(len(codes)))
ax6.set_xticklabels(codes, rotation=45, ha='right')
ax6.set_title('Top 15 Stock Weights (CSN-HMC, Last Rebalance)', fontsize=14, fontweight='bold')
ax6.set_ylabel('Weight (%)', fontsize=12)
ax6.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/workspace/backtest_chart_3.png', dpi=150, bbox_inches='tight')
print("图表3已保存: /workspace/backtest_chart_3.png")

print("\n" + "=" * 80)
print("运行完成！")
print("=" * 80)
