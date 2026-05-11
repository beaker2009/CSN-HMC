#!/usr/bin/env python3
"""
快速运行脚本 - 使用较短时间范围展示完整结果
"""
import warnings
import os
warnings.filterwarnings("ignore")
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
from sklearn.covariance import LedoitWolf
import sys

sys.path.insert(0, '/workspace')
from v1_0_mod import get_market_data, get_index_constituents, get_stock_name_mapping

# 配置
@dataclass
class Config:
    data_mode: str = "SH50"
    n_assets: int = 50
    start_date: str = "2022-01-01"  # 缩短到2年
    end_date: str = "2024-12-31"
    random_seed: int = 42
    train_window_months: int = 3  # 缩短训练窗口
    rebalance_freq: str = "ME"
    transaction_cost: float = 0.003
    risk_aversion: float = 0.4
    risk_free_annual: float = 0.025
    weight_prior_alpha: float = 1.0
    weight_smoothing: float = 0.1
    max_single_weight: float = 0.35
    min_single_weight: float = 0.001
    kappa0: float = 0.005
    nu0_offset: int = 2
    lambda0_diag: bool = True
    use_ledoit_wolf: bool = True
    objective_scale: float = 252.0
    n_samples: int = 200  # 减少采样
    n_burnin: int = 100   # 减少预热
    n_chains: int = 2
    target_accept: float = 0.40
    initial_step_size: float = 0.8
    leapfrog_steps: int = 3
    step_size_min: float = 0.05
    step_size_max: float = 5.0
    newton_tol: float = 1e-8
    newton_max_iter: int = 20
    dual_averaging_gamma: float = 0.15
    use_chinese_font: bool = True
    show_plots: bool = True

# 导入核心函数
from v1_0_mod import (
    bayesian_posterior, posterior_predictive_moments,
    simplex_to_sphere, sphere_to_simplex, tangent_projection,
    sphere_exponential_map, potential_energy, riemannian_gradient,
    hessian_matrix, make_positive_definite, robust_mass_matrix_and_sample,
    effective_sample_size, gelman_rubin, concentration_index, portfolio_stats,
    csn_hmc_sampler, equal_weight_portfolio, mean_variance_portfolio,
    rolling_backtest, summarize_results, annual_turnover_rate,
    print_config, print_tables, plot_results
)

def main():
    cfg = Config()
    print("=" * 80)
    print("CSN-HMC 投资组合优化器 - 快速运行模式")
    print("=" * 80)
    print_config(cfg)
    
    np.random.seed(cfg.random_seed)
    
    print("\n===== 1. 数据获取 =====")
    try:
        index_df = get_index_constituents(cfg.data_mode)
        cfg.n_assets = len(index_df)
        returns_df, name_map = get_market_data(index_df, cfg.start_date, cfg.end_date)
    except Exception as e:
        print(f"\n数据获取失败：{e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n===== 2. 滚动回测执行 =====")
    try:
        diag_df, dailies, curves, weight_df = rolling_backtest(returns_df, cfg)
    except Exception as e:
        print(f"\n回测执行失败：{e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n===== 3. 回测结果汇总 =====")
    summary = summarize_results(dailies, curves, cfg)
    summary.loc["CSN-HMC", "年化换手率(%)"] = annual_turnover_rate(weight_df[[c for c in weight_df.columns if c.startswith("CSN_")]])
    summary.loc["MV", "年化换手率(%)"] = annual_turnover_rate(weight_df[[c for c in weight_df.columns if c.startswith("MV_")]])
    summary.loc["EW", "年化换手率(%)"] = annual_turnover_rate(weight_df[[c for c in weight_df.columns if c.startswith("EW_")]])
    
    print_tables(summary, diag_df, weight_df, name_map, cfg)
    plot_results(curves, diag_df, weight_df, name_map, cfg)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary.to_csv(f"csn_hmc_{cfg.data_mode}_summary_{timestamp}.csv", encoding="utf-8-sig")
    diag_df.to_csv(f"csn_hmc_{cfg.data_mode}_diagnostics_{timestamp}.csv", encoding="utf-8-sig")
    weight_df.to_csv(f"csn_hmc_{cfg.data_mode}_weights_{timestamp}.csv", encoding="utf-8-sig")
    print(f"\n结果已导出（带时间戳）：")
    print(f"  - csn_hmc_{cfg.data_mode}_summary_{timestamp}.csv")
    print(f"  - csn_hmc_{cfg.data_mode}_diagnostics_{timestamp}.csv")
    print(f"  - csn_hmc_{cfg.data_mode}_weights_{timestamp}.csv")

if __name__ == "__main__":
    main()
