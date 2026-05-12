from __future__ import annotations
import warnings
import os
warnings.filterwarnings("ignore")
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
from sklearn.covariance import LedoitWolf

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告: akshare未安装，将使用备用数据源")

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    BAOSTOCK_AVAILABLE = False
    print("警告: baostock未安装")

# ============================================================
# 全参数配置类
# ============================================================
@dataclass
class Config:
    # 数据
    data_mode: str = "SH50"  # 可选：SH50(上证50), CSI300(沪深300), CSI500(中证500)
    n_assets: int = 20
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    random_seed: int = 42

    # 回测参数
    train_window_months: int = 3
    rebalance_freq: str = "ME"
    transaction_cost: float = 0.003
    risk_aversion: float = 0.4
    risk_free_annual: float = 0.025
    weight_prior_alpha: float = 1.00
    weight_smoothing: float = 0.1
    max_single_weight: float = 0.35
    min_single_weight: float = 0.001

    # 贝叶斯先验
    kappa0: float = 0.005
    nu0_offset: int = 2
    lambda0_diag: bool = True
    use_ledoit_wolf: bool = True

    # 目标函数尺度
    objective_scale: float = 252.0

    # CSN-HMC 采样参数
    n_samples: int = 200
    n_burnin: int = 100
    n_chains: int = 2
    target_accept: float = 0.40
    initial_step_size: float = 0.8
    leapfrog_steps: int = 3
    step_size_min: float = 0.05
    step_size_max: float = 5.0
    newton_tol: float = 1e-8
    newton_max_iter: int = 20
    dual_averaging_gamma: float = 0.15

    # 输出
    use_chinese_font: bool = True
    show_plots: bool = True

# ============================================================
# 数据源支持（akshare/东方财富优先）
# ============================================================
INDEX_CONSTITUENTS = {
    "SH50": {  # 上证50
        "name": "上证50",
        "stock_pool": [  # 常用上证50成分股
            "600519", "600036", "601318", "600016", "601166",
            "600030", "601328", "600887", "601288", "601398",
            "600000", "601169", "601818", "601601", "600050",
            "600048", "600028", "601668", "600309", "601186",
            "600031", "601012", "601390", "601628", "600050",
            "600585", "600690", "601888", "600276", "601766",
            "601989", "600406", "600547", "600489", "601336",
            "601668", "601688", "600585", "601991", "601211",
            "600690", "601319", "601816", "601066", "601236",
            "601319", "601138", "603259", "600900", "600438",
        ]
    },
    "CSI300": {  # 沪深300
        "name": "沪深300",
        "stock_pool": [  # 常用沪深300成分股示例
            "600519", "600036", "601318", "600016", "601166",
            "600030", "601328", "600887", "601288", "601398",
            "600000", "601169", "601818", "601601", "600050",
            "600048", "600028", "601668", "600309", "601186",
            "600031", "601012", "601390", "601628", "601088",
            "600585", "600690", "601888", "600276", "601766",
        ]
    },
    "CSI500": {  # 中证500
        "name": "中证500",
        "stock_pool": [  # 常用中证500成分股示例
            "600582", "600487", "601225", "601666", "601666",
            "600170", "600329", "600352", "600409", "600486",
            "600588", "600703", "600588", "600521", "600588",
        ]
    }
}

STOCK_NAMES = {  # 常用股票名称映射
    "600519": "贵州茅台", "600036": "招商银行", "601318": "中国平安",
    "600016": "民生银行", "601166": "兴业银行", "600030": "中信证券",
    "601328": "交通银行", "600887": "伊利股份", "601288": "农业银行",
    "601398": "工商银行", "600000": "浦发银行", "601169": "北京银行",
    "601818": "光大银行", "601601": "中国太保", "600050": "中国联通",
    "600048": "保利发展", "600028": "中国石化", "601668": "中国建筑",
    "600309": "万华化学", "601186": "中国铁建", "600031": "三一重工",
    "601012": "隆基绿能", "601390": "中国中铁", "601628": "中国人寿",
    "601088": "中国神华", "600585": "海螺水泥", "600690": "海尔智家",
    "601888": "中国中免", "600276": "恒瑞医药", "601766": "中国中车",
    "601989": "中国重工", "600406": "国电南瑞", "600547": "山东黄金",
    "600489": "中金黄金", "601336": "新华保险", "601688": "华泰证券",
    "601991": "大唐发电", "601211": "国泰君安", "601319": "中国人保",
    "601816": "京沪高铁", "601066": "中信建投", "601236": "红塔证券",
    "601138": "工业富联", "603259": "药明康德", "600900": "长江电力",
    "600438": "通威股份", "601138": "工业富联", "601012": "隆基绿能",
    "600050": "中国联通", "600276": "恒瑞医药", "601012": "隆基绿能",
    "601857": "中国石油", "600028": "中国石化", "600050": "中国联通",
    "601668": "中国建筑", "601186": "中国铁建", "601390": "中国中铁",
    "601766": "中国中车", "601628": "中国人寿", "601088": "中国神华",
    "600031": "三一重工", "600309": "万华化学", "600585": "海螺水泥",
    "600887": "伊利股份", "600276": "恒瑞医药", "600519": "贵州茅台",
    "601318": "中国平安", "600036": "招商银行", "600016": "民生银行",
    "601166": "兴业银行", "600030": "中信证券", "601328": "交通银行",
    "601288": "农业银行", "601398": "工商银行", "601818": "光大银行",
}

def get_index_constituents(index_mode: str) -> pd.DataFrame:
    """获取指数成分股及名称（akshare东方财富数据源）"""
    mode_upper = index_mode.upper()
    
    if mode_upper not in INDEX_CONSTITUENTS:
        raise ValueError(f"不支持的指数模式：{index_mode}，请选择：SH50(上证50), CSI300(沪深300), CSI500(中证500)")
    
    config = INDEX_CONSTITUENTS[mode_upper]
    stocks = config["stock_pool"]
    
    data_list = []
    for code in stocks:
        name = STOCK_NAMES.get(code, code)
        data_list.append({"code": code, "name": name})
    
    df = pd.DataFrame(data_list)
    print(f"成功获取 {config['name']} 成分股，共 {len(df)} 只。")
    return df

def get_stock_name_mapping(index_df: pd.DataFrame) -> Dict[str, str]:
    """代码到名称的映射字典"""
    if index_df.empty:
        return {}
    return dict(zip(index_df["code"], index_df["name"]))

# ============================================================
# 真实行情数据获取（新浪财经API）
# ============================================================
def get_stock_data_sina(code: str, start: str, end: str) -> pd.DataFrame:
    """使用新浪财经API获取单只股票历史数据"""
    import urllib.request
    import json
    import time
    
    for attempt in range(2):
        try:
            url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
            url += f"?symbol={code}&scale=240&ma=no&datalen=1000"
            
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
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
                
                df = df[(df.index >= start) & (df.index <= end)]
                
                if len(df) > 0:
                    time.sleep(0.15)
                    return df
            
            time.sleep(0.3)
        except Exception:
            time.sleep(0.5)
    
    return pd.DataFrame()

def get_stock_data_akshare(code: str, start: str, end: str) -> pd.DataFrame:
    """使用akshare获取单只股票历史数据"""
    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq"
        )
        if df is not None and len(df) > 0:
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume"
            })
            df["date"] = pd.to_datetime(df["date"])
            return df[["date", "close"]].set_index("date")
    except Exception:
        pass
    return pd.DataFrame()

def get_stock_data_baostock(code: str, start: str, end: str) -> pd.DataFrame:
    """使用baostock获取单只股票历史数据"""
    try:
        bs_code = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
        rs = bs.query_history_k_data_plus(
            code=bs_code,
            fields="date,close",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2"
        )
        if rs.error_code == "0":
            data_list = []
            while rs.error_code == "0" and rs.next():
                data_list.append(rs.get_row_data())
            if data_list:
                df = pd.DataFrame(data_list, columns=["date", "close"])
                df["date"] = pd.to_datetime(df["date"])
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                return df.set_index("date").sort_index()
    except Exception:
        pass
    return pd.DataFrame()

def get_stock_data_fallback(code: str, start: str, end: str) -> pd.DataFrame:
    """备用数据生成器（使用随机游走模拟真实市场特征）"""
    rng = np.random.default_rng(int(hash(code)) % (2**31))
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    dates = pd.bdate_range(start_dt, end_dt)
    
    n = len(dates)
    if n == 0:
        return pd.DataFrame(columns=["close"])
    
    returns = rng.standard_normal(n) * 0.02 + 0.0003
    returns = np.clip(returns, -0.1, 0.1)
    
    price = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"close": price}, index=dates)
    return df

def get_market_data(index_df: pd.DataFrame, start: str, end: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """获取多只股票的市场数据（优先使用新浪财经API）"""
    if index_df.empty:
        raise ValueError("指数成分股数据为空，无法获取行情。")
    
    tickers = index_df["code"].tolist()
    name_map = get_stock_name_mapping(index_df)
    
    print(f"\n开始获取 {len(tickers)} 只股票的市场数据（{start} 至 {end}）...")
    
    price_df = pd.DataFrame()
    success_count = 0
    sina_success = 0
    ak_success = 0
    fallback_count = 0
    
    for code in tqdm(tickers, desc="行情数据下载进度"):
        df = pd.DataFrame()
        
        # 优先使用新浪财经API
        df = get_stock_data_sina(code, start, end)
        if not df.empty:
            sina_success += 1
        elif AKSHARE_AVAILABLE:
            df = get_stock_data_akshare(code, start, end)
            if not df.empty:
                ak_success += 1
        
        if df.empty and BAOSTOCK_AVAILABLE:
            df = get_stock_data_baostock(code, start, end)
        
        if df.empty:
            df = get_stock_data_fallback(code, start, end)
            if not df.empty:
                fallback_count += 1
        
        if not df.empty:
            price_df[code] = df["close"]
            success_count += 1
    
    price_df = price_df.sort_index().dropna(how="all")
    
    if price_df.empty:
        raise ValueError("未获取到任何有效行情数据。")
    
    returns = price_df.pct_change().dropna()
    returns = returns.clip(lower=-0.1, upper=0.1)
    
    valid_codes = returns.columns.tolist()
    name_map = {k: v for k, v in name_map.items() if k in valid_codes}
    
    print(f"\n数据获取统计：")
    print(f"  - 新浪财经: {sina_success} 只")
    print(f"  - akshare: {ak_success} 只")
    print(f"  - 备用模拟: {fallback_count} 只")
    print(f"数据获取完成：{len(returns)} 个交易日，{returns.shape[1]} 只有效资产。")
    return returns, name_map

def get_data(cfg: Config) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """获取市场数据"""
    cached_file = "real_stock_returns.csv"
    if os.path.exists(cached_file):
        print(f"\n发现缓存的真实数据文件: {cached_file}")
        print("使用已缓存数据，跳过API下载...")
        try:
            returns = pd.read_csv(cached_file, index_col=0, parse_dates=True)
            returns = returns.dropna(how='all')
            returns = returns[(returns.index >= cfg.start_date) & (returns.index <= cfg.end_date)]
            returns.columns = [c.replace('sh', '').replace('sz', '') for c in returns.columns]
            returns = returns.loc[:, ~returns.columns.duplicated()]
            name_map = {}
            for col in returns.columns:
                name = STOCK_NAMES.get(col, col)
                name_map[col] = name
            print(f"已加载 {len(returns)} 个交易日, {returns.shape[1]} 只股票")
            return returns, name_map
        except Exception as e:
            print(f"加载缓存数据失败: {e}，将重新下载")
    
    index_df = get_index_constituents(cfg.data_mode)
    cfg.n_assets = len(index_df)
    return get_market_data(index_df, cfg.start_date, cfg.end_date)

# ============================================================
# 贝叶斯后验
# ============================================================
def bayesian_posterior(
    returns: np.ndarray,
    cfg: Config,
    mu0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    returns = np.asarray(returns, dtype=float)
    T, d = returns.shape
    if T <= d:
        raise ValueError(f"训练样本数 T={T} 小于资产维度 d={d}")

    sample_mean = returns.mean(axis=0)
    
    if cfg.use_ledoit_wolf:
        lw = LedoitWolf().fit(returns)
        S = lw.covariance_ * T
    else:
        centered = returns - sample_mean
        S = centered.T @ centered

    if mu0 is None:
        mu0 = sample_mean.copy()
    nu0 = d + cfg.nu0_offset
    if cfg.lambda0_diag:
        sample_cov = np.cov(returns, rowvar=False, ddof=0)
        Lambda0 = np.diag(np.diag(sample_cov))
    else:
        Lambda0 = np.cov(returns, rowvar=False, ddof=0)

    kappa_n = cfg.kappa0 + T
    nu_n = nu0 + T
    mu_n = (cfg.kappa0 * mu0 + T * sample_mean) / kappa_n
    mean_diff = sample_mean - mu0
    Lambda_n = Lambda0 + S + (cfg.kappa0 * T / kappa_n) * np.outer(mean_diff, mean_diff)

    return mu_n, kappa_n, nu_n, Lambda_n

def posterior_predictive_moments(
    mu_n: np.ndarray,
    kappa_n: float,
    nu_n: float,
    Lambda_n: np.ndarray,
    d: int,
    objective_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nu_n <= d + 1:
        raise ValueError("nu_n 必须大于 d+1 保证协方差期望存在")

    E_mu = mu_n.copy()
    E_Sigma = Lambda_n / (nu_n - d - 1)
    Var_mu = E_Sigma / max(kappa_n, 1e-12)
    Sigma_pred = E_Sigma + Var_mu

    E_mu = objective_scale * E_mu
    Sigma_pred = objective_scale * Sigma_pred

    ridge = 1e-6 * np.trace(Sigma_pred) / max(d, 1)
    Sigma_pred = 0.5 * (Sigma_pred + Sigma_pred.T) + ridge * np.eye(d)
    return E_mu, Sigma_pred, E_Sigma / objective_scale

# ============================================================
# 共形球面流形几何
# ============================================================
def simplex_to_sphere(w: np.ndarray, cfg: Config) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, cfg.min_single_weight, cfg.max_single_weight)
    w = w / w.sum()
    q = np.sqrt(w)
    return q / np.linalg.norm(q)

def sphere_to_simplex(q: np.ndarray, cfg: Config) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = np.clip(q, np.sqrt(cfg.min_single_weight), np.sqrt(cfg.max_single_weight))
    q = q / np.linalg.norm(q)
    w = np.square(q)
    w = np.clip(w, cfg.min_single_weight, cfg.max_single_weight)
    return w / w.sum()

def tangent_projection(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v - np.dot(q, v) * q

def sphere_exponential_map(q: np.ndarray, v: np.ndarray, eps: float) -> np.ndarray:
    q = q / np.linalg.norm(q)
    v = tangent_projection(q, v)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        return q * (1.0 - 0.5 * (eps * v_norm) ** 2) + eps * v * (1.0 - (eps * v_norm) ** 2 / 6.0)
    return q * np.cos(eps * v_norm) + (v / v_norm) * np.sin(eps * v_norm)

# ============================================================
# 势能函数、梯度
# ============================================================
def potential_energy(q: np.ndarray, E_mu: np.ndarray, Sigma_pred: np.ndarray, lam: float, cfg: Config) -> float:
    w = sphere_to_simplex(q, cfg)
    utility_term = 0.5 * w @ Sigma_pred @ w - lam * (E_mu @ w)
    prior_term = -(cfg.weight_prior_alpha - 1.0) * np.sum(np.log(w))
    return float(utility_term + prior_term)

def euclidean_gradient(q: np.ndarray, E_mu: np.ndarray, Sigma_pred: np.ndarray, lam: float, cfg: Config) -> np.ndarray:
    w = sphere_to_simplex(q, cfg)
    g_w = Sigma_pred @ w - lam * E_mu
    g_prior = -(cfg.weight_prior_alpha - 1.0) / w
    return 2.0 * q * (g_w + g_prior)

def riemannian_gradient(q: np.ndarray, E_mu: np.ndarray, Sigma_pred: np.ndarray, lam: float, cfg: Config) -> np.ndarray:
    return tangent_projection(q, euclidean_gradient(q, E_mu, Sigma_pred, lam, cfg))

def hessian_matrix(q: np.ndarray, E_mu: np.ndarray, Sigma_pred: np.ndarray, lam: float, cfg: Config) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    d = len(q)
    w = sphere_to_simplex(q, cfg)
    g_w = Sigma_pred @ w - lam * E_mu
    g_prior = -(cfg.weight_prior_alpha - 1.0) / w
    H = np.zeros((d, d), dtype=float)

    for i in range(d):
        H[i, i] = 2.0 * (g_w[i] + g_prior[i]) + 4.0 * q[i] * q[i] * (Sigma_pred[i, i] + (cfg.weight_prior_alpha - 1.0) / (w[i] ** 2))
        for j in range(i + 1, d):
            H[i, j] = 4.0 * q[i] * q[j] * Sigma_pred[i, j]
            H[j, i] = H[i, j]

    H = 0.5 * (H + H.T)
    return H

# ============================================================
# 稳健的正定性保证
# ============================================================
def make_positive_definite(M: np.ndarray, min_eig: float = 1e-4) -> np.ndarray:
    d = M.shape[0]
    M = 0.5 * (M + M.T)

    ridge = 1e-6 * np.trace(M) / d
    M_adj = M + ridge * np.eye(d)

    try:
        np.linalg.cholesky(M_adj)
        return M_adj
    except np.linalg.LinAlgError:
        pass

    try:
        eigvals, eigvecs = np.linalg.eigh(M_adj)
        eigvals_clipped = np.maximum(eigvals, min_eig)
        M_fixed = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        M_fixed = 0.5 * (M_fixed + M_fixed.T)
        np.linalg.cholesky(M_fixed)
        return M_fixed
    except np.linalg.LinAlgError:
        pass

    diag_M = np.diag(np.maximum(np.diag(M), min_eig))
    return diag_M

def robust_mass_matrix_and_sample(
    q: np.ndarray,
    E_mu: np.ndarray,
    Sigma_pred: np.ndarray,
    lam: float,
    cfg: Config,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    d = len(q)

    w = sphere_to_simplex(q, cfg)
    vol_scale = np.sqrt(np.diag(Sigma_pred))
    vol_scale = np.maximum(vol_scale, 1e-2)
    M_diag = np.diag(1.0 / (vol_scale ** 2))

    try:
        H = hessian_matrix(q, E_mu, Sigma_pred, lam, cfg)
        M = 0.2 * M_diag + 0.8 * (0.5 * (H + H.T))
    except:
        M = M_diag

    M = make_positive_definite(M)

    try:
        L = np.linalg.cholesky(M)
        z = rng.normal(size=d)
        p = np.linalg.solve(L.T, z)
    except:
        M = np.eye(d)
        p = rng.normal(size=d)

    p = tangent_projection(q, p)
    return M, p

# ============================================================
# 收敛诊断
# ============================================================
def _autocorr_1d(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 2:
        return np.array([1.0])
    if max_lag is None:
        max_lag = min(n - 1, 200)
    fft_len = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=fft_len)
    acf = np.fft.irfft(fx * np.conjugate(fx), n=fft_len)[:n]
    acf = acf / acf[0]
    return acf[: max_lag + 1]

def effective_sample_size(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=float)
    n, d = x.shape
    ess = np.empty(d, dtype=float)
    for j in range(d):
        acf = _autocorr_1d(x[:, j], max_lag=min(n - 1, 200))
        tau = 1.0
        for k in range(1, len(acf)):
            if acf[k] <= 0:
                break
            tau += 2.0 * acf[k]
        ess[j] = n / max(tau, 1e-12)
    return ess

def gelman_rubin(chains: np.ndarray) -> np.ndarray:
    chains = np.asarray(chains, dtype=float)
    m, n, d = chains.shape
    if m < 2:
        return np.full(d, np.nan)
    chain_means = chains.mean(axis=1)
    grand_mean = chain_means.mean(axis=0)
    B = n / max(m - 1, 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
    var_hat = ((n - 1) / n) * W + B / n
    return np.sqrt(np.maximum(var_hat / np.maximum(W, 1e-12), 1.0))

def concentration_index(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w ** 2))

def portfolio_stats(returns: pd.Series, rf_annual: float) -> Dict[str, float]:
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

    return {
        "年化收益率": annual_ret,
        "年化波动率": annual_vol,
        "年化夏普": sharpe,
        "年化索提诺": sortino,
        "最大回撤": max_dd,
        "卡玛比率": calmar,
    }

# ============================================================
# CSN-HMC 采样器
# ============================================================
def csn_hmc_sampler(
    E_mu: np.ndarray,
    Sigma_pred: np.ndarray,
    lam: float,
    cfg: Config,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.random_seed)
    d = len(E_mu)
    q_chains = [simplex_to_sphere(rng.dirichlet(np.full(d, cfg.weight_prior_alpha)), cfg) for _ in range(cfg.n_chains)]
    eps = float(cfg.initial_step_size)
    eps_min, eps_max = cfg.step_size_min, cfg.step_size_max

    mu = np.log(10 * eps)
    gamma = cfg.dual_averaging_gamma
    t0 = 10.0
    kappa = 0.75
    H_bar = 0.0

    total_iters = cfg.n_burnin + cfg.n_samples
    chain_storage: List[List[np.ndarray]] = [[] for _ in range(cfg.n_chains)]
    accept_hist: List[float] = []
    eps_hist: List[float] = []

    pbar = tqdm(total=total_iters, desc="CSN-HMC采样进度", unit="iter")
    for t in range(1, total_iters + 1):
        iteration_accepts = []

        for m in range(cfg.n_chains):
            q0 = q_chains[m].copy()

            _, p0 = robust_mass_matrix_and_sample(q0, E_mu, Sigma_pred, lam, cfg, rng)

            H0 = potential_energy(q0, E_mu, Sigma_pred, lam, cfg) + 0.5 * float(p0 @ p0)
            q_curr = q0.copy()
            p_curr = p0.copy()

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
                accepted = 1.0
            else:
                accepted = 0.0

            iteration_accepts.append(accepted)
            if t > cfg.n_burnin:
                chain_storage[m].append(sphere_to_simplex(q_chains[m], cfg))

        accept_bar = float(np.mean(iteration_accepts))
        accept_hist.append(accept_bar)
        eps_hist.append(eps)

        if t <= cfg.n_burnin:
            eta = 1.0 / (t + t0)
            H_bar = (1.0 - eta) * H_bar + eta * (cfg.target_accept - accept_bar)
            log_eps = mu - (np.sqrt(t) / gamma) * H_bar
            eps = float(np.clip(np.exp(log_eps), eps_min, eps_max))

        phase = "预热期" if t <= cfg.n_burnin else "正式采样"
        pbar.update(1)
        pbar.set_postfix({"阶段": phase, "步长": f"{eps:.4f}", "接受率": f"{accept_bar:.2f}"})

    pbar.close()

    samples = np.vstack([np.asarray(v) for v in chain_storage if len(v) > 0])
    if samples.ndim == 1:
        samples = samples[None, :]

    min_len = min((len(v) for v in chain_storage), default=0)
    if min_len > 1 and cfg.n_chains >= 2:
        chain_arrs = np.asarray([np.asarray(v[:min_len]) for v in chain_storage])
        rhat = gelman_rubin(chain_arrs)
    else:
        rhat = np.full(d, np.nan)

    ess = effective_sample_size(samples) if len(samples) > 1 else np.full(d, np.nan)
    diagnostics = {
        "accept_rate_history": np.asarray(accept_hist),
        "accept_rate_mean": float(np.mean(accept_hist)),
        "rhat": rhat,
        "rhat_mean": float(np.nanmean(rhat)),
        "ess": ess,
        "ess_mean": float(np.nanmean(ess)),
        "ess_min": float(np.nanmin(ess)),
        "step_size_history": np.asarray(eps_hist),
        "final_step_size": eps,
    }

    return samples, diagnostics

# ============================================================
# 对比基准：完全无约束MV模型
# ============================================================
def equal_weight_portfolio(d: int) -> np.ndarray:
    return np.full(d, 1.0 / d)

def mean_variance_portfolio(E_mu: np.ndarray, Sigma: np.ndarray, lam: float, cfg: Config) -> np.ndarray:
    d = len(E_mu)
    x0 = equal_weight_portfolio(d)

    def obj(w: np.ndarray) -> float:
        return float(0.5 * w @ Sigma @ w - lam * (E_mu @ w))

    def grad(w: np.ndarray) -> np.ndarray:
        return Sigma @ w - lam * E_mu

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)},)
    bounds = [(0.0, 1.0) for _ in range(d)]

    res = minimize(
        obj, x0, method="SLSQP", jac=grad, bounds=bounds, constraints=cons,
        options={"ftol": 1e-12, "maxiter": 5000, "disp": False},
    )
    w = res.x if res.success else x0
    w = np.clip(w, 0.0, 1.0)
    return w / w.sum()

# ============================================================
# 滚动回测
# ============================================================
def _next_trading_position(index: pd.DatetimeIndex, date: pd.Timestamp) -> int:
    return int(index.searchsorted(date, side="right"))

def rolling_backtest(returns_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series], pd.DataFrame]:
    returns_df = returns_df.sort_index().copy()
    d = returns_df.shape[1]
    dates = returns_df.index
    assets = returns_df.columns

    rebalance_dates = returns_df.resample(cfg.rebalance_freq).last().index
    rebalance_dates = rebalance_dates[rebalance_dates >= dates[0]]
    rebalance_dates = rebalance_dates[rebalance_dates <= dates[-1]]

    if len(rebalance_dates) <= cfg.train_window_months + 1:
        raise ValueError("样本区间太短，无法进行滚动训练")

    tasks = list(range(cfg.train_window_months, len(rebalance_dates) - 1))
    print(f"\n开始滚动回测：总调仓次数 = {len(tasks)}")

    w_csn = pd.DataFrame(index=dates, columns=assets, dtype=float)
    w_mv = pd.DataFrame(index=dates, columns=assets, dtype=float)
    w_eq = pd.DataFrame(index=dates, columns=assets, dtype=float)

    diag_rows = []
    weight_rows = []
    last_csn_w = equal_weight_portfolio(d)

    for idx in tqdm(tasks, desc="滚动调仓进度", unit="期"):
        train_end = rebalance_dates[idx]
        train_start = rebalance_dates[idx - cfg.train_window_months]
        train_returns = returns_df.loc[train_start:train_end].dropna()

        if len(train_returns) <= d:
            raw_csn_w = equal_weight_portfolio(d)
            mv_w = raw_csn_w.copy()
            diag = {k: np.nan for k in ["accept_rate_mean", "final_step_size", "rhat_mean", "ess_mean", "ess_min"]}
        else:
            mu_n, kappa_n, nu_n, Lambda_n = bayesian_posterior(train_returns.values, cfg)
            E_mu, Sigma_pred, E_Sigma = posterior_predictive_moments(
                mu_n, kappa_n, nu_n, Lambda_n, d,
                objective_scale=cfg.objective_scale
            )

            samples, diag = csn_hmc_sampler(E_mu, Sigma_pred, cfg.risk_aversion, cfg)
            raw_csn_w = samples.mean(axis=0)
            raw_csn_w = np.clip(raw_csn_w, cfg.min_single_weight, cfg.max_single_weight)
            raw_csn_w = raw_csn_w / raw_csn_w.sum()

            mv_w = mean_variance_portfolio(E_mu, Sigma_pred, cfg.risk_aversion, cfg)

        csn_w = cfg.weight_smoothing * last_csn_w + (1 - cfg.weight_smoothing) * raw_csn_w
        csn_w = np.clip(csn_w, cfg.min_single_weight, cfg.max_single_weight)
        csn_w = csn_w / csn_w.sum()
        last_csn_w = csn_w.copy()

        eq_w = equal_weight_portfolio(d)

        start_pos = _next_trading_position(dates, train_end)
        end_pos = _next_trading_position(dates, rebalance_dates[idx + 1])
        if idx + 1 == len(rebalance_dates) - 1:
            end_pos = len(dates)

        if start_pos < end_pos:
            w_csn.iloc[start_pos:end_pos, :] = csn_w
            w_mv.iloc[start_pos:end_pos, :] = mv_w
            w_eq.iloc[start_pos:end_pos, :] = eq_w

        diag_rows.append({
            "rebalance_date": train_end,
            **diag,
            "csn_herfindahl": concentration_index(csn_w),
            "mv_herfindahl": concentration_index(mv_w),
        })

        weight_rows.append({
            "rebalance_date": train_end,
            **{f"CSN_{c}": csn_w[i] for i, c in enumerate(assets)},
            **{f"MV_{c}": mv_w[i] for i, c in enumerate(assets)},
            **{f"EW_{c}": eq_w[i] for i, c in enumerate(assets)},
        })

    # 填充权重数据
    for df in (w_csn, w_mv, w_eq):
        first_valid = df.dropna().index.min()
        if pd.notna(first_valid):
            first_row = df.loc[first_valid].values
            df.ffill(inplace=True)
            df.fillna(pd.Series(first_row, index=df.columns), inplace=True)
        else:
            df.iloc[:, :] = equal_weight_portfolio(d)

    # 构建用于计算交易成本的权重序列（仅包含调仓日的权重变化）
    # 首先提取调仓日的权重（使用 asof 获取最近的可交易日期）
    valid_rebalance_dates = [d for d in rebalance_dates if d in w_csn.index]
    if len(valid_rebalance_dates) == 0:
        valid_rebalance_dates = [w_csn.index.asof(d) for d in rebalance_dates if pd.notna(w_csn.index.asof(d))]
    
    rebalance_weight_csn = w_csn.loc[valid_rebalance_dates].dropna(how='all')
    rebalance_weight_mv = w_mv.loc[valid_rebalance_dates].dropna(how='all')
    rebalance_weight_eq = w_eq.loc[valid_rebalance_dates].dropna(how='all')
    
    # 计算调仓日的权重变化
    # 换手率 = 0.5 × |w_new - w_old| 的绝对值之和
    weight_change_csn = 0.5 * rebalance_weight_csn.diff().abs().sum(axis=1).fillna(0.0)
    weight_change_mv = 0.5 * rebalance_weight_mv.diff().abs().sum(axis=1).fillna(0.0)
    weight_change_eq = 0.5 * rebalance_weight_eq.diff().abs().sum(axis=1).fillna(0.0)
    
    # 将权重变化映射回完整的日期序列（交易成本在调仓日之后的第一个交易日扣除）
    turnover_csn = pd.Series(0.0, index=dates)
    turnover_mv = pd.Series(0.0, index=dates)
    turnover_eq = pd.Series(0.0, index=dates)
    
    valid_dates_for_loop = [d for d in valid_rebalance_dates if d in dates]
    for idx, date in enumerate(valid_dates_for_loop):
        if idx == 0:
            continue
        next_pos = _next_trading_position(dates, date)
        if next_pos < len(dates):
            next_date = dates[next_pos]
            if next_date in turnover_csn.index:
                turnover_csn.loc[next_date] = weight_change_csn.get(date, 0.0)
                turnover_mv.loc[next_date] = weight_change_mv.get(date, 0.0)
                turnover_eq.loc[next_date] = weight_change_eq.get(date, 0.0)
    
    # 计算组合收益（使用滞后一期的权重）
    lag_csn = w_csn.shift(1).bfill()
    lag_mv = w_mv.shift(1).bfill()
    lag_eq = w_eq.shift(1).bfill()

    gross_csn = (returns_df * lag_csn).sum(axis=1)
    gross_mv = (returns_df * lag_mv).sum(axis=1)
    gross_eq = (returns_df * lag_eq).sum(axis=1)

    # 正确的复利计算：交易成本按比例减少资金
    # 
    # 交易过程：
    # 1. 调仓日收盘后：资金变为 1 × (1 - turnover × cost)
    # 2. 次日开始用剩余资金投资
    # 3. 次日收益：剩余资金 × (1 + 日收益率)
    #
    # 正确公式：
    # value_t = value_{t-1} × (1 + r_t) × (1 - turnover_{t-1} × cost)
    #
    # 日收益因子 = (1 + gross_return) × (1 - turnover × cost_rate)
    
    # 日收益因子（考虑交易成本）
    daily_factor_csn = (1.0 + gross_csn) * (1.0 - turnover_csn * cfg.transaction_cost)
    daily_factor_mv = (1.0 + gross_mv) * (1.0 - turnover_mv * cfg.transaction_cost)
    daily_factor_eq = (1.0 + gross_eq) * (1.0 - turnover_eq * cfg.transaction_cost)

    # 净值曲线（从索引1开始累积，索引0为初始资金1.0）
    curves = {}
    for name, factor in [("CSN-HMC", daily_factor_csn), ("MV", daily_factor_mv), ("EW", daily_factor_eq)]:
        cumprod_series = pd.Series(1.0, index=factor.index)
        for i in range(1, len(factor)):
            cumprod_series.iloc[i] = cumprod_series.iloc[i-1] * factor.iloc[i]
        curves[name] = cumprod_series
    
    # 日收益率
    dailies = {
        "CSN-HMC": daily_factor_csn - 1.0,
        "MV": daily_factor_mv - 1.0,
        "EW": daily_factor_eq - 1.0,
    }

    diag_df = pd.DataFrame(diag_rows).set_index("rebalance_date")
    weight_df = pd.DataFrame(weight_rows).set_index("rebalance_date")

    return diag_df, dailies, curves, weight_df

# ============================================================
# 结果汇总与可视化（含股票名称）
# ============================================================
def summarize_results(dailies: Dict[str, pd.Series], curves: Dict[str, pd.Series], cfg: Config) -> pd.DataFrame:
    rows = {}
    for name, ret in dailies.items():
        stats = portfolio_stats(ret, cfg.risk_free_annual)
        stats["期末净值"] = float(curves[name].iloc[-1])
        rows[name] = stats
    return pd.DataFrame(rows).T

def annual_turnover_rate(weights: pd.DataFrame) -> float:
    monthly_weights = weights.resample("ME").last()
    monthly_changes = monthly_weights.diff().abs().sum(axis=1).dropna()
    avg_monthly_turnover = monthly_changes.mean()
    return float(avg_monthly_turnover * 12 * 100)

def print_config(cfg: Config) -> None:
    print("\n===== 当前运行配置 =====")
    items = asdict(cfg)
    for k, v in items.items():
        print(f"{k}: {v}")

def print_tables(summary: pd.DataFrame, diag_df: pd.DataFrame, weight_df: pd.DataFrame, name_map: Dict[str, str], cfg: Config) -> None:
    print("\n===== 最终回测业绩总表（扣除交易成本后） =====")
    out = summary.copy()
    out.loc["CSN-HMC", "年化换手率(%)"] = annual_turnover_rate(weight_df[[c for c in weight_df.columns if c.startswith("CSN_")]])
    out.loc["MV", "年化换手率(%)"] = annual_turnover_rate(weight_df[[c for c in weight_df.columns if c.startswith("MV_")]])
    out.loc["EW", "年化换手率(%)"] = annual_turnover_rate(weight_df[[c for c in weight_df.columns if c.startswith("EW_")]])
    display_cols = ["年化收益率", "年化波动率", "年化夏普", "年化索提诺", "最大回撤", "卡玛比率", "期末净值", "年化换手率(%)"]
    print(out[display_cols].map(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) and np.isfinite(x) else x))

    print("\n===== CSN-HMC 采样诊断总表（前5行） =====")
    show_cols = [c for c in ["accept_rate_mean", "final_step_size", "rhat_mean", "ess_mean", "ess_min", "csn_herfindahl", "mv_herfindahl"] if c in diag_df.columns]
    print(diag_df[show_cols].head().round(4))

    print("\n===== 最近一期 CSN-HMC 权重（降序，含名称） =====")
    last = weight_df.iloc[-1]
    csn_cols = [c for c in weight_df.columns if c.startswith("CSN_")]
    csn_series = last[csn_cols].sort_values(ascending=False)
    
    # 映射名称
    csn_df = csn_series.reset_index()
    csn_df.columns = ["code_weight", "权重"]
    csn_df["code"] = csn_df["code_weight"].str.replace("CSN_", "")
    csn_df["名称"] = csn_df["code"].map(name_map).fillna(csn_df["code"])
    csn_df = csn_df[["code", "名称", "权重"]]
    print(csn_df.to_string(index=False))

    print("\n===== 最近一期 无约束MV 权重（降序，前10，含名称） =====")
    last_mv = weight_df.iloc[-1]
    mv_cols = [c for c in weight_df.columns if c.startswith("MV_")]
    mv_series = last_mv[mv_cols].sort_values(ascending=False).head(10)
    
    mv_df = mv_series.reset_index()
    mv_df.columns = ["code_weight", "权重"]
    mv_df["code"] = mv_df["code_weight"].str.replace("MV_", "")
    mv_df["名称"] = mv_df["code"].map(name_map).fillna(mv_df["code"])
    mv_df = mv_df[["code", "名称", "权重"]]
    print(mv_df.to_string(index=False))

def plot_results(curves: Dict[str, pd.Series], diag_df: pd.DataFrame, weight_df: pd.DataFrame, name_map: Dict[str, str], cfg: Config) -> None:
    if not cfg.show_plots:
        return
    if cfg.use_chinese_font:
        plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={"height_ratios": [2, 1.4, 1.4]})

    for name, curve in curves.items():
        axes[0].plot(curve.index, curve.values, label=name, linewidth=2, alpha=0.85)
    axes[0].set_title(f"累计净值曲线对比（{cfg.data_mode}）", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("净值", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3, linestyle="--")

    for name, curve in curves.items():
        dd = curve / curve.cummax() - 1.0
        axes[1].plot(dd.index, dd.values, label=name, linewidth=1.5, alpha=0.85)
    axes[1].set_title("回撤曲线对比", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("回撤", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3, linestyle="--")

    if "accept_rate_mean" in diag_df.columns:
        axes[2].plot(diag_df.index, diag_df["accept_rate_mean"], label="平均接受率", linewidth=1.8, color="#1f77b4")
    if "final_step_size" in diag_df.columns:
        ax2 = axes[2].twinx()
        ax2.plot(diag_df.index, diag_df["final_step_size"], label="最终步长", linewidth=1.8, color="#ff7f0e", linestyle="--")
    axes[2].set_title("CSN-HMC 采样诊断概览", fontsize=14, fontweight="bold")
    axes[2].set_ylabel("接受率", fontsize=12)
    if "final_step_size" in diag_df.columns:
        ax2.set_ylabel("步长", fontsize=12)
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if "final_step_size" in diag_df.columns else ([], [])
    if lines1 or lines2:
        axes[2].legend(lines1 + lines2, labels1 + labels2, fontsize=11)
    axes[2].grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.show()

    # 权重对比图
    csn_cols = [c for c in weight_df.columns if c.startswith("CSN_")]
    if len(csn_cols) > 0:
        plt.figure(figsize=(14, 7))
        last_csn = weight_df.iloc[-1][csn_cols].sort_values(ascending=False).head(20)
        codes = [c.replace("CSN_", "") for c in last_csn.index]
        names = [name_map.get(c, c) for c in codes]
        labels = [f"{n}\n({c})" for n, c in zip(names, codes)]
        
        plt.bar(labels, last_csn.values, color="#1f77b4", alpha=0.8)
        plt.title(f"最近一期 CSN-HMC 个股权重（Top20，{cfg.data_mode}）", fontsize=14, fontweight="bold")
        plt.ylabel("权重", fontsize=12)
        plt.xticks(rotation=45, fontsize=10, ha='right')
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.show()

# ============================================================
# 全参数交互式配置
# ============================================================
def interactive_config() -> Config:
    cfg = Config()
    
    print("=" * 80)
    print("CSN-HMC 投资组合优化器")
    print("=" * 80)
    print("提示：直接回车使用默认值\n")

    # 基础设置
    print("--- 1. 基础设置 ---")
    print("支持的指数：SH50(上证50), CSI300(沪深300), CSI500(中证500)")
    mode_input = input(f"数据模式 [默认: {cfg.data_mode}]: ").strip()
    if mode_input:
        cfg.data_mode = mode_input.upper()

    start_input = input(f"回测开始日期 (YYYY-MM-DD) [默认: {cfg.start_date}]: ").strip()
    if start_input:
        cfg.start_date = start_input
    
    end_input = input(f"回测结束日期 (YYYY-MM-DD) [默认: {cfg.end_date}]: ").strip()
    if end_input:
        cfg.end_date = end_input

    # 回测核心参数
    print("\n--- 2. 回测核心参数 ---")
    show_backtest = input("是否修改回测核心参数? (y/n) [默认: n]: ").strip().lower()
    if show_backtest == 'y':
        cfg.train_window_months = int(input(f"训练窗口月数 [默认: {cfg.train_window_months}]: ") or cfg.train_window_months)
        cfg.rebalance_freq = input(f"调仓频率 (ME=月末, W=周末, D=日) [默认: {cfg.rebalance_freq}]: ") or cfg.rebalance_freq
        cfg.transaction_cost = float(input(f"双边交易成本 (如0.003=千三) [默认: {cfg.transaction_cost}]: ") or cfg.transaction_cost)
        cfg.risk_aversion = float(input(f"风险厌恶系数 (越小越激进) [默认: {cfg.risk_aversion}]: ") or cfg.risk_aversion)
        cfg.max_single_weight = float(input(f"单只个股权重上限 [默认: {cfg.max_single_weight}]: ") or cfg.max_single_weight)
        cfg.min_single_weight = float(input(f"单只个股权重下限 [默认: {cfg.min_single_weight}]: ") or cfg.min_single_weight)
        cfg.weight_smoothing = float(input(f"权重平滑系数 (0-1，越小越灵活) [默认: {cfg.weight_smoothing}]: ") or cfg.weight_smoothing)

    # 贝叶斯先验
    print("\n--- 3. 贝叶斯先验参数 ---")
    show_bayes = input("是否修改贝叶斯先验参数? (y/n) [默认: n]: ").strip().lower()
    if show_bayes == 'y':
        cfg.kappa0 = float(input(f"收益先验强度 kappa0 (越小越相信历史) [默认: {cfg.kappa0}]: ") or cfg.kappa0)
        cfg.use_ledoit_wolf = (input(f"是否使用Ledoit-Wolf协方差收缩? (y/n) [默认: {'y' if cfg.use_ledoit_wolf else 'n'}]: ").strip().lower() != 'n')

    # 采样参数
    print("\n--- 4. CSN-HMC 采样参数 (高级) ---")
    show_sampler = input("是否修改采样参数? (y/n) [默认: n]: ").strip().lower()
    if show_sampler == 'y':
        cfg.n_samples = int(input(f"采样数 n_samples [默认: {cfg.n_samples}]: ") or cfg.n_samples)
        cfg.n_burnin = int(input(f"预热步数 n_burnin [默认: {cfg.n_burnin}]: ") or cfg.n_burnin)
        cfg.target_accept = float(input(f"目标接受率 [默认: {cfg.target_accept}]: ") or cfg.target_accept)
        cfg.step_size_max = float(input(f"最大步长 [默认: {cfg.step_size_max}]: ") or cfg.step_size_max)

    print("\n" + "=" * 80)
    print("配置完成！开始运行...")
    print("=" * 80 + "\n")
    
    return cfg

# ============================================================
# 主函数
# ============================================================
def main() -> None:
    import sys
    interactive = len(sys.argv) > 1 and sys.argv[1] == "--interactive"
    
    if interactive:
        cfg = interactive_config()
    else:
        print("使用默认配置运行（非交互模式）")
        print("使用 --interactive 参数启动交互式配置")
        cfg = Config()
        cfg.show_plots = True
    
    print_config(cfg)
    np.random.seed(cfg.random_seed)

    print("\n===== 1. 数据获取 =====")
    try:
        returns_df, name_map = get_data(cfg)
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