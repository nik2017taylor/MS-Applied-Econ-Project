# -*- coding: utf-8 -*-
"""BITCOIN-CONTROL REPAIR FOR THE EXISTING THREE-MARKET MINING STUDY.

Open this .py from inside Spyder; press F5. No companion Python file is needed.
Reuses the unchanged local price/weather/gas checkpoints from LMP_single_file.
Downloads ONLY the public Coin Metrics BTC CSV if no pinned BTC snapshot exists.
No AWS, no GridStatus quota, no new electricity download, no package installation.

Restores the earlier Massena mining specification to the common daily framework:
  (1) weather-only and weather+gas incremental comparisons;
  (2) lagged log BTC/USD price and its post interaction;
  (3) preferred lagged log hashprice and its post interaction.
Price-only and hashprice regressions are SEPARATE specifications, not an
unidentified collection of redundant profitability components in one equation.
All common-sample specifications use the same hourly/daily eligibility mask.
An original-sample weather+gas model is retained ONLY as an omission comparator.

Every model using BTC includes its specified BTC term in BOTH the hourly
mean regression and ALL daily mean/log-moment/tail regressions. Design-matrix
assertions block a model if its declared BTC terms are missing. Missing BTC
is never forward-filled, replaced by zero, or silently omitted.

Primary results: crypto_key_results_df / crypto_run_summary_df.
Comparisons: crypto_specification_comparison_df (post mean and total moment).
Audit: crypto_control_audit_df / crypto_centering_df / crypto_coverage_df.

All existing location/date/mapping limitations remain. These are conditional
mining-event associations, NOT causal AI or metered-load effects. The existing
PDF is the OMITTED-BTC version until a new empirical run is inspected.

Hashprice: (IssTotUSD + FeeTotNtv*PriceUSD)/(HashRate/1000), USD/PH/s-day.
HashRate is Coin Metrics BTC TH/s. This is a public network revenue indicator,
not a facility's observed profit or an instrument. No LMP-subtracted margin
enters any regression. ASIC scenarios are descriptive, never fitted controls.
The prior-day UTC label is mapped to local date d using label d-1. Archive
vintages are recorded; this is not a point-in-time publication-vintage database.
HAC and block intervals condition on the fitted first stage as before.

Sources: the user's massena_end_to_end_nodal_volatility.py mining-fundamentals
module and LMP_Analyze_Existing_Data.py common estimation/checkpoint code;
https://github.com/coinmetrics/data (CC BY-NC 4.0 community data)
https://docs.coinmetrics.io/network-data/network-data-overview/mining/hash-rate
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import itertools
import math
import re
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from contextlib import nullcontext

# %% SETTINGS -- normally no changes needed
PROJECT_FOLDER = Path.home() / "data_center_study_data" / "LMP_single_file"
CACHE_DIR = PROJECT_FOLDER / "cache"
RUN_SITES = ("massena", "rockdale", "nautilus")
MAKE_PLOTS = False  # Tables first; no extra interactive plotting needed
BOOTSTRAP_DRAWS = 499
BOOTSTRAP_BLOCK_DAYS = 7
MIN_HOURLY_COVERAGE = 0.80
MIN_DAILY_COVERAGE = 0.90
MIN_DAYS_PER_SIDE = 90
HAC_DAYS = 30
MAX_BLAS_THREADS = 2
# Match any custom settings you used in the downloader. Defaults match it exactly.
NYISO_SOURCE = "public"
PJM_MANUAL_ROUTES = ()
SITE_OVERRIDES = {}
# Prefer validated checkpoints to mutable/stale workspace objects.
ALLOW_WORKSPACE_FALLBACK = True
# Only needed when the checkpoint is elsewhere: exact site_inputs directory per site.
CHECKPOINT_DIRECTORY_OVERRIDES = {}
# Example: {"massena": r"C:\...\cache\site_inputs\massena_<token>"}

# %% BITCOIN REPAIR SETTINGS
# Old checkpoints are READ ONLY. New BTC inputs/results use separate directories.
BTC_LOCAL_CSV = ""  # Optional path to the original-format Coin Metrics btc.csv.
BTC_ALLOW_DOWNLOAD = True
BTC_REFRESH = False  # Reuse the pinned local BTC snapshot on later F5 runs.
BTC_CACHE_DIR = PROJECT_FOLDER / "bitcoin_controls_v1"
BTC_RESULTS_DIR = PROJECT_FOLDER / "outputs" / "bitcoin_controls_v1"
SAVE_BTC_RESULTS = True
BTC_DOWNLOAD_MAX_BYTES = 40_000_000
BTC_DOWNLOAD_TIMEOUT = 45
BTC_DOWNLOAD_ATTEMPTS = 2
# Restores the original lag-one UTC-date-label convention. No forward fill.
BTC_LAG_CALENDAR_DAYS = 1
BTC_ARCHIVE_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
# Actual market/network observations, not assumptions about a facility's hardware.
BTC_METRICS = ("time", "PriceUSD", "FeeTotNtv", "BlkCnt", "IssTotNtv", "IssTotUSD", "HashRate")
# Difficulty is retained if available, but is NOT a separate mandatory regressor.
# Hashprice already summarizes revenue per unit of network hashrate.
SPECIFICATIONS = (
    "weather_gas_original_sample",
    "weather_only_common_sample",
    "weather_gas_common_sample",
    "btc_price_interaction",
    "hashprice_interaction",
)
PRIMARY_SPECIFICATION = "hashprice_interaction"
SPEC_CONTROLS = {
    "weather_gas_original_sample": None,
    "weather_only_common_sample": None,
    "weather_gas_common_sample": None,
    "btc_price_interaction": "btc_log_price_lag1d_centered",
    "hashprice_interaction": "btc_log_hashprice_lag1d_centered",
}
ACTIVE_SPECIFICATION = PRIMARY_SPECIFICATION
_ACTIVE_SITE = ""
_ACTIVE_NODE = ""
_MINING_DAILY = None
_CONTROL_AUDIT_ROWS = []
_BTC_CENTER_ROWS = []
_SOURCE_RECORD = {}


try:
    import numpy as np
    import pandas as pd
    import patsy
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
except ImportError as exc:
    raise RuntimeError(
        "Use the same Spyder interpreter as the successful download. "
        "This script does not install packages; only the BTC source can require a network call. "
        "Missing import: " + str(exc)
    ) from exc


def numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric extension arrays to ordinary floats without imputing values."""
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col].dtype) or pd.api.types.is_bool_dtype(out[col].dtype):
            out[col] = pd.to_numeric(out[col], errors="coerce").to_numpy(
                dtype="float64", na_value=np.nan
            )
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    return out


def _redact_message(error):
    text = str(error)
    for variable in ("GRIDSTATUS_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        secret = os.getenv(variable)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return re.sub(r"(?i)(api[_-]?key=)[^&\s]+", r"\1[REDACTED]", text)


# %% Inherited registry (unchanged nominal dates; not new operational verification)
@dataclass(frozen=True)
class Site:
    key: str
    name: str
    iso: str
    timezone: str
    start: str
    event: str
    end: str
    benchmark: str
    nodes: tuple[str, ...]
    noaa_station: str
    noaa_name: str
    dataset: str
    interval_minutes: int
    ai_tier: str = 'C'
    date_status: str = 'legacy nominal date; operational ramp not verified'
    mapping_status: str = 'regional/nodal proxy, not a verified facility meter'
    alternatives: tuple[str, ...] = ()

    def timestamp(self, field: str) -> pd.Timestamp:
        return pd.Timestamp(getattr(self, field)).tz_localize(self.timezone)

SITES = {'massena': Site('massena',
        'Massena computing/mining pilot',
        'NYISO',
        'America/New_York',
        '2015-05-01',
        '2018-05-01',
        '2021-05-01',
        'NORTH',
        ('SITHE___MASSENA',
            'ALCOA___DSASP',
            'HQ_GEN_CEDARS',
            'ST LAWRENCE____',
            'ALCOA_RYNLDS___DRP'),
        '72622394725',
        'Massena International Airport',
        'nyiso_lmp_real_time_hourly',
        60),
    'rockdale': Site('rockdale',
        'Whinstone/Rockdale mining pilot',
        'ERCOT',
        'America/Chicago',
        '2017-05-01',
        '2020-05-01',
        '2023-05-01',
        'HB_SOUTH',
        ('LZ_SOUTH',
            ),
        '72103400351',
        'Cameron Municipal Airport',
        'ercot_public_rtm_spp',
        15,
        mapping_status='legacy SOUTH zone minus SOUTH hub; Rockdale electrical mapping unverified',
        alternatives=('2021-05-26',
            )),
    'nautilus': Site('nautilus',
        'Nautilus/Susquehanna mining-phase pilot',
        'PJM',
        'America/New_York',
        '2020-02-01',
        '2023-02-01',
        '2026-02-01',
        'PPL',
        (),
        '72513014777',
        'Wilkes-Barre/Scranton International Airport',
        'pjm_lmp_real_time_hourly',
        60,
        mapping_status='name-screened candidate pnodes; connection not verified',
        alternatives=('2023-04-01',
            ))}

def utc_series(values, name='time') -> pd.Series:
    """Accept explicit UTC/offsets, reject naive timestamps and missing timestamps."""
    s = pd.Series(values)
    if s.isna().any():
        raise ValueError(f'{name}: missing timestamps are not accepted.')
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert('UTC')
    if not all(getattr(pd.Timestamp(v), 'tzinfo', None) is not None for v in s):
        raise ValueError(f'{name}: timezone-naive times are not accepted; provide explicit UTC/offsets.')
    # pandas 2+ needs format=mixed when strings span different offsets or layouts.
    return pd.to_datetime(s, utc=True, errors='raise', format='mixed')

def alias_table(prices: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    wide = prices.pivot(index='hour_utc', columns='location', values='lmp')
    rows = []
    for a, b in itertools.combinations([n for n in nodes if n in wide], 2):
        pair = wide[[a, b]].dropna()
        equal_masks = wide[a].notna().equals(wide[b].notna())
        exact = bool(len(pair) > 0 and equal_masks and np.array_equal(pair[a], pair[b]))
        rows.append(dict(node_a=a,
                node_b=b,
                common_hours=len(pair),
                exact_alias=exact,
                max_abs_difference=float((pair[a] - pair[b]).abs().max()) if len(pair) else np.nan))
    return pd.DataFrame(rows)

def build_price_panel(prices: pd.DataFrame,
    site: Site,
    node: str) -> pd.DataFrame:
    p = numeric_frame(prices)
    if 'complete_hour' in p:
        p.loc[~p.complete_hour.eq(1), 'lmp'] = np.nan
    p['lmp'] = pd.to_numeric(p['lmp'], errors='coerce').to_numpy(dtype='float64', na_value=np.nan)
    p['hour_utc'] = utc_series(p.hour_utc).array
    if p.duplicated(['hour_utc', 'location']).any():
        raise ValueError('Hourly prices contain duplicate location-times.')
    wide = p.pivot(index='hour_utc', columns='location', values='lmp')
    if node not in wide or site.benchmark not in wide:
        raise ValueError(f'Missing focal/benchmark series: {node}, {site.benchmark}')
    start, end, event = (site.timestamp(k).tz_convert('UTC') for k in ('start', 'end', 'event'))
    hours = pd.date_range(start, end, freq='h', inclusive='left')
    out = wide[[node, site.benchmark]].reindex(hours).rename(columns={node: 'local_price', site.benchmark: 'benchmark_price'})
    out.index.name = 'hour_utc'
    out = out.reset_index()
    out['basis'] = out.local_price - out.benchmark_price
    out['hour_local'] = out.hour_utc.dt.tz_convert(site.timezone)
    out['date_local'] = out.hour_local.dt.tz_localize(None).dt.normalize()
    out['hour'] = out.hour_local.dt.hour
    out['dow'] = out.hour_local.dt.dayofweek
    out['month'] = out.hour_local.dt.month
    out['post'] = (out.hour_utc >= event).astype(int)
    out['trend_years'] = (out.hour_utc - event).dt.total_seconds() / (86400 * 365.25)
    out['post_trend'] = out.post * out.trend_years
    out['site_key'], out['node'], out['ai_tier'] = site.key, node, site.ai_tier
    return out

def build_panel(prices, weather, gas, site: Site, node: str):
    out = build_price_panel(prices, site, node)
    notes = []
    if weather.empty or not {'hour_utc', 'temperature_c'}.issubset(weather):
        out['temperature_c'] = np.nan
        notes.append('Temperature frame missing/empty; adjusted model blocked, not omitted.')
    else:
        w = numeric_frame(weather[['hour_utc', 'temperature_c']])
        w['temperature_c'] = pd.to_numeric(w['temperature_c'], errors='coerce').to_numpy(dtype='float64', na_value=np.nan)
        w['hour_utc'] = utc_series(w.hour_utc).array
        w = w.groupby('hour_utc', as_index=False).temperature_c.mean()
        out = out.merge(w, on='hour_utc', how='left', validate='one_to_one')
    if gas.empty or not {'date_local', 'henry_hub_usd_per_mmbtu'}.issubset(gas):
        out['gas'] = np.nan
        notes.append('Gas frame missing/empty; adjusted model blocked, not omitted.')
    else:
        g = numeric_frame(gas[['date_local', 'henry_hub_usd_per_mmbtu']])
        g['date_local'] = pd.to_datetime(g.date_local, errors='raise').dt.normalize()
        if getattr(g.date_local.dt, 'tz', None) is not None:
            raise ValueError('Gas dates must be timezone-naive local calendar dates.')
        if g.groupby('date_local').henry_hub_usd_per_mmbtu.nunique().gt(1).any():
            raise ValueError('Conflicting Henry Hub observations for the same date.')
        g = g.drop_duplicates('date_local').set_index('date_local').sort_index()
        dates = pd.date_range(min(g.index.min(), out.date_local.min()), out.date_local.max(), freq='D')
        # Preserve the inherited seven-calendar-day, backward-looking gas carry.
        g = g.reindex(dates).ffill(limit=7).rename_axis('date_local').reset_index()
        out = out.merge(g, on='date_local', how='left', validate='many_to_one')
        out['gas'] = out.henry_hub_usd_per_mmbtu
    out['temp2'] = out.temperature_c ** 2
    out = attach_bitcoin_controls(out, site)
    out = numeric_frame(out)
    rows = []
    for post, period in [(0, 'pre'), (1, 'post')]:
        z = out[out.post.eq(post)]
        rows.append(dict(site_key=site.key, node=node, period=period,
            expected_hours=len(z), paired_price_hours=int(z.basis.notna().sum()),
            paired_coverage=z.basis.notna().mean(), weather_coverage=z.temperature_c.notna().mean(),
            gas_coverage=z.gas.notna().mean(),
            base_weather_gas_coverage=z[['basis','temperature_c','gas']].notna().all(axis=1).mean(),
            bitcoin_price_coverage=z.btc_log_price_lag1d.notna().mean(),
            bitcoin_hashprice_coverage=z.btc_log_hashprice_lag1d.notna().mean(),
            complete_controls_coverage=z.estimation_valid.eq(1).mean(),
            notes=' '.join(notes)))
    return out, pd.DataFrame(rows)


def daily_panel(hourly: pd.DataFrame) -> pd.DataFrame:
    d = numeric_frame(hourly)
    valid = d.estimation_valid.eq(1)
    counts = d.groupby('date_local').size().rename('expected_hours')
    for c in ['basis', 'temperature_c', 'temp2', 'gas', 'local_price', 'benchmark_price']:
        d.loc[~valid, c] = np.nan
    daily = d.groupby('date_local').agg(basis=('basis',
            'mean'),
        temperature_c=('temperature_c',
            'mean'),
        temp2=('temp2',
            'mean'),
        gas=('gas',
            'mean'),
        local_price=('local_price',
            'mean'),
        benchmark_price=('benchmark_price',
            'mean'),
        valid_hours=('basis',
            'count'),
        post=('post',
            'first'),
        trend_years=('trend_years',
            'mean'),
        post_trend=('post_trend',
            'mean'))
    # Use precisely the same eligible hours as basis and weather.
    extra_columns = ['btc_log_price_lag1d_centered', 'btc_log_hashprice_lag1d_centered',
                     'btc_log_price_lag1d', 'btc_log_hashprice_lag1d']
    for c in extra_columns:
        d.loc[~valid, c] = np.nan
    daily = daily.join(d.groupby('date_local')[extra_columns].mean())
    daily = daily.join(counts)
    daily['daily_coverage'] = daily.valid_hours / daily.expected_hours
    daily['usable_day'] = daily.daily_coverage.ge(MIN_DAILY_COVERAGE)
    daily.loc[~daily.usable_day, ['basis', 'temperature_c', 'temp2', 'gas', 'local_price', 'benchmark_price']] = np.nan
    daily['month'] = daily.index.month
    daily['dow'] = daily.index.dayofweek
    return daily.reset_index()

def fit_daily_hac(formula: str, data: pd.DataFrame, maxlags=HAC_DAYS):
    """OLS with Bartlett HAC on calendar-day distances, not compressed row distances.
    Large-sample normal CIs; covariance conditions on regressors/first-stage residuals.
    """
    data = numeric_frame(data)
    y, x = patsy.dmatrices(formula, data, return_type='dataframe', NA_action='drop')
    verify_required_controls(x, formula, stage='daily')
    if len(x) < max(60, 3 * x.shape[1]):
        raise ValueError('Too few complete days for this model.')
    if np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
        raise ValueError('Rank-deficient design; inspect window length and constant controls.')
    fit = sm.OLS(y.iloc[:, 0], x).fit()
    score = x.to_numpy() * fit.resid.to_numpy()[:, None]
    days = pd.to_datetime(data.loc[x.index, 'date_local'])
    if days.duplicated().any():
        raise ValueError('HAC input must have one observation per local calendar day.')
    positions = (days - days.min()).dt.days.to_numpy()
    full = np.zeros((int(positions.max()) + 1, score.shape[1]))
    full[positions] = score
    meat = full.T @ full
    for lag in range(1, min(maxlags, len(full) - 1) + 1):
        cross = full[lag:].T @ full[:-lag]
        meat += (1 - lag / (maxlags + 1)) * (cross + cross.T)
    bread = np.asarray(fit.normalized_cov_params, dtype=float)
    cov = bread @ meat @ bread
    cov *= len(x) / (len(x) - x.shape[1])
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    values = fit.params.to_numpy()
    z = np.divide(values, se, out=np.full_like(values, np.nan), where=se > 0)
    table = pd.DataFrame(dict(term=x.columns,
            estimate=values,
            std_error=se,
            ci_low=values - 1.95996398454 * se,
            ci_high=values + 1.95996398454 * se,
            p_value=2 * stats.norm.sf(np.abs(z)),
            n_days=len(x),
            hac_calendar_days=maxlags))
    # .summary() on the saved fitted object now uses the SAME calendar-HAC
    # covariance as coefficients_df, rather than ordinary OLS standard errors.
    fit._results.cov_params_default = cov
    fit._results.cov_type = 'HAC-calendar'
    fit._results.cov_kwds = {'description': f'Bartlett HAC on actual calendar-day distances; {maxlags} lags.'}
    fit._results.use_t = False
    fit._results._cache.clear()
    return (fit, table, cov)

MEAN_RHS = ''  # Set explicitly by activate_specification(), not a user omission switch.

def block_ratio_ci(daily: pd.DataFrame, field: str, draws=BOOTSTRAP_DRAWS, block=BOOTSTRAP_BLOCK_DAYS, seed=20260921) -> dict:
    """Moving-block CI for post/pre mean residual moment, conditional on first-stage fit.
    Sampling is on a complete daily grid within each regime, including missing days.
    This is descriptive, not a treatment-effect CI, and is not a null-centered test.
    """
    arrays = []
    for post in (0, 1):
        z = daily.loc[daily.post.eq(post), ['date_local', field]].set_index('date_local').sort_index()
        z = z.reindex(pd.date_range(z.index.min(), z.index.max(), freq='D'))[field].to_numpy(float)
        arrays.append(z)
    a, b = arrays
    pre, post = (np.nanmean(a), np.nanmean(b))
    row = dict(outcome=field,
        pre_level=pre,
        post_level=post,
        level_difference=post - pre,
        ratio=np.nan,
        ratio_ci_low=np.nan,
        ratio_ci_high=np.nan,
        draws=draws,
        block_days=block,
        near_zero_denominator=bool(pre <= 1e-08),
        inference='descriptive block CI; conditional on fitted hourly mean; no causal interpretation')
    if pre <= 1e-08 or any((np.isfinite(z).sum() < 2 * block for z in arrays)):
        return row
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(draws):
        means = []
        for z in arrays:
            starts = rng.integers(0, len(z) - block + 1, math.ceil(len(z) / block))
            sample = np.concatenate([z[i:i + block] for i in starts])[:len(z)]
            means.append(np.nanmean(sample))
        if means[0] > 1e-08 and np.isfinite(means).all():
            ratios.append(means[1] / means[0])
    row['ratio'] = post / pre
    if len(ratios) > draws * 0.9:
        row['ratio_ci_low'], row['ratio_ci_high'] = np.quantile(ratios, [0.025, 0.975])
    return row

def event_study(daily: pd.DataFrame, event: str) -> pd.DataFrame:
    """Descriptive quarterly event bins vs final pre-event quarter, not a DiD ATT."""
    d = daily.copy()
    t = pd.Timestamp(event)
    months = (d.date_local.dt.year - t.year) * 12 + d.date_local.dt.month - t.month
    d['event_bin'] = np.floor_divide(months, 3).clip(-8, 7).astype(int)
    d['sin_year'] = np.sin(2 * np.pi * d.date_local.dt.dayofyear / 365.25)
    d['cos_year'] = np.cos(2 * np.pi * d.date_local.dt.dayofyear / 365.25)
    if -1 not in set(d.loc[d.basis.notna(), 'event_bin']):
        return pd.DataFrame()
    try:
        _, tab, _ = fit_daily_hac('basis ~ C(event_bin, Treatment(reference=-1)) + sin_year + cos_year + C(dow) + ' + continuous_rhs(),
            d)
        tab = tab[tab.term.str.contains('event_bin', regex=False)].copy()
        tab['estimand'] = 'descriptive event-bin difference; no untreated donor control'
        return tab
    except ValueError as error:
        return pd.DataFrame([dict(estimand='event study not estimated', error=str(error))])

def node_analysis(hourly: pd.DataFrame, site: Site, bootstrap_draws=BOOTSTRAP_DRAWS) -> dict:
    hourly = numeric_frame(hourly)
    daily = daily_panel(hourly)
    usable = daily.dropna(subset=['basis', 'temperature_c', 'gas'])
    if any((usable.post.eq(i).sum() < MIN_DAYS_PER_SIDE for i in (0, 1))):
        raise ValueError(f'Need at least {MIN_DAYS_PER_SIDE} complete days on each side.')
    results = []
    models = {}
    meanfit, tab, mean_cov = fit_daily_hac('basis ~ ' + MEAN_RHS, daily)
    tab['outcome'] = 'mean_basis'
    tab['units'] = '$/MWh; post_trend is $/MWh per year'
    results.append(tab)
    models['mean_basis'] = meanfit
    stage_errors = []
    partial = {key: pd.DataFrame() for key in (
        'horizon_effects_df','variance_ratios_df','descriptive_df',
        'event_study_df','sensitivity_df','placebo_df','stage_errors_df')}
    partial.update(hourly_df=hourly, daily_df=daily, coefficients_df=tab.copy(),
        fitted_models=models, hourly_conditional_mean_model=None)
    try:
        stage = 'hourly_conditional_mean'
        h = hourly.copy()
        # Match the hourly first-stage sample to the declared specification.
        h.loc[~h.estimation_valid.eq(1), 'basis'] = np.nan
        formula = 'basis ~ post + trend_years + post_trend + C(hour) + C(dow) + C(month) + ' + continuous_rhs()
        yh, xh = patsy.dmatrices(formula, h, return_type='dataframe', NA_action='drop')
        verify_required_controls(xh, formula, stage='hourly_mean')
        if np.linalg.matrix_rank(xh.to_numpy()) < xh.shape[1]:
            raise ValueError('Hourly conditional-mean model is rank deficient.')
        hourlyfit = sm.OLS(yh.iloc[:, 0], xh).fit()
        h['residual'] = np.nan
        h.loc[xh.index, 'residual'] = hourlyfit.resid
        h['residual_m2'] = h.residual ** 2
        h['positive_semivariance'] = h.residual.clip(lower=0) ** 2
        h['negative_semivariance'] = h.residual.clip(upper=0) ** 2
        # Thresholds remain based on ALL paired pre-event raw basis observations,
        # as in the previous analysis, rather than changing with BTC eligibility.
        pre = hourly.loc[hourly.post.eq(0), 'basis'].dropna()
        lo, hi = pre.quantile([0.01, 0.99])
        valid = h.basis.notna()
        h['lower_tail'] = np.where(valid, (h.basis < lo).astype(float), np.nan)
        h['upper_tail'] = np.where(valid, (h.basis > hi).astype(float), np.nan)
        momentcols = ['residual_m2', 'positive_semivariance', 'negative_semivariance', 'lower_tail', 'upper_tail']
        for field in momentcols:
            h.loc[h.residual.isna(), field] = np.nan
        m = h.groupby('date_local')[momentcols].mean().reset_index()
        daily = daily.merge(m, on='date_local', how='left', validate='one_to_one')
        daily.loc[~daily.usable_day, momentcols] = np.nan
        partial.update(hourly_df=h, daily_df=daily, hourly_conditional_mean_model=hourlyfit)
        stage = 'volatility_and_tail_models'
        eps = 1e-08
        for field in momentcols:
            if field in ('lower_tail', 'upper_tail'):
                dependent = field
                units = 'probability (multiply by 100 for percentage points)'
            else:
                dependent = 'log_' + field
                daily[dependent] = np.log(daily[field] + eps)
                units = 'log moment; exp(coef)-1 is proportional change in conditional geometric moment'
            try:
                fit, t, _ = fit_daily_hac(dependent + ' ~ ' + MEAN_RHS, daily)
                t['outcome'], t['units'] = (field, units)
                results.append(t)
                models[field] = fit
            except Exception as error:
                stage_errors.append(dict(stage='model_' + field, message=_redact_message(error)))
                results.append(pd.DataFrame([dict(outcome=field, term='not_estimated', error=str(error))]))
        coef = pd.concat(results, ignore_index=True)
        coef['p_fdr_bh_within_node'] = np.nan
        mask = coef.term.eq('post') & coef.p_value.notna()
        if mask.any():
            coef.loc[mask, 'p_fdr_bh_within_node'] = multipletests(coef.loc[mask, 'p_value'], method='fdr_bh')[1]
        partial['coefficients_df'] = coef
        stage = 'horizon_contrasts'
        contrasts = []
        for months in (3, 6, 12, 24, 36):
            if hourly.hour_utc.max() + pd.Timedelta(hours=1) < site.timestamp('event').tz_convert('UTC') + pd.DateOffset(months=months):
                continue
            beta = meanfit.params
            cov = mean_cov
            v = np.zeros(len(beta))
            v[beta.index.get_loc('post')] = 1
            v[beta.index.get_loc('post_trend')] = months / 12
            est = float(v @ beta)
            se = float(np.sqrt(v @ cov @ v))
            contrasts.append(dict(horizon_months=months,
                    mean_basis_change=est,
                    ci_low=est - 1.95996 * se,
                    ci_high=est + 1.95996 * se,
                    estimand='fitted ITS difference at horizon, holding Bitcoin conditions at the original event reference; not an average or actual-path effect'))
        partial['horizon_effects_df'] = pd.DataFrame(contrasts)
        stage = 'variance_ratio_bootstrap'
        ratios = pd.DataFrame([block_ratio_ci(daily, c, draws=bootstrap_draws) for c in momentcols[:3]])
        partial['variance_ratios_df'] = ratios
        stage = 'descriptive_statistics'
        desc = []
        for post in (0, 1):
            z = h.loc[h.post.eq(post), 'basis'].dropna()
            desc.append(dict(period='post' if post else 'pre',
                    n_hours=len(z),
                    mean=z.mean(),
                    std=z.std(),
                    q01=z.quantile(0.01),
                    q05=z.quantile(0.05),
                    median=z.median(),
                    q95=z.quantile(0.95),
                    q99=z.quantile(0.99),
                    min=z.min(),
                    max=z.max(),
                    lower_fixed_threshold=lo,
                    upper_fixed_threshold=hi))
        partial['descriptive_df'] = pd.DataFrame(desc)
        stage = 'robustness_checks'
        sensitivity = []
        for years in (1, 2, 3):
            e = pd.Timestamp(site.event)
            left, right = (e - pd.DateOffset(years=years), e + pd.DateOffset(years=years))
            if daily.date_local.min() > left or daily.date_local.max() + pd.Timedelta(days=1) < right:
                continue
            z = daily[daily.date_local.between(left, right, inclusive='left')]
            if all((z.loc[z.post.eq(i), 'basis'].notna().sum() >= MIN_DAYS_PER_SIDE for i in (0, 1))):
                try:
                    _, t, _ = fit_daily_hac('basis ~ ' + MEAN_RHS, z)
                    t = t[t.term.isin(['post', 'post_trend'])].copy()
                    t['check'] = f'symmetric_{years}y'
                    sensitivity.append(t)
                except ValueError:
                    pass
        for newdate in site.alternatives:
            z = daily.copy()
            e = pd.Timestamp(newdate)
            z['post'] = (z.date_local >= e).astype(int)
            z['trend_years'] = (z.date_local - e).dt.days / 365.25
            z['post_trend'] = z.post * z.trend_years
            try:
                _, t, _ = fit_daily_hac('basis ~ ' + MEAN_RHS, z)
                t = t[t.term.isin(['post', 'post_trend'])].copy()
                t['check'] = 'alternative_nominal_date_' + newdate
                sensitivity.append(t)
            except Exception as error:
                stage_errors.append(dict(stage=stage, message=_redact_message(error)))
        placebo = []
        z = daily[daily.date_local < pd.Timestamp(site.event)].copy()
        e = pd.Timestamp(site.event) - pd.DateOffset(years=1)
        z['post'] = (z.date_local >= e).astype(int)
        z['trend_years'] = (z.date_local - e).dt.days / 365.25
        z['post_trend'] = z.post * z.trend_years
        if all((z.loc[z.post.eq(i), 'basis'].notna().sum() >= MIN_DAYS_PER_SIDE for i in (0, 1))):
            try:
                _, t, _ = fit_daily_hac('basis ~ ' + MEAN_RHS, z)
                t = t[t.term.isin(['post', 'post_trend'])].copy()
                t['placebo_date'] = str(e.date())
                placebo.append(t)
            except Exception as error:
                stage_errors.append(dict(stage=stage, message=_redact_message(error)))
        winsor = daily.copy()
        q = daily.loc[daily.post.eq(0), 'basis'].quantile([0.005, 0.995])
        winsor['basis'] = winsor.basis.clip(q.iloc[0], q.iloc[1])
        try:
            _, t, _ = fit_daily_hac('basis ~ ' + MEAN_RHS, winsor)
            t = t[t.term.isin(['post', 'post_trend'])].copy()
            t['check'] = 'pre_threshold_winsor_0.5pct'
            sensitivity.append(t)
        except Exception as error:
            stage_errors.append(dict(stage=stage, message=_redact_message(error)))
        partial['sensitivity_df'] = pd.concat(sensitivity, ignore_index=True) if sensitivity else pd.DataFrame()
        partial['placebo_df'] = pd.concat(placebo, ignore_index=True) if placebo else pd.DataFrame()
        stage = 'event_time_diagnostics'
        return dict(stage_errors_df=pd.DataFrame(stage_errors), hourly_df=h,
            daily_df=daily,
            coefficients_df=coef,
            horizon_effects_df=pd.DataFrame(contrasts),
            variance_ratios_df=ratios,
            descriptive_df=pd.DataFrame(desc),
            event_study_df=event_study(daily,
                site.event),
            sensitivity_df=pd.concat(sensitivity,
                ignore_index=True) if sensitivity else pd.DataFrame(),
            placebo_df=pd.concat(placebo,
                ignore_index=True) if placebo else pd.DataFrame(),
            fitted_models=models,
            hourly_conditional_mean_model=hourlyfit)
    except Exception as error:
        stage_errors.append(dict(stage=stage, message=_redact_message(error),
            exception_type=type(error).__name__))
        partial['stage_errors_df'] = pd.DataFrame(stage_errors)
        return partial


def analyze_site(prices, weather, gas, site: Site, bootstrap_draws=BOOTSTRAP_DRAWS):
    nodes = [c for c in prices.location.unique() if c != site.benchmark]
    ordered = [n for n in site.nodes if n in nodes] + [n for n in nodes if n not in site.nodes]
    aliases = alias_table(prices, ordered)
    excluded = set(aliases.loc[aliases.exact_alias, 'node_b']) if not aliases.empty else set()
    outputs = {}
    coverage = []
    errors = []
    inspection_panels = []
    for node in ordered:
        global _ACTIVE_SITE, _ACTIVE_NODE
        _ACTIVE_SITE, _ACTIVE_NODE = site.key, node
        try:
            hourly, cov = build_panel(prices, weather, gas, site, node)
            coverage.append(cov)
            inspection_panels.append(hourly)
            if node in excluded:
                errors.append(dict(node=node, status='excluded_exact_alias', message='Not an independent robustness node'))
                continue
            if cov.complete_controls_coverage.isna().any() or cov.complete_controls_coverage.min() < MIN_HOURLY_COVERAGE:
                errors.append(dict(node=node,
                        status='coverage_failed',
                        message='Pre/post complete-control coverage below threshold: ' + cov[['period','paired_coverage','weather_coverage','gas_coverage','complete_controls_coverage']].to_dict('records').__str__()))
                continue
            outputs[node] = node_analysis(hourly, site, bootstrap_draws)
            for item in outputs[node].get('stage_errors_df', pd.DataFrame()).to_dict('records'):
                errors.append(dict(node=node, status='partial_model_failure', **item))
        except Exception as error:
            errors.append(dict(node=node, status='analysis_failed', message=_redact_message(error)))
    combined = {}
    keys = ['hourly_df',
        'daily_df',
        'coefficients_df',
        'horizon_effects_df',
        'variance_ratios_df',
        'descriptive_df',
        'event_study_df',
        'sensitivity_df',
        'placebo_df']
    for key in keys:
        frames = []
        for node, out in outputs.items():
            f = out[key].copy()
            f['node'] = node
            f['site_key'] = site.key
            f['ai_tier'] = site.ai_tier
            f['evidence_class'] = 'mining/legacy proxy diagnostic; not AI evidence'
            frames.append(f)
        combined[key] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined['inspection_hourly_df'] = pd.concat(inspection_panels, ignore_index=True) if inspection_panels else pd.DataFrame()
    combined.update(node_outputs=outputs,
        aliases_df=aliases,
        coverage_df=pd.concat(coverage,
            ignore_index=True) if coverage else pd.DataFrame(),
        analysis_errors_df=pd.DataFrame(errors))
    return combined

# %% Descriptive price-only comparisons -- independent of weather/gas availability

def _bootstrap_hourly_mean_variance(panel, field, draws, block_days, seed):
    """Resample calendar-day blocks, retaining hourly weighting and DST day length.

    Sufficient statistics are computed from the observed hourly values. Missing
    calendar days remain in the block grid (zero counts), not interpolated.
    The bootstrap is within each regime and descriptive, not a causal test.
    """
    rng = np.random.default_rng(seed)
    boot = []
    valid_days = []
    for period in (0, 1):
        z = panel.loc[panel.post.eq(period), ['date_local', field]].copy()
        finite = z[field].dropna()
        center = float(finite.mean()) if len(finite) else 0.0
        z['_dev'] = z[field] - center
        z['_dev2'] = z['_dev'] ** 2
        moments = z.groupby('date_local').agg(
            n=(field, 'count'), total=('_dev', 'sum'), total2=('_dev2', 'sum'))
        if moments.empty:
            return None, 'No observations on one side.'
        grid = pd.date_range(moments.index.min(), moments.index.max(), freq='D')
        moments = moments.reindex(grid, fill_value=0.0)
        nday = len(moments)
        valid_days.append(int(moments.n.gt(0).sum()))
        if nday < block_days or valid_days[-1] < MIN_DAYS_PER_SIDE:
            return None, f'Need {MIN_DAYS_PER_SIDE} observed calendar days per side for intervals.'
        # Add block sufficient statistics without constructing enormous hourly arrays.
        prefix = np.vstack([np.zeros(3), np.cumsum(moments.to_numpy(dtype=float), axis=0)])
        nblocks = math.ceil(nday / block_days)
        starts = rng.integers(0, nday - block_days + 1, size=(draws, nblocks))
        lengths = np.full(nblocks, block_days, dtype=int)
        lengths[-1] = nday - block_days * (nblocks - 1)
        agg = (prefix[starts + lengths[None, :]] - prefix[starts]).sum(axis=1)
        count, total, total2 = agg.T
        means = np.full(draws, np.nan)
        variances = np.full(draws, np.nan)
        good = count > 1
        means[good] = center + total[good] / count[good]
        variances[good] = np.maximum((total2[good] - total[good] ** 2 / count[good]) / (count[good] - 1), 0.0)
        boot.append((means, variances))
    diff = boot[1][0] - boot[0][0]
    vdiff = boot[1][1] - boot[0][1]
    vr = np.divide(boot[1][1], boot[0][1], out=np.full(draws, np.nan),
                   where=boot[0][1] > 1e-8)
    result = {'bootstrap_draws': draws, 'bootstrap_block_days': block_days,
              'pre_observed_days': valid_days[0], 'post_observed_days': valid_days[1]}
    for key, values in [('mean_difference', diff), ('variance_difference', vdiff), ('variance_ratio', vr)]:
        good = np.isfinite(values)
        result[key + '_valid_draws'] = int(good.sum())
        low, high = (np.quantile(values[good], [0.025, 0.975])
                     if good.sum() >= 0.9 * draws else (np.nan, np.nan))
        result[key + '_ci_low'] = low
        result[key + '_ci_high'] = high
    return result, 'estimated; descriptive moving-calendar-block percentile intervals'


def describe_price_changes(prices, site, node, draws=BOOTSTRAP_DRAWS):
    h = build_price_panel(prices, site, node)
    # All three outcomes use the SAME paired-price hours, not different missing samples.
    pair = h[['local_price', 'benchmark_price']].notna().all(axis=1)
    h.loc[~pair, ['local_price', 'benchmark_price', 'basis']] = np.nan
    desc, changes = [], []
    for field in ('local_price', 'benchmark_price', 'basis'):
        data = {}
        for post in (0, 1):
            z = h.loc[h.post.eq(post), field]
            x = z.dropna()
            stats_row = dict(site_key=site.key, node=node, benchmark=site.benchmark,
                outcome=field, period='post' if post else 'pre',
                expected_hours=len(z), n_hours=len(x),
                paired_price_coverage=len(x) / len(z) if len(z) else np.nan,
                mean=x.mean(), variance=x.var(ddof=1), std=x.std(ddof=1),
                median=x.median(), q01=x.quantile(0.01), q05=x.quantile(0.05),
                q95=x.quantile(0.95), q99=x.quantile(0.99), minimum=x.min(), maximum=x.max(),
                negative_price_share=x.lt(0).mean() if len(x) else np.nan,
                nominal_event_date=site.event,
                sample='paired observed hourly prices; no weather/gas screen',
                ai_tier=site.ai_tier, causal_interpretation=False)
            desc.append(stats_row)
            data[post] = stats_row
        a, b = data[0], data[1]
        ratio = b['variance'] / a['variance'] if a['variance'] > 1e-8 else np.nan
        row = dict(site_key=site.key, node=node, benchmark=site.benchmark, outcome=field,
            pre_hours=a['n_hours'], post_hours=b['n_hours'],
            pre_mean=a['mean'], post_mean=b['mean'], mean_difference=b['mean']-a['mean'],
            pre_variance=a['variance'], post_variance=b['variance'],
            variance_difference=b['variance']-a['variance'], variance_ratio=ratio,
            variance_percent_change=100*(ratio-1),
            near_zero_pre_variance=bool(a['variance'] <= 1e-8),
            units_mean='$/MWh', units_variance='($/MWh)^2',
            estimand='raw post-minus-pre mean / raw post-to-pre hourly variance ratio',
            nominal_event_date=site.event, ai_tier=site.ai_tier,
            inference='descriptive; not an adjusted estimate or causal effect')
        for key in ('mean_difference','variance_difference','variance_ratio'):
            row[key+'_ci_low'] = np.nan
            row[key+'_ci_high'] = np.nan
        sufficient = all(np.isfinite(data[i]['paired_price_coverage']) and
                         data[i]['paired_price_coverage'] >= MIN_HOURLY_COVERAGE for i in (0,1))
        if not sufficient:
            row['bootstrap_status'] = 'not estimated: paired coverage below unchanged threshold'
        elif draws < 20:
            row['bootstrap_status'] = 'not estimated: fewer than 20 bootstrap draws configured'
        else:
            seed = int(hashlib.sha256((site.key+'|'+node+'|'+field).encode()).hexdigest()[:8],16)
            try:
                ci, status = _bootstrap_hourly_mean_variance(h, field, draws,
                    BOOTSTRAP_BLOCK_DAYS, seed)
                if ci:
                    row.update(ci)
                row['bootstrap_status'] = status
            except Exception as exc:
                row['bootstrap_status'] = 'failed: ' + _redact_message(exc)
        changes.append(row)
    return h, pd.DataFrame(desc), pd.DataFrame(changes)


# %% Strict local checkpoint loading -- no network/credential/provider code

def _cache_identity_token(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _checkpoint_identity(site):
    return dict(site=asdict(site), source=NYISO_SOURCE if site.iso == 'NYISO' else site.dataset,
                manual_pjm_routes=list(PJM_MANUAL_ROUTES) if site.iso == 'PJM' else [],
                input_format_version=1, synthetic_test_only=False)


def _read_cached_csv(path):
    path = Path(path)
    meta = path.with_name(path.name.removesuffix('.csv.gz') + '.schema.json')
    if not meta.exists():
        raise ValueError('Missing dtype sidecar: ' + str(meta))
    schema = json.loads(meta.read_text(encoding='utf-8'))
    strings = {c: 'string' for c, dtype in schema.items()
               if dtype in ('object','str') or dtype.startswith('string')}
    try:
        frame = pd.read_csv(path, dtype=strings, float_precision='round_trip')
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(schema))
    for col, dtype in schema.items():
        if dtype.startswith('datetime64'):
            if ', ' in dtype:
                timezone = dtype.split(', ',1)[1].rstrip(']')
                frame[col] = pd.to_datetime(frame[col], utc=True, errors='raise', format='mixed').dt.tz_convert(timezone)
            else:
                frame[col] = pd.to_datetime(frame[col], errors='raise', format='mixed')
        elif dtype in ('bool','boolean'):
            if not pd.api.types.is_bool_dtype(frame[col]):
                frame[col] = frame[col].astype('string').map({'True':True, 'False':False})
            frame[col] = frame[col].astype(dtype)
        elif dtype not in ('object','str') and not dtype.startswith('string'):
            frame[col] = frame[col].astype(dtype)
    return frame


def _is_synthetic(frame):
    if frame.attrs.get('synthetic_test_only', False):
        return True
    if 'synthetic_test_only' in frame:
        return frame.synthetic_test_only.astype('string').str.lower().isin(['true','1','1.0']).any()
    return False


def load_inputs(site, workspace=None):
    """Load exactly matched real checkpoints; never guess across different studies."""
    identity = _checkpoint_identity(site)
    folder = Path(CHECKPOINT_DIRECTORY_OVERRIDES.get(site.key,
        Path(CACHE_DIR) / 'site_inputs' / (site.key + '_' + _cache_identity_token(identity))))
    manifest = folder / 'manifest.json'
    if manifest.exists():
        info = json.loads(manifest.read_text(encoding='utf-8'))
        if info.get('synthetic_test_only') is not False:
            raise ValueError('Refusing synthetic/unclassified checkpoint: ' + str(folder))
        expected = json.loads(json.dumps(identity, default=str))
        if info.get('identity') != expected:
            raise ValueError('Checkpoint settings differ. Match SITE_OVERRIDES/NYISO_SOURCE/PJM_MANUAL_ROUTES; no data were mixed.')
        frames = {}
        for name in ('lmp_hourly_df','weather_hourly_df','gas_daily_df','price_mapping_df'):
            path = folder / (name + '.csv.gz')
            if not path.exists() and name == 'lmp_hourly_df':
                raise FileNotFoundError('Price checkpoint missing: ' + str(path))
            frames[name] = _read_cached_csv(path) if path.exists() else pd.DataFrame()
            if _is_synthetic(frames[name]):
                raise ValueError('Refusing a synthetic-tagged input frame.')
        return frames, str(folder), bool(info.get('download_complete', False))
    if ALLOW_WORKSPACE_FALLBACK and workspace:
        settings = workspace.get('run_settings_df')
        registry = workspace.get('treatment_registry_df')
        results = workspace.get('site_results', {})
        if (isinstance(settings, pd.DataFrame) and not settings.empty and
            'mode' in settings and settings['mode'].isin(['full','analyze_cached']).all() and
            not _is_synthetic(settings) and isinstance(registry,pd.DataFrame) and 'key' in registry):
            rows = registry.loc[registry['key'].eq(site.key)]
            fields = ('start','event','end','timezone','benchmark')
            matched = len(rows) == 1 and all(str(rows.iloc[0][f]) == str(getattr(site,f)) for f in fields)
            source = results.get(site.key,{}) if isinstance(results,dict) else {}
            if matched and isinstance(source,dict) and isinstance(source.get('lmp_hourly_df'),pd.DataFrame):
                names = ('lmp_hourly_df','weather_hourly_df','gas_daily_df','price_mapping_df')
                frames = {name:source.get(name,pd.DataFrame()).copy() for name in names}
                if any(_is_synthetic(d) for d in frames.values()):
                    raise ValueError('Refusing synthetic workspace inputs.')
                return frames, 'verified real-run frames still in Spyder', False
    alternatives = sorted(str(p) for p in (Path(CACHE_DIR)/'site_inputs').glob(site.key+'_*'))
    detail = ('\nOther directories found (not selected automatically):\n'+'\n'.join(alternatives)) if alternatives else ''
    raise FileNotFoundError('No matching real input checkpoint. Expected:\n'+str(folder)+detail+
        '\nSet PROJECT_FOLDER/CACHE_DIR to the original data folder. No download was attempted.')


def prepare_inputs(frames, site):
    result, diagnostic = {}, []
    for name, frame in frames.items():
        out = numeric_frame(frame)
        for col in frame:
            if str(frame[col].dtype) != str(out[col].dtype) or frame[col].isna().any():
                diagnostic.append(dict(site_key=site.key, frame=name, column=col,
                    original_dtype=str(frame[col].dtype), analysis_dtype=str(out[col].dtype),
                    original_missing=int(frame[col].isna().sum()),
                    analysis_missing=int(out[col].isna().sum()),
                    action='numeric representation only; no missing-value imputation'))
        result[name] = out
    return result, pd.DataFrame(diagnostic)


def _stack(outputs, key):
    frames = []
    for site_key, out in outputs.items():
        frame = out.get(key)
        if isinstance(frame,pd.DataFrame) and not frame.empty:
            frame = frame.copy()
            frame['site_key'] = site_key
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_outputs(results, summary, old_errors, old_summary):
    output = dict(reanalyzed_site_results=results, analysis_run_summary_df=pd.DataFrame(summary),
        previous_analysis_errors_df=old_errors, previous_run_summary_df=old_summary)
    for key in ('coefficients_df','coverage_df','aliases_df','variance_ratios_df','horizon_effects_df',
                'event_study_df','sensitivity_df','placebo_df','analysis_errors_df','hourly_df','daily_df',
                'inspection_hourly_df'):
        output['all_'+key] = _stack(results,key)
    for name in ('raw_prepost_summary_df','raw_mean_variance_changes_df','dtype_diagnostics_df','raw_analysis_errors_df'):
        output[name] = _stack(results,name)
    coef = output['all_coefficients_df']
    if not coef.empty:
        coef['p_fdr_bh_all_primary_tests'] = np.nan
        mask = coef.term.eq('post') & coef.p_value.notna()
        if mask.any():
            coef.loc[mask,'p_fdr_bh_all_primary_tests'] = multipletests(coef.loc[mask,'p_value'],method='fdr_bh')[1]
        output['key_results_df'] = coef.loc[coef.term.isin(['post','post_trend'])].copy()
    else:
        output['key_results_df'] = pd.DataFrame()
    for site_key, data in results.items():
        for name in ('hourly_df','daily_df','inspection_hourly_df','raw_price_hourly_df','coverage_df'):
            output[site_key+'_'+name] = data.get(name,pd.DataFrame())
    output['adjusted_mean_results_df'] = (output['key_results_df'].loc[
        output['key_results_df'].outcome.eq('mean_basis')].copy()
        if not output['key_results_df'].empty else pd.DataFrame())
    output['adjusted_volatility_results_df'] = (output['key_results_df'].loc[
        output['key_results_df'].outcome.isin(['residual_m2','positive_semivariance','negative_semivariance'])].copy()
        if not output['key_results_df'].empty else pd.DataFrame())
    return output


# %% Public Bitcoin data and mandatory design-matrix checks

def continuous_rhs():
    parts = ['temperature_c', 'temp2']
    if ACTIVE_SPECIFICATION != 'weather_only_common_sample':
        parts.append('gas')
    column = SPEC_CONTROLS[ACTIVE_SPECIFICATION]
    if column is not None:
        parts += [column, 'post:' + column]
    return ' + '.join(parts)


def activate_specification(name):
    global ACTIVE_SPECIFICATION, MEAN_RHS
    if name not in SPEC_CONTROLS:
        raise ValueError('Unknown specification: ' + str(name))
    ACTIVE_SPECIFICATION = name
    MEAN_RHS = ('post + trend_years + post_trend + C(month) + C(dow) + '
                + continuous_rhs())


def verify_required_controls(design, formula, stage):
    """Test the ACTUAL Patsy design matrix, not just a configuration label."""
    column = SPEC_CONTROLS[ACTIVE_SPECIFICATION]
    expected = ['temperature_c', 'temp2']
    if ACTIVE_SPECIFICATION != 'weather_only_common_sample':
        expected.append('gas')
    if column is not None:
        expected += [column, 'post:' + column]
    missing = [c for c in expected if c not in design.columns]
    _CONTROL_AUDIT_ROWS.append(dict(
        site_key=_ACTIVE_SITE, node=_ACTIVE_NODE,
        specification=ACTIVE_SPECIFICATION, stage=stage, formula=formula,
        n_observations=len(design), expected_controls=' | '.join(expected),
        actual_columns=' | '.join(map(str, design.columns)),
        bitcoin_required=column is not None,
        required_controls_present=not missing))
    if missing:
        raise ValueError('Required controls missing from fitted design: ' + ', '.join(missing))


def parse_coinmetrics_csv(payload):
    """Validate exact published metric names. Reject conflicting daily revisions."""
    from io import BytesIO
    raw = pd.read_csv(BytesIO(payload), low_memory=False)
    missing = sorted(set(BTC_METRICS) - set(raw.columns))
    if missing:
        raise ValueError('Coin Metrics archive missing required columns: ' + str(missing))
    keep = list(BTC_METRICS) + (['DiffMean'] if 'DiffMean' in raw else [])
    d = raw[keep].copy()
    stamps = pd.to_datetime(d['time'], utc=True, errors='coerce', format='mixed')
    if stamps.isna().any():
        raise ValueError('Unparseable Coin Metrics daily date; source rejected.')
    if not stamps.eq(stamps.dt.normalize()).all():
        raise ValueError('Expected daily midnight-UTC metric labels.')
    d['source_date_utc'] = stamps.dt.tz_localize(None).dt.normalize()
    d = d.drop(columns='time')
    for c in d.columns.drop('source_date_utc'):
        d[c] = pd.to_numeric(d[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    values = d.columns.drop('source_date_utc')
    if d.groupby('source_date_utc')[values].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Conflicting BTC values for a date; do not average revisions.')
    d = d.drop_duplicates('source_date_utc').sort_values('source_date_utc')
    for c in ('PriceUSD', 'HashRate', 'BlkCnt', 'IssTotNtv', 'IssTotUSD'):
        d[c] = d[c].where(d[c].gt(0))
    d['FeeTotNtv'] = d.FeeTotNtv.where(d.FeeTotNtv.ge(0))
    if 'DiffMean' in d:
        d['DiffMean'] = d.DiffMean.where(d.DiffMean.gt(0))
    if d.empty:
        raise ValueError('Coin Metrics source is empty.')
    return d.reset_index(drop=True)


def derive_bitcoin_daily(source, start, end):
    """Calendar-day lag, not previous-available-row lag. Missing BTC stays missing."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end <= start or BTC_LAG_CALENDAR_DAYS != 1:
        raise ValueError('This restored specification requires an increasing window and exactly one calendar-day BTC lag.')
    grid = pd.date_range(start - pd.Timedelta(days=14 + BTC_LAG_CALENDAR_DAYS),
                         end - pd.Timedelta(days=1), freq='D')
    d = source.set_index('source_date_utc').reindex(grid)
    d.index.name = 'source_date_utc'
    d['btc_price_usd'] = d.PriceUSD
    d['btc_subsidy_btc_per_block'] = d.IssTotNtv / d.BlkCnt
    d['btc_fees_btc_per_block'] = d.FeeTotNtv / d.BlkCnt
    d['btc_network_revenue_usd_day'] = d.IssTotUSD + d.FeeTotNtv * d.PriceUSD
    d['btc_hashrate_ph_s'] = d.HashRate / 1000.0
    d['btc_hashprice_usd_per_ph_day'] = d.btc_network_revenue_usd_day / d.btc_hashrate_ph_s
    d['btc_hashprice_usd_per_ph_day'] = d.btc_hashprice_usd_per_ph_day.where(
        d[list(BTC_METRICS)[1:]].notna().all(axis=1) & d.btc_hashprice_usd_per_ph_day.gt(0))
    # Replicate the original descriptive ASIC scenarios; these do not enter a fit.
    for efficiency in (30, 60, 100):
        d[f'btc_energy_only_break_even_{efficiency}jth_usd_mwh'] = (
            d.btc_hashprice_usd_per_ph_day / (0.024 * efficiency))
    d['btc_log_price_lag1d'] = np.log(d.btc_price_usd.shift(BTC_LAG_CALENDAR_DAYS))
    d['btc_log_hashprice_lag1d'] = np.log(d.btc_hashprice_usd_per_ph_day.shift(BTC_LAG_CALENDAR_DAYS))
    d['btc_source_date_for_control'] = d.index - pd.Timedelta(days=BTC_LAG_CALENDAR_DAYS)
    d = d.reset_index().rename(columns={'source_date_utc':'date_local'})
    # date_local is a join label: local day d receives source UTC label d-1.
    d = d[d.date_local.ge(start) & d.date_local.lt(end)].reset_index(drop=True)
    return d


def load_bitcoin_daily(start, end):
    """One public CSV download at most per run; pinned snapshot reused thereafter."""
    global _SOURCE_RECORD
    import time as _time
    from datetime import datetime, timezone
    from urllib.request import Request, urlopen
    cache = Path(BTC_CACHE_DIR)
    snapshot = cache / 'coinmetrics_btc_snapshot.csv'
    metadata = cache / 'coinmetrics_btc_snapshot.json'
    source_type = ''
    acquired = None
    if BTC_LOCAL_CSV:
        source_path = Path(BTC_LOCAL_CSV).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError('BTC_LOCAL_CSV does not exist: ' + str(source_path))
        payload = source_path.read_bytes()
        source_type = 'explicit_local_coinmetrics_csv'
    elif snapshot.exists() and not BTC_REFRESH:
        source_path = snapshot
        payload = snapshot.read_bytes()
        source_type = 'pinned_local_snapshot'
        if metadata.exists():
            meta = json.loads(metadata.read_text(encoding='utf-8'))
            if meta.get('sha256') != hashlib.sha256(payload).hexdigest():
                raise ValueError('Pinned BTC snapshot hash differs from its manifest.')
            acquired = meta.get('downloaded_utc')
    else:
        if not BTC_ALLOW_DOWNLOAD:
            raise FileNotFoundError('No BTC snapshot. Provide BTC_LOCAL_CSV or enable BTC_ALLOW_DOWNLOAD.')
        last_error = None
        payload = None
        for attempt in range(BTC_DOWNLOAD_ATTEMPTS):
            try:
                request = Request(BTC_ARCHIVE_URL, headers={
                    'User-Agent':'Academic-LMP-Bitcoin-Control-Repair/1.0',
                    'Accept':'text/csv,application/octet-stream'})
                with urlopen(request, timeout=BTC_DOWNLOAD_TIMEOUT) as response:
                    payload = response.read(BTC_DOWNLOAD_MAX_BYTES + 1)
                if len(payload) > BTC_DOWNLOAD_MAX_BYTES:
                    raise ValueError('BTC source exceeds configured byte cap.')
                parse_coinmetrics_csv(payload)  # Never pin an HTML/error document.
                break
            except Exception as exc:
                last_error = exc
                payload = None
                if attempt + 1 < BTC_DOWNLOAD_ATTEMPTS:
                    _time.sleep(2)
        if payload is None:
            raise RuntimeError('BTC download failed; no Bitcoin-adjusted estimates were made. '
                               'No AWS/GridStatus calls were made. ' + str(last_error))
        acquired = datetime.now(timezone.utc).isoformat()
        cache.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(payload).hexdigest()
        # Keep old snapshots when explicitly refreshing, without overwriting evidence.
        version = cache / ('coinmetrics_btc_' + digest[:16] + '.csv')
        if not version.exists():
            version.write_bytes(payload)
        part = snapshot.with_suffix('.csv.part')
        part.write_bytes(payload)
        part.replace(snapshot)
        metadata.write_text(json.dumps(dict(sha256=digest, downloaded_utc=acquired,
             source_url=BTC_ARCHIVE_URL, license='Coin Metrics Community CC BY-NC 4.0'), indent=2), encoding='utf-8')
        source_path = snapshot
        source_type = 'downloaded_public_coinmetrics_csv'
    if len(payload) > BTC_DOWNLOAD_MAX_BYTES:
        raise ValueError('BTC local file exceeds the configured byte cap.')
    source = parse_coinmetrics_csv(payload)
    _SOURCE_RECORD = dict(source=source_type, path=str(source_path), source_url=BTC_ARCHIVE_URL,
        sha256=hashlib.sha256(payload).hexdigest(), downloaded_utc=acquired,
        min_source_date=str(source.source_date_utc.min().date()),
        max_source_date=str(source.source_date_utc.max().date()),
        raw_rows=len(source), lag_calendar_days=BTC_LAG_CALENDAR_DAYS,
        price_metric='PriceUSD', hash_rate_unit='TH/s',
        hashprice_formula='(IssTotUSD + FeeTotNtv * PriceUSD) / (HashRate / 1000)',
        no_btc_imputation=True, point_in_time_publication_vintage=False)
    return derive_bitcoin_daily(source, start, end)


def attach_bitcoin_controls(hourly, site):
    if _MINING_DAILY is None or _MINING_DAILY.empty:
        raise ValueError('Bitcoin data absent. Refusing to omit required controls.')
    columns = ['date_local', 'btc_log_price_lag1d', 'btc_log_hashprice_lag1d',
               'btc_source_date_for_control']
    h = hourly.merge(_MINING_DAILY[columns], on='date_local', how='left', validate='many_to_one')
    for label in ('price', 'hashprice'):
        raw = 'btc_log_' + label + '_lag1d'
        eligible = _MINING_DAILY.loc[_MINING_DAILY.date_local.le(pd.Timestamp(site.event))
                                    & _MINING_DAILY[raw].notna()].sort_values('date_local')
        if eligible.empty:
            raise ValueError('No pre/event-reference Bitcoin value for ' + label)
        row = eligible.iloc[-1]
        center = float(row[raw])
        h[raw + '_centered'] = h[raw] - center
        _BTC_CENTER_ROWS.append(dict(site_key=site.key, nominal_event=site.event,
            variable=raw, reference_log_value=center, reference_level=float(np.exp(center)),
            reference_local_date=str(row.date_local.date()),
            source_utc_date=str(row.btc_source_date_for_control.date()),
            anchor_age_days=int((pd.Timestamp(site.event)-row.date_local).days)))
    base = h[['basis', 'temperature_c', 'gas']].notna().all(axis=1)
    common = base & h[['btc_log_price_lag1d', 'btc_log_hashprice_lag1d']].notna().all(axis=1)
    h['base_weather_gas_valid'] = base
    h['bitcoin_common_valid'] = common
    h['estimation_valid'] = base if ACTIVE_SPECIFICATION == 'weather_gas_original_sample' else common
    # No current/future date label may enter any fitted Bitcoin value.
    valid = h[['btc_log_price_lag1d', 'btc_log_hashprice_lag1d']].notna().any(axis=1)
    if not (h.loc[valid, 'btc_source_date_for_control'] < h.loc[valid, 'date_local']).all():
        raise ValueError('Bitcoin date alignment is not strictly lagged.')
    return h


def _tag(frame, site_key, specification):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    f = frame.copy()
    f['site_key'], f['specification'] = site_key, specification
    f['bitcoin_adjusted'] = SPEC_CONTROLS[specification] is not None
    f['primary_specification'] = specification == PRIMARY_SPECIFICATION
    return f


def _combine(frames):
    return pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()


def _export_crypto_outputs(out):
    if not SAVE_BTC_RESULTS:
        return
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = Path(BTC_RESULTS_DIR) / stamp
    folder.mkdir(parents=True, exist_ok=False)
    for name, value in out.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(folder / (name + '.csv.gz'), index=False, compression='gzip')
    code_path = globals().get('__file__')
    code_sha = hashlib.sha256(Path(code_path).read_bytes()).hexdigest() if code_path and Path(code_path).is_file() else None
    manifest = dict(primary_specification=PRIMARY_SPECIFICATION, specifications=SPECIFICATIONS,
        code_sha256=code_sha, btc_source=_SOURCE_RECORD,
        inputs_read_only=True, results_are_new_estimates=True,
        conditional_not_causal=True,
        prior_pdf_status='omitted-Bitcoin specification; not updated by this script')
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    print('New results saved separately:', folder)
    out['crypto_results_folder'] = str(folder)


def main(namespace=None):
    """Restore BTC controls without changing electricity inputs or using quotas."""
    global _MINING_DAILY, _ACTIVE_SITE, _ACTIVE_NODE
    namespace = globals() if namespace is None else namespace
    workspace = dict(namespace)  # Capture old input workspace before clearing NEW tables.
    table_names = ('crypto_run_summary_df','crypto_key_results_df','crypto_specification_comparison_df',
        'crypto_control_audit_df','crypto_centering_df','crypto_coverage_df','crypto_source_status_df',
        'crypto_all_coefficients_df','crypto_variance_ratios_df','crypto_event_study_df',
        'crypto_sensitivity_df','crypto_placebo_df','crypto_horizon_effects_df','crypto_analysis_errors_df',
        'crypto_raw_mean_variance_changes_df','crypto_raw_prepost_summary_df','crypto_zero_moment_df',
        'crypto_common_sample_check_df','crypto_daily_primary_df','crypto_hourly_primary_df',
        'crypto_mining_economics_daily_df','crypto_input_provenance_df')
    out = {n: pd.DataFrame() for n in table_names}
    namespace.update(out)
    _CONTROL_AUDIT_ROWS.clear()
    _BTC_CENTER_ROWS.clear()
    print('\nBITCOIN CONTROL REPAIR | old electricity inputs read only')
    print('Primary:', PRIMARY_SPECIFICATION, '| separate price-only benchmark included')
    print('Old PDF results omit BTC. This run writes NEW, separately labelled estimates.')
    if PRIMARY_SPECIFICATION not in SPECIFICATIONS or SPEC_CONTROLS.get(PRIMARY_SPECIFICATION) is None:
        raise ValueError('Primary specification must actually contain Bitcoin controls.')
    prepared, summaries, provenance, errors = {}, [], [], []
    for key in RUN_SITES:
        if key not in SITES:
            raise ValueError('Unknown site: ' + key)
        site = replace(SITES[key], **SITE_OVERRIDES.get(key, {}))
        try:
            frames, source_path, complete = load_inputs(site, workspace)
            frames, _ = prepare_inputs(frames, site)
            prepared[key] = (site, frames)
            provenance.append(dict(site_key=key, input_source=source_path,
                download_complete_recorded=complete, start=site.start, end_exclusive=site.end,
                nominal_event=site.event, benchmark=site.benchmark,
                price_rows=len(frames['lmp_hourly_df']), input_files_modified=False))
        except Exception as exc:
            summaries.append(dict(site_key=key,specification='all',status='input_blocked',
                                  message=_redact_message(exc)))
    out['crypto_input_provenance_df'] = pd.DataFrame(provenance)
    if not prepared:
        out['crypto_run_summary_df'] = pd.DataFrame(summaries)
        namespace.update(out)
        print(out['crypto_run_summary_df'].to_string(index=False))
        return out
    start = min(pd.Timestamp(s.start) for s, _ in prepared.values())
    end = max(pd.Timestamp(s.end) for s, _ in prepared.values())
    try:
        _MINING_DAILY = load_bitcoin_daily(start, end)
    except Exception as exc:
        summaries.append(dict(site_key='all',specification='all',status='bitcoin_source_blocked',
                              message=_redact_message(exc)))
        out['crypto_run_summary_df'] = pd.DataFrame(summaries)
        out['crypto_source_status_df'] = pd.DataFrame([dict(status='failed',message=_redact_message(exc))])
        namespace.update(out)
        print(out['crypto_run_summary_df'].to_string(index=False))
        return out
    out['crypto_source_status_df'] = pd.DataFrame([dict(status='loaded', **_SOURCE_RECORD)])
    out['crypto_mining_economics_daily_df'] = _MINING_DAILY.copy()
    buckets = {k: [] for k in ('coefficients_df','coverage_df','variance_ratios_df','event_study_df',
        'sensitivity_df','placebo_df','horizon_effects_df','analysis_errors_df')}
    primary_hourly, primary_daily, raw_changes, raw_summaries, zeros = [], [], [], [], []
    fitted = {}
    try:
        from threadpoolctl import threadpool_limits
        context = threadpool_limits(limits=MAX_BLAS_THREADS)
    except ImportError:
        context = nullcontext()
    with context:
        for key, (site, frames) in prepared.items():
            prices, weather, gas = [frames[n] for n in ('lmp_hourly_df','weather_hourly_df','gas_daily_df')]
            print('\n' + key.upper() + ': cached prices/weather/gas loaded.')
            for specification in SPECIFICATIONS:
                activate_specification(specification)
                print('  Estimating:', specification, flush=True)
                try:
                    result = analyze_site(prices, weather, gas, site, BOOTSTRAP_DRAWS)
                    for name in buckets:
                        buckets[name].append(_tag(result.get(name,pd.DataFrame()),key,specification))
                    for node, z in result.get('node_outputs', {}).items():
                        fitted[(key,node,specification)] = z.get('fitted_models', {})
                        if specification == PRIMARY_SPECIFICATION:
                            primary_hourly.append(_tag(z['hourly_df'].assign(node=node),key,specification))
                            primary_daily.append(_tag(z['daily_df'].assign(node=node),key,specification))
                        for period in (0,1):
                            dd=z['daily_df'].loc[z['daily_df'].post.eq(period)]
                            for outcome in ('residual_m2','positive_semivariance','negative_semivariance'):
                                if outcome in dd:
                                    v=dd[outcome].dropna()
                                    zeros.append(dict(site_key=key,node=node,specification=specification,
                                        period='post' if period else 'pre',outcome=outcome,n_days=len(v),
                                        zero_days=int(v.eq(0).sum()),zero_share=float(v.eq(0).mean()) if len(v) else np.nan,
                                        below_log_offset_days=int(v.lt(1e-8).sum())))
                    n_mean=sum('mean_basis' in z.get('fitted_models',{}) for z in result['node_outputs'].values())
                    n_vol=sum('residual_m2' in z.get('fitted_models',{}) for z in result['node_outputs'].values())
                    bad=result['analysis_errors_df']
                    bad=bad.loc[~bad.status.eq('excluded_exact_alias')] if not bad.empty else bad
                    status='complete' if n_mean and n_mean==n_vol and bad.empty else 'partial_or_blocked'
                    summaries.append(dict(site_key=key,specification=specification,status=status,
                        mean_nodes=n_mean,total_moment_nodes=n_vol,
                        bitcoin_adjusted=SPEC_CONTROLS[specification] is not None,
                        message='Inspect crypto_analysis_errors_df' if status!='complete' else 'Estimated; conditional associations only'))
                except Exception as exc:
                    errors.append(dict(site_key=key,specification=specification,status='failed',message=_redact_message(exc)))
                    summaries.append(dict(site_key=key,specification=specification,status='failed',message=_redact_message(exc)))
                out['crypto_run_summary_df']=pd.DataFrame(summaries)
                namespace.update(out)
            # Price-only descriptive tables remain exactly the old paired-price sample.
            nodes=[n for n in prices.location.unique() if n!=site.benchmark]
            ordered=[n for n in site.nodes if n in nodes]+[n for n in nodes if n not in site.nodes]
            aliases=alias_table(prices,ordered)
            excluded=set(aliases.loc[aliases.exact_alias,'node_b']) if not aliases.empty else set()
            for node in ordered:
                if node in excluded:
                    continue
                try:
                    _, desc, change=describe_price_changes(prices,site,node,BOOTSTRAP_DRAWS)
                    raw_summaries.append(desc);raw_changes.append(change)
                except Exception as exc:
                    errors.append(dict(site_key=key,node=node,status='raw_failed',message=_redact_message(exc)))
    coef=_combine(buckets['coefficients_df'])
    if not coef.empty:
        # Preserve the six-outcome immediate-post BH family WITHIN each spec;
        # additionally expose an across-specification adjustment to disclose search.
        coef['p_fdr_bh_within_specification']=np.nan
        coef['p_fdr_bh_all_specifications']=np.nan
        coef['fdr_tests_within_specification']=np.nan
        for spec, indices in coef.groupby('specification').groups.items():
            valid=coef.loc[indices].term.eq('post') & coef.loc[indices].p_value.notna()
            ii=coef.loc[indices].index[valid]
            if len(ii):
                coef.loc[ii,'p_fdr_bh_within_specification']=multipletests(coef.loc[ii,'p_value'],method='fdr_bh')[1]
                coef.loc[ii,'fdr_tests_within_specification']=len(ii)
        ii=coef.index[coef.term.eq('post') & coef.p_value.notna()]
        if len(ii):
            coef.loc[ii,'p_fdr_bh_all_specifications']=multipletests(coef.loc[ii,'p_value'],method='fdr_bh')[1]
        coef['percent_geometric_moment_change']=np.where(coef.outcome.isin(
            ['residual_m2','positive_semivariance','negative_semivariance']),100*np.expm1(coef.estimate),np.nan)
        out['crypto_all_coefficients_df']=coef
        out['crypto_key_results_df']=coef.loc[coef.specification.eq(PRIMARY_SPECIFICATION)
            & (coef.term.isin(['post','post_trend']) | coef.term.str.contains('btc_',regex=False))].copy()
        comparison=coef.loc[coef.term.eq('post') & coef.outcome.isin(['mean_basis','residual_m2'])].copy()
        out['crypto_specification_comparison_df']=comparison
        common=comparison.loc[comparison.specification.ne('weather_gas_original_sample')
            & comparison.outcome.eq('mean_basis')]
        check=common.groupby(['site_key','node']).agg(specifications=('specification','nunique'),
            distinct_day_counts=('n_days','nunique'),n_days=('n_days','first')).reset_index()
        check['same_day_count']=check.distinct_day_counts.eq(1)
        out['crypto_common_sample_check_df']=check
    mapping={'coverage_df':'crypto_coverage_df','variance_ratios_df':'crypto_variance_ratios_df',
        'event_study_df':'crypto_event_study_df','sensitivity_df':'crypto_sensitivity_df',
        'placebo_df':'crypto_placebo_df','horizon_effects_df':'crypto_horizon_effects_df'}
    for key,name in mapping.items():
        out[name]=_combine(buckets[key])
    out['crypto_analysis_errors_df']=_combine(buckets['analysis_errors_df']+[pd.DataFrame(errors)])
    out['crypto_control_audit_df']=pd.DataFrame(_CONTROL_AUDIT_ROWS)
    out['crypto_centering_df']=pd.DataFrame(_BTC_CENTER_ROWS).drop_duplicates()
    out['crypto_run_summary_df']=pd.DataFrame(summaries)
    out['crypto_daily_primary_df']=_combine(primary_daily)
    out['crypto_hourly_primary_df']=_combine(primary_hourly)
    out['crypto_raw_mean_variance_changes_df']=_combine(raw_changes)
    out['crypto_raw_prepost_summary_df']=_combine(raw_summaries)
    out['crypto_zero_moment_df']=pd.DataFrame(zeros)
    out['crypto_fitted_models']=fitted
    _export_crypto_outputs(out)
    namespace.update(out)
    print('\nRUN SUMMARY\n'+out['crypto_run_summary_df'].to_string(index=False))
    print('\nSTART HERE: crypto_specification_comparison_df, crypto_control_audit_df, crypto_analysis_errors_df')
    if not out['crypto_specification_comparison_df'].empty:
        cols=['site_key','node','specification','outcome','estimate','p_value','n_days']
        print(out['crypto_specification_comparison_df'][cols].to_string(index=False))
    print('Old manuscript/results were not overwritten. No causal or AI claim is made.')
    return out


if __name__ == '__main__':
    globals().update(main(globals()))
