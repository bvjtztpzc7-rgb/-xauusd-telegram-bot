import os
import time
import math
import requests
import warnings
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAZIONE
# ============================================================

SYMBOL = "XAU/USD"
INTERVAL = "5min"

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurato")

OUTPUT_DIR = "backtest_results_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Strategia base
# -------------------------

EMA_FAST = 20
EMA_SLOW = 50

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

RSI_PERIOD = 14
ATR_PERIOD = 14

RSI_MIN = 30
RSI_MAX = 65

# Base attuale
BASE_TP_ATR = 2.5
BASE_SL_ATR = 1.5

# -------------------------
# Orizzonte analisi
# -------------------------

MFE_MAE_HORIZON_BARS = 288      # 24 ore su 5m
TRADE_MAX_BARS = 288             # massimo 24 ore

# -------------------------
# Download
# -------------------------

BLOCK_SIZE = 5000
N_BLOCKS = 2

# -------------------------
# Split
# -------------------------

DEVELOPMENT_RATIO = 0.70

# -------------------------
# Progress
# -------------------------

PRINT_EVERY = 100


# ============================================================
# UTILITY
# ============================================================

def print_separator(title=""):
    print("\n" + "=" * 80)
    if title:
        print(title)
        print("=" * 80)


def safe_div(a, b):
    if b is None or b == 0 or pd.isna(b):
        return np.nan
    return a / b


def profit_factor_from_R(results):
    if not results:
        return np.nan

    gains = sum(x for x in results if x > 0)
    losses = abs(sum(x for x in results if x < 0))

    if losses == 0:
        return np.inf if gains > 0 else np.nan

    return gains / losses


def max_drawdown(values):
    if len(values) == 0:
        return np.nan

    equity = np.cumsum(values)
    peak = np.maximum.accumulate(np.concatenate([[0], equity]))
    dd = peak[1:] - equity

    return float(np.max(dd)) if len(dd) else 0.0


def r_value(outcome, tp_atr, sl_atr):
    if outcome == "TP":
        return tp_atr / sl_atr

    if outcome == "SL":
        return -1.0

    return 0.0


# ============================================================
# DOWNLOAD DATI
# ============================================================

def download_block(end_date=None):
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": BLOCK_SIZE,
        "apikey": API_KEY,
        "format": "JSON",
        "timezone": "UTC",
    }

    if end_date is not None:
        params["end_date"] = end_date

    print("Download dati:", {
        k: "***" if k == "apikey" else v
        for k, v in params.items()
    })

    response = requests.get(
        "https://api.twelvedata.com/time_series",
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise RuntimeError(f"Risposta Twelve Data non valida: {data}")

    df = pd.DataFrame(data["values"])

    if df.empty:
        raise RuntimeError("Blocco dati vuoto")

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce"
        )

    df = df.dropna(
        subset=["datetime", "open", "high", "low", "close"]
    )

    df = df.sort_values("datetime")
    df = df.drop_duplicates("datetime")

    print(
        f"Ricevute {len(df)} candele | "
        f"{df['datetime'].min()} -> {df['datetime'].max()}"
    )

    return df


def download_history():
    print_separator("DOWNLOAD STORICO")

    blocks = []

    end_date = None

    for i in range(N_BLOCKS):
        print_separator(f"DOWNLOAD BLOCCO {i + 1}/{N_BLOCKS}")

        df = download_block(end_date)

        blocks.append(df)

        if i < N_BLOCKS - 1:
            oldest = df["datetime"].min()

            # piccolo margine per evitare duplicazioni
            end_date = (
                oldest - pd.Timedelta(minutes=5)
            ).strftime("%Y-%m-%d %H:%M:%S")

        time.sleep(1)

    data = pd.concat(blocks, ignore_index=True)

    data = (
        data
        .drop_duplicates("datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print_separator("DATI STORICI")

    print(f"Candele totali: {len(data)}")
    print(
        f"Periodo: "
        f"{data['datetime'].min()} -> "
        f"{data['datetime'].max()}"
    )

    return data


# ============================================================
# INDICATORI
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(df, period=14):
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()


def calculate_indicators(df):

    print_separator("CALCOLO INDICATORI")

    df = df.copy()

    # EMA
    df["ema20"] = df["close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    # EMA distance
    df["ema_distance"] = (
        (df["ema20"] - df["ema50"])
        / df["close"]
        * 100
    )

    # EMA slope
    df["ema20_slope"] = (
        df["ema20"] -
        df["ema20"].shift(5)
    )

    df["ema50_slope"] = (
        df["ema50"] -
        df["ema50"].shift(5)
    )

    # MACD
    ema_fast = df["close"].ewm(
        span=MACD_FAST,
        adjust=False
    ).mean()

    ema_slow = df["close"].ewm(
        span=MACD_SLOW,
        adjust=False
    ).mean()

    df["macd"] = ema_fast - ema_slow

    df["macd_signal"] = df["macd"].ewm(
        span=MACD_SIGNAL,
        adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    df["macd_change"] = (
        df["macd_hist"] -
        df["macd_hist"].shift(3)
    )

    # RSI
    df["rsi"] = calculate_rsi(
        df["close"],
        RSI_PERIOD
    )

    df["rsi_change"] = (
        df["rsi"] -
        df["rsi"].shift(3)
    )

    # ATR
    df["atr"] = calculate_atr(
        df,
        ATR_PERIOD
    )

    df["atr_pct"] = (
        df["atr"] /
        df["close"] *
        100
    )

    # ATR regime
    atr_median = (
        df["atr"]
        .rolling(200)
        .median()
    )

    df["atr_ratio"] = (
        df["atr"] /
        atr_median
    )

    df["atr_regime"] = np.select(
        [
            df["atr_ratio"] < 0.75,
            df["atr_ratio"] > 1.25
        ],
        [
            "LOW",
            "HIGH"
        ],
        default="NORMAL"
    )

    # Candle structure
    df["body"] = (
        df["close"] -
        df["open"]
    )

    df["body_abs"] = df["body"].abs()

    df["range"] = (
        df["high"] -
        df["low"]
    )

    df["body_ratio"] = (
        df["body_abs"] /
        df["range"].replace(0, np.nan)
    )

    df["upper_wick"] = (
        df["high"] -
        df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1) -
        df["low"]
    )

    df["body_atr"] = (
        df["body_abs"] /
        df["atr"]
    )

    # Body buckets
    df["body_bucket"] = pd.cut(
        df["body_atr"],
        bins=[
            -np.inf,
            0.25,
            0.50,
            1.00,
            1.50,
            np.inf
        ],
        labels=[
            "VERY_SMALL",
            "SMALL",
            "MEDIUM",
            "LARGE",
            "VERY_LARGE"
        ]
    ).astype(str)

    # EMA strength
    df["ema_strength_value"] = (
        df["ema_distance"].abs()
    )

    df["ema_strength"] = pd.cut(
        df["ema_strength_value"],
        bins=[
            -np.inf,
            0.02,
            0.05,
            0.10,
            np.inf
        ],
        labels=[
            "VERY_WEAK",
            "WEAK",
            "MEDIUM",
            "STRONG"
        ]
    ).astype(str)

    # MACD strength
    macd_abs = df["macd_hist"].abs()

    macd_median = (
        macd_abs
        .rolling(200)
        .median()
    )

    macd_ratio = (
        macd_abs /
        macd_median.replace(0, np.nan)
    )

    df["macd_strength"] = np.select(
        [
            macd_ratio < 0.75,
            macd_ratio > 1.50
        ],
        [
            "WEAK",
            "STRONG"
        ],
        default="NORMAL"
    )

    # RSI buckets
    df["rsi_bucket"] = pd.cut(
        df["rsi"],
        bins=[
            0,
            30,
            40,
            50,
            60,
            70,
            100
        ],
        labels=[
            "<30",
            "30-40",
            "40-50",
            "50-60",
            "60-70",
            ">70"
        ]
    ).astype(str)

    # Momentum
    for n in [3, 6, 12, 24]:
        df[f"momentum_{n}"] = (
            df["close"] /
            df["close"].shift(n) -
            1
        ) * 100

    # Recent highs/lows
    for n in [6, 12, 24, 48]:
        df[f"high_distance_{n}"] = (
            df[f"high_distance_{n}"] if
            f"high_distance_{n}" in df.columns
            else (
                df["close"] /
                df["high"].rolling(n).max() -
                1
            ) * 100
        )

        df[f"low_distance_{n}"] = (
            df["close"] /
            df["low"].rolling(n).min() -
            1
        ) * 100

    return df


# ============================================================
# TIME FEATURES
# ============================================================

def add_time_features(df):

    df = df.copy()

    df["hour_utc"] = df["datetime"].dt.hour

    df["minute"] = df["datetime"].dt.minute

    df["day_of_week"] = (
        df["datetime"]
        .dt.day_name()
    )

    # Sessioni approssimative UTC
    df["session"] = np.select(
        [
            df["hour_utc"].between(0, 6),
            df["hour_utc"].between(7, 12),
            df["hour_utc"].between(13, 17),
            df["hour_utc"].between(18, 23)
        ],
        [
            "ASIA",
            "EUROPE",
            "US",
            "LATE"
        ],
        default="OTHER"
    )

    return df


# ============================================================
# MULTI TIMEFRAME
# ============================================================

def build_higher_timeframes(df):

    print_separator("COSTRUZIONE 15M + 1H")

    base = df.set_index("datetime")

    # 15m
    df15 = (
        base
        .resample("15min")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        })
        .dropna()
    )

    df15["ema20_15"] = df15["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df15["ema50_15"] = df15["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df15["trend15"] = np.where(
        df15["ema20_15"] >
        df15["ema50_15"],
        "BULLISH",
        "BEARISH"
    )

    # 1H
    df1h = (
        base
        .resample("1h")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        })
        .dropna()
    )

    df1h["ema20_1h"] = df1h["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df1h["ema50_1h"] = df1h["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df1h["trend1h"] = np.where(
        df1h["ema20_1h"] >
        df1h["ema50_1h"],
        "BULLISH",
        "BEARISH"
    )

    # Merge backward:
    # important: use last CLOSED higher-timeframe candle
    df = pd.merge_asof(
        df.sort_values("datetime"),
        df15[
            ["trend15", "ema20_15", "ema50_15"]
        ].reset_index(),
        on="datetime",
        direction="backward"
    )

    df = pd.merge_asof(
        df.sort_values("datetime"),
        df1h[
            ["trend1h", "ema20_1h", "ema50_1h"]
        ].reset_index(),
        on="datetime",
        direction="backward"
    )

    return df


# ============================================================
# SEGNALI
# ============================================================

def generate_signals(df):

    print_separator("GENERAZIONE SEGNALI BASE")

    df = df.copy()

    bullish = (
        (df["ema20"] > df["ema50"]) &
        (df["macd"] > df["macd_signal"]) &
        (df["rsi"] > RSI_MIN) &
        (df["rsi"] < RSI_MAX) &
        (df["trend15"] == "BULLISH")
    )

    bearish = (
        (df["ema20"] < df["ema50"]) &
        (df["macd"] < df["macd_signal"]) &
        (df["rsi"] > RSI_MIN) &
        (df["rsi"] < RSI_MAX) &
        (df["trend15"] == "BEARISH")
    )

    df["signal"] = np.select(
        [bullish, bearish],
        ["BUY", "SELL"],
        default=""
    )

    df["signal_id"] = (
        df["signal"]
        .ne("")
        .cumsum()
    )

    total = (df["signal"] != "").sum()
    buys = (df["signal"] == "BUY").sum()
    sells = (df["signal"] == "SELL").sum()

    print(f"BUY/SELL totali: {total}")
    print(f"BUY: {buys}")
    print(f"SELL: {sells}")

    return df


# ============================================================
# SIGNAL EPISODES
# ============================================================

def classify_signal_episodes(df):

    df = df.copy()

    signal_mask = df["signal"] != ""

    df["episode_id"] = np.nan
    df["episode_first"] = False

    current_episode = 0
    previous_signal = None

    for idx in df.index:

        signal = df.at[idx, "signal"]

        if signal == "":
            continue

        if signal != previous_signal:
            current_episode += 1

        df.at[idx, "episode_id"] = current_episode

        previous_signal = signal

    signal_rows = df[signal_mask].copy()

    if not signal_rows.empty:
        first_indices = (
            signal_rows
            .groupby("episode_id")
            .head(1)
            .index
        )

        df.loc[first_indices, "episode_first"] = True

    return df


# ============================================================
# TRADE ENGINE
# ============================================================

def simulate_trade(
    df,
    entry_index,
    direction,
    tp_atr,
    sl_atr,
    max_bars=TRADE_MAX_BARS
):

    # signal candle -> next candle open
    if entry_index + 1 >= len(df):
        return None

    signal_row = df.iloc[entry_index]
    entry_row = df.iloc[entry_index + 1]

    entry_price = float(entry_row["open"])
    atr = float(signal_row["atr"])

    if not np.isfinite(entry_price):
        return None

    if not np.isfinite(atr) or atr <= 0:
        return None

    if direction == "BUY":

        tp_price = (
            entry_price +
            tp_atr * atr
        )

        sl_price = (
            entry_price -
            sl_atr * atr
        )

    else:

        tp_price = (
            entry_price -
            tp_atr * atr
        )

        sl_price = (
            entry_price +
            sl_atr * atr
        )

    end_index = min(
        entry_index + 1 + max_bars,
        len(df) - 1
    )

    outcome = "TIMEOUT"
    exit_index = end_index

    mfe = 0.0
    mae = 0.0

    same_candle = False

    for j in range(
        entry_index + 1,
        end_index + 1
    ):

        row = df.iloc[j]

        high = float(row["high"])
        low = float(row["low"])

        if direction == "BUY":

            favorable = (
                high - entry_price
            ) / atr

            adverse = (
                entry_price - low
            ) / atr

            tp_hit = high >= tp_price
            sl_hit = low <= sl_price

        else:

            favorable = (
                entry_price - low
            ) / atr

            adverse = (
                high - entry_price
            ) / atr

            tp_hit = low <= tp_price
            sl_hit = high >= sl_price

        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

        if tp_hit and sl_hit:

            same_candle = True

            # Conservative assumption
            outcome = "SL"
            exit_index = j
            break

        if tp_hit:

            outcome = "TP"
            exit_index = j
            break

        if sl_hit:

            outcome = "SL"
            exit_index = j
            break

    result_R = r_value(
        outcome,
        tp_atr,
        sl_atr
    )

    return {
        "signal_index": entry_index,
        "signal_time": signal_row["datetime"],
        "entry_time": entry_row["datetime"],
        "exit_time": df.iloc[exit_index]["datetime"],
        "direction": direction,

        "entry_price": entry_price,
        "tp_price": tp_price,
        "sl_price": sl_price,

        "atr": atr,

        "tp_atr": tp_atr,
        "sl_atr": sl_atr,

        "outcome": outcome,
        "result_R": result_R,

        "bars_held": exit_index - entry_index - 1,

        "minutes_held": (
            exit_index - entry_index - 1
        ) * 5,

        "MFE_ATR": mfe,
        "MAE_ATR": mae,

        "same_candle_tp_sl": same_candle,

        # Features at signal time
        "rsi": signal_row["rsi"],
        "rsi_bucket": signal_row["rsi_bucket"],

        "ema_distance": signal_row["ema_distance"],
        "ema_strength": signal_row["ema_strength"],

        "macd_hist": signal_row["macd_hist"],
        "macd_strength": signal_row["macd_strength"],

        "atr_ratio": signal_row["atr_ratio"],
        "atr_regime": signal_row["atr_regime"],

        "body_atr": signal_row["body_atr"],
        "body_bucket": signal_row["body_bucket"],

        "momentum_3": signal_row["momentum_3"],
        "momentum_6": signal_row["momentum_6"],
        "momentum_12": signal_row["momentum_12"],
        "momentum_24": signal_row["momentum_24"],

        "ema20_slope": signal_row["ema20_slope"],
        "ema50_slope": signal_row["ema50_slope"],

        "trend15": signal_row["trend15"],
        "trend1h": signal_row["trend1h"],

        "hour_utc": signal_row["hour_utc"],
        "session": signal_row["session"],
        "day_of_week": signal_row["day_of_week"],

        "episode_id": signal_row["episode_id"],
        "episode_first": signal_row["episode_first"],
    }


# ============================================================
# BUILD ALL TRADES
# ============================================================

def build_trade_dataset(df):

    print_separator("COSTRUZIONE DATASET DEI TRADE")

    signal_indices = df.index[
        df["signal"] != ""
    ].tolist()

    print(
        f"Segnali trovati: "
        f"{len(signal_indices)}"
    )

    trades = []

    for counter, idx in enumerate(signal_indices, 1):

        direction = df.at[idx, "signal"]

        result = simulate_trade(
            df,
            idx,
            direction,
            BASE_TP_ATR,
            BASE_SL_ATR
        )

        if result is not None:
            trades.append(result)

        if counter % PRINT_EVERY == 0:
            print(
                f"Processati "
                f"{counter}/{len(signal_indices)} segnali..."
            )

    trades = pd.DataFrame(trades)

    return trades


# ============================================================
# DATASET NON OVERLAPPING
# ============================================================

def build_non_overlapping(df):

    signal_indices = df.index[
        df["signal"] != ""
    ].tolist()

    selected = []

    next_allowed_index = -1

    for idx in signal_indices:

        if idx < next_allowed_index:
            continue

        selected.append(idx)

        # Simuliamo il trade base
        direction = df.at[idx, "signal"]

        result = simulate_trade(
            df,
            idx,
            direction,
            BASE_TP_ATR,
            BASE_SL_ATR
        )

        if result is None:
            continue

        # il prossimo segnale può essere considerato
        # solo dopo la chiusura del trade
        next_allowed_index = (
            idx +
            result["bars_held"] +
            2
        )

    return selected


# ============================================================
# SUMMARY
# ============================================================

def summarize_trades(trades, dataset_name):

    if trades.empty:

        return {
            "dataset": dataset_name,
            "trades": 0,
            "TP": 0,
            "SL": 0,
            "TIMEOUT": 0,
            "win_rate_pct": np.nan,
            "avg_R": np.nan,
            "total_R": 0,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "avg_MFE_ATR": np.nan,
            "avg_MAE_ATR": np.nan,
            "median_MFE_ATR": np.nan,
            "median_MAE_ATR": np.nan,
        }

    results = trades["result_R"].tolist()

    tp = (
        trades["outcome"] == "TP"
    ).sum()

    sl = (
        trades["outcome"] == "SL"
    ).sum()

    timeout = (
        trades["outcome"] == "TIMEOUT"
    ).sum()

    wins = tp

    return {
        "dataset": dataset_name,

        "trades": len(trades),

        "TP": int(tp),
        "SL": int(sl),
        "TIMEOUT": int(timeout),

        "win_rate_pct": (
            wins /
            len(trades) *
            100
        ),

        "avg_R": trades["result_R"].mean(),

        "total_R": trades["result_R"].sum(),

        "profit_factor": profit_factor_from_R(
            results
        ),

        "max_drawdown_R": max_drawdown(
            results
        ),

        "avg_MFE_ATR": trades["MFE_ATR"].mean(),
        "avg_MAE_ATR": trades["MAE_ATR"].mean(),

        "median_MFE_ATR": trades["MFE_ATR"].median(),
        "median_MAE_ATR": trades["MAE_ATR"].median(),
    }


# ============================================================
# SUMMARY MULTI DATASET
# ============================================================

def make_summary_table(trades):

    rows = []

    if trades.empty:
        return pd.DataFrame()

    split_time = trades["signal_time"].quantile(
        DEVELOPMENT_RATIO
    )

    development = trades[
        trades["signal_time"] <= split_time
    ]

    verification = trades[
        trades["signal_time"] > split_time
    ]

    datasets = {
        "DEVELOPMENT": development,
        "VERIFICATION": verification,
        "ALL": trades,
    }

    for name, subset in datasets.items():

        rows.append(
            summarize_trades(
                subset,
                name
            )
        )

    return pd.DataFrame(rows)


# ============================================================
# BUY / SELL
# ============================================================

def analyze_direction(trades):

    rows = []

    split_time = trades["signal_time"].quantile(
        DEVELOPMENT_RATIO
    )

    datasets = {
        "DEVELOPMENT": trades[
            trades["signal_time"] <= split_time
        ],
        "VERIFICATION": trades[
            trades["signal_time"] > split_time
        ],
    }

    for dataset_name, dataset in datasets.items():

        for direction in ["BUY", "SELL"]:

            subset = dataset[
                dataset["direction"] == direction
            ]

            row = summarize_trades(
                subset,
                f"{dataset_name}_{direction}"
            )

            same = (
                subset["same_candle_tp_sl"].sum()
                if not subset.empty
                else 0
            )

            row["SL_AND_TP_SAME_CANDLE"] = int(
                same
            )

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# GENERIC CONDITION ANALYSIS
# ============================================================

def condition_stats(
    trades,
    condition_name,
    mask
):

    subset = trades[mask]

    if len(subset) < 20:
        return None

    row = summarize_trades(
        subset,
        condition_name
    )

    row["condition"] = condition_name

    return row


def analyze_conditions(trades):

    rows = []

    conditions = {}

    # Base
    conditions["15_ONLY"] = (
        trades["trend15"].isin(
            ["BULLISH", "BEARISH"]
        )
    )

    conditions["1H_ONLY"] = (
        trades["trend1h"].isin(
            ["BULLISH", "BEARISH"]
        )
    )

    conditions["15_AND_1H"] = (
        trades["trend15"].eq(
            np.where(
                trades["direction"] == "BUY",
                "BULLISH",
                "BEARISH"
            )
        )
        &
        trades["trend1h"].eq(
            np.where(
                trades["direction"] == "BUY",
                "BULLISH",
                "BEARISH"
            )
        )
    )

    # Categorical features
    for col in [
        "rsi_bucket",
        "ema_strength",
        "macd_strength",
        "atr_regime",
        "body_bucket",
        "session",
        "day_of_week",
    ]:

        for value in trades[col].dropna().unique():

            conditions[
                f"{col}={value}"
            ] = (
                trades[col] == value
            )

    # Momentum
    conditions["momentum_3_positive"] = (
        trades["momentum_3"] > 0
    )

    conditions["momentum_3_negative"] = (
        trades["momentum_3"] < 0
    )

    conditions["momentum_6_positive"] = (
        trades["momentum_6"] > 0
    )

    conditions["momentum_6_negative"] = (
        trades["momentum_6"] < 0
    )

    # Strong trend
    conditions["ema_strong"] = (
        trades["ema_strength"].isin(
            ["MEDIUM", "STRONG"]
        )
    )

    conditions["ema_weak"] = (
        trades["ema_strength"].isin(
            ["VERY_WEAK", "WEAK"]
        )
    )

    conditions["macd_strong"] = (
        trades["macd_strength"] == "STRONG"
    )

    conditions["macd_weak"] = (
        trades["macd_strength"] == "WEAK"
    )

    # Direction-specific trend alignment
    conditions["direction_15_aligned"] = (
        (
            (trades["direction"] == "BUY") &
            (trades["trend15"] == "BULLISH")
        )
        |
        (
            (trades["direction"] == "SELL") &
            (trades["trend15"] == "BEARISH")
        )
    )

    conditions["direction_1h_aligned"] = (
        (
            (trades["direction"] == "BUY") &
            (trades["trend1h"] == "BULLISH")
        )
        |
        (
            (trades["direction"] == "SELL") &
            (trades["trend1h"] == "BEARISH")
        )
    )

    for name, mask in conditions.items():

        row = condition_stats(
            trades,
            name,
            mask
        )

        if row is not None:
            rows.append(row)

    # Useful combinations
    combo_conditions = {

        "EMA_STRONG_AND_MACD_STRONG":
            conditions["ema_strong"] &
            conditions["macd_strong"],

        "EMA_WEAK_AND_MACD_WEAK":
            conditions["ema_weak"] &
            conditions["macd_weak"],

        "EMA_STRONG_AND_MACD_STRONG_AND_1H_ALIGNED":
            conditions["ema_strong"] &
            conditions["macd_strong"] &
            conditions["direction_1h_aligned"],

        "EMA_STRONG_AND_15_ALIGNED":
            conditions["ema_strong"] &
            conditions["direction_15_aligned"],

        "MACD_STRONG_AND_15_ALIGNED":
            conditions["macd_strong"] &
            conditions["direction_15_aligned"],

        "HIGH_VOLATILITY_AND_STRONG_EMA":
            (trades["atr_regime"] == "HIGH") &
            conditions["ema_strong"],

        "NORMAL_VOLATILITY_AND_STRONG_EMA":
            (trades["atr_regime"] == "NORMAL") &
            conditions["ema_strong"],

        "LOW_VOLATILITY_AND_STRONG_EMA":
            (trades["atr_regime"] == "LOW") &
            conditions["ema_strong"],
    }

    for name, mask in combo_conditions.items():

        row = condition_stats(
            trades,
            name,
            mask
        )

        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# FEATURE CORRELATIONS
# ============================================================

def feature_correlations(trades):

    features = [
        "rsi",
        "ema_distance",
        "macd_hist",
        "atr_ratio",
        "body_atr",
        "momentum_3",
        "momentum_6",
        "momentum_12",
        "momentum_24",
        "ema20_slope",
        "ema50_slope",
        "hour_utc",
    ]

    rows = []

    for feature in features:

        if feature not in trades.columns:
            continue

        subset = trades[
            [feature, "MFE_ATR", "MAE_ATR", "result_R"]
        ].dropna()

        if len(subset) < 20:
            continue

        corr_result = subset[
            feature
        ].corr(
            subset["result_R"]
        )

        corr_mfe = subset[
            feature
        ].corr(
            subset["MFE_ATR"]
        )

        corr_mae = subset[
            feature
        ].corr(
            subset["MAE_ATR"]
        )

        rows.append({
            "feature": feature,
            "corr_MFE_ATR": corr_mfe,
            "corr_MAE_ATR": corr_mae,
            "corr_result_R": corr_result,
            "abs_corr_result_R": abs(corr_result),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "abs_corr_result_R",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# TIME HORIZONS
# ============================================================

def analyze_time_horizons(trades):

    rows = []

    for direction in ["BUY", "SELL"]:

        subset = trades[
            trades["direction"] == direction
        ]

        if subset.empty:
            continue

        for max_minutes in [
            15,
            30,
            60,
            120,
            240,
            480,
            720,
            1440,
        ]:

            limited = subset[
                subset["minutes_held"] <= max_minutes
            ]

            if len(limited) < 20:
                continue

            row = summarize_trades(
                limited,
                f"{direction}_{max_minutes}m"
            )

            row["direction"] = direction
            row["max_minutes"] = max_minutes

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# TARGET ANALYSIS
# ============================================================

def target_analysis(df, trades):

    rows = []

    targets = [
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
    ]

    split_time = trades["signal_time"].quantile(
        DEVELOPMENT_RATIO
    )

    for dataset_name, dataset in [
        (
            "DEVELOPMENT",
            trades[
                trades["signal_time"] <= split_time
            ]
        ),
        (
            "VERIFICATION",
            trades[
                trades["signal_time"] > split_time
            ]
        )
    ]:

        for direction in ["BUY", "SELL"]:

            subset = dataset[
                dataset["direction"] == direction
            ]

            if subset.empty:
                continue

            for target in targets:

                reached = (
                    subset["MFE_ATR"] >= target
                )

                reached_count = reached.sum()

                reached_subset = subset[
                    reached
                ]

                rows.append({
                    "dataset": dataset_name,
                    "direction": direction,
                    "target_atr": target,
                    "trades": len(subset),
                    "reached_count": int(
                        reached_count
                    ),
                    "reach_pct": (
                        reached_count /
                        len(subset) *
                        100
                    ),
                    "avg_minutes_to_target": np.nan,
                    "median_minutes_to_target": np.nan,
                })

    return pd.DataFrame(rows)


# ============================================================
# TP / SL MATRIX
# ============================================================

def evaluate_matrix(
    df,
    signals,
    dataset_name,
    tp_values,
    sl_values
):

    rows = []

    for direction in ["BUY", "SELL"]:

        direction_signals = [
            x for x in signals
            if df.at[x, "signal"] == direction
        ]

        if not direction_signals:
            continue

        signal_times = df.loc[
            direction_signals,
            "datetime"
        ]

        split_time = df["datetime"].quantile(
            DEVELOPMENT_RATIO
        )

        if dataset_name == "DEVELOPMENT":
            direction_signals = [
                x for x in direction_signals
                if df.at[x, "datetime"] <= split_time
            ]

        elif dataset_name == "VERIFICATION":
            direction_signals = [
                x for x in direction_signals
                if df.at[x, "datetime"] > split_time
            ]

        for tp_atr, sl_atr in product(
            tp_values,
            sl_values
        ):

            results = []

            wins = 0
            losses = 0

            for idx in direction_signals:

                result = simulate_trade(
                    df,
                    idx,
                    direction,
                    tp_atr,
                    sl_atr
                )

                if result is None:
                    continue

                results.append(
                    result["result_R"]
                )

                if result["outcome"] == "TP":
                    wins += 1

                elif result["outcome"] == "SL":
                    losses += 1

            if not results:
                continue

            rows.append({
                "dataset": dataset_name,
                "direction": direction,

                "TP_ATR": tp_atr,
                "SL_ATR": sl_atr,

                "trades": len(results),

                "wins": wins,
                "losses": losses,

                "win_rate_pct": (
                    wins /
                    len(results) *
                    100
                ),

                "total_R": sum(results),

                "avg_R": np.mean(results),

                "profit_factor":
                    profit_factor_from_R(
                        results
                    ),

                "max_drawdown_R":
                    max_drawdown(
                        results
                    ),
            })

    return pd.DataFrame(rows)


# ============================================================
# SAME CANDLE ANALYSIS
# ============================================================

def same_candle_analysis(trades):

    rows = []

    for mode in [
        "CONSERVATIVE",
        "OPTIMISTIC",
        "EXCLUDE"
    ]:

        temp = trades.copy()

        same = (
            temp["same_candle_tp_sl"]
        )

        if mode == "OPTIMISTIC":

            temp.loc[
                same,
                "outcome"
            ] = "TP"

            temp.loc[
                same,
                "result_R"
            ] = (
                temp.loc[
                    same,
                    "tp_atr"
                ] /
                temp.loc[
                    same,
                    "sl_atr"
                ]
            )

        elif mode == "EXCLUDE":

            temp = temp[
                ~same
            ]

        row = summarize_trades(
            temp,
            mode
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# TIME SPLIT
# ============================================================

def temporal_blocks(trades, n_blocks=6):

    if trades.empty:
        return pd.DataFrame()

    ordered = trades.sort_values(
        "signal_time"
    ).reset_index(drop=True)

    blocks = np.array_split(
        ordered,
        n_blocks
    )

    rows = []

    for i, block in enumerate(blocks, 1):

        row = summarize_trades(
            block,
            f"BLOCK_{i}"
        )

        row["block"] = i

        row["start"] = (
            block["signal_time"].min()
        )

        row["end"] = (
            block["signal_time"].max()
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# INDEPENDENT SIGNAL DATASETS
# ============================================================

def build_signal_modes(df):

    print_separator(
        "COSTRUZIONE MODALITÀ SEGNALI"
    )

    all_indices = df.index[
        df["signal"] != ""
    ].tolist()

    episode_first_indices = df.index[
        (
            (df["signal"] != "") &
            df["episode_first"]
        )
    ].tolist()

    non_overlapping_indices = (
        build_non_overlapping(df)
    )

    print(
        f"ALL: {len(all_indices)}"
    )

    print(
        f"FIRST-IN-EPISODE: "
        f"{len(episode_first_indices)}"
    )

    print(
        f"NON-OVERLAPPING: "
        f"{len(non_overlapping_indices)}"
    )

    return {
        "ALL": all_indices,
        "FIRST_IN_EPISODE": episode_first_indices,
        "NON_OVERLAPPING": non_overlapping_indices,
    }


# ============================================================
# BUILD TRADES FOR MODE
# ============================================================

def trades_from_indices(
    df,
    indices,
    tp_atr=BASE_TP_ATR,
    sl_atr=BASE_SL_ATR
):

    results = []

    for idx in indices:

        direction = df.at[
            idx,
            "signal"
        ]

        result = simulate_trade(
            df,
            idx,
            direction,
            tp_atr,
            sl_atr
        )

        if result is not None:
            results.append(result)

    return pd.DataFrame(results)


# ============================================================
# PRINT TOP ANALYSIS
# ============================================================

def print_top_analysis(
    df,
    title,
    n=15
):

    print_separator(title)

    if df is None or df.empty:
        print("Nessun dato disponibile.")
        return

    if "feature" in df.columns:

        output = df.copy()

        if "abs_corr_result_R" in output.columns:

            output = (
                output
                .sort_values(
                    "abs_corr_result_R",
                    ascending=False
                )
                .head(n)
            )

        print(
            output.to_string(
                index=False
            )
        )

        return

    if "condition" in df.columns:

        display_cols = [
            "condition",
            "trades",
            "win_rate_pct",
            "avg_R",
            "total_R",
            "profit_factor",
            "max_drawdown_R",
            "avg_MFE_ATR",
            "avg_MAE_ATR",
        ]

        display_cols = [
            c for c in display_cols
            if c in df.columns
        ]

        output = (
            df
            .sort_values(
                ["trades", "avg_R"],
                ascending=[False, False]
            )
            .head(n)
        )

        print(
            output[
                display_cols
            ].to_string(index=False)
        )

        return

    print(
        "Formato dataframe non riconosciuto."
    )

    print(
        "Colonne disponibili:",
        list(df.columns)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print_separator(
        "BACKTEST V3 - XAU/USD"
    )

    print(f"Symbol: {SYMBOL}")
    print(f"Interval: {INTERVAL}")

    print(
        f"TP base: {BASE_TP_ATR} ATR"
    )

    print(
        f"SL base: {BASE_SL_ATR} ATR"
    )

    print(
        f"MFE/MAE horizon: "
        f"{MFE_MAE_HORIZON_BARS} barre"
    )

    print(
        f"Development: "
        f"{DEVELOPMENT_RATIO * 100:.0f}%"
    )

    print(
        f"Verification: "
        f"{(1 - DEVELOPMENT_RATIO) * 100:.0f}%"
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    df = download_history()

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    df = calculate_indicators(df)

    df = add_time_features(df)

    df = build_higher_timeframes(df)

    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    df = generate_signals(df)

    df = classify_signal_episodes(df)

    # --------------------------------------------------------
    # RAW DATA SAVE
    # --------------------------------------------------------

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "raw_5m_data_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # SIGNAL MODES
    # --------------------------------------------------------

    modes = build_signal_modes(df)

    # --------------------------------------------------------
    # ALL BASE TRADES
    # --------------------------------------------------------

    print_separator(
        "COSTRUZIONE TRADE BASE"
    )

    all_trades = trades_from_indices(
        df,
        modes["ALL"]
    )

    episode_trades = trades_from_indices(
        df,
        modes["FIRST_IN_EPISODE"]
    )

    non_overlap_trades = trades_from_indices(
        df,
        modes["NON_OVERLAPPING"]
    )

    # --------------------------------------------------------
    # MODE SUMMARY
    # --------------------------------------------------------

    mode_rows = []

    for name, dataset in [
        ("ALL", all_trades),
        (
            "FIRST_IN_EPISODE",
            episode_trades
        ),
        (
            "NON_OVERLAPPING",
            non_overlap_trades
        ),
    ]:

        row = summarize_trades(
            dataset,
            name
        )

        mode_rows.append(row)

    mode_summary = pd.DataFrame(
        mode_rows
    )

    print_separator(
        "CONFRONTO MODALITÀ SEGNALI"
    )

    print(
        mode_summary.to_string(
            index=False
        )
    )

    mode_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "signal_modes_summary.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # BASE SUMMARY
    # --------------------------------------------------------

    summary = make_summary_table(
        all_trades
    )

    print_separator(
        "SUMMARY GENERALE - ALL SIGNALS"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "summary_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # BUY / SELL
    # --------------------------------------------------------

    direction = analyze_direction(
        all_trades
    )

    print_separator(
        "BUY / SELL"
    )

    print(
        direction.to_string(
            index=False
        )
    )

    direction.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "by_direction_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # CONDITIONS
    # --------------------------------------------------------

    conditions = analyze_conditions(
        all_trades
    )

    print_top_analysis(
        conditions,
        "CONDIZIONI / COMBINAZIONI"
    )

    conditions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "condition_combinations_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # CORRELATIONS
    # --------------------------------------------------------

    correlations = feature_correlations(
        all_trades
    )

    print_top_analysis(
        correlations,
        "FEATURE CON MAGGIORE CORRELAZIONE"
    )

    correlations.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_correlations_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # TIME HORIZONS
    # --------------------------------------------------------

    horizons = analyze_time_horizons(
        all_trades
    )

    print_separator(
        "TIME HORIZONS"
    )

    print(
        horizons.to_string(
            index=False
        )
    )

    horizons.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "time_horizons_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # TARGET ANALYSIS
    # --------------------------------------------------------

    targets = target_analysis(
        df,
        all_trades
    )

    print_separator(
        "TARGET RAGGIUNTI"
    )

    print(
        targets.to_string(
            index=False
        )
    )

    targets.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "target_analysis_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # TP / SL MATRIX
    # --------------------------------------------------------

    tp_values = [
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
    ]

    sl_values = [
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
    ]

    all_signal_indices = modes["ALL"]

    matrix_dev = evaluate_matrix(
        df,
        all_signal_indices,
        "DEVELOPMENT",
        tp_values,
        sl_values
    )

    matrix_ver = evaluate_matrix(
        df,
        all_signal_indices,
        "VERIFICATION",
        tp_values,
        sl_values
    )

    matrix = pd.concat(
        [
            matrix_dev,
            matrix_ver
        ],
        ignore_index=True
    )

    print_separator(
        "MATRICE TP / SL"
    )

    print(
        matrix.to_string(
            index=False
        )
    )

    matrix.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "tp_sl_matrix_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # TP / SL FOR INDEPENDENT MODES
    # --------------------------------------------------------

    mode_matrix_rows = []

    for mode_name, mode_indices in modes.items():

        if mode_name == "ALL":
            continue

        matrix_mode = evaluate_matrix(
            df,
            mode_indices,
            "ALL",
            tp_values,
            sl_values
        )

        if not matrix_mode.empty:

            matrix_mode.insert(
                0,
                "signal_mode",
                mode_name
            )

            mode_matrix_rows.append(
                matrix_mode
            )

    if mode_matrix_rows:

        independent_matrix = pd.concat(
            mode_matrix_rows,
            ignore_index=True
        )

    else:

        independent_matrix = pd.DataFrame()

    independent_matrix.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "tp_sl_matrix_independent_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # SAME CANDLE
    # --------------------------------------------------------

    same_candle = same_candle_analysis(
        all_trades
    )

    print_separator(
        "SAME CANDLE TP + SL"
    )

    print(
        same_candle.to_string(
            index=False
        )
    )

    same_candle.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "same_candle_analysis_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # TEMPORAL BLOCKS
    # --------------------------------------------------------

    blocks = temporal_blocks(
        all_trades,
        n_blocks=6
    )

    print_separator(
        "STABILITÀ TEMPORALE - 6 BLOCCHI"
    )

    print(
        blocks.to_string(
            index=False
        )
    )

    blocks.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "temporal_blocks_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # MODE TEMPORAL BLOCKS
    # --------------------------------------------------------

    mode_block_rows = []

    for mode_name, dataset in [
        ("ALL", all_trades),
        (
            "FIRST_IN_EPISODE",
            episode_trades
        ),
        (
            "NON_OVERLAPPING",
            non_overlap_trades
        )
    ]:

        block_df = temporal_blocks(
            dataset,
            n_blocks=6
        )

        if not block_df.empty:

            block_df.insert(
                0,
                "signal_mode",
                mode_name
            )

            mode_block_rows.append(
                block_df
            )

    if mode_block_rows:

        mode_blocks = pd.concat(
            mode_block_rows,
            ignore_index=True
        )

    else:

        mode_blocks = pd.DataFrame()

    mode_blocks.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "signal_mode_temporal_blocks_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # DETAILED TRADE DATA
    # --------------------------------------------------------

    all_trades.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "diagnostic_v3_all_trades.csv"
        ),
        index=False
    )

    episode_trades.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "diagnostic_v3_first_episode.csv"
        ),
        index=False
    )

    non_overlap_trades.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "diagnostic_v3_non_overlapping.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # DEVELOPMENT / VERIFICATION FILES
    # --------------------------------------------------------

    split_time = all_trades[
        "signal_time"
    ].quantile(
        DEVELOPMENT_RATIO
    )

    development = all_trades[
        all_trades["signal_time"] <= split_time
    ]

    verification = all_trades[
        all_trades["signal_time"] > split_time
    ]

    development.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "diagnostic_v3_development.csv"
        ),
        index=False
    )

    verification.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "diagnostic_v3_verification.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # QUICK DEVELOPMENT / VERIFICATION CONDITIONS
    # --------------------------------------------------------

    dev_conditions = analyze_conditions(
        development
    )

    ver_conditions = analyze_conditions(
        verification
    )

    dev_conditions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "conditions_development_v3.csv"
        ),
        index=False
    )

    ver_conditions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "conditions_verification_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # DIRECTION × REGIME
    # --------------------------------------------------------

    regime_rows = []

    for direction_name in [
        "BUY",
        "SELL"
    ]:

        for regime_col in [
            "atr_regime",
            "ema_strength",
            "macd_strength",
            "body_bucket",
            "rsi_bucket",
            "session",
            "day_of_week",
        ]:

            for value in all_trades[
                regime_col
            ].dropna().unique():

                subset = all_trades[
                    (
                        all_trades["direction"] ==
                        direction_name
                    )
                    &
                    (
                        all_trades[regime_col] ==
                        value
                    )
                ]

                if len(subset) < 20:
                    continue

                row = summarize_trades(
                    subset,
                    f"{direction_name}_{regime_col}_{value}"
                )

                row["direction"] = direction_name
                row["feature"] = regime_col
                row["value"] = value

                regime_rows.append(row)

    direction_regimes = pd.DataFrame(
        regime_rows
    )

    direction_regimes.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "direction_regimes_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # FINAL DIAGNOSTICS
    # --------------------------------------------------------

    print_separator(
        "DIAGNOSTICA FINALE"
    )

    print(
        "Segnali totali:",
        len(modes["ALL"])
    )

    print(
        "Primo segnale per episodio:",
        len(modes["FIRST_IN_EPISODE"])
    )

    print(
        "Segnali non sovrapposti:",
        len(modes["NON_OVERLAPPING"])
    )

    print(
        "Trade ALL:",
        len(all_trades)
    )

    print(
        "Trade FIRST-IN-EPISODE:",
        len(episode_trades)
    )

    print(
        "Trade NON-OVERLAPPING:",
        len(non_overlap_trades)
    )

    print(
        "Split development:",
        split_time
    )

    print(
        "MFE mediana ALL:",
        round(
            all_trades["MFE_ATR"].median(),
            3
        )
    )

    print(
        "MAE mediana ALL:",
        round(
            all_trades["MAE_ATR"].median(),
            3
        )
    )

    # --------------------------------------------------------
    # FILE LIST
    # --------------------------------------------------------

    print_separator(
        "FILE RISULTATO"
    )

    for filename in sorted(
        os.listdir(OUTPUT_DIR)
    ):

        print(
            f" - {filename}"
        )

    elapsed = time.time() - start_time

    print_separator(
        "FINE BACKTEST V3"
    )

    print(
        f"Tempo totale: "
        f"{elapsed:.2f} secondi"
    )

    print(
        f"Risultati salvati in: "
        f"{os.path.abspath(OUTPUT_DIR)}"
    )


if __name__ == "__main__":
    main()
