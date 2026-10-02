import os
import time
import math
import warnings
from itertools import product

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAZIONE
# ============================================================

SYMBOL = "XAU/USD"
INTERVAL = "5min"

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurato")

OUTPUT_DIR = "backtest_results_v6"

TOTAL_CANDLES = 10000
BLOCK_SIZE = 5000

DEV_RATIO = 0.70

BASE_SL_ATR = 1.5
BASE_TP_ATR = 2.5

ATR_PERIOD = 14

MIN_TRADES_DEV = 30
MIN_TRADES_VER = 15

MAX_COMBINATIONS = 10000

# Orizzonti espressi in candele da 5 minuti
# 3=15m, 6=30m, 12=1h, 24=2h, 48=4h, 96=8h
HORIZONS = [3, 6, 12, 24, 48, 96, 288]

# ============================================================
# GRIGLIA DELLO SCANNER
# ============================================================

EMA_STRENGTHS = [
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
]

RSI_RANGES = [
    (0, 100),
    (30, 70),
    (30, 65),
    (35, 65),
    (35, 60),
    (40, 60),
]

ATR_REGIMES = [
    "ALL",
    "NORMAL",
    "HIGH",
]

TREND_MODES = [
    "NONE",
    "15M",
    "1H",
    "15M_1H",
]

MOMENTUM_MODES = [
    "NONE",
    "MOM6",
    "MOM12",
]

SIGNAL_MODES = [
    "ALL",
    "FIRST_IN_EPISODE",
    "NON_OVERLAPPING",
]

TP_VALUES = [
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
]

SL_VALUES = [
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
]

# ============================================================
# UTILITY
# ============================================================


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def fmt(x, digits=3):
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


# ============================================================
# DOWNLOAD TWELVE DATA
# ============================================================


def download_block(outputsize=5000, end_date=None):
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": outputsize,
        "timezone": "UTC",
        "apikey": API_KEY,
        "order": "ASC",
    }

    if end_date is not None:
        params["end_date"] = end_date

    r = requests.get(url, params=params, timeout=30)

    if r.status_code != 200:
        raise RuntimeError(
            f"Twelve Data HTTP {r.status_code}: {r.text[:500]}"
        )

    data = r.json()

    if "values" not in data:
        raise RuntimeError(
            f"Risposta Twelve Data non valida: {data}"
        )

    df = pd.DataFrame(data["values"])

    required = ["datetime", "open", "high", "low", "close"]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Colonna mancante: {col}")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["datetime", "open", "high", "low", "close"]
    )

    return df


def download_data():
    print("=" * 70)
    print("DOWNLOAD DATI")
    print("=" * 70)

    blocks = []

    first = download_block(BLOCK_SIZE)
    blocks.append(first)

    print(
        f"Blocco 1: {len(first)} candele "
        f"{first['datetime'].min()} -> {first['datetime'].max()}"
    )

    if len(first) >= BLOCK_SIZE:
        oldest = first["datetime"].min()

        second = download_block(
            BLOCK_SIZE,
            end_date=oldest.strftime("%Y-%m-%d %H:%M:%S"),
        )

        blocks.append(second)

        print(
            f"Blocco 2: {len(second)} candele "
            f"{second['datetime'].min()} -> {second['datetime'].max()}"
        )

    df = pd.concat(blocks, ignore_index=True)

    df = (
        df.drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if len(df) > TOTAL_CANDLES:
        df = df.tail(TOTAL_CANDLES).reset_index(drop=True)

    print()
    print(f"Candele finali: {len(df)}")
    print(f"Da: {df['datetime'].iloc[0]}")
    print(f"A : {df['datetime'].iloc[-1]}")

    return df


# ============================================================
# INDICATORI
# ============================================================


def add_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema100"] = close.ewm(span=100, adjust=False).mean()

    # EMA distance normalizzata
    df["ema_distance"] = (
        (df["ema20"] - df["ema50"]).abs() / close * 100
    )

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    # ATR
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.rolling(
        ATR_PERIOD
    ).mean()

    df["atr_pct"] = (
        df["atr"] / close * 100
    )

    df["atr_pct_median"] = (
        df["atr_pct"]
        .rolling(200, min_periods=50)
        .median()
    )

    # Momentum
    df["mom6"] = close.pct_change(6) * 100
    df["mom12"] = close.pct_change(12) * 100

    # Candela
    df["body"] = (close - df["open"]).abs()

    df["body_pct"] = (
        df["body"] / close * 100
    )

    df["range"] = high - low

    df["body_ratio"] = (
        df["body"] /
        df["range"].replace(0, np.nan)
    )

    # Direzione candela
    df["candle_direction"] = np.where(
        close > df["open"],
        1,
        np.where(close < df["open"], -1, 0),
    )

    # Ora UTC
    df["hour"] = df["datetime"].dt.hour

    return df


# ============================================================
# MULTI-TIMEFRAME
# ============================================================


def add_higher_timeframes(df):
    df = df.copy()

    base = df.set_index("datetime")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    # 15 minuti
    tf15 = (
        base.resample("15min")
        .agg(agg)
        .dropna()
    )

    tf15["ema20"] = tf15["close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    tf15["ema50"] = tf15["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    tf15["trend"] = np.where(
        tf15["ema20"] > tf15["ema50"],
        1,
        np.where(
            tf15["ema20"] < tf15["ema50"],
            -1,
            0,
        ),
    )

    # 1 ora
    tf1h = (
        base.resample("1h")
        .agg(agg)
        .dropna()
    )

    tf1h["ema20"] = tf1h["close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    tf1h["ema50"] = tf1h["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    tf1h["trend"] = np.where(
        tf1h["ema20"] > tf1h["ema50"],
        1,
        np.where(
            tf1h["ema20"] < tf1h["ema50"],
            -1,
            0,
        ),
    )

    # IMPORTANTISSIMO:
    # usiamo solo il timeframe superiore già CHIUSO.
    # shift(1) evita look-ahead.
    tf15 = tf15[
        ["trend"]
    ].rename(
        columns={"trend": "trend_15m"}
    )

    tf1h = tf1h[
        ["trend"]
    ].rename(
        columns={"trend": "trend_1h"}
    )

    tf15 = tf15.shift(1)
    tf1h = tf1h.shift(1)

    result = df.set_index("datetime")

    result = pd.merge_asof(
        result.sort_index(),
        tf15.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    result = pd.merge_asof(
        result.sort_index(),
        tf1h.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    result = result.reset_index()

    return result


# ============================================================
# EMA STRENGTH
# ============================================================


def ema_strength_ok(row, threshold):
    if threshold <= 0:
        return True

    return row["ema_distance"] >= threshold


# ============================================================
# ATR REGIME
# ============================================================


def atr_regime_ok(row, regime):
    if regime == "ALL":
        return True

    atr = row["atr_pct"]
    median = row["atr_pct_median"]

    if pd.isna(atr) or pd.isna(median):
        return False

    if regime == "NORMAL":
        return atr <= median

    if regime == "HIGH":
        return atr > median

    return True


# ============================================================
# TREND FILTER
# ============================================================


def trend_ok(row, direction, mode):
    if mode == "NONE":
        return True

    required = 1 if direction == "BUY" else -1

    if mode == "15M":
        return row["trend_15m"] == required

    if mode == "1H":
        return row["trend_1h"] == required

    if mode == "15M_1H":
        return (
            row["trend_15m"] == required
            and row["trend_1h"] == required
        )

    return True


# ============================================================
# MOMENTUM FILTER
# ============================================================


def momentum_ok(row, direction, mode):
    if mode == "NONE":
        return True

    if direction == "BUY":
        if mode == "MOM6":
            return row["mom6"] > 0
        if mode == "MOM12":
            return row["mom12"] > 0

    if direction == "SELL":
        if mode == "MOM6":
            return row["mom6"] < 0
        if mode == "MOM12":
            return row["mom12"] < 0

    return True


# ============================================================
# BASE SIGNAL
# ============================================================


def base_direction(row):
    ema20 = row["ema20"]
    ema50 = row["ema50"]

    macd = row["macd"]
    signal = row["macd_signal"]

    rsi = row["rsi"]

    if any(
        pd.isna(x)
        for x in [
            ema20,
            ema50,
            macd,
            signal,
            rsi,
            row["atr"],
        ]
    ):
        return None

    # BUY
    if (
        ema20 > ema50
        and macd > signal
        and 30 < rsi < 70
    ):
        return "BUY"

    # SELL
    if (
        ema20 < ema50
        and macd < signal
        and 30 < rsi < 70
    ):
        return "SELL"

    return None


# ============================================================
# SIGNAL FILTER
# ============================================================


def signal_allowed(row, direction, cfg):
    if direction is None:
        return False

    rsi_low, rsi_high = cfg["rsi_range"]

    rsi = row["rsi"]

    if pd.isna(rsi):
        return False

    if not (
        rsi_low < rsi < rsi_high
    ):
        return False

    # EMA strength
    if not ema_strength_ok(
        row,
        cfg["ema_strength"],
    ):
        return False

    # ATR regime
    if not atr_regime_ok(
        row,
        cfg["atr_regime"],
    ):
        return False

    # Higher timeframe
    if not trend_ok(
        row,
        direction,
        cfg["trend_mode"],
    ):
        return False

    # Momentum
    if not momentum_ok(
        row,
        direction,
        cfg["momentum_mode"],
    ):
        return False

    # Candela non minuscola
    if cfg["body_filter"]:
        if (
            pd.isna(row["body_ratio"])
            or row["body_ratio"] < 0.25
        ):
            return False

    return True


# ============================================================
# SIGNAL MODE
# ============================================================


def build_signals(df, cfg):
    signals = []

    last_signal_index = {
        "BUY": -10_000,
        "SELL": -10_000,
    }

    previous_direction = None

    for i in range(len(df) - 1):
        row = df.iloc[i]

        direction = base_direction(row)

        if direction is None:
            previous_direction = None
            continue

        if not signal_allowed(
            row,
            direction,
            cfg,
        ):
            previous_direction = None
            continue

        mode = cfg["signal_mode"]

        # --------------------------------------------
        # FIRST IN EPISODE
        # --------------------------------------------
        if mode == "FIRST_IN_EPISODE":
            if previous_direction == direction:
                continue

        # --------------------------------------------
        # NON OVERLAPPING
        # --------------------------------------------
        if mode == "NON_OVERLAPPING":
            if (
                i - last_signal_index[direction]
                < cfg["horizon"]
            ):
                continue

        signals.append(
            {
                "signal_index": i,
                "entry_index": i + 1,
                "direction": direction,
                "signal_time": row["datetime"],
                "entry_price": df.iloc[i + 1]["open"],
                "atr": row["atr"],
                "rsi": row["rsi"],
                "ema_distance": row["ema_distance"],
                "trend_15m": row["trend_15m"],
                "trend_1h": row["trend_1h"],
            }
        )

        last_signal_index[direction] = i
        previous_direction = direction

    return signals


# ============================================================
# SIMULAZIONE TRADE
# ============================================================


def simulate_trade(
    df,
    signal,
    tp_atr,
    sl_atr,
    horizon,
):
    entry_idx = signal["entry_index"]

    if entry_idx >= len(df):
        return None

    entry = signal["entry_price"]
    atr = signal["atr"]
    direction = signal["direction"]

    if pd.isna(entry) or pd.isna(atr):
        return None

    if atr <= 0:
        return None

    tp_distance = atr * tp_atr
    sl_distance = atr * sl_atr

    if direction == "BUY":
        tp = entry + tp_distance
        sl = entry - sl_distance
    else:
        tp = entry - tp_distance
        sl = entry + sl_distance

    end_idx = min(
        entry_idx + horizon,
        len(df) - 1,
    )

    result = "TIMEOUT"
    exit_price = df.iloc[end_idx]["close"]
    exit_idx = end_idx

    for j in range(
        entry_idx,
        end_idx + 1,
    ):
        candle = df.iloc[j]

        high = candle["high"]
        low = candle["low"]

        if direction == "BUY":
            hit_sl = low <= sl
            hit_tp = high >= tp

        else:
            hit_sl = high >= sl
            hit_tp = low <= tp

        # Se entrambi vengono toccati nella stessa candela,
        # assumiamo SL prima: scelta conservativa.
        if hit_sl and hit_tp:
            result = "SL"
            exit_price = sl
            exit_idx = j
            break

        if hit_sl:
            result = "SL"
            exit_price = sl
            exit_idx = j
            break

        if hit_tp:
            result = "TP"
            exit_price = tp
            exit_idx = j
            break

    if direction == "BUY":
        pnl = exit_price - entry
    else:
        pnl = entry - exit_price

    risk = sl_distance

    if risk <= 0:
        return None

    r_multiple = pnl / risk

    return {
        "signal_time": signal["signal_time"],
        "entry_time": df.iloc[entry_idx]["datetime"],
        "exit_time": df.iloc[exit_idx]["datetime"],
        "direction": direction,
        "entry": entry,
        "exit": exit_price,
        "tp": tp,
        "sl": sl,
        "atr": atr,
        "rsi": signal["rsi"],
        "ema_distance": signal["ema_distance"],
        "trend_15m": signal["trend_15m"],
        "trend_1h": signal["trend_1h"],
        "result": result,
        "R": r_multiple,
        "bars_held": exit_idx - entry_idx,
        "tp_atr": tp_atr,
        "sl_atr": sl_atr,
        "horizon": horizon,
    }


# ============================================================
# STATISTICHE
# ============================================================


def calculate_stats(trades):
    if trades is None or len(trades) == 0:
        return {
            "trades": 0,
            "tp": 0,
            "sl": 0,
            "timeout": 0,
            "win_rate": np.nan,
            "avg_R": np.nan,
            "total_R": np.nan,
            "PF": np.nan,
            "max_DD_R": np.nan,
        }

    r = pd.Series(
        trades["R"].values,
        dtype=float,
    )

    wins = trades[
        trades["result"] == "TP"
    ]

    losses = trades[
        trades["result"] == "SL"
    ]

    gross_profit = wins["R"].sum()
    gross_loss = abs(losses["R"].sum())

    if gross_loss > 0:
        pf = gross_profit / gross_loss
    else:
        pf = np.inf if gross_profit > 0 else 0

    equity = r.cumsum()
    running_max = equity.cummax()
    dd = running_max - equity

    max_dd = dd.max() if len(dd) else 0

    return {
        "trades": len(trades),
        "tp": int(
            (trades["result"] == "TP").sum()
        ),
        "sl": int(
            (trades["result"] == "SL").sum()
        ),
        "timeout": int(
            (trades["result"] == "TIMEOUT").sum()
        ),
        "win_rate": (
            len(wins) / len(trades) * 100
        ),
        "avg_R": r.mean(),
        "total_R": r.sum(),
        "PF": pf,
        "max_DD_R": max_dd,
    }


# ============================================================
# SCORE DI ROBUSTEZZA
# ============================================================


def robustness_score(dev, ver):
    """
    Score volutamente conservativo.

    Premi:
      - PF > 1 in entrambe le parti
      - avg_R positivo in entrambe
      - profitto positivo in entrambe
      - buon numero di trade

    Penalizza:
      - grande drawdown
      - pochi trade
      - forte differenza Dev/Verification
    """

    if (
        dev["trades"] < MIN_TRADES_DEV
        or ver["trades"] < MIN_TRADES_VER
    ):
        return -9999

    if any(
        pd.isna(x)
        for x in [
            dev["PF"],
            ver["PF"],
            dev["avg_R"],
            ver["avg_R"],
        ]
    ):
        return -9999

    score = 0.0

    # PF
    score += min(dev["PF"], 2.0) * 20
    score += min(ver["PF"], 2.0) * 35

    # Avg R
    score += max(
        -1,
        min(dev["avg_R"], 0.5),
    ) * 30

    score += max(
        -1,
        min(ver["avg_R"], 0.5),
    ) * 50

    # Total R
    score += max(
        -50,
        min(dev["total_R"], 50),
    ) * 0.10

    score += max(
        -50,
        min(ver["total_R"], 50),
    ) * 0.20

    # Stabilità Dev -> Verification
    pf_diff = abs(
        dev["PF"] - ver["PF"]
    )

    score -= min(
        pf_diff,
        2
    ) * 5

    # Drawdown
    score -= min(
        dev["max_DD_R"],
        50,
    ) * 0.20

    score -= min(
        ver["max_DD_R"],
        50,
    ) * 0.40

    # Bonus se entrambe sono positive
    if (
        dev["PF"] > 1
        and ver["PF"] > 1
    ):
        score += 25

    if (
        dev["avg_R"] > 0
        and ver["avg_R"] > 0
    ):
        score += 25

    if (
        dev["total_R"] > 0
        and ver["total_R"] > 0
    ):
        score += 15

    return score


# ============================================================
# GENERAZIONE CONFIGURAZIONI
# ============================================================


def generate_configs():
    configs = []

    # Evitiamo una combinazione mostruosa:
    # costruiamo combinazioni mirate.
    for (
        direction,
        ema_strength,
        rsi_range,
        atr_regime,
        trend_mode,
        momentum_mode,
        signal_mode,
        body_filter,
       
