import os
import time
import math
import warnings
import traceback

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

OUTPUT_DIR = "backtest_v8"

N_CANDLES = 10000
BATCH_SIZE = 5000

# ------------------------------------------------------------
# COSTI SIMULATI
# ------------------------------------------------------------

SPREAD_PER_SIDE = 0.05
SLIPPAGE_PER_SIDE = 0.05

ENTRY_COST = SPREAD_PER_SIDE + SLIPPAGE_PER_SIDE
EXIT_COST = SPREAD_PER_SIDE + SLIPPAGE_PER_SIDE

TOTAL_COST = ENTRY_COST + EXIT_COST

# ------------------------------------------------------------
# PARAMETRI SCANSIONE
# ------------------------------------------------------------

TP_VALUES = [1.5, 1.75, 2.0, 2.25, 2.5]
SL_VALUES = [0.75, 1.0, 1.25]

HORIZONS = [30, 60]

DIRECTIONS = ["BUY", "SELL"]

MODES = [
    "BASE",
    "TREND_STRONG",
    "ATR_HIGH",
    "BODY_STRONG",
    "MOMENTUM_STRONG",
    "RSI_MID",
    "TREND_1H",
    "TREND_STRONG_1H",
    "CONFLUENCE",
]

ENTRY_MODES = [
    "FIRST",
    "COOLDOWN_30",
    "COOLDOWN_60",
]

# ------------------------------------------------------------
# SPLIT
# ------------------------------------------------------------

DEV_PCT = 0.60
VAL_PCT = 0.20
TEST_PCT = 0.20

MIN_TRADES_DEV = 30
MIN_TRADES_VAL = 15
MIN_TRADES_TEST = 15


# ============================================================
# UTILITY
# ============================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# DOWNLOAD TWELVE DATA
# ============================================================

def download_batch(end_date=None, outputsize=5000):
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "timezone": "UTC",
        "format": "JSON",
    }

    if end_date is not None:
        params["end_date"] = end_date

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise RuntimeError(
            f"Errore Twelve Data: {data}"
        )

    return data["values"]


def download_data():
    print_section("DOWNLOAD DATI")

    all_values = []

    end_date = None

    while len(all_values) < N_CANDLES:

        remaining = N_CANDLES - len(all_values)
        batch_size = min(BATCH_SIZE, remaining)

        print(
            f"Scarico blocco da {end_date if end_date else 'ultimo dato'} "
            f"({batch_size} candele)..."
        )

        values = download_batch(
            end_date=end_date,
            outputsize=batch_size,
        )

        if not values:
            break

        all_values.extend(values)

        oldest = min(v["datetime"] for v in values)

        print(
            f"Ricevute: {len(values)} | "
            f"Totale grezzo: {len(all_values)} | "
            f"Oldest: {oldest}"
        )

        if len(values) < batch_size:
            break

        end_date = oldest

        time.sleep(0.5)

    df = pd.DataFrame(all_values)

    if df.empty:
        raise RuntimeError("Nessun dato scaricato.")

    df = df.drop_duplicates(subset=["datetime"])

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
    )

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=numeric_cols
    )

    df = df.sort_values("datetime")
    df = df.reset_index(drop=True)

    if len(df) > N_CANDLES:
        df = df.iloc[-N_CANDLES:].copy()
        df.reset_index(drop=True, inplace=True)

    print()
    print(f"Candele finali: {len(df)}")
    print(f"Periodo: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    return df


# ============================================================
# INDICATORI
# ============================================================

def calculate_indicators(df):
    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["EMA20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA100"] = df["close"].ewm(
        span=100,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_HIST"] = (
        df["MACD"] - df["MACD_SIGNAL"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    df["MOM6"] = (
        df["close"] - df["close"].shift(6)
    )

    df["MOM12"] = (
        df["close"] - df["close"].shift(12)
    )

    # --------------------------------------------------------
    # CANDLE BODY
    # --------------------------------------------------------

    df["BODY"] = (
        df["close"] - df["open"]
    ).abs()

    df["RANGE"] = (
        df["high"] - df["low"]
    )

    df["BODY_RATIO"] = np.where(
        df["RANGE"] > 0,
        df["BODY"] / df["RANGE"],
        0,
    )

    # --------------------------------------------------------
    # EMA STRENGTH
    # --------------------------------------------------------

    df["EMA_GAP_20_50"] = (
        df["EMA20"] - df["EMA50"]
    )

    df["EMA_GAP_50_100"] = (
        df["EMA50"] - df["EMA100"]
    )

    df["EMA_GAP_NORM"] = (
        df["EMA_GAP_20_50"].abs()
        / df["ATR"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # ATR RELATIVO
    # --------------------------------------------------------

    df["ATR_MEDIAN_100"] = (
        df["ATR"]
        .rolling(100)
        .median()
    )

    df["ATR_RATIO"] = (
        df["ATR"]
        / df["ATR_MEDIAN_100"].replace(0, np.nan)
    )

    return df


# ============================================================
# HIGHER TIMEFRAME
# ============================================================

def build_higher_timeframes(df):

    base = df.set_index("datetime")

    # --------------------------------------------------------
    # 15 MIN
    # --------------------------------------------------------

    tf15 = base.resample(
        "15min",
        label="left",
        closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()

    tf15["EMA20"] = tf15["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    tf15["EMA50"] = tf15["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    tf15["TREND"] = np.where(
        tf15["EMA20"] > tf15["EMA50"],
        "BULLISH",
        np.where(
            tf15["EMA20"] < tf15["EMA50"],
            "BEARISH",
            "NEUTRAL"
        )
    )

    tf15 = tf15.reset_index()

    # La candela 15m che parte alle 10:00
    # è disponibile solo dalle 10:15.
    tf15["available_at"] = (
        tf15["datetime"]
        + pd.Timedelta(minutes=15)
    )

    # --------------------------------------------------------
    # 1 HOUR
    # --------------------------------------------------------

    tf60 = base.resample(
        "60min",
        label="left",
        closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()

    tf60["EMA20"] = tf60["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    tf60["EMA50"] = tf60["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    tf60["TREND"] = np.where(
        tf60["EMA20"] > tf60["EMA50"],
        "BULLISH",
        np.where(
            tf60["EMA20"] < tf60["EMA50"],
            "BEARISH",
            "NEUTRAL"
        )
    )

    tf60 = tf60.reset_index()

    tf60["available_at"] = (
        tf60["datetime"]
        + pd.Timedelta(hours=1)
    )

    # --------------------------------------------------------
    # MERGE LEAKAGE-FREE
    # --------------------------------------------------------

    df = df.sort_values("datetime").reset_index(drop=True)

    tf15_merge = tf15[
        [
            "available_at",
            "TREND"
        ]
    ].rename(
        columns={
            "TREND": "TREND_15M"
        }
    ).sort_values("available_at")

    tf60_merge = tf60[
        [
            "available_at",
            "TREND"
        ]
    ].rename(
        columns={
            "TREND": "TREND_1H"
        }
    ).sort_values("available_at")

    df = pd.merge_asof(
        df.sort_values("datetime"),
        tf15_merge,
        left_on="datetime",
        right_on="available_at",
        direction="backward",
    )

    df.drop(
        columns=["available_at"],
        inplace=True,
        errors="ignore"
    )

    df = pd.merge_asof(
        df.sort_values("datetime"),
        tf60_merge,
        left_on="datetime",
        right_on="available_at",
        direction="backward",
    )

    df.drop(
        columns=["available_at"],
        inplace=True,
        errors="ignore"
    )

    return df.reset_index(drop=True)


# ============================================================
# SIGNAL
# ============================================================

def signal_matches(row, direction, mode):

    if direction == "BUY":

        base = (
            row["EMA20"] > row["EMA50"]
            and row["EMA50"] > row["EMA100"]
            and row["TREND_15M"] == "BULLISH"
            and row["MACD"] > row["MACD_SIGNAL"]
            and 30 < row["RSI"] < 65
            and row["MOM6"] > 0
        )

    else:

        base = (
            row["EMA20"] < row["EMA50"]
            and row["EMA50"] < row["EMA100"]
            and row["TREND_15M"] == "BEARISH"
            and row["MACD"] < row["MACD_SIGNAL"]
            and 35 < row["RSI"] < 70
            and row["MOM6"] < 0
        )

    if not base:
        return False

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------

    if mode == "BASE":
        return True

    # --------------------------------------------------------
    # TREND STRONG
    # --------------------------------------------------------

    if mode == "TREND_STRONG":

        if direction == "BUY":
            return (
                row["EMA20"] > row["EMA50"]
                and row["EMA50"] > row["EMA100"]
                and row["EMA_GAP_NORM"] >= 0.30
            )

        return (
            row["EMA20"] < row["EMA50"]
            and row["EMA50"] < row["EMA100"]
            and row["EMA_GAP_NORM"] >= 0.30
        )

    # --------------------------------------------------------
    # ATR HIGH
    # --------------------------------------------------------

    if mode == "ATR_HIGH":

        return (
            pd.notna(row["ATR_RATIO"])
            and row["ATR_RATIO"] >= 1.10
        )

    # --------------------------------------------------------
    # BODY STRONG
    # --------------------------------------------------------

    if mode == "BODY_STRONG":

        return (
            row["BODY_RATIO"] >= 0.60
        )

    # --------------------------------------------------------
    # MOMENTUM STRONG
    # --------------------------------------------------------

    if mode == "MOMENTUM_STRONG":

        if direction == "BUY":
            return (
                row["MOM6"] > 0
                and row["MOM12"] > 0
                and row["MACD_HIST"] > 0
            )

        return (
            row["MOM6"] < 0
            and row["MOM12"] < 0
            and row["MACD_HIST"] < 0
        )

    # --------------------------------------------------------
    # RSI MID
    # --------------------------------------------------------

    if mode == "RSI_MID":

        if direction == "BUY":
            return 45 < row["RSI"] < 60

        return 40 < row["RSI"] < 55

    # --------------------------------------------------------
    # TREND 1H
    # --------------------------------------------------------

    if mode == "TREND_1H":

        if direction == "BUY":
            return row["TREND_1H"] == "BULLISH"

        return row["TREND_1H"] == "BEARISH"

    # --------------------------------------------------------
    # TREND STRONG 1H
    # --------------------------------------------------------

    if mode == "TREND_STRONG_1H":

        if direction == "BUY":

            return (
                row["TREND_1H"] == "BULLISH"
                and row["EMA20"] > row["EMA50"]
                and row["EMA50"] > row["EMA100"]
            )

        return (
            row["TREND_1H"] == "BEARISH"
            and row["EMA20"] < row["EMA50"]
            and row["EMA50"] < row["EMA100"]
        )

    # --------------------------------------------------------
    # CONFLUENCE
    # --------------------------------------------------------

    if mode == "CONFLUENCE":

        if direction == "BUY":

            return (
                row["TREND_1H"] == "BULLISH"
                and row["TREND_15M"] == "BULLISH"
                and row["MACD"] > row["MACD_SIGNAL"]
                and row["MOM6"] > 0
                and row["RSI"] > 45
                and row["RSI"] < 60
            )

        return (
            row["TREND_1H"] == "BEARISH"
            and row["TREND_15M"] == "BEARISH"
            and row["MACD"] < row["MACD_SIGNAL"]
            and row["MOM6"] < 0
            and row["RSI"] > 40
            and row["RSI"] < 55
        )

    return False


# ============================================================
# ENTRY MODE
# ============================================================

def entry_allowed(
    idx,
    selected_entries,
    entry_mode
):

    if not selected_entries:
        return True

    last_entry = selected_entries[-1]

    if entry_mode == "FIRST":
        minimum_bars = 1

    elif entry_mode == "COOLDOWN_30":
        minimum_bars = 6

    elif entry_mode == "COOLDOWN_60":
        minimum_bars = 12

    else:
        minimum_bars = 1

    return (
        idx - last_entry >= minimum_bars
    )


# ============================================================
# SIMULAZIONE TRADE
# ============================================================

def simulate_trade(
    df,
    signal_idx,
    direction,
    tp_atr,
    sl_atr,
    horizon_min,
    split_end_idx,
):
    """
    IMPORTANTISSIMO:

    split_end_idx è ESCLUSIVO.

    Il trade non può mai leggere candele appartenenti
    alla partizione successiva.

    Questo impedisce contaminazione DEV -> VAL
    e VAL -> TEST.
    """

    entry_idx = signal_idx + 1

    # --------------------------------------------------------
    # NON POSSIAMO ENTRARE FUORI DALLO SPLIT
    # --------------------------------------------------------

    if entry_idx >= split_end_idx:
        return None

    if entry_idx >= len(df):
        return None

    signal_row = df.iloc[signal_idx]
    entry_row = df.iloc[entry_idx]

    atr = signal_row["ATR"]

    if pd.isna(atr) or atr <= 0:
        return None

    raw_entry = entry_row["open"]

    # --------------------------------------------------------
    # COSTO ENTRATA
    # --------------------------------------------------------

    if direction == "BUY":
        entry_price = raw_entry + ENTRY_COST
    else:
        entry_price = raw_entry - ENTRY_COST

    # --------------------------------------------------------
    # TARGET / STOP
    # --------------------------------------------------------

    if direction == "BUY":

        tp_price = (
            entry_price
            + tp_atr * atr
        )

        sl_price = (
            entry_price
            - sl_atr * atr
        )

    else:

        tp_price = (
            entry_price
            - tp_atr * atr
        )

        sl_price = (
            entry_price
            + sl_atr * atr
        )

    # --------------------------------------------------------
    # ORIZZONTE
    # --------------------------------------------------------

    bars = max(
        1,
        int(horizon_min / 5)
    )

    theoretical_end = entry_idx + bars

    # --------------------------------------------------------
    # CORREZIONE CRITICA V8
    #
    # NON oltrepassare lo split.
    #
    # split_end_idx è esclusivo.
    # Ultima candela utilizzabile = split_end_idx - 1
    # --------------------------------------------------------

    end_idx = min(
        theoretical_end,
        split_end_idx - 1,
        len(df) - 1
    )

    if end_idx < entry_idx:
        return None

    # --------------------------------------------------------
    # SCANSIONE CANDLE
    # --------------------------------------------------------

    for j in range(
        entry_idx,
        end_idx + 1
    ):

        row = df.iloc[j]

        high = row["high"]
        low = row["low"]

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if direction == "BUY":

            hit_tp = high >= tp_price
            hit_sl = low <= sl_price

            # Se entrambi nella stessa candela:
            # conservativo -> SL
            if hit_tp and hit_sl:

                exit_price = sl_price

                gross_move = (
                    exit_price
                    - entry_price
                )

                net_move = (
                    gross_move
                    - EXIT_COST
                )

                r = net_move / (
                    sl_atr * atr
                )

                return {
                    "result": "SL",
                    "entry_idx": entry_idx,
                    "exit_idx": j,
                    "entry": entry_price,
                    "exit": exit_price,
                    "R": r,
                }

            if hit_sl:

                exit_price = sl_price

                gross_move = (
                    exit_price
                    - entry_price
                )

                net_move = (
                    gross_move
                    - EXIT_COST
                )

                r = net_move / (
                    sl_atr * atr
                )

                return {
                    "result": "SL",
                    "entry_idx": entry_idx,
                    "exit_idx": j,
                    "entry": entry_price,
                    "exit": exit_price,
                    "R": r,
                }

            if hit_tp:

                exit_price = tp_price

                gross_move = (
                    exit_price
                    - entry_price
                )

                net_move = (
                    gross_move
                    - EXIT_COST
                )

                r = net_move / (
                    sl_atr * atr
                )

                return {
                    "result": "TP",
                    "entry_idx": entry_idx,
                    "exit_idx": j,
                    "entry": entry_price,
                    "exit": exit_price,
                    "R": r,
                }

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        else:

            hit_tp = low <= tp_price
            hit_sl = high >= sl_price

            # Conservativo
            if hit_tp and hit_sl:

                exit_price = sl_price

                gross_move = (
                    entry_price
                    - exit_price
                )

                net_move = (
                    gross_move
                    - EXIT_COST
                )

                r = net_move / (
                    sl_atr * atr
                )

                return {
                    "result": "SL",
                    "entry_idx": entry_idx,
                    "exit_idx": j,
                    "entry": entry_price,
                    "exit": exit_price,
                    "R": r,
                }

            if hit_sl:

                exit_price = sl_price

                gross_move = (
                    entry_price
                    - exit_price
                )

                net_move = (
                    gross_move
                    - EXIT_COST
                )

                r = net_move / (
                    sl_atr * atr
                )

                return {
                    "result": "SL",
                    "entry_idx": entry_idx,
                    "exit_idx": j,
                    "entry": entry_price,
                    "exit": exit_price,
                    "R": r,
                }

            if hit_tp:

                exit_price = tp_price

                gross_move = (
                    entry_price
                    - exit_price
                )

                net_move = (
                    gross_move
                    - EXIT_COST
                )

                r = net_move / (
                    sl_atr * atr
                )

                return {
                    "result": "TP",
                    "entry_idx": entry_idx,
                    "exit_idx": j,
                    "entry": entry_price,
                    "exit": exit_price,
                    "R": r,
                }

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    final_row = df.iloc[end_idx]

    raw_exit = final_row["close"]

    if direction == "BUY":
        exit_price = raw_exit - EXIT_COST

        gross_move = (
            exit_price
            - entry_price
        )

    else:
        exit_price = raw_exit + EXIT_COST

        gross_move = (
            entry_price
            - exit_price
        )

    net_move = gross_move

    r = net_move / (
        sl_atr * atr
    )

    return {
        "result": "TIMEOUT",
        "entry_idx": entry_idx,
        "exit_idx": end_idx,
        "entry": entry_price,
        "exit": exit_price,
        "R": r,
    }


# ============================================================
# STATISTICHE
# ============================================================

def calculate_stats(trades):

    if not trades:
        return {
            "trades": 0,
            "tp": 0,
            "sl": 0,
            "timeout": 0,
            "win_rate": np.nan,
            "avg_R": np.nan,
            "total_R": np.nan,
            "PF": np.nan,
            "maxDD": np.nan,
        }

    r_values = np.array(
        [t["R"] for t in trades],
        dtype=float
    )

    tp_count = sum(
        t["result"] == "TP"
        for t in trades
    )

    sl_count = sum(
        t["result"] == "SL"
        for t in trades
    )

    timeout_count = sum(
        t["result"] == "TIMEOUT"
        for t in trades
    )

    total_R = r_values.sum()

    avg_R = r_values.mean()

    wins = r_values[r_values > 0]
    losses = r_values[r_values < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    if gross_loss > 0:
        PF = gross_profit / gross_loss
    else:
        PF = np.inf

    equity = np.cumsum(r_values)

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = running_max - equity

    maxDD = drawdown.max()

    win_rate = (
        tp_count / len(trades) * 100
    )

    return {
        "trades": len(trades),
        "tp": tp_count,
        "sl": sl_count,
        "timeout": timeout_count,
        "win_rate": win_rate,
        "avg_R": avg_R,
        "total_R": total_R,
        "PF": PF,
        "maxDD": maxDD,
    }


# ============================================================
# STRATEGY SIMULATION
# ============================================================

def simulate_strategy(
    df,
    direction,
    mode,
    entry_mode,
    tp_atr,
    sl_atr,
    horizon_min,
    start_idx,
    split_end_idx,
):

    trades = []

    selected_entries = []

    # --------------------------------------------------------
    # SCANSIONE SIGNAL
    # --------------------------------------------------------

    for idx in range(
        start_idx,
        split_end_idx
    ):

        # Entry deve avere una candela successiva
        if idx + 1 >= split_end_idx:
            break

        row = df.iloc[idx]

        # Indicatori disponibili?
        required = [
            row["EMA20"],
            row["EMA50"],
            row["EMA100"],
            row["MACD"],
            row["MACD_SIGNAL"],
            row["RSI"],
            row["ATR"],
            row["MOM6"],
            row["TREND_15M"],
            row["TREND_1H"],
        ]

        if any(pd.isna(x) for x in required):
            continue

        if not signal_matches(
            row,
            direction,
            mode
        ):
            continue

        # ----------------------------------------------------
        # ENTRY FILTER
        # ----------------------------------------------------

        if not entry_allowed(
            idx,
            selected_entries,
            entry_mode
        ):
            continue

        # ----------------------------------------------------
        # TRADE
        # ----------------------------------------------------

        trade = simulate_trade(
            df=df,
            signal_idx=idx,
            direction=direction,
            tp_atr=tp_atr,
            sl_atr=sl_atr,
            horizon_min=horizon_min,
            split_end_idx=split_end_idx,
        )

        if trade is None:
            continue

        trade["signal_idx"] = idx
        trade["direction"] = direction
        trade["mode"] = mode
        trade["entry_mode"] = entry_mode
        trade["TP_ATR"] = tp_atr
        trade["SL_ATR"] = sl_atr
        trade["HORIZON_MIN"] = horizon_min
        trade["signal_time"] = row["datetime"]
        trade["entry_time"] = df.iloc[
            trade["entry_idx"]
        ]["datetime"]

        trade["exit_time"] = df.iloc[
            trade["exit_idx"]
        ]["datetime"]

        trades.append(trade)

        selected_entries.append(
            trade["entry_idx"]
        )

    return trades


# ============================================================
# ROBUSTNESS SCORE
# ============================================================

def robustness_score(
    dev,
    val,
    test=None,
):

    if dev["trades"] == 0:
        return -999999

    if val["trades"] == 0:
        return -999999

    score = 0

    # DEV
    score += (
        dev["avg_R"] * 100
    )

    score += (
        max(
            0,
            dev["PF"] - 1
        ) * 20
    )

    # VAL
    score += (
        val["avg_R"] * 150
    )

    score += (
        max(
            0,
            val["PF"] - 1
        ) * 30
    )

    # Penalità degradation
    degradation = (
        dev["avg_R"]
        - val["avg_R"]
    )

    if degradation > 0:
        score -= (
            degradation * 50
        )

    # TEST
    if test is not None:

        if test["trades"] > 0:

            score += (
                test["avg_R"] * 250
            )

            score += (
                max(
                    0,
                    test["PF"] - 1
                ) * 50
            )

            # Forte penalità se TEST peggiora
            if test["avg_R"] < 0:
                score -= 40

            if test["PF"] < 1:
                score -= 30

    return score


# ============================================================
# PARAMETRI RESULT
# ============================================================

def result_row(
    direction,
    mode,
    entry_mode,
    tp_atr,
    sl_atr,
    horizon,
    dev,
    val,
    test,
):

    dev_score = robustness_score(
        dev,
        val,
        None
    )

    total_score = robustness_score(
        dev,
        val,
        test
    )

    robust = (
        dev["trades"] >= MIN_TRADES_DEV
        and val["trades"] >= MIN_TRADES_VAL
        and dev["avg_R"] > 0
        and val["avg_R"] > 0
        and dev["PF"] > 1
        and val["PF"] > 1
    )

    robust_test = (
        robust
        and test["trades"] >= MIN_TRADES_TEST
        and test["avg_R"] > 0
        and test["PF"] > 1
    )

    test_positive = (
        test["trades"] >= MIN_TRADES_TEST
        and test["avg_R"] > 0
        and test["total_R"] > 0
    )

    return {
        "direction": direction,
        "mode": mode,
        "entry_mode": entry_mode,
        "TP_ATR": tp_atr,
        "SL_ATR": sl_atr,
        "HORIZON_MIN": horizon,

        "DEV_trades": dev["trades"],
        "DEV_TP": dev["tp"],
        "DEV_SL": dev["sl"],
        "DEV_TIMEOUT": dev["timeout"],
        "DEV_win": dev["win_rate"],
        "DEV_avg_R": dev["avg_R"],
        "DEV_total_R": dev["total_R"],
        "DEV_PF": dev["PF"],
        "DEV_DD": dev["maxDD"],

        "VAL_trades": val["trades"],
        "VAL_TP": val["tp"],
        "VAL_SL": val["sl"],
        "VAL_TIMEOUT": val["timeout"],
        "VAL_win": val["win_rate"],
        "VAL_avg_R": val["avg_R"],
        "VAL_total_R": val["total_R"],
        "VAL_PF": val["PF"],
        "VAL_DD": val["maxDD"],

        "TEST_trades": test["trades"],
        "TEST_TP": test["tp"],
        "TEST_SL": test["sl"],
        "TEST_TIMEOUT": test["timeout"],
        "TEST_win": test["win_rate"],
        "TEST_avg_R": test["avg_R"],
        "TEST_total_R": test["total_R"],
        "TEST_PF": test["PF"],
        "TEST_DD": test["maxDD"],

        "DEV_VAL_score": dev_score,
        "ROBUST_SCORE": total_score,

        "ROBUST_DEV_VAL": robust,
        "ROBUST_DEV_VAL_TEST": robust_test,
        "TEST_POSITIVE": test_positive,
    }


# ============================================================
# MAIN SCANNER
# ============================================================

def main():

    ensure_output_dir()

    print_section(
        "BACKTEST V8 - ROBUSTNESS SCANNER"
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    df = download_data()

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    print_section(
        "CALCOLO INDICATORI"
    )

    df = calculate_indicators(df)

    df = build_higher_timeframes(df)

    # --------------------------------------------------------
    # SALVA DATASET
    # --------------------------------------------------------

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "candles_with_indicators.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    n = len(df)

    dev_end = int(
        n * DEV_PCT
    )

    val_end = int(
        n * (DEV_PCT + VAL_PCT)
    )

    test_end = n

    print()
    print("SPLIT")
    print("-" * 78)

    print(
        f"DEV   : 0 -> {dev_end - 1} "
        f"({dev_end} candles)"
    )

    print(
        f"VAL   : {dev_end} -> {val_end - 1} "
        f"({val_end - dev_end} candles)"
    )

    print(
        f"TEST  : {val_end} -> {test_end - 1} "
        f"({test_end - val_end} candles)"
    )

    print()
    print(
        "CORREZIONE V8: ogni trade è obbligato "
        "a chiudere dentro il proprio split."
    )

    # --------------------------------------------------------
    # SCANSIONE
    # --------------------------------------------------------

    total_combinations = (
        len(DIRECTIONS)
        * len(MODES)
        * len(ENTRY_MODES)
        * len(TP_VALUES)
        * len(SL_VALUES)
        * len(HORIZONS)
    )

    print()
    print(
        f"Combinazioni da testare: "
        f"{total_combinations}"
    )

    all_results = []
    all_trades = []

    counter = 0

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    for direction in DIRECTIONS:

        for mode in MODES:

            for entry_mode in ENTRY_MODES:

                for tp_atr in TP_VALUES:

                    for sl_atr in SL_VALUES:

                        for horizon in HORIZONS:

                            counter += 1

                            if counter % 50 == 0:
                                print(
                                    f"Progress: "
                                    f"{counter}/"
                                    f"{total_combinations}"
                                )

                            # =================================
                            # DEV
                            # =================================

                            dev_trades = simulate_strategy(
                                df=df,
                                direction=direction,
                                mode=mode,
                                entry_mode=entry_mode,
                                tp_atr=tp_atr,
                                sl_atr=sl_atr,
                                horizon_min=horizon,
                                start_idx=0,
                                split_end_idx=dev_end,
                            )

                            dev_stats = calculate_stats(
                                dev_trades
                            )

                            # =================================
                            # VAL
                            # =================================

                            val_trades = simulate_strategy(
                                df=df,
                                direction=direction,
                                mode=mode,
                                entry_mode=entry_mode,
                                tp_atr=tp_atr,
                                sl_atr=sl_atr,
                                horizon_min=horizon,
                                start_idx=dev_end,
                                split_end_idx=val_end,
                            )

                            val_stats = calculate_stats(
                                val_trades
                            )

                            # =================================
                            # TEST
                            # =================================

                            test_trades = simulate_strategy(
                                df=df,
                                direction=direction,
                                mode=mode,
                                entry_mode=entry_mode,
                                tp_atr=tp_atr,
                                sl_atr=sl_atr,
                                horizon_min=horizon,
                                start_idx=val_end,
                                split_end_idx=test_end,
                            )

                            test_stats = calculate_stats(
                                test_trades
                            )

                            # =================================
                            # RESULT
                            # =================================

                            result = result_row(
                                direction=direction,
                                mode=mode,
                                entry_mode=entry_mode,
                                tp_atr=tp_atr,
                                sl_atr=sl_atr,
                                horizon=horizon,
                                dev=dev_stats,
                                val=val_stats,
                                test=test_stats,
                            )

                            all_results.append(
                                result
                            )

                            # =================================
                            # SAVE TRADES
                            # =================================

                            for t in (
                                dev_trades
                                + val_trades
                                + test_trades
                            ):

                                t["split"] = (
                                    "DEV"
                                    if t["signal_idx"] < dev_end
                                    else
                                    "VAL"
                                    if t["signal_idx"] < val_end
                                    else
                                    "TEST"
                                )

                                all_trades.append(
                                    t
                                )

    # --------------------------------------------------------
    # DATAFRAME RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    trades_df = pd.DataFrame(
        all_trades
    )

    # --------------------------------------------------------
    # SAVE ALL
    # --------------------------------------------------------

    scan_path = os.path.join(
        OUTPUT_DIR,
        "scan_all.csv"
    )

    results_df.to_csv(
        scan_path,
        index=False
    )

    trades_path = os.path.join(
        OUTPUT_DIR,
        "trades_all.csv"
    )

    trades_df.to_csv(
        trades_path,
        index=False
    )

    # --------------------------------------------------------
    # ROBUST DEV + VAL
    # --------------------------------------------------------

    robust_dev_val = results_df[
        results_df["ROBUST_DEV_VAL"]
        == True
    ].copy()

    robust_dev_val = robust_dev_val.sort_values(
        "ROBUST_SCORE",
        ascending=False
    )

    robust_dev_val.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "robust_dev_val.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # ROBUST DEV + VAL + TEST
    # --------------------------------------------------------

    robust_dev_val_test = results_df[
        results_df["ROBUST_DEV_VAL_TEST"]
        == True
    ].copy()

    robust_dev_val_test = robust_dev_val_test.sort_values(
        "ROBUST_SCORE",
        ascending=False
    )

    robust_dev_val_test.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "robust_dev_val_test.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # TEST POSITIVE
    # --------------------------------------------------------

    test_positive = results_df[
        results_df["TEST_POSITIVE"]
        == True
    ].copy()

    test_positive = test_positive.sort_values(
        [
            "TEST_avg_R",
            "TEST_PF",
        ],
        ascending=False
    )

    test_positive.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_positive.csv"
        ),
        index=False
    )

    # ========================================================
    # MODE SUMMARY TEST
    # ========================================================

    mode_summary = (
        results_df
        .groupby(
            [
                "direction",
                "mode",
            ],
            dropna=False
        )
        .agg(
            combinations=("TEST_trades", "count"),
            avg_test_R=("TEST_avg_R", "mean"),
            median_test_R=("TEST_avg_R", "median"),
            avg_test_PF=("TEST_PF", "mean"),
            positive_test_pct=(
                "TEST_POSITIVE",
                "mean"
            ),
            total_test_R=(
                "TEST_total_R",
                "mean"
            ),
        )
        .reset_index()
    )

    mode_summary[
        "positive_test_pct"
    ] *= 100

    mode_summary = mode_summary.sort_values(
        [
            "avg_test_R",
            "avg_test_PF",
        ],
        ascending=False
    )

    mode_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "mode_summary_test.csv"
        ),
        index=False
    )

    # ========================================================
    # OUTPUT CONSOLE
    # ========================================================

    print_section(
        "TOP 30 DEV + VALIDATION"
    )

    cols = [
        "direction",
        "mode",
        "entry_mode",
        "TP_ATR",
        "SL_ATR",
        "HORIZON_MIN",
        "DEV_trades",
        "DEV_avg_R",
        "DEV_PF",
        "VAL_trades",
        "VAL_avg_R",
        "VAL_PF",
        "TEST_trades",
        "TEST_avg_R",
        "TEST_PF",
        "ROBUST_SCORE",
    ]

    top_dev_val = results_df.sort_values(
        "DEV_VAL_score",
        ascending=False
    ).head(30)

    print(
        top_dev_val[
            cols
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # TOP 30 TEST
    # ========================================================

    print_section(
        "TOP 30 TEST"
    )

    top_test = results_df.sort_values(
        [
            "TEST_avg_R",
            "TEST_PF",
            "TEST_total_R",
        ],
        ascending=False
    ).head(30)

    print(
        top_test[
            cols
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # ROBUST DEV + VAL
    # ========================================================

    print_section(
        "ROBUST DEV + VALIDATION"
    )

    if robust_dev_val.empty:

        print(
            "NESSUNA COMBINAZIONE supera "
            "i criteri DEV + VAL."
        )

    else:

        print(
            robust_dev_val[
                cols
            ].head(30).to_string(
                index=False
            )
        )

    # ========================================================
    # ROBUST DEV + VAL + TEST
    # ========================================================

    print_section(
        "ROBUST DEV + VALIDATION + TEST"
    )

    if robust_dev_val_test.empty:

        print(
            "NESSUNA COMBINAZIONE supera "
            "tutti i criteri inclusa la TEST."
        )

    else:

        print(
            robust_dev_val_test[
                cols
            ].head(30).to_string(
                index=False
            )
        )

    # ========================================================
    # TEST POSITIVE
    # ========================================================

    print_section(
        "TEST POSITIVE"
    )

    if test_positive.empty:

        print(
            "Nessuna combinazione con "
            "TEST positiva secondo i criteri."
        )

    else:

        print(
            test_positive[
                cols
            ].head(30).to_string(
                index=False
            )
        )

    # ========================================================
    # MODE SUMMARY
    # ========================================================

    print_section(
        "MODE SUMMARY TEST"
    )

    print(
        mode_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_section(
        "FILE GENERATI"
    )

    print(
        f"- {scan_path}"
    )

    print(
        f"- {os.path.join(OUTPUT_DIR, 'robust_dev_val.csv')}"
    )

    print(
        f"- {os.path.join(OUTPUT_DIR, 'robust_dev_val_test.csv')}"
    )

    print(
        f"- {os.path.join(OUTPUT_DIR, 'test_positive.csv')}"
    )

    print(
        f"- {os.path.join(OUTPUT_DIR, 'mode_summary_test.csv')}"
    )

    print(
        f"- {trades_path}"
    )

    print(
        f"- {os.path.join(OUTPUT_DIR, 'candles_with_indicators.csv')}"
    )

    print()
    print(
        "BACKTEST V8 COMPLETATO."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as e:

        print()
        print("=" * 78)
        print("ERRORE BACKTEST V8")
        print("=" * 78)

        print(
            repr(e)
        )

        traceback.print_exc()

        raise
