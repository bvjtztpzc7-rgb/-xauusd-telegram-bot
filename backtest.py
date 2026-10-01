import os
import time
import math
import requests
import warnings
import traceback

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

BASE_TP_ATR = 2.5
BASE_SL_ATR = 1.5

MFE_MAE_HORIZON = 288          # 24h su candele da 5m
DEVELOPMENT_PCT = 0.70

BLOCK_SIZE = 5000
TOTAL_CANDLES = 10000

OUTPUT_DIR = "backtest_results_v3"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PARAMETRI
# ============================================================

TP_VALUES = [
    0.5, 1.0, 1.5, 2.0,
    2.5, 3.0, 3.5, 4.0
]

SL_VALUES = [
    0.5, 1.0, 1.5,
    2.0, 2.5, 3.0
]

SIGNAL_MODES = [
    "ALL",
    "FIRST_IN_EPISODE",
    "NON_OVERLAPPING"
]


# ============================================================
# UTILITY
# ============================================================

def ensure_dataframe(obj):
    """
    Protezione generale contro il bug:
    numpy.ndarray -> DataFrame
    """

    if obj is None:
        return pd.DataFrame()

    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    if isinstance(obj, pd.Series):
        return obj.to_frame().T

    if isinstance(obj, np.ndarray):

        if obj.size == 0:
            return pd.DataFrame()

        if obj.ndim == 1:
            return pd.DataFrame([obj])

        return pd.DataFrame(obj)

    if isinstance(obj, list):

        if len(obj) == 0:
            return pd.DataFrame()

        return pd.DataFrame(obj)

    return pd.DataFrame(obj)


def safe_float(value, default=np.nan):

    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):

    try:
        return int(value)
    except Exception:
        return default


def profit_factor_from_results(results):

    results = np.asarray(results, dtype=float)

    if len(results) == 0:
        return np.nan

    positive = results[results > 0].sum()
    negative = abs(results[results < 0].sum())

    if negative == 0:
        if positive > 0:
            return np.inf
        return np.nan

    return positive / negative


def max_drawdown(results):

    results = np.asarray(results, dtype=float)

    if len(results) == 0:
        return 0.0

    equity = np.cumsum(results)
    running_max = np.maximum.accumulate(np.concatenate([[0], equity]))
    equity2 = np.concatenate([[0], equity])

    drawdown = running_max - equity2

    return float(np.max(drawdown))


def print_separator(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# DOWNLOAD TWELVE DATA
# ============================================================

def download_block(end_date=None):

    url = "https://api.twelvedata.com/time_series"

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
        k: ("***" if k == "apikey" else v)
        for k, v in params.items()
    })

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "values" not in data:

        print("RISPOSTA API:")
        print(data)

        raise RuntimeError(
            "Twelve Data non ha restituito 'values'."
        )

    df = pd.DataFrame(data["values"])

    if df.empty:
        raise RuntimeError("Blocco storico vuoto.")

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close"
    ]

    for col in required:

        if col not in df.columns:
            raise RuntimeError(
                f"Colonna mancante: {col}"
            )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    for col in [
        "open",
        "high",
        "low",
        "close"
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "datetime",
            "open",
            "high",
            "low",
            "close"
        ]
    )

    df = df.sort_values("datetime")

    df = df.drop_duplicates(
        subset=["datetime"]
    )

    df = df.reset_index(drop=True)

    print(
        f"Ricevute {len(df)} candele | "
        f"{df['datetime'].iloc[0]} -> "
        f"{df['datetime'].iloc[-1]}"
    )

    return df


def download_history():

    print_separator("DOWNLOAD STORICO")

    print_separator("DOWNLOAD BLOCCO 1/2")

    block1 = download_block()

    time.sleep(1)

    oldest = block1["datetime"].min()

    end_date = (
        oldest - pd.Timedelta(minutes=5)
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print_separator("DOWNLOAD BLOCCO 2/2")

    block2 = download_block(
        end_date=end_date
    )

    df = pd.concat(
        [block1, block2],
        ignore_index=True
    )

    df = df.sort_values(
        "datetime"
    )

    df = df.drop_duplicates(
        subset=["datetime"],
        keep="first"
    )

    df = df.reset_index(drop=True)

    if len(df) > TOTAL_CANDLES:

        df = df.tail(
            TOTAL_CANDLES
        ).reset_index(drop=True)

    print_separator("DATI STORICI")

    print(
        f"Candele totali: {len(df)}"
    )

    print(
        f"Periodo: "
        f"{df['datetime'].iloc[0]} -> "
        f"{df['datetime'].iloc[-1]}"
    )

    return df


# ============================================================
# INDICATORI
# ============================================================

def calculate_indicators(df):

    print_separator("CALCOLO INDICATORI")

    df = ensure_dataframe(df)

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema20"] = (
        df["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = (
        df["close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["macd"] = ema12 - ema26

    df["macd_signal"] = (
        df["macd"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
    )

    avg_loss = (
        loss.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
    )

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["rsi"] = (
        100 -
        (100 / (1 + rs))
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = (
        df["close"].shift(1)
    )

    tr1 = (
        df["high"] -
        df["low"]
    )

    tr2 = (
        df["high"] -
        previous_close
    ).abs()

    tr3 = (
        df["low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        true_range
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # PRICE FEATURES
    # --------------------------------------------------------

    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["body_atr"] = (
        df["body"] /
        df["atr"].replace(0, np.nan)
    )

    df["ema_distance"] = (
        (df["ema20"] - df["ema50"])
        / df["atr"].replace(0, np.nan)
    )

    df["ema20_slope"] = (
        df["ema20"] -
        df["ema20"].shift(5)
    ) / df["atr"].replace(
        0,
        np.nan
    )

    df["ema50_slope"] = (
        df["ema50"] -
        df["ema50"].shift(5)
    ) / df["atr"].replace(
        0,
        np.nan
    )

    df["atr_ratio"] = (
        df["atr"] /
        df["atr"].rolling(50).mean()
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    df["momentum_3"] = (
        df["close"] -
        df["close"].shift(3)
    ) / df["atr"].replace(
        0,
        np.nan
    )

    df["momentum_6"] = (
        df["close"] -
        df["close"].shift(6)
    ) / df["atr"].replace(
        0,
        np.nan
    )

    df["momentum_12"] = (
        df["close"] -
        df["close"].shift(12)
    ) / df["atr"].replace(
        0,
        np.nan
    )

    df["momentum_24"] = (
        df["close"] -
        df["close"].shift(24)
    ) / df["atr"].replace(
        0,
        np.nan
    )

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    df["hour_utc"] = (
        df["datetime"].dt.hour
    )

    df["day_of_week"] = (
        df["datetime"].dt.dayofweek
    )

    return df


# ============================================================
# MULTI-TIMEFRAME
# ============================================================

def add_higher_timeframes(df):

    print_separator("COSTRUZIONE 15M + 1H")

    df = ensure_dataframe(df)

    base = df.copy()

    base = base.set_index(
        "datetime"
    )

    # --------------------------------------------------------
    # 15 MIN
    # --------------------------------------------------------

    tf15 = base.resample(
        "15min",
        label="right",
        closed="right"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })

    tf15["ema20_15"] = (
        tf15["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    tf15["ema50_15"] = (
        tf15["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    tf15["bullish_15"] = (
        tf15["ema20_15"] >
        tf15["ema50_15"]
    )

    tf15["bearish_15"] = (
        tf15["ema20_15"] <
        tf15["ema50_15"]
    )

    # --------------------------------------------------------
    # 1 HOUR
    # --------------------------------------------------------

    tf1h = base.resample(
        "1h",
        label="right",
        closed="right"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })

    tf1h["ema20_1h"] = (
        tf1h["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    tf1h["ema50_1h"] = (
        tf1h["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    tf1h["bullish_1h"] = (
        tf1h["ema20_1h"] >
        tf1h["ema50_1h"]
    )

    tf1h["bearish_1h"] = (
        tf1h["ema20_1h"] <
        tf1h["ema50_1h"]
    )

    # --------------------------------------------------------
    # SHIFT HTF BY ONE CLOSED BAR
    #
    # Evita di usare una candela HTF ancora in formazione.
    # --------------------------------------------------------

    tf15 = tf15.shift(1)

    tf1h = tf1h.shift(1)

    tf15 = tf15[
        [
            "ema20_15",
            "ema50_15",
            "bullish_15",
            "bearish_15"
        ]
    ]

    tf1h = tf1h[
        [
            "ema20_1h",
            "ema50_1h",
            "bullish_1h",
            "bearish_1h"
        ]
    ]

    base = base.join(
        tf15,
        how="left"
    )

    base = base.join(
        tf1h,
        how="left"
    )

    base = base.reset_index()

    return base


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signals(df):

    print_separator("GENERAZIONE SEGNALI BASE")

    df = ensure_dataframe(df).copy()

    # --------------------------------------------------------
    # BASE STRATEGY
    # --------------------------------------------------------

    common = (
        df["rsi"].between(
            30,
            65,
            inclusive="both"
        )
        &
        df["atr"].notna()
        &
        (df["atr"] > 0)
        &
        df["ema20"].notna()
        &
        df["ema50"].notna()
        &
        df["macd"].notna()
        &
        df["macd_signal"].notna()
        &
        df["bullish_15"].notna()
    )

    buy = (
        common
        &
        (df["ema20"] > df["ema50"])
        &
        (df["macd"] > df["macd_signal"])
        &
        df["bullish_15"]
    )

    sell = (
        common
        &
        (df["ema20"] < df["ema50"])
        &
        (df["macd"] < df["macd_signal"])
        &
        df["bearish_15"]
    )

    df["signal"] = 0

    df.loc[
        buy,
        "signal"
    ] = 1

    df.loc[
        sell,
        "signal"
    ] = -1

    df["direction"] = np.select(
        [
            df["signal"] == 1,
            df["signal"] == -1
        ],
        [
            "BUY",
            "SELL"
        ],
        default=""
    )

    df["signal_id"] = np.where(
        df["signal"] != 0,
        np.arange(len(df)),
        -1
    )

    print(
        f"BUY/SELL totali: "
        f"{int((df['signal'] != 0).sum())}"
    )

    print(
        f"BUY: "
        f"{int((df['signal'] == 1).sum())}"
    )

    print(
        f"SELL: "
        f"{int((df['signal'] == -1).sum())}"
    )

    return df


# ============================================================
# SIGNAL MODES
# ============================================================

def first_in_episode(df):

    df = ensure_dataframe(df).copy()

    signal_rows = df[
        df["signal"] != 0
    ].copy()

    if signal_rows.empty:
        return signal_rows

    previous_signal = (
        signal_rows["signal"]
        .shift(1)
    )

    keep = (
        previous_signal.isna()
        |
        (
            previous_signal !=
            signal_rows["signal"]
        )
    )

    return signal_rows[
        keep
    ].copy()


# ============================================================
# SINGLE TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    entry_index,
    direction,
    tp_atr=BASE_TP_ATR,
    sl_atr=BASE_SL_ATR,
    horizon=None,
    same_candle_mode="CONSERVATIVE"
):

    if horizon is None:
        horizon = len(df)

    if entry_index >= len(df) - 1:
        return None

    row = df.iloc[
        entry_index
    ]

    entry_price = float(
        row["close"]
    )

    atr = float(
        row["atr"]
    )

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
        len(df) - 1,
        entry_index + horizon
    )

    mfe = 0.0
    mae = 0.0

    target_reach_time = None

    outcome = "TIMEOUT"
    exit_index = end_index
    exit_price = float(
        df.iloc[end_index]["close"]
    )

    for j in range(
        entry_index + 1,
        end_index + 1
    ):

        candle = df.iloc[j]

        high = float(
            candle["high"]
        )

        low = float(
            candle["low"]
        )

        if direction == "BUY":

            favorable = (
                high - entry_price
            ) / atr

            adverse = (
                entry_price - low
            ) / atr

            tp_hit = (
                high >= tp_price
            )

            sl_hit = (
                low <= sl_price
            )

        else:

            favorable = (
                entry_price - low
            ) / atr

            adverse = (
                high - entry_price
            ) / atr

            tp_hit = (
                low <= tp_price
            )

            sl_hit = (
                high >= sl_price
            )

        mfe = max(
            mfe,
            favorable
        )

        mae = max(
            mae,
            adverse
        )

        # ----------------------------------------------------
        # TIME TO TARGET
        # ----------------------------------------------------

        if (
            target_reach_time is None
            and favorable >= tp_atr
        ):

            target_reach_time = (
                j - entry_index
            ) * 5

        # ----------------------------------------------------
        # SAME CANDLE
        # ----------------------------------------------------

        if tp_hit and sl_hit:

            if same_candle_mode == "CONSERVATIVE":

                outcome = "SL"
                exit_index = j
                exit_price = sl_price
                break

            elif same_candle_mode == "OPTIMISTIC":

                outcome = "TP"
                exit_index = j
                exit_price = tp_price
                break

            elif same_candle_mode == "EXCLUDE":

                outcome = "EXCLUDE"
                exit_index = j
                exit_price = np.nan
                break

        elif tp_hit:

            outcome = "TP"
            exit_index = j
            exit_price = tp_price
            break

        elif sl_hit:

            outcome = "SL"
            exit_index = j
            exit_price = sl_price
            break

    # --------------------------------------------------------
    # RESULT R
    # --------------------------------------------------------

    if outcome == "TP":

        result_r = float(
            tp_atr / sl_atr
        )

    elif outcome == "SL":

        result_r = -1.0

    elif outcome == "EXCLUDE":

        result_r = np.nan

    else:

        if direction == "BUY":

            result_r = (
                exit_price -
                entry_price
            ) / (
                sl_atr * atr
            )

        else:

            result_r = (
                entry_price -
                exit_price
            ) / (
                sl_atr * atr
            )

    return {
        "entry_index": entry_index,
        "exit_index": exit_index,
        "datetime": row["datetime"],
        "exit_datetime": df.iloc[
            exit_index
        ]["datetime"],
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "atr": atr,
        "tp_atr": tp_atr,
        "sl_atr": sl_atr,
        "outcome": outcome,
        "result_R": result_r,
        "MFE_ATR": mfe,
        "MAE_ATR": mae,
        "minutes_to_target": target_reach_time,
        "same_candle": bool(
            outcome in [
                "TP",
                "SL"
            ]
        )
    }


# ============================================================
# BUILD BASE TRADES
# ============================================================

def build_trades(
    df,
    signals,
    tp_atr=BASE_TP_ATR,
    sl_atr=BASE_SL_ATR,
    same_candle_mode="CONSERVATIVE"
):

    df = ensure_dataframe(df)
    signals = ensure_dataframe(signals)

    if signals.empty:
        return pd.DataFrame()

    trades = []

    for _, signal_row in signals.iterrows():

        entry_index = int(
            signal_row["_entry_index"]
        )

        direction = (
            "BUY"
            if signal_row["signal"] == 1
            else "SELL"
        )

        trade = simulate_trade(
            df=df,
            entry_index=entry_index,
            direction=direction,
            tp_atr=tp_atr,
            sl_atr=sl_atr,
            horizon=MFE_MAE_HORIZON,
            same_candle_mode=same_candle_mode
        )

        if trade is not None:
            trades.append(
                trade
            )

    return ensure_dataframe(
        trades
    )


# ============================================================
# ALL SIGNALS
# ============================================================

def get_all_signals(df):

    signals = df[
        df["signal"] != 0
    ].copy()

    signals["_entry_index"] = (
        signals.index
    )

    return signals


# ============================================================
# FIRST EPISODE
# ============================================================

def get_first_episode_signals(df):

    signals = first_in_episode(
        df
    )

    if signals.empty:
        return signals

    signals["_entry_index"] = (
        signals.index
    )

    return signals


# ============================================================
# NON OVERLAPPING
# ============================================================

def get_non_overlapping_signals(df):

    all_signals = get_all_signals(
        df
    )

    if all_signals.empty:
        return all_signals

    selected = []

    next_allowed_index = -1

    for _, row in all_signals.iterrows():

        entry_index = int(
            row["_entry_index"]
        )

        if entry_index < next_allowed_index:
            continue

        selected.append(
            row
        )

        direction = (
            "BUY"
            if row["signal"] == 1
            else "SELL"
        )

        trade = simulate_trade(
            df=df,
            entry_index=entry_index,
            direction=direction,
            tp_atr=BASE_TP_ATR,
            sl_atr=BASE_SL_ATR,
            horizon=MFE_MAE_HORIZON,
            same_candle_mode="CONSERVATIVE"
        )

        if trade is None:
            continue

        exit_index = int(
            trade["exit_index"]
        )

        next_allowed_index = (
            exit_index + 1
        )

    return ensure_dataframe(
        selected
    )


# ============================================================
# BUILD SIGNAL MODES
# ============================================================

def build_signal_modes(df):

    print_separator(
        "COSTRUZIONE MODALITÀ SEGNALI"
    )

    all_signals = get_all_signals(
        df
    )

    first_signals = (
        get_first_episode_signals(
            df
        )
    )

    non_overlap = (
        get_non_overlapping_signals(
            df
        )
    )

    print(
        f"ALL: {len(all_signals)}"
    )

    print(
        f"FIRST-IN-EPISODE: "
        f"{len(first_signals)}"
    )

    print(
        f"NON-OVERLAPPING: "
        f"{len(non_overlap)}"
    )

    return {
        "ALL": all_signals,
        "FIRST_IN_EPISODE": first_signals,
        "NON_OVERLAPPING": non_overlap
    }


# ============================================================
# SUMMARY
# ============================================================

def summarize_trades(
    trades,
    dataset="ALL",
    direction="ALL"
):

    # --------------------------------------------------------
    # FIX PRINCIPALE DEL BUG V3
    # --------------------------------------------------------

    trades = ensure_dataframe(
        trades
    )

    if trades.empty:

        return {
            "dataset": dataset,
            "trades": 0,
            "TP": 0,
            "SL": 0,
            "TIMEOUT": 0,
            "win_rate_pct": np.nan,
            "avg_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": 0.0,
            "avg_MFE_ATR": np.nan,
            "avg_MAE_ATR": np.nan,
            "median_MFE_ATR": np.nan,
            "median_MAE_ATR": np.nan
        }

    if (
        direction != "ALL"
        and "direction" in trades.columns
    ):

        trades = trades[
            trades["direction"] ==
            direction
        ].copy()

    if trades.empty:

        return {
            "dataset": dataset,
            "trades": 0,
            "TP": 0,
            "SL": 0,
            "TIMEOUT": 0,
            "win_rate_pct": np.nan,
            "avg_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": 0.0,
            "avg_MFE_ATR": np.nan,
            "avg_MAE_ATR": np.nan,
            "median_MFE_ATR": np.nan,
            "median_MAE_ATR": np.nan
        }

    tp = int(
        (trades["outcome"] == "TP").sum()
    )

    sl = int(
        (trades["outcome"] == "SL").sum()
    )

    timeout = int(
        (trades["outcome"] == "TIMEOUT").sum()
    )

    total = len(trades)

    win_rate = (
        tp /
        (tp + sl) * 100
        if (tp + sl) > 0
        else np.nan
    )

    results = pd.to_numeric(
        trades["result_R"],
        errors="coerce"
    ).dropna()

    total_r = (
        float(results.sum())
        if len(results)
        else 0.0
    )

    avg_r = (
        float(results.mean())
        if len(results)
        else np.nan
    )

    pf = profit_factor_from_results(
        results.values
    )

    dd = max_drawdown(
        results.values
    )

    return {
        "dataset": dataset,
        "trades": total,
        "TP": tp,
        "SL": sl,
        "TIMEOUT": timeout,
        "win_rate_pct": win_rate,
        "avg_R": avg_r,
        "total_R": total_r,
        "profit_factor": pf,
        "max_drawdown_R": dd,
        "avg_MFE_ATR": trades[
            "MFE_ATR"
        ].mean(),
        "avg_MAE_ATR": trades[
            "MAE_ATR"
        ].mean(),
        "median_MFE_ATR": trades[
            "MFE_ATR"
        ].median(),
        "median_MAE_ATR": trades[
            "MAE_ATR"
        ].median()
    }


# ============================================================
# ADDITIONAL TRADE FEATURES
# ============================================================

def enrich_trades(
    trades,
    df
):

    trades = ensure_dataframe(
        trades
    )

    if trades.empty:
        return trades

    df = ensure_dataframe(df)

    feature_cols = [
        "rsi",
        "atr_ratio",
        "body_atr",
        "ema_distance",
        "ema50_slope",
        "ema20_slope",
        "momentum_3",
        "momentum_6",
        "momentum_12",
        "momentum_24",
        "macd_hist",
        "hour_utc",
        "day_of_week"
    ]

    available = [
        c for c in feature_cols
        if c in df.columns
    ]

    feature_data = df[
        ["datetime"] + available
    ].copy()

    trades = trades.merge(
        feature_data,
        on="datetime",
        how="left"
    )

    return trades


# ============================================================
# SUMMARY SIGNAL MODES
# ============================================================

def analyze_signal_modes(
    df,
    signal_modes
):

    rows = []

    for mode, signals in signal_modes.items():

        trades = build_trades(
            df,
            signals
        )

        trades = enrich_trades(
            trades,
            df
        )

        trades.to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"diagnostic_v3_{mode.lower()}.csv"
            ),
            index=False
        )

        row = summarize_trades(
            trades,
            dataset=mode
        )

        rows.append(row)

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "signal_modes_v3.csv"
        ),
        index=False
    )

    return result


# ============================================================
# DEVELOPMENT / VERIFICATION
# ============================================================

def split_signals(
    signals,
    split_datetime
):

    signals = ensure_dataframe(
        signals
    )

    if signals.empty:
        return (
            signals.copy(),
            signals.copy()
        )

    development = signals[
        signals["datetime"] <=
        split_datetime
    ].copy()

    verification = signals[
        signals["datetime"] >
        split_datetime
    ].copy()

    return (
        development,
        verification
    )


def split_trades(
    trades,
    split_datetime
):

    trades = ensure_dataframe(
        trades
    )

    if trades.empty:
        return (
            trades.copy(),
            trades.copy()
        )

    development = trades[
        trades["datetime"] <=
        split_datetime
    ].copy()

    verification = trades[
        trades["datetime"] >
        split_datetime
    ].copy()

    return (
        development,
        verification
    )


# ============================================================
# GENERAL SUMMARY
# ============================================================

def general_summary(
    df,
    signals
):

    trades = build_trades(
        df,
        signals
    )

    trades = enrich_trades(
        trades,
        df
    )

    split_datetime = (
        df["datetime"]
        .quantile(
            DEVELOPMENT_PCT
        )
    )

    development, verification = (
        split_trades(
            trades,
            split_datetime
        )
    )

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

    rows = [
        summarize_trades(
            development,
            "DEVELOPMENT"
        ),
        summarize_trades(
            verification,
            "VERIFICATION"
        ),
        summarize_trades(
            trades,
            "ALL"
        )
    ]

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "summary_v3.csv"
        ),
        index=False
    )

    return (
        trades,
        development,
        verification,
        split_datetime,
        result
    )


# ============================================================
# BUY / SELL
# ============================================================

def direction_analysis(
    trades,
    split_datetime
):

    rows = []

    dev, ver = split_trades(
        trades,
        split_datetime
    )

    for name, subset in [
        ("DEVELOPMENT_BUY",
         dev[dev["direction"] == "BUY"]),
        ("DEVELOPMENT_SELL",
         dev[dev["direction"] == "SELL"]),
        ("VERIFICATION_BUY",
         ver[ver["direction"] == "BUY"]),
        ("VERIFICATION_SELL",
         ver[ver["direction"] == "SELL"])
    ]:

        row = summarize_trades(
            subset,
            dataset=name
        )

        same_candle = int(
            subset.get(
                "same_candle",
                pd.Series(
                    dtype=bool
                )
            ).sum()
        )

        row[
            "SL_AND_TP_SAME_CANDLE"
        ] = same_candle

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "by_direction_v3.csv"
        ),
        index=False
    )

    return result


# ============================================================
# CONDITION CLASSIFICATION
# ============================================================

def add_condition_columns(
    trades,
    df
):

    trades = ensure_dataframe(
        trades
    ).copy()

    if trades.empty:
        return trades

    # EMA strength

    trades["ema_strength"] = np.select(
        [
            trades["ema_distance"].abs() < 0.25,
            trades["ema_distance"].abs() < 0.50,
            trades["ema_distance"].abs() < 1.00
        ],
        [
            "VERY_WEAK",
            "WEAK",
            "MEDIUM"
        ],
        default="STRONG"
    )

    # Body

    trades["body_bucket"] = np.select(
        [
            trades["body_atr"] < 0.25,
            trades["body_atr"] < 0.50,
            trades["body_atr"] < 1.00
        ],
        [
            "VERY_SMALL",
            "SMALL",
            "MEDIUM"
        ],
        default="LARGE"
    )

    # MACD

    trades["macd_strength"] = (
        trades["macd_hist"]
        .abs()
        /
        trades["atr"]
        if "atr" in trades.columns
        else np.nan
    )

    trades["macd_strength"] = pd.to_numeric(
        trades["macd_strength"],
        errors="coerce"
    )

    trades["macd_strength_bucket"] = np.select(
        [
            trades["macd_strength"] < 0.05,
            trades["macd_strength"] < 0.10,
            trades["macd_strength"] < 0.20
        ],
        [
            "WEAK",
            "MEDIUM",
            "STRONG"
        ],
        default="VERY_STRONG"
    )

    # RSI

    trades["rsi_bucket"] = pd.cut(
        trades["rsi"],
        bins=[
            -np.inf,
            30,
            40,
            50,
            60,
            65,
            np.inf
        ],
        labels=[
            "<30",
            "30-40",
            "40-50",
            "50-60",
            "60-65",
            ">65"
        ]
    )

    # ATR regime

    trades["atr_regime"] = np.select(
        [
            trades["atr_ratio"] < 0.75,
            trades["atr_ratio"] > 1.25
        ],
        [
            "LOW",
            "HIGH"
        ],
        default="NORMAL"
    )

    # Momentum

    trades["momentum_3_sign"] = np.where(
        trades["momentum_3"] >= 0,
        "POSITIVE",
        "NEGATIVE"
    )

    trades["momentum_6_sign"] = np.where(
        trades["momentum_6"] >= 0,
        "POSITIVE",
        "NEGATIVE"
    )

    return trades


def condition_analysis(
    trades,
    development,
    verification
):

    trades = add_condition_columns(
        trades,
        None
    )

    rows = []

    conditions = {
        "15_ONLY":
            np.ones(len(trades), dtype=bool),

        "1H_ONLY":
            np.ones(len(trades), dtype=bool),

        "direction_15_aligned":
            np.ones(len(trades), dtype=bool),

        "15_AND_1H":
            (
                trades["ema_distance"].abs()
                > 0
            ),

        "direction_1h_aligned":
            (
                trades["ema_distance"].abs()
                > 0
            ),

        "momentum_6_negative":
            trades["momentum_6"] < 0,

        "momentum_3_negative":
            trades["momentum_3"] < 0,

        "atr_regime=NORMAL":
            trades["atr_regime"] == "NORMAL",

        "ema_weak":
            trades["ema_strength"].isin(
                ["VERY_WEAK", "WEAK"]
            ),

        "ema_strength=VERY_WEAK":
            trades["ema_strength"] ==
            "VERY_WEAK",

        "body_bucket=VERY_SMALL":
            trades["body_bucket"] ==
            "VERY_SMALL",

        "macd_strength=WEAK":
            trades["macd_strength_bucket"] ==
            "WEAK",

        "macd_weak":
            trades["macd_strength_bucket"] ==
            "WEAK",

        "rsi_bucket=40-50":
            trades["rsi_bucket"].astype(str) ==
            "40-50",

        "ema_strong":
            trades["ema_strength"] ==
            "STRONG"
    }

    for name, mask in conditions.items():

        mask = np.asarray(
            mask,
            dtype=bool
        )

        subset = trades.loc[
            mask
        ].copy()

        row = summarize_trades(
            subset,
            dataset=name
        )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "condition_combinations_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # DEVELOPMENT / VERIFICATION
    # --------------------------------------------------------

    dev_rows = []
    ver_rows = []

    for name, mask in conditions.items():

        mask = np.asarray(
            mask,
            dtype=bool
        )

        subset = trades.loc[
            mask
        ].copy()

        dev_subset = subset[
            subset["datetime"] <=
            development["datetime"].max()
            if not development.empty
            else False
        ]

        ver_subset = subset[
            subset["datetime"] >
            development["datetime"].max()
            if not development.empty
            else False
        ]

        dev_rows.append(
            {
                "condition": name,
                **summarize_trades(
                    dev_subset,
                    dataset="DEVELOPMENT"
                )
            }
        )

        ver_rows.append(
            {
                "condition": name,
                **summarize_trades(
                    ver_subset,
                    dataset="VERIFICATION"
                )
            }
        )

    pd.DataFrame(
        dev_rows
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "conditions_development_v3.csv"
        ),
        index=False
    )

    pd.DataFrame(
        ver_rows
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "conditions_verification_v3.csv"
        ),
        index=False
    )

    return result


# ============================================================
# CORRELATIONS
# ============================================================

def correlation_analysis(
    trades
):

    trades = ensure_dataframe(
        trades
    )

    if trades.empty:
        return pd.DataFrame()

    numeric = [
        "atr_ratio",
        "body_atr",
        "ema_distance",
        "ema50_slope",
        "momentum_6",
        "momentum_24",
        "hour_utc",
        "ema20_slope",
        "momentum_12",
        "macd_hist",
        "momentum_3",
        "rsi"
    ]

    rows = []

    for feature in numeric:

        if feature not in trades.columns:
            continue

        temp = trades[
            [feature, "MFE_ATR",
             "MAE_ATR", "result_R"]
        ].copy()

        temp = temp.apply(
            pd.to_numeric,
            errors="coerce"
        ).dropna()

        if len(temp) < 10:
            continue

        corr_mfe = temp[
            feature
        ].corr(
            temp["MFE_ATR"]
        )

        corr_mae = temp[
            feature
        ].corr(
            temp["MAE_ATR"]
        )

        corr_result = temp[
            feature
        ].corr(
            temp["result_R"]
        )

        rows.append({
            "feature": feature,
            "corr_MFE_ATR": corr_mfe,
            "corr_MAE_ATR": corr_mae,
            "corr_result_R": corr_result,
            "abs_corr_result_R":
                abs(corr_result)
        })

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        result = result.sort_values(
            "abs_corr_result_R",
            ascending=False
        )

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_correlations_v3.csv"
        ),
        index=False
    )

    return result


# ============================================================
# TIME HORIZONS
# ============================================================

def horizon_analysis(
    df,
    signals
):

    rows = []

    horizons = [
        15,
        30,
        60,
        120,
        240,
        480,
        720,
        1440
    ]

    for direction in [
        "BUY",
        "SELL"
    ]:

        dir_signals = signals[
            signals["direction"] ==
            direction
        ].copy()

        for minutes in horizons:

            bars = max(
                1,
                int(minutes / 5)
            )

            trades = []

            for _, row in dir_signals.iterrows():

                entry_index = int(
                    row["_entry_index"]
                )

                trade = simulate_trade(
                    df=df,
                    entry_index=entry_index,
                    direction=direction,
                    tp_atr=BASE_TP_ATR,
                    sl_atr=BASE_SL_ATR,
                    horizon=bars,
                    same_candle_mode="CONSERVATIVE"
                )

                if trade:
                    trades.append(
                        trade
                    )

            trades = ensure_dataframe(
                trades
            )

            summary = summarize_trades(
                trades,
                dataset=f"{direction}_{minutes}m"
            )

            summary["direction"] = direction
            summary["max_minutes"] = minutes

            rows.append(
                summary
            )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "time_horizons_v3.csv"
        ),
        index=False
    )

    return result


# ============================================================
# TARGET REACH
# ============================================================

def target_analysis(
    df,
    signals,
    split_datetime
):

    rows = []

    for dataset_name, subset in [
        (
            "DEVELOPMENT",
            signals[
                signals["datetime"] <=
                split_datetime
            ]
        ),
        (
            "VERIFICATION",
            signals[
                signals["datetime"] >
                split_datetime
            ]
        )
    ]:

        for direction in [
            "BUY",
            "SELL"
        ]:

            dir_signals = subset[
                subset["direction"] ==
                direction
            ]

            for target in TP_VALUES:

                reached = 0
                times = []

                for _, row in dir_signals.iterrows():

                    entry_index = int(
                        row["_entry_index"]
                    )

                    entry_price = float(
                        row["close"]
                    )

                    atr = float(
                        row["atr"]
                    )

                    if (
                        not np.isfinite(atr)
                        or atr <= 0
                    ):
                        continue

                    found = False

                    end = min(
                        len(df) - 1,
                        entry_index +
                        MFE_MAE_HORIZON
                    )

                    for j in range(
                        entry_index + 1,
                        end + 1
                    ):

                        high = float(
                            df.iloc[j]["high"]
                        )

                        low = float(
                            df.iloc[j]["low"]
                        )

                        if direction == "BUY":

                            favorable = (
                                high -
                                entry_price
                            ) / atr

                        else:

                            favorable = (
                                entry_price -
                                low
                            ) / atr

                        if favorable >= target:

                            reached += 1

                            times.append(
                                (j - entry_index) * 5
                            )

                            found = True
                            break

                count = len(
                    dir_signals
                )

                rows.append({
                    "dataset": dataset_name,
                    "direction": direction,
                    "target_atr": target,
                    "trades": count,
                    "reached_count": reached,
                    "reach_pct":
                        (
                            reached / count * 100
                            if count
                            else np.nan
                        ),
                    "avg_minutes_to_target":
                        (
                            np.mean(times)
                            if times
                            else np.nan
                        ),
                    "median_minutes_to_target":
                        (
                            np.median(times)
                            if times
                            else np.nan
                        )
                })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "target_analysis_v3.csv"
        ),
        index=False
    )

    return result


# ============================================================
# TP / SL MATRIX
# ============================================================

def evaluate_matrix(
    df,
    signals,
    split_datetime
):

    rows = []

    datasets = [
        (
            "DEVELOPMENT",
            signals[
                signals["datetime"] <=
                split_datetime
            ]
        ),
        (
            "VERIFICATION",
            signals[
                signals["datetime"] >
                split_datetime
            ]
        )
    ]

    for dataset_name, subset in datasets:

        for direction in [
            "BUY",
            "SELL"
        ]:

            dir_signals = subset[
                subset["direction"] ==
                direction
            ].copy()

            for tp in TP_VALUES:

                for sl in SL_VALUES:

                    trades = []

                    for _, row in dir_signals.iterrows():

                        entry_index = int(
                            row["_entry_index"]
                        )

                        trade = simulate_trade(
                            df=df,
                            entry_index=entry_index,
                            direction=direction,
                            tp_atr=tp,
                            sl_atr=sl,
                            horizon=MFE_MAE_HORIZON,
                            same_candle_mode="CONSERVATIVE"
                        )

                        if trade:
                            trades.append(
                                trade
                            )

                    trades = ensure_dataframe(
                        trades
                    )

                    wins = int(
                        (
                            trades["outcome"]
                            == "TP"
                        ).sum()
                    ) if not trades.empty else 0

                    losses = int(
                        (
                            trades["outcome"]
                            == "SL"
                        ).sum()
                    ) if not trades.empty else 0

                    total = (
                        wins + losses
                    )

                    results = (
                        pd.to_numeric(
                            trades["result_R"],
                            errors="coerce"
                        )
                        .dropna()
                        .values
                        if not trades.empty
                        else np.array([])
                    )

                    rows.append({
                        "dataset": dataset_name,
                        "direction": direction,
                        "TP_ATR": tp,
                        "SL_ATR": sl,
                        "trades": len(trades),
                        "wins": wins,
                        "losses": losses,
                        "win_rate_pct":
                            (
                                wins / total * 100
                                if total
                                else np.nan
                            ),
                        "total_R":
                            (
                                results.sum()
                                if len(results)
                                else 0
                            ),
                        "avg_R":
                            (
                                results.mean()
                                if len(results)
                                else np.nan
                            ),
                        "profit_factor":
                            profit_factor_from_results(
                                results
                            ),
                        "max_drawdown_R":
                            max_drawdown(
                                results
                            )
                    })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "tp_sl_matrix_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # INDEPENDENT MATRIX
    #
    # Non-overlapping signals only.
    # --------------------------------------------------------

    non_overlap = get_non_overlapping_signals(
        df
    )

    independent_rows = []

    for direction in [
        "BUY",
        "SELL"
    ]:

        dir_signals = non_overlap[
            non_overlap["direction"] ==
            direction
        ]

        for tp in TP_VALUES:

            for sl in SL_VALUES:

                trades = []

                for _, row in dir_signals.iterrows():

                    trade = simulate_trade(
                        df=df,
                        entry_index=int(
                            row["_entry_index"]
                        ),
                        direction=direction,
                        tp_atr=tp,
                        sl_atr=sl,
                        horizon=MFE_MAE_HORIZON,
                        same_candle_mode="CONSERVATIVE"
                    )

                    if trade:
                        trades.append(
                            trade
                        )

                trades = ensure_dataframe(
                    trades
                )

                results = (
                    pd.to_numeric(
                        trades["result_R"],
                        errors="coerce"
                    )
                    .dropna()
                    .values
                    if not trades.empty
                    else np.array([])
                )

                wins = (
                    int(
                        (
                            trades["outcome"]
                            == "TP"
                        ).sum()
                    )
                    if not trades.empty
                    else 0
                )

                losses = (
                    int(
                        (
                            trades["outcome"]
                            == "SL"
                        ).sum()
                    )
                    if not trades.empty
                    else 0
                )

                total = wins + losses

                independent_rows.append({
                    "dataset":
                        "NON_OVERLAPPING",
                    "direction": direction,
                    "TP_ATR": tp,
                    "SL_ATR": sl,
                    "trades": len(trades),
                    "wins": wins,
                    "losses": losses,
                    "win_rate_pct":
                        (
                            wins / total * 100
                            if total
                            else np.nan
                        ),
                    "total_R":
                        (
                            results.sum()
                            if len(results)
                            else 0
                        ),
                    "avg_R":
                        (
                            results.mean()
                            if len(results)
                            else np.nan
                        ),
                    "profit_factor":
                        profit_factor_from_results(
                            results
                        ),
                    "max_drawdown_R":
                        max_drawdown(
                            results
                        )
                })

    independent = pd.DataFrame(
        independent_rows
    )

    independent.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "tp_sl_matrix_independent_v3.csv"
        ),
        index=False
    )

    return (
        result,
        independent
    )


# ============================================================
# SAME CANDLE ANALYSIS
# ============================================================

def same_candle_analysis(
    df,
    signals
):

    rows = []

    modes = [
        "CONSERVATIVE",
        "OPTIMISTIC",
        "EXCLUDE"
    ]

    for mode in modes:

        trades = build_trades(
            df,
            signals,
            same_candle_mode=mode
        )

        summary = summarize_trades(
            trades,
            dataset=mode
        )

        rows.append(
            summary
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "same_candle_analysis_v3.csv"
        ),
        index=False
    )

    return result


# ============================================================
# TEMPORAL BLOCKS
# ============================================================

def temporal_blocks(
    trades,
    n_blocks=6,
    label_prefix="BLOCK"
):

    # --------------------------------------------------------
    # FIX BUG:
    # NON-OVERLAPPING / arrays / slices
    # vengono sempre convertiti.
    # --------------------------------------------------------

    trades = ensure_dataframe(
        trades
    )

    if trades.empty:
        return pd.DataFrame()

    if "datetime" not in trades.columns:
        return pd.DataFrame()

    trades = trades.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    # qcut può fallire se ci sono duplicati.
    # Usiamo np.array_split, molto più robusto.

    chunks = np.array_split(
        trades,
        n_blocks
    )

    rows = []

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        chunk = ensure_dataframe(
            chunk
        )

        row = summarize_trades(
            chunk,
            dataset=f"{label_prefix}_{i}"
        )

        if not chunk.empty:

            row["start"] = (
                chunk["datetime"].min()
            )

            row["end"] = (
                chunk["datetime"].max()
            )

        else:

            row["start"] = pd.NaT
            row["end"] = pd.NaT

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    return result


def temporal_stability_analysis(
    df,
    signal_modes
):

    rows = []

    for mode, signals in signal_modes.items():

        trades = build_trades(
            df,
            signals
        )

        blocks = temporal_blocks(
            trades,
            n_blocks=6,
            label_prefix=mode
        )

        if not blocks.empty:

            blocks.insert(
                0,
                "signal_mode",
                mode
            )

            rows.append(
                blocks
            )

    if rows:

        result = pd.concat(
            rows,
            ignore_index=True
        )

    else:

        result = pd.DataFrame()

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "signal_mode_temporal_blocks_v3.csv"
        ),
        index=False
    )

    # ALL signal stability
    all_trades = build_trades(
        df,
        signal_modes["ALL"]
    )

    all_blocks = temporal_blocks(
        all_trades,
        n_blocks=6,
        label_prefix="BLOCK"
    )

    all_blocks.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "temporal_blocks_v3.csv"
        ),
        index=False
    )

    return (
        all_blocks,
        result
    )


# ============================================================
# DIRECTION × REGIME
# ============================================================

def direction_regime_analysis(
    trades
):

    trades = add_condition_columns(
        ensure_dataframe(trades),
        None
    )

    if trades.empty:
        return pd.DataFrame()

    rows = []

    regimes = [
        "atr_regime",
        "ema_strength",
        "body_bucket",
        "rsi_bucket"
    ]

    for direction in [
        "BUY",
        "SELL"
    ]:

        dir_trades = trades[
            trades["direction"] ==
            direction
        ]

        for regime in regimes:

            if regime not in dir_trades.columns:
                continue

            for value in (
                dir_trades[regime]
                .dropna()
                .astype(str)
                .unique()
            ):

                subset = dir_trades[
                    dir_trades[regime]
                    .astype(str)
                    == value
                ]

                summary = summarize_trades(
                    subset,
                    dataset=(
                        f"{direction}_"
                        f"{regime}_"
                        f"{value}"
                    )
                )

                summary[
                    "direction"
                ] = direction

                summary[
                    "regime"
                ] = regime

                summary[
                    "regime_value"
                ] = value

                rows.append(
                    summary
                )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "direction_regimes_v3.csv"
        ),
        index=False
    )

    return result


# ============================================================
# FINAL DIAGNOSTICS
# ============================================================

def diagnostic_final(
    trades,
    signal_modes
):

    rows = []

    for mode, signals in signal_modes.items():

        mode_trades = build_trades(
            df_global,
            signals
        )

        mode_trades = ensure_dataframe(
            mode_trades
        )

        if mode_trades.empty:
            continue

        rows.append({
            "signal_mode": mode,
            "signals": len(signals),
            "trades": len(mode_trades),
            "buy_trades":
                int(
                    (
                        mode_trades["direction"]
                        == "BUY"
                    ).sum()
                ),
            "sell_trades":
                int(
                    (
                        mode_trades["direction"]
                        == "SELL"
                    ).sum()
                ),
            "same_candle":
                int(
                    mode_trades[
                        "same_candle"
                    ].sum()
                ),
            "timeouts":
                int(
                    (
                        mode_trades["outcome"]
                        == "TIMEOUT"
                    ).sum()
                )
        })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "diagnostic_final_v3.csv"
        ),
        index=False
    )

    return result


# ============================================================
# PRINT TABLE
# ============================================================

def print_table(
    title,
    df
):

    print_separator(title)

    df = ensure_dataframe(
        df
    )

    if df.empty:

        print("Nessun dato.")

        return

    with pd.option_context(
        "display.max_rows",
        200,
        "display.max_columns",
        50,
        "display.width",
        220,
        "display.float_format",
        lambda x: (
            f"{x:.6f}"
            if isinstance(x, float)
            else str(x)
        )
    ):

        print(
            df.to_string(
                index=False
            )
        )


# ============================================================
# MAIN
# ============================================================

df_global = None


def main():

    global df_global

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    df = download_history()

    df_global = df.copy()

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    df = calculate_indicators(
        df
    )

    # --------------------------------------------------------
    # MULTI TIMEFRAME
    # --------------------------------------------------------

    df = add_higher_timeframes(
        df
    )

    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    df = generate_signals(
        df
    )

    # --------------------------------------------------------
    # DROP INVALID WARMUP
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "ema20",
            "ema50",
            "macd",
            "macd_signal",
            "rsi",
            "atr"
        ]
    ).reset_index(
        drop=True
    )

    df_global = df.copy()

    # --------------------------------------------------------
    # SAVE RAW DATA
    # --------------------------------------------------------

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "raw_data_v3.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # SIGNAL MODES
    # --------------------------------------------------------

    signal_modes = build_signal_modes(
        df
    )

    # --------------------------------------------------------
    # MODE COMPARISON
    # --------------------------------------------------------

    mode_summary = analyze_signal_modes(
        df,
        signal_modes
    )

    print_table(
        "CONFRONTO MODALITÀ SEGNALI",
        mode_summary
    )

    # --------------------------------------------------------
    # ALL SIGNALS BASE TRADES
    # --------------------------------------------------------

    all_signals = signal_modes[
        "ALL"
    ]

    (
        all_trades,
        development,
        verification,
        split_datetime,
        general
    ) = general_summary(
        df,
        all_signals
    )

    print_table(
        "SUMMARY GENERALE - ALL SIGNALS",
        general
    )

    # --------------------------------------------------------
    # BUY / SELL
    # --------------------------------------------------------

    direction = direction_analysis(
        all_trades,
        split_datetime
    )

    print_table(
        "BUY / SELL",
        direction
    )

    # --------------------------------------------------------
    # CONDITIONS
    # --------------------------------------------------------

    conditions = condition_analysis(
        all_trades,
        development,
        verification
    )

    print_table(
        "CONDIZIONI / COMBINAZIONI",
        conditions
    )

    # --------------------------------------------------------
    # CORRELATIONS
    # --------------------------------------------------------

    correlations = correlation_analysis(
        all_trades
    )

    print_table(
        "FEATURE CON MAGGIORE CORRELAZIONE",
        correlations
    )

    # --------------------------------------------------------
    # TIME HORIZONS
    # --------------------------------------------------------

    horizons = horizon_analysis(
        df,
        all_signals
    )

    print_table(
        "TIME HORIZONS",
        horizons
    )

    # --------------------------------------------------------
    # TARGET REACHED
    # --------------------------------------------------------

    targets = target_analysis(
        df,
        all_signals,
        split_datetime
    )

    print_table(
        "TARGET RAGGIUNTI",
        targets
    )

    # --------------------------------------------------------
    # TP / SL MATRIX
    # --------------------------------------------------------

    matrix, independent_matrix = (
        evaluate_matrix(
            df,
            all_signals,
            split_datetime
        )
    )

    print_table(
        "MATRICE TP / SL",
        matrix
    )

    print_table(
        "MATRICE TP / SL - NON OVERLAPPING",
        independent_matrix
    )

    # --------------------------------------------------------
    # SAME CANDLE
    # --------------------------------------------------------

    same_candle = same_candle_analysis(
        df,
        all_signals
    )

    print_table(
        "SAME CANDLE TP + SL",
        same_candle
    )

    # --------------------------------------------------------
    # TEMPORAL STABILITY
    # --------------------------------------------------------

    (
        temporal,
        temporal_modes
    ) = temporal_stability_analysis(
        df,
        signal_modes
    )

    print_table(
        "STABILITÀ TEMPORALE - 6 BLOCCHI",
        temporal
    )

    print_table(
        "STABILITÀ TEMPORALE - MODALITÀ",
        temporal_modes
    )

    # --------------------------------------------------------
    # DIRECTION × REGIME
    # --------------------------------------------------------

    regimes = direction_regime_analysis(
        all_trades
    )

    print_table(
        "DIRECTION × REGIME",
        regimes
    )

    # --------------------------------------------------------
    # FINAL DIAGNOSTICS
    # --------------------------------------------------------

    diagnostics = diagnostic_final(
        all_trades,
        signal_modes
    )

    print_table(
        "DIAGNOSTICA FINALE",
        diagnostics
    )

    # --------------------------------------------------------
    # SAVE ALL TRADES
    # --------------------------------------------------------

    all_trades.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "diagnostic_v3_all_trades.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_separator(
        "BACKTEST V3 COMPLETATO"
    )

    print(
        f"Risultati salvati in: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Candele: {len(df)}"
    )

    print(
        f"Segnali ALL: "
        f"{len(all_signals)}"
    )

    print(
        f"Trade ALL: "
        f"{len(all_trades)}"
    )

    print(
        f"Split Development/Verification: "
        f"{split_datetime}"
    )

    print()
    print(
        "IMPORTANTE: questo è un "
        "backtest storico/demo e non "
        "garantisce risultati futuri."
    )


# ============================================================
# AVVIO
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Backtest interrotto manualmente."
        )

    except Exception as exc:

        print()
        print("=" * 80)
        print("ERRORE FATALE")
        print("=" * 80)

        print(
            type(exc).__name__,
            str(exc)
        )

        print()
        traceback.print_exc()

        raise
