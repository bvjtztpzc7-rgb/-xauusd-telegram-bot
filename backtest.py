import os
import time
import math
import requests
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAZIONE
# ============================================================

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

SYMBOL = "XAU/USD"
INTERVAL = "5min"

# Due blocchi da 5000 = fino a ~10.000 candele
# circa 34 giorni di dati 5m complessivi.
OUTPUTSIZE = 5000
BLOCKS = 2

# Strategia BASE attuale
EMA_FAST = 20
EMA_SLOW = 50

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

RSI_PERIOD = 14
ATR_PERIOD = 14

RSI_MIN = 30
RSI_MAX = 65

# Parametri standard del backtest
BASE_TP_ATR = 2.5
BASE_SL_ATR = 1.5

# Orizzonte massimo per analizzare il comportamento successivo
MAX_FORWARD_BARS = 288       # 24 ore su candele 5m

# Orizzonti per MFE/MAE
HORIZONS = {
    "5m": 1,
    "15m": 3,
    "30m": 6,
    "60m": 12,
    "120m": 24,
    "240m": 48,
    "480m": 96,
}

# Target da studiare
TP_ATR_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Stop da studiare
SL_ATR_LEVELS = [1.0, 1.5, 2.0]

# Percentuale development / verification
DEVELOPMENT_RATIO = 0.70

# Cartella risultati
OUTPUT_DIR = Path("backtest_results")


# ============================================================
# CONTROLLO API
# ============================================================

if not API_KEY:
    raise RuntimeError(
        "TWELVE_DATA_API_KEY non configurata nelle GitHub Secrets."
    )


# ============================================================
# UTILITY
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def print_separator(title=None):
    print("\n" + "=" * 80)
    if title:
        print(title)
        print("=" * 80)


# ============================================================
# DOWNLOAD DATI
# ============================================================

def download_block(start_date=None, end_date=None, outputsize=5000):
    """
    Scarica un blocco di dati da Twelve Data.
    """

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "format": "JSON",
        "timezone": "UTC",
    }

    if start_date:
        params["start_date"] = start_date

    if end_date:
        params["end_date"] = end_date

    print(f"Download dati: {params}")

    response = requests.get(url, params=params, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"Twelve Data HTTP {response.status_code}: {response.text[:1000]}"
        )

    data = response.json()

    if "status" in data and data["status"] == "error":
        raise RuntimeError(
            f"Twelve Data error: {data.get('message', data)}"
        )

    values = data.get("values")

    if not values:
        raise RuntimeError(
            f"Nessun dato ricevuto da Twelve Data: {data}"
        )

    df = pd.DataFrame(values)

    if "datetime" not in df.columns:
        raise RuntimeError("Colonna datetime mancante.")

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce"
    )

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.dropna(
        subset=["datetime", "open", "high", "low", "close"]
    )

    df = df.sort_values("datetime")
    df = df.drop_duplicates("datetime")

    return df


def download_history():
    """
    Scarica più blocchi.

    Twelve Data restituisce normalmente le candele più recenti.
    Per ottenere uno storico più ampio utilizziamo end_date
    progressivamente più vecchi.
    """

    all_blocks = []

    end_date = None

    for block_number in range(BLOCKS):

        print_separator(
            f"DOWNLOAD BLOCCO {block_number + 1}/{BLOCKS}"
        )

        try:
            df = download_block(
                end_date=end_date,
                outputsize=OUTPUTSIZE
            )

            if df.empty:
                break

            all_blocks.append(df)

            oldest = df["datetime"].min()

            print(
                f"Ricevute {len(df)} candele | "
                f"{df['datetime'].min()} -> {df['datetime'].max()}"
            )

            # Spostiamo la finestra indietro.
            end_date = (
                oldest - pd.Timedelta(minutes=5)
            ).strftime("%Y-%m-%d %H:%M:%S")

            time.sleep(1)

        except Exception as e:
            print(f"Errore nel blocco {block_number + 1}: {e}")
            break

    if not all_blocks:
        raise RuntimeError("Impossibile scaricare dati storici.")

    df = pd.concat(all_blocks, ignore_index=True)

    df = df.sort_values("datetime")
    df = df.drop_duplicates("datetime")
    df = df.reset_index(drop=True)

    return df


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

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (df["high"] - previous_close).abs()

    tr3 = (df["low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    return atr


def add_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema20"] = df["close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = df["close"].ewm(
        span=MACD_FAST,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=MACD_SLOW,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=MACD_SIGNAL,
        adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi"] = calculate_rsi(
        df["close"],
        RSI_PERIOD
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["atr"] = calculate_atr(
        df,
        ATR_PERIOD
    )

    # --------------------------------------------------------
    # EMA FEATURES
    # --------------------------------------------------------

    df["ema_distance"] = (
        df["ema20"] -
        df["ema50"]
    )

    df["ema_distance_abs"] = (
        df["ema_distance"].abs()
    )

    df["ema_distance_atr"] = (
        df["ema_distance_abs"] /
        df["atr"]
    )

    df["ema20_slope_1"] = (
        df["ema20"] -
        df["ema20"].shift(1)
    )

    df["ema20_slope_3"] = (
        df["ema20"] -
        df["ema20"].shift(3)
    )

    df["ema20_slope_6"] = (
        df["ema20"] -
        df["ema20"].shift(6)
    )

    df["ema50_slope_1"] = (
        df["ema50"] -
        df["ema50"].shift(1)
    )

    df["ema50_slope_3"] = (
        df["ema50"] -
        df["ema50"].shift(3)
    )

    df["ema50_slope_6"] = (
        df["ema50"] -
        df["ema50"].shift(6)
    )

    # --------------------------------------------------------
    # RSI MOMENTUM
    # --------------------------------------------------------

    df["rsi_change_1"] = (
        df["rsi"] -
        df["rsi"].shift(1)
    )

    df["rsi_change_3"] = (
        df["rsi"] -
        df["rsi"].shift(3)
    )

    df["rsi_change_6"] = (
        df["rsi"] -
        df["rsi"].shift(6)
    )

    # --------------------------------------------------------
    # MACD MOMENTUM
    # --------------------------------------------------------

    df["macd_change_1"] = (
        df["macd"] -
        df["macd"].shift(1)
    )

    df["macd_change_3"] = (
        df["macd"] -
        df["macd"].shift(3)
    )

    df["macd_change_6"] = (
        df["macd"] -
        df["macd"].shift(6)
    )

    df["macd_hist_change_1"] = (
        df["macd_hist"] -
        df["macd_hist"].shift(1)
    )

    df["macd_hist_change_3"] = (
        df["macd_hist"] -
        df["macd_hist"].shift(3)
    )

    # --------------------------------------------------------
    # ATR / VOLATILITY
    # --------------------------------------------------------

    df["atr_change_1"] = (
        df["atr"] -
        df["atr"].shift(1)
    )

    df["atr_change_3"] = (
        df["atr"] -
        df["atr"].shift(3)
    )

    df["atr_mean_24"] = (
        df["atr"]
        .rolling(24)
        .mean()
    )

    df["atr_relative_24"] = (
        df["atr"] /
        df["atr_mean_24"]
    )

    df["atr_mean_48"] = (
        df["atr"]
        .rolling(48)
        .mean()
    )

    df["atr_relative_48"] = (
        df["atr"] /
        df["atr_mean_48"]
    )

    # --------------------------------------------------------
    # CANDLE FEATURES
    # --------------------------------------------------------

    df["candle_range"] = (
        df["high"] -
        df["low"]
    )

    df["candle_body"] = (
        df["close"] -
        df["open"]
    )

    df["candle_body_abs"] = (
        df["candle_body"].abs()
    )

    df["body_ratio"] = (
        df["candle_body_abs"] /
        df["candle_range"].replace(0, np.nan)
    )

    df["upper_wick"] = (
        df["high"] -
        df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1) -
        df["low"]
    )

    df["upper_wick_ratio"] = (
        df["upper_wick"] /
        df["candle_range"].replace(0, np.nan)
    )

    df["lower_wick_ratio"] = (
        df["lower_wick"] /
        df["candle_range"].replace(0, np.nan)
    )

    df["close_position"] = (
        (df["close"] - df["low"]) /
        df["candle_range"].replace(0, np.nan)
    )

    df["candle_bullish"] = (
        df["close"] > df["open"]
    ).astype(int)

    df["candle_bearish"] = (
        df["close"] < df["open"]
    ).astype(int)

    # --------------------------------------------------------
    # MULTI-CANDLE MOMENTUM
    # --------------------------------------------------------

    for bars, name in [
        (3, "15m"),
        (6, "30m"),
        (12, "60m"),
        (24, "120m"),
    ]:
        df[f"momentum_{name}"] = (
            df["close"] /
            df["close"].shift(bars) -
            1
        ) * 100

        df[f"range_{name}"] = (
            df["high"]
            .rolling(bars)
            .max()
            -
            df["low"]
            .rolling(bars)
            .min()
        )

    # --------------------------------------------------------
    # CANDLE SEQUENCES
    # --------------------------------------------------------

    df["bullish_count_3"] = (
        df["candle_bullish"]
        .rolling(3)
        .sum()
    )

    df["bullish_count_6"] = (
        df["candle_bullish"]
        .rolling(6)
        .sum()
    )

    df["bullish_count_12"] = (
        df["candle_bullish"]
        .rolling(12)
        .sum()
    )

    # --------------------------------------------------------
    # RECENT HIGH / LOW
    # --------------------------------------------------------

    for bars in [12, 24, 48, 96]:

        recent_high = (
            df["high"]
            .rolling(bars)
            .max()
            .shift(1)
        )

        recent_low = (
            df["low"]
            .rolling(bars)
            .min()
            .shift(1)
        )

        df[f"recent_high_{bars}"] = recent_high

        df[f"recent_low_{bars}"] = recent_low

        df[f"distance_high_{bars}_atr"] = (
            (recent_high - df["close"]).abs() /
            df["atr"]
        )

        df[f"distance_low_{bars}_atr"] = (
            (df["close"] - recent_low).abs() /
            df["atr"]
        )

    return df


# ============================================================
# TIMEFRAME SUPERIORI
# ============================================================

def build_higher_timeframe_context(df):

    base = df.copy()

    temp = base.set_index("datetime")

    # --------------------------------------------------------
    # 15 MINUTI
    # --------------------------------------------------------

    tf15 = temp.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

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

    tf15["trend_15"] = np.where(
        tf15["ema20_15"] >
        tf15["ema50_15"],
        1,
        np.where(
            tf15["ema20_15"] <
            tf15["ema50_15"],
            -1,
            0
        )
    )

    tf15["ema_distance_15"] = (
        tf15["ema20_15"] -
        tf15["ema50_15"]
    )

    tf15["ema20_slope_15"] = (
        tf15["ema20_15"] -
        tf15["ema20_15"].shift(1)
    )

    tf15["ema50_slope_15"] = (
        tf15["ema50_15"] -
        tf15["ema50_15"].shift(1)
    )

    # IMPORTANT:
    # shift(1) = usiamo solamente la candela 15m
    # completamente chiusa prima del segnale.
    tf15_context = tf15[
        [
            "trend_15",
            "ema_distance_15",
            "ema20_slope_15",
            "ema50_slope_15"
        ]
    ].shift(1)

    tf15_context = tf15_context.rename(
        columns={
            "trend_15": "trend_15",
            "ema_distance_15": "ema_distance_15",
            "ema20_slope_15": "ema20_slope_15",
            "ema50_slope_15": "ema50_slope_15",
        }
    )

    # --------------------------------------------------------
    # 1 ORA
    # --------------------------------------------------------

    tf1h = temp.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

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

    tf1h["trend_1h"] = np.where(
        tf1h["ema20_1h"] >
        tf1h["ema50_1h"],
        1,
        np.where(
            tf1h["ema20_1h"] <
            tf1h["ema50_1h"],
            -1,
            0
        )
    )

    tf1h["ema_distance_1h"] = (
        tf1h["ema20_1h"] -
        tf1h["ema50_1h"]
    )

    tf1h["ema20_slope_1h"] = (
        tf1h["ema20_1h"] -
        tf1h["ema20_1h"].shift(1)
    )

    tf1h["ema50_slope_1h"] = (
        tf1h["ema50_1h"] -
        tf1h["ema50_1h"].shift(1)
    )

    tf1h_context = tf1h[
        [
            "trend_1h",
            "ema_distance_1h",
            "ema20_slope_1h",
            "ema50_slope_1h"
        ]
    ].shift(1)

    # --------------------------------------------------------
    # MERGE TEMPORALE
    # --------------------------------------------------------

    result = base.copy()

    result = pd.merge_asof(
        result.sort_values("datetime"),
        tf15_context.reset_index().sort_values("datetime"),
        on="datetime",
        direction="backward"
    )

    result = pd.merge_asof(
        result.sort_values("datetime"),
        tf1h_context.reset_index().sort_values("datetime"),
        on="datetime",
        direction="backward"
    )

    return result


# ============================================================
# SEGNALI STRATEGIA BASE
# ============================================================

def generate_base_signals(df):

    df = df.copy()

    bullish = (
        (df["ema20"] > df["ema50"]) &
        (df["macd"] > df["macd_signal"]) &
        (df["rsi"] > RSI_MIN) &
        (df["rsi"] < RSI_MAX) &
        (df["trend_15"] == 1)
    )

    bearish = (
        (df["ema20"] < df["ema50"]) &
        (df["macd"] < df["macd_signal"]) &
        (df["rsi"] > RSI_MIN) &
        (df["rsi"] < RSI_MAX) &
        (df["trend_15"] == -1)
    )

    df["signal"] = np.where(
        bullish,
        "BUY",
        np.where(
            bearish,
            "SELL",
            ""
        )
    )

    return df


# ============================================================
# CLASSIFICAZIONI
# ============================================================

def classify_rsi(value):

    if pd.isna(value):
        return "NA"

    if value < 40:
        return "30-40"

    if value < 50:
        return "40-50"

    if value < 60:
        return "50-60"

    return "60-65"


def classify_atr_relative(value):

    if pd.isna(value):
        return "NA"

    if value < 0.80:
        return "LOW"

    if value < 1.20:
        return "NORMAL"

    return "HIGH"


def classify_strength(value):

    if pd.isna(value):
        return "NA"

    if value < 0.25:
        return "WEAK"

    if value < 0.75:
        return "MEDIUM"

    return "STRONG"


def classify_body(value):

    if pd.isna(value):
        return "NA"

    if value < 0.30:
        return "SMALL"

    if value < 0.60:
        return "MEDIUM"

    return "LARGE"


# ============================================================
# MFE / MAE
# ============================================================

def calculate_forward_metrics(
    df,
    entry_index,
    direction,
    entry_price,
    atr
):

    result = {}

    if pd.isna(atr) or atr <= 0:
        return result

    end_index = min(
        len(df) - 1,
        entry_index + MAX_FORWARD_BARS
    )

    future = df.iloc[
        entry_index:end_index + 1
    ]

    if direction == "BUY":

        favorable = (
            future["high"] -
            entry_price
        )

        adverse = (
            entry_price -
            future["low"]
        )

    else:

        favorable = (
            entry_price -
            future["low"]
        )

        adverse = (
            future["high"] -
            entry_price
        )

    # --------------------------------------------------------
    # MFE / MAE assoluti
    # --------------------------------------------------------

    result["mfe_price"] = max(
        0.0,
        float(favorable.max())
    )

    result["mae_price"] = max(
        0.0,
        float(adverse.max())
    )

    result["mfe_atr"] = (
        result["mfe_price"] / atr
    )

    result["mae_atr"] = (
        result["mae_price"] / atr
    )

    # --------------------------------------------------------
    # MFE / MAE per orizzonte
    # --------------------------------------------------------

    for horizon_name, bars in HORIZONS.items():

        horizon_end = min(
            entry_index + bars,
            end_index
        )

        horizon = df.iloc[
            entry_index:horizon_end + 1
        ]

        if horizon.empty:
            continue

        if direction == "BUY":

            fav = (
                horizon["high"] -
                entry_price
            )

            adv = (
                entry_price -
                horizon["low"]
            )

        else:

            fav = (
                entry_price -
                horizon["low"]
            )

            adv = (
                horizon["high"] -
                entry_price
            )

        result[
            f"mfe_{horizon_name}_atr"
        ] = max(
            0.0,
            float(fav.max()) / atr
        )

        result[
            f"mae_{horizon_name}_atr"
        ] = max(
            0.0,
            float(adv.max()) / atr
        )

    # --------------------------------------------------------
    # TARGET RAGGIUNTI
    # --------------------------------------------------------

    for target in TP_ATR_LEVELS:

        target_price_distance = target * atr

        if direction == "BUY":

            reached = (
                future["high"] >=
                entry_price +
                target_price_distance
            )

        else:

            reached = (
                future["low"] <=
                entry_price -
                target_price_distance
            )

        indices = np.where(
            reached.values
        )[0]

        result[
            f"reach_{target:g}atr"
        ] = int(len(indices) > 0)

        if len(indices) > 0:

            bars_to_target = int(
                indices[0]
            )

            result[
                f"bars_to_{target:g}atr"
            ] = bars_to_target

            result[
                f"minutes_to_{target:g}atr"
            ] = bars_to_target * 5

        else:

            result[
                f"bars_to_{target:g}atr"
            ] = np.nan

            result[
                f"minutes_to_{target:g}atr"
            ] = np.nan

    return result


# ============================================================
# TP / SL REALISTICO
# ============================================================

def evaluate_trade(
    df,
    entry_index,
    direction,
    entry_price,
    atr
):

    if pd.isna(atr) or atr <= 0:
        return {}

    tp_distance = BASE_TP_ATR * atr
    sl_distance = BASE_SL_ATR * atr

    if direction == "BUY":

        tp_price = (
            entry_price +
            tp_distance
        )

        sl_price = (
            entry_price -
            sl_distance
        )

    else:

        tp_price = (
            entry_price -
            tp_distance
        )

        sl_price = (
            entry_price +
            sl_distance
        )

    end_index = min(
        len(df) - 1,
        entry_index + MAX_FORWARD_BARS
    )

    future = df.iloc[
        entry_index:end_index + 1
    ]

    outcome = "TIMEOUT"
    exit_price = np.nan
    exit_index = np.nan

    for offset, (_, candle) in enumerate(
        future.iterrows()
    ):

        high = candle["high"]
        low = candle["low"]

        if direction == "BUY":

            hit_tp = high >= tp_price
            hit_sl = low <= sl_price

        else:

            hit_tp = low <= tp_price
            hit_sl = high >= sl_price

        # ----------------------------------------------------
        # Se TP e SL vengono toccati nella stessa candela,
        # NON possiamo sapere dal solo OHLC quale sia avvenuto
        # prima.
        #
        # Per evitare un'ottimizzazione favorevole assumiamo
        # CONSERVATIVAMENTE che abbia colpito lo SL.
        # ----------------------------------------------------

        if hit_tp and hit_sl:

            outcome = "SL_AND_TP_SAME_CANDLE"

            exit_price = sl_price
            exit_index = (
                entry_index +
                offset
            )

            break

        elif hit_tp:

            outcome = "TP"

            exit_price = tp_price
            exit_index = (
                entry_index +
                offset
            )

            break

        elif hit_sl:

            outcome = "SL"

            exit_price = sl_price
            exit_index = (
                entry_index +
                offset
            )

            break

    if outcome == "TIMEOUT":

        last_close = future.iloc[-1]["close"]

        exit_price = last_close

        exit_index = end_index

    if direction == "BUY":

        result_r = (
            exit_price -
            entry_price
        ) / sl_distance

    else:

        result_r = (
            entry_price -
            exit_price
        ) / sl_distance

    bars_in_trade = (
        int(exit_index) -
        entry_index
    )

    return {
        "tp_price": tp_price,
        "sl_price": sl_price,
        "outcome": outcome,
        "exit_price": exit_price,
        "bars_in_trade": bars_in_trade,
        "minutes_in_trade": bars_in_trade * 5,
        "result_R": result_r,
    }


# ============================================================
# COSTRUZIONE DATASET TRADE
# ============================================================

def build_trade_dataset(df):

    trades = []

    print_separator("COSTRUZIONE DATASET DEI TRADE")

    signal_indices = np.where(
        df["signal"].isin(["BUY", "SELL"])
    )[0]

    print(
        f"Segnali trovati: {len(signal_indices)}"
    )

    for counter, signal_index in enumerate(
        signal_indices,
        start=1
    ):

        # Il segnale viene generato sulla candela chiusa.
        # Entriamo all'OPEN della candela successiva.
        entry_index = signal_index + 1

        if entry_index >= len(df):
            continue

        row = df.iloc[signal_index]

        direction = row["signal"]

        entry_row = df.iloc[entry_index]

        entry_price = float(
            entry_row["open"]
        )

        atr = float(row["atr"])

        if not np.isfinite(atr) or atr <= 0:
            continue

        trade = {
            "signal_index": signal_index,
            "entry_index": entry_index,
            "signal_datetime": row["datetime"],
            "entry_datetime": entry_row["datetime"],
            "direction": direction,
            "entry_price": entry_price,
            "atr": atr,
        }

        # ----------------------------------------------------
        # INDICATORI
        # ----------------------------------------------------

        feature_columns = [
            "rsi",
            "rsi_change_1",
            "rsi_change_3",
            "rsi_change_6",

            "ema20",
            "ema50",
            "ema_distance",
            "ema_distance_abs",
            "ema_distance_atr",

            "ema20_slope_1",
            "ema20_slope_3",
            "ema20_slope_6",

            "ema50_slope_1",
            "ema50_slope_3",
            "ema50_slope_6",

            "macd",
            "macd_signal",
            "macd_hist",

            "macd_change_1",
            "macd_change_3",
            "macd_change_6",

            "macd_hist_change_1",
            "macd_hist_change_3",

            "atr_change_1",
            "atr_change_3",
            "atr_relative_24",
            "atr_relative_48",

            "candle_range",
            "candle_body",
            "candle_body_abs",
            "body_ratio",

            "upper_wick",
            "lower_wick",
            "upper_wick_ratio",
            "lower_wick_ratio",
            "close_position",

            "momentum_15m",
            "momentum_30m",
            "momentum_60m",
            "momentum_120m",

            "range_15m",
            "range_30m",
            "range_60m",
            "range_120m",

            "bullish_count_3",
            "bullish_count_6",
            "bullish_count_12",

            "trend_15",
            "ema_distance_15",
            "ema20_slope_15",
            "ema50_slope_15",

            "trend_1h",
            "ema_distance_1h",
            "ema20_slope_1h",
            "ema50_slope_1h",

            "recent_high_12",
            "recent_low_12",
            "distance_high_12_atr",
            "distance_low_12_atr",

            "recent_high_24",
            "recent_low_24",
            "distance_high_24_atr",
            "distance_low_24_atr",

            "recent_high_48",
            "recent_low_48",
            "distance_high_48_atr",
            "distance_low_48_atr",

            "recent_high_96",
            "recent_low_96",
            "distance_high_96_atr",
            "distance_low_96_atr",
        ]

        for col in feature_columns:

            if col in row.index:
                trade[col] = row[col]

        # ----------------------------------------------------
        # CLASSIFICAZIONI
        # ----------------------------------------------------

        trade["rsi_bucket"] = classify_rsi(
            row["rsi"]
        )

        trade["ema_strength"] = classify_strength(
            row["ema_distance_atr"]
        )

        trade["macd_strength"] = classify_strength(
            abs(row["macd_hist"]) /
            atr
        )

        trade["atr_regime"] = classify_atr_relative(
            row["atr_relative_24"]
        )

        trade["body_bucket"] = classify_body(
            row["body_ratio"]
        )

        # ----------------------------------------------------
        # CONCORDANZA MULTI-TIMEFRAME
        # ----------------------------------------------------

        if direction == "BUY":

            trade["trend_15_aligned"] = int(
                row["trend_15"] == 1
            )

            trade["trend_1h_aligned"] = int(
                row["trend_1h"] == 1
            )

        else:

            trade["trend_15_aligned"] = int(
                row["trend_15"] == -1
            )

            trade["trend_1h_aligned"] = int(
                row["trend_1h"] == -1
            )

        trade["mtf_aligned"] = int(
            trade["trend_15_aligned"] == 1 and
            trade["trend_1h_aligned"] == 1
        )

        # ----------------------------------------------------
        # RISULTATO TP/SL
        # ----------------------------------------------------

        trade.update(
            evaluate_trade(
                df,
                entry_index,
                direction,
                entry_price,
                atr
            )
        )

        # ----------------------------------------------------
        # MFE / MAE / TARGET
        # ----------------------------------------------------

        trade.update(
            calculate_forward_metrics(
                df,
                entry_index,
                direction,
                entry_price,
                atr
            )
        )

        trades.append(trade)

        if counter % 50 == 0:
            print(
                f"Processati {counter}/"
                f"{len(signal_indices)} segnali..."
            )

    result = pd.DataFrame(trades)

    if result.empty:
        raise RuntimeError(
            "Nessun trade generato."
        )

    return result


# ============================================================
# SPLIT TEMPORALE
# ============================================================

def add_dataset_split(trades):

    trades = trades.sort_values(
        "signal_datetime"
    ).reset_index(drop=True)

    split_index = int(
        len(trades) *
        DEVELOPMENT_RATIO
    )

    trades["dataset"] = "VERIFICATION"

    trades.loc[
        :split_index - 1,
        "dataset"
    ] = "DEVELOPMENT"

    return trades


# ============================================================
# STATISTICHE
# ============================================================

def summarize(df, label):

    if df.empty:
        return {
            "dataset": label,
            "trades": 0
        }

    tp = (
        df["outcome"] == "TP"
    ).sum()

    sl = (
        df["outcome"] == "SL"
    ).sum()

    same = (
        df["outcome"] ==
        "SL_AND_TP_SAME_CANDLE"
    ).sum()

    timeout = (
        df["outcome"] == "TIMEOUT"
    ).sum()

    total = len(df)

    win_rate = (
        tp / total * 100
        if total
        else 0
    )

    avg_r = df["result_R"].mean()

    total_r = df["result_R"].sum()

    gross_profit = (
        df.loc[
            df["result_R"] > 0,
            "result_R"
        ].sum()
    )

    gross_loss = abs(
        df.loc[
            df["result_R"] < 0,
            "result_R"
        ].sum()
    )

    profit_factor = (
        gross_profit /
        gross_loss
        if gross_loss > 0
        else np.nan
    )

    # --------------------------------------------------------
    # MAX DRAWDOWN
    # --------------------------------------------------------

    equity = (
        df["result_R"]
        .fillna(0)
        .cumsum()
    )

    running_max = equity.cummax()

    drawdown = (
        running_max -
        equity
    )

    max_dd = drawdown.max()

    return {
        "dataset": label,
        "trades": total,
        "TP": int(tp),
        "SL": int(sl),
        "SL_AND_TP_SAME_CANDLE": int(same),
        "TIMEOUT": int(timeout),
        "win_rate_pct": win_rate,
        "avg_R": avg_r,
        "total_R": total_r,
        "profit_factor": profit_factor,
        "max_drawdown_R": max_dd,
        "avg_MFE_ATR": df["mfe_atr"].mean(),
        "avg_MAE_ATR": df["mae_atr"].mean(),
        "median_MFE_ATR": df["mfe_atr"].median(),
        "median_MAE_ATR": df["mae_atr"].median(),
    }


def grouped_analysis(
    trades,
    column,
    min_trades=10
):

    rows = []

    for value, group in trades.groupby(
        column,
        dropna=False
    ):

        if len(group) < min_trades:
            continue

        summary = summarize(
            group,
            str(value)
        )

        summary["group"] = value

        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# ANALISI TARGET
# ============================================================

def target_analysis(trades):

    rows = []

    for dataset_name, dataset in trades.groupby(
        "dataset"
    ):

        for direction, group in dataset.groupby(
            "direction"
        ):

            for target in TP_ATR_LEVELS:

                reach_col = (
                    f"reach_{target:g}atr"
                )

                if reach_col not in group.columns:
                    continue

                rows.append({
                    "dataset": dataset_name,
                    "direction": direction,
                    "target_atr": target,
                    "trades": len(group),
                    "reached_count": int(
                        group[reach_col].sum()
                    ),
                    "reach_pct": (
                        group[reach_col].mean() *
                        100
                    ),
                    "avg_minutes_to_target": (
                        group[
                            f"minutes_to_{target:g}atr"
                        ].mean()
                    ),
                    "median_minutes_to_target": (
                        group[
                            f"minutes_to_{target:g}atr"
                        ].median()
                    )
                })

    return pd.DataFrame(rows)


# ============================================================
# TP/SL COMBINATIONS
# ============================================================

def simulate_tp_sl_matrix(
    df,
    trades
):

    rows = []

    for dataset_name, dataset in trades.groupby(
        "dataset"
    ):

        for direction, group in dataset.groupby(
            "direction"
        ):

            for tp_atr in TP_ATR_LEVELS:

                for sl_atr in SL_ATR_LEVELS:

                    outcomes = []

                    for _, trade in group.iterrows():

                        entry_index = int(
                            trade["entry_index"]
                        )

                        entry = float(
                            trade["entry_price"]
                        )

                        atr = float(
                            trade["atr"]
                        )

                        if not np.isfinite(atr):
                            continue

                        tp_distance = (
                            tp_atr * atr
                        )

                        sl_distance = (
                            sl_atr * atr
                        )

                        if direction == "BUY":

                            tp_price = (
                                entry +
                                tp_distance
                            )

                            sl_price = (
                                entry -
                                sl_distance
                            )

                        else:

                            tp_price = (
                                entry -
                                tp_distance
                            )

                            sl_price = (
                                entry +
                                sl_distance
                            )

                        end_index = min(
                            len(df) - 1,
                            entry_index +
                            MAX_FORWARD_BARS
                        )

                        future = df.iloc[
                            entry_index:
                            end_index + 1
                        ]

                        outcome = "TIMEOUT"

                        for _, candle in future.iterrows():

                            high = candle["high"]
                            low = candle["low"]

                            if direction == "BUY":

                                hit_tp = (
                                    high >= tp_price
                                )

                                hit_sl = (
                                    low <= sl_price
                                )

                            else:

                                hit_tp = (
                                    low <= tp_price
                                )

                                hit_sl = (
                                    high >= sl_price
                                )

                            if hit_tp and hit_sl:

                                outcome = "SL"
                                break

                            if hit_tp:

                                outcome = "TP"
                                break

                            if hit_sl:

                                outcome = "SL"
                                break

                        if outcome == "TP":

                            result_r = (
                                tp_atr /
                                sl_atr
                            )

                        elif outcome == "SL":

                            result_r = -1

                        else:

                            result_r = 0

                        outcomes.append(
                            result_r
                        )

                    if not outcomes:
                        continue

                    arr = np.array(
                        outcomes,
                        dtype=float
                    )

                    wins = (
                        arr > 0
                    ).sum()

                    losses = (
                        arr < 0
                    ).sum()

                    gross_profit = arr[
                        arr > 0
                    ].sum()

                    gross_loss = abs(
                        arr[arr < 0].sum()
                    )

                    pf = (
                        gross_profit /
                        gross_loss
                        if gross_loss > 0
                        else np.nan
                    )

                    equity = np.cumsum(arr)

                    running_max = np.maximum.accumulate(
                        equity
                    )

                    dd = (
                        running_max -
                        equity
                    )

                    rows.append({
                        "dataset": dataset_name,
                        "direction": direction,
                        "TP_ATR": tp_atr,
                        "SL_ATR": sl_atr,
                        "trades": len(arr),
                        "wins": int(wins),
                        "losses": int(losses),
                        "win_rate_pct": (
                            wins /
                            len(arr) *
                            100
                        ),
                        "total_R": arr.sum(),
                        "avg_R": arr.mean(),
                        "profit_factor": pf,
                        "max_drawdown_R": dd.max()
                    })

    return pd.DataFrame(rows)


# ============================================================
# ANALISI COMBINAZIONI
# ============================================================

def combination_analysis(trades):

    rows = []

    # Queste sono combinazioni relativamente compatte.
    # Non facciamo un'esplosione combinatoria enorme.

    conditions = {
        "rsi_bucket": [
            "30-40",
            "40-50",
            "50-60",
            "60-65"
        ],

        "ema_strength": [
            "WEAK",
            "MEDIUM",
            "STRONG"
        ],

        "macd_strength": [
            "WEAK",
            "MEDIUM",
            "STRONG"
        ],

        "atr_regime": [
            "LOW",
            "NORMAL",
            "HIGH"
        ],

        "body_bucket": [
            "SMALL",
            "MEDIUM",
            "LARGE"
        ]
    }

    # --------------------------------------------------------
    # Singoli fattori
    # --------------------------------------------------------

    for col in conditions:

        for value in conditions[col]:

            subset = trades[
                trades[col] == value
            ]

            if len(subset) < 15:
                continue

            s = summarize(
                subset,
                f"{col}={value}"
            )

            rows.append({
                "type": "SINGLE",
                "condition": (
                    f"{col}={value}"
                ),
                **s
            })

    # --------------------------------------------------------
    # Coppie
    # --------------------------------------------------------

    pair_columns = [
        "rsi_bucket",
        "ema_strength",
        "macd_strength",
        "atr_regime",
        "body_bucket"
    ]

    for col_a, col_b in itertools.combinations(
        pair_columns,
        2
    ):

        for value_a in trades[
            col_a
        ].dropna().unique():

            for value_b in trades[
                col_b
            ].dropna().unique():

                subset = trades[
                    (trades[col_a] == value_a) &
                    (trades[col_b] == value_b)
                ]

                if len(subset) < 15:
                    continue

                s = summarize(
                    subset,
                    f"{col_a}={value_a} & "
                    f"{col_b}={value_b}"
                )

                rows.append({
                    "type": "PAIR",
                    "condition": (
                        f"{col_a}={value_a} & "
                        f"{col_b}={value_b}"
                    ),
                    **s
                })

    # --------------------------------------------------------
    # Multi-timeframe
    # --------------------------------------------------------

    mtf_cases = {
        "15_ONLY": trades[
            trades["trend_15_aligned"] == 1
        ],

        "1H_ONLY": trades[
            trades["trend_1h_aligned"] == 1
        ],

        "15_AND_1H": trades[
            (trades["trend_15_aligned"] == 1) &
            (trades["trend_1h_aligned"] == 1)
        ],

        "15_AND_1H_NOT_ALIGNED": trades[
            (trades["trend_15_aligned"] == 1) &
            (trades["trend_1h_aligned"] == 0)
        ]
    }

    for name, subset in mtf_cases.items():

        if len(subset) < 15:
            continue

        s = summarize(
            subset,
            name
        )

        rows.append({
            "type": "MTF",
            "condition": name,
            **s
        })

    return pd.DataFrame(rows)


# ============================================================
# CORRELAZIONI
# ============================================================

def correlation_analysis(trades):

    numeric_features = [
        "rsi",
        "rsi_change_1",
        "rsi_change_3",
        "rsi_change_6",

        "ema_distance_atr",
        "ema20_slope_1",
        "ema20_slope_3",
        "ema20_slope_6",

        "ema50_slope_1",
        "ema50_slope_3",
        "ema50_slope_6",

        "macd",
        "macd_hist",
        "macd_change_1",
        "macd_change_3",
        "macd_change_6",

        "atr_relative_24",
        "atr_relative_48",

        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "close_position",

        "momentum_15m",
        "momentum_30m",
        "momentum_60m",
        "momentum_120m",

        "bullish_count_3",
        "bullish_count_6",
        "bullish_count_12",

        "distance_high_12_atr",
        "distance_low_12_atr",
        "distance_high_24_atr",
        "distance_low_24_atr",
        "distance_high_48_atr",
        "distance_low_48_atr",
        "distance_high_96_atr",
        "distance_low_96_atr",

        "mfe_atr",
        "mae_atr"
    ]

    available = [
        c for c in numeric_features
        if c in trades.columns
    ]

    correlations = []

    for feature in available:

        try:

            corr_mfe = trades[
                [feature, "mfe_atr"]
            ].corr().iloc[0, 1]

            corr_mae = trades[
                [feature, "mae_atr"]
            ].corr().iloc[0, 1]

            corr_result = trades[
                [feature, "result_R"]
            ].corr().iloc[0, 1]

            correlations.append({
                "feature": feature,
                "corr_MFE_ATR": corr_mfe,
                "corr_MAE_ATR": corr_mae,
                "corr_result_R": corr_result
            })

        except Exception:
            pass

    result = pd.DataFrame(
        correlations
    )

    if not result.empty:

        result["abs_corr_result_R"] = (
            result["corr_result_R"].abs()
        )

        result = result.sort_values(
            "abs_corr_result_R",
            ascending=False
        )

    return result


# ============================================================
# STAMPA RISULTATI
# ============================================================

def print_summary_table(summary_df):

    print_separator(
        "SUMMARY GENERALE"
    )

    if summary_df.empty:
        print("Nessun risultato.")
        return

    cols = [
        "dataset",
        "trades",
        "TP",
        "SL",
        "TIMEOUT",
        "win_rate_pct",
        "avg_R",
        "total_R",
        "profit_factor",
        "max_drawdown_R",
        "avg_MFE_ATR",
        "avg_MAE_ATR"
    ]

    available = [
        c for c in cols
        if c in summary_df.columns
    ]

    print(
        summary_df[
            available
        ].to_string(
            index=False
        )
    )


def print_top_analysis(
    df,
    title,
    n=15
):

    print_separator(title)

    if df.empty:
        print("Nessun dato disponibile.")
        return

    # --------------------------------------------------------
    # DATAFRAME DELLE CORRELAZIONI
    # --------------------------------------------------------

    if "feature" in df.columns:

        output = df.copy()

        if "abs_corr_result_R" in output.columns:

            output = output.sort_values(
                "abs_corr_result_R",
                ascending=False
            ).head(n)

        else:

            output = output.head(n)

        print(
            output.to_string(
                index=False
            )
        )

        return

    # --------------------------------------------------------
    # DATAFRAME DELLE CONDIZIONI
    # --------------------------------------------------------

    if "trades" not in df.columns:

        print(
            "Formato dataframe non riconosciuto."
        )

        print(
            "Colonne disponibili:",
            list(df.columns)
        )

        return

    display_cols = [
        "condition",
        "trades",
        "win_rate_pct",
        "avg_R",
        "total_R",
        "profit_factor",
        "max_drawdown_R",
        "avg_MFE_ATR",
        "avg_MAE_ATR"
    ]

    display_cols = [
        c for c in display_cols
        if c in df.columns
    ]

    output = df.sort_values(
        ["trades", "avg_R"],
        ascending=[False, False]
    ).head(n)

    print(
        output[
            display_cols
        ].to_string(
            index=False
        )
    )

    print_separator(title)

    if df.empty:
        print("Nessun gruppo sufficientemente grande.")
        return

    display_cols = [
        "condition",
        "trades",
        "win_rate_pct",
        "avg_R",
        "total_R",
        "profit_factor",
        "max_drawdown_R",
        "avg_MFE_ATR",
        "avg_MAE_ATR"
    ]

    display_cols = [
        c for c in display_cols
        if c in df.columns
    ]

    # ATTENZIONE:
    # non stiamo dichiarando "migliore".
    # Stampiamo solamente i gruppi con maggiore
    # risultato statistico per ispezione.
    output = df.sort_values(
        ["trades", "avg_R"],
        ascending=[False, False]
    ).head(n)

    print(
        output[
            display_cols
        ].to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print_separator(
        "BACKTEST V2 - XAU/USD"
    )

    print(
        f"Symbol: {SYMBOL}"
    )

    print(
        f"Interval: {INTERVAL}"
    )

    print(
        f"TP base: {BASE_TP_ATR} ATR"
    )

    print(
        f"SL base: {BASE_SL_ATR} ATR"
    )

    print(
        f"Development: {DEVELOPMENT_RATIO:.0%}"
    )

    print(
        f"Verification: "
        f"{1 - DEVELOPMENT_RATIO:.0%}"
    )

    # --------------------------------------------------------
    # 1. DOWNLOAD
    # --------------------------------------------------------

    df = download_history()

    print_separator(
        "DATI STORICI"
    )

    print(
        f"Candele totali: {len(df)}"
    )

    print(
        f"Periodo: "
        f"{df['datetime'].min()} -> "
        f"{df['datetime'].max()}"
    )

    # Salva dati grezzi
    df.to_csv(
        OUTPUT_DIR /
        "raw_5m_data.csv",
        index=False
    )

    # --------------------------------------------------------
    # 2. INDICATORI
    # --------------------------------------------------------

    print_separator(
        "CALCOLO INDICATORI"
    )

    df = add_indicators(df)

    # --------------------------------------------------------
    # 3. TIMEFRAME SUPERIORI
    # --------------------------------------------------------

    print_separator(
        "COSTRUZIONE 15M + 1H"
    )

    df = build_higher_timeframe_context(
        df
    )

    # --------------------------------------------------------
    # 4. SEGNALI
    # --------------------------------------------------------

    print_separator(
        "GENERAZIONE SEGNALI BASE"
    )

    df = generate_base_signals(df)

    signal_count = (
        df["signal"]
        .isin(["BUY", "SELL"])
        .sum()
    )

    print(
        f"BUY/SELL totali: {signal_count}"
    )

    print(
        "BUY:",
        (df["signal"] == "BUY").sum()
    )

    print(
        "SELL:",
        (df["signal"] == "SELL").sum()
    )

    # --------------------------------------------------------
    # 5. TRADE DATASET
    # --------------------------------------------------------

    trades = build_trade_dataset(
        df
    )

    # --------------------------------------------------------
    # 6. SPLIT
    # --------------------------------------------------------

    trades = add_dataset_split(
        trades
    )

    # --------------------------------------------------------
    # 7. SALVA DATASET COMPLETO
    # --------------------------------------------------------

    trades.to_csv(
        OUTPUT_DIR /
        "diagnostic_v2_all_trades.csv",
        index=False
    )

    # --------------------------------------------------------
    # 8. DEVELOPMENT / VERIFICATION
    # --------------------------------------------------------

    development = trades[
        trades["dataset"] ==
        "DEVELOPMENT"
    ].copy()

    verification = trades[
        trades["dataset"] ==
        "VERIFICATION"
    ].copy()

    development.to_csv(
        OUTPUT_DIR /
        "diagnostic_v2_development.csv",
        index=False
    )

    verification.to_csv(
        OUTPUT_DIR /
        "diagnostic_v2_verification.csv",
        index=False
    )

    # --------------------------------------------------------
    # 9. SUMMARY
    # --------------------------------------------------------

    summary = pd.DataFrame([
        summarize(
            development,
            "DEVELOPMENT"
        ),

        summarize(
            verification,
            "VERIFICATION"
        ),

        summarize(
            trades,
            "ALL"
        )
    ])

    summary.to_csv(
        OUTPUT_DIR /
        "summary.csv",
        index=False
    )

    print_summary_table(
        summary
    )

    # --------------------------------------------------------
    # 10. BUY / SELL
    # --------------------------------------------------------

    direction_rows = []

    for dataset_name, dataset in trades.groupby(
        "dataset"
    ):

        for direction, group in dataset.groupby(
            "direction"
        ):

            s = summarize(
                group,
                f"{dataset_name}_{direction}"
            )

            direction_rows.append(s)

    direction_df = pd.DataFrame(
        direction_rows
    )

    direction_df.to_csv(
        OUTPUT_DIR /
        "by_direction.csv",
        index=False
    )

    print_separator(
        "BUY / SELL"
    )

    print(
        direction_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 11. RSI
    # --------------------------------------------------------

    rsi_df = grouped_analysis(
        trades,
        "rsi_bucket"
    )

    rsi_df.to_csv(
        OUTPUT_DIR /
        "by_rsi.csv",
        index=False
    )

    # --------------------------------------------------------
    # 12. EMA STRENGTH
    # --------------------------------------------------------

    ema_df = grouped_analysis(
        trades,
        "ema_strength"
    )

    ema_df.to_csv(
        OUTPUT_DIR /
        "by_ema_strength.csv",
        index=False
    )

    # --------------------------------------------------------
    # 13. MACD STRENGTH
    # --------------------------------------------------------

    macd_df = grouped_analysis(
        trades,
        "macd_strength"
    )

    macd_df.to_csv(
        OUTPUT_DIR /
        "by_macd_strength.csv",
        index=False
    )

    # --------------------------------------------------------
    # 14. ATR REGIME
    # --------------------------------------------------------

    atr_df = grouped_analysis(
        trades,
        "atr_regime"
    )

    atr_df.to_csv(
        OUTPUT_DIR /
        "by_atr_regime.csv",
        index=False
    )

    # --------------------------------------------------------
    # 15. BODY
    # --------------------------------------------------------

    body_df = grouped_analysis(
        trades,
        "body_bucket"
    )

    body_df.to_csv(
        OUTPUT_DIR /
        "by_body.csv",
        index=False
    )

    # --------------------------------------------------------
    # 16. TARGET ANALYSIS
    # --------------------------------------------------------

    targets_df = target_analysis(
        trades
    )

    targets_df.to_csv(
        OUTPUT_DIR /
        "target_analysis.csv",
        index=False
    )

    # --------------------------------------------------------
    # 17. TP/SL MATRIX
    # --------------------------------------------------------

    matrix_df = simulate_tp_sl_matrix(
        df,
        trades
    )

    matrix_df.to_csv(
        OUTPUT_DIR /
        "tp_sl_matrix.csv",
        index=False
    )

    # --------------------------------------------------------
    # 18. COMBINATIONS
    # --------------------------------------------------------

    combinations_df = combination_analysis(
        trades
    )

    combinations_df.to_csv(
        OUTPUT_DIR /
        "condition_combinations.csv",
        index=False
    )

    # --------------------------------------------------------
    # 19. CORRELAZIONI
    # --------------------------------------------------------

    correlations_df = correlation_analysis(
        trades
    )

    correlations_df.to_csv(
        OUTPUT_DIR /
        "feature_correlations.csv",
        index=False
    )

    # --------------------------------------------------------
    # 20. ANALISI ORIZZONTI
    # --------------------------------------------------------

    horizon_rows = []

    for dataset_name, dataset in trades.groupby(
        "dataset"
    ):

        for horizon_name in HORIZONS:

            mfe_col = (
                f"mfe_{horizon_name}_atr"
            )

            mae_col = (
                f"mae_{horizon_name}_atr"
            )

            if mfe_col not in dataset.columns:
                continue

            horizon_rows.append({
                "dataset": dataset_name,
                "horizon": horizon_name,
                "avg_MFE_ATR": dataset[
                    mfe_col
                ].mean(),
                "median_MFE_ATR": dataset[
                    mfe_col
                ].median(),
                "avg_MAE_ATR": dataset[
                    mae_col
                ].mean(),
                "median_MAE_ATR": dataset[
                    mae_col
                ].median()
            })

    horizons_df = pd.DataFrame(
        horizon_rows
    )

    horizons_df.to_csv(
        OUTPUT_DIR /
        "time_horizons.csv",
        index=False
    )

    # --------------------------------------------------------
    # 21. STAMPA ANALISI
    # --------------------------------------------------------

    print_top_analysis(
        combinations_df,
        "CONDIZIONI / COMBINAZIONI"
    )

    print_top_analysis(
        correlations_df.rename(
            columns={
                "feature": "condition"
            }
        ),
        "FEATURE CON MAGGIORE CORRELAZIONE"
    )

    print_separator(
        "TARGET RAGGIUNTI"
    )

    print(
        targets_df.to_string(
            index=False
        )
    )

    print_separator(
        "MATRICE TP / SL"
    )

    print(
        matrix_df.to_string(
            index=False
        )
    )

    print_separator(
        "FINE BACKTEST"
    )

    elapsed = (
        time.time() -
        start_time
    )

    print(
        f"Tempo totale: "
        f"{elapsed:.2f} secondi"
    )

    print(
        f"Risultati salvati in: "
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        "\nFile principali:"
    )

    for file in sorted(
        OUTPUT_DIR.glob("*.csv")
    ):
        print(
            f" - {file.name}"
        )


if __name__ == "__main__":
    main()
