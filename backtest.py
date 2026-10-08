import os
import time
import warnings
import requests
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# V10 — AGGRESSIVE BUT PRUDENT
# TREND + BREAKOUT + MOMENTUM
# ============================================================

SYMBOL = "XAU/USD"
INTERVAL = "5min"

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurato")


# ============================================================
# PARAMETRI FISSI V10
# ============================================================

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 100

BREAKOUT_LOOKBACK = 24

BODY_RATIO_MIN = 0.60
CLOSE_LOCATION_MAX = 0.20

MOMENTUM_ATR_MIN = 0.25

ATR_PERIOD = 14
ATR_AVG_PERIOD = 50

RSI_PERIOD = 14

BUY_RSI_MIN = 52
BUY_RSI_MAX = 68

SELL_RSI_MIN = 32
SELL_RSI_MAX = 48

EMA_GAP_ATR_MIN = 0.15

SL_ATR = 1.00
TP_ATR = 2.50

HORIZON_MIN = 60

COOLDOWN_MIN = 60

# Costi totali stimati:
# spread + slippage
ROUND_TRIP_COST = 0.10

OUTPUT_FILE = "backtest_results_v10.csv"


# ============================================================
# DOWNLOAD DATI
# ============================================================

def download_data():

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": 5000,
        "apikey": API_KEY,
        "timezone": "UTC",
        "order": "ASC"
    }

    print("=" * 78)
    print("V10 — AGGRESSIVE BUT PRUDENT")
    print("=" * 78)

    print("\nDownload dati Twelve Data...")

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise RuntimeError(
            f"Errore Twelve Data:\n{data}"
        )

    df = pd.DataFrame(data["values"])

    if df.empty:
        raise RuntimeError("Nessun dato ricevuto.")

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    df = df.sort_values("datetime").reset_index(drop=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=numeric_cols
    ).reset_index(drop=True)

    print(f"Candele scaricate: {len(df)}")

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

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema20"] = df["close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=EMA_MID,
        adjust=False
    ).mean()

    df["ema100"] = df["close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # TRUE RANGE
    # --------------------------------------------------------

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (df["high"] - prev_close).abs()

    tr3 = (df["low"] - prev_close).abs()

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    # --------------------------------------------------------
    # ATR — Wilder style
    # --------------------------------------------------------

    df["atr"] = df["tr"].ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False
    ).mean()

    df["atr_avg50"] = df["atr"].rolling(
        ATR_AVG_PERIOD
    ).mean()

    # --------------------------------------------------------
    # MOM6
    # --------------------------------------------------------

    df["mom6"] = (
        df["close"] -
        df["close"].shift(6)
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # --------------------------------------------------------
    # Candle body
    # --------------------------------------------------------

    candle_range = (
        df["high"] -
        df["low"]
    )

    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["body_ratio"] = (
        df["body"] /
        candle_range.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Close location
    #
    # BUY:
    # close nel 20% superiore
    #
    # SELL:
    # close nel 20% inferiore
    # --------------------------------------------------------

    df["close_position"] = (
        df["close"] -
        df["low"]
    ) / candle_range.replace(
        0,
        np.nan
    )

    # --------------------------------------------------------
    # BREAKOUT LEVEL
    #
    # IMPORTANTISSIMO:
    # shift(1) = non usa la candela corrente.
    #
    # Quindi non c'è lookahead.
    # --------------------------------------------------------

    df["previous_high"] = (
        df["high"]
        .rolling(BREAKOUT_LOOKBACK)
        .max()
        .shift(1)
    )

    df["previous_low"] = (
        df["low"]
        .rolling(BREAKOUT_LOOKBACK)
        .min()
        .shift(1)
    )

    # --------------------------------------------------------
    # EMA DISTANCE NORMALIZED BY ATR
    # --------------------------------------------------------

    df["ema_gap_atr"] = (
        (df["ema20"] - df["ema50"]) /
        df["atr"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # 15 MIN TREND
    #
    # Costruiamo candele 15m.
    # La candela 15m diventa disponibile
    # SOLO dopo la sua chiusura.
    # --------------------------------------------------------

    temp = df.set_index("datetime")

    tf15 = temp.resample(
        "15min",
        label="left",
        closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })

    tf15 = tf15.dropna()

    tf15["ema20_15"] = tf15["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    tf15["ema50_15"] = tf15["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    tf15["trend15"] = np.where(
        tf15["ema20_15"] >
        tf15["ema50_15"],
        "BULLISH",
        "BEARISH"
    )

    # La candela 15m iniziata alle 10:00
    # è disponibile alle 10:15.
    tf15["available_at"] = (
        tf15.index +
        pd.Timedelta(minutes=15)
    )

    tf15 = tf15[
        [
            "available_at",
            "trend15"
        ]
    ].sort_values("available_at")

    base = df.sort_values(
        "datetime"
    ).copy()

    # Entry sulla chiusura della candela 5m.
    base["entry_time"] = (
        base["datetime"] +
        pd.Timedelta(minutes=5)
    )

    base = pd.merge_asof(
        base.sort_values("entry_time"),
        tf15.sort_values("available_at"),
        left_on="entry_time",
        right_on="available_at",
        direction="backward"
    )

    base = base.drop(
        columns=["available_at"],
        errors="ignore"
    )

    return base


# ============================================================
# GENERAZIONE SEGNALI
# ============================================================

def generate_signal(row):

    required = [
        "ema20",
        "ema50",
        "ema100",
        "atr",
        "atr_avg50",
        "mom6",
        "rsi",
        "body_ratio",
        "close_position",
        "previous_high",
        "previous_low",
        "ema_gap_atr",
        "trend15"
    ]

    for col in required:

        if pd.isna(row[col]):
            return None

    # ========================================================
    # BUY
    # ========================================================

    buy = (

        # Trend principale
        row["ema20"] >
        row["ema50"] >

        row["ema100"]

        # EMA20 sufficientemente distante
        and row["ema_gap_atr"] >=
        EMA_GAP_ATR_MIN

        # Trend 15m
        and row["trend15"] ==
        "BULLISH"

        # Breakout
        and row["close"] >
        row["previous_high"]

        # Candela forte
        and row["body_ratio"] >=
        BODY_RATIO_MIN

        # Chiusura nel 20% superiore
        and row["close_position"] >=
        (1 - CLOSE_LOCATION_MAX)

        # Momentum
        and row["mom6"] >=
        MOMENTUM_ATR_MIN *
        row["atr"]

        # Volatilità sufficiente
        and row["atr"] >
        row["atr_avg50"]

        # RSI
        and row["rsi"] >=
        BUY_RSI_MIN

        and row["rsi"] <=
        BUY_RSI_MAX
    )

    if buy:
        return "BUY"

    # ========================================================
    # SELL
    # ========================================================

    sell = (

        # Trend principale
        row["ema20"] <
        row["ema50"] <
        row["ema100"]

        # EMA20 sufficientemente distante
        and row["ema_gap_atr"] <=
        -EMA_GAP_ATR_MIN

        # Trend 15m
        and row["trend15"] ==
        "BEARISH"

        # Breakout
        and row["close"] <
        row["previous_low"]

        # Candela forte
        and row["body_ratio"] >=
        BODY_RATIO_MIN

        # Chiusura nel 20% inferiore
        and row["close_position"] <=
        CLOSE_LOCATION_MAX

        # Momentum
        and row["mom6"] <=
        -MOMENTUM_ATR_MIN *
        row["atr"]

        # Volatilità sufficiente
        and row["atr"] >
        row["atr_avg50"]

        # RSI
        and row["rsi"] >=
        SELL_RSI_MIN

        and row["rsi"] <=
        SELL_RSI_MAX
    )

    if sell:
        return "SELL"

    return None


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(df, entry_index, direction):

    entry_row = df.iloc[entry_index]

    entry_price = float(
        entry_row["close"]
    )

    atr = float(
        entry_row["atr"]
    )

    if not np.isfinite(atr) or atr <= 0:
        return None

    if direction == "BUY":

        sl = (
            entry_price -
            SL_ATR * atr
        )

        tp = (
            entry_price +
            TP_ATR * atr
        )

    else:

        sl = (
            entry_price +
            SL_ATR * atr
        )

        tp = (
            entry_price -
            TP_ATR * atr
        )

    horizon_bars = HORIZON_MIN // 5

    last_index = min(
        len(df) - 1,
        entry_index + horizon_bars
    )

    exit_index = last_index
    exit_price = float(
        df.iloc[last_index]["close"]
    )

    result = "TIMEOUT"

    # ========================================================
    # SCANSIONE FUTURE CANDLE
    # ========================================================

    for j in range(
        entry_index + 1,
        last_index + 1
    ):

        candle = df.iloc[j]

        high = float(
            candle["high"]
        )

        low = float(
            candle["low"]
        )

        if direction == "BUY":

            hit_sl = low <= sl
            hit_tp = high >= tp

            # Se SL e TP vengono toccati
            # nella stessa candela:
            # scelta conservativa -> SL.
            if hit_sl and hit_tp:

                exit_index = j
                exit_price = sl
                result = "SL"
                break

            if hit_sl:

                exit_index = j
                exit_price = sl
                result = "SL"
                break

            if hit_tp:

                exit_index = j
                exit_price = tp
                result = "TP"
                break

        else:

            hit_sl = high >= sl
            hit_tp = low <= tp

            # Conservativo:
            # se entrambi nella stessa candela -> SL.
            if hit_sl and hit_tp:

                exit_index = j
                exit_price = sl
                result = "SL"
                break

            if hit_sl:

                exit_index = j
                exit_price = sl
                result = "SL"
                break

            if hit_tp:

                exit_index = j
                exit_price = tp
                result = "TP"
                break

    # ========================================================
    # R NETTO
    # ========================================================

    if direction == "BUY":

        gross_move = (
            exit_price -
            entry_price
        )

    else:

        gross_move = (
            entry_price -
            exit_price
        )

    net_move = (
        gross_move -
        ROUND_TRIP_COST
    )

    risk = SL_ATR * atr

    r_multiple = (
        net_move /
        risk
    )

    return {
        "entry_index": entry_index,
        "exit_index": exit_index,
        "entry_time": entry_row["entry_time"],
        "exit_time": df.iloc[exit_index]["entry_time"],
        "direction": direction,
        "entry": entry_price,
        "sl": sl,
        "tp": tp,
        "exit": exit_price,
        "atr": atr,
        "r": r_multiple,
        "result": result
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(df):

    trades = []

    last_entry_time = None
    next_available
