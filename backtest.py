import os
import time
import math
import warnings
import traceback
from itertools import product

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


# ============================================================
# V9 - ROBUSTNESS / OPTIMIZATION SCANNER
# ============================================================

SYMBOL = "XAU/USD"
INTERVAL = "5min"

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurato")


# ============================================================
# CONFIGURAZIONE DATI
# ============================================================

TOTAL_CANDLES = 10000
BLOCK_SIZE = 5000

DEV_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

OUTPUT_DIR = "backtest_v9"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PARAMETRI V9
# ============================================================

# Body strength
BODY_RATIOS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

# RSI
RSI_RANGES = [
    (30, 65),
    (32, 65),
    (35, 65),
    (35, 60),
    (38, 62),
    (40, 65),
]

# Momentum minimo
MOM6_MIN_ATR = [
    0.00,
    0.10,
    0.20,
    0.30,
]

# EMA trend strength.
# Per SELL:
# EMA20 < EMA50 < EMA100
# Per BUY:
# EMA20 > EMA50 > EMA100

EMA_GAP_ATR = [
    0.00,
    0.05,
    0.10,
    0.20,
]

# ATR regime.
# ATR relativo alla sua media.
ATR_REGIMES = [
    "ANY",
    "NORMAL_HIGH",
    "HIGH",
]

# TP / SL
TP_ATR_VALUES = [
    1.75,
    2.00,
    2.25,
    2.50,
]

SL_ATR_VALUES = [
    0.75,
    1.00,
    1.25,
]

# Durata massima trade
HORIZON_MIN_VALUES = [
    30,
    45,
    60,
]

# Modalità ingresso
ENTRY_MODES = [
    "FIRST",
    "COOLDOWN_30",
    "COOLDOWN_60",
]

# Direzioni
DIRECTIONS = [
    "SELL",
    "BUY",
]


# ============================================================
# DOWNLOAD DATI
# ============================================================

def richiesta_dati(outputsize=5000, start_date=None):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": outputsize,
        "timezone": "UTC",
        "apikey": API_KEY,
    }

    if start_date is not None:
        params["end_date"] = start_date

    risposta = requests.get(
        url,
        params=params,
        timeout=30,
    )

    risposta.raise_for_status()

    dati = risposta.json()

    if "values" not in dati:
        raise RuntimeError(
            f"Errore Twelve Data: {dati}"
        )

    return pd.DataFrame(dati["values"])


def scarica_dati():

    print("=" * 78)
    print("DOWNLOAD DATI")
    print("=" * 78)

    blocchi = []
    end_date = None

    for i in range(2):

        print(
            f"Scarico blocco {i + 1}/2 "
            f"({BLOCK_SIZE} candele)..."
        )

        df_block = richiesta_dati(
            outputsize=BLOCK_SIZE,
            start_date=end_date,
        )

        if df_block.empty:
            break

        blocchi.append(df_block)

        df_block["datetime"] = pd.to_datetime(
            df_block["datetime"],
            utc=True,
        )

        oldest = df_block["datetime"].min()

        print(
            f"Ricevute: {len(df_block)} | "
            f"Oldest: {oldest}"
        )

        end_date = (
            oldest.strftime("%Y-%m-%d %H:%M:%S")
        )

        time.sleep(1)

    if not blocchi:
        raise RuntimeError("Nessun dato scaricato")

    df = pd.concat(
        blocchi,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # NORMALIZZAZIONE
    # --------------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = (
        df
        .dropna(
            subset=[
                "datetime",
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # SOLO CANDELE CHIUSE
    # --------------------------------------------------------

    now_utc = pd.Timestamp.now(tz="UTC")

    candle_duration = pd.Timedelta(minutes=5)

    df = df[
        df["datetime"] + candle_duration <= now_utc
    ].copy()

    df = (
        df
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # Ultime 10000
    if len(df) > TOTAL_CANDLES:
        df = df.iloc[-TOTAL_CANDLES:].reset_index(
            drop=True
        )

    print("")
    print(f"Candele finali: {len(df)}")

    if len(df):
        print(
            f"Periodo: "
            f"{df['datetime'].iloc[0]} -> "
            f"{df['datetime'].iloc[-1]}"
        )

    return df


# ============================================================
# INDICATORI
# ============================================================

def ema(series, span):

    return (
        series
        .ewm(
            span=span,
            adjust=False,
        )
        .mean()
    )


def calcola_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain
        .ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False,
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False,
        )
        .mean()
    )

    rs = avg_gain / avg_loss

    return (
        100 -
        (100 / (1 + rs))
    )


def calcola_atr(df, period=14):

    prev_close = df["close"].shift()

    tr1 = (
        df["high"] -
        df["low"]
    )

    tr2 = (
        df["high"] -
        prev_close
    ).abs()

    tr3 = (
        df["low"] -
        prev_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    return (
        tr
        .ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False,
        )
        .mean()
    )


def calcola_indicatori(df):

    df = df.copy()

    close = df["close"]

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["EMA20"] = ema(close, 20)
    df["EMA50"] = ema(close, 50)
    df["EMA100"] = ema(close, 100)

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = ema(
        df["MACD"],
        9,
    )

    df["MACD_HIST"] = (
        df["MACD"] -
        df["MACD_SIGNAL"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["RSI"] = calcola_rsi(
        close,
        14,
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["ATR"] = calcola_atr(
        df,
        14,
    )

    # --------------------------------------------------------
    # MOMENTUM 6 CANDELE
    # --------------------------------------------------------

    df["MOM6"] = (
        close -
        close.shift(6)
    )

    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    df["BODY"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["RANGE"] = (
        df["high"] -
        df["low"]
    )

    df["BODY_RATIO"] = np.where(
        df["RANGE"] > 0,
        df["BODY"] / df["RANGE"],
        0,
    )

    # --------------------------------------------------------
    # CLOSE LOCATION
    #
    # 0 = close vicino al minimo
    # 1 = close vicino al massimo
    # --------------------------------------------------------

    df["CLOSE_LOCATION"] = np.where(
        df["RANGE"] > 0,
        (
            df["close"] -
            df["low"]
        ) / df["RANGE"],
        0.5,
    )

    # --------------------------------------------------------
    # ATR RELATIVO
    # --------------------------------------------------------

    df["ATR_MEAN_50"] = (
        df["ATR"]
        .rolling(50)
        .mean()
    )

    df["ATR_RELATIVE"] = np.where(
        df["ATR_MEAN_50"] > 0,
        df["ATR"] /
        df["ATR_MEAN_50"],
        np.nan,
    )

    # --------------------------------------------------------
    # DISTANZE EMA NORMALIZZATE ATR
    # --------------------------------------------------------

    df["EMA_GAP_20_50_ATR"] = (
        (
            df["EMA20"] -
            df["EMA50"]
        ).abs()
        / df["ATR"]
    )

    df["EMA_GAP_50_100_ATR"] = (
        (
            df["EMA50"] -
            df["EMA100"]
        ).abs()
        / df["ATR"]
    )

    # --------------------------------------------------------
    # 15 MINUTI
    # --------------------------------------------------------

    temp = (
        df
        .set_index("datetime")
        .resample("15min")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        })
        .dropna()
        .reset_index()
    )

    temp["EMA20_15"] = ema(
        temp["close"],
        20,
    )

    temp["EMA50_15"] = ema(
        temp["close"],
        50,
    )

    temp["EMA100_15"] = ema(
        temp["close"],
        100,
    )

    temp["ATR_15"] = calcola_atr(
        temp,
        14,
    )

    temp["TREND_15"] = np.select(
        [
            (
                (temp["EMA20_15"] > temp["EMA50_15"])
                &
                (temp["EMA50_15"] > temp["EMA100_15"])
            ),
            (
                (temp["EMA20_15"] < temp["EMA50_15"])
                &
                (temp["EMA50_15"] < temp["EMA100_15"])
            ),
        ],
        [
            "BULLISH_STRONG",
            "BEARISH_STRONG",
        ],
        default="NEUTRAL",
    )

    # Solo candele 15m completamente chiuse
    temp["CLOSE_TIME"] = (
        temp["datetime"] +
        pd.Timedelta(minutes=15)
    )

    temp = temp[
        temp["CLOSE_TIME"]
        <= df["datetime"].max()
        + pd.Timedelta(minutes=5)
    ].copy()

    # --------------------------------------------------------
    # ASOF JOIN
    # --------------------------------------------------------

    df = pd.merge_asof(
        df.sort_values("datetime"),
        temp[
            [
                "datetime",
                "EMA20_15",
                "EMA50_15",
                "EMA100_15",
                "ATR_15",
                "TREND_15",
            ]
        ].sort_values("datetime"),
        on="datetime",
        direction="backward",
    )

    return df.reset_index(drop=True)


# ============================================================
# SPLIT
# ============================================================

def crea_split(df):

    n = len(df)

    dev_end = int(
        n * DEV_RATIO
    )

    val_end = int(
        n * (DEV_RATIO + VAL_RATIO)
    )

    dev = df.iloc[
        :dev_end
    ].copy()

    val = df.iloc[
        dev_end:val_end
    ].copy()

    test = df.iloc[
        val_end:
    ].copy()

    return dev, val, test


# ============================================================
# FILTRO ATR
# ============================================================

def atr_regime_ok(value, regime):

    if pd.isna(value):
        return False

    if regime == "ANY":
        return True

    if regime == "NORMAL_HIGH":
        return value >= 1.00

    if regime == "HIGH":
        return value >= 1.15

    return True


# ============================================================
# GENERAZIONE SETUP
# ============================================================

def setup_valido(row, params):

    direction = params["direction"]

    rsi_low = params["rsi_low"]
    rsi_high = params["rsi_high"]

    body_ratio = params["body_ratio"]
    mom_min_atr = params["mom_min_atr"]
    ema_gap = params["ema_gap_atr"]
    atr_regime = params["atr_regime"]

    if (
        pd.isna(row["ATR"])
        or row["ATR"] <= 0
    ):
        return False

    if pd.isna(row["RSI"]):
        return False

    if pd.isna(row["TREND_15"]):
        return False

    # --------------------------------------------------------
    # FILTRO BODY
    # --------------------------------------------------------

    if row["BODY_RATIO"] < body_ratio:
        return False

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    if not atr_regime_ok(
        row["ATR_RELATIVE"],
        atr_regime,
    ):
        return False

    # --------------------------------------------------------
    # EMA GAP
    # --------------------------------------------------------

    if (
        row["EMA_GAP_20_50_ATR"]
        < ema_gap
    ):
        return False

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    if direction == "SELL":

        trend_ok = (
            row["TREND_15"]
            == "BEARISH_STRONG"
        )

        ema_ok = (
            row["EMA20"]
            < row["EMA50"]
            < row["EMA100"]
        )

        macd_ok = (
            row["MACD"]
            < row["MACD_SIGNAL"]
        )

        rsi_ok = (
            rsi_low
            < row["RSI"]
            < rsi_high
        )

        mom_ok = (
            row["MOM6"]
            <= -mom_min_atr * row["ATR"]
        )

        # Candela ribassista:
        # close sotto open
        candle_direction_ok = (
            row["close"]
            < row["open"]
        )

        # Chiusura nella parte bassa
        close_location_ok = (
            row["CLOSE_LOCATION"]
            <= 0.40
        )

        return (
            trend_ok
            and ema_ok
            and macd_ok
            and rsi_ok
            and mom_ok
            and candle_direction_ok
            and close_location_ok
        )

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if direction == "BUY":

        trend_ok = (
            row["TREND_15"]
            == "BULLISH_STRONG"
        )

        ema_ok = (
            row["EMA20"]
            > row["EMA50"]
            > row["EMA100"]
        )

        macd_ok = (
            row["MACD"]
            > row["MACD_SIGNAL"]
        )

        rsi_ok = (
            rsi_low
            < row["RSI"]
            < rsi_high
        )

        mom_ok = (
            row["MOM6"]
            >= mom_min_atr * row["ATR"]
        )

        candle_direction_ok = (
            row["close"]
            > row["open"]
        )

        close_location_ok = (
            row["CLOSE_LOCATION"]
            >= 0.60
        )

        return (
            trend_ok
            and ema_ok
            and macd_ok
            and rsi_ok
            and mom_ok
            and candle_direction_ok
            and close_location_ok
        )

    return False


# ============================================================
# SIMULAZIONE TRADE
# ============================================================

def simula_trade(
    df,
    entry_index,
    direction,
    tp_atr,
    sl_atr,
    horizon_min,
):

    entry_row = df.iloc[entry_index]

    entry_price = float(
        entry_row["close"]
    )

    atr = float(
        entry_row["ATR"]
    )

    if (
        not np.isfinite(entry_price)
        or not np.isfinite(atr)
        or atr <= 0
    ):
        return None

    if direction == "SELL":

        sl = (
            entry_price +
            sl_atr * atr
        )

        tp = (
            entry_price -
            tp_atr * atr
        )

    else:

        sl = (
            entry_price -
            sl_atr * atr
        )

        tp = (
            entry_price +
            tp_atr * atr
        )

    max_bars = int(
        horizon_min / 5
    )

    last_index = min(
        entry_index + max_bars,
        len(df) - 1,
    )

    exit_index = last_index
    result = "TIMEOUT"

    # --------------------------------------------------------
    # BAR SUCCESSIVE
    # --------------------------------------------------------

    for j in range(
        entry_index + 1,
        last_index + 1,
    ):

        row = df.iloc[j]

        high = float(row["high"])
        low = float(row["low"])

        if direction == "SELL":

            hit_sl = high >= sl
            hit_tp = low <= tp

            # Se entrambi vengono colpiti nella stessa
            # candela usiamo conservativamente SL.
            if hit_sl and hit_tp:

                result = "SL"
                exit_index = j
                break

            if hit_sl:

                result = "SL"
                exit_index = j
                break

            if hit_tp:

                result = "TP"
                exit_index = j
                break

        else:

            hit_sl = low <= sl
            hit_tp = high >= tp

            if hit_sl and hit_tp:

                result = "SL"
                exit_index = j
                break

            if hit_sl:

                result = "SL"
                exit_index = j
                break

            if hit_tp:

                result = "TP"
                exit_index = j
                break

    # --------------------------------------------------------
    # EXIT PRICE
    # --------------------------------------------------------

    if result == "TP":

        exit_price = tp

    elif result == "SL":

        exit_price = sl

    else:

        exit_price = float(
            df.iloc[exit_index]["close"]
        )

    # --------------------------------------------------------
    # R MULTIPLE
    # --------------------------------------------------------

    risk = sl_atr * atr

    if risk <= 0:
        return None

    if direction == "SELL":

        pnl = (
            entry_price -
            exit_price
        )

    else:

        pnl = (
            exit_price -
            entry_price
        )

    R = pnl / risk

    return {
        "entry_index": entry_index,
        "exit_index": exit_index,
        "entry_time": entry_row["datetime"],
        "exit_time": df.iloc[exit_index]["datetime"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "SL": sl,
        "TP": tp,
        "ATR": atr,
        "result": result,
        "R": R,
    }


# ============================================================
# SIMULAZIONE STRATEGIA
# ============================================================

def run_strategy(
    df,
    params,
):

    trades = []

    last_entry_index = -10_000

    cooldown_bars = 0

    if params["entry_mode"] == "COOLDOWN_30":
        cooldown_bars = 6

    elif params["entry_mode"] == "COOLDOWN_60":
        cooldown_bars = 12

    # --------------------------------------------------------
    # SCANSIONE
    # --------------------------------------------------------

    i = 0

    while i < len(df):

        row = df.iloc[i]

        # ----------------------------------------------------
        # COOLDOWN
        # ----------------------------------------------------

        if (
            i - last_entry_index
            < cooldown_bars
        ):
            i += 1
            continue

        # ----------------------------------------------------
        # SETUP
        # ----------------------------------------------------

        if not setup_valido(
            row,
            params,
        ):
            i += 1
            continue

        # ----------------------------------------------------
        # TRADE
        # ----------------------------------------------------

        trade = simula_trade(
            df=df,
            entry_index=i,
            direction=params["direction"],
            tp_atr=params["tp_atr"],
            sl_atr=params["sl_atr"],
            horizon_min=params["horizon_min"],
        )

        if trade is None:
            i += 1
            continue

        trades.append(trade)

        last_entry_index = i

        # FIRST:
        # saltiamo fino alla chiusura del trade.
        #
        # COOLDOWN:
        # possiamo comunque considerare il trade chiuso,
        # ma applichiamo il cooldown.

        if params["entry_mode"] == "FIRST":

            i = max(
                i + 1,
                trade["exit_index"] + 1,
            )

        else:

            i += 1

    return trades


# ============================================================
# METRICHE
# ============================================================

def metriche(trades):

    if not trades:

        return {
            "trades": 0,
            "TP": 0,
            "SL": 0,
            "TIMEOUT": 0,
            "win_rate": 0.0,
            "avg_R": 0.0,
            "total_R": 0.0,
            "PF": 0.0,
            "maxDD": 0.0,
            "median_R": 0.0,
        }

    df = pd.DataFrame(trades)

    R = df["R"].astype(float)

    tp = int(
        (df["result"] == "TP").sum()
    )

    sl = int(
        (df["result"] == "SL").sum()
    )

    timeout = int(
        (df["result"] == "TIMEOUT").sum()
    )

    wins = int(
        (R > 0).sum()
    )

    gross_profit = float(
        R[R > 0].sum()
    )

    gross_loss = float(
        -R[R < 0].sum()
    )

    if gross_loss > 0:
        pf = (
            gross_profit /
            gross_loss
        )
    elif gross_profit > 0:
        pf = float("inf")
    else:
        pf = 0.0

    equity = R.cumsum()

    running_max = equity.cummax()

    drawdown = (
        running_max -
        equity
    )

    max_dd = float(
        drawdown.max()
    ) if len(drawdown) else 0.0

    return {
        "trades": len(df),
        "TP": tp,
        "SL": sl,
        "TIMEOUT": timeout,
        "win_rate": (
            wins / len(df) * 100
        ),
        "avg_R": float(
            R.mean()
        ),
        "total_R": float(
            R.sum()
        ),
        "PF": float(pf),
        "maxDD": max_dd,
        "median_R": float(
            R.median()
        ),
    }


# ============================================================
# ROBUST SCORE
# ============================================================

def safe_pf(pf):

    if np.isinf(pf):
        return 3.0

    if pd.isna(pf):
        return 0.0

    return float(
        np.clip(pf, 0, 3)
    )


def robust_score(
    dev,
    val,
    test=None,
):

    # --------------------------------------------------------
    # DEV + VAL
    #
    # Il TEST NON entra nel ranking principale.
    # --------------------------------------------------------

    if (
        dev["trades"] < 30
        or val["trades"] < 10
    ):
        base = -1000.0

    else:

        avg_r = (
            0.50 * dev["avg_R"]
            +
            0.50 * val["avg_R"]
        )

        pf = (
            0.45 * safe_pf(dev["PF"])
            +
            0.55 * safe_pf(val["PF"])
        )

        stability = (
            1.0
            if (
                dev["avg_R"] > 0
                and val["avg_R"] > 0
            )
            else 0.25
        )

        trade_factor = min(
            1.0,
            math.sqrt(
                min(
                    dev["trades"],
                    val["trades"],
                ) / 50
            ),
        )

        dd_penalty = (
            1.0 /
            (
                1.0 +
                0.05 *
                (
                    dev["maxDD"]
                    +
                    val["maxDD"]
                )
            )
        )

        base = (
            100 *
            avg_r *
            pf *
            stability *
            trade_factor *
            dd_penalty
        )

    return float(base)


def combined_score(
    dev,
    val,
    test,
):

    # Questo è SOLO diagnostico.
    # Non viene usato per scegliere la strategia.

    avg_r = (
        0.30 * dev["avg_R"]
        +
        0.30 * val["avg_R"]
        +
        0.40 * test["avg_R"]
    )

    pf = (
        0.30 * safe_pf(dev["PF"])
        +
        0.30 * safe_pf(val["PF"])
        +
        0.40 * safe_pf(test["PF"])
    )

    positive = (
        (
            dev["avg_R"] > 0
        )
        +
        (
            val["avg_R"] > 0
        )
        +
        (
            test["avg_R"] > 0
        )
    )

    stability = (
        1.0
        if positive == 3
        else 0.50
        if positive == 2
        else 0.20
    )

    return float(
        100 *
        avg_r *
        pf *
        stability
    )


# ============================================================
# ESECUZIONE COMBINAZIONE
# ============================================================

def valuta_combinazione(
    dev,
    val,
    test,
    params,
):

    dev_trades = run_strategy(
        dev,
        params,
    )

    val_trades = run_strategy(
        val,
        params,
    )

    test_trades = run_strategy(
        test,
        params,
    )

    dev_m = metriche(dev_trades)
    val_m = metriche(val_trades)
    test_m = metriche(test_trades)

    score = robust_score(
        dev_m,
        val_m,
        test_m,
    )

    combined = combined_score(
        dev_m,
        val_m,
        test_m,
    )

    row = {
        **params,

        "DEV_trades": dev_m["trades"],
        "DEV_TP": dev_m["TP"],
        "DEV_SL": dev_m["SL"],
        "DEV_TIMEOUT": dev_m["TIMEOUT"],
        "DEV_win_rate": dev_m["win_rate"],
        "DEV_avg_R": dev_m["avg_R"],
        "DEV_total_R": dev_m["total_R"],
        "DEV_PF": dev_m["PF"],
        "DEV_maxDD": dev_m["maxDD"],

        "VAL_trades": val_m["trades"],
        "VAL_TP": val_m["TP"],
        "VAL_SL": val_m["SL"],
        "VAL_TIMEOUT": val_m["TIMEOUT"],
        "VAL_win_rate": val_m["win_rate"],
        "VAL_avg_R": val_m["avg_R"],
        "VAL_total_R": val_m["total_R"],
        "VAL_PF": val_m["PF"],
        "VAL_maxDD": val_m["maxDD"],

        "TEST_trades": test_m["trades"],
        "TEST_TP": test_m["TP"],
        "TEST_SL": test_m["SL"],
        "TEST_TIMEOUT": test_m["TIMEOUT"],
        "TEST_win_rate": test_m["win_rate"],
        "TEST_avg_R": test_m["avg_R"],
        "TEST_total_R": test_m["total_R"],
        "TEST_PF": test_m["PF"],
        "TEST_maxDD": test_m["maxDD"],

        "ROBUST_SCORE": score,
        "COMBINED_SCORE_DIAGNOSTIC": combined,
    }

    return row


# ============================================================
# SCANNER
# ============================================================

def costruisci_parametri():

    params_list = []

    for (
        direction,
        body_ratio,
        rsi_range,
        mom_min_atr,
        ema_gap_atr,
        atr_regime,
        tp_atr,
        sl_atr,
        horizon_min,
        entry_mode,
    ) in product(
        DIRECTIONS,
        BODY_RATIOS,
        RSI_RANGES,
        MOM6_MIN_ATR,
        EMA_GAP_ATR,
        ATR_REGIMES,
        TP_ATR_VALUES,
        SL_ATR_VALUES,
        HORIZON_MIN_VALUES,
        ENTRY_MODES,
    ):

        params_list.append({
            "direction": direction,
            "body_ratio": body_ratio,
            "rsi_low": rsi_range[0],
            "rsi_high": rsi_range[1],
            "mom_min_atr": mom_min_atr,
            "ema_gap_atr": ema_gap_atr,
            "atr_regime": atr_regime,
            "tp_atr": tp_atr,
            "sl_atr": sl_atr,
            "horizon_min": horizon_min,
            "entry_mode": entry_mode,
        })

    return params_list


# ============================================================
# RUN SCANNER
# ============================================================

def main():

    print("")
    print("=" * 78)
    print("BACKTEST V9 - ROBUSTNESS OPTIMIZATION SCANNER")
    print("=" * 78)
    print("")
    print("V9:")
    print("- multi-filter optimization")
    print("- BUY / SELL separati")
    print("- Body Ratio")
    print("- RSI")
    print("- MOM6 normalizzato ATR")
    print("- EMA trend 20/50/100")
    print("- EMA gap normalizzato ATR")
    print("- ATR regime")
    print("- 15m strong trend")
    print("- TP / SL")
    print("- Horizon")
    print("- Cooldown")
    print("- DEV / VAL / TEST")
    print("- TEST escluso dal ranking principale")
    print("")

    # --------------------------------------------------------
    # DATI
    # --------------------------------------------------------

    df = scarica_dati()

    if len(df) < 1000:
        raise RuntimeError(
            "Dati insufficienti"
        )

    # --------------------------------------------------------
    # INDICATORI
    # --------------------------------------------------------

    print("")
    print("=" * 78)
    print("CALCOLO INDICATORI")
    print("=" * 78)

    df = calcola_indicatori(df)

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "candles_with_indicators.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    dev, val, test = crea_split(df)

    print("")
    print("=" * 78)
    print("SPLIT")
    print("=" * 78)

    print(
        f"DEV   : 0 -> {len(dev)-1} "
        f"({len(dev)} candles)"
    )

    print(
        f"VAL   : {len(dev)} -> "
        f"{len(dev)+len(val)-1} "
        f"({len(val)} candles)"
    )

    print(
        f"TEST  : {len(dev)+len(val)} -> "
        f"{len(df)-1} "
        f"({len(test)} candles)"
    )

    print("")
    print(
        "CORREZIONE V9: "
        "ogni trade viene chiuso dentro il proprio split."
    )

    # --------------------------------------------------------
    # PARAMETRI
    # --------------------------------------------------------

    params_list = costruisci_parametri()

    print("")
    print(
        f"Combinazioni da testare: "
        f"{len(params_list)}"
    )

    # --------------------------------------------------------
    # SCAN
    # --------------------------------------------------------

    risultati = []

    start = time.time()

    for counter, params in enumerate(
        params_list,
        start=1,
    ):

        try:

            result = valuta_combinazione(
                dev,
                val,
                test,
                params,
            )

            risultati.append(result)

        except Exception as e:

            print(
                f"Errore combinazione "
                f"{counter}: {e}"
            )

        if (
            counter % 100 == 0
            or counter == len(params_list)
        ):

            elapsed = (
                time.time() -
                start
            )

            print(
                f"Progress: "
                f"{counter}/{len(params_list)} "
                f" | "
                f"{elapsed:.1f}s"
            )

    results = pd.DataFrame(
        risultati
    )

    # --------------------------------------------------------
    # CSV COMPLETO
    # --------------------------------------------------------

    results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "scan_all.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # TOP DEV + VAL
    # --------------------------------------------------------

    top_dev_val = (
        results
        .sort_values(
            "ROBUST_SCORE",
            ascending=False,
        )
        .head(50)
        .reset_index(drop=True)
    )

    top_dev_val.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "top_dev_val.csv",
        ),
        index=False,
    )

    print("")
    print("=" * 78)
    print("TOP 30 DEV + VALIDATION")
    print("=" * 78)

    print(
        top_dev_val.head(30).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # ROBUST POSITIVI DEV + VAL
    # --------------------------------------------------------

    robust = results[
        (results["DEV_trades"] >= 30)
        &
        (results["VAL_trades"] >= 10)
        &
        (results["DEV_avg_R"] > 0)
        &
        (results["VAL_avg_R"] > 0)
        &
        (results["DEV_PF"] > 1.0)
        &
        (results["VAL_PF"] > 1.0)
    ].copy()

    robust = (
        robust
        .sort_values(
            "ROBUST_SCORE",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    robust.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "robust_dev_val.csv",
        ),
        index=False,
    )

    print("")
    print("=" * 78)
    print("ROBUST DEV + VALIDATION")
    print("=" * 78)

    if robust.empty:

        print(
            "NESSUNA configurazione "
            "positiva contemporaneamente su DEV + VAL."
        )

    else:

        print(
            robust.head(30).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # CONTROLLO TEST
    # --------------------------------------------------------

    # ATTENZIONE:
    # questo NON viene usato per scegliere
    # la configurazione.
    #
    # Serve solo a vedere cosa è successo
    # sul periodo completamente fuori campione.

    test_positive = results[
        (results["TEST_trades"] >= 10)
        &
        (results["TEST_avg_R"] > 0)
        &
        (results["TEST_PF"] > 1.0)
    ].copy()

    test_positive = (
        test_positive
        .sort_values(
            "TEST_avg_R",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    test_positive.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_positive.csv",
        ),
        index=False,
    )

    print("")
    print("=" * 78)
    print("TEST POSITIVE - SOLO DIAGNOSTICA")
    print("=" * 78)

    if test_positive.empty:

        print(
            "Nessuna configurazione "
            "positiva nel TEST con almeno 10 trade."
        )

    else:

        print(
            test_positive.head(30).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # TOP TEST DEI CANDIDATI ROBUSTI
    # --------------------------------------------------------

    if not robust.empty:

        robust_test = (
            robust
            .sort_values(
                [
                    "TEST_avg_R",
                    "TEST_PF",
                ],
                ascending=False,
            )
            .reset_index(drop=True)
        )

        robust_test.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "robust_dev_val_test.csv",
            ),
            index=False,
        )

        print("")
        print("=" * 78)
        print("ROBUST DEV + VAL -> CONTROLLO TEST")
        print("=" * 78)

        print(
            robust_test.head(30).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # SUMMARY PER DIREZIONE
    # --------------------------------------------------------

    direction_summary = (
        results
        .groupby("direction")
        .agg(
            combinations=("direction", "size"),
            avg_test_R=("TEST_avg_R", "mean"),
            median_test_R=("TEST_avg_R", "median"),
            avg_test_PF=("TEST_PF", "mean"),
            positive_test_pct=(
                "TEST_avg_R",
                lambda x:
                (
                    (x > 0).sum()
                    /
                    len(x)
                    * 100
                )
            ),
            total_test_R=("TEST_total_R", "sum"),
        )
        .reset_index()
        .sort_values(
            "avg_test_R",
            ascending=False,
        )
    )

    direction_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "direction_summary_test.csv",
        ),
        index=False,
    )

    print("")
    print("=" * 78)
    print("DIRECTION SUMMARY TEST")
    print("=" * 78)

    print(
        direction_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SUMMARY BODY RATIO
    # --------------------------------------------------------

    body_summary = (
        results
        .groupby(
            [
                "direction",
                "body_ratio",
            ]
        )
        .agg(
            combinations=("direction", "size"),
            avg_test_R=("TEST_avg_R", "mean"),
            median_test_R=("TEST_avg_R", "median"),
            avg_test_PF=("TEST_PF", "mean"),
            positive_test_pct=(
                "TEST_avg_R",
                lambda x:
                (
                    (x > 0).sum()
                    /
                    len(x)
                    * 100
                )
            ),
        )
        .reset_index()
        .sort_values(
            "avg_test_R",
            ascending=False,
        )
    )

    body_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "body_ratio_summary.csv",
        ),
        index=False,
    )

    print("")
    print("=" * 78)
    print("BODY RATIO SUMMARY")
    print("=" * 78)

    print(
        body_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SUMMARY MODE
    # --------------------------------------------------------

    mode_summary = (
        results
        .groupby(
            [
                "direction",
                "atr_regime",
                "entry_mode",
            ]
        )
        .agg(
            combinations=("direction", "size"),
            avg_test_R=("TEST_avg_R", "mean"),
            median_test_R=("TEST_avg_R", "median"),
            avg_test_PF=("TEST_PF", "mean"),
            positive_test_pct=(
                "TEST_avg_R",
                lambda x:
                (
                    (x > 0).sum()
                    /
                    len(x)
                    * 100
                )
            ),
            total_test_R=("TEST_total_R", "sum"),
        )
        .reset_index()
        .sort_values(
            "avg_test_R",
            ascending=False,
        )
    )

    mode_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "mode_summary_test.csv",
        ),
        index=False,
    )

    print("")
    print("=" * 78)
    print("MODE SUMMARY TEST")
    print("=" * 78)

    print(
        mode_summary.head(50).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # TRADES DELLE MIGLIORI CONFIGURAZIONI
    # --------------------------------------------------------

    candidate_source = robust

    if candidate_source.empty:
        candidate_source = top_dev_val

    trades_rows = []

    for _, candidate in (
        candidate_source
        .head(20)
        .iterrows()
    ):

        params = {
            "direction": candidate["direction"],
            "body_ratio": candidate["body_ratio"],
            "rsi_low": candidate["rsi_low"],
            "rsi_high": candidate["rsi_high"],
            "mom_min_atr": candidate["mom_min_atr"],
            "ema_gap_atr": candidate["ema_gap_atr"],
            "atr_regime": candidate["atr_regime"],
            "tp_atr": candidate["tp_atr"],
            "sl_atr": candidate["sl_atr"],
            "horizon_min": candidate["horizon_min"],
            "entry_mode": candidate["entry_mode"],
        }

        # Salviamo trades DEV
        for split_name, split_df in [
            ("DEV", dev),
            ("VAL", val),
            ("TEST", test),
        ]:

            trades = run_strategy(
                split_df,
                params,
            )

            for trade in trades:

                trades_rows.append({
                    "split": split_name,
                    **params,
                    **trade,
                })

    if trades_rows:

        trades_df = pd.DataFrame(
            trades_rows
        )

        trades_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "trades_top_candidates.csv",
            ),
            index=False,
        )

    # --------------------------------------------------------
    # INDICAZIONE FINALE
    # --------------------------------------------------------

    print("")
    print("=" * 78)
    print("RISULTATO V9")
    print("=" * 78)

    if not robust.empty:

        best = robust.iloc[0]

        print("")
        print("CONFIGURAZIONE PRINCIPALE V9")
        print("----------------------------------------")
        print(
            f"Direction       : "
            f"{best['direction']}"
        )
        print(
            f"Body Ratio      : "
            f">= {best['body_ratio']:.2f}"
        )
        print(
            f"RSI             : "
            f"{best['rsi_low']:.0f} - "
            f"{best['rsi_high']:.0f}"
        )
        print(
            f"MOM6 min ATR    : "
            f"{best['mom_min_atr']:.2f}"
        )
        print(
            f"EMA gap ATR     : "
            f"{best['ema_gap_atr']:.2f}"
        )
        print(
            f"ATR regime      : "
            f"{best['atr_regime']}"
        )
        print(
            f"Entry mode      : "
            f"{best['entry_mode']}"
        )
        print(
            f"TP              : "
            f"{best['tp_atr']:.2f} ATR"
        )
        print(
            f"SL              : "
            f"{best['sl_atr']:.2f} ATR"
        )
        print(
            f"Horizon         : "
            f"{best['horizon_min']} min"
        )

        print("")
        print("DEV")
        print(
            f"Trades={best['DEV_trades']} | "
            f"AvgR={best['DEV_avg_R']:.4f} | "
            f"PF={best['DEV_PF']:.3f}"
        )

        print("VAL")
        print(
            f"Trades={best['VAL_trades']} | "
            f"AvgR={best['VAL_avg_R']:.4f} | "
            f"PF={best['VAL_PF']:.3f}"
        )

        print("TEST - controllo fuori campione")
        print(
            f"Trades={best['TEST_trades']} | "
            f"AvgR={best['TEST_avg_R']:.4f} | "
            f"PF={best['TEST_PF']:.3f}"
        )

    else:

        print("")
        print(
            "Nessuna configurazione ha superato "
            "contemporaneamente i filtri di robustezza DEV + VAL."
        )

        print(
            "Questo NON significa che il sistema sia inutilizzabile."
        )

        print(
            "Significa che dobbiamo allargare o modificare "
            "la ricerca prima di trasformarla in bot live."
        )

    # --------------------------------------------------------
    # FILE
    # --------------------------------------------------------

    print("")
    print("=" * 78)
    print("FILE GENERATI")
    print("=" * 78)

    for filename in sorted(
        os.listdir(OUTPUT_DIR)
    ):

        print(
            f"- {OUTPUT_DIR}/{filename}"
        )

    print("")
    print("=" * 78)
    print("BACKTEST V9 COMPLETATO")
    print("=" * 78)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print("")
        print("=" * 78)
        print("ERRORE BACKTEST V9")
        print("=" * 78)

        print(
            repr(e)
        )

        traceback.print_exc()

        raise
