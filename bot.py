
import os
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ============================================================
# V10 — AGGRESSIVE TREND ANTICIPATION
# PAPER / DEMO ONLY
# ============================================================

VERSION = "V10 — AGGRESSIVE TREND ANTICIPATION"

SYMBOL = "XAU/USD"
INTERVAL = "5min"
OUTPUTSIZE = 500

API_KEY = os.getenv("TWELVE_DATA_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurato")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN non configurato")

if not TELEGRAM_CHAT_ID:
    raise RuntimeError("TELEGRAM_CHAT_ID non configurato")

# Parametri sperimentali
BREAKOUT_LOOKBACK = 12
COOLDOWN_BARS = 3

SL_ATR = 0.80
TP_ATR = 1.50

MIN_SCORE = 6
MIN_SCORE_DIFFERENCE = 2

API_URL = "https://api.twelvedata.com/time_series"
TELEGRAM_URL = (
    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
)

# ============================================================
# DOWNLOAD DATI
# ============================================================

def download_data():
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": OUTPUTSIZE,
        "timezone": "UTC",
        "apikey": API_KEY,
        "format": "JSON",
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") == "error" or "values" not in data:
        message = data.get("message", str(data))
        raise RuntimeError(f"Errore Twelve Data: {message}")

    df = pd.DataFrame(data["values"])

    required = ["datetime", "open", "high", "low", "close"]
    missing = set(required) - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Colonne mancanti nei dati: {sorted(missing)}"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # Esclude la candela 5m non ancora chiusa.
    now = pd.Timestamp.now(tz="UTC")
    df = df[
        df["datetime"] + pd.Timedelta(minutes=5) <= now
    ].copy()

    df = df.reset_index(drop=True)

    if len(df) < 150:
        raise RuntimeError(
            f"Dati insufficienti: solo {len(df)} candele chiuse"
        )

    return df


# ============================================================
# INDICATORI BASE
# ============================================================

def calculate_base_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Medie mobili
    df["ema9"] = close.ewm(
        span=9, adjust=False
    ).mean()

    df["ema20"] = close.ewm(
        span=20, adjust=False
    ).mean()

    df["ema50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    df["ema100"] = close.ewm(
        span=100, adjust=False
    ).mean()

    # ATR 14
    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    df["atr_avg50"] = df["atr"].rolling(
        50, min_periods=30
    ).mean()

    df["atr_ratio"] = df["atr"] / df["atr_avg50"]

    # RSI 14
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

    df.loc[
        (avg_loss == 0) & (avg_gain > 0), "rsi"
    ] = 100

    df.loc[
        (avg_loss == 0) & (avg_gain == 0), "rsi"
    ] = 50

    # MACD
    ema12 = close.ewm(
        span=12, adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26, adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    # Momentum a 3 e 6 candele, normalizzato per ATR
    df["mom3"] = (
        close - close.shift(3)
    ) / df["atr"]

    df["mom6"] = (
        close - close.shift(6)
    ) / df["atr"]

    # Pendenza EMA20
    df["ema20_slope"] = (
        df["ema20"] - df["ema20"].shift(3)
    ) / df["atr"]

    # Qualità della candela
    candle_range = (high - low).replace(0, np.nan)

    df["body_ratio"] = (
        (close - df["open"]).abs() / candle_range
    )

    df["close_location"] = (
        (close - low) / candle_range
    )

    # Livelli di breakout: solo candele precedenti
    df["breakout_high"] = (
        high.shift(1)
        .rolling(BREAKOUT_LOOKBACK)
        .max()
    )

    df["breakout_low"] = (
        low.shift(1)
        .rolling(BREAKOUT_LOOKBACK)
        .min()
    )

    # Ampiezza del gap fra EMA
    df["ema_gap_atr"] = (
        (df["ema20"] - df["ema50"]).abs()
        / df["atr"]
    )

    return df


# ============================================================
# TREND MULTI-TIMEFRAME
# ============================================================

def build_higher_timeframe(df, minutes):
    """
    Costruisce candele complete da 15m o 60m.
    available_at impedisce di usare una candela HTF
    prima della sua chiusura.
    """

    temp = df.set_index("datetime")[
        ["open", "high", "low", "close"]
    ].copy()

    rule = f"{minutes}min"

    tf = temp.resample(
        rule,
        label="left",
        closed="left",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    })

    # Conteggio separato: datetime è l'indice, non una colonna.
    tf["candle_count"] = (
        temp["close"]
        .resample(rule, label="left", closed="left")
        .count()
    )

    expected_count = minutes // 5

    tf = tf[
        tf["candle_count"] == expected_count
    ].copy()

    tf = tf.dropna(
        subset=["open", "high", "low", "close"]
    )

    tf["ema20"] = tf["close"].ewm(
        span=20, adjust=False
    ).mean()

    tf["ema50"] = tf["close"].ewm(
        span=50, adjust=False
    ).mean()

    tf["ema20_previous"] = tf["ema20"].shift(3)

    tf["trend"] = np.select(
        [
            (tf["ema20"] > tf["ema50"])
            & (tf["ema20"] > tf["ema20_previous"]),

            (tf["ema20"] < tf["ema50"])
            & (tf["ema20"] < tf["ema20_previous"]),
        ],
        [1, -1],
        default=0,
    )

    # Una candela HTF è utilizzabile solo dopo la chiusura.
    tf["available_at"] = (
        tf.index + pd.Timedelta(minutes=minutes)
    )

    return tf.reset_index(drop=True)[
        ["available_at", "trend"]
    ]


def attach_higher_timeframes(df):
    df = df.copy()

    df["entry_time"] = (
        df["datetime"] + pd.Timedelta(minutes=5)
    )

    tf15 = build_higher_timeframe(df, 15)
    tf60 = build_higher_timeframe(df, 60)

    # Merge backward: mai usare trend HTF futuro.
    df = pd.merge_asof(
        df.sort_values("entry_time"),
        tf15.sort_values("available_at"),
        left_on="entry_time",
        right_on="available_at",
        direction="backward",
    )

    df = df.rename(columns={"trend": "trend15"})
    df = df.drop(columns=["available_at"])

    df = pd.merge_asof(
        df.sort_values("entry_time"),
        tf60.sort_values("available_at"),
        left_on="entry_time",
        right_on="available_at",
        direction="backward",
    )

    df = df.rename(columns={"trend": "trend60"})
    df = df.drop(columns=["available_at"])

    df["trend15"] = df["trend15"].fillna(0)
    df["trend60"] = df["trend60"].fillna(0)

    return df.sort_values("datetime").reset_index(drop=True)


def calculate_indicators(df):
    df = calculate_base_indicators(df)
    df = attach_higher_timeframes(df)
    return df


# ============================================================
# PUNTEGGIO DIREZIONALE
# ============================================================

def score_row(row):
    """
    Punteggio euristico, NON probabilità statistica.
    Permette di rilevare anche accelerazioni iniziali,
    senza richiedere sempre tutte le EMA allineate.
    """

    values = [
        "close", "atr", "rsi", "ema9", "ema20",
        "ema50", "ema100", "macd_hist", "mom3",
        "mom6", "ema20_slope", "body_ratio",
        "close_location", "breakout_high",
        "breakout_low", "atr_ratio", "trend15",
        "trend60",
    ]

    if any(
        pd.isna(row.get(key, np.nan))
        for key in values
    ):
        return 0, 0

    if row["atr"] <= 0:
        return 0, 0

    buy = 0
    sell = 0

    # Struttura e velocità delle EMA
    buy += int(row["ema9"] > row["ema20"])
    buy += int(row["ema20"] > row["ema50"])
    buy += int(row["ema50"] > row["ema100"])
    buy += int(row["ema20_slope"] > 0.02)

    sell += int(row["ema9"] < row["ema20"])
    sell += int(row["ema20"] < row["ema50"])
    sell += int(row["ema50"] < row["ema100"])
    sell += int(row["ema20_slope"] < -0.02)

    # Momentum: cerca l'accelerazione prima che sia estrema
    buy += int(row["mom3"] > 0.08)
    buy += int(row["mom6"] > 0.10)

    sell += int(row["mom3"] < -0.08)
    sell += int(row["mom6"] < -0.10)

    # MACD e sua accelerazione
    buy += int(row["macd_hist"] > 0)
    buy += int(row["macd_hist"] > row["macd_hist_previous"])

    sell += int(row["macd_hist"] < 0)
    sell += int(row["macd_hist"] < row["macd_hist_previous"])

    # RSI: evita di richiedere un RSI già molto estremo
    buy += int(48 <= row["rsi"] <= 72)
    sell += int(28 <= row["rsi"] <= 52)

    # Trend superiori: conferme, non blocchi assoluti
    buy += int(row["trend15"] == 1)
    buy += int(row["trend60"] == 1)

    sell += int(row["trend15"] == -1)
    sell += int(row["trend60"] == -1)

    # Breakout e vicinanza al livello di rottura
    buy += int(
        row["close"] >= row["breakout_high"]
        or (
            row["breakout_high"] - row["close"]
        ) <= 0.15 * row["atr"]
    )

    sell += int(
        row["close"] <= row["breakout_low"]
        or (
            row["close"] - row["breakout_low"]
        ) <= 0.15 * row["atr"]
    )

    # Qualità della candela e posizione della chiusura
    buy += int(
        row["body_ratio"] >= 0.35
        and row["close_location"] >= 0.65
    )

    sell += int(
        row["body_ratio"] >= 0.35
        and row["close_location"] <= 0.35
    )

    # Volatilità sufficiente, ma non necessariamente elevata
    buy += int(row["atr_ratio"] >= 0.80)
    sell += int(row["atr_ratio"] >= 0.80)

    return buy, sell


def calculate_scores(df):
    df = df.copy()

    df["macd_hist_previous"] = df["macd_hist"].shift(1)

    scores = df.apply(
        score_row,
        axis=1,
        result_type="expand",
    )

    df["buy_score"] = scores[0].astype(int)
    df["sell_score"] = scores[1].astype(int)

    return df


def get_signal(row):
    buy = int(row["buy_score"])
    sell = int(row["sell_score"])

    if (
        buy >= MIN_SCORE
        and buy - sell >= MIN_SCORE_DIFFERENCE
    ):
        return "BUY", buy, sell

    if (
        sell >= MIN_SCORE
        and sell - buy >= MIN_SCORE_DIFFERENCE
    ):
        return "SELL", buy, sell

    return None, buy, sell


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    response = requests.post(
        TELEGRAM_URL,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
        },
        timeout=20,
    )

    response.raise_for_status()
    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(
            f"Telegram ha rifiutato il messaggio: {result}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 64)
    print(VERSION)
    print("=" * 64)

    df = download_data()

    print(f"Candele chiuse: {len(df)}")
    print(f"Ultima candela: {df.iloc[-1]['datetime']}")

    df = calculate_indicators(df)
    df = calculate_scores(df)

    row = df.iloc[-1]

    signal, buy_score, sell_score = get_signal(row)

    print(f"Prezzo: {row['close']:.2f}")
    print(f"RSI: {row['rsi']:.2f}")
    print(f"ATR: {row['atr']:.3f}")
    print(f"Momentum 3: {row['mom3']:.3f} ATR")
    print(f"Momentum 6: {row['mom6']:.3f} ATR")
    print(f"Trend 15m: {int(row['trend15'])}")
    print(f"Trend 1h: {int(row['trend60'])}")
    print(f"BUY score: {buy_score}")
    print(f"SELL score: {sell_score}")

    # Evita segnali consecutivi della stessa direzione
    # nelle tre candele precedenti.
    recent = df.iloc[-(COOLDOWN_BARS + 1):-1]

    previous_signals = [
        get_signal(previous)[0]
        for _, previous in recent.iterrows()
    ]

    if signal and signal in previous_signals:
        print(
            f"Segnale {signal} ignorato: "
            "condizione già presente nelle candele recenti."
        )
        return

    if signal is None:
        print("Nessun segnale: punteggio insufficiente.")
        return

    entry = float(row["close"])
    atr = float(row["atr"])

    if signal == "BUY":
        sl = entry - SL_ATR * atr
        tp = entry + TP_ATR * atr
        score = buy_score
    else:
        sl = entry + SL_ATR * atr
        tp = entry - TP_ATR * atr
        score = sell_score

    risk = abs(entry - sl)
    reward = abs(tp - entry)

    message = (
        f"📊 {VERSION}\n"
        f"🧪 MODALITÀ PAPER/DEMO — NESSUN ORDINE REALE\n\n"
        f"Strumento: {SYMBOL}\n"
        f"Segnale: {signal}\n"
        f"Prezzo teorico: {entry:.2f}\n"
        f"Stop Loss: {sl:.2f}\n"
        f"Take Profit: {tp:.2f}\n\n"
        f"Punteggio direzionale: {score}\n"
        f"BUY score: {buy_score}\n"
        f"SELL score: {sell_score}\n"
        f"RSI: {row['rsi']:.2f}\n"
        f"ATR: {atr:.3f}\n"
        f"Momentum 3: {row['mom3']:.2f} ATR\n"
        f"Momentum 6: {row['mom6']:.2f} ATR\n"
        f"Trend 15m: {int(row['trend15'])}\n"
        f"Trend 1h: {int(row['trend60'])}\n"
        f"Rapporto teorico rischio/rendimento: "
        f"1:{reward / risk:.2f}\n"
        f"Candela UTC: {row['datetime']}\n\n"
        f"Il punteggio è euristico e non rappresenta "
        f"una probabilità di successo."
    )

    send_telegram(message)
    print(f"Segnale {signal} inviato a Telegram.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRORE: {exc}")
        traceback.print_exc()
        raise
