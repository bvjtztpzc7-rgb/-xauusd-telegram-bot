
import os
import traceback
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ============================================================
# V10 — AGGRESSIVE BUT PRUDENT
# PAPER / DEMO ONLY — NESSUN ORDINE REALE
# ============================================================

VERSION = "V10 — AGGRESSIVE BUT PRUDENT"

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

# Strategia V10
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

COOLDOWN_MIN = 60
COOLDOWN_BARS = COOLDOWN_MIN // 5

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
        "apikey": API_KEY,
        "timezone": "UTC",
        "order": "ASC",
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
        raise RuntimeError(
            f"Errore Twelve Data: {data.get('message', data)}"
        )

    df = pd.DataFrame(data["values"])

    required = ["datetime", "open", "high", "low", "close"]
    if df.empty or not set(required).issubset(df.columns):
        raise RuntimeError("Dati Twelve Data mancanti o non validi.")

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # Non usare la candela 5m finché non è chiusa.
    now = pd.Timestamp.now(tz="UTC")
    df = df[
        df["datetime"] + pd.Timedelta(minutes=5) <= now
    ].reset_index(drop=True)

    if len(df) < 200:
        raise RuntimeError(
            f"Dati insufficienti: {len(df)} candele chiuse."
        )

    print(f"Candele chiuse: {len(df)}")
    print(f"Prima candela: {df.iloc[0]['datetime']}")
    print(f"Ultima candela: {df.iloc[-1]['datetime']}")

    return df


# ============================================================
# INDICATORI 5 MINUTI
# ============================================================

def calculate_base_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema20"] = close.ewm(
        span=EMA_FAST, adjust=False
    ).mean()

    df["ema50"] = close.ewm(
        span=EMA_MID, adjust=False
    ).mean()

    df["ema100"] = close.ewm(
        span=EMA_SLOW, adjust=False
    ).mean()

    # ATR Wilder
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
        alpha=1 / ATR_PERIOD,
        adjust=False,
    ).mean()

    df["atr_avg50"] = df["atr"].rolling(
        ATR_AVG_PERIOD
    ).mean()

    # Momentum su 6 candele, in unità di prezzo.
    df["mom6"] = close - close.shift(6)

    # RSI Wilder
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

    df.loc[
        (avg_loss == 0) & (avg_gain > 0), "rsi"
    ] = 100

    df.loc[
        (avg_loss == 0) & (avg_gain == 0), "rsi"
    ] = 50

    # Forza della candela
    candle_range = (high - low).replace(0, np.nan)

    df["body_ratio"] = (
        (close - df["open"]).abs() / candle_range
    )

    df["close_position"] = (
        (close - low) / candle_range
    )

    # Livelli di breakout: esclusa la candela corrente.
    df["previous_high"] = (
        high.rolling(BREAKOUT_LOOKBACK)
        .max()
        .shift(1)
    )

    df["previous_low"] = (
        low.rolling(BREAKOUT_LOOKBACK)
        .min()
        .shift(1)
    )

    # Distanza firmata tra EMA20 ed EMA50, normalizzata ATR.
    df["ema_gap_atr"] = (
        (df["ema20"] - df["ema50"])
        / df["atr"].replace(0, np.nan)
    )

    return df


# ============================================================
# TREND 15 MINUTI SENZA LOOKAHEAD
# ============================================================

def attach_15m_trend(df):
    df = df.copy()

    temp = df.set_index("datetime")[
        ["open", "high", "low", "close"]
    ]

    tf15 = temp.resample(
        "15min",
        label="left",
        closed="left",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    })

    # Una candela 15m è valida solo se contiene tutte
    # e tre le candele da 5m attese.
    tf15["count"] = temp["close"].resample(
        "15min",
        label="left",
        closed="left",
    ).count()

    tf15 = tf15[
        (tf15["count"] == 3)
    ].dropna(
        subset=["open", "high", "low", "close"]
    ).copy()

    tf15["ema20_15"] = tf15["close"].ewm(
        span=20, adjust=False
    ).mean()

    tf15["ema50_15"] = tf15["close"].ewm(
        span=50, adjust=False
    ).mean()

    tf15["trend15"] = np.select(
        [
            tf15["ema20_15"] > tf15["ema50_15"],
            tf15["ema20_15"] < tf15["ema50_15"],
        ],
        ["BULLISH", "BEARISH"],
        default="NEUTRAL",
    )

    # Disponibile soltanto dopo la chiusura della candela 15m.
    tf15["available_at"] = (
        tf15.index + pd.Timedelta(minutes=15)
    )

    tf15 = tf15.reset_index(drop=True)[
        ["available_at", "trend15"]
    ].sort_values("available_at")

    # Momento in cui la candela 5m risulta chiusa.
    df["entry_time"] = (
        df["datetime"] + pd.Timedelta(minutes=5)
    )

    df = pd.merge_asof(
        df.sort_values("entry_time"),
        tf15,
        left_on="entry_time",
        right_on="available_at",
        direction="backward",
    )

    df = df.drop(
        columns=["available_at"],
        errors="ignore",
    )

    return df.sort_values("datetime").reset_index(drop=True)


def calculate_indicators(df):
    df = calculate_base_indicators(df)
    df = attach_15m_trend(df)
    return df


# ============================================================
# GENERAZIONE SEGNALI V10
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
        "trend15",
    ]

    for col in required:
        if pd.isna(row.get(col, np.nan)):
            return None

    if row["atr"] <= 0 or row["atr_avg50"] <= 0:
        return None

    # BUY
    buy = (
        row["ema20"] > row["ema50"] > row["ema100"]
        and row["ema_gap_atr"] >= EMA_GAP_ATR_MIN
        and row["trend15"] == "BULLISH"
        and row["close"] > row["previous_high"]
        and row["body_ratio"] >= BODY_RATIO_MIN
        and row["close_position"] >= 1 - CLOSE_LOCATION_MAX
        and row["mom6"] >= MOMENTUM_ATR_MIN * row["atr"]
        and row["atr"] > row["atr_avg50"]
        and BUY_RSI_MIN <= row["rsi"] <= BUY_RSI_MAX
    )

    if buy:
        return "BUY"

    # SELL
    sell = (
        row["ema20"] < row["ema50"] < row["ema100"]
        and row["ema_gap_atr"] <= -EMA_GAP_ATR_MIN
        and row["trend15"] == "BEARISH"
        and row["close"] < row["previous_low"]
        and row["body_ratio"] >= BODY_RATIO_MIN
        and row["close_position"] <= CLOSE_LOCATION_MAX
        and row["mom6"] <= -MOMENTUM_ATR_MIN * row["atr"]
        and row["atr"] > row["atr_avg50"]
        and SELL_RSI_MIN <= row["rsi"] <= SELL_RSI_MAX
    )

    if sell:
        return "SELL"

    return None


# ============================================================
# DIAGNOSTICA FILTRI V10
# Non modifica la generazione dei segnali.
# ============================================================

def diagnose_filters(row):
    required = [
        "ema20", "ema50", "ema100", "atr", "atr_avg50",
        "mom6", "rsi", "body_ratio", "close_position",
        "previous_high", "previous_low", "ema_gap_atr",
        "trend15",
    ]

    print("\n" + "=" * 58)
    print("DIAGNOSTICA FILTRI V10")
    print("=" * 58)

    missing = [
        col for col in required
        if pd.isna(row.get(col, np.nan))
    ]

    if missing:
        print("DATI MANCANTI:", ", ".join(missing))
        print("Diagnostica incompleta: alcuni indicatori non disponibili.")
        return

    # Condizioni BUY
    buy_checks = {
        "EMA20 > EMA50 > EMA100":
            row["ema20"] > row["ema50"] > row["ema100"],

        "Distanza EMA sufficiente":
            row["ema_gap_atr"] >= EMA_GAP_ATR_MIN,

        "Trend 15m rialzista":
            row["trend15"] == "BULLISH",

        "Breakout sopra il massimo precedente":
            row["close"] > row["previous_high"],

        "Corpo candela >= 60%":
            row["body_ratio"] >= BODY_RATIO_MIN,

        "Chiusura nel 20% superiore":
            row["close_position"] >= 1 - CLOSE_LOCATION_MAX,

        "Momentum BUY sufficiente":
            row["mom6"] >= MOMENTUM_ATR_MIN * row["atr"],

        "ATR sopra media 50":
            row["atr"] > row["atr_avg50"],

        "RSI BUY tra 52 e 68":
            BUY_RSI_MIN <= row["rsi"] <= BUY_RSI_MAX,
    }

    # Condizioni SELL
    sell_checks = {
        "EMA20 < EMA50 < EMA100":
            row["ema20"] < row["ema50"] < row["ema100"],

        "Distanza EMA sufficiente":
            row["ema_gap_atr"] <= -EMA_GAP_ATR_MIN,

        "Trend 15m ribassista":
            row["trend15"] == "BEARISH",

        "Breakout sotto il minimo precedente":
            row["close"] < row["previous_low"],

        "Corpo candela >= 60%":
            row["body_ratio"] >= BODY_RATIO_MIN,

        "Chiusura nel 20% inferiore":
            row["close_position"] <= CLOSE_LOCATION_MAX,

        "Momentum SELL sufficiente":
            row["mom6"] <= -MOMENTUM_ATR_MIN * row["atr"],

        "ATR sopra media 50":
            row["atr"] > row["atr_avg50"],

        "RSI SELL tra 32 e 48":
            SELL_RSI_MIN <= row["rsi"] <= SELL_RSI_MAX,
    }

    for direction, checks in [
        ("BUY", buy_checks),
        ("SELL", sell_checks),
    ]:
        passed = sum(checks.values())

        print(f"\n--- {direction}: {passed}/{len(checks)} filtri superati ---")

        for name, ok in checks.items():
            print(f"[{'OK' if ok else 'NO'}] {name}")

    print("=" * 58)



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
    print("MODALITÀ PAPER/DEMO — NESSUN ORDINE REALE")
    print("=" * 64)

    df = download_data()
    df = calculate_indicators(df)

    row = df.iloc[-1]

diagnose_filters(row)

signal = generate_signal(row)

    print(f"Ultima candela UTC: {row['datetime']}")
    print(f"Prezzo di riferimento: {row['close']:.2f}")
    print(f"RSI: {row['rsi']:.2f}")
    print(f"ATR: {row['atr']:.3f}")
    print(f"Momentum 6: {row['mom6']:.3f}")
    print(f"Trend 15m: {row['trend15']}")

    if signal is None:
        print("Nessun segnale: condizioni V10 non soddisfatte.")
        return

    # Cooldown direzionale: evita un nuovo segnale BUY o SELL
    # se la stessa direzione si è verificata nelle 12 candele
    # precedenti (60 minuti).
    recent = df.iloc[-(COOLDOWN_BARS + 1):-1]

    for _, previous in recent.iterrows():
        if generate_signal(previous) == signal:
            print(
                f"Segnale {signal} ignorato: "
                f"stessa direzione rilevata negli ultimi "
                f"{COOLDOWN_MIN} minuti."
            )
            return

    entry = float(row["close"])
    atr = float(row["atr"])

    if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
        raise RuntimeError("Prezzo o ATR non validi.")

    if signal == "BUY":
        sl = entry - SL_ATR * atr
        tp = entry + TP_ATR * atr
    else:
        sl = entry + SL_ATR * atr
        tp = entry - TP_ATR * atr

    risk = abs(entry - sl)
    reward = abs(tp - entry)

    message = (
        f"📊 {VERSION}\n"
        f"🧪 PAPER/DEMO — NESSUN ORDINE REALE\n\n"
        f"Strumento: {SYMBOL}\n"
        f"Segnale: {signal}\n"
        f"Prezzo teorico: {entry:.2f}\n"
        f"Stop Loss: {sl:.2f}\n"
        f"Take Profit: {tp:.2f}\n\n"
        f"RSI: {row['rsi']:.2f}\n"
        f"ATR: {atr:.3f}\n"
        f"Momentum 6: {row['mom6']:.3f}\n"
        f"Trend 15m: {row['trend15']}\n"
        f"Rapporto teorico rischio/rendimento: "
        f"1:{reward / risk:.2f}\n"
        f"Candela UTC: {row['datetime']}\n\n"
        f"Segnale sperimentale: non è una garanzia "
        f"di risultato. Verificare sempre i dati."
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
