
import os
import traceback
import requests
import numpy as np
import pandas as pd


# ============================================================
# V10.1 — AGGRESSIVE BUT PRUDENT
# Breakout più frequenti, filtri di rischio mantenuti
# ============================================================

SYMBOL = "XAU/USD"
INTERVAL = "5min"
DATA_SIZE = 500

API_KEY = os.getenv("TWELVE_DATA_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurato")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN non configurato")
if not TELEGRAM_CHAT_ID:
    raise RuntimeError("TELEGRAM_CHAT_ID non configurato")


# PARAMETRI STRATEGIA

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 100

BREAKOUT_LOOKBACK = 12
BODY_RATIO_MIN = 0.40
CLOSE_LOCATION_MAX = 0.20

MOMENTUM_ATR_MIN = 0.10

ATR_PERIOD = 14
ATR_AVG_PERIOD = 50
ATR_REGIME_MIN = 0.85

RSI_PERIOD = 14
BUY_RSI_MIN, BUY_RSI_MAX = 50, 70
SELL_RSI_MIN, SELL_RSI_MAX = 30, 50

EMA_GAP_ATR_MIN = 0.10

SL_ATR = 0.80
TP_ATR = 1.50

COOLDOWN_MIN = 15


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
        timeout=20
    )
    response.raise_for_status()

    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(f"Errore Telegram: {result}")


# ============================================================
# DATI
# ============================================================

def get_data():
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": DATA_SIZE,
        "apikey": API_KEY,
        "timezone": "UTC",
        "order": "ASC"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "values" not in data:
        raise RuntimeError(f"Errore Twelve Data: {data}")

    df = pd.DataFrame(data["values"])
    if df.empty:
        raise RuntimeError("Twelve Data ha restituito dati vuoti.")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["open", "high", "low", "close"])
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    # Escludi la candela 5m non ancora chiusa.
    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df["datetime"] + pd.Timedelta(minutes=5) <= now_utc]
    df = df.reset_index(drop=True)

    if len(df) < 200:
        raise RuntimeError(
            f"Candele chiuse insufficienti: {len(df)}"
        )

    return df


# ============================================================
# INDICATORI
# ============================================================

def calculate_indicators(df):
    df = df.copy()

    # EMA
    for period, name in [
        (EMA_FAST, "ema20"),
        (EMA_MID, "ema50"),
        (EMA_SLOW, "ema100")
    ]:
        df[name] = df["close"].ewm(
            span=period, adjust=False
        ).mean()

    # ATR
    previous_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / ATR_PERIOD, adjust=False
    ).mean()

    df["atr_avg50"] = df["atr"].rolling(
        ATR_AVG_PERIOD
    ).mean()

    # Momentum a 6 candele
    df["mom6"] = df["close"] - df["close"].shift(6)

    # RSI di Wilder, con gestione dei casi limite
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100
    df.loc[(avg_gain == 0) & (avg_loss > 0), "rsi"] = 0
    df.loc[(avg_gain == 0) & (avg_loss == 0), "rsi"] = 50

    # Forza della candela e posizione della chiusura
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)

    df["body_ratio"] = (
        (df["close"] - df["open"]).abs() / candle_range
    )

    df["close_position"] = (
        (df["close"] - df["low"]) / candle_range
    )

    # Livelli breakout: esclude la candela corrente
    df["previous_high"] = (
        df["high"].rolling(BREAKOUT_LOOKBACK)
        .max().shift(1)
    )

    df["previous_low"] = (
        df["low"].rolling(BREAKOUT_LOOKBACK)
        .min().shift(1)
    )

    # Distanza EMA normalizzata per ATR
    df["ema_gap_atr"] = (
        (df["ema20"] - df["ema50"]) /
        df["atr"].replace(0, np.nan)
    )

    # Ora di chiusura della candela operativa
    
temp = df.set_index("datetime").copy()

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

# Conta quante candele da 5 minuti compongono
# ogni candela da 15 minuti.
tf15["candle_count"] = (
    temp["close"]
    .resample("15min", label="left", closed="left")
    .count()
)


    tf15 = tf15[tf15["datetime"] == 3].copy()

    tf15["ema20_15"] = tf15["close"].ewm(
        span=20, adjust=False
    ).mean()

    tf15["ema50_15"] = tf15["close"].ewm(
        span=50, adjust=False
    ).mean()

    tf15["trend15"] = np.select(
        [
            tf15["ema20_15"] > tf15["ema50_15"],
            tf15["ema20_15"] < tf15["ema50_15"]
        ],
        ["BULLISH", "BEARISH"],
        default="NEUTRAL"
    )

    # Ogni candela 15m è disponibile solo alla chiusura.
    tf15["available_at"] = (
        tf15.index + pd.Timedelta(minutes=15)
    )

    tf15 = tf15[["available_at", "trend15"]].sort_values(
        "available_at"
    )

    df = pd.merge_asof(
        df.sort_values("entry_time"),
        tf15,
        left_on="entry_time",
        right_on="available_at",
        direction="backward"
    )

    return df.drop(columns=["available_at"], errors="ignore")


# ============================================================
# SEGNALI
# ============================================================

def check_signal(row):
    required = [
        "ema20", "ema50", "ema100", "atr", "atr_avg50",
        "mom6", "rsi", "body_ratio", "close_position",
        "previous_high", "previous_low", "ema_gap_atr",
        "trend15"
    ]

    if any(pd.isna(row[col]) for col in required):
        return None

    # Filtro volatilità meno restrittivo della V10
    if row["atr"] < ATR_REGIME_MIN * row["atr_avg50"]:
        return None

    common_candle = row["body_ratio"] >= BODY_RATIO_MIN

    buy = (
        row["ema20"] > row["ema50"] > row["ema100"]
        and row["ema_gap_atr"] >= EMA_GAP_ATR_MIN
        and row["trend15"] == "BULLISH"
        and row["close"] > row["previous_high"]
        and common_candle
        and row["close_position"] >= 1 - CLOSE_LOCATION_MAX
        and row["mom6"] >= MOMENTUM_ATR_MIN * row["atr"]
        and BUY_RSI_MIN <= row["rsi"] <= BUY_RSI_MAX
    )

    if buy:
        return "BUY"

    sell = (
        row["ema20"] < row["ema50"] < row["ema100"]
        and row["ema_gap_atr"] <= -EMA_GAP_ATR_MIN
        and row["trend15"] == "BEARISH"
        and row["close"] < row["previous_low"]
        and common_candle
        and row["close_position"] <= CLOSE_LOCATION_MAX
        and row["mom6"] <= -MOMENTUM_ATR_MIN * row["atr"]
        and SELL_RSI_MIN <= row["rsi"] <= SELL_RSI_MAX
    )

    if sell:
        return "SELL"

    return None


# ============================================================
# MESSAGGIO TELEGRAM
# ============================================================

def build_message(direction, row):
    entry = float(row["close"])
    atr = float(row["atr"])

    if direction == "BUY":
        sl = entry - SL_ATR * atr
        tp = entry + TP_ATR * atr
        emoji = "🟢"
    else:
        sl = entry + SL_ATR * atr
        tp = entry - TP_ATR * atr
        emoji = "🔴"

    candle_time = row["datetime"].strftime("%d/%m/%Y %H:%M")

    return f"""
🧪 PAPER/DEMO — XAU/USD V10.1

{emoji} {direction}

⏰ Candela: {candle_time} UTC
💰 Entry teorica: {entry:.2f}
🛑 SL teorico: {sl:.2f}
🎯 TP teorico: {tp:.2f}

RSI: {row["rsi"]:.2f}
ATR: {atr:.2f}
MOM6: {row["mom6"]:.2f}
Body Ratio: {row["body_ratio"]:.2f}
Trend 15m: {row["trend15"]}

Breakout: {BREAKOUT_LOOKBACK} candele
SL: {SL_ATR:.2f} ATR
TP: {TP_ATR:.2f} ATR
R:R teorico: 1:{TP_ATR / SL_ATR:.2f}
Cooldown obiettivo: {COOLDOWN_MIN} min

⚠️ Segnale sperimentale PAPER/DEMO.
Nessun ordine reale viene eseguito.
""".strip()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("V10.1 — AGGRESSIVE BUT PRUDENT")
    print("=" * 65)

    df = get_data()
    print(f"Candele chiuse: {len(df)}")
    print(f"Ultima candela: {df['datetime'].iloc[-1]}")

    df = calculate_indicators(df)
    row = df.iloc[-1]

    print("\nCandela analizzata:", row["datetime"])
    for col in [
        "close", "rsi", "atr", "mom6", "ema20",
        "ema50", "ema100", "body_ratio", "close_position"
    ]:
        value = row[col]
        print(f"{col}: {value:.3f}" if pd.notna(value) else f"{col}: n/d")

    print("Trend 15m:", row["trend15"])

    signal = check_signal(row)

    if signal is None:
        print("Nessun segnale V10.1.")
        return

    # Il cooldown è controllato sulle candele precedenti.
    # Tre candele da 5m corrispondono a 15 minuti.
    recent = df.iloc[-4:-1]
    for _, previous_row in recent.iterrows():
        if check_signal(previous_row) is not None:
            print("Cooldown attivo: segnale recente già presente.")
            return

    message = build_message(signal, row)

    print(f"\nSegnale rilevato: {signal}")
    send_telegram(message)
    print("Messaggio Telegram inviato.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
