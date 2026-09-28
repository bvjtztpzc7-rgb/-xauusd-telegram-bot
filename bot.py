import os
import time
import requests
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("TWELVE_DATA_API_KEY")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN non configurato")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurata")

if not CHAT_ID:
    raise RuntimeError("TELEGRAM_CHAT_ID non configurato")


# ============================================================
# TELEGRAM
# ============================================================

def invia_telegram(testo):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    dati = {
        "chat_id": CHAT_ID,
        "text": testo
    }

    risposta = requests.post(url, data=dati, timeout=15)

    if risposta.ok:
        print("Telegram: messaggio inviato")
        return True

    print("Errore Telegram:", risposta.text)
    return False


# ============================================================
# DATI XAU/USD
# ============================================================

def scarica_dati():
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": "XAU/USD",
        "interval": "5min",
        "outputsize": 500,
        "apikey": API_KEY
    }

    risposta = requests.get(url, params=params, timeout=20)
    dati = risposta.json()

    if "values" not in dati:
        print("Errore Twelve Data:")
        print(dati)
        return None

    df = pd.DataFrame(dati["values"])

    df["datetime"] = pd.to_datetime(df["datetime"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col])

    df = df.sort_values("datetime").reset_index(drop=True)

    return df


# ============================================================
# INDICATORI
# ============================================================

def calcola_indicatori(df):

    df = df.copy()

    # EMA
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    return df


# ============================================================
# TREND 15 MINUTI
# ============================================================

def calcola_trend_15m(df):

    df = df.copy()

    df_15m = (
        df.set_index("datetime")
        .resample("15min")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        })
        .dropna()
        .reset_index()
    )

    df_15m["EMA20_15"] = (
        df_15m["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df_15m["EMA50_15"] = (
        df_15m["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    # Ultima candela 15m completamente chiusa
    ultima_15m = df_15m.iloc[-2]

    if ultima_15m["EMA20_15"] > ultima_15m["EMA50_15"]:
        trend = "BULLISH"

    elif ultima_15m["EMA20_15"] < ultima_15m["EMA50_15"]:
        trend = "BEARISH"

    else:
        trend = "NEUTRAL"

    return trend, ultima_15m["datetime"]


# ============================================================
# ANALISI
# ============================================================

def analizza_xauusd():

    df = scarica_dati()

    if df is None:
        return None

    if len(df) < 100:
        print("Dati insufficienti.")
        return None

    df = calcola_indicatori(df)

    # Ultima candela 5m completamente chiusa
    candela = df.iloc[-2]

    trend_15m, datetime_15m = calcola_trend_15m(df)

    prezzo = candela["close"]
    rsi = candela["RSI"]
    atr = candela["ATR"]

    ema20 = candela["EMA20"]
    ema50 = candela["EMA50"]

    macd = candela["MACD"]
    macd_signal = candela["MACD_signal"]

    segnale = "NONE"

    # BUY
    if (
        ema20 > ema50
        and macd > macd_signal
        and 30 < rsi < 65
        and trend_15m == "BULLISH"
    ):
        segnale = "BUY"

    # SELL
    elif (
        ema20 < ema50
        and macd < macd_signal
        and 30 < rsi < 65
        and trend_15m == "BEARISH"
    ):
        segnale = "SELL"

    sl = None
    tp = None

    if segnale == "BUY":

        sl = prezzo - (1.5 * atr)
        tp = prezzo + (2.0 * atr)

    elif segnale == "SELL":

        sl = prezzo + (1.5 * atr)
        tp = prezzo - (2.0 * atr)

    return {
        "signal": segnale,
        "datetime": candela["datetime"],
        "price": prezzo,
        "sl": sl,
        "tp": tp,
        "rsi": rsi,
        "atr": atr,
        "macd": macd,
        "macd_signal": macd_signal,
        "ema20": ema20,
        "ema50": ema50,
        "trend_15m": trend_15m,
        "trend_datetime": datetime_15m
    }


# ============================================================
# MESSAGGIO TELEGRAM
# ============================================================

def crea_messaggio(r):

    if r["signal"] == "BUY":
        emoji = "🟢"
    else:
        emoji = "🔴"

    return (
        f"{emoji} XAU/USD — {r['signal']}\n\n"
        f"⏰ Candela: {r['datetime']}\n"
        f"💰 Entry: {r['price']:.2f}\n"
        f"🛑 SL: {r['sl']:.2f}\n"
        f"🎯 TP: {r['tp']:.2f}\n\n"
        f"RSI: {r['rsi']:.2f}\n"
        f"ATR: {r['atr']:.2f}\n"
        f"Trend 15m: {r['trend_15m']}\n"
    )


# ============================================================
# LOOP PRINCIPALE
# ============================================================

ultima_candela = None

print("========================================")
print("🤖 XAU/USD BOT — SERVER")
print("========================================")
print("Bot avviato.")
print("Controllo ogni 20 secondi.")
print("Invio Telegram: SOLO BUY / SELL")
print()

while True:

    try:

        df = scarica_dati()

        if df is None or len(df) < 100:
            print("⚠️ Dati insufficienti.")
            time.sleep(20)
            continue

        candela_attuale = df.iloc[-2]["datetime"]

        # Nessuna nuova candela
        if candela_attuale == ultima_candela:
            time.sleep(20)
            continue

        # Nuova candela
        ultima_candela = candela_attuale

        print("----------------------------------------")
        print(f"🕐 Nuova candela: {candela_attuale}")
        print("🔎 Analisi XAU/USD...")

        risultato = analizza_xauusd()

        if risultato is None:
            time.sleep(20)
            continue

        segnale = risultato["signal"]

        print(f"📊 Segnale: {segnale}")

        if segnale in ["BUY", "SELL"]:

            messaggio = crea_messaggio(risultato)

            if invia_telegram(messaggio):
                print("📨 ✅ Segnale inviato a Telegram!")

        else:

            print("⏳ Nessun segnale.")

        time.sleep(20)

    except Exception as e:

        print("❌ ERRORE:")
        print(e)

        print("🔄 Nuovo tentativo tra 20 secondi...")
        time.sleep(20)
