import os
import requests
import pandas as pd
from zoneinfo import ZoneInfo


# ============================================================
# CONFIGURAZIONE
# ============================================================

TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("TWELVE_DATA_API_KEY")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN non configurato")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurato")

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

    risposta = requests.post(
        url,
        data=dati,
        timeout=15
    )

    if risposta.ok:
        print("✅ Telegram: messaggio inviato")
        return True

    print("❌ Errore Telegram:")
    print(risposta.text)

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

    risposta = requests.get(
        url,
        params=params,
        timeout=20
    )

    risposta.raise_for_status()

    dati = risposta.json()

    if "values" not in dati:

        print("❌ Errore Twelve Data:")
        print(dati)

        return None

    df = pd.DataFrame(dati["values"])

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

    df = (
        df
        .dropna()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# INDICATORI
# ============================================================

def calcola_indicatori(df):

    df = df.copy()

    # EMA 20
    df["EMA20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # EMA 50
    df["EMA50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # MACD

    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["MACD"] = ema12 - ema26

    df["MACD_signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # RSI 14

    delta = df["close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = (
        gain
        .ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        )
        .mean()
    )

    rs = avg_gain / avg_loss

    df["RSI"] = (
        100 -
        (100 / (1 + rs))
    )

    # ATR 14

    high_low = (
        df["high"] -
        df["low"]
    )

    high_close = abs(
        df["high"] -
        df["close"].shift()
    )

    low_close = abs(
        df["low"] -
        df["close"].shift()
    )

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        true_range
        .ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        )
        .mean()
    )

    return df


# ============================================================
# TREND 15 MINUTI
# ============================================================

def calcola_trend_15m(df):

    df_15m = (
        df
        .set_index("datetime")
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
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df_15m["EMA50_15"] = (
        df_15m["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    if len(df_15m) < 3:
        return "NEUTRAL", None

    ultima = df_15m.iloc[-2]

    if ultima["EMA20_15"] > ultima["EMA50_15"]:
        trend = "BULLISH"

    elif ultima["EMA20_15"] < ultima["EMA50_15"]:
        trend = "BEARISH"

    else:
        trend = "NEUTRAL"

    return trend, ultima["datetime"]


# ============================================================
# ANALISI
# ============================================================

def analizza_xauusd():

    df = scarica_dati()

    if df is None:
        return None

    if len(df) < 100:

        print("❌ Dati insufficienti")

        return None

    df = calcola_indicatori(df)

    # Ultima candela completamente chiusa
    candela = df.iloc[-2]

    trend_15m, trend_datetime = (
        calcola_trend_15m(df)
    )

    prezzo = candela["close"]
    ema20 = candela["EMA20"]
    ema50 = candela["EMA50"]
    macd = candela["MACD"]
    macd_signal = candela["MACD_signal"]
    rsi = candela["RSI"]
    atr = candela["ATR"]

    # --------------------------------------------------------
    # CONDIZIONI BUY
    # --------------------------------------------------------

    buy_ema = ema20 > ema50
    buy_macd = macd > macd_signal
    buy_rsi = 30 < rsi < 65
    buy_trend = trend_15m == "BULLISH"

    # --------------------------------------------------------
    # CONDIZIONI SELL
    # --------------------------------------------------------

    sell_ema = ema20 < ema50
    sell_macd = macd < macd_signal
    sell_rsi = 30 < rsi < 65
    sell_trend = trend_15m == "BEARISH"

    segnale = "NONE"

    if (
        buy_ema
        and buy_macd
        and buy_rsi
        and buy_trend
    ):

        segnale = "BUY"

    elif (
        sell_ema
        and sell_macd
        and sell_rsi
        and sell_trend
    ):

        segnale = "SELL"

    # --------------------------------------------------------
    # SL / TP
    # --------------------------------------------------------

    sl = None
    tp = None

    if segnale == "BUY":

        sl = prezzo - (1.5 * atr)
        tp = prezzo + (2.5 * atr)

    elif segnale == "SELL":

        sl = prezzo + (1.5 * atr)
        tp = prezzo - (2.5 * atr)

    # --------------------------------------------------------
    # DIAGNOSTICA
    # --------------------------------------------------------

    print("")
    print("========================================")
    print("📊 ANALISI XAU/USD")
    print("========================================")

    print(f"⏰ Candela: {r['datetime'].astimezone(ZoneInfo('Europe/Rome')).strftime('%d/%m/%Y %H:%M')}\n")
    print(f"💰 Prezzo: {prezzo:.2f}")

    print("")
    print(f"EMA20: {ema20:.4f}")
    print(f"EMA50: {ema50:.4f}")
    print(f"MACD: {macd:.4f}")
    print(f"MACD Signal: {macd_signal:.4f}")
    print(f"RSI: {rsi:.2f}")
    print(f"ATR: {atr:.4f}")

    print("")
    print(f"📈 Trend 15m: {trend_15m}")
    print(f"🕐 Ultimo trend 15m chiuso: {trend_datetime}")

    print("")
    print("🟢 BUY")

    print(
        f"EMA: {'✅' if buy_ema else '❌'}"
    )

    print(
        f"MACD: {'✅' if buy_macd else '❌'}"
    )

    print(
        f"RSI: {'✅' if buy_rsi else '❌'}"
    )

    print(
        f"Trend: {'✅' if buy_trend else '❌'}"
    )

    print("")
    print("🔴 SELL")

    print(
        f"EMA: {'✅' if sell_ema else '❌'}"
    )

    print(
        f"MACD: {'✅' if sell_macd else '❌'}"
    )

    print(
        f"RSI: {'✅' if sell_rsi else '❌'}"
    )

    print(
        f"Trend: {'✅' if sell_trend else '❌'}"
    )

    print("")
    print(f"➡️ SEGNALE: {segnale}")
    print("========================================")

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
        "trend_datetime": trend_datetime
    }


# ============================================================
# MESSAGGIO TELEGRAM
# ============================================================

def crea_messaggio(r):

    emoji = "🟢" if r["signal"] == "BUY" else "🔴"

    return (
        "🧪 PAPER/DEMO\n\n"
        f"{emoji} XAU/USD — {r['signal']}\n\n"
        f"⏰ Candela: {r['datetime'].astimezone(ZoneInfo('Europe/Rome')).strftime('%d/%m/%Y %H:%M')}\n"
        f"💰 Entry: {r['price']:.2f}\n"
        f"🛑 SL: {r['sl']:.2f}\n"
        f"🎯 TP: {r['tp']:.2f}\n\n"
        f"RSI: {r['rsi']:.2f}\n"
        f"ATR: {r['atr']:.2f}\n"
        f"Trend 15m: {r['trend_15m']}\n\n"
        "⚠️ Segnale simulato — nessun ordine reale."
    )


# ============================================================
# ESECUZIONE
# ============================================================

print("========================================")
print("🤖 XAU/USD BOT — PAPER/DEMO")
print("========================================")

try:

    risultato = analizza_xauusd()

    if risultato is None:

        print("⚠️ Analisi non disponibile")

        raise SystemExit(0)

    segnale = risultato["signal"]

    if segnale in ["BUY", "SELL"]:

        print("📨 Segnale trovato")

        messaggio = crea_messaggio(risultato)

        invia_telegram(messaggio)

    else:

        print("⏳ Nessun nuovo segnale")

    print("")
    print("✅ Esecuzione terminata")

except Exception as e:

    print("❌ ERRORE:")
    print(e)
    raise
