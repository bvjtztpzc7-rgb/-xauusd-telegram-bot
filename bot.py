import os
import requests
import pandas as pd
from datetime import timezone
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
SYMBOL = "XAU/USD"
INTERVAL = "5min"
ROME_TZ = ZoneInfo("Europe/Rome")
UTC_TZ = timezone.utc
# ============================================================
# STRATEGIA V8
# ============================================================
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 100
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MOMENTUM_PERIOD = 6
# BODY_STRONG V8
BODY_RATIO_MIN = 0.60
# Parametri candidati V8
SL_ATR = 1.25
TP_ATR = 2.50
# Cooldown V8
COOLDOWN_MINUTES = 30
# Orizzonte di riferimento del backtest
HORIZON_MINUTES = 30
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
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": 500,
        "timezone": "UTC",
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
    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )
    # --------------------------------------------------------
    # PREZZI
    # --------------------------------------------------------
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
        .dropna(
            subset=[
                "datetime",
                "open",
                "high",
                "low",
                "close"
            ]
        )
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last"
        )
        .reset_index(drop=True)
    )
    # --------------------------------------------------------
    # ORA ATTUALE
    # --------------------------------------------------------
    adesso_utc = pd.Timestamp.now(tz="UTC")
    adesso_roma = adesso_utc.tz_convert(ROME_TZ)
    print("")
    print("🕐 CONTROLLO ORARIO")
    print("----------------------------------------")
    print(
        "Ora attuale UTC:",
        adesso_utc.strftime("%d/%m/%Y %H:%M:%S")
    )
    print(
        "Ora attuale Roma:",
        adesso_roma.strftime("%d/%m/%Y %H:%M:%S")
    )
    print("----------------------------------------")
    # --------------------------------------------------------
    # ELIMINA DATI FUTURI
    # --------------------------------------------------------
    df = df[
        df["datetime"] <= adesso_utc
    ].copy()
    if df.empty:
        print("❌ Nessun dato valido")
        return None
    # --------------------------------------------------------
    # SOLO CANDELE 5M COMPLETAMENTE CHIUSE
    #
    # Se una candela parte alle 21:10, termina alle 21:15.
    # Alle 21:19 quella candela è quindi completamente chiusa.
    # --------------------------------------------------------
    durata_5m = pd.Timedelta(minutes=5)
    df_chiuse = df[
        df["datetime"] + durata_5m
        <= adesso_utc
    ].copy()
    if df_chiuse.empty:
        print("❌ Nessuna candela 5m completamente chiusa")
        return None
    ultima = df_chiuse["datetime"].iloc[-1]
    print(
        "Ultima candela 5m chiusa UTC:",
        ultima.strftime("%d/%m/%Y %H:%M")
    )
    print(
        "Ultima candela 5m chiusa Roma:",
        ultima
        .astimezone(ROME_TZ)
        .strftime("%d/%m/%Y %H:%M")
    )
    print("----------------------------------------")
    return df_chiuse.reset_index(drop=True)
# ============================================================
# INDICATORI
# ============================================================
def calcola_indicatori(df):
    df = df.copy()
    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------
    df["EMA20"] = (
        df["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False
        )
        .mean()
    )
    df["EMA50"] = (
        df["close"]
        .ewm(
            span=EMA_MID,
            adjust=False
        )
        .mean()
    )
    df["EMA100"] = (
        df["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False
        )
        .mean()
    )
    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------
    ema12 = (
        df["close"]
        .ewm(
            span=MACD_FAST,
            adjust=False
        )
        .mean()
    )
    ema26 = (
        df["close"]
        .ewm(
            span=MACD_SLOW,
            adjust=False
        )
        .mean()
    )
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = (
        df["MACD"]
        .ewm(
            span=MACD_SIGNAL,
            adjust=False
        )
        .mean()
    )
    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = (
        gain
        .ewm(
            alpha=1 / RSI_PERIOD,
            min_periods=RSI_PERIOD,
            adjust=False
        )
        .mean()
    )
    avg_loss = (
        loss
        .ewm(
            alpha=1 / RSI_PERIOD,
            min_periods=RSI_PERIOD,
            adjust=False
        )
        .mean()
    )
    rs = avg_gain / avg_loss
    df["RSI"] = (
        100 -
        (100 / (1 + rs))
    )
    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------
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
            alpha=1 / ATR_PERIOD,
            min_periods=ATR_PERIOD,
            adjust=False
        )
        .mean()
    )
    # --------------------------------------------------------
    # MOM6
    # --------------------------------------------------------
    df["MOM6"] = (
        df["close"] -
        df["close"].shift(MOMENTUM_PERIOD)
    )
    # --------------------------------------------------------
    # BODY RATIO
    # --------------------------------------------------------
    df["BODY"] = abs(
        df["close"] -
        df["open"]
    )
    df["RANGE"] = (
        df["high"] -
        df["low"]
    )
    df["BODY_RATIO"] = 0.0
    mask = df["RANGE"] > 0
    df.loc[mask, "BODY_RATIO"] = (
        df.loc[mask, "BODY"] /
        df.loc[mask, "RANGE"]
    )
    return df
# ============================================================
# TREND 15M
# ============================================================
def calcola_trend_15m(df, adesso_utc):
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
    # --------------------------------------------------------
    # SOLO CANDELE 15M COMPLETAMENTE CHIUSE
    # --------------------------------------------------------
    df_15m = df_15m[
        df_15m["datetime"] +
        pd.Timedelta(minutes=15)
        <= adesso_utc
    ].copy()
    if len(df_15m) < 50:
        return "NEUTRAL", None
    # --------------------------------------------------------
    # EMA 15M
    # --------------------------------------------------------
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
    ultima = df_15m.iloc[-1]
    if (
        ultima["EMA20_15"] >
        ultima["EMA50_15"]
    ):
        trend = "BULLISH"
    elif (
        ultima["EMA20_15"] <
        ultima["EMA50_15"]
    ):
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"
    return trend, ultima["datetime"]
# ============================================================
# CONTROLLO SETUP PRECEDENTE
# ============================================================
def setup_sell_valido(row):
    if pd.isna(row["EMA20"]):
        return False
    if pd.isna(row["EMA50"]):
        return False
    if pd.isna(row["EMA100"]):
        return False
    if pd.isna(row["MACD"]):
        return False
    if pd.isna(row["MACD_signal"]):
        return False
    if pd.isna(row["RSI"]):
        return False
    if pd.isna(row["MOM6"]):
        return False
    if pd.isna(row["BODY_RATIO"]):
        return False
    return (
        row["EMA20"] < row["EMA50"] < row["EMA100"]
        and row["MACD"] < row["MACD_signal"]
        and 30 < row["RSI"] < 65
        and row["MOM6"] < 0
        and row["BODY_RATIO"] >= BODY_RATIO_MIN
    )
# ============================================================
# COOLDOWN
# ============================================================
def controllo_cooldown(df):
    cutoff = (
        df["datetime"].iloc[-1]
        - pd.Timedelta(minutes=COOLDOWN_MINUTES)
    )
    precedenti = df[
        (df["datetime"] < df["datetime"].iloc[-1])
        &
        (df["datetime"] >= cutoff)
    ].copy()
    # --------------------------------------------------------
    # IMPORTANTE:
    #
    # Non consideriamo un semplice setup come "trade".
    # Il cooldown viene attivato solo se una candela precedente
    # ha generato un setup SELL completo.
    # --------------------------------------------------------
    for _, row in precedenti.iterrows():
        if setup_sell_valido(row):
            return False, row["datetime"]
    return True, None
# ============================================================
# ANALISI
# ============================================================
def analizza_xauusd():
    df = scarica_dati()
    if df is None:
        return None
    if len(df) < 150:
        print("❌ Dati insufficienti")
        return None
    # Ora usata per tutti i controlli temporali
    adesso_utc = pd.Timestamp.now(tz="UTC")
    df = calcola_indicatori(df)
    # --------------------------------------------------------
    # TREND 15M
    # --------------------------------------------------------
    trend_15m, trend_datetime = (
        calcola_trend_15m(
            df,
            adesso_utc
        )
    )
    # --------------------------------------------------------
    # ULTIMA 5M CHIUSA
    # --------------------------------------------------------
    candela = df.iloc[-1]
    prezzo = candela["close"]
    ema20 = candela["EMA20"]
    ema50 = candela["EMA50"]
    ema100 = candela["EMA100"]
    macd = candela["MACD"]
    macd_signal = candela["MACD_signal"]
    rsi = candela["RSI"]
    atr = candela["ATR"]
    mom6 = candela["MOM6"]
    body = candela["BODY"]
    candle_range = candela["RANGE"]
    body_ratio = candela["BODY_RATIO"]
    # --------------------------------------------------------
    # FILTRI SELL
    # --------------------------------------------------------
    sell_ema = (
        ema20 < ema50 < ema100
    )
    sell_macd = (
        macd < macd_signal
    )
    sell_rsi = (
        30 < rsi < 65
    )
    sell_trend = (
        trend_15m == "BEARISH"
    )
    sell_momentum = (
        mom6 < 0
    )
    sell_body = (
        body_ratio >= BODY_RATIO_MIN
    )
    cooldown_ok, cooldown_datetime = (
        controllo_cooldown(df)
    )
    # --------------------------------------------------------
    # SEGNALE
    # --------------------------------------------------------
    segnale = "NONE"
    if (
        sell_ema
        and sell_macd
        and sell_rsi
        and sell_trend
        and sell_momentum
        and sell_body
        and cooldown_ok
    ):
        segnale = "SELL"
    # --------------------------------------------------------
    # SL / TP
    # --------------------------------------------------------
    sl = None
    tp = None
    if segnale == "SELL":
        sl = prezzo + (
            SL_ATR * atr
        )
        tp = prezzo - (
            TP_ATR * atr
        )
    # ========================================================
    # DIAGNOSTICA
    # ========================================================
    candela_roma = (
        candela["datetime"]
        .astimezone(ROME_TZ)
    )
    print("")
    print("========================================")
    print("📊 ANALISI XAU/USD — V8.1")
    print("========================================")
    print(
        "⏰ Ora esecuzione UTC:",
        adesso_utc.strftime("%d/%m/%Y %H:%M:%S")
    )
    print(
        "🇮🇹 Ora esecuzione Roma:",
        adesso_utc
        .astimezone(ROME_TZ)
        .strftime("%d/%m/%Y %H:%M:%S")
    )
    print("")
    print(
        "⏰ Candela UTC:",
        candela["datetime"]
        .strftime("%d/%m/%Y %H:%M")
    )
    print(
        "🇮🇹 Candela Roma:",
        candela_roma.strftime("%d/%m/%Y %H:%M")
    )
    print(
        f"💰 Prezzo: {prezzo:.2f}"
    )
    print("")
    print("📐 INDICATORI")
    print("----------------------------------------")
    print(
        f"EMA20:  {ema20:.4f}"
    )
    print(
        f"EMA50:  {ema50:.4f}"
    )
    print(
        f"EMA100: {ema100:.4f}"
    )
    print(
        f"MACD: {macd:.4f}"
    )
    print(
        f"MACD Signal: {macd_signal:.4f}"
    )
    print(
        f"RSI: {rsi:.2f}"
    )
    print(
        f"ATR: {atr:.4f}"
    )
    print(
        f"MOM6: {mom6:.4f}"
    )
    print("")
    print("🕯️ FORZA CANDELA")
    print("----------------------------------------")
    print(
        f"Body: {body:.4f}"
    )
    print(
        f"Range: {candle_range:.4f}"
    )
    print(
        f"Body Ratio: {body_ratio:.3f}"
    )
    print(
        f"Soglia: {BODY_RATIO_MIN:.2f}"
    )
    print("")
    print(
        f"📈 Trend 15m: {trend_15m}"
    )
    if trend_datetime is not None:
        print(
            "🕐 Candela trend 15m:",
            trend_datetime
            .astimezone(ROME_TZ)
            .strftime("%d/%m/%Y %H:%M")
        )
    print("")
    print("🔴 FILTRI SELL")
    print("----------------------------------------")
    print(
        f"EMA20 < EMA50 < EMA100: "
        f"{'✅' if sell_ema else '❌'}"
    )
    print(
        f"MACD < Signal: "
        f"{'✅' if sell_macd else '❌'}"
    )
    print(
        f"RSI 30–65: "
        f"{'✅' if sell_rsi else '❌'}"
    )
    print(
        f"Trend 15m BEARISH: "
        f"{'✅' if sell_trend else '❌'}"
    )
    print(
        f"MOM6 < 0: "
        f"{'✅' if sell_momentum else '❌'}"
    )
    print(
        f"Body Ratio >= {BODY_RATIO_MIN:.2f}: "
        f"{'✅' if sell_body else '❌'}"
    )
    print(
        f"Cooldown {COOLDOWN_MINUTES}m: "
        f"{'✅' if cooldown_ok else '❌'}"
    )
    if (
        not cooldown_ok
        and cooldown_datetime is not None
    ):
        print(
            "Setup precedente:",
            cooldown_datetime
            .astimezone(ROME_TZ)
            .strftime("%d/%m/%Y %H:%M")
        )
    print("")
    print(
        f"➡️ SEGNALE: {segnale}"
    )
    if sl is not None:
        print(
            f"🛑 SL: {sl:.2f}"
        )
    if tp is not None:
        print(
            f"🎯 TP: {tp:.2f}"
        )
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
        "ema100": ema100,
        "mom6": mom6,
        "body": body,
        "range": candle_range,
        "body_ratio": body_ratio,
        "trend_15m": trend_15m,
        "trend_datetime": trend_datetime,
        "cooldown_ok": cooldown_ok
    }
# ============================================================
# TELEGRAM
# ============================================================
def crea_messaggio(r):
    candela_roma = (
        r["datetime"]
        .astimezone(ROME_TZ)
        .strftime("%d/%m/%Y %H:%M")
    )
    return (
        "🧪 PAPER/DEMO — XAU/USD V8.1\n\n"
        "🔴 SELL\n\n"
        f"⏰ Candela: {candela_roma}\n"
        f"💰 Entry: {r['price']:.2f}\n"
        f"🛑 SL: {r['sl']:.2f}\n"
        f"🎯 TP: {r['tp']:.2f}\n\n"
        f"RSI: {r['rsi']:.2f}\n"
        f"ATR: {r['atr']:.2f}\n"
        f"MOM6: {r['mom6']:.2f}\n"
        f"Body Ratio: {r['body_ratio']:.2f}\n"
        f"Trend 15m: {r['trend_15m']}\n\n"
        f"SL: {SL_ATR:.2f} ATR\n"
        f"TP: {TP_ATR:.2f} ATR\n"
        f"Cooldown: {COOLDOWN_MINUTES} min\n"
        f"Orizzonte test: {HORIZON_MINUTES} min\n\n"
        "⚠️ Segnale sperimentale PAPER/DEMO.\n"
        "Nessun ordine reale viene eseguito."
    )
# ============================================================
# ESECUZIONE
# ============================================================
print("========================================")
print("🤖 XAU/USD BOT — V8.1 PAPER/DEMO")
print("========================================")
print("")
print("STRATEGIA:")
print("SELL + BODY_STRONG")
print(
    f"Body Ratio >= {BODY_RATIO_MIN:.2f}"
)
print(
    f"SL = {SL_ATR:.2f} ATR"
)
print(
    f"TP = {TP_ATR:.2f} ATR"
)
print(
    f"Cooldown = {COOLDOWN_MINUTES} min"
)
print(
    f"Horizon = {HORIZON_MINUTES} min"
)
print("")
try:
    risultato = analizza_xauusd()
    if risultato is None:
        print("⚠️ Analisi non disponibile")
        raise SystemExit(0)
    segnale = risultato["signal"]
    if segnale == "SELL":
        print("")
        print("📨 SEGNALE SELL TROVATO")
        messaggio = crea_messaggio(
            risultato
        )
        invia_telegram(
            messaggio
        )
    else:
        print("")
        print("⏳ Nessun segnale SELL valido")
    print("")
    print("✅ Esecuzione terminata")
except Exception as e:
    print("")
    print("❌ ERRORE:")
    print(e)
    raise
