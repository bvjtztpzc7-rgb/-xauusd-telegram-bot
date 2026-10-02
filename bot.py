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
# PARAMETRI STRATEGIA V8
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
# V8
BODY_RATIO_MIN = 0.60
# V8 candidato principale
SL_ATR = 1.25
TP_ATR = 2.50
# Cooldown 30 minuti = 6 candele da 5 minuti
COOLDOWN_MINUTES = 30
COOLDOWN_CANDLES = COOLDOWN_MINUTES // 5
# Orizzonte del backtest V8
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
    # CONTROLLO ORARIO
    # --------------------------------------------------------
    adesso_utc = pd.Timestamp.now(tz="UTC")
    print("")
    print("🕐 CONTROLLO DATI")
    print("----------------------------------------")
    print(
        "Ora attuale UTC:",
        adesso_utc.strftime("%d/%m/%Y %H:%M:%S")
    )
    print(
        "Ultimo timestamp ricevuto:",
        df["datetime"].iloc[-1]
        .strftime("%d/%m/%Y %H:%M:%S UTC")
    )
    # --------------------------------------------------------
    # ELIMINA DATI FUTURI
    # --------------------------------------------------------
    df = df[
        df["datetime"] <= adesso_utc
    ].copy()
    if df.empty:
        print("❌ Nessun dato valido non futuro")
        return None
    # --------------------------------------------------------
    # SOLO CANDELE 5M COMPLETAMENTE CHIUSE
    # --------------------------------------------------------
    durata_candela = pd.Timedelta(minutes=5)
    df_chiuse = df[
        df["datetime"] + durata_candela
        <= adesso_utc
    ].copy()
    if df_chiuse.empty:
        print("❌ Nessuna candela 5m completamente chiusa")
        return None
    ultima = df_chiuse["datetime"].iloc[-1]
    print(
        "Ultima candela 5m chiusa:",
        ultima.strftime("%d/%m/%Y %H:%M UTC")
    )
    print(
        "Ultima candela 5m chiusa (Roma):",
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
    # EMA 20 / 50 / 100
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
    # RSI 14
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
    # ATR 14
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
    #
    # Corpo della candela / range totale
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
    # --------------------------------------------------------
    # SOLO CANDELE 15M COMPLETAMENTE CHIUSE
    # --------------------------------------------------------
    adesso_utc = pd.Timestamp.now(tz="UTC")
    df_15m = df_15m[
        df_15m["datetime"] +
        pd.Timedelta(minutes=15)
        <= adesso_utc
    ].copy()
    if len(df_15m) < 50:
        return "NEUTRAL", None
    # --------------------------------------------------------
    # EMA 20 / 50
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
# CONTROLLO COOLDOWN
# ============================================================
def controllo_cooldown(df):
    if len(df) <= COOLDOWN_CANDLES:
        return True, None
    # --------------------------------------------------------
    # Analizziamo le candele precedenti alla corrente.
    #
    # Se nelle ultime 30m c'è già stata una configurazione
    # SELL valida, evitiamo di generare un altro segnale.
    # --------------------------------------------------------
    precedenti = df.iloc[
        -COOLDOWN_CANDLES - 1:-1
    ].copy()
    for _, row in precedenti.iterrows():
        sell_ema = (
            row["EMA20"] <
            row["EMA50"] <
            row["EMA100"]
        )
        sell_macd = (
            row["MACD"] <
            row["MACD_signal"]
        )
        sell_rsi = (
            30 < row["RSI"] < 65
        )
        sell_momentum = (
            row["MOM6"] < 0
        )
        body_strong = (
            row["BODY_RATIO"] >=
            BODY_RATIO_MIN
        )
        if (
            sell_ema
            and sell_macd
            and sell_rsi
            and sell_momentum
            and body_strong
        ):
            return (
                False,
                row["datetime"]
            )
    return True, None
# ============================================================
# ANALISI XAU/USD
# ============================================================
def analizza_xauusd():
    df = scarica_dati()
    if df is None:
        return None
    if len(df) < 150:
        print("❌ Dati insufficienti")
        return None
    df = calcola_indicatori(df)
    # --------------------------------------------------------
    # TREND 15M
    # --------------------------------------------------------
    trend_15m, trend_datetime = (
        calcola_trend_15m(df)
    )
    # --------------------------------------------------------
    # ULTIMA CANDELA 5M CHIUSA
    # --------------------------------------------------------
    candela = df.iloc[-1]
    preco = candela["close"]
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
    # FILTRI SELL V8
    # --------------------------------------------------------
    sell_ema = (
        ema20 <
        ema50 <
        ema100
    )
    sell_macd = (
        macd <
        macd_signal
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
        body_ratio >=
        BODY_RATIO_MIN
    )
    # --------------------------------------------------------
    # COOLDOWN
    # --------------------------------------------------------
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
        sl = preco + (
            SL_ATR * atr
        )
        tp = preco - (
            TP_ATR * atr
        )
    # --------------------------------------------------------
    # DIAGNOSTICA
    # --------------------------------------------------------
    candela_roma = (
        candela["datetime"]
        .astimezone(ROME_TZ)
    )
    print("")
    print("========================================")
    print("📊 ANALISI XAU/USD — V8")
    print("========================================")
    print(
        f"⏰ Candela UTC: "
        f"{candela['datetime'].strftime('%d/%m/%Y %H:%M')}"
    )
    print(
        f"🇮🇹 Candela Roma: "
        f"{candela_roma.strftime('%d/%m/%Y %H:%M')}"
    )
    print(
        f"💰 Prezzo: {preco:.2f}"
    )
    print("")
    print("📐 INDICATORI")
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
        f"Soglia Body: {BODY_RATIO_MIN:.2f}"
    )
    print("")
    print(
        f"📈 Trend 15m: {trend_15m}"
    )
    if trend_datetime is not None:
        print(
            "🕐 Ultimo trend 15m chiuso:",
            trend_datetime
            .astimezone(ROME_TZ)
            .strftime("%d/%m/%Y %H:%M")
        )
    print("")
    print("🔴 FILTRI SELL")
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
        f"Cooldown 30m: "
        f"{'✅' if cooldown_ok else '❌'}"
    )
    if not cooldown_ok and cooldown_datetime is not None:
        print(
            "Ultimo setup trovato:",
            cooldown_datetime
            .astimezone(ROME_TZ)
            .strftime("%d/%m/%Y %H:%M")
        )
    print("")
    print(f"➡️ SEGNALE: {segnale}")
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
        "price": preco,
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
# MESSAGGIO TELEGRAM
# ============================================================
def crea_messaggio(r):
    candela_roma = (
        r["datetime"]
        .astimezone(ROME_TZ)
        .strftime("%d/%m/%Y %H:%M")
    )
    return (
        "🧪 PAPER/DEMO — V8\n\n"
        "🔴 XAU/USD — SELL\n\n"
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
        f"Orizzonte test: {HORIZON_MINUTES} min\n"
        f"Cooldown: {COOLDOWN_MINUTES} min\n\n"
        "⚠️ Segnale sperimentale PAPER/DEMO.\n"
        "Nessun ordine reale viene eseguito."
    )
# ============================================================
# ESECUZIONE
# ============================================================
print("========================================")
print("🤖 XAU/USD BOT — V8 PAPER/DEMO")
print("========================================")
print("")
print("Strategia:")
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
print("")
try:
    risultato = analizza_xauusd()
    if risultato is None:
        print("⚠️ Analisi non disponibile")
        raise SystemExit(0)
    segnale = risultato["signal"]
    if segnale == "SELL":
        print("📨 Segnale SELL trovato")
        messaggio = crea_messaggio(
            risultato
        )
        invia_telegram(
            messaggio
        )
    else:
        print(
            "⏳ Nessun segnale SELL valido"
        )
    print("")
    print("✅ Esecuzione terminata")
except Exception as e:
    print("❌ ERRORE:")
    print(e)
    raise
