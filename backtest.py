import os
import requests
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURAZIONE
# ============================================================

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurato")


# ============================================================
# SCARICA DATI
# ============================================================

def scarica_dati():

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": "XAU/USD",
        "interval": "5min",
        "outputsize": 500,
        "apikey": API_KEY,
        "format": "JSON"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    data = response.json()

    if "values" not in data:
        raise RuntimeError(f"Errore Twelve Data: {data}")

    df = pd.DataFrame(data["values"])

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.sort_values("datetime")
    df = df.reset_index(drop=True)

    return df


# ============================================================
# INDICATORI
# ============================================================

def calcola_indicatori(df):

    # EMA
    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    # MACD
    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # ATR
    high_low = df["high"] - df["low"]

    high_close = (
        df["high"] -
        df["close"].shift()
    ).abs()

    low_close = (
        df["low"] -
        df["close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    # MACD precedente
    df["macd_prev"] = df["macd"].shift(1)
    df["signal_prev"] = df["macd_signal"].shift(1)

    return df


# ============================================================
# TREND 15 MINUTI
# ============================================================

def aggiungi_trend_15m(df):

    temp = df.set_index("datetime")

    df15 = temp.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

    df15["ema20"] = df15["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df15["ema50"] = df15["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df15["trend"] = np.where(
        df15["ema20"] > df15["ema50"],
        "BULLISH",
        "BEARISH"
    )

    # Usiamo solamente candele 15m CHIUSE.
    # La candela 15m in corso non viene utilizzata.

    df15["trend_closed"] = df15["trend"].shift(1)

    result = df.copy()

    result["trend_15m"] = None

    for i in range(len(result)):

        timestamp = result.iloc[i]["datetime"]

        periodo = timestamp.floor("15min")

        periodi_precedenti = df15.index[
            df15.index < periodo
        ]

        if len(periodi_precedenti) == 0:
            continue

        ultimo_periodo = periodi_precedenti[-1]

        result.loc[
            i,
            "trend_15m"
        ] = df15.loc[
            ultimo_periodo,
            "trend"
        ]

    return result


# ============================================================
# GENERAZIONE SEGNALE
# ============================================================

def genera_segnale(
    riga,
    strategia
):

    if pd.isna(riga["atr"]):
        return None

    trend = riga["trend_15m"]

    # --------------------------------------------------------
    # STRATEGIA A
    # Strategia attuale
    # --------------------------------------------------------

    if strategia == "A":

        buy = (
            riga["ema20"] > riga["ema50"]
            and
            riga["macd"] > riga["macd_signal"]
            and
            30 < riga["rsi"] < 65
            and
            trend == "BULLISH"
        )

        sell = (
            riga["ema20"] < riga["ema50"]
            and
            riga["macd"] < riga["macd_signal"]
            and
            30 < riga["rsi"] < 65
            and
            trend == "BEARISH"
        )

    # --------------------------------------------------------
    # STRATEGIA B
    # RSI 30-70
    # --------------------------------------------------------

    elif strategia == "B":

        buy = (
            riga["ema20"] > riga["ema50"]
            and
            riga["macd"] > riga["macd_signal"]
            and
            30 < riga["rsi"] < 70
            and
            trend == "BULLISH"
        )

        sell = (
            riga["ema20"] < riga["ema50"]
            and
            riga["macd"] < riga["macd_signal"]
            and
            30 < riga["rsi"] < 70
            and
            trend == "BEARISH"
        )

    # --------------------------------------------------------
    # STRATEGIA C
    # Incrocio MACD
    # --------------------------------------------------------

    elif strategia == "C":

        bullish_cross = (
            riga["macd_prev"] <=
            riga["signal_prev"]
            and
            riga["macd"] >
            riga["macd_signal"]
        )

        bearish_cross = (
            riga["macd_prev"] >=
            riga["signal_prev"]
            and
            riga["macd"] <
            riga["macd_signal"]
        )

        buy = (
            riga["ema20"] > riga["ema50"]
            and
            bullish_cross
            and
            30 < riga["rsi"] < 70
            and
            trend == "BULLISH"
        )

        sell = (
            riga["ema20"] < riga["ema50"]
            and
            bearish_cross
            and
            30 < riga["rsi"] < 70
            and
            trend == "BEARISH"
        )

    else:
        return None

    if buy:
        return "BUY"

    if sell:
        return "SELL"

    return None


# ============================================================
# BACKTEST
# ============================================================

def esegui_backtest(
    df,
    strategia
):

    operazioni = []

    posizione = None

    for i in range(1, len(df) - 1):

        riga = df.iloc[i]

        # ----------------------------------------------------
        # GESTIONE POSIZIONE
        # ----------------------------------------------------

        if posizione is not None:

            high = riga["high"]
            low = riga["low"]

            entry = posizione["entry"]
            sl = posizione["sl"]
            tp = posizione["tp"]

            risultato = None
            uscita = None

            if posizione["tipo"] == "BUY":

                if low <= sl:
                    risultato = "SL"
                    uscita = sl

                elif high >= tp:
                    risultato = "TP"
                    uscita = tp

            else:

                if high >= sl:
                    risultato = "SL"
                    uscita = sl

                elif low <= tp:
                    risultato = "TP"
                    uscita = tp

            if risultato:

                if posizione["tipo"] == "BUY":
                    profitto = uscita - entry
                else:
                    profitto = entry - uscita

                operazioni.append({
                    "tipo": posizione["tipo"],
                    "risultato": risultato,
                    "profitto": profitto,
                    "data": posizione["data"]
                })

                posizione = None

            continue

        # ----------------------------------------------------
        # CERCA NUOVO SEGNALE
        # ----------------------------------------------------

        segnale = genera_segnale(
            riga,
            strategia
        )

        if segnale is None:
            continue

        # Entriamo all'apertura della candela successiva
        entry = df.iloc[i + 1]["open"]

        atr = riga["atr"]

        if segnale == "BUY":

            sl = entry - (
                1.5 * atr
            )

            tp = entry + (
                2.0 * atr
            )

        else:

            sl = entry + (
                1.5 * atr
            )

            tp = entry - (
                2.0 * atr
            )

        posizione = {
            "tipo": segnale,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "data": df.iloc[i + 1]["datetime"]
        }

    return operazioni


# ============================================================
# RISULTATI
# ============================================================

def risultati(operazioni):

    totale = len(operazioni)

    if totale == 0:

        return {
            "operazioni": 0,
            "tp": 0,
            "sl": 0,
            "winrate": 0,
            "profitto": 0
        }

    tp = sum(
        x["risultato"] == "TP"
        for x in operazioni
    )

    sl = sum(
        x["risultato"] == "SL"
        for x in operazioni
    )

    winrate = (
        tp / totale
    ) * 100

    profitto = sum(
        x["profitto"]
        for x in operazioni
    )

    return {
        "operazioni": totale,
        "tp": tp,
        "sl": sl,
        "winrate": winrate,
        "profitto": profitto
    }


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 65)
print("🤖 CONFRONTO STRATEGIE XAU/USD — PAPER/DEMO")
print("=" * 65)

df = scarica_dati()

print(
    f"\n📊 Candele analizzate: {len(df)}"
)

print(
    f"📅 Da: {df.iloc[0]['datetime']}"
)

print(
    f"📅 A:  {df.iloc[-1]['datetime']}"
)

df = calcola_indicatori(df)

df = aggiungi_trend_15m(df)

print()
print("=" * 65)

strategie = {
    "A": "Strategia attuale — RSI 30-65",
    "B": "RSI ampliato — RSI 30-70",
    "C": "Incrocio MACD — RSI 30-70"
}

for codice, nome in strategie.items():

    operazioni = esegui_backtest(
        df,
        codice
    )

    r = risultati(operazioni)

    print()
    print(f"📌 STRATEGIA {codice}")
    print(nome)
    print("-" * 65)

    print(
        f"Operazioni: {r['operazioni']}"
    )

    print(
        f"TP: {r['tp']}"
    )

    print(
        f"SL: {r['sl']}"
    )

    print(
        f"Win rate: {r['winrate']:.2f}%"
    )

    print(
        f"Risultato prezzo: {r['profitto']:.2f}"
    )

print()
print("=" * 65)
print("✅ CONFRONTO TERMINATO")
print("=" * 65)
print()
print(
    "⚠️ I risultati sono simulazioni storiche "
    "e non garantiscono risultati futuri."
)
