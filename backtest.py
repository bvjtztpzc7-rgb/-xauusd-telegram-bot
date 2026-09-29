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
# SCARICA DATI STORICI
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

    response = requests.get(url, params=params, timeout=30)
    data = response.json()

    if "values" not in data:
        raise RuntimeError(f"Errore Twelve Data: {data}")

    df = pd.DataFrame(data["values"])

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("datetime")
    df = df.reset_index(drop=True)

    return df


# ============================================================
# INDICATORI
# ============================================================

def calcola_indicatori(df):

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

    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["high"] - df["low"]

    high_close = (
        df["high"] - df["close"].shift()
    ).abs()

    low_close = (
        df["low"] - df["close"].shift()
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

    return df


# ============================================================
# TREND 15 MINUTI
# ============================================================

def calcola_trend_15m(df):

    df15 = (
        df.set_index("datetime")
        .resample("15min")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        })
        .dropna()
    )

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

    # Riportiamo il trend 15m su ogni candela 5m
    df = df.copy()

    df["trend_15m"] = (
        df.set_index("datetime")
        .join(
            df15["trend"].rename("trend_15m"),
            how="left"
        )["trend_15m"]
        .ffill()
        .values
    )

    return df


# ============================================================
# BACKTEST
# ============================================================

def backtest(df):

    operazioni = []

    posizione = None

    for i in range(1, len(df) - 1):

        riga = df.iloc[i]

        # Ignora righe senza indicatori
        if pd.isna(riga["atr"]):
            continue

        # ----------------------------------------------------
        # Se abbiamo una posizione aperta
        # ----------------------------------------------------

        if posizione is not None:

            high = riga["high"]
            low = riga["low"]

            entry = posizione["entry"]
            sl = posizione["sl"]
            tp = posizione["tp"]

            risultato = None
            prezzo_uscita = None

            # Caso BUY
            if posizione["tipo"] == "BUY":

                # Se nella stessa candela vengono colpiti entrambi,
                # assumiamo SL prima (scenario conservativo)
                if low <= sl:
                    risultato = "SL"
                    prezzo_uscita = sl

                elif high >= tp:
                    risultato = "TP"
                    prezzo_uscita = tp

            # Caso SELL
            elif posizione["tipo"] == "SELL":

                if high >= sl:
                    risultato = "SL"
                    prezzo_uscita = sl

                elif low <= tp:
                    risultato = "TP"
                    prezzo_uscita = tp

            # Chiudi posizione
            if risultato:

                if posizione["tipo"] == "BUY":
                    profitto = prezzo_uscita - entry
                else:
                    profitto = entry - prezzo_uscita

                operazioni.append({
                    "tipo": posizione["tipo"],
                    "entry": entry,
                    "uscita": prezzo_uscita,
                    "risultato": risultato,
                    "profitto": profitto,
                    "data": posizione["data"]
                })

                posizione = None

            continue

        # ----------------------------------------------------
        # Condizioni BUY
        # ----------------------------------------------------

        buy_ema = riga["ema20"] > riga["ema50"]

        buy_macd = riga["macd"] > riga["macd_signal"]

        buy_rsi = (
            riga["rsi"] > 30
            and riga["rsi"] < 65
        )

        buy_trend = riga["trend_15m"] == "BULLISH"

        # ----------------------------------------------------
        # Condizioni SELL
        # ----------------------------------------------------

        sell_ema = riga["ema20"] < riga["ema50"]

        sell_macd = riga["macd"] < riga["macd_signal"]

        sell_rsi = (
            riga["rsi"] > 30
            and riga["rsi"] < 65
        )

        sell_trend = riga["trend_15m"] == "BEARISH"

        # ----------------------------------------------------
        # Segnale BUY
        # ----------------------------------------------------

        if (
            buy_ema
            and buy_macd
            and buy_rsi
            and buy_trend
        ):

            entry = df.iloc[i + 1]["open"]

            atr = riga["atr"]

            sl = entry - (1.5 * atr)
            tp = entry + (2.0 * atr)

            posizione = {
                "tipo": "BUY",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "data": df.iloc[i + 1]["datetime"]
            }

        # ----------------------------------------------------
        # Segnale SELL
        # ----------------------------------------------------

        elif (
            sell_ema
            and sell_macd
            and sell_rsi
            and sell_trend
        ):

            entry = df.iloc[i + 1]["open"]

            atr = riga["atr"]

            sl = entry + (1.5 * atr)
            tp = entry - (2.0 * atr)

            posizione = {
                "tipo": "SELL",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "data": df.iloc[i + 1]["datetime"]
            }

    return operazioni


# ============================================================
# RISULTATI
# ============================================================

def mostra_risultati(operazioni):

    print()
    print("=" * 55)
    print("📊 RISULTATI BACKTEST XAU/USD")
    print("=" * 55)

    totale = len(operazioni)

    if totale == 0:
        print("❌ Nessuna operazione trovata.")
        return

    tp = sum(
        1 for x in operazioni
        if x["risultato"] == "TP"
    )

    sl = sum(
        1 for x in operazioni
        if x["risultato"] == "SL"
    )

    win_rate = (tp / totale) * 100

    profitto = sum(
        x["profitto"]
        for x in operazioni
    )

    print(f"Operazioni: {totale}")
    print(f"TP: {tp}")
    print(f"SL: {sl}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Risultato prezzo: {profitto:.2f}")

    print()
    print("Ultime operazioni:")

    for op in operazioni[-10:]:

        print(
            f"{op['data']} | "
            f"{op['tipo']} | "
            f"{op['risultato']} | "
            f"{op['profitto']:.2f}"
        )

    print("=" * 55)


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 55)
print("🤖 BACKTEST XAU/USD — PAPER/DEMO")
print("=" * 55)

df = scarica_dati()

df = calcola_indicatori(df)

df = calcola_trend_15m(df)

operazioni = backtest(df)

mostra_risultati(operazioni)
