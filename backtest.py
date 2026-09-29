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
        "outputsize": 5000,
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

    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

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

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

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

def genera_segnale(riga, strategia):

    if pd.isna(riga["atr"]):
        return None

    trend = riga["trend_15m"]

    # --------------------------------------------------------
    # STRATEGIA A
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
    # --------------------------------------------------------

    elif strategia == "C":

        bullish_cross = (
            riga["macd_prev"] <= riga["signal_prev"]
            and
            riga["macd"] > riga["macd_signal"]
        )

        bearish_cross = (
            riga["macd_prev"] >= riga["signal_prev"]
            and
            riga["macd"] < riga["macd_signal"]
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

def esegui_backtest(df, strategia):

    operazioni = []
    posizione = None

    for i in range(1, len(df) - 1):

        riga = df.iloc[i]

        # ----------------------------------------------------
        # POSIZIONE APERTA
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
        # NUOVO SEGNALE
        # ----------------------------------------------------

        segnale = genera_segnale(
            riga,
            strategia
        )

        if segnale is None:
            continue

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
# METRICHE
# ============================================================

def calcola_metriche(operazioni):

    if not operazioni:

        return {
            "totale": 0,
            "tp": 0,
            "sl": 0,
            "winrate": 0,
            "profitto": 0,
            "profit_factor": 0,
            "media_win": 0,
            "media_loss": 0,
            "drawdown": 0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "buy": 0,
            "buy_tp": 0,
            "buy_sl": 0,
            "sell": 0,
            "sell_tp": 0,
            "sell_sl": 0
        }

    totale = len(operazioni)

    vincenti = [
        x["profitto"]
        for x in operazioni
        if x["risultato"] == "TP"
    ]

    perdenti = [
        x["profitto"]
        for x in operazioni
        if x["risultato"] == "SL"
    ]

    tp = len(vincenti)
    sl = len(perdenti)

    winrate = (
        tp / totale
    ) * 100

    profitto = sum(
        x["profitto"]
        for x in operazioni
    )

    profitto_lordo = sum(vincenti)

    perdita_lorda = abs(
        sum(perdenti)
    )

    if perdita_lorda > 0:
        profit_factor = (
            profitto_lordo /
            perdita_lorda
        )
    else:
        profit_factor = float("inf")

    media_win = (
        np.mean(vincenti)
        if vincenti
        else 0
    )

    media_loss = (
        np.mean(perdenti)
        if perdenti
        else 0
    )

    # --------------------------------------------------------
    # EQUITY CURVE E DRAWDOWN
    # --------------------------------------------------------

    capitale = 0
    massimo = 0
    max_drawdown = 0

    for op in operazioni:

        capitale += op["profitto"]

        if capitale > massimo:
            massimo = capitale

        drawdown = massimo - capitale

        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # --------------------------------------------------------
    # STREAK
    # --------------------------------------------------------

    max_win_streak = 0
    max_loss_streak = 0

    win_streak = 0
    loss_streak = 0

    for op in operazioni:

        if op["risultato"] == "TP":

            win_streak += 1
            loss_streak = 0

        else:

            loss_streak += 1
            win_streak = 0

        max_win_streak = max(
            max_win_streak,
            win_streak
        )

        max_loss_streak = max(
            max_loss_streak,
            loss_streak
        )

    # --------------------------------------------------------
    # BUY / SELL
    # --------------------------------------------------------

    buy_ops = [
        x for x in operazioni
        if x["tipo"] == "BUY"
    ]

    sell_ops = [
        x for x in operazioni
        if x["tipo"] == "SELL"
    ]

    buy_tp = sum(
        x["risultato"] == "TP"
        for x in buy_ops
    )

    buy_sl = sum(
        x["risultato"] == "SL"
        for x in buy_ops
    )

    sell_tp = sum(
        x["risultato"] == "TP"
        for x in sell_ops
    )

    sell_sl = sum(
        x["risultato"] == "SL"
        for x in sell_ops
    )

    return {
        "totale": totale,
        "tp": tp,
        "sl": sl,
        "winrate": winrate,
        "profitto": profitto,
        "profit_factor": profit_factor,
        "media_win": media_win,
        "media_loss": media_loss,
        "drawdown": max_drawdown,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "buy": len(buy_ops),
        "buy_tp": buy_tp,
        "buy_sl": buy_sl,
        "sell": len(sell_ops),
        "sell_tp": sell_tp,
        "sell_sl": sell_sl
    }


# ============================================================
# STAMPA RISULTATI
# ============================================================

def stampa_risultati(codice, nome, metriche):

    print()
    print("=" * 65)

    print(f"📌 STRATEGIA {codice}")
    print(nome)

    print("-" * 65)

    print(
        f"Operazioni: {metriche['totale']}"
    )

    print(
        f"TP: {metriche['tp']}"
    )

    print(
        f"SL: {metriche['sl']}"
    )

    print(
        f"Win rate: {metriche['winrate']:.2f}%"
    )

    print(
        f"Risultato prezzo: "
        f"{metriche['profitto']:.2f}"
    )

    print(
        f"Profit Factor: "
        f"{metriche['profit_factor']:.2f}"
    )

    print(
        f"Media vincita: "
        f"{metriche['media_win']:.2f}"
    )

    print(
        f"Media perdita: "
        f"{metriche['media_loss']:.2f}"
    )

    print(
        f"Drawdown massimo: "
        f"{metriche['drawdown']:.2f}"
    )

    print(
        f"Max serie vittorie: "
        f"{metriche['max_win_streak']}"
    )

    print(
        f"Max serie perdite: "
        f"{metriche['max_loss_streak']}"
    )

    print()
    print("📈 BUY")

    print(
        f"Operazioni: "
        f"{metriche['buy']}"
    )

    print(
        f"TP: "
        f"{metriche['buy_tp']}"
    )

    print(
        f"SL: "
        f"{metriche['buy_sl']}"
    )

    print()
    print("📉 SELL")

    print(
        f"Operazioni: "
        f"{metriche['sell']}"
    )

    print(
        f"TP: "
        f"{metriche['sell_tp']}"
    )

    print(
        f"SL: "
        f"{metriche['sell_sl']}"
    )


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 65)
print("🤖 ANALISI STRATEGIE XAU/USD — PAPER/DEMO")
print("=" * 65)

df = scarica_dati()

print()
print(
    f"📊 Candele analizzate: {len(df)}"
)

print(
    f"📅 Da: {df.iloc[0]['datetime']}"
)

print(
    f"📅 A: {df.iloc[-1]['datetime']}"
)

df = calcola_indicatori(df)

df = aggiungi_trend_15m(df)

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

    metriche = calcola_metriche(
        operazioni
    )

    stampa_risultati(
        codice,
        nome,
        metriche
    )

print()
print("=" * 65)
print("✅ ANALISI TERMINATA")
print("=" * 65)

print()
print(
    "⚠️ Simulazione storica: "
    "i risultati passati non garantiscono risultati futuri."
)
