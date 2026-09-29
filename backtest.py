import os
import requests
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "TWELVE_DATA_API_KEY non configurata"
    )


# ============================================================
# SCARICA DATI
# ============================================================

def scarica_dati():

    url = "https://api.twelvedata.com/time_series"

    def richiesta(params):

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        data = response.json()

        if data.get("status") == "error":
            raise RuntimeError(
                f"Errore Twelve Data: {data}"
            )

        if "values" not in data:
            raise RuntimeError(
                f"Risposta Twelve Data inattesa: {data}"
            )

        return data["values"]

    # ========================================================
    # BLOCCO 1
    # ========================================================

    params1 = {
        "symbol": "XAU/USD",
        "interval": "5min",
        "outputsize": 5000,
        "apikey": API_KEY,
        "format": "JSON",
        "timezone": "UTC",
        "order": "desc"
    }

    valori1 = richiesta(params1)

    df1 = pd.DataFrame(valori1)

    df1["datetime"] = pd.to_datetime(
        df1["datetime"],
        utc=True
    )

    df1 = df1[
        [
            "datetime",
            "open",
            "high",
            "low",
            "close"
        ]
    ].copy()

    for col in [
        "open",
        "high",
        "low",
        "close"
    ]:
        df1[col] = pd.to_numeric(
            df1[col],
            errors="coerce"
        )

    df1 = df1.dropna()

    data_piu_vecchia = df1["datetime"].min()

    # ========================================================
    # BLOCCO 2
    # ========================================================

    data_fine_secondo_blocco = (
        data_piu_vecchia -
        pd.Timedelta(minutes=5)
    )

    params2 = {
        "symbol": "XAU/USD",
        "interval": "5min",
        "outputsize": 5000,
        "apikey": API_KEY,
        "format": "JSON",
        "timezone": "UTC",
        "end_date": data_fine_secondo_blocco.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "order": "desc"
    }

    valori2 = richiesta(params2)

    df2 = pd.DataFrame(valori2)

    df2["datetime"] = pd.to_datetime(
        df2["datetime"],
        utc=True
    )

    df2 = df2[
        [
            "datetime",
            "open",
            "high",
            "low",
            "close"
        ]
    ].copy()

    for col in [
        "open",
        "high",
        "low",
        "close"
    ]:
        df2[col] = pd.to_numeric(
            df2[col],
            errors="coerce"
        )

    df2 = df2.dropna()

    # ========================================================
    # UNIONE
    # ========================================================

    df = pd.concat(
        [df1, df2],
        ignore_index=True
    )

    df = df.drop_duplicates(
        subset=["datetime"]
    )

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    print(
        f"Candele scaricate: {len(df)}"
    )

    print(
        f"Da: {df['datetime'].iloc[0]}"
    )

    print(
        f"A: {df['datetime'].iloc[-1]}"
    )

    return df


# ============================================================
# INDICATORI
# ============================================================

def calcola_indicatori(df):

    df = df.copy()

    # EMA 20
    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    # EMA 50
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
    high_low = (
        df["high"] -
        df["low"]
    )

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

    df["signal_prev"] = (
        df["macd_signal"].shift(1)
    )

    return df


# ============================================================
# TENDENZA 15 MINUTI
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

        timestamp = result.loc[
            i,
            "datetime"
        ]

        periodo = timestamp.floor("15min")

        periodi_precedenti = df15.index[
            df15.index < periodo
        ]

        if len(periodi_precedenti) == 0:
            continue

        ultimo_periodo = (
            periodi_precedenti[-1]
        )

        result.loc[
            i,
            "trend_15m"
        ] = df15.loc[
            ultimo_periodo,
            "trend"
        ]

    return result


def genera_segnale(row, strategia):

    # ========================================================
    # DATI NECESSARI
    # ========================================================

    if pd.isna(row["atr"]):
        return None

    ema_bull = (
        row["ema20"] >
        row["ema50"]
    )

    ema_bear = (
        row["ema20"] <
        row["ema50"]
    )

    macd_bull = (
        row["macd"] >
        row["macd_signal"]
    )

    macd_bear = (
        row["macd"] <
        row["macd_signal"]
    )

    trend_bull = (
        row["trend_15m"] == "BULLISH"
    )

    trend_bear = (
        row["trend_15m"] == "BEARISH"
    )

    # ========================================================
    # STRATEGIA A
    # RSI 30-65
    # ========================================================

    if strategia == "A":

        rsi_ok = (
            30 <
            row["rsi"] <
            65
        )

        if (
            ema_bull
            and macd_bull
            and rsi_ok
            and trend_bull
        ):
            return "BUY"

        if (
            ema_bear
            and macd_bear
            and rsi_ok
            and trend_bear
        ):
            return "SELL"

        return None

    # ========================================================
    # STRATEGIA B
    # RSI 30-70
    # ========================================================

    if strategia == "B":

        rsi_ok = (
            30 <
            row["rsi"] <
            70
        )

        if (
            ema_bull
            and macd_bull
            and rsi_ok
            and trend_bull
        ):
            return "BUY"

        if (
            ema_bear
            and macd_bear
            and rsi_ok
            and trend_bear
        ):
            return "SELL"

        return None

    # ========================================================
    # STRATEGIA C
    # INCROCIO MACD
    # ========================================================

    if strategia == "C":

        if (
            pd.isna(row["macd_prev"])
            or
            pd.isna(row["signal_prev"])
        ):
            return None

        rsi_ok = (
            30 <
            row["rsi"] <
            70
        )

        bullish_cross = (
            row["macd_prev"]
            <=
            row["signal_prev"]
            and
            row["macd"]
            >
            row["macd_signal"]
        )

        bearish_cross = (
            row["macd_prev"]
            >=
            row["signal_prev"]
            and
            row["macd"]
            <
            row["macd_signal"]
        )

        if (
            ema_bull
            and
            bullish_cross
            and
            rsi_ok
            and
            trend_bull
        ):
            return "BUY"

        if (
            ema_bear
            and
            bearish_cross
            and
            rsi_ok
            and
            trend_bear
        ):
            return "SELL"

        return None

    # ========================================================
    # STRATEGIA NON RICONOSCIUTA
    # ========================================================

    return None

# ============================================================
# BACKTEST
# ============================================================

def esegui_backtest(df, strategia):

    operazioni = []

    posizione = None

    # ========================================================
    # DIAGNOSTICA BUY
    # ========================================================

    buy = {
        "ema": 0,
        "macd": 0,
        "rsi": 0,
        "trend": 0,
        "ema_macd": 0,
        "ema_macd_rsi": 0,
        "finale": 0
    }

    # ========================================================
    # DIAGNOSTICA SELL
    # ========================================================

    sell = {
        "ema": 0,
        "macd": 0,
        "rsi": 0,
        "trend": 0,
        "ema_macd": 0,
        "ema_macd_rsi": 0,
        "finale": 0
    }

    for i in range(
        1,
        len(df) - 1
    ):

        row = df.iloc[i]

        # ----------------------------------------------------
        # POSIZIONE APERTA
        # ----------------------------------------------------

        if posizione is not None:

            high = row["high"]
            low = row["low"]

            entry = posizione["entry"]
            sl = posizione["sl"]
            tp = posizione["tp"]

            risultato = None
            exit_price = None

            if posizione["tipo"] == "BUY":

                if low <= sl:

                    risultato = "SL"
                    exit_price = sl

                elif high >= tp:

                    risultato = "TP"
                    exit_price = tp

            else:

                if high >= sl:

                    risultato = "SL"
                    exit_price = sl

                elif low <= tp:

                    risultato = "TP"
                    exit_price = tp

            if risultato is not None:

                if posizione["tipo"] == "BUY":

                    profitto = (
                        exit_price -
                        entry
                    )

                else:

                    profitto = (
                        entry -
                        exit_price
                    )

                operazioni.append({
                    "tipo": posizione["tipo"],
                    "risultato": risultato,
                    "profitto": profitto,
                    "datetime": posizione["datetime"]
                })

                posizione = None

            continue

        # ----------------------------------------------------
        # DATI NON VALIDI
        # ----------------------------------------------------

        if pd.isna(row["atr"]):
            continue

        trend = row["trend_15m"]

        ema_bull = (
            row["ema20"] >
            row["ema50"]
        )

        ema_bear = (
            row["ema20"] <
            row["ema50"]
        )

        macd_bull = (
            row["macd"] >
            row["macd_signal"]
        )

        macd_bear = (
            row["macd"] <
            row["macd_signal"]
        )

        # ----------------------------------------------------
        # DIAGNOSTICA
        # ----------------------------------------------------

        if strategia == "A":

            rsi_ok = (
                30 <
                row["rsi"] <
                65
            )

        elif strategia == "B":

            rsi_ok = (
                30 <
                row["rsi"] <
                70
            )

        else:

            rsi_ok = (
                30 <
                row["rsi"] <
                70
            )

        # ====================================================
        # BUY DIAGNOSTICA
        # ====================================================

        if ema_bull:
            buy["ema"] += 1

        if macd_bull:
            buy["macd"] += 1

        if rsi_ok:
            buy["rsi"] += 1

        if trend == "BULLISH":
            buy["trend"] += 1

        if (
            ema_bull
            and
            macd_bull
        ):
            buy["ema_macd"] += 1

        if (
            ema_bull
            and
            macd_bull
            and
            rsi_ok
        ):
            buy["ema_macd_rsi"] += 1

        # ====================================================
        # SELL DIAGNOSTICA
        # ====================================================

        if ema_bear:
            sell["ema"] += 1

        if macd_bear:
            sell["macd"] += 1

        if trend == "BEARISH":
            sell["trend"] += 1

        if (
            ema_bear
            and
            macd_bear
        ):
            sell["ema_macd"] += 1

        if (
            ema_bear
            and
            macd_bear
            and
            rsi_ok
        ):
            sell["ema_macd_rsi"] += 1

        # ====================================================
        # STRATEGIA C
        # ====================================================

        if strategia == "C":

            bullish_cross = (
                row["macd_prev"]
                <=
                row["signal_prev"]
                and
                row["macd"]
                >
                row["macd_signal"]
            )

            bearish_cross = (
                row["macd_prev"]
                >=
                row["signal_prev"]
                and
                row["macd"]
                <
                row["macd_signal"]
            )

            if (
                ema_bull
                and
                bullish_cross
                and
                rsi_ok
                and
                trend == "BULLISH"
            ):
                buy["finale"] += 1

            if (
                ema_bear
                and
                bearish_cross
                and
                rsi_ok
                and
                trend == "BEARISH"
            ):
                sell["finale"] += 1

        else:

            # =================================================
            # BUY FINALE A/B
            # =================================================

            if strategia == "A":

                buy_finale = (
                    ema_bull
                    and
                    macd_bull
                    and
                    30 < row["rsi"] < 65
                    and
                    trend == "BULLISH"
                )

                sell_finale = (
                    ema_bear
                    and
                    macd_bear
                    and
                    30 < row["rsi"] < 65
                    and
                    trend == "BEARISH"
                )

            else:

                buy_finale = (
                    ema_bull
                    and
                    macd_bull
                    and
                    30 < row["rsi"] < 70
                    and
                    trend == "BULLISH"
                )

                sell_finale = (
                    ema_bear
                    and
                    macd_bear
                    and
                    30 < row["rsi"] < 70
                    and
                    trend == "BEARISH"
                )

            if buy_finale:
                buy["finale"] += 1

            if sell_finale:
                sell["finale"] += 1

        # ----------------------------------------------------
        # SEGNALE REALE
        # ----------------------------------------------------

        segnale = genera_segnale(
            row,
            strategia
        )

        if segnale is None:
            continue

        entry = df.iloc[
            i + 1
        ]["open"]

        atr = row["atr"]

        if segnale == "BUY":

            sl = (
                entry -
                (1.5 * atr)
            )

            tp = (
                entry +
                (2.0 * atr)
            )

        else:

            sl = (
                entry +
                (1.5 * atr)
            )

            tp = (
                entry -
                (2.0 * atr)
            )

        posizione = {
            "tipo": segnale,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "datetime": df.iloc[
                i + 1
            ]["datetime"]
        }

    # ========================================================
    # STAMPA DIAGNOSTICA
    # ========================================================

    print()
    print(
        "🔎 DIAGNOSTICA — STRATEGIA",
        strategia
    )
    print("=" * 60)

    print("BUY")
    print("-" * 60)

    print(
        "EMA20 > EMA50:",
        buy["ema"]
    )

    print(
        "MACD > SIGNAL:",
        buy["macd"]
    )

    print(
        "RSI OK:",
        buy["rsi"]
    )

    print(
        "TREND BULLISH:",
        buy["trend"]
    )

    print(
        "EMA + MACD:",
        buy["ema_macd"]
    )

    print(
        "EMA + MACD + RSI:",
        buy["ema_macd_rsi"]
    )

    print(
        "SEGNALE BUY FINALE:",
        buy["finale"]
    )

    print()
    print("SELL")
    print("-" * 60)

    print(
        "EMA20 < EMA50:",
        sell["ema"]
    )

    print(
        "MACD < SIGNAL:",
        sell["macd"]
    )

    print(
        "TREND BEARISH:",
        sell["trend"]
    )

    print(
        "EMA + MACD:",
        sell["ema_macd"]
    )

    print(
        "EMA + MACD + RSI:",
        sell["ema_macd_rsi"]
    )

    print(
        "SEGNALE SELL FINALE:",
        sell["finale"]
    )

    print("=" * 60)

# ============================================================
# CALCOLO METRICHE
# ============================================================

def calcola_metriche(operazioni):

    if not operazioni:
        return {
            "operazioni": 0,
            "tp": 0,
            "sl": 0,
            "win_rate": 0,
            "risultato": 0,
            "profit_factor": 0,
            "media_vincita": 0,
            "media_perdita": 0,
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

    # ========================================================
    # RISULTATI
    # ========================================================

    risultati = [
        op.get("risultato")
        for op in operazioni
    ]

    tp = risultati.count("TP")
    sl = risultati.count("SL")

    totale = len(operazioni)

    win_rate = (
        tp / totale * 100
        if totale > 0
        else 0
    )

    # ========================================================
    # PROFITTI
    # ========================================================

    profitti = []

    for op in operazioni:

        profitto = op.get("profitto", 0)

        # Se per qualsiasi motivo il valore non è numerico,
        # non lo usiamo come profitto.
        try:
            profitto = float(profitto)
        except (TypeError, ValueError):
            profitto = 0.0

        profitti.append(profitto)

    risultato_totale = sum(profitti)

    # ========================================================
    # VINCITE / PERDITE
    # ========================================================

    vincite = [
        x for x in profitti
        if x > 0
    ]

    perdite = [
        x for x in profitti
        if x < 0
    ]

    totale_vincite = sum(vincite)

    totale_perdite = abs(
        sum(perdite)
    )

    if totale_perdite > 0:

        profit_factor = (
            totale_vincite /
            totale_perdite
        )

    else:

        profit_factor = 0

    media_vincita = (
        np.mean(vincite)
        if vincite
        else 0
    )

    media_perdita = (
        np.mean(perdite)
        if perdite
        else 0
    )

    # ========================================================
    # DRAWDOWN
    # ========================================================

    equity = 0
    massimo = 0
    max_drawdown = 0

    for profitto in profitti:

        equity += profitto

        if equity > massimo:
            massimo = equity

        drawdown = massimo - equity

        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # ========================================================
    # STREAK
    # ========================================================

    max_win_streak = 0
    max_loss_streak = 0

    win_streak = 0
    loss_streak = 0

    for risultato_operazione in risultati:

        if risultato_operazione == "TP":

            win_streak += 1
            loss_streak = 0

        elif risultato_operazione == "SL":

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

    # ========================================================
    # BUY / SELL
    # ========================================================

    buy_ops = [
        op for op in operazioni
        if op.get("tipo") == "BUY"
    ]

    sell_ops = [
        op for op in operazioni
        if op.get("tipo") == "SELL"
    ]

    buy_tp = sum(
        1
        for op in buy_ops
        if op.get("risultato") == "TP"
    )

    buy_sl = sum(
        1
        for op in buy_ops
        if op.get("risultato") == "SL"
    )

    sell_tp = sum(
        1
        for op in sell_ops
        if op.get("risultato") == "TP"
    )

    sell_sl = sum(
        1
        for op in sell_ops
        if op.get("risultato") == "SL"
    )

    # ========================================================
    # RISULTATO FINALE
    # ========================================================

    return {
        "operazioni": totale,
        "tp": tp,
        "sl": sl,
        "win_rate": float(win_rate),
        "risultato": float(risultato_totale),
        "profit_factor": float(profit_factor),
        "media_vincita": float(media_vincita),
        "media_perdita": float(media_perdita),
        "drawdown": float(max_drawdown),
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
# STAMPA METRICHE
# ============================================================

def stampa_metriche(nome, descrizione, metriche):

    print()
    print("=" * 65)
    print("📌", nome)
    print(descrizione)
    print("-" * 65)

    print(
        f"Operazioni: {metriche['operazioni']}"
    )

    print(
        f"TP: {metriche['tp']}"
    )

    print(
        f"SL: {metriche['sl']}"
    )

    print(
        f"Win rate: {float(metriche['win_rate']):.2f}%"
    )

    print(
        f"Risultato prezzo: "
        f"{float(metriche['risultato']):.2f}"
    )

    print(
        f"Profit Factor: "
        f"{float(metriche['profit_factor']):.2f}"
    )

    print(
        f"Media vincita: "
        f"{float(metriche['media_vincita']):.2f}"
    )

    print(
        f"Media perdita: "
        f"{float(metriche['media_perdita']):.2f}"
    )

    print(
        f"Drawdown massimo: "
        f"{float(metriche['drawdown']):.2f}"
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
# ESECUZIONE
# ============================================================

def esegui_fase(
    df,
    nome_fase,
    percentuale
):

    print()
    print("#" * 65)
    print(
        f"📊 {nome_fase}"
    )
    print("#" * 65)

    print(
        f"Candele: {len(df)}"
    )

    print(
        f"Da: {df['datetime'].iloc[0]}"
    )

    print(
        f"A: {df['datetime'].iloc[-1]}"
    )

    strategie = [
        (
            "A",
            "Strategia attuale — RSI 30-65"
        ),
        (
            "B",
            "RSI ampliato — RSI 30-70"
        ),
        (
            "C",
            "Incrocio MACD — RSI 30-70"
        )
    ]

    for strategia, descrizione in strategie:

        operazioni = esegui_backtest(
            df,
            strategia
        )

        metriche = calcola_metriche(
            operazioni
        )

        stampa_metriche(
            f"STRATEGIA {strategia}",
            descrizione,
            metriche
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("XAU/USD — BACKTEST PAPER/DEMO")
    print(
        "📊 TEST SVILUPPO 70% / VERIFICA 30%"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # SCARICA
    # --------------------------------------------------------

    df = scarica_dati()

    print()
    print(
        f"📊 Candele totali scaricate: "
        f"{len(df)}"
    )

    print(
        f"📅 Periodo completo: "
        f"{df['datetime'].iloc[0]}"
        f" → "
        f"{df['datetime'].iloc[-1]}"
    )

    # --------------------------------------------------------
    # INDICATORI
    # --------------------------------------------------------

    df = calcola_indicatori(df)

    # --------------------------------------------------------
    # TREND 15M
    # --------------------------------------------------------

    df = aggiungi_trend_15m(df)

    # --------------------------------------------------------
    # SPLIT 70 / 30
    # --------------------------------------------------------

    split_index = int(
        len(df) * 0.70
    )

    df_sviluppo = df.iloc[
        :split_index
    ].copy()

    df_verifica = df.iloc[
        split_index:
    ].copy()

    # --------------------------------------------------------
    # FASE 1
    # --------------------------------------------------------

    esegui_fase(
        df_sviluppo,
        "FASE 1 — SVILUPPO 70%",
        70
    )

    # --------------------------------------------------------
    # FASE 2
    # --------------------------------------------------------

    esegui_fase(
        df_verifica,
        "FASE 2 — VERIFICA 30%",
        30
    )

    # --------------------------------------------------------
    # FINE
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("✅ BACKTEST TERMINATO")
    print("=" * 65)


# ============================================================
# AVVIO PROGRAMMA
# ============================================================

if __name__ == "__main__":
    main()
