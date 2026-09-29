# ============================================================
# BACKTEST
# ============================================================

def esegui_backtest(df, strategia):

    operazioni = []
    posizione = None

    # --------------------------------------------------------
    # CONTATORI DIAGNOSTICI
    # --------------------------------------------------------

    buy = {
        "ema": 0,
        "macd": 0,
        "rsi": 0,
        "trend": 0,
        "ema_macd": 0,
        "ema_macd_rsi": 0,
        "finale": 0
    }

    sell = {
        "ema": 0,
        "macd": 0,
        "rsi": 0,
        "trend": 0,
        "ema_macd": 0,
        "ema_macd_rsi": 0,
        "finale": 0
    }

    for i in range(1, len(df) - 1):

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
                    profitto = exit_price - entry
                else:
                    profitto = entry - exit_price

                operazioni.append({
                    "tipo": posizione["tipo"],
                    "risultato": risultato,
                    "profitto": profitto,
                    "datetime": posizione["datetime"]
                })

                posizione = None

            continue

        # ----------------------------------------------------
        # DATI BASE
        # ----------------------------------------------------

        if pd.isna(row["atr"]):
            continue

        trend = row["trend_15m"]

        ema_bull = row["ema20"] > row["ema50"]
        ema_bear = row["ema20"] < row["ema50"]

        macd_bull = row["macd"] > row["macd_signal"]
        macd_bear = row["macd"] < row["macd_signal"]

        # ----------------------------------------------------
        # STRATEGIE A / B
        # ----------------------------------------------------

        if strategia == "A":

            rsi_ok = 30 < row["rsi"] < 65

            if ema_bull:
                buy["ema"] += 1

            if macd_bull:
                buy["macd"] += 1

            if rsi_ok:
                buy["rsi"] += 1

            if trend == "BULLISH":
                buy["trend"] += 1

            if ema_bull and macd_bull:
                buy["ema_macd"] += 1

            if ema_bull and macd_bull and rsi_ok:
                buy["ema_macd_rsi"] += 1

            if (
                ema_bull
                and macd_bull
                and rsi_ok
                and trend == "BULLISH"
            ):
                buy["finale"] += 1

            # SELL

            if ema_bear:
                sell["ema"] += 1

            if macd_bear:
                sell["macd"] += 1

            if trend == "BEARISH":
                sell["trend"] += 1

            if ema_bear and macd_bear:
                sell["ema_macd"] += 1

            if ema_bear and macd_bear and rsi_ok:
                sell["ema_macd_rsi"] += 1

            if (
                ema_bear
                and macd_bear
                and rsi_ok
                and trend == "BEARISH"
            ):
                sell["finale"] += 1

        # ----------------------------------------------------
        # STRATEGIA B
        # ----------------------------------------------------

        elif strategia == "B":

            rsi_ok = 30 < row["rsi"] < 70

            if ema_bull:
                buy["ema"] += 1

            if macd_bull:
                buy["macd"] += 1

            if rsi_ok:
                buy["rsi"] += 1

            if trend == "BULLISH":
                buy["trend"] += 1

            if ema_bull and macd_bull:
                buy["ema_macd"] += 1

            if ema_bull and macd_bull and rsi_ok:
                buy["ema_macd_rsi"] += 1

            if (
                ema_bull
                and macd_bull
                and rsi_ok
                and trend == "BULLISH"
            ):
                buy["finale"] += 1

            # SELL

            if ema_bear:
                sell["ema"] += 1

            if macd_bear:
                sell["macd"] += 1

            if trend == "BEARISH":
                sell["trend"] += 1

            if ema_bear and macd_bear:
                sell["ema_macd"] += 1

            if ema_bear and macd_bear and rsi_ok:
                sell["ema_macd_rsi"] += 1

            if (
                ema_bear
                and macd_bear
                and rsi_ok
                and trend == "BEARISH"
            ):
                sell["finale"] += 1

        # ----------------------------------------------------
        # STRATEGIA C
        # ----------------------------------------------------

        elif strategia == "C":

            bullish_cross = (
                row["macd_prev"] <= row["signal_prev"]
                and
                row["macd"] > row["macd_signal"]
            )

            bearish_cross = (
                row["macd_prev"] >= row["signal_prev"]
                and
                row["macd"] < row["macd_signal"]
            )

            rsi_ok = 30 < row["rsi"] < 70

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
            continue

        # ----------------------------------------------------
        # SEGNALE REALE
        # ----------------------------------------------------

        segnale = genera_segnale(
            row,
            strategia
        )

        if segnale is None:
            continue

        entry = df.iloc[i + 1]["open"]
        atr = row["atr"]

        if segnale == "BUY":

            sl = entry - (1.5 * atr)
            tp = entry + (2.0 * atr)

        else:

            sl = entry + (1.5 * atr)
            tp = entry - (2.0 * atr)

        posizione = {
            "tipo": segnale,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "datetime": df.iloc[i + 1]["datetime"]
        }

    # --------------------------------------------------------
    # DIAGNOSTICA
    # --------------------------------------------------------

    print()
    print("🔎 DIAGNOSTICA — STRATEGIA", strategia)
    print("=" * 60)

    print("BUY")
    print("-" * 60)
    print("EMA:", buy["ema"])
    print("MACD:", buy["macd"])
    print("RSI:", buy["rsi"])
    print("TREND BULLISH:", buy["trend"])
    print("EMA + MACD:", buy["ema_macd"])
    print("EMA + MACD + RSI:", buy["ema_macd_rsi"])
    print("SEGNALE BUY FINALE:", buy["finale"])

    print()
    print("SELL")
    print("-" * 60)
    print("EMA:", sell["ema"])
    print("MACD:", sell["macd"])
    print("TREND BEARISH:", sell["trend"])
    print("EMA + MACD:", sell["ema_macd"])
    print("EMA + MACD + RSI:", sell["ema_macd_rsi"])
    print("SEGNALE SELL FINALE:", sell["finale"])

    print("=" * 60)

    return operazioni
