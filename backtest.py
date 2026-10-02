import os
import time
import warnings
import traceback
from itertools import product

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAZIONE
# ============================================================

SYMBOL = "XAU/USD"
INTERVAL = "5min"

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError("TWELVE_DATA_API_KEY non configurato")

OUTPUT_DIR = "backtest_results_v6"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CANDLES = 10000

DEV_FRAC = 0.70

TP_VALUES = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
]

SL_VALUES = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
]

HORIZONS = [
    15,
    30,
    60,
]

MIN_TRADES_DEV = 20
MIN_TRADES_VER = 10


# ============================================================
# DOWNLOAD DATI
# ============================================================

def get_data():

    url = "https://api.twelvedata.com/time_series"

    frames = []

    end = None
    remaining = N_CANDLES

    while remaining > 0:

        size = min(5000, remaining)

        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "outputsize": size,
            "apikey": API_KEY,
            "timezone": "UTC",
            "order": "ASC",
        }

        if end:
            params["end_date"] = end

        response = requests.get(
            url,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        if "values" not in data:
            raise RuntimeError(
                f"Twelve Data ha restituito: {data}"
            )

        df = pd.DataFrame(data["values"])

        frames.append(df)

        if len(df) < size:
            break

        end = str(df["datetime"].min())

        remaining -= len(df)

        time.sleep(0.2)

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = df.drop_duplicates(
        subset=["datetime"]
    )

    df = df.tail(N_CANDLES)

    for column in [
        "open",
        "high",
        "low",
        "close",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
    )

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return df


# ============================================================
# INDICATORI
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False,
    ).mean()


def rsi(series, period=14):

    delta = series.diff()

    gains = delta.clip(
        lower=0
    )

    losses = -delta.clip(
        upper=0
    )

    avg_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    return 100 - (
        100 / (1 + rs)
    )


def atr(df, period=14):

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],

            (
                df["high"] -
                previous_close
            ).abs(),

            (
                df["low"] -
                previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()


def macd(series):

    macd_line = (
        ema(series, 12) -
        ema(series, 26)
    )

    signal_line = ema(
        macd_line,
        9,
    )

    return macd_line, signal_line


# ============================================================
# AGGIUNTA INDICATORI
# ============================================================

def add_indicators(df):

    df = df.copy()

    df["ema20"] = ema(
        df["close"],
        20,
    )

    df["ema50"] = ema(
        df["close"],
        50,
    )

    df["ema100"] = ema(
        df["close"],
        100,
    )

    df["rsi"] = rsi(
        df["close"],
        14,
    )

    df["atr"] = atr(
        df,
        14,
    )

    df["atr_pct"] = (
        df["atr"] /
        df["close"] *
        100
    )

    (
        df["macd"],
        df["macd_sig"],
    ) = macd(
        df["close"]
    )

    df["mom6"] = (
        df["close"].pct_change(6)
        * 100
    )

    df["mom12"] = (
        df["close"].pct_change(12)
        * 100
    )

    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["range"] = (
        df["high"] -
        df["low"]
    )

    df["body_ratio"] = (
        df["body"] /
        df["range"].replace(
            0,
            np.nan,
        )
    )

    df["ema_gap"] = (
        (
            df["ema20"] -
            df["ema50"]
        ).abs()
        /
        df["atr"].replace(
            0,
            np.nan,
        )
    )

    return df


# ============================================================
# TIMEFRAME SUPERIORI
# ============================================================

def make_timeframe(df, rule):

    x = (
        df.set_index("datetime")
        [
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        .resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }
        )
        .dropna()
        .reset_index()
    )

    x["ema20"] = ema(
        x["close"],
        20,
    )

    x["ema50"] = ema(
        x["close"],
        50,
    )

    x["trend"] = np.where(
        x["ema20"] >
        x["ema50"],
        1,
        -1,
    )

    return x


def merge_trends(df):

    result = df.copy()

    trend15 = make_timeframe(
        result,
        "15min",
    )

    trend15 = trend15[
        [
            "datetime",
            "trend",
        ]
    ].rename(
        columns={
            "trend": "trend15"
        }
    )

    trend1h = make_timeframe(
        result,
        "1h",
    )

    trend1h = trend1h[
        [
            "datetime",
            "trend",
        ]
    ].rename(
        columns={
            "trend": "trend1h"
        }
    )

    result = pd.merge_asof(
        result.sort_values(
            "datetime"
        ),
        trend15.sort_values(
            "datetime"
        ),
        on="datetime",
        direction="backward",
    )

    result = pd.merge_asof(
        result.sort_values(
            "datetime"
        ),
        trend1h.sort_values(
            "datetime"
        ),
        on="datetime",
        direction="backward",
    )

    return result


# ============================================================
# GENERAZIONE SEGNALI
# ============================================================

def signal_mask(
    df,
    direction,
    mode,
):

    bullish = (
        (df["ema20"] > df["ema50"])
        &
        (df["ema50"] > df["ema100"])
        &
        (df["trend15"] == 1)
    )

    bearish = (
        (df["ema20"] < df["ema50"])
        &
        (df["ema50"] < df["ema100"])
        &
        (df["trend15"] == -1)
    )

    if direction == "BUY":

        mask = bullish.copy()

        mask &= (
            df["macd"] >
            df["macd_sig"]
        )

        mask &= df["rsi"].between(
            30,
            65,
        )

        mask &= (
            df["mom6"] > 0
        )

    else:

        mask = bearish.copy()

        mask &= (
            df["macd"] <
            df["macd_sig"]
        )

        mask &= df["rsi"].between(
            30,
            65,
        )

        mask &= (
            df["mom6"] < 0
        )

    # --------------------------------------------------------
    # FILTRI
    # --------------------------------------------------------

    if mode == "EMA_STRONG":

        mask &= (
            df["ema_gap"] >= 0.5
        )

    elif mode == "EMA_VERY_STRONG":

        mask &= (
            df["ema_gap"] >= 1.0
        )

    elif mode == "ATR_HIGH":

        median_atr = (
            df["atr_pct"]
            .rolling(200)
            .median()
        )

        mask &= (
            df["atr_pct"] >=
            median_atr
        )

    elif mode == "ATR_NORMAL":

        median_atr = (
            df["atr_pct"]
            .rolling(200)
            .median()
        )

        mask &= (
            df["atr_pct"] <
            median_atr
        )

    elif mode == "BODY":

        mask &= (
            df["body_ratio"] >= 0.45
        )

    elif mode == "RSI_MID":

        mask &= df["rsi"].between(
            40,
            60,
        )

    elif mode == "TREND_1H":

        required_trend = (
            1
            if direction == "BUY"
            else -1
        )

        mask &= (
            df["trend1h"] ==
            required_trend
        )

    elif mode == "EMA_ATR":

        median_atr = (
            df["atr_pct"]
            .rolling(200)
            .median()
        )

        mask &= (
            df["ema_gap"] >= 0.5
        )

        mask &= (
            df["atr_pct"] <
            median_atr
        )

    return mask.fillna(False)


# ============================================================
# PRIMO SEGNALE DI OGNI EPISODIO
# ============================================================

def first_in_episode(mask):

    previous = mask.shift(
        1,
        fill_value=False,
    )

    return (
        mask &
        ~previous
    )


# ============================================================
# SIMULAZIONE TRADE
# ============================================================

def simulate(
    df,
    signals,
    direction,
    tp,
    sl,
    horizon,
):

    trades = []

    signal_indexes = np.flatnonzero(
        signals.to_numpy()
    )

    for signal_index in signal_indexes:

        if signal_index + 1 >= len(df):
            continue

        entry_index = (
            signal_index + 1
        )

        entry = float(
            df.iloc[
                entry_index
            ]["open"]
        )

        atr_value = float(
            df.iloc[
                signal_index
            ]["atr"]
        )

        if (
            not np.isfinite(
                atr_value
            )
            or atr_value <= 0
        ):
            continue

        if direction == "BUY":

            tp_price = (
                entry +
                tp * atr_value
            )

            sl_price = (
                entry -
                sl * atr_value
            )

        else:

            tp_price = (
                entry -
                tp * atr_value
            )

            sl_price = (
                entry +
                sl * atr_value
            )

        candles_horizon = max(
            1,
            int(horizon / 5),
        )

        end_index = min(
            len(df) - 1,
            entry_index +
            candles_horizon,
        )

        result = "TIMEOUT"

        exit_price = float(
            df.iloc[
                end_index
            ]["close"]
        )

        exit_index = end_index

        for j in range(
            entry_index,
            end_index + 1,
        ):

            high = float(
                df.iloc[j]["high"]
            )

            low = float(
                df.iloc[j]["low"]
            )

            if direction == "BUY":

                hit_tp = (
                    high >= tp_price
                )

                hit_sl = (
                    low <= sl_price
                )

            else:

                hit_tp = (
                    low <= tp_price
                )

                hit_sl = (
                    high >= sl_price
                )

            # ------------------------------------------------
            # Se TP e SL vengono toccati
            # nella stessa candela:
            # scelta conservativa = SL
            # ------------------------------------------------

            if hit_tp and hit_sl:

                result = "SL"

                exit_price = (
                    sl_price
                )

                exit_index = j

                break

            if hit_tp:

                result = "TP"

                exit_price = (
                    tp_price
                )

                exit_index = j

                break

            if hit_sl:

                result = "SL"

                exit_price = (
                    sl_price
                )

                exit_index = j

                break

        # ----------------------------------------------------
        # CALCOLO R
        # ----------------------------------------------------

        if result == "TP":

            rr = tp / sl

        elif result == "SL":

            rr = -1.0

        else:

            if direction == "BUY":

                rr = (
                    exit_price -
                    entry
                ) / (
                    sl *
                    atr_value
                )

            else:

                rr = (
                    entry -
                    exit_price
                ) / (
                    sl *
                    atr_value
                )

        trades.append(
            {
                "signal_time":
                    df.iloc[
                        signal_index
                    ]["datetime"],

                "entry_time":
                    df.iloc[
                        entry_index
                    ]["datetime"],

                "exit_time":
                    df.iloc[
                        exit_index
                    ]["datetime"],

                "direction":
                    direction,

                "entry":
                    entry,

                "exit":
                    exit_price,

                "tp":
                    tp_price,

                "sl":
                    sl_price,

                "atr":
                    atr_value,

                "result":
                    result,

                "R":
                    rr,
            }
        )

    return pd.DataFrame(
        trades
    )


# ============================================================
# STATISTICHE
# ============================================================

def calculate_stats(trades):

    if trades.empty:

        return {
            "trades": 0,
            "win": np.nan,
            "avg_R": np.nan,
            "total_R": 0.0,
            "PF": np.nan,
            "DD": 0.0,
        }

    r = trades["R"].astype(
        float
    )

    wins = (
        trades["result"] ==
        "TP"
    ).sum()

    gross_profit = (
        r[r > 0].sum()
    )

    gross_loss = (
        -r[r < 0].sum()
    )

    if gross_loss > 0:

        pf = (
            gross_profit /
            gross_loss
        )

    else:

        pf = np.inf

    equity = r.cumsum()

    drawdown = (
        equity.cummax() -
        equity
    )

    max_dd = drawdown.max()

    return {
        "trades":
            len(trades),

        "win":
            wins /
            len(trades) *
            100,

        "avg_R":
            r.mean(),

        "total_R":
            r.sum(),

        "PF":
            pf,

        "DD":
            max_dd,
    }


# ============================================================
# SCORE ROBUSTEZZA
# ============================================================

def robustness_score(
    development,
    verification,
):

    if (
        development["trades"] <
        MIN_TRADES_DEV
    ):
        return -999

    if (
        verification["trades"] <
        MIN_TRADES_VER
    ):
        return -999

    if (
        not np.isfinite(
            development["PF"]
        )
        or
        not np.isfinite(
            verification["PF"]
        )
    ):
        return -999

    score = (

        0.45 *
        verification["total_R"]

        +

        0.30 *
        development["total_R"]

        +

        8.0 *
        min(
            verification["avg_R"],
            0,
        )

        +

        5.0 *
        min(
            development["avg_R"],
            0,
        )

        -

        0.10 *
        verification["DD"]

        -

        0.05 *
        development["DD"]
    )

    return score


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("BACKTEST V6 - ROBUSTNESS SCANNER")
    print("=" * 72)
    print()

    # --------------------------------------------------------
    # DATI
    # --------------------------------------------------------

    df = get_data()

    print(
        f"Candele scaricate: {len(df)}"
    )

    print(
        f"Periodo: "
        f"{df['datetime'].min()} "
        f"-> "
        f"{df['datetime'].max()}"
    )

    print()

    # --------------------------------------------------------
    # INDICATORI
    # --------------------------------------------------------

    df = add_indicators(
        df
    )

    df = merge_trends(
        df
    )

    df = df.dropna().reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # SPLIT DEVELOPMENT / VERIFICATION
    # --------------------------------------------------------

    split_index = int(
        len(df) *
        DEV_FRAC
    )

    development_end = (
        df.iloc[
            split_index - 1
        ]["datetime"]
    )

    print(
        "SVILUPPO:"
    )

    print(
        f"fino a {development_end}"
    )

    print()

    print(
        "VERIFICA:"
    )

    print(
        f"dopo {development_end}"
    )

    print()

    # --------------------------------------------------------
    # MODALITÀ
    # --------------------------------------------------------

    modes = [
        "BASE",
        "EMA_STRONG",
        "EMA_VERY_STRONG",
        "ATR_HIGH",
        "ATR_NORMAL",
        "BODY",
        "RSI_MID",
        "TREND_1H",
        "EMA_ATR",
    ]

    results = []

    all_trades = []

    # --------------------------------------------------------
    # SCANSIONE
    # --------------------------------------------------------

    combinations = product(
        [
            "BUY",
            "SELL",
        ],
        modes,
        TP_VALUES,
        SL_VALUES,
        HORIZONS,
    )

    for (
        direction,
        mode,
        tp,
        sl,
        horizon,
    ) in combinations:

        filter_mode = (
            "BASE"
            if mode == "BASE"
            else mode
        )

        signals = signal_mask(
            df,
            direction,
            filter_mode,
        )

        # Solo il primo segnale
        # di ogni episodio continuo.
        signals = first_in_episode(
            signals
        )

        if not signals.any():
            continue

        trades = simulate(
            df,
            signals,
            direction,
            tp,
            sl,
            horizon,
        )

        if trades.empty:
            continue

        trades["mode"] = mode

        trades["TP_ATR"] = tp

        trades["SL_ATR"] = sl

        trades["HORIZON_MIN"] = (
            horizon
        )

        trades["period"] = np.where(
            trades["entry_time"] <=
            development_end,
            "DEV",
            "VER",
        )

        development_trades = (
            trades[
                trades["period"] ==
                "DEV"
            ]
        )

        verification_trades = (
            trades[
                trades["period"] ==
                "VER"
            ]
        )

        development = (
            calculate_stats(
                development_trades
            )
        )

        verification = (
            calculate_stats(
                verification_trades
            )
        )

        score = robustness_score(
            development,
            verification,
        )

        results.append(
            {
                "direction":
                    direction,

                "mode":
                    mode,

                "TP_ATR":
                    tp,

                "SL_ATR":
                    sl,

                "HORIZON_MIN":
                    horizon,

                "DEV_TRADES":
                    development[
                        "trades"
                    ],

                "DEV_WIN":
                    development[
                        "win"
                    ],

                "DEV_avg_R":
                    development[
                        "avg_R"
                    ],

                "DEV_TOTAL_R":
                    development[
                        "total_R"
                    ],

                "DEV_PF":
                    development[
                        "PF"
                    ],

                "DEV_DD":
                    development[
                        "DD"
                    ],

                "VER_TRADES":
                    verification[
                        "trades"
                    ],

                "VER_WIN":
                    verification[
                        "win"
                    ],

                "VER_avg_R":
                    verification[
                        "avg_R"
                    ],

                "VER_TOTAL_R":
                    verification[
                        "total_R"
                    ],

                "VER_PF":
                    verification[
                        "PF"
                    ],

                "VER_DD":
                    verification[
                        "DD"
                    ],

                "SCORE":
                    score,
            }
        )

        all_trades.append(
            trades
        )

    # --------------------------------------------------------
    # RISULTATI
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        raise RuntimeError(
            "Nessun risultato generato."
        )

    trades_df = pd.concat(
        all_trades,
        ignore_index=True,
    )

    results_df = results_df.sort_values(
        "SCORE",
        ascending=False,
    )

    # --------------------------------------------------------
    # SALVATAGGIO
    # --------------------------------------------------------

    results_df.to_csv(
        f"{OUTPUT_DIR}/scan_all.csv",
        index=False,
    )

    trades_df.to_csv(
        f"{OUTPUT_DIR}/trades_all.csv",
        index=False,
    )

    df.to_csv(
        f"{OUTPUT_DIR}/candles_with_indicators.csv",
        index=False,
    )

    # --------------------------------------------------------
    # CANDIDATI ROBUSTI
    # --------------------------------------------------------

    robust_candidates = results_df[
        (results_df["DEV_TRADES"] >= MIN_TRADES_DEV)
        &
        (results_df["VER_TRADES"] >= MIN_TRADES_VER)
        &
        (results_df["DEV_avg_R"] > 0)
        &
        (results_df["VER_avg_R"] > 0)
        &
        (results_df["DEV_PF"] > 1)
        &
        (results_df["VER_PF"] > 1)
    ].copy()

    robust_candidates = (
        robust_candidates.sort_values(
            [
                "VER_TOTAL_R",
                "VER_PF",
                "SCORE",
            ],
            ascending=False,
        )
    )

    robust_candidates.to_csv(
        f"{OUTPUT_DIR}/robust_candidates.csv",
        index=False,
    )

    # --------------------------------------------------------
    # STAMPA TOP ROBUSTI
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TOP 25 CANDIDATI ROBUSTI")
    print("=" * 72)
    print()

    display_columns = [
        "direction",
        "mode",
        "TP_ATR",
        "SL_ATR",
        "HORIZON_MIN",
        "DEV_TRADES",
        "DEV_avg_R",
        "DEV_TOTAL_R",
        "DEV_PF",
        "VER_TRADES",
        "VER_avg_R",
        "VER_TOTAL_R",
        "VER_PF",
        "VER_DD",
    ]

    if robust_candidates.empty:

        print(
            "NESSUN CANDIDATO SODDISFA "
            "TUTTI I CRITERI DI ROBUSTEZZA."
        )

    else:

        print(
            robust_candidates[
                display_columns
            ].head(25).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # TOP GENERALE
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TOP 25 GENERALI")
    print("=" * 72)
    print()

    print(
        results_df[
            display_columns +
            ["SCORE"]
        ].head(25).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # RIEPILOGO
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("COMPLETATO")
    print("=" * 72)

    print(
        f"Combinazioni analizzate: "
        f"{len(results_df)}"
    )

    print(
        f"Candidati robusti: "
        f"{len(robust_candidates)}"
    )

    print()
    print(
        f"File risultati: "
        f"{OUTPUT_DIR}/"
    )

    print(
        f"- scan_all.csv"
    )

    print(
        f"- robust_candidates.csv"
    )

    print(
        f"- trades_all.csv"
    )

    print(
        f"- candles_with_indicators.csv"
    )

    print("=" * 72)


# ============================================================
# AVVIO
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        traceback.print_exc()

        raise
