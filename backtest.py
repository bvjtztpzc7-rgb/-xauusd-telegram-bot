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
    raise RuntimeError(
        "TWELVE_DATA_API_KEY non configurato"
    )

OUTPUT_DIR = "backtest_results_v7"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# DATI
# ------------------------------------------------------------

N_CANDLES = 10000

# V7:
# DEV = 60%
# VALIDATION = 20%
# TEST = 20%
DEV_FRAC = 0.60
VAL_FRAC = 0.20

# ------------------------------------------------------------
# TP / SL
# ------------------------------------------------------------

# Ricerca principale.
# Include il candidato V6:
# BUY + ATR_HIGH + TP 2.0 / SL 1.0 + 60m

TP_VALUES = [
    1.0,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    3.0,
]

SL_VALUES = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
]

HORIZONS = [
    30,
    60,
]

# ------------------------------------------------------------
# FILTRI
# ------------------------------------------------------------

MODES = [
    "BASE",
    "ATR_HIGH",
    "BODY",
    "RSI_MID",
    "TREND_1H",
    "EMA_STRONG",
]

DIRECTIONS = [
    "BUY",
    "SELL",
]

# ------------------------------------------------------------
# MINIMO TRADE
# ------------------------------------------------------------

MIN_TRADES_DEV = 20
MIN_TRADES_VAL = 10
MIN_TRADES_TEST = 10

# ------------------------------------------------------------
# COSTI / SLIPPAGE
# ------------------------------------------------------------

# Valori espressi in punti prezzo XAU/USD.
#
# Esempio:
# spread medio simulato per lato = 0.05
# slippage per lato = 0.05
#
# Totale teorico round-trip = 0.20
#
# Sono parametri volutamente configurabili.
# NON rappresentano una promessa sul costo reale del broker.

SPREAD_PER_SIDE = 0.05
SLIPPAGE_PER_SIDE = 0.05

ENTRY_COST = (
    SPREAD_PER_SIDE +
    SLIPPAGE_PER_SIDE
)

EXIT_COST = (
    SPREAD_PER_SIDE +
    SLIPPAGE_PER_SIDE
)

# ------------------------------------------------------------
# GESTIONE SEGNALI
# ------------------------------------------------------------

ENTRY_MODES = [
    "FIRST",
    "NON_OVERLAP",
    "COOLDOWN_30",
    "COOLDOWN_60",
]

# ------------------------------------------------------------
# SENSITIVITY
# ------------------------------------------------------------

SENSITIVITY_TP = [
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
]

SENSITIVITY_SL = [
    0.75,
    1.0,
    1.25,
]

# ============================================================
# DOWNLOAD DATI
# ============================================================

def get_data():

    url = "https://api.twelvedata.com/time_series"

    frames = []

    end = None
    remaining = N_CANDLES

    while remaining > 0:

        size = min(
            5000,
            remaining
        )

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

        chunk = pd.DataFrame(
            data["values"]
        )

        if chunk.empty:
            break

        frames.append(chunk)

        if len(chunk) < size:
            break

        end = str(
            chunk["datetime"].min()
        )

        remaining -= len(chunk)

        time.sleep(0.2)

    if not frames:
        raise RuntimeError(
            "Nessun dato scaricato."
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = df.drop_duplicates(
        subset=["datetime"]
    )

    df = df.tail(
        N_CANDLES
    )

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

    df = (
        df
        .sort_values("datetime")
        .reset_index(drop=True)
    )

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
        avg_loss.replace(
            0,
            np.nan,
        )
    )

    return 100 - (
        100 / (1 + rs)
    )


def atr(df, period=14):

    previous_close = (
        df["close"].shift(1)
    )

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
        ema(series, 12)
        -
        ema(series, 26)
    )

    signal_line = ema(
        macd_line,
        9,
    )

    return (
        macd_line,
        signal_line,
    )


# ============================================================
# INDICATORI 5m
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

    # --------------------------------------------------------
    # ATR REGIME
    # --------------------------------------------------------

    df["atr_median_200"] = (
        df["atr_pct"]
        .rolling(200)
        .median()
    )

    return df


# ============================================================
# TIMEFRAME SUPERIORI - LEAKAGE FREE
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
        .resample(
            rule,
            label="left",
            closed="left",
        )
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

    # --------------------------------------------------------
    # IMPORTANTISSIMO
    #
    # La candela 15m/1H viene etichettata all'inizio
    # dell'intervallo.
    #
    # Esempio:
    #
    # 10:00 -> 10:15
    #
    # diventa disponibile SOLO dalle 10:15.
    #
    # Spostiamo quindi il timestamp alla fine della candela.
    # --------------------------------------------------------

    rule_delta = pd.Timedelta(
        rule
    )

    x["available_at"] = (
        x["datetime"] +
        rule_delta
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

    # --------------------------------------------------------
    # 15 MIN
    # --------------------------------------------------------

    trend15 = make_timeframe(
        result,
        "15min",
    )

    trend15 = trend15[
        [
            "available_at",
            "trend",
        ]
    ].rename(
        columns={
            "available_at":
                "datetime",

            "trend":
                "trend15",
        }
    )

    # --------------------------------------------------------
    # 1 HOUR
    # --------------------------------------------------------

    trend1h = make_timeframe(
        result,
        "1h",
    )

    trend1h = trend1h[
        [
            "available_at",
            "trend",
        ]
    ].rename(
        columns={
            "available_at":
                "datetime",

            "trend":
                "trend1h",
        }
    )

    # --------------------------------------------------------
    # MERGE
    #
    # allow_exact_matches=True:
    #
    # se siamo esattamente alle 10:15,
    # la 15m 10:00-10:15 è appena chiusa
    # ed è quindi utilizzabile.
    # --------------------------------------------------------

    result = pd.merge_asof(
        result.sort_values(
            "datetime"
        ),
        trend15.sort_values(
            "datetime"
        ),
        on="datetime",
        direction="backward",
        allow_exact_matches=True,
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
        allow_exact_matches=True,
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
    # FILTRI V7
    # --------------------------------------------------------

    if mode == "ATR_HIGH":

        mask &= (
            df["atr_pct"] >=
            df["atr_median_200"]
        )

    elif mode == "BODY":

        mask &= (
            df["body_ratio"] >=
            0.45
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

    elif mode == "EMA_STRONG":

        mask &= (
            df["ema_gap"] >=
            0.5
        )

    return mask.fillna(False)


# ============================================================
# GESTIONE EPISODI
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


def select_signals(
    df,
    raw_signals,
    entry_mode,
):

    indexes = np.flatnonzero(
        raw_signals.to_numpy()
    )

    if len(indexes) == 0:
        return raw_signals * False

    selected = []

    last_selected = -10**9

    for idx in indexes:

        if entry_mode == "FIRST":

            # Il primo segnale di ogni episodio
            if idx == indexes[0]:

                selected.append(idx)

            else:

                previous_idx = (
                    indexes[
                        np.where(
                            indexes == idx
                        )[0][0] - 1
                    ]
                )

                if idx > previous_idx + 1:
                    selected.append(idx)

        elif entry_mode == "NON_OVERLAP":

            if idx > last_selected:

                selected.append(idx)

        elif entry_mode == "COOLDOWN_30":

            cooldown_bars = 6

            if (
                idx -
                last_selected
                >= cooldown_bars
            ):
                selected.append(idx)

        elif entry_mode == "COOLDOWN_60":

            cooldown_bars = 12

            if (
                idx -
                last_selected
                >= cooldown_bars
            ):
                selected.append(idx)

        if (
            len(selected) > 0
            and selected[-1] == idx
        ):
            last_selected = idx

    result = pd.Series(
        False,
        index=df.index,
    )

    if selected:
        result.iloc[
            selected
        ] = True

    return result


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

        if (
            signal_index + 1
            >= len(df)
        ):
            continue

        entry_index = (
            signal_index + 1
        )

        raw_entry = float(
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

        # ----------------------------------------------------
        # ENTRY COST
        #
        # BUY: cost aumenta il prezzo di entrata
        # SELL: cost aumenta il prezzo effettivo
        # ----------------------------------------------------

        if direction == "BUY":

            entry = (
                raw_entry +
                ENTRY_COST
            )

            tp_price = (
                entry +
                tp * atr_value
            )

            sl_price = (
                entry -
                sl * atr_value
            )

        else:

            entry = (
                raw_entry -
                ENTRY_COST
            )

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
            int(
                horizon / 5
            ),
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
            # TP + SL stessa candela
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
        # EXIT COST
        # ----------------------------------------------------

        if result == "TP":

            if direction == "BUY":

                exit_price -= EXIT_COST

            else:

                exit_price += EXIT_COST

        elif result == "SL":

            if direction == "BUY":

                exit_price -= EXIT_COST

            else:

                exit_price += EXIT_COST

        else:

            if direction == "BUY":

                exit_price -= EXIT_COST

            else:

                exit_price += EXIT_COST

        # ----------------------------------------------------
        # CALCOLO R
        #
        # IMPORTANTE:
        # il denominatore è il rischio iniziale.
        # ----------------------------------------------------

        risk = (
            sl *
            atr_value
        )

        if risk <= 0:
            continue

        if direction == "BUY":

            rr = (
                exit_price -
                entry
            ) / risk

        else:

            rr = (
                entry -
                exit_price
            ) / risk

        # ----------------------------------------------------
        # Classificazione
        #
        # Dopo i costi un TP può teoricamente diventare
        # leggermente inferiore al TP nominale.
        # Manteniamo comunque TP/SL come evento di mercato.
        # ----------------------------------------------------

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

    trades = (
        trades
        .sort_values(
            "entry_time"
        )
        .reset_index(
            drop=True
        )
    )

    r = trades["R"].astype(
        float
    )

    wins = (
        trades["R"] > 0
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

def safe_pf(value):

    if not np.isfinite(value):
        return 10.0

    return float(value)


def robustness_score(
    dev,
    val,
):

    if (
        dev["trades"] <
        MIN_TRADES_DEV
    ):
        return -9999

    if (
        val["trades"] <
        MIN_TRADES_VAL
    ):
        return -9999

    if (
        not np.isfinite(
            dev["avg_R"]
        )
        or
        not np.isfinite(
            val["avg_R"]
        )
    ):
        return -9999

    # --------------------------------------------------------
    # Non premiamo semplicemente il profitto.
    #
    # Vogliamo:
    # - avg R positivo
    # - PF positivo
    # - stabilità DEV -> VAL
    # - drawdown contenuto
    # --------------------------------------------------------

    dev_pf = safe_pf(
        dev["PF"]
    )

    val_pf = safe_pf(
        val["PF"]
    )

    score = (

        # Validation conta di più
        40.0 *
        val["avg_R"]

        +

        20.0 *
        dev["avg_R"]

        +

        5.0 *
        min(
            val_pf - 1.0,
            2.0,
        )

        +

        2.0 *
        min(
            dev_pf - 1.0,
            2.0,
        )

        -

        0.10 *
        val["DD"]

        -

        0.05 *
        dev["DD"]
    )

    # Penalità se DEV è molto migliore
    # della Validation.
    degradation = (
        dev["avg_R"] -
        val["avg_R"]
    )

    if degradation > 0:

        score -= (
            10.0 *
            degradation
        )

    return score


# ============================================================
# SPLIT TEMPORALE
# ============================================================

def add_periods(
    trades,
    dev_end,
    val_end,
):

    trades = trades.copy()

    trades["period"] = np.select(
        [
            trades["entry_time"]
            <= dev_end,

            trades["entry_time"]
            <= val_end,
        ],
        [
            "DEV",
            "VAL",
        ],
        default="TEST",
    )

    return trades


# ============================================================
# ANALISI TEMPORALE
# ============================================================

def time_stability(
    trades,
    period,
):

    x = trades[
        trades["period"] ==
        period
    ].copy()

    if x.empty:
        return {}

    x = x.sort_values(
        "entry_time"
    )

    midpoint = (
        len(x) // 2
    )

    if midpoint == 0:
        return {}

    first = x.iloc[
        :midpoint
    ]

    second = x.iloc[
        midpoint:
    ]

    first_stats = calculate_stats(
        first
    )

    second_stats = calculate_stats(
        second
    )

    return {
        "FIRST_avg_R":
            first_stats["avg_R"],

        "SECOND_avg_R":
            second_stats["avg_R"],

        "FIRST_PF":
            first_stats["PF"],

        "SECOND_PF":
            second_stats["PF"],

        "FIRST_TRADES":
            first_stats["trades"],

        "SECOND_TRADES":
            second_stats["trades"],
    }


# ============================================================
# MAIN SCANNER
# ============================================================

def main():

    print()
    print("=" * 78)
    print("BACKTEST V7 - TARGETED ROBUSTNESS VALIDATION")
    print("=" * 78)
    print()

    print(
        "ATTENZIONE: risultati solo DEMO/PAPER."
    )

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

    # --------------------------------------------------------
    # TIMEFRAME SUPERIORI
    # --------------------------------------------------------

    df = merge_trends(
        df
    )

    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # SPLIT 60 / 20 / 20
    # --------------------------------------------------------

    n = len(df)

    dev_index = int(
        n *
        DEV_FRAC
    )

    val_index = int(
        n *
        (
            DEV_FRAC +
            VAL_FRAC
        )
    )

    dev_end = (
        df.iloc[
            dev_index - 1
        ]["datetime"]
    )

    val_end = (
        df.iloc[
            val_index - 1
        ]["datetime"]
    )

    print(
        "SVILUPPO:"
    )

    print(
        f"fino a {dev_end}"
    )

    print()

    print(
        "VALIDATION:"
    )

    print(
        f"{dev_end} -> {val_end}"
    )

    print()

    print(
        "TEST FINALE:"
    )

    print(
        f"dopo {val_end}"
    )

    print()

    # --------------------------------------------------------
    # SCANNER
    # --------------------------------------------------------

    results = []

    all_trades = []

    combinations = product(
        DIRECTIONS,
        MODES,
        ENTRY_MODES,
        TP_VALUES,
        SL_VALUES,
        HORIZONS,
    )

    total_combinations = 0

    for (
        direction,
        mode,
        entry_mode,
        tp,
        sl,
        horizon,
    ) in combinations:

        total_combinations += 1

        # ----------------------------------------------------
        # SEGNALI
        # ----------------------------------------------------

        raw_signals = signal_mask(
            df,
            direction,
            mode,
        )

        signals = first_in_episode(
            raw_signals
        )

        signals = select_signals(
            df,
            signals,
            entry_mode,
        )

        if not signals.any():
            continue

        # ----------------------------------------------------
        # TRADE
        # ----------------------------------------------------

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

        trades["entry_mode"] = (
            entry_mode
        )

        trades["TP_ATR"] = tp

        trades["SL_ATR"] = sl

        trades["HORIZON_MIN"] = (
            horizon
        )

        trades = add_periods(
            trades,
            dev_end,
            val_end,
        )

        # ----------------------------------------------------
        # STATISTICHE
        # ----------------------------------------------------

        dev_trades = trades[
            trades["period"] ==
            "DEV"
        ]

        val_trades = trades[
            trades["period"] ==
            "VAL"
        ]

        test_trades = trades[
            trades["period"] ==
            "TEST"
        ]

        dev = calculate_stats(
            dev_trades
        )

        val = calculate_stats(
            val_trades
        )

        test = calculate_stats(
            test_trades
        )

        score = robustness_score(
            dev,
            val,
        )

        # ----------------------------------------------------
        # DEGRADATION
        # ----------------------------------------------------

        if (
            np.isfinite(
                dev["avg_R"]
            )
            and
            np.isfinite(
                val["avg_R"]
            )
            and
            dev["avg_R"] != 0
        ):

            degradation_pct = (
                1 -
                (
                    val["avg_R"] /
                    dev["avg_R"]
                )
            ) * 100

        else:

            degradation_pct = np.nan

        results.append(
            {
                "direction":
                    direction,

                "mode":
                    mode,

                "entry_mode":
                    entry_mode,

                "TP_ATR":
                    tp,

                "SL_ATR":
                    sl,

                "HORIZON_MIN":
                    horizon,

                "DEV_TRADES":
                    dev["trades"],

                "DEV_WIN":
                    dev["win"],

                "DEV_avg_R":
                    dev["avg_R"],

                "DEV_TOTAL_R":
                    dev["total_R"],

                "DEV_PF":
                    dev["PF"],

                "DEV_DD":
                    dev["DD"],

                "VAL_TRADES":
                    val["trades"],

                "VAL_WIN":
                    val["win"],

                "VAL_avg_R":
                    val["avg_R"],

                "VAL_TOTAL_R":
                    val["total_R"],

                "VAL_PF":
                    val["PF"],

                "VAL_DD":
                    val["DD"],

                "TEST_TRADES":
                    test["trades"],

                "TEST_WIN":
                    test["win"],

                "TEST_avg_R":
                    test["avg_R"],

                "TEST_TOTAL_R":
                    test["total_R"],

                "TEST_PF":
                    test["PF"],

                "TEST_DD":
                    test["DD"],

                "DEV_VAL_DEGRADATION_%":
                    degradation_pct,

                "SCORE":
                    score,
            }
        )

        all_trades.append(
            trades
        )

    # --------------------------------------------------------
    # DATAFRAME RISULTATI
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

    # --------------------------------------------------------
    # SALVATAGGIO COMPLETO
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

    # ========================================================
    # CANDIDATI ROBUSTI DEV + VALIDATION
    # ========================================================

    robust = results_df[
        (results_df["DEV_TRADES"]
         >= MIN_TRADES_DEV)
        &
        (results_df["VAL_TRADES"]
         >= MIN_TRADES_VAL)
        &
        (results_df["DEV_avg_R"] > 0)
        &
        (results_df["VAL_avg_R"] > 0)
        &
        (results_df["DEV_PF"] > 1)
        &
        (results_df["VAL_PF"] > 1)
    ].copy()

    robust = robust.sort_values(
        [
            "SCORE",
            "VAL_PF",
            "VAL_avg_R",
        ],
        ascending=False,
    )

    robust.to_csv(
        f"{OUTPUT_DIR}/robust_dev_val.csv",
        index=False,
    )

    # ========================================================
    # TEST FINALE
    #
    # IL TEST NON È USATO PER SELEZIONARE.
    # ========================================================

    test_candidates = robust[
        robust["TEST_TRADES"]
        >= MIN_TRADES_TEST
    ].copy()

    test_candidates = (
        test_candidates.sort_values(
            [
                "TEST_avg_R",
                "TEST_PF",
            ],
            ascending=False,
        )
    )

    test_candidates.to_csv(
        f"{OUTPUT_DIR}/test_results_for_robust_candidates.csv",
        index=False,
    )

    # ========================================================
    # CANDIDATO V6 SPECIFICO
    # ========================================================

    v6_candidate = results_df[
        (results_df["direction"] == "BUY")
        &
        (results_df["mode"] == "ATR_HIGH")
        &
        (results_df["TP_ATR"] == 2.0)
        &
        (results_df["SL_ATR"] == 1.0)
        &
        (results_df["HORIZON_MIN"] == 60)
    ].copy()

    v6_candidate.to_csv(
        f"{OUTPUT_DIR}/v6_candidate_check.csv",
        index=False,
    )

    # ========================================================
    # OUTPUT TOP DEV/VALIDATION
    # ========================================================

    display_columns = [
        "direction",
        "mode",
        "entry_mode",
        "TP_ATR",
        "SL_ATR",
        "HORIZON_MIN",

        "DEV_TRADES",
        "DEV_avg_R",
        "DEV_PF",
        "DEV_DD",

        "VAL_TRADES",
        "VAL_avg_R",
        "VAL_PF",
        "VAL_DD",

        "TEST_TRADES",
        "TEST_avg_R",
        "TEST_PF",
        "TEST_DD",

        "SCORE",
    ]

    print()
    print("=" * 78)
    print("TOP 30 DEV + VALIDATION")
    print("=" * 78)
    print()

    if robust.empty:

        print(
            "NESSUN CANDIDATO SUPERA "
            "I CRITERI DEV + VALIDATION."
        )

    else:

        print(
            robust[
                display_columns
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # TEST DEI CANDIDATI ROBUSTI
    # ========================================================

    print()
    print("=" * 78)
    print("TEST FINALE DEI CANDIDATI ROBUSTI")
    print("=" * 78)
    print()

    if test_candidates.empty:

        print(
            "Nessun candidato robusto ha "
            "abbastanza trade nel TEST."
        )

    else:

        print(
            test_candidates[
                display_columns
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # CHECK CANDIDATO V6
    # ========================================================

    print()
    print("=" * 78)
    print("CHECK CANDIDATO V6")
    print("=" * 78)
    print()

    if v6_candidate.empty:

        print(
            "Candidato V6 non trovato."
        )

    else:

        print(
            v6_candidate[
                display_columns
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # SENSITIVITY DEL CANDIDATO V6
    # ========================================================

    print()
    print("=" * 78)
    print("SENSITIVITY BUY + ATR_HIGH + 60m")
    print("=" * 78)
    print()

    sensitivity_results = []

    for tp in SENSITIVITY_TP:

        for sl in SENSITIVITY_SL:

            raw = signal_mask(
                df,
                "BUY",
                "ATR_HIGH",
            )

            raw = first_in_episode(
                raw
            )

            selected = select_signals(
                df,
                raw,
                "NON_OVERLAP",
            )

            trades = simulate(
                df,
                selected,
                "BUY",
                tp,
                sl,
                60,
            )

            if trades.empty:
                continue

            trades = add_periods(
                trades,
                dev_end,
                val_end,
            )

            dev = calculate_stats(
                trades[
                    trades["period"] ==
                    "DEV"
                ]
            )

            val = calculate_stats(
                trades[
                    trades["period"] ==
                    "VAL"
                ]
            )

            test = calculate_stats(
                trades[
                    trades["period"] ==
                    "TEST"
                ]
            )

            sensitivity_results.append(
                {
                    "TP_ATR":
                        tp,

                    "SL_ATR":
                        sl,

                    "DEV_TRADES":
                        dev["trades"],

                    "DEV_avg_R":
                        dev["avg_R"],

                    "DEV_PF":
                        dev["PF"],

                    "VAL_TRADES":
                        val["trades"],

                    "VAL_avg_R":
                        val["avg_R"],

                    "VAL_PF":
                        val["PF"],

                    "TEST_TRADES":
                        test["trades"],

                    "TEST_avg_R":
                        test["avg_R"],

                    "TEST_PF":
                        test["PF"],
                }
            )

    sensitivity_df = pd.DataFrame(
        sensitivity_results
    )

    if not sensitivity_df.empty:

        sensitivity_df.to_csv(
            f"{OUTPUT_DIR}/sensitivity_atr_high.csv",
            index=False,
        )

        print(
            sensitivity_df.to_string(
                index=False
            )
        )

    # ========================================================
    # STABILITÀ TEMPORALE
    # ========================================================

    print()
    print("=" * 78)
    print("STABILITÀ TEMPORALE TOP CANDIDATI")
    print("=" * 78)
    print()

    stability_rows = []

    for _, row in robust.head(20).iterrows():

        mask = (
            (trades_df["direction"]
             == row["direction"])
            &
            (trades_df["mode"]
             == row["mode"])
            &
            (trades_df["entry_mode"]
             == row["entry_mode"])
            &
            (trades_df["TP_ATR"]
             == row["TP_ATR"])
            &
            (trades_df["SL_ATR"]
             == row["SL_ATR"])
            &
            (trades_df["HORIZON_MIN"]
             == row["HORIZON_MIN"])
        )

        candidate_trades = trades_df[
            mask
        ].copy()

        stability = time_stability(
            candidate_trades,
            "VAL",
        )

        if stability:

            stability_rows.append(
                {
                    "direction":
                        row["direction"],

                    "mode":
                        row["mode"],

                    "entry_mode":
                        row["entry_mode"],

                    "TP_ATR":
                        row["TP_ATR"],

                    "SL_ATR":
                        row["SL_ATR"],

                    "HORIZON_MIN":
                        row["HORIZON_MIN"],

                    **stability,
                }
            )

    stability_df = pd.DataFrame(
        stability_rows
    )

    if not stability_df.empty:

        stability_df.to_csv(
            f"{OUTPUT_DIR}/temporal_stability.csv",
            index=False,
        )

        print(
            stability_df.to_string(
                index=False
            )
        )

    # ========================================================
    # RIEPILOGO
    # ========================================================

    print()
    print("=" * 78)
    print("COMPLETATO")
    print("=" * 78)
    print()

    print(
        f"Combinazioni analizzate: "
        f"{total_combinations}"
    )

    print(
        f"Risultati validi: "
        f"{len(results_df)}"
    )

    print(
        f"Candidati robusti DEV+VAL: "
        f"{len(robust)}"
    )

    print(
        f"Candidati arrivati al TEST: "
        f"{len(test_candidates)}"
    )

    print()
    print(
        f"Output: {OUTPUT_DIR}/"
    )

    print(
        "- scan_all.csv"
    )

    print(
        "- robust_dev_val.csv"
    )

    print(
        "- test_results_for_robust_candidates.csv"
    )

    print(
        "- v6_candidate_check.csv"
    )

    print(
        "- sensitivity_atr_high.csv"
    )

    print(
        "- temporal_stability.csv"
    )

    print(
        "- trades_all.csv"
    )

    print(
        "- candles_with_indicators.csv"
    )

    print()
    print("=" * 78)


# ============================================================
# AVVIO
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        traceback.print_exc()

        raise
