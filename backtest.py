import os
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAZIONE V5
# ============================================================

SYMBOL = "XAU/USD"
INTERVAL = "5min"

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "TWELVE_DATA_API_KEY non configurato"
    )

OUTPUT_DIR = "backtest_results_v5"
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# STORICO
# ------------------------------------------------------------

BLOCK_SIZE = 5000
TOTAL_CANDLES = 10000

# ------------------------------------------------------------
# SPLIT TEMPORALE
# ------------------------------------------------------------

DEVELOPMENT_PCT = 0.70

# ------------------------------------------------------------
# TRADE
# ------------------------------------------------------------

HORIZON_BARS = 288          # 24 ore
DEFAULT_TP_ATR = 2.5
DEFAULT_SL_ATR = 1.5

# ------------------------------------------------------------
# MATRICE TP / SL
# ------------------------------------------------------------

TP_VALUES = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0
]

SL_VALUES = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0
]

# ------------------------------------------------------------
# SAME CANDLE
# ------------------------------------------------------------

SAME_CANDLE_MODE = "CONSERVATIVE"

# ------------------------------------------------------------
# COSTI
# ------------------------------------------------------------

COST_R = 0.0


# ============================================================
# UTILITY
# ============================================================

def safe_float(value, default=np.nan):

    try:
        return float(value)

    except Exception:
        return default


def ensure_dataframe(obj):

    if obj is None:
        return pd.DataFrame()

    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    if isinstance(obj, pd.Series):
        return obj.to_frame().T

    if isinstance(obj, np.ndarray):

        if obj.size == 0:
            return pd.DataFrame()

        if obj.ndim == 1:
            return pd.DataFrame([obj])

        return pd.DataFrame(obj)

    if isinstance(obj, list):

        if not obj:
            return pd.DataFrame()

        return pd.DataFrame(obj)

    return pd.DataFrame(obj)


def print_separator(title):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def profit_factor(results):

    values = np.asarray(
        results,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    gains = values[
        values > 0
    ].sum()

    losses = abs(
        values[
            values < 0
        ].sum()
    )

    if losses == 0:

        if gains > 0:
            return np.inf

        return np.nan

    return float(
        gains / losses
    )


def max_drawdown(results):

    values = np.asarray(
        results,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return 0.0

    equity = np.concatenate(
        [
            [0.0],
            np.cumsum(values)
        ]
    )

    peaks = np.maximum.accumulate(
        equity
    )

    drawdown = peaks - equity

    return float(
        drawdown.max()
    )


def print_table(title, data):

    print_separator(title)

    data = ensure_dataframe(data)

    if data.empty:

        print("Nessun dato.")

        return

    with pd.option_context(
        "display.max_rows",
        200,
        "display.max_columns",
        80,
        "display.width",
        260,
        "display.float_format",
        lambda x: f"{x:.4f}"
    ):

        print(
            data.to_string(
                index=False
            )
        )


# ============================================================
# DOWNLOAD
# ============================================================

def download_block(end_date=None):

    url = (
        "https://api.twelvedata.com/"
        "time_series"
    )

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": BLOCK_SIZE,
        "apikey": API_KEY,
        "format": "JSON",
        "timezone": "UTC"
    }

    if end_date is not None:

        params["end_date"] = end_date

    print(
        "Download:",
        {
            k: "***"
            if k == "apikey"
            else v
            for k, v in params.items()
        }
    )

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "values" not in data:

        print("RISPOSTA API:")
        print(data)

        raise RuntimeError(
            "Twelve Data non ha restituito values."
        )

    df = pd.DataFrame(
        data["values"]
    )

    if df.empty:

        raise RuntimeError(
            "Blocco storico vuoto."
        )

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close"
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Colonne mancanti: {missing}"
        )

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

    df = df.dropna(
        subset=required
    )

    df = df.sort_values(
        "datetime"
    )

    df = df.drop_duplicates(
        "datetime"
    )

    df = df.reset_index(
        drop=True
    )

    print(
        f"Ricevute {len(df)} candele | "
        f"{df['datetime'].iloc[0]} -> "
        f"{df['datetime'].iloc[-1]}"
    )

    return df


def download_history():

    print_separator(
        "DOWNLOAD STORICO"
    )

    block1 = download_block()

    time.sleep(1)

    oldest = block1[
        "datetime"
    ].min()

    end_date = (
        oldest -
        pd.Timedelta(minutes=5)
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    block2 = download_block(
        end_date=end_date
    )

    df = pd.concat(
        [
            block1,
            block2
        ],
        ignore_index=True
    )

    df = df.sort_values(
        "datetime"
    )

    df = df.drop_duplicates(
        "datetime"
    )

    df = df.reset_index(
        drop=True
    )

    if len(df) > TOTAL_CANDLES:

        df = df.tail(
            TOTAL_CANDLES
        ).reset_index(
            drop=True
        )

    print_separator(
        "STORICO FINALE"
    )

    print(
        f"Candele: {len(df)}"
    )

    print(
        f"Inizio: {df['datetime'].iloc[0]}"
    )

    print(
        f"Fine:   {df['datetime'].iloc[-1]}"
    )

    return df


# ============================================================
# INDICATORI
# ============================================================

def calculate_indicators(df):

    df = ensure_dataframe(
        df
    ).copy()

    close = df["close"]

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema20"] = (
        close
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["ema50"] = (
        close
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = (
        close
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        close
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["macd"] = (
        ema12 - ema26
    )

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    df["rsi"] = (
        100 -
        100 / (1 + rs)
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = (
        close.shift(1)
    )

    tr1 = (
        df["high"] -
        df["low"]
    )

    tr2 = (
        df["high"] -
        previous_close
    ).abs()

    tr3 = (
        df["low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(
        axis=1
    )

    df["atr"] = (
        true_range
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["body_atr"] = (
        df["body"] /
        df["atr"].replace(
            0,
            np.nan
        )
    )

    df["ema_distance"] = (
        (
            df["ema20"] -
            df["ema50"]
        )
        /
        df["atr"].replace(
            0,
            np.nan
        )
    )

    df["ema20_slope"] = (
        (
            df["ema20"] -
            df["ema20"].shift(5)
        )
        /
        df["atr"].replace(
            0,
            np.nan
        )
    )

    df["ema50_slope"] = (
        (
            df["ema50"] -
            df["ema50"].shift(5)
        )
        /
        df["atr"].replace(
            0,
            np.nan
        )
    )

    atr_mean = (
        df["atr"]
        .rolling(50)
        .mean()
    )

    df["atr_ratio"] = (
        df["atr"] /
        atr_mean
    )

    for bars in [
        3,
        6,
        12,
        24
    ]:

        df[
            f"momentum_{bars}"
        ] = (
            (
                df["close"] -
                df["close"].shift(bars)
            )
            /
            df["atr"].replace(
                0,
                np.nan
            )
        )

    df["hour_utc"] = (
        df["datetime"].dt.hour
    )

    df["day_of_week"] = (
        df["datetime"].dt.dayofweek
    )

    return df


# ============================================================
# MULTI TIMEFRAME
# ============================================================

def add_higher_timeframes(df):

    print_separator(
        "COSTRUZIONE TIMEFRAME SUPERIORI"
    )

    base = df.copy()

    base = base.set_index(
        "datetime"
    )

    # --------------------------------------------------------
    # 15 MIN
    # --------------------------------------------------------

    tf15 = base.resample(
        "15min",
        label="right",
        closed="right"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })

    tf15 = tf15.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    tf15["ema20_15"] = (
        tf15["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    tf15["ema50_15"] = (
        tf15["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    tf15["bullish_15"] = (
        tf15["ema20_15"] >
        tf15["ema50_15"]
    )

    tf15["bearish_15"] = (
        tf15["ema20_15"] <
        tf15["ema50_15"]
    )

    # Solo candela 15m completamente chiusa
    tf15 = tf15[
        [
            "ema20_15",
            "ema50_15",
            "bullish_15",
            "bearish_15"
        ]
    ].shift(1)

    # --------------------------------------------------------
    # 1 HOUR
    # --------------------------------------------------------

    tf1h = base.resample(
        "1h",
        label="right",
        closed="right"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })

    tf1h = tf1h.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    tf1h["ema20_1h"] = (
        tf1h["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    tf1h["ema50_1h"] = (
        tf1h["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    tf1h["bullish_1h"] = (
        tf1h["ema20_1h"] >
        tf1h["ema50_1h"]
    )

    tf1h["bearish_1h"] = (
        tf1h["ema20_1h"] <
        tf1h["ema50_1h"]
    )

    # Solo candela 1H completamente chiusa
    tf1h = tf1h[
        [
            "ema20_1h",
            "ema50_1h",
            "bullish_1h",
            "bearish_1h"
        ]
    ].shift(1)

    # --------------------------------------------------------
    # JOIN
    # --------------------------------------------------------

    base = base.join(
        tf15,
        how="left"
    )

    base = base.join(
        tf1h,
        how="left"
    )

    base = base.reset_index()

    return base


# ============================================================
# GENERAZIONE SEGNALI
# ============================================================

def generate_signals(df):

    print_separator(
        "GENERAZIONE SEGNALI"
    )

    df = df.copy()

    valid = (
        df["rsi"].notna()
        &
        df["atr"].notna()
        &
        (df["atr"] > 0)
        &
        df["ema20"].notna()
        &
        df["ema50"].notna()
        &
        df["macd"].notna()
        &
        df["macd_signal"].notna()
        &
        df["bullish_15"].notna()
        &
        df["bullish_1h"].notna()
    )

    # --------------------------------------------------------
    # STRATEGIA BASE V5
    #
    # ATTENZIONE:
    # Questa parte viene mantenuta volutamente uguale
    # alla strategia che stiamo diagnosticando.
    # Prima misuriamo i risultati, poi cambiamo i filtri.
    # --------------------------------------------------------

    buy = (
        valid
        &
        (df["ema20"] > df["ema50"])
        &
        (df["macd"] > df["macd_signal"])
        &
        df["bullish_15"]
    )

    sell = (
        valid
        &
        (df["ema20"] < df["ema50"])
        &
        (df["macd"] < df["macd_signal"])
        &
        df["bearish_15"]
    )

    df["signal"] = 0

    df.loc[
        buy,
        "signal"
    ] = 1

    df.loc[
        sell,
        "signal"
    ] = -1

    df["direction"] = np.select(
        [
            df["signal"] == 1,
            df["signal"] == -1
        ],
        [
            "BUY",
            "SELL"
        ],
        default=""
    )

    print(
        "Segnali BUY:",
        int(
            (
                df["signal"] == 1
            ).sum()
        )
    )

    print(
        "Segnali SELL:",
        int(
            (
                df["signal"] == -1
            ).sum()
        )
    )

    print(
        "Segnali totali:",
        int(
            (
                df["signal"] != 0
            ).sum()
        )
    )

    return df


# ============================================================
# SIGNAL EPISODES
# ============================================================

def first_in_episode(df):

    signals = df[
        df["signal"] != 0
    ].copy()

    if signals.empty:
        return signals

    previous = (
        signals["signal"]
        .shift(1)
    )

    keep = (
        previous.isna()
        |
        (
            previous !=
            signals["signal"]
        )
    )

    return signals[
        keep
    ].copy()


# ============================================================
# SIMULAZIONE TRADE
# ============================================================

def simulate_trade(
    df,
    signal_index,
    tp_atr,
    sl_atr,
    horizon=HORIZON_BARS,
    same_candle_mode=SAME_CANDLE_MODE
):

    # --------------------------------------------------------
    # Il segnale nasce sulla candela N.
    # L'ingresso avviene all'OPEN di N+1.
    # --------------------------------------------------------

    if signal_index >= len(df) - 1:
        return None

    signal_row = df.iloc[
        signal_index
    ]

    direction = (
        "BUY"
        if signal_row["signal"] == 1
        else "SELL"
    )

    entry_index = (
        signal_index + 1
    )

    entry_row = df.iloc[
        entry_index
    ]

    entry_price = safe_float(
        entry_row["open"]
    )

    atr = safe_float(
        signal_row["atr"]
    )

    if (
        not np.isfinite(entry_price)
        or not np.isfinite(atr)
        or atr <= 0
    ):
        return None

    if direction == "BUY":

        tp_price = (
            entry_price +
            tp_atr * atr
        )

        sl_price = (
            entry_price -
            sl_atr * atr
        )

    else:

        tp_price = (
            entry_price -
            tp_atr * atr
        )

        sl_price = (
            entry_price +
            sl_atr * atr
        )

    end_index = min(
        len(df) - 1,
        entry_index + horizon
    )

    mfe = 0.0
    mae = 0.0

    outcome = "TIMEOUT"

    exit_index = end_index

    exit_price = safe_float(
        df.iloc[
            end_index
        ]["close"]
    )

    minutes_to_target = np.nan

    same_candle = False

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

    for j in range(
        entry_index,
        end_index + 1
    ):

        candle = df.iloc[j]

        high = safe_float(
            candle["high"]
        )

        low = safe_float(
            candle["low"]
        )

        if direction == "BUY":

            favorable = (
                high -
                entry_price
            ) / atr

            adverse = (
                entry_price -
                low
            ) / atr

            tp_hit = (
                high >= tp_price
            )

            sl_hit = (
                low <= sl_price
            )

        else:

            favorable = (
                entry_price -
                low
            ) / atr

            adverse = (
                high -
                entry_price
            ) / atr

            tp_hit = (
                low <= tp_price
            )

            sl_hit = (
                high >= sl_price
            )

        mfe = max(
            mfe,
            favorable
        )

        mae = max(
            mae,
            adverse
        )

        if (
            np.isnan(
                minutes_to_target
            )
            and
            favorable >= tp_atr
        ):

            minutes_to_target = (
                j -
                entry_index
            ) * 5

        # ----------------------------------------------------
        # TP + SL NELLA STESSA CANDELA
        # ----------------------------------------------------

        if tp_hit and sl_hit:

            same_candle = True

            if (
                same_candle_mode ==
                "CONSERVATIVE"
            ):

                outcome = "SL"

                exit_index = j

                exit_price = sl_price

            elif (
                same_candle_mode ==
                "OPTIMISTIC"
            ):

                outcome = "TP"

                exit_index = j

                exit_price = tp_price

            else:

                outcome = "EXCLUDE"

                exit_index = j

                exit_price = np.nan

            break

        # ----------------------------------------------------
        # SOLO TP
        # ----------------------------------------------------

        if tp_hit:

            outcome = "TP"

            exit_index = j

            exit_price = tp_price

            break

        # ----------------------------------------------------
        # SOLO SL
        # ----------------------------------------------------

        if sl_hit:

            outcome = "SL"

            exit_index = j

            exit_price = sl_price

            break

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    if outcome == "TIMEOUT":

        exit_price = safe_float(
            df.iloc[
                exit_index
            ]["close"]
        )

    # --------------------------------------------------------
    # RISULTATO IN R
    # --------------------------------------------------------

    if outcome == "TP":

        result_r = (
            tp_atr /
            sl_atr
        )

    elif outcome == "SL":

        result_r = -1.0

    elif outcome == "EXCLUDE":

        result_r = np.nan

    else:

        if not np.isfinite(
            exit_price
        ):

            result_r = np.nan

        elif direction == "BUY":

            result_r = (
                exit_price -
                entry_price
            ) / (
                sl_atr * atr
            )

        else:

            result_r = (
                entry_price -
                exit_price
            ) / (
                sl_atr * atr
            )

    if np.isfinite(
        result_r
    ):

        result_r -= COST_R

    return {
        "signal_index":
            signal_index,

        "entry_index":
            entry_index,

        "exit_index":
            exit_index,

        "signal_datetime":
            signal_row["datetime"],

        "datetime":
            entry_row["datetime"],

        "exit_datetime":
            df.iloc[
                exit_index
            ]["datetime"],

        "direction":
            direction,

        "signal_close":
            safe_float(
                signal_row["close"]
            ),

        "entry_price":
            entry_price,

        "exit_price":
            exit_price,

        "atr":
            atr,

        "tp_atr":
            tp_atr,

        "sl_atr":
            sl_atr,

        "outcome":
            outcome,

        "result_R":
            result_r,

        "MFE_ATR":
            mfe,

        "MAE_ATR":
            mae,

        "minutes_to_target":
            minutes_to_target,

        "same_candle":
            same_candle
    }


# ============================================================
# COSTRUZIONE TRADE
# ============================================================

def build_trades(
    df,
    signals,
    tp_atr=DEFAULT_TP_ATR,
    sl_atr=DEFAULT_SL_ATR
):

    signals = ensure_dataframe(
        signals
    )

    if signals.empty:
        return pd.DataFrame()

    trades = []

    for _, row in signals.iterrows():

        signal_index = int(
            row["_signal_index"]
        )

        trade = simulate_trade(
            df=df,
            signal_index=signal_index,
            tp_atr=tp_atr,
            sl_atr=sl_atr
        )

        if trade is not None:

            trades.append(
                trade
            )

    return ensure_dataframe(
        trades
    )


# ============================================================
# SIGNAL MODES
# ============================================================

def get_all_signals(df):

    result = df[
        df["signal"] != 0
    ].copy()

    result["_signal_index"] = (
        result.index
    )

    return result


def get_first_episode_signals(df):

    result = first_in_episode(
        df
    )

    if result.empty:
        return result

    result["_signal_index"] = (
        result.index
    )

    return result


def get_non_overlapping_signals(
    df,
    tp_atr=DEFAULT_TP_ATR,
    sl_atr=DEFAULT_SL_ATR
):

    signals = get_all_signals(
        df
    )

    if signals.empty:
        return signals

    selected = []

    next_allowed = -1

    for _, row in signals.iterrows():

        signal_index = int(
            row["_signal_index"]
        )

        if signal_index < next_allowed:
            continue

        selected.append(
            row
        )

        trade = simulate_trade(
            df=df,
            signal_index=signal_index,
            tp_atr=tp_atr,
            sl_atr=sl_atr
        )

        if trade is not None:

            next_allowed = (
                int(
                    trade["exit_index"]
                ) + 1
            )

    return ensure_dataframe(
        selected
    )


def build_signal_modes(df):

    modes = {
        "ALL":
            get_all_signals(df),

        "FIRST_IN_EPISODE":
            get_first_episode_signals(df),

        "NON_OVERLAPPING":
            get_non_overlapping_signals(df)
    }

    print_separator(
        "MODALITÀ SEGNALI"
    )

    for name, signals in modes.items():

        print(
            f"{name}: {len(signals)}"
        )

    return modes


# ============================================================
# ENRICH TRADE
# ============================================================

def enrich_trades(
    trades,
    df
):

    trades = ensure_dataframe(
        trades
    )

    if trades.empty:
        return trades

    feature_cols = [
        "rsi",
        "atr_ratio",
        "body_atr",
        "ema_distance",
        "ema20_slope",
        "ema50_slope",
        "momentum_3",
        "momentum_6",
        "momentum_12",
        "momentum_24",
        "macd_hist",
        "hour_utc",
        "day_of_week",
        "bullish_15",
        "bearish_15",
        "bullish_1h",
        "bearish_1h"
    ]

    available = [
        c
        for c in feature_cols
        if c in df.columns
    ]

    source = df[
        ["datetime"] + available
    ].copy()

    trades = trades.merge(
        source,
        on="datetime",
        how="left"
    )

    return trades


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    trades,
    name="ALL"
):

    trades = ensure_dataframe(
        trades
    )

    if trades.empty:

        return {
            "dataset": name,
            "trades": 0,
            "TP": 0,
            "SL": 0,
            "TIMEOUT": 0,
            "EXCLUDE": 0,
            "win_rate_pct": np.nan,
            "avg_R": np.nan,
            "total_R": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_R": 0.0,
            "avg_MFE_ATR": np.nan,
            "avg_MAE_ATR": np.nan
        }

    tp = int(
        (
            trades["outcome"] ==
            "TP"
        ).sum()
    )

    sl = int(
        (
            trades["outcome"] ==
            "SL"
        ).sum()
    )

    timeout = int(
        (
            trades["outcome"] ==
            "TIMEOUT"
        ).sum()
    )

    exclude = int(
        (
            trades["outcome"] ==
            "EXCLUDE"
        ).sum()
    )

    decided = tp + sl

    win_rate = (
        tp / decided * 100
        if decided
        else np.nan
    )

    results = pd.to_numeric(
        trades["result_R"],
        errors="coerce"
    ).dropna()

    return {
        "dataset": name,

        "trades":
            len(trades),

        "TP":
            tp,

        "SL":
            sl,

        "TIMEOUT":
            timeout,

        "EXCLUDE":
            exclude,

        "win_rate_pct":
            win_rate,

        "avg_R":
            results.mean()
            if len(results)
            else np.nan,

        "total_R":
            results.sum()
            if len(results)
            else 0.0,

        "profit_factor":
            profit_factor(
                results.values
            ),

        "max_drawdown_R":
            max_drawdown(
                results.values
            ),

        "avg_MFE_ATR":
            trades[
                "MFE_ATR"
            ].mean(),

        "avg_MAE_ATR":
            trades[
                "MAE_ATR"
            ].mean()
    }


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def get_split_datetime(df):

    position = int(
        len(df) *
        DEVELOPMENT_PCT
    )

    position = max(
        1,
        min(
            position,
            len(df) - 1
        )
    )

    return df.iloc[
        position
    ]["datetime"]


def split_trades(
    trades,
    split_datetime
):

    trades = ensure_dataframe(
        trades
    )

    if trades.empty:

        return (
            trades.copy(),
            trades.copy()
        )

    development = trades[
        trades["signal_datetime"] <=
        split_datetime
    ].copy()

    verification = trades[
        trades["signal_datetime"] >
        split_datetime
    ].copy()

    return (
        development,
        verification
    )


# ============================================================
# GENERAL ANALYSIS
# ============================================================

def general_analysis(
    df,
    signals,
    split_datetime
):

    trades = build_trades(
        df,
        signals
    )

    trades = enrich_trades(
        trades,
        df
    )

    development, verification = (
        split_trades(
            trades,
            split_datetime
        )
    )

    result = pd.DataFrame([
        summarize(
            development,
            "DEVELOPMENT"
        ),
        summarize(
            verification,
            "VERIFICATION"
        ),
        summarize(
            trades,
            "ALL"
        )
    ])

    return (
        trades,
        development,
        verification,
        result
    )


# ============================================================
# BUY / SELL
# ============================================================

def direction_analysis(
    trades,
    split_datetime
):

    development, verification = (
        split_trades(
            trades,
            split_datetime
        )
    )

    rows = []

    for dataset_name, subset in [
        (
            "DEVELOPMENT",
            development
        ),
        (
            "VERIFICATION",
            verification
        ),
        (
            "ALL",
            trades
        )
    ]:

        for direction in [
            "BUY",
            "SELL"
        ]:

            part = subset[
                subset["direction"] ==
                direction
            ]

            row = summarize(
                part,
                f"{dataset_name}_{direction}"
            )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FILTRI DIAGNOSTICI
# ============================================================

def add_condition_columns(
    trades
):

    trades = ensure_dataframe(
        trades
    ).copy()

    if trades.empty:
        return trades

    trades["ema_strength"] = np.select(
        [
            trades[
                "ema_distance"
            ].abs() < 0.25,

            trades[
                "ema_distance"
            ].abs() < 0.50,

            trades[
                "ema_distance"
            ].abs() < 1.00
        ],
        [
            "VERY_WEAK",
            "WEAK",
            "MEDIUM"
        ],
        default="STRONG"
    )

    trades["body_bucket"] = np.select(
        [
            trades["body_atr"] < 0.25,
            trades["body_atr"] < 0.50,
            trades["body_atr"] < 1.00
        ],
        [
            "VERY_SMALL",
            "SMALL",
            "MEDIUM"
        ],
        default="LARGE"
    )

    trades["atr_regime"] = np.select(
        [
            trades["atr_ratio"] < 0.75,
            trades["atr_ratio"] > 1.25
        ],
        [
            "LOW",
            "HIGH"
        ],
        default="NORMAL"
    )

    trades["momentum_6_sign"] = np.where(
        trades["momentum_6"] >= 0,
        "POSITIVE",
        "NEGATIVE"
    )

    trades["momentum_12_sign"] = np.where(
        trades["momentum_12"] >= 0,
        "POSITIVE",
        "NEGATIVE"
    )

    trades["rsi_bucket"] = pd.cut(
        trades["rsi"],
        bins=[
            -np.inf,
            30,
            40,
            50,
            60,
            65,
            np.inf
        ],
        labels=[
            "<30",
            "30-40",
            "40-50",
            "50-60",
            "60-65",
            ">65"
        ]
    )

    return trades


def condition_analysis(
    trades,
    split_datetime
):

    trades = add_condition_columns(
        trades
    )

    if trades.empty:
        return pd.DataFrame()

    conditions = {

        "ALL":
            np.ones(
                len(trades),
                dtype=bool
            ),

        "ATR_NORMAL":
            trades["atr_regime"] ==
            "NORMAL",

        "ATR_HIGH":
            trades["atr_regime"] ==
            "HIGH",

        "EMA_STRONG":
            trades["ema_strength"] ==
            "STRONG",

        "EMA_MEDIUM_STRONG":
            trades["ema_strength"].isin(
                [
                    "MEDIUM",
                    "STRONG"
                ]
            ),

        "BODY_NOT_TINY":
            trades["body_bucket"].isin(
                [
                    "SMALL",
                    "MEDIUM",
                    "LARGE"
                ]
            ),

        "MOMENTUM_6_POSITIVE":
            trades["momentum_6"] > 0,

        "MOMENTUM_6_NEGATIVE":
            trades["momentum_6"] < 0,

        "MOMENTUM_12_POSITIVE":
            trades["momentum_12"] > 0,

        "MOMENTUM_12_NEGATIVE":
            trades["momentum_12"] < 0,

        "EMA_STRONG_AND_ATR_NORMAL":
            (
                (
                    trades[
                        "ema_strength"
                    ] == "STRONG"
                )
                &
                (
                    trades[
                        "atr_regime"
                    ] == "NORMAL"
                )
            ),

        "EMA_STRONG_AND_MOM6":
            (
                (
                    trades[
                        "ema_strength"
                    ] == "STRONG"
                )
                &
                (
                    trades[
                        "momentum_6"
                    ] > 0
                )
            )
    }

    rows = []

    for name, mask in conditions.items():

        subset = trades.loc[
            mask
        ]

        dev = subset[
            subset[
                "signal_datetime"
            ] <= split_datetime
        ]

        ver = subset[
            subset[
                "signal_datetime"
            ] > split_datetime
        ]

        row = {
            "condition": name
        }

        dev_summary = summarize(
            dev,
            "DEV"
        )

        ver_summary = summarize(
            ver,
            "VER"
        )

        row.update({
            f"dev_{k}": v
            for k, v in dev_summary.items()
            if k != "dataset"
        })

        row.update({
            f"ver_{k}": v
            for k, v in ver_summary.items()
            if k != "dataset"
        })

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TP / SL MATRIX
# ============================================================

def evaluate_matrix(
    df,
    signals,
    split_datetime
):

    signals = ensure_dataframe(
        signals
    )

    if signals.empty:
        return pd.DataFrame()

    rows = []

    datasets = [
        (
            "DEVELOPMENT",
            signals[
                signals["datetime"]
                <= split_datetime
            ]
        ),
        (
            "VERIFICATION",
            signals[
                signals["datetime"]
                > split_datetime
            ]
        )
    ]

    for dataset_name, subset in datasets:

        for direction in [
            "BUY",
            "SELL"
        ]:

            direction_signals = subset[
                subset["direction"] ==
                direction
            ]

            for tp in TP_VALUES:

                for sl in SL_VALUES:

                    trades = []

                    for _, row in (
                        direction_signals.iterrows()
                    ):

                        signal_index = int(
                            row["_signal_index"]
                        )

                        trade = simulate_trade(
                            df=df,
                            signal_index=signal_index,
                            tp_atr=tp,
                            sl_atr=sl
                        )

                        if trade is not None:

                            trades.append(
                                trade
                            )

                    summary = summarize(
                        trades,
                        dataset_name
                    )

                    summary[
                        "direction"
                    ] = direction

                    summary[
                        "TP_ATR"
                    ] = tp

                    summary[
                        "SL_ATR"
                    ] = sl

                    rows.append(
                        summary
                    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TARGET REACH ANALYSIS
# ============================================================

def target_analysis(
    df,
    signals,
    split_datetime
):

    rows = []

    datasets = [
        (
            "DEVELOPMENT",
            signals[
                signals["datetime"]
                <= split_datetime
            ]
        ),
        (
            "VERIFICATION",
            signals[
                signals["datetime"]
                > split_datetime
            ]
        )
    ]

    for dataset_name, subset in datasets:

        for direction in [
            "BUY",
            "SELL"
        ]:

            direction_signals = subset[
                subset["direction"] ==
                direction
            ]

            for target in TP_VALUES:

                reached = 0
                times = []

                for _, row in (
                    direction_signals.iterrows()
                ):

                    signal_index = int(
                        row["_signal_index"]
                    )

                    if (
                        signal_index
                        >= len(df) - 1
                    ):
                        continue

                    entry_index = (
                        signal_index + 1
                    )

                    entry_price = safe_float(
                        df.iloc[
                            entry_index
                        ]["open"]
                    )

                    atr = safe_float(
                        row["atr"]
                    )

                    if (
                        not np.isfinite(
                            entry_price
                        )
                        or
                        not np.isfinite(
                            atr
                        )
                        or
                        atr <= 0
                    ):
                        continue

                    end = min(
                        len(df) - 1,
                        entry_index +
                        HORIZON_BARS
                    )

                    found = False

                    for j in range(
                        entry_index,
                        end + 1
                    ):

                        high = safe_float(
                            df.iloc[j]["high"]
                        )

                        low = safe_float(
                            df.iloc[j]["low"]
                        )

                        if direction == "BUY":

                            favorable = (
                                high -
                                entry_price
                            ) / atr

                        else:

                            favorable = (
                                entry_price -
                                low
                            ) / atr

                        if favorable >= target:

                            reached += 1

                            times.append(
                                (
                                    j -
                                    entry_index
                                ) * 5
                            )

                            found = True

                            break

                    if found:
                        continue

                count = len(
                    direction_signals
                )

                rows.append({

                    "dataset":
                        dataset_name,

                    "direction":
                        direction,

                    "target_ATR":
                        target,

                    "signals":
                        count,

                    "reached":
                        reached,

                    "reach_pct":
                        (
                            reached /
                            count *
                            100
                            if count
                            else np.nan
                        ),

                    "avg_minutes":
                        (
                            np.mean(times)
                            if times
                            else np.nan
                        ),

                    "median_minutes":
                        (
                            np.median(times)
                            if times
                            else np.nan
                        )
                })

    return pd.DataFrame(
        rows
    )


# ============================================================
# TIME HORIZONS V5
# ============================================================

def horizon_analysis(
    df,
    signals
):

    horizons_minutes = [
        15,
        30,
        60,
        120,
        240,
        480,
        720,
        1440
    ]

    rows = []

    if signals is None or signals.empty:

        return pd.DataFrame(
            columns=[
                "dataset",
                "trades",
                "TP",
                "SL",
                "TIMEOUT",
                "EXCLUDE",
                "win_rate_pct",
                "avg_R",
                "total_R",
                "profit_factor",
                "max_drawdown_R",
                "avg_MFE_ATR",
                "avg_MAE_ATR",
                "direction",
                "horizon_minutes",
                "horizon_bars"
            ]
        )

    for direction in [
        "BUY",
        "SELL"
    ]:

        direction_signals = signals[
            signals["direction"] ==
            direction
        ]

        if direction_signals.empty:
            continue

        for minutes in horizons_minutes:

            bars = max(
                1,
                minutes // 5
            )

            trades = []

            for _, row in (
                direction_signals.iterrows()
            ):

                try:

                    signal_index = int(
                        row["_signal_index"]
                    )

                except (
                    KeyError,
                    TypeError,
                    ValueError
                ):

                    continue

                trade = simulate_trade(
                    df=df,
                    signal_index=signal_index,
                    tp_atr=DEFAULT_TP_ATR,
                    sl_atr=DEFAULT_SL_ATR,
                    horizon=bars
                )

                if trade is not None:

                    trades.append(
                        trade
                    )

            summary = summarize(
                trades,
                f"{direction}_{minutes}m"
            )

            summary[
                "direction"
            ] = direction

            summary[
                "horizon_minutes"
            ] = minutes

            summary[
                "horizon_bars"
            ] = bars

            rows.append(
                summary
            )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        result = result.sort_values(
            [
                "direction",
                "horizon_minutes"
            ]
        ).reset_index(
            drop=True
        )

    return result


# ============================================================
# SALVATAGGIO CSV
# ============================================================

def save_csv(
    data,
    filename
):

    data = ensure_dataframe(
        data
    )

    if data.empty:

        print(
            f"Nessun dato da salvare: {filename}"
        )

        return

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    data.to_csv(
        path,
        index=False
    )

    print(
        f"Salvato: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print_separator(
        "XAU/USD BACKTEST V5"
    )

    print(
        f"Symbol:       {SYMBOL}"
    )

    print(
        f"Interval:     {INTERVAL}"
    )

    print(
        f"Candele:      {TOTAL_CANDLES}"
    )

    print(
        f"Development:  {DEVELOPMENT_PCT * 100:.0f}%"
    )

    print(
        f"Verification: {(1 - DEVELOPMENT_PCT) * 100:.0f}%"
    )

    print(
        f"TP base:      {DEFAULT_TP_ATR} ATR"
    )

    print(
        f"SL base:      {DEFAULT_SL_ATR} ATR"
    )

    print(
        f"Horizon:      {HORIZON_BARS * 5 / 60:.1f} ore"
    )

    print(
        f"Same candle:  {SAME_CANDLE_MODE}"
    )

    # --------------------------------------------------------
    # 1. DOWNLOAD
    # --------------------------------------------------------

    df = download_history()

    # --------------------------------------------------------
    # 2. INDICATORI
    # --------------------------------------------------------

    print_separator(
        "CALCOLO INDICATORI"
    )

    df = calculate_indicators(
        df
    )

    # --------------------------------------------------------
    # 3. MULTI TIMEFRAME
    # --------------------------------------------------------

    df = add_higher_timeframes(
        df
    )

    # --------------------------------------------------------
    # 4. SEGNALI
    # --------------------------------------------------------

    df = generate_signals(
        df
    )

    # --------------------------------------------------------
    # 5. SPLIT TEMPORALE
    # --------------------------------------------------------

    split_datetime = get_split_datetime(
        df
    )

    print_separator(
        "SPLIT TEMPORALE"
    )

    print(
        "Split:",
        split_datetime
    )

    print(
        "Development:",
        df["datetime"].iloc[0],
        "->",
        split_datetime
    )

    print(
        "Verification:",
        split_datetime,
        "->",
        df["datetime"].iloc[-1]
    )

    # --------------------------------------------------------
    # 6. SIGNAL MODES
    # --------------------------------------------------------

    modes = build_signal_modes(
        df
    )

    # --------------------------------------------------------
    # 7. ANALISI PRINCIPALE
    # --------------------------------------------------------

    print_separator(
        "ANALISI STRATEGIA BASE"
    )

    signals = modes["ALL"]

    (
        trades,
        development,
        verification,
        general
    ) = general_analysis(
        df,
        signals,
        split_datetime
    )

    print_table(
        "SUMMARY GENERALE",
        general
    )

    # --------------------------------------------------------
    # 8. BUY / SELL
    # --------------------------------------------------------

    direction = direction_analysis(
        trades,
        split_datetime
    )

    print_table(
        "ANALISI BUY / SELL",
        direction
    )

    # --------------------------------------------------------
    # 9. FILTRI DIAGNOSTICI
    # --------------------------------------------------------

    conditions = condition_analysis(
        trades,
        split_datetime
    )

    print_table(
        "DIAGNOSTICA FILTRI",
        conditions
    )

    # --------------------------------------------------------
    # 10. TP / SL MATRIX
    # --------------------------------------------------------

    print_separator(
        "MATRICE TP / SL"
    )

    matrix = evaluate_matrix(
        df,
        signals,
        split_datetime
    )

    print_table(
        "MATRICE TP / SL",
        matrix
    )

    # --------------------------------------------------------
    # 11. TARGET REACH
    # --------------------------------------------------------

    print_separator(
        "TARGET REACH ANALYSIS"
    )

    targets = target_analysis(
        df,
        signals,
        split_datetime
    )

    print_table(
        "TARGET REACH",
        targets
    )

    # --------------------------------------------------------
    # 12. TIME HORIZONS
    # --------------------------------------------------------

    print_separator(
        "TIME HORIZON ANALYSIS"
    )

    horizons = horizon_analysis(
        df,
        signals
    )

    print_table(
        "TIME HORIZONS",
        horizons
    )

    # --------------------------------------------------------
    # 13. SALVATAGGIO
    # --------------------------------------------------------

    print_separator(
        "SALVATAGGIO RISULTATI"
    )

    save_csv(
        df,
        "candles_with_indicators.csv"
    )

    save_csv(
        signals,
        "signals_all.csv"
    )

    save_csv(
        trades,
        "trades_all.csv"
    )

    save_csv(
        development,
        "trades_development.csv"
    )

    save_csv(
        verification,
        "trades_verification.csv"
    )

    save_csv(
        general,
        "summary_general.csv"
    )

    save_csv(
        direction,
        "summary_direction.csv"
    )

    save_csv(
        conditions,
        "diagnostic_conditions.csv"
    )

    save_csv(
        matrix,
        "matrix_tp_sl.csv"
    )

    save_csv(
        targets,
        "target_analysis.csv"
    )

    save_csv(
        horizons,
        "horizon_analysis.csv"
    )

    # --------------------------------------------------------
    # 14. RIEPILOGO MODALITÀ
    # --------------------------------------------------------

    mode_rows = []

    for name, mode_signals in modes.items():

        mode_trades = build_trades(
            df,
            mode_signals
        )

        mode_trades = enrich_trades(
            mode_trades,
            df
        )

        mode_dev, mode_ver = (
            split_trades(
                mode_trades,
                split_datetime
            )
        )

        summary_dev = summarize(
            mode_dev,
            f"{name}_DEVELOPMENT"
        )

        summary_ver = summarize(
            mode_ver,
            f"{name}_VERIFICATION"
        )

        mode_rows.append({
            "mode":
                name,

            "development_trades":
                summary_dev["trades"],

            "development_win_rate":
                summary_dev[
                    "win_rate_pct"
                ],

            "development_avg_R":
                summary_dev[
                    "avg_R"
                ],

            "development_total_R":
                summary_dev[
                    "total_R"
                ],

            "development_PF":
                summary_dev[
                    "profit_factor"
                ],

            "verification_trades":
                summary_ver["trades"],

            "verification_win_rate":
                summary_ver[
                    "win_rate_pct"
                ],

            "verification_avg_R":
                summary_ver[
                    "avg_R"
                ],

            "verification_total_R":
                summary_ver[
                    "total_R"
                ],

            "verification_PF":
                summary_ver[
                    "profit_factor"
                ]
        })

    modes_summary = pd.DataFrame(
        mode_rows
    )

    print_table(
        "CONFRONTO SIGNAL MODES",
        modes_summary
    )

    save_csv(
        modes_summary,
        "signal_modes_summary.csv"
    )

    # --------------------------------------------------------
    # 15. FINE
    # --------------------------------------------------------

    elapsed = (
        time.time() -
        start_time
    )

    print_separator(
        "BACKTEST V5 COMPLETATO"
    )

    print(
        f"Tempo totale: {elapsed:.2f} secondi"
    )

    print(
        f"Risultati salvati in: {OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print_separator(
            "ERRORE BACKTEST V5"
        )

        print(
            repr(e)
        )

        traceback.print_exc()

        raise
