"""
VIX vs Support Ticket Correlation — Streamlit App
==================================================
Requirements:
    pip install streamlit pandas numpy scipy matplotlib openpyxl

Run:
    streamlit run vix_ticket_app.py
"""

import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import streamlit as st

warnings.filterwarnings("ignore")

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BLUE  = "#1A5276"
C_RED   = "#C0392B"
C_TEAL  = "#148F77"
C_GREY  = "#717D7E"
C_BG    = "#F8F9FA"
C_AMBER = "#F39C12"
C_GREEN = "#1E8449"


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_vix(file) -> pd.DataFrame:
    df = pd.read_excel(file, engine="openpyxl")
    df.columns = [c.strip().upper() for c in df.columns]
    date_col = next((c for c in df.columns if "DATE" in c), df.columns[0])
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, infer_datetime_format=True)
    for col in ["OPEN", "HIGH", "LOW", "CLOSE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "CLOSE"]).sort_values("date").reset_index(drop=True)


@st.cache_data
def load_tickets(file) -> pd.DataFrame:
    df = pd.read_excel(file, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in ("date", "day")), df.columns[0])
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, infer_datetime_format=True)
    for col in ["ticket_count", "active_client_count", "avg_ticket_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if ("avg_ticket_count" not in df.columns
            and "ticket_count" in df.columns
            and "active_client_count" in df.columns):
        df["avg_ticket_count"] = df["ticket_count"] / df["active_client_count"]
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def deseasonalise(df: pd.DataFrame, col: str) -> pd.Series:
    grand_mean = df[col].mean()
    dow_means  = df.groupby("dow")[col].transform("mean")
    mon_means  = df.groupby("month_num")[col].transform("mean")
    dow_factor = dow_means / grand_mean
    mon_factor = mon_means / grand_mean
    return df[col] / (dow_factor * mon_factor)


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def compute_ccf(x: pd.Series, y: pd.Series, max_lag: int = 10):
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = x.iloc[:lag].reset_index(drop=True)
            b = y.iloc[-lag:].reset_index(drop=True)
        elif lag == 0:
            a, b = x.reset_index(drop=True), y.reset_index(drop=True)
        else:
            a = x.iloc[lag:].reset_index(drop=True)
            b = y.iloc[:-lag].reset_index(drop=True)
        valid = pd.concat([a, b], axis=1).dropna()
        if len(valid) < 10:
            rows.append({"lag": lag, "r": np.nan, "p": np.nan})
            continue
        r, p = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        rows.append({"lag": lag, "r": r, "p": p})
    return pd.DataFrame(rows)


def build_corr_table(df: pd.DataFrame, ticket_col: str) -> pd.DataFrame:
    vix_vars   = ["CLOSE", "vix_range", "vix_change", "vix_pct"]
    vix_labels = ["VIX Close (level)", "VIX Range H-L", "VIX Day Change", "VIX % Change"]
    rows = []
    for var, lbl in zip(vix_vars, vix_labels):
        if var not in df.columns:
            continue
        clean = df[[var, ticket_col]].dropna()
        if len(clean) < 5:
            continue
        pr, pp = stats.pearsonr(clean[var], clean[ticket_col])
        sr, sp = stats.spearmanr(clean[var], clean[ticket_col])
        rows.append({
            "VIX Metric":   lbl,
            "Pearson r":    round(pr, 3),
            "Pearson p":    round(pp, 4),
            "Spearman r":   round(sr, 3),
            "Spearman p":   round(sp, 4),
            "Significant?": "Yes" if pp < 0.05 else "No",
        })
    return pd.DataFrame(rows)


def strength_label(r: float):
    a = abs(r)
    if a >= 0.6:   return "strong",     "red"
    elif a >= 0.4: return "moderate",   "orange"
    elif a >= 0.2: return "weak",       "yellow"
    else:          return "negligible", "grey"


# ══════════════════════════════════════════════════════════════════════════════
# Chart helpers
# ══════════════════════════════════════════════════════════════════════════════

def fig_timeseries(df):
    fig, ax1 = plt.subplots(figsize=(13, 3.8), facecolor=C_BG)
    ax2 = ax1.twinx()
    ax1.plot(df["date"], df["tickets_z"], color=C_TEAL, lw=1.3, alpha=0.85,
             label="Tickets - deseasonalised (z-score)")
    ax2.plot(df["date"], df["vix_z"],     color=C_RED,  lw=1.0, alpha=0.70,
             label="VIX Close (z-score)")
    ax1.set_ylabel("Tickets z-score", color=C_TEAL, fontsize=10)
    ax2.set_ylabel("VIX z-score",     color=C_RED,  fontsize=10)
    ax1.set_title("Deseasonalised Tickets vs VIX Close", fontsize=12, fontweight="bold")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs  = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labs, loc="upper right", fontsize=8)
    ax1.set_facecolor(C_BG)
    fig.tight_layout()
    return fig


def fig_ccf(ccf_df, ci):
    best_lag = int(ccf_df.loc[ccf_df["r"].abs().idxmax(), "lag"])
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=C_BG)
    colors = [C_RED if l == best_lag else (C_BLUE if r > 0 else C_GREY)
              for l, r in zip(ccf_df["lag"], ccf_df["r"])]
    ax.bar(ccf_df["lag"], ccf_df["r"], color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline( ci, ls="--", lw=1, color="black", alpha=0.5, label=f"95% CI +/-{ci:.3f}")
    ax.axhline(-ci, ls="--", lw=1, color="black", alpha=0.5)
    ax.axhline(0,   lw=0.5,  color="black")
    ax.set_xlabel("Lag (trading days) - negative = VIX leads tickets", fontsize=9)
    ax.set_ylabel("Pearson r", fontsize=9)
    ax.set_title(f"Cross-Correlation Function  -  Peak at lag {best_lag}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_facecolor(C_BG)
    fig.tight_layout()
    return fig, best_lag


def fig_scatter(df, ticket_col, vix_col="CLOSE"):
    clean = df[[vix_col, ticket_col, "dow"]].dropna()
    slope, intercept, r_val, p_val, _ = stats.linregress(clean[vix_col], clean[ticket_col])
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=C_BG)
    sc = ax.scatter(clean[vix_col], clean[ticket_col],
                    c=clean["dow"], cmap="cool", alpha=0.45, s=16, edgecolors="none")
    x_fit = np.linspace(clean[vix_col].min(), clean[vix_col].max(), 200)
    ax.plot(x_fit, intercept + slope * x_fit, color=C_RED, lw=2,
            label=f"OLS  r2={r_val**2:.3f}  p={p_val:.4f}")
    plt.colorbar(sc, ax=ax, label="Day of week (0=Mon)")
    ax.set_xlabel("VIX Close", fontsize=10)
    ax.set_ylabel("Avg tickets/client (deseasonalised)", fontsize=10)
    ax.set_title("Scatter: VIX Level vs Deseasonalised Tickets", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_facecolor(C_BG)
    fig.tight_layout()
    return fig, r_val, p_val, slope, intercept


def fig_seasonal(df, raw_col):
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    mon_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    dow_means = df.groupby("dow")[raw_col].mean()
    mon_means = df.groupby("month_num")[raw_col].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.5), facecolor=C_BG)

    bar_colors_dow = [C_AMBER if i == dow_means.values.argmax() else C_TEAL
                      for i in range(len(dow_means))]
    ax1.bar([dow_labels[i] for i in dow_means.index], dow_means.values,
            color=bar_colors_dow, edgecolor="white")
    ax1.axhline(dow_means.mean(), color=C_RED, lw=1.5, ls="--", label="Mean")
    ax1.set_title("Day-of-Week Seasonal Pattern", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean avg tickets/client", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.set_facecolor(C_BG)

    bar_colors_mon = [C_AMBER if i == mon_means.values.argmax() else C_BLUE
                      for i in range(len(mon_means))]
    ax2.bar([mon_labels[i-1] for i in mon_means.index], mon_means.values,
            color=bar_colors_mon, edgecolor="white")
    ax2.axhline(mon_means.mean(), color=C_RED, lw=1.5, ls="--", label="Mean")
    ax2.set_title("Annual Seasonal Pattern", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean avg tickets/client", fontsize=9)
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(fontsize=8)
    ax2.set_facecolor(C_BG)

    fig.tight_layout()
    return fig


def fig_vix_dist(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.5), facecolor=C_BG)
    ax1.hist(df["CLOSE"], bins=40, color=C_BLUE, edgecolor="white", alpha=0.8)
    ax1.axvline(df["CLOSE"].median(), color=C_RED, lw=2, ls="--",
                label=f"Median {df['CLOSE'].median():.1f}")
    ax1.set_title("VIX Close Distribution", fontsize=11, fontweight="bold")
    ax1.set_xlabel("VIX Close")
    ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=8)
    ax1.set_facecolor(C_BG)
    ax2.plot(df["date"], df["CLOSE"], color=C_RED, lw=0.9, alpha=0.8)
    ax2.fill_between(df["date"], df["LOW"], df["HIGH"],
                     color=C_RED, alpha=0.15, label="Daily H-L range")
    ax2.set_title("VIX Over Time (with H-L Band)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("VIX")
    ax2.legend(fontsize=8)
    ax2.set_facecolor(C_BG)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Plain-English summary
# ══════════════════════════════════════════════════════════════════════════════

def render_summary(df, corr_tbl, ccf_df, r_val, p_val, slope, ticket_col_raw, max_lag):

    close_row  = corr_tbl[corr_tbl["VIX Metric"] == "VIX Close (level)"]
    range_row  = corr_tbl[corr_tbl["VIX Metric"] == "VIX Range H-L"]
    change_row = corr_tbl[corr_tbl["VIX Metric"] == "VIX Day Change"]

    r_close  = float(close_row["Pearson r"].values[0])  if len(close_row)  else 0.0
    r_range  = float(range_row["Pearson r"].values[0])  if len(range_row)  else 0.0
    r_change = float(change_row["Pearson r"].values[0]) if len(change_row) else 0.0
    p_close  = float(close_row["Pearson p"].values[0])  if len(close_row)  else 1.0

    best_lag = int(ccf_df.loc[ccf_df["r"].abs().idxmax(), "lag"])
    ci       = 1.96 / np.sqrt(len(df))
    sig_lags = ccf_df[ccf_df["r"].abs() > ci]["lag"].tolist()

    strength, _ = strength_label(r_close)

    dow_means  = df.groupby("dow")[ticket_col_raw].mean()
    mon_means  = df.groupby("month_num")[ticket_col_raw].mean()
    peak_dow   = ["Monday","Tuesday","Wednesday","Thursday","Friday"][int(dow_means.idxmax())]
    low_dow    = ["Monday","Tuesday","Wednesday","Thursday","Friday"][int(dow_means.idxmin())]
    mon_map    = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                  7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
    peak_month = mon_map[int(mon_means.idxmax())]
    low_month  = mon_map[int(mon_means.idxmin())]
    dow_swing  = (dow_means.max() - dow_means.min()) / dow_means.mean() * 100
    mon_swing  = (mon_means.max() - mon_means.min()) / mon_means.mean() * 100

    vix_mean   = df["CLOSE"].mean()
    vix_median = df["CLOSE"].median()
    date_start = df["date"].min().strftime("%d %b %Y")
    date_end   = df["date"].max().strftime("%d %b %Y")
    n_days     = len(df)

    st.markdown("---")
    st.markdown(
        "<h2 style='color:#1A5276'>Plain-English Summary</h2>"
        "<p style='color:#717D7E'>A jargon-free explanation of what the analysis found.</p>",
        unsafe_allow_html=True,
    )

    # 1. What was analysed
    st.markdown("### 1 · What was analysed")
    st.info(
        f"This analysis compared **{n_days:,} trading days** of data "
        f"from **{date_start}** to **{date_end}**. "
        f"The VIX (often called the 'fear index') measures how nervous financial markets are — "
        f"a higher number means more uncertainty. The average VIX over this period was "
        f"**{vix_mean:.1f}** (median **{vix_median:.1f}**). "
        f"Before comparing to VIX, the ticket data was cleaned of its regular weekly and "
        f"seasonal patterns, so those rhythms don't create false results."
    )

    # 2. Seasonal patterns
    st.markdown("### 2 · Seasonal patterns in ticket volume")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"**Day-of-week pattern**\n\n"
            f"Ticket volumes follow a clear weekly rhythm. **{peak_dow}** is the busiest day "
            f"and **{low_dow}** is the quietest — a swing of roughly **{dow_swing:.0f}%** "
            f"between the two. This pattern was removed before any VIX comparison."
        )
    with col2:
        st.markdown(
            f"**Time-of-year pattern**\n\n"
            f"Volumes also vary through the year. **{peak_month}** tends to be the busiest month "
            f"and **{low_month}** the quietest — a swing of roughly **{mon_swing:.0f}%**. "
            f"This annual cycle was also stripped out before the VIX comparison."
        )

    # 3. Is there a relationship?
    st.markdown("### 3 · Is there a relationship between VIX and ticket volume?")
    if p_close < 0.05:
        st.success(
            f"**Yes — there is a statistically significant {strength} relationship.** "
            f"When market volatility is higher, support ticket volume also tends to be higher, "
            f"even after removing seasonal patterns. The correlation is **r = {r_close:.3f}** "
            f"(p {'< 0.0001' if p_close < 0.0001 else f'= {p_close:.4f}'}), "
            f"which means the result is very unlikely to be due to chance."
        )
    else:
        st.warning(
            f"**No clear relationship was found after removing seasonal patterns.** "
            f"The correlation coefficient is **r = {r_close:.3f}** (p = {p_close:.4f}), "
            f"which is {strength} and not statistically reliable. "
            f"VIX does not appear to be a meaningful driver of your ticket volume."
        )

    # 4. Which VIX measure matters most?
    st.markdown("### 4 · Which aspect of VIX matters most?")
    metrics_info = [
        ("The overall VIX level (close price)",  r_close,
         "Captures the broad fear regime — whether markets are generally calm or stressed over days/weeks."),
        ("The daily high-low range",             r_range,
         "Captures intraday panic — how wild the swings were within a single day."),
        ("The day-over-day change in VIX",       r_change,
         "Captures sudden moves — whether fear spiked or dropped compared to yesterday."),
    ]
    icons = {
        "strong":     "🔴",
        "moderate":   "🟠",
        "weak":       "🟡",
        "negligible": "⚪",
    }
    for name, r, explanation in metrics_info:
        s, _ = strength_label(r)
        direction = "positive" if r > 0 else "negative"
        icon = icons[s]
        st.markdown(
            f"**{icon} {name}** — {s} {direction} link (r = {r:.3f}). {explanation}"
        )

    st.markdown(
        "> **Rule of thumb:** if the *level* matters most, clients react to sustained periods "
        "of market stress. If the *daily change* matters most, they react to sudden shocks. "
        "If neither is significant, VIX is probably not a useful predictor."
    )

    # 5. Timing — does VIX lead?
    st.markdown("### 5 · Does market volatility come before the ticket spike?")
    if best_lag < 0:
        st.success(
            f"**VIX moves before tickets do.** The strongest correlation occurs at a lag of "
            f"**{abs(best_lag)} trading day(s)** — a rise in VIX today tends to be followed "
            f"by more support tickets roughly {abs(best_lag)} day(s) later. "
            f"This is useful: it suggests VIX could work as an **early warning signal** "
            f"for upcoming ticket volume spikes, giving you time to prepare."
        )
    elif best_lag == 0:
        st.info(
            "**VIX and tickets move together on the same day.** "
            "There is no meaningful lead or lag — the relationship is simultaneous. "
            "VIX is unlikely to be a forecasting tool in this case, but it does confirm "
            "a same-day co-movement with your ticket volume."
        )
    else:
        st.warning(
            f"**Ticket volume appears to move before VIX** (peak at lag +{best_lag} days). "
            "This is unusual and worth investigating — it could indicate a data anomaly "
            "or that your clients are reacting to something that later flows through to markets."
        )

    sig_lag_str = ", ".join([str(l) for l in sorted(sig_lags)]) if sig_lags else "none"
    st.caption(f"Lags outside the 95% confidence band: {sig_lag_str}")

    # 6. Practical meaning
    st.markdown("### 6 · What does this mean in practice?")
    if p_close < 0.05 and abs(r_close) >= 0.2:
        vix10_impact = slope * 10
        r2_pct = r_val ** 2 * 100
        st.markdown(
            f"Each **1-point rise in VIX** is associated with a change of **{slope:+.4f}** "
            f"in the average number of tickets per client per day (seasonally adjusted).\n\n"
            f"To put that in context: if VIX rises from 20 to 30 (a significant stress event), "
            f"you might expect roughly a **{vix10_impact:+.3f} tickets/client/day** change in volume.\n\n"
            f"VIX explains about **{r2_pct:.1f}%** of the day-to-day variation in your "
            f"deseasonalised ticket volume. The remaining **{100 - r2_pct:.1f}%** is driven "
            f"by other factors not captured here."
        )
    else:
        st.markdown(
            "The relationship is too weak to be practically useful for forecasting. "
            "Other factors — such as product releases, onboarding cycles, or operational changes — "
            "are likely more important drivers of your ticket volume than market volatility."
        )

    # 7. Next steps
    st.markdown("### 7 · Recommended next steps")
    recs = []

    if p_close < 0.05 and abs(r_close) >= 0.3 and best_lag < 0:
        recs.append(
            f"**Build a simple alert.** Monitor VIX daily. If it rises above a threshold "
            f"(e.g. VIX > 25 or 30), flag the next {abs(best_lag)} day(s) as elevated-risk "
            f"and consider pre-emptive staffing or proactive client comms."
        )
    if p_close < 0.05 and abs(r_close) >= 0.2:
        recs.append(
            "**Add VIX as a feature in a forecasting model.** Even a modest correlation "
            "can improve a regression or ML model when combined with other predictors "
            "like day-of-week, month, and active client count."
        )
    if abs(r_range) > abs(r_close):
        recs.append(
            "**Focus on intraday VIX range (H-L) rather than the close level.** "
            "Your data suggests daily volatility swings matter more than where VIX settled."
        )
    recs.append(
        "**Investigate other drivers.** VIX explains only part of your ticket variation. "
        "Consider overlaying product release dates, onboarding waves, or outage events "
        "to build a more complete picture."
    )
    recs.append(
        "**Download the deseasonalised CSV** (in the Correlation Table tab) to run "
        "your own regressions or share results with stakeholders."
    )

    for i, rec in enumerate(recs, 1):
        st.markdown(f"{i}. {rec}")


# ══════════════════════════════════════════════════════════════════════════════
# Page config & layout
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VIX vs Support Tickets",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 VIX vs Tickets")
    st.markdown("---")
    st.markdown("### 📁 Upload your files")

    vix_file = st.file_uploader(
        "VIX history  (VIXhistory.xlsx)",
        type=["xlsx", "xls"],
        help="Expected columns: DATE, OPEN, HIGH, LOW, CLOSE",
    )
    ticket_file = st.file_uploader(
        "Jira tickets  (Jira Daily Ticket Count.xlsx)",
        type=["xlsx", "xls"],
        help="Expected columns: Date, ticket_count, active_client_count, avg_ticket_count",
    )

    st.markdown("---")
    st.markdown("### Settings")
    max_lag = st.slider("Max CCF lag (trading days)", 5, 20, 10)
    use_avg = st.radio(
        "Ticket metric",
        ["avg_ticket_count (per client)", "ticket_count (raw)"],
        index=0,
        help="Per-client metric adjusts for changes in client count over time.",
    )
    ticket_col_raw = "avg_ticket_count" if "avg" in use_avg else "ticket_count"

    st.markdown("---")
    st.markdown("### Correlation strength guide")
    st.markdown(
        "| r value | Strength |\n"
        "|---|---|\n"
        "| 0.6+ | Strong 🔴 |\n"
        "| 0.4–0.6 | Moderate 🟠 |\n"
        "| 0.2–0.4 | Weak 🟡 |\n"
        "| < 0.2 | Negligible ⚪ |"
    )
    st.markdown(
        "**Lag guide**\n"
        "- Negative = VIX leads tickets\n"
        "- Zero = simultaneous\n"
        "- Positive = tickets lead VIX"
    )


# ── Page header ─────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#1A5276;margin-bottom:0'>📊 VIX vs Support Ticket Correlation</h1>"
    "<p style='color:#717D7E;font-size:1.05rem;margin-top:4px'>"
    "Does market volatility drive support ticket volume? Upload your files to find out."
    "</p>"
    "<hr style='border:1px solid #D5D8DC;margin:12px 0 20px 0'>",
    unsafe_allow_html=True,
)

vix_ready    = vix_file    is not None
ticket_ready = ticket_file is not None

# ══════════════════════════════════════════════════════════════════════════════
# Upload landing page (shown until both files are loaded)
# ══════════════════════════════════════════════════════════════════════════════

if not vix_ready or not ticket_ready:

    st.markdown("## Step 1 — Upload your two Excel files")
    st.markdown(
        "Use the **sidebar on the left** to upload both files. "
        "The analysis will run automatically once both are loaded."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        border = "#1E8449" if vix_ready else "#C0392B"
        status = "Uploaded" if vix_ready else "Waiting for upload..."
        st.markdown(
            f"<div style='border:2px solid {border};border-radius:10px;"
            f"padding:22px;background:#fff;min-height:200px'>"
            f"<h3 style='margin:0 0 6px 0'>📈 VIX History File</h3>"
            f"<p style='color:#717D7E;margin:0 0 14px 0'>Filename: <code>VIXhistory.xlsx</code></p>"
            f"<p style='margin:0 0 6px 0'><strong>Required columns:</strong></p>"
            f"<ul style='margin:0 0 14px 0;padding-left:18px;color:#444'>"
            f"<li>DATE</li><li>OPEN</li><li>HIGH</li><li>LOW</li><li>CLOSE</li>"
            f"</ul>"
            f"<p style='font-size:1.05rem;margin:0'>"
            f"{'<strong style=\"color:#1E8449\">Uploaded</strong>' if vix_ready else '<em style=\"color:#C0392B\">Waiting for upload...</em>'}"
            f"</p></div>",
            unsafe_allow_html=True,
        )

    with col_b:
        border = "#1E8449" if ticket_ready else "#C0392B"
        st.markdown(
            f"<div style='border:2px solid {border};border-radius:10px;"
            f"padding:22px;background:#fff;min-height:200px'>"
            f"<h3 style='margin:0 0 6px 0'>🎫 Jira Ticket Count File</h3>"
            f"<p style='color:#717D7E;margin:0 0 14px 0'>Filename: <code>Jira Daily Ticket Count.xlsx</code></p>"
            f"<p style='margin:0 0 6px 0'><strong>Required columns:</strong></p>"
            f"<ul style='margin:0 0 14px 0;padding-left:18px;color:#444'>"
            f"<li>Date</li><li>ticket_count</li><li>active_client_count</li>"
            f"<li>avg_ticket_count</li><li>month</li>"
            f"</ul>"
            f"<p style='font-size:1.05rem;margin:0'>"
            f"{'<strong style=\"color:#1E8449\">Uploaded</strong>' if ticket_ready else '<em style=\"color:#C0392B\">Waiting for upload...</em>'}"
            f"</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("## What this tool does")

    steps = [
        ("1. Removes seasonal noise",
         "Strips out predictable weekly (Mon-Fri) and annual patterns from your ticket data "
         "so they can't create misleading correlations with VIX."),
        ("2. Tests four VIX measures",
         "Compares your tickets against the VIX level, intraday high-low range, "
         "day-over-day change, and percentage change — each tells a different story."),
        ("3. Checks timing with cross-correlation",
         "Determines whether VIX rises *before* ticket spikes (making it a leading indicator), "
         "at the same time, or after."),
        ("4. Quantifies the relationship",
         "Runs both Pearson and Spearman correlation tests with p-values so you know "
         "whether any relationship is statistically meaningful or could be random chance."),
        ("5. Delivers a plain-English summary",
         "Translates every chart and statistic into clear findings and actionable next steps "
         "— no data science background needed."),
    ]

    for title, desc in steps:
        st.markdown(f"**{title}** — {desc}")

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Load, validate & process
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading and processing files..."):
    vix_df    = load_vix(vix_file)
    ticket_df = load_tickets(ticket_file)

overlap_start = max(vix_df["date"].min(), ticket_df["date"].min())
overlap_end   = min(vix_df["date"].max(), ticket_df["date"].max())

if overlap_start >= overlap_end:
    st.error("The two files have no overlapping date range. Please check your data.")
    st.stop()

df = (pd.merge(vix_df, ticket_df, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True))
df["dow"]        = df["date"].dt.dayofweek
df["month_num"]  = df["date"].dt.month
df["vix_range"]  = df["HIGH"] - df["LOW"]
df["vix_change"] = df["CLOSE"].diff()
df["vix_pct"]    = df["CLOSE"].pct_change() * 100

if ticket_col_raw not in df.columns:
    st.error(f"Column '{ticket_col_raw}' not found in the ticket file. Check column names.")
    st.stop()

df["tickets_deseas"] = deseasonalise(df, ticket_col_raw)
df["tickets_z"]      = zscore(df["tickets_deseas"])
df["vix_z"]          = zscore(df["CLOSE"])
ticket_col = "tickets_deseas"


# ── Dataset overview ─────────────────────────────────────────────────────────────
st.subheader("Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trading days (merged)", f"{len(df):,}")
c2.metric("Date range",
          f"{df['date'].min().strftime('%d %b %Y')} to {df['date'].max().strftime('%d %b %Y')}")
c3.metric("VIX Close mean", f"{df['CLOSE'].mean():.1f}")
c4.metric("Avg tickets/client mean", f"{df[ticket_col_raw].mean():.3f}")

with st.expander("Preview merged data (first 10 rows)"):
    show_cols = ["date","OPEN","HIGH","LOW","CLOSE","vix_range", ticket_col_raw,"tickets_deseas"]
    st.dataframe(df[[c for c in show_cols if c in df.columns]].head(10),
                 use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# Pre-compute shared results
# ══════════════════════════════════════════════════════════════════════════════

ccf_df   = compute_ccf(df["CLOSE"], df[ticket_col], max_lag=max_lag)
ci       = 1.96 / np.sqrt(len(df))
corr_tbl = build_corr_table(df, ticket_col)
_, r_val, p_val, slope, intercept = fig_scatter(df, ticket_col)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis tabs
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Time Series",
    "🔁 Cross-Correlation",
    "🔵 Scatter & OLS",
    "📅 Seasonality",
    "📊 Correlation Table",
])

with tab1:
    st.markdown("#### Deseasonalised Tickets vs VIX — Full Period")
    st.caption(
        "Both series shown as z-scores (0 = long-run average, +1 = one standard deviation above). "
        "This lets two very different scales share one axis."
    )
    st.pyplot(fig_timeseries(df), use_container_width=True)
    st.markdown("#### VIX History")
    st.pyplot(fig_vix_dist(df), use_container_width=True)

with tab2:
    st.markdown("#### Cross-Correlation Function (VIX Close vs Deseasonalised Tickets)")
    st.caption(
        "Negative lags = VIX leads tickets. Red bar = peak correlation lag. "
        "Dashed lines = 95% significance bands."
    )
    ccf_fig, best_lag = fig_ccf(ccf_df, ci)
    st.pyplot(ccf_fig, use_container_width=True)
    if best_lag < 0:
        st.success(f"Peak at lag **{best_lag}** — VIX leads tickets by {abs(best_lag)} trading day(s).")
    elif best_lag == 0:
        st.info("Peak at lag **0** — VIX and tickets move together on the same day.")
    else:
        st.warning(f"Peak at lag **+{best_lag}** — tickets appear to lead VIX.")
    with st.expander("CCF values table"):
        st.dataframe(ccf_df.style.format({"r": "{:.3f}", "p": "{:.4f}"}),
                     use_container_width=True)

with tab3:
    st.markdown("#### VIX Close vs Deseasonalised Tickets — Scatter + OLS")
    st.caption("Points coloured by day-of-week (0=Mon...4=Fri). Red line = OLS fit.")
    scatter_fig, r_val, p_val, slope, intercept = fig_scatter(df, ticket_col)
    st.pyplot(scatter_fig, use_container_width=True)
    s1, s2, s3 = st.columns(3)
    s1.metric("Pearson r2", f"{r_val**2:.3f}")
    s2.metric("OLS slope (per VIX point)", f"{slope:+.5f}")
    s3.metric("p-value", "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}")
    if p_val < 0.05:
        st.success(
            f"Statistically significant (p={'< 0.0001' if p_val < 0.0001 else f'{p_val:.4f}'}). "
            f"Each 1-point VIX rise is associated with a {slope:+.5f} change in "
            f"deseasonalised avg tickets/client."
        )
    else:
        st.warning(f"Not statistically significant at p < 0.05 (p={p_val:.4f}).")

with tab4:
    st.markdown("#### Seasonal Patterns Removed Before Correlation")
    st.caption(
        "Amber bars = peak period. These effects were stripped from the ticket series "
        "before any VIX comparison to prevent spurious correlations."
    )
    st.pyplot(fig_seasonal(df, ticket_col_raw), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        dow_labels = ["Mon","Tue","Wed","Thu","Fri"]
        dow_tbl = df.groupby("dow")[ticket_col_raw].agg(["mean","std","count"]).reset_index()
        dow_tbl["dow"] = dow_tbl["dow"].map(dict(enumerate(dow_labels)))
        dow_tbl.columns = ["Day","Mean","Std Dev","N"]
        st.dataframe(dow_tbl.style.format({"Mean":"{:.3f}","Std Dev":"{:.3f}"}),
                     use_container_width=True)
    with col_b:
        mon_map_abbr = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                        7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        mon_tbl = df.groupby("month_num")[ticket_col_raw].agg(["mean","std","count"]).reset_index()
        mon_tbl["month_num"] = mon_tbl["month_num"].map(mon_map_abbr)
        mon_tbl.columns = ["Month","Mean","Std Dev","N"]
        st.dataframe(mon_tbl.style.format({"Mean":"{:.3f}","Std Dev":"{:.3f}"}),
                     use_container_width=True)

with tab5:
    st.markdown("#### Pearson & Spearman Correlations — All VIX Metrics")
    st.caption(
        "Green = r >= 0.4 (moderate+), Yellow = 0.2-0.4, Red = < 0.2. "
        "Both tests shown for robustness."
    )

    def colour_r(val):
        try:
            v = abs(float(val))
            if v >= 0.4:   return "background-color:#D5F5E3"
            elif v >= 0.2: return "background-color:#FEF9E7"
            else:          return "background-color:#FDEDEC"
        except Exception:
            return ""

    styled = (corr_tbl.style
              .applymap(colour_r, subset=["Pearson r", "Spearman r"])
              .format({"Pearson r": "{:.3f}", "Pearson p": "{:.4f}",
                       "Spearman r": "{:.3f}", "Spearman p": "{:.4f}"}))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")
    with st.expander("Download deseasonalised data as CSV"):
        csv_cols = ["date","OPEN","HIGH","LOW","CLOSE","vix_range","vix_change","vix_pct",
                    ticket_col_raw,"tickets_deseas","tickets_z","vix_z","dow","month_num"]
        csv_buf = io.StringIO()
        df[[c for c in csv_cols if c in df.columns]].to_csv(csv_buf, index=False)
        st.download_button("Download CSV", csv_buf.getvalue(),
                           "vix_tickets_deseasonalised.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# Plain-English Summary — always rendered below the tabs
# ══════════════════════════════════════════════════════════════════════════════

render_summary(df, corr_tbl, ccf_df, r_val, p_val, slope, ticket_col_raw, max_lag)