# ================================================================
# ALPHAQUANT TERMINAL — DATA-DRIVEN 3D QUANT DASHBOARD
# Streamlit version | CSV upload | 3D animation | HTML export
# Default data is SYNTHETIC DEMO DATA, never presented as live data.
# CSV: price (required), time (optional), volume (optional)
# ================================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------ PAGE / CONFIG ------------------------
st.set_page_config(
    page_title="ALPHAQUANT TERMINAL",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG = "#070A10"
PANEL = "#101722"
GOLD = "#F5C451"
CYAN = "#31D7E8"
WHITE = "#EAF0F8"
MUTED = "#91A0B5"
HTML_FILE = "alphaquant_terminal_3d.html"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {BG}; color: {WHITE}; }}
    [data-testid="stHeader"] {{ background: rgba(7,10,16,0.95); }}
    [data-testid="stSidebar"] {{ background: {PANEL}; }}
    h1, h2, h3 {{ color: {GOLD} !important; }}
    div[data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid #2A3749;
        padding: 14px; border-radius: 12px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    div[data-testid="stMetricValue"] {{ color: {GOLD}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="padding:8px 0 14px;font-family:Arial,sans-serif">
      <div style="font-size:12px;color:{CYAN};letter-spacing:3px">
        QUANT RESEARCH • 3D DATA ENGINE
      </div>
      <div style="font-size:clamp(25px,4vw,38px);font-weight:800;color:{GOLD};margin-top:5px">
        ALPHAQUANT TERMINAL
      </div>
      <div style="font-size:12px;color:{MUTED};margin-top:5px">
        DATA-DRIVEN 3D QUANT INTELLIGENCE LAB
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------ SIDEBAR / DATA INPUT -----------------
st.sidebar.markdown("## DATA CONTROL")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV data",
    type=["csv"],
    help="Required column: price. Optional columns: time and volume.",
)
use_demo = st.sidebar.checkbox(
    "Use synthetic demo data when no CSV is uploaded",
    value=True,
)
st.sidebar.caption(
    "Synthetic data is for software demonstration only. "
    "It is not live XAU/USD market data."
)

# ------------------------ LOAD DATA ----------------------------
@st.cache_data(show_spinner=False)
def read_csv_data(file_bytes):
    from io import BytesIO

    raw = pd.read_csv(BytesIO(file_bytes))
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "price" not in raw.columns:
        raise ValueError(
            "CSV harus memiliki kolom 'price'. "
            "Ubah nama kolom harga menjadi price lalu upload ulang."
        )

    result = pd.DataFrame()
    result["price"] = pd.to_numeric(raw["price"], errors="coerce")

    if "time" in raw.columns:
        result["time"] = raw["time"].astype(str)
    else:
        result["time"] = np.arange(len(raw)).astype(str)

    if "volume" in raw.columns:
        result["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    else:
        result["volume"] = np.nan

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=["price"]).reset_index(drop=True)
    return result


if uploaded_file is not None:
    try:
        data = read_csv_data(uploaded_file.getvalue())
        DATA_MODE = "USER CSV"
    except Exception as exc:
        st.error(f"Gagal membaca CSV: {exc}")
        st.stop()
elif use_demo:
    rng = np.random.default_rng(42)
    n = 180
    time_index = np.arange(n)
    innovations = rng.normal(0, 0.55, n)
    synthetic_returns = 0.035 + 0.32 * np.sin(time_index / 10) + innovations
    synthetic_price = 2350 + np.cumsum(synthetic_returns)
    data = pd.DataFrame({
        "time": [f"T{i:03d}" for i in range(n)],
        "price": synthetic_price,
        "volume": rng.lognormal(mean=5.0, sigma=0.35, size=n),
    })
    DATA_MODE = "SYNTHETIC DEMO"
else:
    st.info("Upload CSV atau aktifkan synthetic demo data di sidebar.")
    st.stop()

if len(data) < 12:
    st.error("Diperlukan minimal 12 observasi harga yang valid.")
    st.stop()

if not np.isfinite(data["price"].to_numpy(dtype=float)).all():
    st.error("Kolom price mengandung nilai non-finite yang tidak valid.")
    st.stop()

data["return"] = data["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
data["momentum"] = data["price"].diff(5).fillna(0)
data["volatility"] = data["return"].rolling(12, min_periods=2).std().fillna(0)
data["index"] = np.arange(len(data))

# ------------------------ METRICS -------------------------------
latest_price = float(data["price"].iloc[-1])
first_price = float(data["price"].iloc[0])
price_change = ((latest_price / first_price) - 1) * 100 if first_price != 0 else 0.0
volatility_proxy = float(data["return"].std() * np.sqrt(252) * 100)
latest_momentum = float(data["momentum"].iloc[-1])
drawdown = data["price"] / data["price"].cummax() - 1
max_drawdown = float(drawdown.min() * 100)

m1, m2, m3, m4 = st.columns(4)
m1.metric("LATEST VALUE", f"{latest_price:,.4f}")
m2.metric("TOTAL CHANGE", f"{price_change:+.2f}%")
m3.metric("VOLATILITY PROXY", f"{volatility_proxy:.2f}%")
m4.metric("MAX DRAWDOWN", f"{max_drawdown:.2f}%")

st.markdown(
    f"""
    <div style="margin:8px 0 16px;padding:11px 13px;border:1px solid #26364B;
                border-radius:8px;color:{MUTED};font-size:12px">
      DATA MODE: <b style="color:{GOLD}">{DATA_MODE}</b>
      &nbsp; | &nbsp; OBSERVATIONS: <b style="color:{WHITE}">{len(data)}</b>
      &nbsp; | &nbsp; PRICE SOURCE:
      <b style="color:{WHITE}">{'Uploaded CSV' if DATA_MODE == 'USER CSV' else 'Synthetic simulation'}</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------ CUSTOM COLORS ------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("## VISUAL SETTINGS")
bg_color = st.sidebar.color_picker("Background", BG)
panel_color = st.sidebar.color_picker("Panel / secondary background", PANEL)
gold_color = st.sidebar.color_picker("Primary / price line", GOLD)
cyan_color = st.sidebar.color_picker("Trajectory / momentum", CYAN)
white_color = st.sidebar.color_picker("Text / neutral", WHITE)
vol_color = st.sidebar.color_picker("Volatility line", "#FF9C6E")
surface_low = st.sidebar.color_picker("3D surface low", "#111C2B")
surface_mid = st.sidebar.color_picker("3D surface middle", "#087E87")
surface_high = st.sidebar.color_picker("3D surface high", "#FFE6A0")

# ------------------------ 3D QUANT SURFACE ---------------------
vol = data["volatility"].to_numpy(dtype=float)
mom = data["momentum"].to_numpy(dtype=float)
price = data["price"].to_numpy(dtype=float)
x_idx = data["index"].to_numpy(dtype=float)

factor_axis = np.linspace(-2, 2, 35)
X, Y = np.meshgrid(x_idx, factor_axis)

price_center = float(np.nanmedian(price))
price_scale = float(np.nanstd(price))
if not np.isfinite(price_scale) or price_scale == 0:
    price_scale = 1.0
price_norm = (price - price_center) / price_scale

vol_scale = float(np.nanstd(vol))
if not np.isfinite(vol_scale) or vol_scale == 0:
    vol_scale = 1.0
vol_norm = (vol - np.nanmean(vol)) / vol_scale

mom_scale = float(np.nanstd(mom))
if not np.isfinite(mom_scale) or mom_scale == 0:
    mom_scale = 1.0
mom_norm = (mom - np.nanmean(mom)) / mom_scale

surface_rows = []
for factor in factor_axis:
    z = price_norm + 0.22 * factor * vol_norm + 0.12 * (factor**2 - 1) * mom_norm
    surface_rows.append(z)
Z = np.asarray(surface_rows)

path_y = np.clip(0.45 * vol_norm + 0.25 * mom_norm, -1.8, 1.8)
path_z = price_norm + 0.22 * path_y * vol_norm + 0.12 * (path_y**2 - 1) * mom_norm + 0.35

fig = go.Figure()
fig.add_trace(go.Surface(
    x=X, y=Y, z=Z,
    colorscale=[
        [0.00, surface_low],
        [0.48, surface_mid],
        [1.00, surface_high],
    ],
    opacity=0.88,
    showscale=False,
    name="Data-derived quant surface",
    hovertemplate=(
        "Observation %{x:.0f}<br>Factor %{y:.2f}"
        "<br>Normalized response %{z:.3f}<extra></extra>"
    ),
))

initial_end = min(4, len(data))
fig.add_trace(go.Scatter3d(
    x=x_idx[:initial_end], y=path_y[:initial_end], z=path_z[:initial_end],
    mode="lines",
    line=dict(color=cyan_color, width=7),
    name="Data trajectory",
    customdata=price[:initial_end],
    hovertemplate="Observation %{x}<br>Price %{customdata:.4f}<extra></extra>",
))
fig.add_trace(go.Scatter3d(
    x=[x_idx[initial_end - 1]],
    y=[path_y[initial_end - 1]],
    z=[path_z[initial_end - 1]],
    mode="markers",
    marker=dict(size=7, color=gold_color, line=dict(color=white_color, width=2)),
    name="Current data point",
    customdata=[price[initial_end - 1]],
    hovertemplate="CURRENT POINT<br>Price %{customdata:.4f}<extra></extra>",
))

n_frames = min(65, max(12, len(data) - 3))
ends = np.unique(np.linspace(3, len(data), n_frames, dtype=int))
frames = []
for frame_no, end in enumerate(ends):
    end = int(end)
    frames.append(go.Frame(
        name=str(frame_no),
        data=[
            go.Scatter3d(
                x=x_idx[:end], y=path_y[:end], z=path_z[:end],
                customdata=price[:end], mode="lines",
                line=dict(color=cyan_color, width=7), name="Data trajectory",
            ),
            go.Scatter3d(
                x=[x_idx[end - 1]], y=[path_y[end - 1]], z=[path_z[end - 1]],
                customdata=[price[end - 1]], mode="markers",
                marker=dict(size=7, color=gold_color, line=dict(color=white_color, width=2)),
                name="Current data point",
            ),
        ],
        traces=[1, 2],
    ))
fig.frames = frames

fig.update_layout(
    title=dict(
        text="<b>3D QUANT MANIFOLD</b><br><sup>PRICE × VOLATILITY × MOMENTUM • DATA-DRIVEN</sup>",
        x=0.025, y=0.97, font=dict(color=gold_color, size=21),
    ),
    paper_bgcolor=bg_color,
    plot_bgcolor=bg_color,
    font=dict(color=white_color, family="Arial"),
    height=760,
    margin=dict(l=0, r=0, t=100, b=35),
    legend=dict(bgcolor="rgba(7,10,16,0.65)", bordercolor="#253246", borderwidth=1),
    scene=dict(
        bgcolor=bg_color,
        xaxis=dict(title="OBSERVATION", color=cyan_color, backgroundcolor=bg_color, gridcolor="#26364B"),
        yaxis=dict(title="VOLATILITY / FACTOR STATE", color=gold_color, backgroundcolor=bg_color, gridcolor="#26364B"),
        zaxis=dict(title="NORMALIZED RESPONSE", color=white_color, backgroundcolor=bg_color, gridcolor="#26364B"),
        camera=dict(eye=dict(x=1.65, y=1.55, z=1.0)),
        aspectratio=dict(x=1.5, y=1, z=0.8),
    ),
    updatemenus=[dict(
        type="buttons", showactive=False, x=0.02, y=0.02,
        bgcolor=panel_color, bordercolor=gold_color, font=dict(color=gold_color),
        buttons=[
            dict(label="▶ PLAY", method="animate", args=[None, {
                "frame": {"duration": 85, "redraw": True},
                "transition": {"duration": 0},
                "fromcurrent": True, "mode": "immediate",
            }]),
            dict(label="Ⅱ PAUSE", method="animate", args=[[None], {
                "frame": {"duration": 0, "redraw": False},
                "transition": {"duration": 0}, "mode": "immediate",
            }]),
        ],
    )],
    sliders=[dict(
        active=0, x=0.23, y=0.025, len=0.72,
        currentvalue=dict(prefix="DATA FRAME  ", font=dict(color=gold_color)),
        steps=[
            dict(label=str(i + 1), method="animate", args=[[str(i)], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            }])
            for i in range(len(frames))
        ],
        bgcolor=panel_color, bordercolor="#34445C", tickcolor=gold_color,
        font=dict(color=MUTED),
    )],
)
fig.update_layout(uirevision="alphaquant-3d")

st.subheader("3D QUANT MANIFOLD")
st.caption("The 3D surface is a normalized visualization derived from the selected dataset; it is not a forecast or trading signal.")
st.plotly_chart(fig, use_container_width=True, key="quant_3d_chart")

# ------------------------ SUPPORTING DATA PANELS ---------------
tail_n = min(80, len(data))
sub = data.tail(tail_n)

fig2 = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
    subplot_titles=("PRICE / INDEX SERIES", "MOMENTUM AND VOLATILITY PROXIES"),
)
fig2.add_trace(go.Scatter(
    x=sub["index"], y=sub["price"], mode="lines",
    line=dict(color=gold_color, width=2), name="Price",
), row=1, col=1)
fig2.add_trace(go.Bar(
    x=sub["index"], y=sub["momentum"],
    marker_color=cyan_color, name="5-step momentum",
), row=2, col=1)
fig2.add_trace(go.Scatter(
    x=sub["index"], y=sub["volatility"], mode="lines",
    line=dict(color=vol_color, width=2), name="Volatility proxy",
), row=2, col=1)
fig2.update_layout(
    title=dict(text="<b>QUANT DATA MONITOR</b>", font=dict(color=gold_color, size=19)),
    paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=white_color),
    height=600, margin=dict(l=35, r=25, t=75, b=35),
    legend=dict(orientation="h", y=1.03, x=0.01, bgcolor="rgba(0,0,0,0)"),
)
fig2.update_xaxes(gridcolor="#26364B", color=MUTED)
fig2.update_yaxes(gridcolor="#26364B", color=MUTED)
st.subheader("QUANT DATA MONITOR")
st.plotly_chart(fig2, use_container_width=True, key="quant_data_monitor")

# ------------------------ DATA TABLE ---------------------------
with st.expander("View source data and derived features"):
    st.dataframe(data.tail(100), use_container_width=True, hide_index=True)

# ------------------------ EXPORT STANDALONE HTML ---------------
# Embed Plotly.js so the downloaded HTML works offline after download.
html_3d = fig.to_html(full_html=False, include_plotlyjs=True, auto_play=False)
html_2d = fig2.to_html(full_html=False, include_plotlyjs=False, auto_play=False)
summary = f"""
<div style="background:{bg_color};color:{white_color};padding:24px;font-family:Arial,sans-serif">
  <div style="font-size:12px;letter-spacing:3px;color:{cyan_color}">QUANT RESEARCH • 3D DATA ENGINE</div>
  <h1 style="color:{gold_color};margin:8px 0">ALPHAQUANT TERMINAL</h1>
  <p style="color:{MUTED}">Mode: <b>{DATA_MODE}</b> | Observations: <b>{len(data)}</b> |
  Latest value: <b>{latest_price:.4f}</b> | Total change: <b>{price_change:+.2f}%</b></p>
  <p style="color:{MUTED}">Charts and metrics are derived from the selected dataset. Synthetic demo data is not live market data.</p>
</div>
"""
full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ALPHAQUANT TERMINAL</title>
<style>
body{{margin:0;background:{bg_color};color:{white_color};font-family:Arial,sans-serif}}
.wrap{{max-width:1500px;margin:auto;padding:12px}}
</style></head><body><div class="wrap">{summary}{html_3d}{html_2d}</div></body></html>"""

st.download_button(
    label="⬇️ Download standalone HTML dashboard",
    data=full_html.encode("utf-8"),
    file_name=HTML_FILE,
    mime="text/html",
    use_container_width=True,
)

st.caption(
    "Note: volatility is an annualized proxy based on the observation returns and √252; "
    "this convention is most meaningful for daily observations. "
    "Maximum drawdown here is computed on the price series, not on a strategy equity curve."
)
