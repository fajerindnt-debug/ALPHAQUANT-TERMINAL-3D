# ================================================================
# ALPHAQUANT TERMINAL — DATA-DRIVEN 3D QUANT DASHBOARD
# Google Colab / Jupyter / browser via exported HTML
# IMPORTANT: default source is SYNTHETIC. Upload a CSV to use your data.
# CSV can contain: time (optional), price (required), volume (optional)
# ================================================================


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.colab import files
import os

# ------------------------ CONFIG ------------------------
np.random.seed(42)
HTML_FILE = "alphaquant_terminal_3d.html"
BG = "#070A10"
PANEL = "#101722"
GOLD = "#F5C451"
CYAN = "#31D7E8"
WHITE = "#EAF0F8"
MUTED = "#91A0B5"

# ------------------------ LOAD DATA ---------------------
# Option 1: Upload a CSV with a 'price' column.
# Optional columns: 'time' and 'volume'.
# Option 2: Skip upload to create clearly labeled synthetic data.
print("Optional: upload a CSV containing a 'price' column.")
print("If you skip, the dashboard uses SYNTHETIC DEMO DATA.")
try:
    uploaded = files.upload()
except Exception:
    uploaded = {}

csv_files = [name for name in uploaded.keys() if name.lower().endswith(".csv")]

if csv_files:
    raw = pd.read_csv(csv_files[0])
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "price" not in raw.columns:
        raise ValueError("CSV must contain a column named 'price'. Rename your price column to 'price' and run again.")
    data = pd.DataFrame()
    data["price"] = pd.to_numeric(raw["price"], errors="coerce")
    if "time" in raw.columns:
        data["time"] = raw["time"].astype(str)
    else:
        data["time"] = np.arange(len(raw)).astype(str)
    if "volume" in raw.columns:
        data["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    else:
        data["volume"] = np.nan
    data = data.dropna(subset=["price"]).reset_index(drop=True)
    DATA_MODE = "USER CSV"
else:
    n = 180
    time_index = np.arange(n)
    innovations = np.random.normal(0, 0.55, n)
    synthetic_returns = 0.035 + 0.32*np.sin(time_index/10) + innovations
    synthetic_price = 2350 + np.cumsum(synthetic_returns)
    data = pd.DataFrame({
        "time": [f"T{i:03d}" for i in range(n)],
        "price": synthetic_price,
        "volume": np.random.lognormal(mean=5.0, sigma=0.35, size=n)
    })
    DATA_MODE = "SYNTHETIC DEMO"

if len(data) < 12:
    raise ValueError("Need at least 12 valid price observations for this dashboard.")

data["return"] = data["price"].pct_change().fillna(0)
data["momentum"] = data["price"].diff(5).fillna(0)
data["volatility"] = data["return"].rolling(12, min_periods=2).std().fillna(0)
data["index"] = np.arange(len(data))

# ------------------------ METRICS -----------------------
latest_price = float(data["price"].iloc[-1])
price_change = float((data["price"].iloc[-1] / data["price"].iloc[0] - 1) * 100)
volatility_proxy = float(data["return"].std() * np.sqrt(252) * 100)
latest_momentum = float(data["momentum"].iloc[-1])
drawdown = data["price"] / data["price"].cummax() - 1
max_drawdown = float(drawdown.min() * 100)

def metric_card(label, value, color=GOLD, sub="Derived from current dataset"):
    return f"""
    <div style="background:{PANEL};border:1px solid #2A3749;border-radius:12px;padding:15px;min-width:0">
      <div style="color:{MUTED};font-size:11px;letter-spacing:1.3px">{label}</div>
      <div style="color:{color};font-size:25px;font-weight:700;margin-top:8px;overflow-wrap:anywhere">{value}</div>
      <div style="color:{MUTED};font-size:10px;margin-top:6px">{sub}</div>
    </div>"""

display(HTML(f"""
<div style="background:{BG};color:{WHITE};padding:8px 2px 16px;font-family:Arial,sans-serif">
  <div style="font-size:12px;color:{CYAN};letter-spacing:3px">QUANT RESEARCH • 3D DATA ENGINE</div>
  <div style="font-size:30px;font-weight:800;color:{GOLD};margin-top:5px">ALPHAQUANT TERMINAL</div>
  <div style="font-size:12px;color:{MUTED};margin-top:5px">DATA-DRIVEN 3D QUANT INTELLIGENCE LAB</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px;margin-top:18px">
    {metric_card("LATEST VALUE", f"{latest_price:,.4f}", GOLD, "Last valid observation")}
    {metric_card("TOTAL CHANGE", f"{price_change:+.2f}%", CYAN if price_change >= 0 else "#FF7C8A", "First to last observation")}
    {metric_card("VOLATILITY PROXY", f"{volatility_proxy:.2f}%", WHITE, "Std. dev. of returns × √252")}
    {metric_card("MAX DRAWDOWN", f"{max_drawdown:.2f}%", "#FF7C8A", "Peak-to-trough in dataset")}
  </div>
  <div style="margin-top:12px;padding:10px 12px;border:1px solid #26364B;border-radius:8px;color:{MUTED};font-size:12px">
    DATA MODE: <b style="color:{GOLD}">{DATA_MODE}</b>
    &nbsp; | &nbsp; OBSERVATIONS: <b style="color:{WHITE}">{len(data)}</b>
    &nbsp; | &nbsp; PRICE SOURCE: <b style="color:{WHITE}">{'Uploaded CSV' if DATA_MODE == 'USER CSV' else 'Synthetic simulation'}</b>
  </div>
</div>
"""))

# ------------------------ 3D DATA SURFACE ---------------
# X = observation index, Y = volatility state, Z = price response.
# The surface is computed from the loaded dataset; it changes with the data.
window = min(30, max(8, len(data)//4))
vol = data["volatility"].to_numpy()
mom = data["momentum"].to_numpy()
price = data["price"].to_numpy()
x_idx = data["index"].to_numpy()

factor_axis = np.linspace(-2, 2, 35)
X, Y = np.meshgrid(x_idx, factor_axis)

# Normalize factors safely for display; price remains tied to actual input data.
price_center = np.nanmedian(price)
price_scale = np.nanstd(price)
if not np.isfinite(price_scale) or price_scale == 0:
    price_scale = 1.0
price_norm = (price - price_center) / price_scale

vol_scale = np.nanstd(vol)
if not np.isfinite(vol_scale) or vol_scale == 0:
    vol_scale = 1.0
vol_norm = (vol - np.nanmean(vol)) / vol_scale

mom_scale = np.nanstd(mom)
if not np.isfinite(mom_scale) or mom_scale == 0:
    mom_scale = 1.0
mom_norm = (mom - np.nanmean(mom)) / mom_scale

surface_rows = []
for f in factor_axis:
    # Each row is derived from actual price/volatility/momentum observations.
    z = price_norm + 0.22 * f * vol_norm + 0.12 * (f**2 - 1) * mom_norm
    surface_rows.append(z)
Z = np.array(surface_rows)

# 3D animated path sits above the response surface.
path_y = np.clip(0.45 * vol_norm + 0.25 * mom_norm, -1.8, 1.8)
path_z = price_norm + 0.22 * path_y * vol_norm + 0.12 * (path_y**2 - 1) * mom_norm + 0.35

fig = go.Figure()
fig.add_trace(go.Surface(
    x=X, y=Y, z=Z,
    colorscale=[
        [0.00, "#111C2B"], [0.22, "#123F53"],
        [0.48, "#087E87"], [0.72, "#B38B35"], [1.00, "#FFE6A0"]
    ],
    opacity=0.88,
    showscale=False,
    name="Data-derived quant surface",
    hovertemplate="Observation %{x:.0f}<br>Factor %{y:.2f}<br>Normalized response %{z:.3f}<extra></extra>"
))

initial_end = min(4, len(data))
fig.add_trace(go.Scatter3d(
    x=x_idx[:initial_end], y=path_y[:initial_end], z=path_z[:initial_end],
    mode="lines", line=dict(color=CYAN, width=7), name="Data trajectory",
    customdata=price[:initial_end],
    hovertemplate="Observation %{x}<br>Price %{customdata:.4f}<extra></extra>"
))
fig.add_trace(go.Scatter3d(
    x=[x_idx[initial_end-1]], y=[path_y[initial_end-1]], z=[path_z[initial_end-1]],
    mode="markers", marker=dict(size=7, color=GOLD, line=dict(color=WHITE, width=2)),
    name="Current data point", customdata=[price[initial_end-1]],
    hovertemplate="CURRENT POINT<br>Price %{customdata:.4f}<extra></extra>"
))

# Animate through real loaded observations, not a decorative random animation.
n_frames = min(65, max(12, len(data)-3))
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
                line=dict(color=CYAN, width=7), name="Data trajectory"
            ),
            go.Scatter3d(
                x=[x_idx[end-1]], y=[path_y[end-1]], z=[path_z[end-1]],
                customdata=[price[end-1]], mode="markers",
                marker=dict(size=7, color=GOLD, line=dict(color=WHITE, width=2)),
                name="Current data point"
            )
        ],
        traces=[1, 2]
    ))
fig.frames = frames

fig.update_layout(
    title=dict(
        text="<b>3D QUANT MANIFOLD</b><br><sup>PRICE × VOLATILITY × MOMENTUM • DATA-DRIVEN</sup>",
        x=0.025, y=0.97, font=dict(color=GOLD, size=21)
    ),
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color=WHITE, family="Arial"),
    height=760, margin=dict(l=0, r=0, t=100, b=35),
    legend=dict(bgcolor="rgba(7,10,16,0.65)", bordercolor="#253246", borderwidth=1),
    scene=dict(
        bgcolor=BG,
        xaxis=dict(title="OBSERVATION", color=CYAN, backgroundcolor=BG, gridcolor="#26364B"),
        yaxis=dict(title="VOLATILITY / FACTOR STATE", color=GOLD, backgroundcolor=BG, gridcolor="#26364B"),
        zaxis=dict(title="NORMALIZED RESPONSE", color=WHITE, backgroundcolor=BG, gridcolor="#26364B"),
        camera=dict(eye=dict(x=1.65, y=1.55, z=1.0)),
        aspectratio=dict(x=1.5, y=1, z=0.8)
    ),
    updatemenus=[dict(
        type="buttons", showactive=False, x=0.02, y=0.02,
        bgcolor=PANEL, bordercolor=GOLD, font=dict(color=GOLD),
        buttons=[
            dict(label="▶ PLAY", method="animate", args=[None, {
                "frame": {"duration": 85, "redraw": True},
                "transition": {"duration": 0}, "fromcurrent": True, "mode": "immediate"
            }]),
            dict(label="Ⅱ PAUSE", method="animate", args=[[None], {
                "frame": {"duration": 0, "redraw": False},
                "transition": {"duration": 0}, "mode": "immediate"
            }])
        ]
    )],
    sliders=[dict(
        active=0, x=0.23, y=0.025, len=0.72,
        currentvalue=dict(prefix="DATA FRAME  ", font=dict(color=GOLD)),
        steps=[dict(label=str(i+1), method="animate", args=[[str(i)], {
            "mode": "immediate", "frame": {"duration": 0, "redraw": True},
            "transition": {"duration": 0}
        }]) for i in range(len(frames))],
        bgcolor=PANEL, bordercolor="#34445C", tickcolor=GOLD,
        font=dict(color=MUTED)
    )]
)

fig.show()

# ------------------------ SUPPORTING DATA PANELS --------
# These panels are also calculated from the same dataset.
tail_n = min(80, len(data))
sub = data.tail(tail_n)

fig2 = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
    subplot_titles=("PRICE / INDEX SERIES", "MOMENTUM AND VOLATILITY PROXIES")
)
fig2.add_trace(go.Scatter(
    x=sub["index"], y=sub["price"], mode="lines",
    line=dict(color=GOLD, width=2), name="Price"
), row=1, col=1)
fig2.add_trace(go.Bar(
    x=sub["index"], y=sub["momentum"],
    marker_color=CYAN, name="5-step momentum"
), row=2, col=1)
fig2.add_trace(go.Scatter(
    x=sub["index"], y=sub["volatility"],
    mode="lines", line=dict(color="#FF9C6E", width=2), name="Volatility proxy"
), row=2, col=1)
fig2.update_layout(
    title=dict(text="<b>QUANT DATA MONITOR</b>", font=dict(color=GOLD, size=19)),
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=WHITE),
    height=600, margin=dict(l=35, r=25, t=75, b=35),
    legend=dict(orientation="h", y=1.03, x=0.01, bgcolor="rgba(0,0,0,0)")
)
fig2.update_xaxes(gridcolor="#26364B", color=MUTED)
fig2.update_yaxes(gridcolor="#26364B", color=MUTED)
fig2.show()

# ------------------------ EXPORT TO BROWSER --------------
# Export both 3D animation and data monitor to a single standalone HTML page.
html_3d = fig.to_html(full_html=False, include_plotlyjs="cdn", auto_play=False)
html_2d = fig2.to_html(full_html=False, include_plotlyjs=False, auto_play=False)
summary = f"""
<div style="background:{BG};color:{WHITE};padding:24px;font-family:Arial,sans-serif">
  <div style="font-size:12px;letter-spacing:3px;color:{CYAN}">ALPHAQUANT • RESEARCH TERMINAL</div>
  <h1 style="color:{GOLD};margin:8px 0">ALPHAQUANT TERMINAL</h1>
  <p style="color:{MUTED}">Mode: <b>{DATA_MODE}</b> | Observations: <b>{len(data)}</b> |
  Latest value: <b>{latest_price:.4f}</b> | Total change: <b>{price_change:+.2f}%</b></p>
  <p style="color:{MUTED}">The animated trajectory follows the loaded data. Metrics and supporting charts are calculated from the same dataset.</p>
</div>
"""
full_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ALPHAQUANT TERMINAL</title>
<style>body{{margin:0;background:{BG};color:{WHITE};font-family:Arial,sans-serif}} .wrap{{max-width:1500px;margin:auto;padding:12px}} </style>
</head><body><div class="wrap">{summary}{html_3d}{html_2d}</div></body></html>"""
with open(HTML_FILE, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"Saved browser dashboard: {HTML_FILE}")
print("Download the standalone HTML dashboard with the next cell.")
