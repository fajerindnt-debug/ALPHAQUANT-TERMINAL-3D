
# ALPHAQUANT TERMINAL

Interactive 3D quantitative research dashboard built with
Python, Streamlit, Plotly, NumPy and pandas.

## Features

- Interactive 3D quant response surface
- Animated data trajectory with Play/Pause and frame slider
- Price, momentum and volatility charts
- Descriptive volatility and drawdown metrics
- CSV upload and processed-data export
- Dark terminal interface with gold and cyan accents
- Synthetic demonstration mode

## Data format

Upload a CSV containing a `price` column.

Optional columns:
- `time`
- `volume`

Example:

```csv
time,price,volume
T001,2350.25,1200
T002,2351.10,1350
T003,2349.80,1100
```

The example values are illustrative, not live market prices.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open https://share.streamlit.io/
3. Connect your GitHub account.
4. Select this repository.
5. Set the main file to `app.py`.
6. Deploy.

## Important

The default dataset is synthetic. This project does not connect
to a live market feed, provide validated trading signals, or
execute broker orders.
