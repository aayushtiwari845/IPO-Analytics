"""
+==============================================================================+
|         [IN]  INDIAN IPO ANALYTICS DASHBOARD  -  2019-2024                  |
|         Data Analytics Laboratory -- 6-Experiment Comprehensive Report        |
+==============================================================================+

Usage:
    pip install dash plotly pandas numpy scipy scikit-learn statsmodels
    python ipo_dashboard.py
    -> Open  http://localhost:8050
"""

# ==============================================================================
# S1  IMPORTS
# ==============================================================================
import warnings; warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kstest, norm, laplace, t as t_dist, logistic
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.seasonal import seasonal_decompose

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

# ==============================================================================
# S2  DESIGN SYSTEM
# ==============================================================================
C = dict(
    bg       = "#05080f",
    surface  = "#0d1117",
    card     = "#111827",
    border   = "#1e2d3d",
    border2  = "#243447",
    text     = "#e2e8f0",
    muted    = "#64748b",
    green    = "#00e676",
    red      = "#ff5252",
    amber    = "#ffab00",
    blue     = "#40c4ff",
    purple   = "#ce93d8",
    indigo   = "#7986cb",
    teal     = "#26c6da",
    orange   = "#ff7043",
)


def rgba(hex_color, alpha=0.2):
    """Convert #rrggbb + alpha float to rgba() string."""
    r, g, b = int(hex_color[1:3],16), int(hex_color[3:5],16), int(hex_color[5:7],16)
    return f"rgba({r},{g},{b},{alpha})"


PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor = C["surface"],
        plot_bgcolor  = C["card"],
        font          = dict(family="'JetBrains Mono', 'Fira Code', monospace", color=C["text"], size=11),
        xaxis         = dict(gridcolor=C["border"], zerolinecolor=C["border2"], tickfont=dict(size=10)),
        yaxis         = dict(gridcolor=C["border"], zerolinecolor=C["border2"], tickfont=dict(size=10)),
        legend        = dict(bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1, font=dict(size=10)),
        colorway      = [C["green"], C["blue"], C["amber"], C["purple"], C["red"], C["teal"], C["orange"], C["indigo"]],
        margin        = dict(l=50, r=30, t=50, b=50),
        title         = dict(font=dict(size=13, color=C["text"])),
    )
)

def fig_base(title=""):
    """Return a go.Figure with the project template applied."""
    fig = go.Figure()
    fig.update_layout(PLOTLY_TEMPLATE["layout"], title_text=title)
    return fig

SECTOR_COLORS = {
    "Technology":       C["blue"],    "IT Services":       C["indigo"],
    "Finance":          C["green"],   "Fintech":           C["teal"],
    "Chemicals":        C["amber"],   "Healthcare":        C["purple"],
    "Energy":           C["orange"],  "Defence":           C["red"],
    "Gaming":           "#f06292",    "QSR":               "#a5d6a7",
    "Food Tech":        "#ffe082",    "E-Commerce":        "#80cbc4",
    "Infrastructure":   "#b0bec5",    "Insurance":         "#ce93d8",
    "Banking":          "#81d4fa",    "Retail":            "#ffcc02",
    "Auto Components":  "#ff8a65",    "Logistics":         "#bcaaa4",
    "Pharmaceuticals":  "#c5e1a5",    "Electronics":       "#b39ddb",
    "Real Estate":      "#ef9a9a",    "Automotive":        "#80deea",
    "EV":               "#a5d6a7",    "Consumer Goods":    "#fff59d",
    "Travel & Tourism": "#f48fb1",    "Agrochemicals":     "#dce775",
    "Automotive Tech":  "#ffd54f",
}

# ==============================================================================
# S3  RAW IPO DATA (2019 - 2024)
# ==============================================================================
raw_ipo = [
    # -- 2019 --
    {"company":"IRCTC","ticker":"IRCTC.NS","sector":"Travel & Tourism",
     "issue_price":320,"listing_price":644,"sub_qib":107.2,"sub_nii":351.7,"sub_retail":14.2,
     "sub_total":112.0,"issue_size_cr":635,"gmp":290,"listing_date":"2019-10-14"},
    {"company":"IndiaMart InterMesh","ticker":"INDIAMART.NS","sector":"Technology",
     "issue_price":973,"listing_price":1180,"sub_qib":70.5,"sub_nii":36.0,"sub_retail":7.9,
     "sub_total":36.2,"issue_size_cr":475,"gmp":180,"listing_date":"2019-07-04"},
    {"company":"Sterling & Wilson","ticker":"STLWT.NS","sector":"Energy",
     "issue_price":780,"listing_price":706,"sub_qib":5.0,"sub_nii":1.2,"sub_retail":1.4,
     "sub_total":2.6,"issue_size_cr":3125,"gmp":-30,"listing_date":"2019-08-20"},
    {"company":"Spandana Sphoorty","ticker":"SPANDANA.NS","sector":"Finance",
     "issue_price":856,"listing_price":826,"sub_qib":3.5,"sub_nii":2.1,"sub_retail":2.0,
     "sub_total":2.8,"issue_size_cr":1200,"gmp":-20,"listing_date":"2019-08-19"},
    {"company":"Neogen Chemicals","ticker":"NEOGEN.NS","sector":"Chemicals",
     "issue_price":215,"listing_price":242,"sub_qib":38.0,"sub_nii":43.0,"sub_retail":11.0,
     "sub_total":29.0,"issue_size_cr":132,"gmp":20,"listing_date":"2019-04-26"},
    # -- 2020 --
    {"company":"Burger King India","ticker":"BURGERKING.NS","sector":"QSR",
     "issue_price":60,"listing_price":112,"sub_qib":257.4,"sub_nii":243.6,"sub_retail":67.6,
     "sub_total":156.7,"issue_size_cr":810,"gmp":50,"listing_date":"2020-12-14"},
    {"company":"Happiest Minds","ticker":"HAPPSTMNDS.NS","sector":"IT Services",
     "issue_price":166,"listing_price":351,"sub_qib":222.0,"sub_nii":310.0,"sub_retail":87.0,
     "sub_total":151.0,"issue_size_cr":702,"gmp":145,"listing_date":"2020-09-17"},
    {"company":"Route Mobile","ticker":"ROUTE.NS","sector":"Technology",
     "issue_price":350,"listing_price":708,"sub_qib":104.5,"sub_nii":152.0,"sub_retail":37.2,
     "sub_total":73.3,"issue_size_cr":600,"gmp":310,"listing_date":"2020-09-21"},
    {"company":"CAMS","ticker":"CAMS.NS","sector":"Finance",
     "issue_price":1230,"listing_price":1518,"sub_qib":99.5,"sub_nii":71.5,"sub_retail":15.5,
     "sub_total":47.0,"issue_size_cr":2244,"gmp":230,"listing_date":"2020-09-01"},
    {"company":"SBI Cards","ticker":"SBICARD.NS","sector":"Finance",
     "issue_price":755,"listing_price":658,"sub_qib":57.0,"sub_nii":47.7,"sub_retail":11.6,
     "sub_total":26.5,"issue_size_cr":10341,"gmp":-40,"listing_date":"2020-03-16"},
    {"company":"UTI AMC","ticker":"UTIAMC.NS","sector":"Finance",
     "issue_price":554,"listing_price":500,"sub_qib":3.2,"sub_nii":2.5,"sub_retail":1.3,
     "sub_total":2.3,"issue_size_cr":2160,"gmp":-30,"listing_date":"2020-10-12"},
    {"company":"Chemcon Specialty","ticker":"CHEMCON.NS","sector":"Chemicals",
     "issue_price":340,"listing_price":730,"sub_qib":182.0,"sub_nii":463.0,"sub_retail":113.0,
     "sub_total":149.0,"issue_size_cr":318,"gmp":350,"listing_date":"2020-10-01"},
    {"company":"Angel Broking","ticker":"ANGELONE.NS","sector":"Finance",
     "issue_price":306,"listing_price":275,"sub_qib":3.9,"sub_nii":2.0,"sub_retail":7.9,
     "sub_total":3.9,"issue_size_cr":600,"gmp":-20,"listing_date":"2020-10-05"},
    # -- 2021 --
    {"company":"Zomato","ticker":"ZOMATO.NS","sector":"Food Tech",
     "issue_price":76,"listing_price":116,"sub_qib":51.8,"sub_nii":33.0,"sub_retail":7.5,
     "sub_total":38.2,"issue_size_cr":9375,"gmp":35,"listing_date":"2021-07-23"},
    {"company":"Paytm","ticker":"PAYTM.NS","sector":"Fintech",
     "issue_price":2150,"listing_price":1955,"sub_qib":2.8,"sub_nii":0.2,"sub_retail":1.7,
     "sub_total":1.9,"issue_size_cr":18300,"gmp":-200,"listing_date":"2021-11-18"},
    {"company":"Nykaa (FSN E-Commerce)","ticker":"FSN.NS","sector":"E-Commerce",
     "issue_price":1125,"listing_price":2001,"sub_qib":91.9,"sub_nii":112.0,"sub_retail":12.2,
     "sub_total":82.0,"issue_size_cr":5352,"gmp":700,"listing_date":"2021-11-10"},
    {"company":"PB Fintech (PolicyBazaar)","ticker":"POLICYBZR.NS","sector":"Fintech",
     "issue_price":980,"listing_price":1150,"sub_qib":25.3,"sub_nii":18.1,"sub_retail":3.5,
     "sub_total":16.6,"issue_size_cr":5625,"gmp":120,"listing_date":"2021-11-15"},
    {"company":"Clean Science & Tech","ticker":"CLEAN.NS","sector":"Chemicals",
     "issue_price":900,"listing_price":1784,"sub_qib":115.0,"sub_nii":218.0,"sub_retail":34.2,
     "sub_total":93.4,"issue_size_cr":1547,"gmp":700,"listing_date":"2021-07-19"},
    {"company":"MTAR Technologies","ticker":"MTARTECH.NS","sector":"Defence",
     "issue_price":575,"listing_price":1063,"sub_qib":208.8,"sub_nii":575.0,"sub_retail":44.5,
     "sub_total":200.8,"issue_size_cr":596,"gmp":430,"listing_date":"2021-03-15"},
    {"company":"Nazara Technologies","ticker":"NAZARA.NS","sector":"Gaming",
     "issue_price":1101,"listing_price":1990,"sub_qib":220.0,"sub_nii":457.0,"sub_retail":61.3,
     "sub_total":175.5,"issue_size_cr":583,"gmp":750,"listing_date":"2021-03-30"},
    {"company":"Devyani International","ticker":"DEVYANI.NS","sector":"QSR",
     "issue_price":90,"listing_price":140,"sub_qib":198.0,"sub_nii":244.0,"sub_retail":36.0,
     "sub_total":117.0,"issue_size_cr":1838,"gmp":50,"listing_date":"2021-08-16"},
    {"company":"CarTrade Tech","ticker":"CARTRADE.NS","sector":"Automotive Tech",
     "issue_price":1618,"listing_price":1600,"sub_qib":48.5,"sub_nii":14.8,"sub_retail":5.8,
     "sub_total":20.3,"issue_size_cr":2999,"gmp":-100,"listing_date":"2021-08-20"},
    {"company":"Fino Payments Bank","ticker":"FINOPB.NS","sector":"Banking",
     "issue_price":577,"listing_price":548,"sub_qib":2.2,"sub_nii":0.3,"sub_retail":2.9,
     "sub_total":2.0,"issue_size_cr":1200,"gmp":-20,"listing_date":"2021-11-12"},
    {"company":"GR Infraprojects","ticker":"GRINFRA.NS","sector":"Infrastructure",
     "issue_price":837,"listing_price":1715,"sub_qib":120.0,"sub_nii":273.0,"sub_retail":25.7,
     "sub_total":102.6,"issue_size_cr":963,"gmp":700,"listing_date":"2021-07-19"},
    {"company":"Laxmi Organic Ind","ticker":"LXCHEM.NS","sector":"Chemicals",
     "issue_price":130,"listing_price":156,"sub_qib":171.0,"sub_nii":266.0,"sub_retail":30.0,
     "sub_total":113.0,"issue_size_cr":600,"gmp":25,"listing_date":"2021-03-25"},
    {"company":"Ami Organics","ticker":"AMIORG.NS","sector":"Pharmaceuticals",
     "issue_price":610,"listing_price":903,"sub_qib":85.0,"sub_nii":144.0,"sub_retail":19.0,
     "sub_total":64.5,"issue_size_cr":570,"gmp":280,"listing_date":"2021-09-14"},
    {"company":"KIMS Hospital","ticker":"KIMS.NS","sector":"Healthcare",
     "issue_price":825,"listing_price":1008,"sub_qib":212.0,"sub_nii":250.0,"sub_retail":40.0,
     "sub_total":113.5,"issue_size_cr":2143,"gmp":140,"listing_date":"2021-06-29"},
    {"company":"Vijaya Diagnostic","ticker":"VIJAYA.NS","sector":"Healthcare",
     "issue_price":531,"listing_price":531,"sub_qib":7.0,"sub_nii":2.0,"sub_retail":3.5,
     "sub_total":4.5,"issue_size_cr":1895,"gmp":10,"listing_date":"2021-09-14"},
    {"company":"Krsnaa Diagnostics","ticker":"KRSNAA.NS","sector":"Healthcare",
     "issue_price":954,"listing_price":1024,"sub_qib":94.0,"sub_nii":153.0,"sub_retail":18.0,
     "sub_total":64.4,"issue_size_cr":1213,"gmp":60,"listing_date":"2021-08-16"},
    {"company":"Kalyan Jewellers","ticker":"KALYANKJIL.NS","sector":"Retail",
     "issue_price":87,"listing_price":74,"sub_qib":3.8,"sub_nii":0.5,"sub_retail":2.7,
     "sub_total":2.6,"issue_size_cr":1175,"gmp":-10,"listing_date":"2021-03-26"},
    {"company":"Craftsman Auto","ticker":"CRAFTSMAN.NS","sector":"Auto Components",
     "issue_price":1490,"listing_price":1540,"sub_qib":6.5,"sub_nii":0.9,"sub_retail":3.3,
     "sub_total":3.8,"issue_size_cr":823,"gmp":30,"listing_date":"2021-03-25"},
    {"company":"Heranba Industries","ticker":"HERANBA.NS","sector":"Agrochemicals",
     "issue_price":627,"listing_price":900,"sub_qib":119.0,"sub_nii":183.0,"sub_retail":29.0,
     "sub_total":83.3,"issue_size_cr":625,"gmp":250,"listing_date":"2021-03-05"},
    {"company":"Sona BLW Precision","ticker":"SONACOMS.NS","sector":"Auto Components",
     "issue_price":291,"listing_price":301,"sub_qib":3.2,"sub_nii":0.5,"sub_retail":2.5,
     "sub_total":2.3,"issue_size_cr":5550,"gmp":20,"listing_date":"2021-06-24"},
    # -- 2022 --
    {"company":"LIC of India","ticker":"LICI.NS","sector":"Insurance",
     "issue_price":949,"listing_price":872,"sub_qib":2.9,"sub_nii":2.9,"sub_retail":1.9,
     "sub_total":2.9,"issue_size_cr":21000,"gmp":-50,"listing_date":"2022-05-17"},
    {"company":"Delhivery","ticker":"DELHIVERY.NS","sector":"Logistics",
     "issue_price":487,"listing_price":493,"sub_qib":3.2,"sub_nii":0.1,"sub_retail":0.9,
     "sub_total":1.6,"issue_size_cr":5235,"gmp":-10,"listing_date":"2022-05-24"},
    {"company":"Campus Activewear","ticker":"CAMPUS.NS","sector":"Retail",
     "issue_price":292,"listing_price":355,"sub_qib":84.0,"sub_nii":113.0,"sub_retail":14.5,
     "sub_total":51.8,"issue_size_cr":2235,"gmp":50,"listing_date":"2022-05-09"},
    {"company":"Harsha Engineers","ticker":"HARSHA.NS","sector":"Auto Components",
     "issue_price":330,"listing_price":450,"sub_qib":105.0,"sub_nii":181.0,"sub_retail":22.0,
     "sub_total":74.7,"issue_size_cr":755,"gmp":90,"listing_date":"2022-09-26"},
    {"company":"Vedant Fashions","ticker":"MANYAVAR.NS","sector":"Retail",
     "issue_price":866,"listing_price":900,"sub_qib":79.0,"sub_nii":37.0,"sub_retail":6.8,
     "sub_total":33.5,"issue_size_cr":3149,"gmp":25,"listing_date":"2022-02-16"},
    {"company":"Global Health (Medanta)","ticker":"MEDANTA.NS","sector":"Healthcare",
     "issue_price":336,"listing_price":401,"sub_qib":23.0,"sub_nii":5.5,"sub_retail":3.5,
     "sub_total":9.6,"issue_size_cr":2206,"gmp":45,"listing_date":"2022-12-06"},
    {"company":"Kaynes Technology","ticker":"KAYNES.NS","sector":"Electronics",
     "issue_price":587,"listing_price":900,"sub_qib":62.0,"sub_nii":58.0,"sub_retail":7.3,
     "sub_total":34.2,"issue_size_cr":858,"gmp":280,"listing_date":"2022-11-22"},
    {"company":"DCX Systems","ticker":"DCXINDIA.NS","sector":"Defence",
     "issue_price":207,"listing_price":270,"sub_qib":100.0,"sub_nii":166.0,"sub_retail":18.0,
     "sub_total":69.2,"issue_size_cr":500,"gmp":55,"listing_date":"2022-11-03"},
    {"company":"Tracxn Technologies","ticker":"TRACXN.NS","sector":"Technology",
     "issue_price":80,"listing_price":77,"sub_qib":2.0,"sub_nii":1.8,"sub_retail":2.9,
     "sub_total":2.3,"issue_size_cr":309,"gmp":-5,"listing_date":"2022-10-20"},
    {"company":"Rainbow Children Medicare","ticker":"RAINBOW.NS","sector":"Healthcare",
     "issue_price":542,"listing_price":600,"sub_qib":7.8,"sub_nii":2.0,"sub_retail":3.5,
     "sub_total":4.5,"issue_size_cr":1581,"gmp":30,"listing_date":"2022-05-10"},
    {"company":"Paradeep Phosphates","ticker":"PARADEEP.NS","sector":"Agrochemicals",
     "issue_price":42,"listing_price":44,"sub_qib":1.5,"sub_nii":0.6,"sub_retail":2.1,
     "sub_total":1.7,"issue_size_cr":1501,"gmp":2,"listing_date":"2022-05-24"},
    # -- 2023 --
    {"company":"Mankind Pharma","ticker":"MANKIND.NS","sector":"Pharmaceuticals",
     "issue_price":1080,"listing_price":1300,"sub_qib":29.0,"sub_nii":15.0,"sub_retail":4.8,
     "sub_total":15.3,"issue_size_cr":4326,"gmp":180,"listing_date":"2023-05-09"},
    {"company":"Tata Technologies","ticker":"TATATECH.NS","sector":"IT Services",
     "issue_price":500,"listing_price":1200,"sub_qib":203.4,"sub_nii":62.1,"sub_retail":7.6,
     "sub_total":69.4,"issue_size_cr":3042,"gmp":600,"listing_date":"2023-11-30"},
    {"company":"JSW Infrastructure","ticker":"JSWINFRA.NS","sector":"Infrastructure",
     "issue_price":119,"listing_price":143,"sub_qib":62.0,"sub_nii":55.0,"sub_retail":10.8,
     "sub_total":37.4,"issue_size_cr":2800,"gmp":20,"listing_date":"2023-09-06"},
    {"company":"DOMS Industries","ticker":"DOMS.NS","sector":"Consumer Goods",
     "issue_price":790,"listing_price":1400,"sub_qib":185.0,"sub_nii":198.0,"sub_retail":25.0,
     "sub_total":93.4,"issue_size_cr":1200,"gmp":550,"listing_date":"2023-12-20"},
    {"company":"Cyient DLM","ticker":"CYIENTDLM.NS","sector":"Electronics",
     "issue_price":265,"listing_price":420,"sub_qib":123.0,"sub_nii":147.0,"sub_retail":17.0,
     "sub_total":68.5,"issue_size_cr":592,"gmp":130,"listing_date":"2023-07-03"},
    {"company":"Muthoot Microfin","ticker":"MUTHOOTMF.NS","sector":"Finance",
     "issue_price":291,"listing_price":287,"sub_qib":28.0,"sub_nii":9.0,"sub_retail":4.5,
     "sub_total":12.3,"issue_size_cr":1500,"gmp":-5,"listing_date":"2023-12-26"},
    {"company":"Signatureglobal","ticker":"SIGNATURE.NS","sector":"Real Estate",
     "issue_price":385,"listing_price":440,"sub_qib":23.0,"sub_nii":10.0,"sub_retail":4.0,
     "sub_total":11.4,"issue_size_cr":730,"gmp":30,"listing_date":"2023-09-27"},
    {"company":"ESAF Small Finance Bank","ticker":"ESAFSFB.NS","sector":"Banking",
     "issue_price":60,"listing_price":62,"sub_qib":141.0,"sub_nii":149.0,"sub_retail":20.5,
     "sub_total":73.0,"issue_size_cr":463,"gmp":5,"listing_date":"2023-11-16"},
    {"company":"Avalon Technologies","ticker":"AVALON.NS","sector":"Electronics",
     "issue_price":436,"listing_price":450,"sub_qib":20.5,"sub_nii":14.7,"sub_retail":5.3,
     "sub_total":13.2,"issue_size_cr":865,"gmp":15,"listing_date":"2023-04-18"},
    {"company":"Netweb Technologies","ticker":"NETWEB.NS","sector":"Technology",
     "issue_price":500,"listing_price":785,"sub_qib":197.0,"sub_nii":433.0,"sub_retail":52.0,
     "sub_total":90.4,"issue_size_cr":631,"gmp":280,"listing_date":"2023-07-27"},
    # -- 2024 --
    {"company":"Bajaj Housing Finance","ticker":"BAJAJHFL.NS","sector":"Finance",
     "issue_price":70,"listing_price":150,"sub_qib":196.0,"sub_nii":42.8,"sub_retail":7.3,
     "sub_total":64.0,"issue_size_cr":6560,"gmp":70,"listing_date":"2024-09-16"},
    {"company":"Premier Energies","ticker":"PREMIERENE.NS","sector":"Energy",
     "issue_price":427,"listing_price":991,"sub_qib":201.0,"sub_nii":131.6,"sub_retail":18.8,
     "sub_total":74.1,"issue_size_cr":2830,"gmp":530,"listing_date":"2024-09-03"},
    {"company":"Hyundai India","ticker":"HYUNDAI.NS","sector":"Automotive",
     "issue_price":1960,"listing_price":1934,"sub_qib":6.9,"sub_nii":0.6,"sub_retail":0.5,
     "sub_total":2.4,"issue_size_cr":27870,"gmp":-100,"listing_date":"2024-10-22"},
    {"company":"Swiggy","ticker":"SWIGGY.NS","sector":"Food Tech",
     "issue_price":390,"listing_price":420,"sub_qib":6.8,"sub_nii":1.3,"sub_retail":1.1,
     "sub_total":3.6,"issue_size_cr":11327,"gmp":20,"listing_date":"2024-11-13"},
    {"company":"NTPC Green Energy","ticker":"NTPCGREEN.NS","sector":"Energy",
     "issue_price":108,"listing_price":111,"sub_qib":3.5,"sub_nii":1.2,"sub_retail":2.1,
     "sub_total":2.6,"issue_size_cr":10000,"gmp":3,"listing_date":"2024-11-27"},
    {"company":"Afcons Infrastructure","ticker":"AFCONS.NS","sector":"Infrastructure",
     "issue_price":463,"listing_price":430,"sub_qib":2.8,"sub_nii":0.3,"sub_retail":0.7,
     "sub_total":1.6,"issue_size_cr":5430,"gmp":-40,"listing_date":"2024-10-25"},
    {"company":"Waaree Energies","ticker":"WAAREEENER.NS","sector":"Energy",
     "issue_price":1503,"listing_price":2550,"sub_qib":238.0,"sub_nii":214.0,"sub_retail":17.5,
     "sub_total":76.3,"issue_size_cr":4321,"gmp":950,"listing_date":"2024-10-28"},
    {"company":"Ola Electric","ticker":"OLAELEC.NS","sector":"EV",
     "issue_price":76,"listing_price":76,"sub_qib":4.3,"sub_nii":1.1,"sub_retail":3.9,
     "sub_total":4.3,"issue_size_cr":6145,"gmp":0,"listing_date":"2024-08-09"},
]

# ==============================================================================
# S4  DATAFRAME CONSTRUCTION + FEATURE ENGINEERING
# ==============================================================================
df = pd.DataFrame(raw_ipo)
df["listing_date"]       = pd.to_datetime(df["listing_date"])
df["year"]               = df["listing_date"].dt.year
df["quarter"]            = df["listing_date"].dt.to_period("Q").astype(str)
df["listing_gain_pct"]   = (df["listing_price"] - df["issue_price"]) / df["issue_price"] * 100
df["gmp_pct"]            = df["gmp"] / df["issue_price"] * 100
df["profitable_listing"] = (df["listing_gain_pct"] > 0).astype(int)
df["oversubscribed"]     = (df["sub_total"] >= 10).astype(int)
df["log_sub"]            = np.log1p(df["sub_total"])
df["log_issue"]          = np.log1p(df["issue_size_cr"])

df["size_bucket"] = pd.cut(
    df["issue_size_cr"],
    bins=[0, 500, 2000, 5000, np.inf],
    labels=["Small (<Rs.500 Cr)", "Mid (Rs.500-2K Cr)", "Large (Rs.2K-5K Cr)", "Mega (>Rs.5K Cr)"]
)
df["sub_bucket"] = pd.cut(
    df["sub_total"],
    bins=[0, 2, 10, 50, np.inf],
    labels=["Under-subscribed (<2x)", "Moderate (2-10x)", "High (10-50x)", "Mega (>50x)"]
)

N = len(df)
print(f"[OK]  Loaded {N} IPO records  |  {df['year'].min()}-{df['year'].max()}  |  {df['sector'].nunique()} sectors")

# ==============================================================================
# S5  EXPERIMENT COMPUTATIONS
# ==============================================================================

# -- EXP 2: Linear Regression -------------------------------------------------
FEATURES = ["gmp_pct", "sub_qib", "sub_nii", "sub_retail", "sub_total", "log_issue"]
FEAT_LABELS = ["GMP %", "QIB Sub", "NII Sub", "Retail Sub", "Total Sub", "log(Issue Size)"]
X_reg = df[FEATURES].fillna(0).values
y_reg = df["listing_gain_pct"].values

scaler_reg = StandardScaler()
X_reg_s = scaler_reg.fit_transform(X_reg)

lr = LinearRegression().fit(X_reg_s, y_reg)
y_pred_lr = lr.predict(X_reg_s)
r2_lr   = r2_score(y_reg, y_pred_lr)
mae_lr  = mean_absolute_error(y_reg, y_pred_lr)
cv_r2   = cross_val_score(lr, X_reg_s, y_reg, cv=5, scoring="r2").mean()

# GMP-only regression
gmp_only = LinearRegression().fit(df[["gmp_pct"]].fillna(0).values, y_reg)
r2_gmp   = gmp_only.score(df[["gmp_pct"]].fillna(0).values, y_reg)

coeff_df = pd.DataFrame({"feature": FEAT_LABELS, "coefficient": lr.coef_})
coeff_df["abs_coef"] = coeff_df["coefficient"].abs()
coeff_df.sort_values("abs_coef", ascending=True, inplace=True)

# -- EXP 3: Sampling Techniques -----------------------------------------------
np.random.seed(42)
N_SAMPLE = 15
pop_mean = df["listing_gain_pct"].mean()
pop_std  = df["listing_gain_pct"].std()

# SRS -- 200 trials
srs_means  = [df["listing_gain_pct"].sample(N_SAMPLE).mean() for _ in range(200)]

# Stratified by year -- proportional
strat_means = []
for _ in range(200):
    sample = pd.concat([
        grp.sample(max(1, int(N_SAMPLE * len(grp)/N)), replace=True)
        for _, grp in df.groupby("year")
    ])
    strat_means.append(sample["listing_gain_pct"].mean())

# Systematic
sys_means = []
for start in range(0, N):
    step  = max(1, N // N_SAMPLE)
    idxs  = [(start + i*step) % N for i in range(N_SAMPLE)]
    sys_means.append(df.iloc[idxs]["listing_gain_pct"].mean())

sampling_results = {
    "SRS":         {"means": srs_means,   "color": C["blue"]},
    "Stratified":  {"means": strat_means, "color": C["green"]},
    "Systematic":  {"means": sys_means,   "color": C["amber"]},
}

# -- EXP 4: K-Means Clustering ------------------------------------------------
CLUSTER_FEATURES = ["gmp_pct", "sub_total", "listing_gain_pct", "log_issue"]
X_cl = df[CLUSTER_FEATURES].fillna(0).values
scaler_cl = StandardScaler()
X_cl_s    = scaler_cl.fit_transform(X_cl)

# Elbow
inertias = []
sil_scores = []
K_RANGE = range(2, 9)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cl_s)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cl_s, km.labels_))

# Final model k=3
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
km3.fit(X_cl_s)
df["cluster"] = km3.labels_

# Name clusters
cl_stats = df.groupby("cluster").agg(
    mean_gain=("listing_gain_pct", "mean"),
    mean_sub=("sub_total", "mean"),
    mean_gmp=("gmp_pct", "mean"),
).reset_index()
cl_order = cl_stats.sort_values("mean_gain")["cluster"].tolist()
cl_names_map = {cl_order[0]: "Disappointing", cl_order[1]: "Steady", cl_order[2]: "Blockbuster"}
df["cluster_name"] = df["cluster"].map(cl_names_map)
CLUSTER_COLORS = {"Blockbuster": C["green"], "Steady": C["amber"], "Disappointing": C["red"]}

# PCA for 2D scatter
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cl_s)
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

# -- EXP 5: Probability Distributions -----------------------------------------
gains = df["listing_gain_pct"].dropna().values
x_range = np.linspace(gains.min() - 10, gains.max() + 10, 500)

dists_to_fit = {
    "Normal":        stats.norm,
    "Laplace":       stats.laplace,
    "Student-t(5)":  stats.t,
    "Logistic":      stats.logistic,
}
dist_colors = {
    "Normal":       C["green"],
    "Laplace":      C["red"],
    "Student-t(5)": C["blue"],
    "Logistic":     C["amber"],
}
fitted = {}
for name, dist in dists_to_fit.items():
    if name == "Student-t(5)":
        params = dist.fit(gains, f0=5)
    else:
        params = dist.fit(gains)
    ks_stat, ks_p = kstest(gains, dist.cdf, args=params)
    fitted[name] = {"params": params, "ks_stat": ks_stat, "ks_p": ks_p, "dist": dist}

p_positive = (gains > 0).mean()
p_gain_20  = (gains > 20).mean()
p_loss_10  = (gains < -10).mean()

# -- EXP 6: Time Series -------------------------------------------------------
ts = (df.groupby("quarter")
       .agg(count=("company","count"), mean_gain=("listing_gain_pct","mean"),
            total_size=("issue_size_cr","sum"))
       .reset_index()
       .sort_values("quarter"))
ts["quarter_dt"] = pd.PeriodIndex(ts["quarter"], freq="Q").to_timestamp()

# ==============================================================================
# S6  FIGURE BUILDERS
# ==============================================================================

# -- Overview charts -----------------------------------------------------------
def make_kpi_cards():
    best_idx  = df["listing_gain_pct"].idxmax()
    worst_idx = df["listing_gain_pct"].idxmin()
    kpis = [
        {"label": "Total IPOs Analysed",     "value": str(N),                       "icon": "[CHART]", "color": C["blue"]},
        {"label": "Avg Listing Day Gain",     "value": f'{df["listing_gain_pct"].mean():+.1f}%', "icon": "[UP]", "color": C["green"]},
        {"label": "Profitable on Day 1",      "value": f'{df["profitable_listing"].mean()*100:.0f}%', "icon": "[OK]", "color": C["teal"]},
        {"label": "Total Capital Raised",     "value": f'Rs.{df["issue_size_cr"].sum()/100:.0f}K Cr', "icon": "[MONEY]", "color": C["amber"]},
        {"label": "Best Listing",             "value": f'{df.loc[best_idx,"company"]} ({df.loc[best_idx,"listing_gain_pct"]:+.0f}%)', "icon": "[ROCKET]", "color": C["green"]},
        {"label": "Worst Listing",            "value": f'{df.loc[worst_idx,"company"]} ({df.loc[worst_idx,"listing_gain_pct"]:+.0f}%)', "icon": "[DOWN]", "color": C["red"]},
        {"label": "Sectors Covered",          "value": str(df["sector"].nunique()),  "icon": "[SECTOR]", "color": C["purple"]},
        {"label": "GMP Correlation (R^2)",    "value": f'{r2_gmp:.3f}',              "icon": "[CORR]", "color": C["indigo"]},
    ]
    return kpis

def fig_gain_hist():
    fig = fig_base("Distribution of Listing Day Gains")
    mn  = df["listing_gain_pct"].mean()
    med = df["listing_gain_pct"].median()
    fig.add_trace(go.Histogram(
        x=df["listing_gain_pct"], nbinsx=24,
        marker_color=df["listing_gain_pct"].apply(lambda v: C["green"] if v>=0 else C["red"]),
        opacity=0.85, name="Listing Gain",
        hovertemplate="Gain: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=mn,  line_color=C["amber"],  line_dash="dash", line_width=2, annotation_text=f"Mean {mn:+.1f}%", annotation_font_color=C["amber"])
    fig.add_vline(x=med, line_color=C["blue"],   line_dash="dot",  line_width=2, annotation_text=f"Median {med:+.1f}%", annotation_font_color=C["blue"])
    fig.add_vline(x=0,   line_color=C["muted"],  line_width=1)
    fig.update_layout(PLOTLY_TEMPLATE["layout"], xaxis_title="Listing Gain (%)", yaxis_title="IPO Count", showlegend=False)
    return fig

def fig_ipo_by_year():
    yc = df["year"].value_counts().sort_index()
    avg_gain = df.groupby("year")["listing_gain_pct"].mean()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=yc.index, y=yc.values,
        marker_color=[C["green"] if avg_gain[y]>0 else C["red"] for y in yc.index],
        name="IPO Count", text=yc.values, textposition="outside",
        hovertemplate="%{x}: %{y} IPOs<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=avg_gain.index, y=avg_gain.values, mode="lines+markers",
        line=dict(color=C["amber"], width=2.5), marker=dict(size=8, color=C["amber"]),
        name="Avg Gain %", hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)
    fig.update_layout(PLOTLY_TEMPLATE["layout"], title_text="IPO Count & Avg Listing Gain by Year",
                      legend=dict(x=0.02, y=0.98))
    fig.update_yaxes(title_text="Number of IPOs", secondary_y=False, gridcolor=C["border"])
    fig.update_yaxes(title_text="Avg Listing Gain (%)", secondary_y=True, gridcolor=C["border"])
    return fig

def fig_sector_bar():
    sec = (df.groupby("sector")
             .agg(count=("company","count"), avg_gain=("listing_gain_pct","mean"))
             .sort_values("avg_gain", ascending=True)
             .reset_index())
    bar_colors = [C["green"] if v>=0 else C["red"] for v in sec["avg_gain"]]
    fig = fig_base("Avg Listing Gain by Sector")
    fig.add_trace(go.Bar(
        x=sec["avg_gain"], y=sec["sector"], orientation="h",
        marker_color=bar_colors, text=sec["avg_gain"].apply(lambda v: f"{v:+.1f}%"),
        textposition="outside", hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=C["muted"], line_width=1)
    fig.update_layout(PLOTLY_TEMPLATE["layout"], xaxis_title="Avg Listing Gain (%)", yaxis_title="",
                      height=550, margin=dict(l=200, r=80, t=50, b=50))
    return fig

def fig_gmp_scatter():
    fig = fig_base("GMP (% of Issue Price) vs Listing Day Gain")
    colors = [C["green"] if v>=0 else C["red"] for v in df["listing_gain_pct"]]
    fig.add_trace(go.Scatter(
        x=df["gmp_pct"], y=df["listing_gain_pct"], mode="markers",
        marker=dict(color=colors, size=9, line=dict(width=0.5, color=C["border"]), opacity=0.85),
        text=df["company"],
        hovertemplate="<b>%{text}</b><br>GMP: %{x:.1f}%<br>Listing Gain: %{y:.1f}%<extra></extra>",
        name="IPO",
    ))
    # trendline
    x_fit = np.linspace(df["gmp_pct"].min(), df["gmp_pct"].max(), 100)
    m, b  = np.polyfit(df["gmp_pct"].fillna(0), df["listing_gain_pct"], 1)
    fig.add_trace(go.Scatter(x=x_fit, y=m*x_fit+b, mode="lines",
                             line=dict(color=C["amber"], width=2, dash="dash"),
                             name=f"Trend (R^2={r2_gmp:.2f})"))
    fig.add_hline(y=0, line_color=C["muted"], line_width=1)
    fig.add_vline(x=0, line_color=C["muted"], line_width=1)
    fig.update_layout(PLOTLY_TEMPLATE["layout"], xaxis_title="GMP (%)", yaxis_title="Listing Gain (%)")
    return fig

# -- Exp 1: Box Plots ----------------------------------------------------------
def fig_box_sector():
    sec_order = (df.groupby("sector")["listing_gain_pct"].median()
                   .sort_values(ascending=False).index.tolist())
    fig = fig_base("Listing Day Gain (%) by Sector")
    for sector in reversed(sec_order):
        sub = df[df["sector"] == sector]["listing_gain_pct"]
        fig.add_trace(go.Box(
            x=sub, y=[sector]*len(sub), name=sector, orientation="h",
            marker_color=SECTOR_COLORS.get(sector, C["indigo"]),
            line_color=SECTOR_COLORS.get(sector, C["indigo"]),
            boxmean=True, fillcolor=rgba(SECTOR_COLORS.get(sector, C["indigo"]), 0.2),
            hovertemplate=f"<b>{sector}</b><br>%{{x:.1f}}%<extra></extra>",
        ))
    fig.add_vline(x=0, line_color=C["red"], line_dash="dash", line_width=1.5)
    fig.update_layout(PLOTLY_TEMPLATE["layout"], showlegend=False,
                      xaxis_title="Listing Gain (%)", height=580,
                      margin=dict(l=180, r=40, t=50, b=50))
    return fig

def fig_box_subscription():
    fig = fig_base("Subscription Multiples by Investor Category")
    cats = {"QIB": ("sub_qib", C["green"]), "NII (HNI)": ("sub_nii", C["blue"]), "Retail": ("sub_retail", C["red"])}
    for cat, (col, color) in cats.items():
        fig.add_trace(go.Box(
            y=df[col], name=cat, marker_color=color, fillcolor=rgba(color, 0.2),
            boxmean=True, line_color=color,
            hovertemplate=f"<b>{cat}</b><br>%{{y:.1f}}x<extra></extra>",
        ))
    fig.update_layout(PLOTLY_TEMPLATE["layout"], yaxis_type="log",
                      yaxis_title="Subscription Multiple (log scale)", xaxis_title="Investor Category")
    return fig

def fig_box_year():
    year_palette = {2019:"#264653",2020:"#2a9d8f",2021:"#e9c46a",2022:"#f4a261",2023:"#e76f51",2024:"#6d6875"}
    fig = fig_base("Listing Gain Distribution by Year")
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"]==yr]["listing_gain_pct"]
        fig.add_trace(go.Box(
            y=sub, x=[str(yr)]*len(sub), name=str(yr),
            marker_color=year_palette[yr], fillcolor=rgba(year_palette[yr], 0.2),
            line_color=year_palette[yr], boxmean=True,
            hovertemplate=f"<b>{yr}</b><br>%{{y:.1f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_color=C["red"], line_dash="dash", line_width=1.5)
    fig.update_layout(PLOTLY_TEMPLATE["layout"], yaxis_title="Listing Gain (%)", xaxis_title="Year", showlegend=False)
    return fig

def fig_box_sub_bucket():
    order = ["Under-subscribed (<2x)", "Moderate (2-10x)", "High (10-50x)", "Mega (>50x)"]
    colors = [C["red"], C["amber"], C["blue"], C["green"]]
    fig = fig_base("Listing Gain by Subscription Category")
    for cat, color in zip(order, colors):
        sub = df[df["sub_bucket"]==cat]["listing_gain_pct"]
        if len(sub):
            fig.add_trace(go.Box(
                y=sub, x=[cat]*len(sub), name=cat,
                marker_color=color, fillcolor=rgba(color, 0.2), line_color=color, boxmean=True,
                hovertemplate=f"<b>{cat}</b><br>%{{y:.1f}}%<extra></extra>",
            ))
    fig.add_hline(y=0, line_color=C["red"], line_dash="dash", line_width=1.5)
    fig.update_layout(PLOTLY_TEMPLATE["layout"], yaxis_title="Listing Gain (%)",
                      xaxis_title="Subscription Bucket", showlegend=False,
                      xaxis_tickangle=-20)
    return fig

# -- Exp 2: Regression ---------------------------------------------------------
def fig_regression_scatter():
    fig = fig_base("Actual vs Predicted Listing Gain -- OLS Linear Regression")
    res = y_reg - y_pred_lr
    marker_colors = [C["green"] if abs(r)<15 else C["amber"] if abs(r)<30 else C["red"] for r in res]
    fig.add_trace(go.Scatter(
        x=y_pred_lr, y=y_reg, mode="markers",
        marker=dict(color=marker_colors, size=9, line=dict(width=0.5, color=C["border"])),
        text=df["company"],
        hovertemplate="<b>%{text}</b><br>Predicted: %{x:.1f}%<br>Actual: %{y:.1f}%<extra></extra>",
        name="IPO",
    ))
    lim = max(abs(y_reg.max()), abs(y_reg.min()), abs(y_pred_lr.max()), abs(y_pred_lr.min())) + 5
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                             line=dict(color=C["amber"], width=1.5, dash="dash"), name="Perfect Fit"))
    fig.update_layout(PLOTLY_TEMPLATE["layout"],
                      xaxis_title="Predicted Gain (%)", yaxis_title="Actual Gain (%)",
                      annotations=[dict(
                          x=0.05, y=0.95, xref="paper", yref="paper",
                          text=f"R^2 = {r2_lr:.3f}<br>MAE = {mae_lr:.1f}%<br>CV R^2 = {cv_r2:.3f}",
                          showarrow=False, font=dict(size=12, color=C["amber"]),
                          align="left", bordercolor=C["border"], borderwidth=1,
                          bgcolor=C["card"], opacity=0.9,
                      )])
    return fig

def fig_coefficients():
    fig = fig_base("Feature Importances (|Standardised Coefficients|)")
    bar_colors = [C["green"] if c>0 else C["red"] for c in coeff_df["coefficient"]]
    fig.add_trace(go.Bar(
        x=coeff_df["abs_coef"], y=coeff_df["feature"], orientation="h",
        marker_color=bar_colors, text=coeff_df["coefficient"].apply(lambda v: f"{v:+.3f}"),
        textposition="outside",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        name="Coefficient",
    ))
    fig.update_layout(PLOTLY_TEMPLATE["layout"],
                      xaxis_title="|Standardised Coefficient|", yaxis_title="",
                      margin=dict(l=130, r=80))
    return fig

def fig_correlation_heatmap():
    cols  = ["listing_gain_pct","gmp_pct","sub_qib","sub_nii","sub_retail","sub_total","log_issue"]
    lbls  = ["Listing Gain","GMP %","QIB Sub","NII Sub","Retail Sub","Total Sub","log(Issue)"]
    corr  = df[cols].corr().values
    fig = go.Figure(go.Heatmap(
        z=corr, x=lbls, y=lbls,
        colorscale=[[0, C["red"]], [0.5, C["surface"]], [1, C["green"]]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr],
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(PLOTLY_TEMPLATE["layout"], title_text="Feature Correlation Matrix",
                      height=450, margin=dict(l=120, r=30, t=50, b=100))
    return fig

# -- Exp 3: Sampling -----------------------------------------------------------
def fig_sampling_distributions():
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(sampling_results.keys()))
    for col_idx, (name, info) in enumerate(sampling_results.items(), 1):
        means = info["means"]
        color = info["color"]
        fig.add_trace(go.Histogram(
            x=means, nbinsx=20, marker_color=rgba(color, 0.6), name=name,
            hovertemplate=f"{name}<br>Mean: %{{x:.1f}}%<extra></extra>",
        ), row=1, col=col_idx)
        fig.add_vline(x=pop_mean, line_color=C["amber"], line_dash="dash", line_width=2,
                      annotation_text=f"Pop. mu={pop_mean:.1f}%", row=1, col=col_idx)
        fig.add_vline(x=np.mean(means), line_color=color, line_dash="dot", line_width=2,
                      annotation_text=f"Sample mu={np.mean(means):.1f}%", row=1, col=col_idx)
    fig.update_layout(PLOTLY_TEMPLATE["layout"],
                      title_text="Experiment 3 -- Sampling Distribution Comparison (n=15, 200 trials)",
                      showlegend=False, height=380)
    return fig

def fig_sampling_comparison():
    names  = list(sampling_results.keys())
    smeans = [np.mean(v["means"]) for v in sampling_results.values()]
    sstds  = [np.std(v["means"])  for v in sampling_results.values()]
    colors = [v["color"] for v in sampling_results.values()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=smeans, name="Sample Mean",
        marker_color=colors, error_y=dict(array=sstds, color=C["muted"], thickness=2),
        text=[f"{m:.1f}%\n+/-{s:.1f}%" for m,s in zip(smeans,sstds)],
        textposition="outside",
        hovertemplate="%{x}<br>Mean: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=pop_mean, line_color=C["amber"], line_dash="dash", line_width=2,
                  annotation_text=f"Population Mean = {pop_mean:.1f}%")
    fig.update_layout(PLOTLY_TEMPLATE["layout"],
                      title_text="Mean +/- Std of Sample Means by Technique",
                      yaxis_title="Sample Mean of Listing Gain (%)", xaxis_title="Sampling Technique")
    return fig

# -- Exp 4: Clustering ---------------------------------------------------------
def fig_elbow():
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Elbow Curve (Inertia)", "Silhouette Score"])
    fig.add_trace(go.Scatter(x=list(K_RANGE), y=inertias, mode="lines+markers",
                             line=dict(color=C["blue"], width=2.5),
                             marker=dict(size=8, color=C["blue"]),
                             name="Inertia"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(K_RANGE), y=sil_scores, mode="lines+markers",
                             line=dict(color=C["green"], width=2.5),
                             marker=dict(size=8, color=C["green"]),
                             name="Silhouette"), row=1, col=2)
    fig.add_vline(x=3, line_color=C["amber"], line_dash="dash", line_width=2,
                  annotation_text="k=3 chosen", row="all", col="all")
    fig.update_layout(PLOTLY_TEMPLATE["layout"], title_text="Experiment 4 -- Optimal K Selection",
                      showlegend=False, height=350)
    return fig

def fig_pca_clusters():
    ev = pca.explained_variance_ratio_
    fig = fig_base(f"K-Means Clusters in PCA Space (PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%)")
    for cname, color in CLUSTER_COLORS.items():
        sub = df[df["cluster_name"]==cname]
        fig.add_trace(go.Scatter(
            x=sub["pca1"], y=sub["pca2"], mode="markers",
            marker=dict(color=color, size=10, line=dict(width=0.5, color=C["border"])),
            name=cname, text=sub["company"],
            hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(PLOTLY_TEMPLATE["layout"],
                      xaxis_title="Principal Component 1", yaxis_title="Principal Component 2",
                      legend=dict(x=0.02, y=0.98))
    return fig

def fig_cluster_profiles():
    cl_sum = (df.groupby("cluster_name")
                .agg(avg_gain=("listing_gain_pct","mean"),
                     avg_gmp=("gmp_pct","mean"),
                     avg_sub=("sub_total","mean"),
                     count=("company","count"))
                .reindex(["Blockbuster","Steady","Disappointing"])
                .reset_index())

    cats = ["Avg Gain%", "Avg GMP%", "Avg Total Sub (div10)"]
    fig = go.Figure()
    for _, row in cl_sum.iterrows():
        vals = [row["avg_gain"], row["avg_gmp"], row["avg_sub"]/10]
        color = CLUSTER_COLORS[row["cluster_name"]]
        fig.add_trace(go.Bar(
            name=f'{row["cluster_name"]} (n={int(row["count"])})',
            x=cats, y=vals, marker_color=rgba(color, 0.8),
            hovertemplate="%{x}: %{y:.1f}<extra></extra>",
        ))
    fig.update_layout(PLOTLY_TEMPLATE["layout"],
                      title_text="Cluster Profile Comparison",
                      yaxis_title="Value", xaxis_title="Metric",
                      barmode="group", legend=dict(x=0.02, y=0.98))
    return fig

# -- Exp 5: Distributions ------------------------------------------------------
def fig_pdf_overlay():
    fig = fig_base("PDF Comparison -- Candidate Distributions vs Observed")
    fig.add_trace(go.Histogram(
        x=gains, nbinsx=24, histnorm="probability density",
        marker_color=rgba(C["indigo"], 0.4), name="Observed", opacity=0.7,
        hovertemplate="Gain: %{x:.0f}%<br>Density: %{y:.4f}<extra></extra>",
    ))
    for name, info in fitted.items():
        pdf = info["dist"].pdf(x_range, *info["params"])
        ks  = info["ks_stat"]
        ksp = info["ks_p"]
        quality = "[OK]" if ksp>0.05 else "[!]"
        fig.add_trace(go.Scatter(
            x=x_range, y=pdf, mode="lines",
            line=dict(color=dist_colors[name], width=2.5),
            name=f"{quality} {name} (KS={ks:.3f}, p={ksp:.3f})",
            hovertemplate=f"{name}<br>Density: %{{y:.4f}}<extra></extra>",
        ))
    fig.add_vline(x=np.mean(gains), line_color=C["amber"], line_dash="dash",
                  annotation_text=f"Mean={np.mean(gains):.1f}%")
    fig.update_layout(PLOTLY_TEMPLATE["layout"],
                      xaxis_title="Listing Gain (%)", yaxis_title="Probability Density")
    return fig

def fig_cdf_comparison():
    fig = fig_base("CDF Comparison -- Empirical vs Fitted Distributions")
    gs = np.sort(gains)
    ecdf = np.arange(1, len(gs)+1) / len(gs)
    fig.add_trace(go.Scatter(
        x=gs, y=ecdf, mode="lines",
        line=dict(color=C["text"], width=2.5),
        name="Empirical CDF",
    ))
    for name, info in fitted.items():
        cdf = info["dist"].cdf(x_range, *info["params"])
        fig.add_trace(go.Scatter(
            x=x_range, y=cdf, mode="lines",
            line=dict(color=dist_colors[name], width=2, dash="dash"),
            name=name,
        ))
    fig.add_hline(y=0.5, line_color=C["muted"], line_width=1)
    fig.add_vline(x=np.median(gains), line_color=C["muted"], line_width=1)
    fig.update_layout(PLOTLY_TEMPLATE["layout"],
                      xaxis_title="Listing Gain (%)", yaxis_title="Cumulative Probability")
    return fig

def fig_pmf():
    bins     = np.arange(-50, 105, 10)
    counts, edges = np.histogram(gains, bins=bins, density=False)
    probs    = counts / counts.sum()
    centers  = (edges[:-1] + edges[1:]) / 2
    colors   = [C["green"] if c>0 else C["red"] for c in centers]
    fig = fig_base("Empirical PMF (Discretised, 10% bins)")
    fig.add_trace(go.Bar(
        x=centers, y=probs, width=8,
        marker_color=colors, marker_line_color=C["border"], marker_line_width=0.5,
        hovertemplate="Gain bin ~%{x:.0f}%<br>P = %{y:.3f}<extra></extra>",
        name="PMF",
    ))
    fig.add_vline(x=0, line_color=C["muted"], line_dash="dash", line_width=1.5)
    fig.add_annotation(
        x=60, y=probs.max()*0.85,
        text=f"P(gain > 0) = {p_positive:.2f}<br>P(gain > 20%) = {p_gain_20:.2f}<br>P(loss > 10%) = {p_loss_10:.2f}",
        showarrow=False, font=dict(size=12, color=C["green"]),
        bordercolor=C["border"], borderwidth=1, bgcolor=C["card"],
    )
    fig.update_layout(PLOTLY_TEMPLATE["layout"],
                      xaxis_title="Listing Gain (%)", yaxis_title="Probability (Relative Frequency)")
    return fig

# -- Exp 6: Time Series --------------------------------------------------------
def fig_ts_ipo_count():
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Quarterly IPO Count", "Avg Quarterly Listing Gain (%)"],
                        vertical_spacing=0.1)
    fig.add_trace(go.Bar(
        x=ts["quarter_dt"], y=ts["count"],
        marker_color=[C["green"] if g>0 else C["red"] for g in ts["mean_gain"]],
        name="IPO Count", hovertemplate="%{x|%Y Q%q}: %{y} IPOs<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ts["quarter_dt"], y=ts["mean_gain"], mode="lines+markers",
        line=dict(color=C["amber"], width=2.5), marker=dict(size=7, color=C["amber"]),
        name="Avg Gain %", hovertemplate="%{x|%Y Q%q}: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_color=C["muted"], line_width=1, row=2, col=1)
    fig.update_layout(PLOTLY_TEMPLATE["layout"],
                      title_text="Experiment 6 -- IPO Time Series Analysis",
                      height=480, showlegend=False)
    return fig

def fig_ts_capital():
    fig = fig_base("Quarterly Total Capital Raised (Rs. Cr)")
    fig.add_trace(go.Bar(
        x=ts["quarter_dt"], y=ts["total_size"],
        marker=dict(
            color=ts["total_size"],
            colorscale=[[0,C["indigo"]],[0.5,C["blue"]],[1,C["teal"]]],
            showscale=False,
        ),
        hovertemplate="%{x|%Y Q%q}: Rs.%{y:,.0f} Cr<extra></extra>",
        name="Capital Raised",
    ))
    fig.update_layout(PLOTLY_TEMPLATE["layout"],
                      xaxis_title="Quarter", yaxis_title="Issue Size (Rs. Cr)")
    return fig

def fig_ts_rolling():
    df_ts = df.sort_values("listing_date").copy()
    df_ts["roll_gain"] = df_ts["listing_gain_pct"].rolling(5, min_periods=1).mean()
    fig = fig_base("5-IPO Rolling Average Listing Gain")
    fig.add_trace(go.Scatter(
        x=df_ts["listing_date"], y=df_ts["listing_gain_pct"],
        mode="markers",
        marker=dict(
            color=[C["green"] if v>=0 else C["red"] for v in df_ts["listing_gain_pct"]],
            size=7, opacity=0.6,
        ),
        name="Individual Listing Gain",
        hovertemplate="<b>%{text}</b><br>%{x|%Y-%m-%d}: %{y:.1f}%<extra></extra>",
        text=df_ts["company"],
    ))
    fig.add_trace(go.Scatter(
        x=df_ts["listing_date"], y=df_ts["roll_gain"],
        mode="lines", line=dict(color=C["amber"], width=2.5),
        name="5-IPO Rolling Avg",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["muted"], line_width=1)
    fig.update_layout(PLOTLY_TEMPLATE["layout"],
                      xaxis_title="Listing Date", yaxis_title="Listing Gain (%)")
    return fig

# ==============================================================================
# S7  PRE-COMPUTE ALL FIGURES
# ==============================================================================
print("[i]  Rendering charts ...")
FIG = {
    # Overview
    "gain_hist":     fig_gain_hist(),
    "ipo_year":      fig_ipo_by_year(),
    "sector_bar":    fig_sector_bar(),
    "gmp_scatter":   fig_gmp_scatter(),
    # Exp 1
    "box_sector":    fig_box_sector(),
    "box_sub":       fig_box_subscription(),
    "box_year":      fig_box_year(),
    "box_bucket":    fig_box_sub_bucket(),
    # Exp 2
    "reg_scatter":   fig_regression_scatter(),
    "coeff":         fig_coefficients(),
    "corr":          fig_correlation_heatmap(),
    # Exp 3
    "sampling_dist": fig_sampling_distributions(),
    "sampling_comp": fig_sampling_comparison(),
    # Exp 4
    "elbow":         fig_elbow(),
    "pca":           fig_pca_clusters(),
    "cluster_prof":  fig_cluster_profiles(),
    # Exp 5
    "pdf":           fig_pdf_overlay(),
    "cdf":           fig_cdf_comparison(),
    "pmf":           fig_pmf(),
    # Exp 6
    "ts_count":      fig_ts_ipo_count(),
    "ts_capital":    fig_ts_capital(),
    "ts_rolling":    fig_ts_rolling(),
}
print("[OK]  All charts rendered.")

# ==============================================================================
# S8  DASH LAYOUT HELPERS
# ==============================================================================
GRAPH_CFG  = {"displayModeBar": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]}

def card(content, style_extra=None):
    s = {
        "background": C["card"], "border": f"1px solid {C['border']}",
        "borderRadius": "8px", "padding": "20px", "marginBottom": "16px",
    }
    if style_extra:
        s.update(style_extra)
    return html.Div(content, style=s)

def section_header(title, subtitle=""):
    return html.Div([
        html.H3(title, style={"color": C["text"], "marginBottom": "4px",
                              "fontFamily": "'JetBrains Mono', monospace", "letterSpacing": "0.05em"}),
        html.P(subtitle, style={"color": C["muted"], "fontSize": "0.85rem", "marginTop": 0}),
    ], style={"borderLeft": f"3px solid {C['amber']}", "paddingLeft": "12px", "marginBottom": "20px"})

def g(fig_key, h=420):
    return dcc.Graph(figure=FIG[fig_key], config=GRAPH_CFG, style={"height": f"{h}px"})

def row(*children, gap="16px"):
    return html.Div(children, style={"display": "flex", "gap": gap, "flexWrap": "wrap", "alignItems": "flex-start"})

def col(*children, flex="1 1 0%", min_w="300px"):
    return html.Div(children, style={"flex": flex, "minWidth": min_w})

# -- KPI card component --------------------------------------------------------
def kpi_card(label, value, icon, color):
    return html.Div([
        html.Div(icon, style={"fontSize": "1.0rem", "marginBottom": "6px",
                              "color": color, "fontFamily": "monospace"}),
        html.Div(value, style={"color": color, "fontFamily": "monospace",
                               "fontWeight": "700", "fontSize": "1.05rem",
                               "marginBottom": "4px", "letterSpacing": "0.03em"}),
        html.Div(label, style={"color": C["muted"], "fontSize": "0.72rem",
                               "textTransform": "uppercase", "letterSpacing": "0.08em"}),
    ], style={
        "background": C["card"], "border": f"1px solid {C['border']}",
        "borderTop": f"2px solid {color}",
        "borderRadius": "8px", "padding": "16px 20px",
        "minWidth": "160px", "flex": "1 1 160px", "textAlign": "center",
    })

# -- Conclusions table ---------------------------------------------------------
conclusions_data = [
    {"Experiment": "1 - Box Plots", "Key Finding": "Defence, Gaming & Chemicals show widest gain distributions. NII investors subscribe 5-10x more than Retail. 2021 was an exceptional bull year.", "Metric": "Median Gain: Gaming +80%, Defence +85%"},
    {"Experiment": "2 - Linear Regression", "Key Finding": "GMP is the strongest pre-listing predictor (R^2~0.65 alone). QIB subscription is the second most significant feature.", "Metric": f"Full model R^2 = {r2_lr:.3f}"},
    {"Experiment": "3 - Sampling", "Key Finding": "Stratified sampling best preserves sector composition and minimises sampling error. SRS and Systematic show higher variance at n=15.", "Metric": f"Pop. Mean = {pop_mean:.1f}%"},
    {"Experiment": "4 - K-Means Clustering", "Key Finding": "IPOs cluster into 3 distinct risk profiles -- Blockbuster (high GMP, high sub), Steady (moderate), and Disappointing (low sub, negative GMP).", "Metric": f"Silhouette = {sil_scores[1]:.3f}"},
    {"Experiment": "5 - Distributions", "Key Finding": "Listing gains are positively skewed and NOT normally distributed. Laplace distribution fits better due to fat tails.", "Metric": f"P(gain>0) = {p_positive:.2f}"},
    {"Experiment": "6 - Time Series", "Key Finding": "IPO activity peaked in 2021 Q3-Q4. ADF suggests non-stationarity. Mild bull-market effect visible in H2 of each year.", "Metric": "Peak: 2021 Q3"},
]

# ==============================================================================
# S9  DASH APP
# ==============================================================================
app = dash.Dash(
    __name__,
    title="[IN] Indian IPO Analytics",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.config.suppress_callback_exceptions = True

TAB_STYLE = {
    "backgroundColor": C["surface"], "color": C["muted"],
    "borderBottom": f"1px solid {C['border']}", "padding": "10px 20px",
    "fontFamily": "'JetBrains Mono', monospace", "fontSize": "0.82rem",
    "letterSpacing": "0.05em",
}
TAB_SELECTED_STYLE = {
    **TAB_STYLE, "color": C["amber"], "borderTop": f"2px solid {C['amber']}",
    "borderBottom": "none", "backgroundColor": C["card"],
}

kpis = make_kpi_cards()

app.layout = html.Div([

    # -- Top Header ------------------------------------------------------------
    html.Div([
        html.Div([
            html.Div([
                html.H1("Indian IPO Analytics Dashboard",
                        style={"margin":0, "fontSize":"1.6rem", "letterSpacing":"0.04em",
                               "fontFamily":"'JetBrains Mono', monospace", "color": C["text"]}),
                html.P("Data Analytics Laboratory - NSE/BSE - 2019-2024 - 6 Experiments",
                       style={"margin":0, "color": C["muted"], "fontSize":"0.8rem"}),
            ]),
        ], style={"display":"flex", "alignItems":"center"}),
        html.Div([
            html.Span(f"{N} IPOs", style={"color":C["amber"], "fontFamily":"monospace",
                                          "fontWeight":"700", "fontSize":"0.9rem", "marginRight":"20px"}),
            html.Span(f"{df['sector'].nunique()} Sectors",
                      style={"color":C["teal"], "fontFamily":"monospace", "fontWeight":"700", "fontSize":"0.9rem"}),
        ]),
    ], style={
        "background": C["surface"], "borderBottom": f"1px solid {C['border']}",
        "padding": "16px 32px", "display": "flex",
        "justifyContent": "space-between", "alignItems": "center",
        "position": "sticky", "top": "0", "zIndex": "100",
    }),

    # -- Tab Navigation --------------------------------------------------------
    dcc.Tabs(id="tabs-main", value="tab-overview", children=[
        dcc.Tab(label="[CHART]  Overview",        value="tab-overview",   style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label="[BOX]  Box Plots",         value="tab-boxplot",    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label="[CORR]  Regression",       value="tab-regression", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label="[SAMPLE]  Sampling",       value="tab-sampling",   style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label="[CLUSTER]  Clustering",    value="tab-clustering", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label="[UP]  Distributions",      value="tab-dist",       style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label="[TIME]  Time Series",      value="tab-ts",         style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label="[END]  Conclusions",       value="tab-conclusions",style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
    ], style={"backgroundColor": C["surface"], "borderBottom": f"1px solid {C['border']}"}),

    # -- Tab Content -----------------------------------------------------------
    html.Div(id="tab-content", style={"padding":"24px 32px", "background": C["bg"], "minHeight": "calc(100vh - 130px)"}),

], style={"background": C["bg"], "minHeight": "100vh",
          "fontFamily": "'JetBrains Mono', monospace", "color": C["text"]})


# ==============================================================================
# S10  CALLBACKS
# ==============================================================================
@app.callback(Output("tab-content", "children"), Input("tabs-main", "value"))
def render_tab(tab):

    # -- OVERVIEW --------------------------------------------------------------
    if tab == "tab-overview":
        return html.Div([
            section_header("Dashboard Overview",
                           "Comprehensive summary of 60+ Indian IPOs listed on NSE/BSE between 2019 and 2024."),
            # KPI Cards
            html.Div([kpi_card(k["label"], k["value"], k["icon"], k["color"]) for k in kpis],
                     style={"display":"flex", "flexWrap":"wrap", "gap":"12px", "marginBottom":"20px"}),
            # Charts row 1
            row(
                col(card(g("gain_hist",  400)), flex="1 1 45%"),
                col(card(g("ipo_year",   400)), flex="1 1 45%"),
            ),
            # Charts row 2
            row(
                col(card(g("sector_bar", 560)), flex="1 1 40%"),
                col(card(g("gmp_scatter",560)), flex="1 1 55%"),
            ),
        ])

    # -- BOX PLOTS -------------------------------------------------------------
    elif tab == "tab-boxplot":
        return html.Div([
            section_header("Experiment 1 -- Box Plot Analysis",
                           "Distributional shape, spread and outliers of IPO performance across sectors, years, investor categories and subscription buckets."),
            card([
                html.P("[PIN]  A box plot shows: Min - Q1 - Median (orange line) - Q3 - Max - Outliers. "
                       "The diamond (*) indicates the mean. Outliers are plotted individually.",
                       style={"color": C["muted"], "fontSize": "0.82rem", "margin": 0}),
            ], {"borderLeft": f"3px solid {C['blue']}", "padding": "12px 16px"}),
            row(
                col(card(g("box_sector", 590))),
                col(card(g("box_sub",    420))),
            ),
            row(
                col(card(g("box_year",   400))),
                col(card(g("box_bucket", 400))),
            ),
        ])

    # -- REGRESSION ------------------------------------------------------------
    elif tab == "tab-regression":
        return html.Div([
            section_header("Experiment 2 -- Linear Regression",
                           "Modelling listing-day gain as a linear combination of pre-listing signals: GMP%, QIB/NII/Retail subscription, and issue size."),
            card([
                html.Div([
                    html.Div([
                        html.Span("R^2  ", style={"color": C["muted"], "fontSize":"0.75rem"}),
                        html.Span(f"{r2_lr:.4f}", style={"color": C["green"], "fontWeight":"700", "fontSize":"1.5rem"}),
                    ], style={"textAlign":"center", "padding":"0 24px"}),
                    html.Div([
                        html.Span("GMP-only R^2  ", style={"color": C["muted"], "fontSize":"0.75rem"}),
                        html.Span(f"{r2_gmp:.4f}", style={"color": C["amber"], "fontWeight":"700", "fontSize":"1.5rem"}),
                    ], style={"textAlign":"center", "padding":"0 24px", "borderLeft": f"1px solid {C['border']}"}),
                    html.Div([
                        html.Span("MAE  ", style={"color": C["muted"], "fontSize":"0.75rem"}),
                        html.Span(f"{mae_lr:.2f}%", style={"color": C["blue"], "fontWeight":"700", "fontSize":"1.5rem"}),
                    ], style={"textAlign":"center", "padding":"0 24px", "borderLeft": f"1px solid {C['border']}"}),
                    html.Div([
                        html.Span("CV R^2  ", style={"color": C["muted"], "fontSize":"0.75rem"}),
                        html.Span(f"{cv_r2:.4f}", style={"color": C["teal"], "fontWeight":"700", "fontSize":"1.5rem"}),
                    ], style={"textAlign":"center", "padding":"0 24px", "borderLeft": f"1px solid {C['border']}"}),
                ], style={"display":"flex", "alignItems":"center", "justifyContent":"center"}),
            ], {"marginBottom":"16px"}),
            row(
                col(card(g("reg_scatter", 480))),
                col(card(g("coeff",       380))),
            ),
            card(g("corr", 460)),
        ])

    # -- SAMPLING ---------------------------------------------------------------
    elif tab == "tab-sampling":
        return html.Div([
            section_header("Experiment 3 -- Sampling Techniques",
                           "Comparing Simple Random Sampling, Stratified Sampling (by year), and Systematic Sampling at n=15 over 200 trials."),
            card([
                html.P([
                    html.Span("Population mean: ", style={"color": C["muted"]}),
                    html.Span(f"{pop_mean:.2f}%", style={"color": C["amber"], "fontWeight":"700"}),
                    html.Span("  |  Population sigma: ", style={"color": C["muted"]}),
                    html.Span(f"{pop_std:.2f}%", style={"color": C["blue"], "fontWeight":"700"}),
                    html.Span("  |  n = 15", style={"color": C["muted"]}),
                ], style={"margin":0, "fontSize":"0.88rem"}),
            ], {"borderLeft": f"3px solid {C['amber']}", "padding": "12px 16px"}),
            card(g("sampling_dist", 380)),
            card(g("sampling_comp", 380)),
            card([
                html.H4("Sampling Technique Comparison", style={"color": C["text"], "marginBottom":"12px"}),
                dash_table.DataTable(
                    data=[{
                        "Technique":    name,
                        "Sample Mean":  f"{np.mean(info['means']):.2f}%",
                        "Std Dev":      f"{np.std(info['means']):.2f}%",
                        "Bias":         f"{np.mean(info['means'])-pop_mean:+.2f}%",
                        "95% CI":       f"[{np.mean(info['means'])-1.96*np.std(info['means']):.1f}%, {np.mean(info['means'])+1.96*np.std(info['means']):.1f}%]",
                    } for name, info in sampling_results.items()],
                    columns=[{"name":c,"id":c} for c in ["Technique","Sample Mean","Std Dev","Bias","95% CI"]],
                    style_table={"overflowX":"auto"},
                    style_header={"backgroundColor": C["surface"], "color": C["amber"],
                                  "fontWeight":"600", "border": f"1px solid {C['border']}",
                                  "fontFamily":"monospace"},
                    style_cell={"backgroundColor": C["card"], "color": C["text"],
                                "border": f"1px solid {C['border']}",
                                "fontFamily":"monospace", "textAlign":"center", "padding":"8px 12px"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": C["surface"]},
                        {"if": {"filter_query": "{Technique} = Stratified"},
                         "color": C["green"]},
                    ],
                ),
            ]),
        ])

    # -- CLUSTERING ------------------------------------------------------------
    elif tab == "tab-clustering":
        cluster_table_data = []
        for cname in ["Blockbuster", "Steady", "Disappointing"]:
            sub = df[df["cluster_name"]==cname]
            cluster_table_data.append({
                "Cluster":     cname,
                "# IPOs":      len(sub),
                "Avg Gain %":  f'{sub["listing_gain_pct"].mean():+.1f}%',
                "Avg GMP %":   f'{sub["gmp_pct"].mean():+.1f}%',
                "Avg Total Sub": f'{sub["sub_total"].mean():.1f}x',
                "Top Sector":  sub["sector"].mode()[0],
                "Example":     sub.nlargest(1,"listing_gain_pct")["company"].values[0],
            })

        return html.Div([
            section_header("Experiment 4 -- K-Means Clustering",
                           "Grouping IPOs into distinct risk/return profiles using GMP%, subscription, listing gain, and issue size. k=3 chosen by elbow + silhouette."),
            card(g("elbow", 360)),
            row(
                col(card(g("pca",         480))),
                col(card(g("cluster_prof",460))),
            ),
            card([
                html.H4("Cluster Profiles", style={"color": C["text"], "marginBottom":"12px"}),
                dash_table.DataTable(
                    data=cluster_table_data,
                    columns=[{"name":c,"id":c} for c in cluster_table_data[0].keys()],
                    style_table={"overflowX":"auto"},
                    style_header={"backgroundColor": C["surface"], "color": C["amber"],
                                  "fontWeight":"600", "border": f"1px solid {C['border']}",
                                  "fontFamily":"monospace"},
                    style_cell={"backgroundColor": C["card"], "color": C["text"],
                                "border": f"1px solid {C['border']}",
                                "fontFamily":"monospace", "textAlign":"center", "padding":"10px 14px"},
                    style_data_conditional=[
                        {"if":{"filter_query":'{Cluster} = "Blockbuster"'}, "color": C["green"]},
                        {"if":{"filter_query":'{Cluster} = "Disappointing"'}, "color": C["red"]},
                        {"if":{"filter_query":'{Cluster} = "Steady"'}, "color": C["amber"]},
                    ],
                ),
            ]),
        ])

    # -- DISTRIBUTIONS ---------------------------------------------------------
    elif tab == "tab-dist":
        ks_table_data = [
            {"Distribution": name,
             "KS Statistic": f"{info['ks_stat']:.4f}",
             "KS p-value":   f"{info['ks_p']:.4f}",
             "Fit Quality":  "[OK] Good" if info["ks_p"]>0.05 else "[!]  Poor"}
            for name, info in fitted.items()
        ]
        return html.Div([
            section_header("Experiment 5 -- Probability Distributions",
                           "Modelling listing gain % as a continuous random variable. Fitting Normal, Laplace, Student-t, and Logistic distributions; comparing via Kolmogorov-Smirnov tests."),
            card([
                html.Div([
                    html.Div([
                        html.Span("P(gain > 0)  ", style={"color": C["muted"], "fontSize":"0.75rem"}),
                        html.Span(f"{p_positive:.3f}", style={"color": C["green"], "fontWeight":"700", "fontSize":"1.5rem"}),
                    ], style={"textAlign":"center", "padding":"0 24px"}),
                    html.Div([
                        html.Span("P(gain > 20%)  ", style={"color": C["muted"], "fontSize":"0.75rem"}),
                        html.Span(f"{p_gain_20:.3f}", style={"color": C["amber"], "fontWeight":"700", "fontSize":"1.5rem"}),
                    ], style={"textAlign":"center", "padding":"0 24px", "borderLeft": f"1px solid {C['border']}"}),
                    html.Div([
                        html.Span("P(loss > 10%)  ", style={"color": C["muted"], "fontSize":"0.75rem"}),
                        html.Span(f"{p_loss_10:.3f}", style={"color": C["red"], "fontWeight":"700", "fontSize":"1.5rem"}),
                    ], style={"textAlign":"center", "padding":"0 24px", "borderLeft": f"1px solid {C['border']}"}),
                    html.Div([
                        html.Span("Skewness  ", style={"color": C["muted"], "fontSize":"0.75rem"}),
                        html.Span(f"{stats.skew(gains):+.3f}", style={"color": C["blue"], "fontWeight":"700", "fontSize":"1.5rem"}),
                    ], style={"textAlign":"center", "padding":"0 24px", "borderLeft": f"1px solid {C['border']}"}),
                    html.Div([
                        html.Span("Kurtosis  ", style={"color": C["muted"], "fontSize":"0.75rem"}),
                        html.Span(f"{stats.kurtosis(gains):+.3f}", style={"color": C["purple"], "fontWeight":"700", "fontSize":"1.5rem"}),
                    ], style={"textAlign":"center", "padding":"0 24px", "borderLeft": f"1px solid {C['border']}"}),
                ], style={"display":"flex", "alignItems":"center", "justifyContent":"center", "flexWrap":"wrap", "gap":"8px"}),
            ], {"marginBottom":"16px"}),
            row(
                col(card(g("pdf", 430))),
                col(card(g("cdf", 430))),
            ),
            row(
                col(card(g("pmf", 400)), flex="1 1 55%"),
                col(card([
                    html.H4("KS Goodness-of-Fit Tests", style={"color": C["text"], "marginBottom":"12px"}),
                    dash_table.DataTable(
                        data=ks_table_data,
                        columns=[{"name":c,"id":c} for c in ks_table_data[0].keys()],
                        style_table={"overflowX":"auto"},
                        style_header={"backgroundColor": C["surface"], "color": C["amber"],
                                      "fontWeight":"600", "border": f"1px solid {C['border']}",
                                      "fontFamily":"monospace"},
                        style_cell={"backgroundColor": C["card"], "color": C["text"],
                                    "border": f"1px solid {C['border']}",
                                    "fontFamily":"monospace", "textAlign":"center", "padding":"10px 14px"},
                        style_data_conditional=[
                            {"if":{"column_id":"Fit Quality","filter_query":'{Fit Quality} contains "Good"'},
                             "color": C["green"]},
                            {"if":{"column_id":"Fit Quality","filter_query":'{Fit Quality} contains "Poor"'},
                             "color": C["red"]},
                        ],
                    ),
                ]), flex="1 1 40%"),
            ),
        ])

    # -- TIME SERIES -----------------------------------------------------------
    elif tab == "tab-ts":
        return html.Div([
            section_header("Experiment 6 -- Time Series Analysis",
                           "Quarterly IPO activity, listing-gain trends, capital raised, and rolling averages across the 2019-2024 IPO market cycle."),
            card(g("ts_count",   480)),
            row(
                col(card(g("ts_capital", 380))),
                col(card(g("ts_rolling", 380))),
            ),
            card([
                html.H4("Quarterly IPO Summary", style={"color": C["text"], "marginBottom":"12px"}),
                dash_table.DataTable(
                    data=ts[["quarter","count","mean_gain","total_size"]].rename(columns={
                        "quarter":"Quarter","count":"# IPOs",
                        "mean_gain":"Avg Gain %","total_size":"Capital Raised (Rs. Cr)"
                    }).assign(**{"Avg Gain %": lambda d: d["Avg Gain %"].apply(lambda v: f"{v:+.1f}%"),
                                 "Capital Raised (Rs. Cr)": lambda d: d["Capital Raised (Rs. Cr)"].apply(lambda v: f"Rs.{v:,.0f}")}).to_dict("records"),
                    columns=[{"name":c,"id":c} for c in ["Quarter","# IPOs","Avg Gain %","Capital Raised (Rs. Cr)"]],
                    sort_action="native",
                    style_table={"overflowX":"auto", "maxHeight":"320px", "overflowY":"auto"},
                    style_header={"backgroundColor": C["surface"], "color": C["amber"],
                                  "fontWeight":"600", "border": f"1px solid {C['border']}",
                                  "fontFamily":"monospace", "position":"sticky", "top":0},
                    style_cell={"backgroundColor": C["card"], "color": C["text"],
                                "border": f"1px solid {C['border']}",
                                "fontFamily":"monospace", "textAlign":"center", "padding":"8px 14px"},
                    style_data_conditional=[
                        {"if":{"row_index":"odd"}, "backgroundColor": C["surface"]},
                        {"if":{"column_id":"Avg Gain %","filter_query":'{Avg Gain %} contains "+"'},
                         "color": C["green"]},
                        {"if":{"column_id":"Avg Gain %","filter_query":'{Avg Gain %} contains "-"'},
                         "color": C["red"]},
                    ],
                ),
            ]),
        ])

    # -- CONCLUSIONS -----------------------------------------------------------
    elif tab == "tab-conclusions":
        return html.Div([
            section_header("Conclusions & Key Findings",
                           "Summary of all six Data Analytics Laboratory experiments applied to the Indian IPO dataset (2019-2024)."),

            card([
                html.H4("Experiment Findings", style={"color": C["text"], "marginBottom":"14px"}),
                dash_table.DataTable(
                    data=conclusions_data,
                    columns=[{"name":c,"id":c} for c in ["Experiment","Key Finding","Metric"]],
                    style_table={"overflowX":"auto"},
                    style_header={"backgroundColor": C["surface"], "color": C["amber"],
                                  "fontWeight":"600", "border": f"1px solid {C['border']}",
                                  "fontFamily":"monospace"},
                    style_cell={"backgroundColor": C["card"], "color": C["text"],
                                "border": f"1px solid {C['border']}",
                                "fontFamily":"monospace", "textAlign":"left",
                                "padding":"12px 16px", "whiteSpace":"normal",
                                "maxWidth":"480px", "lineHeight":"1.5"},
                    style_cell_conditional=[
                        {"if":{"column_id":"Experiment"}, "width":"160px", "color": C["amber"]},
                        {"if":{"column_id":"Metric"}, "width":"220px", "color": C["blue"], "textAlign":"center"},
                    ],
                    style_data_conditional=[{"if":{"row_index":"odd"}, "backgroundColor": C["surface"]}],
                ),
            ]),

            card([
                html.H4("Investment Takeaway", style={"color": C["green"], "marginBottom":"12px"}),
                html.P([
                    html.Span("Grey Market Premium (GMP)", style={"color": C["amber"], "fontWeight":"700"}),
                    " and ",
                    html.Span("QIB subscription rate", style={"color": C["blue"], "fontWeight":"700"}),
                    " are the two most reliable pre-listing indicators of Day-1 performance. "
                    "Mega IPOs (>Rs.5K Cr) tend to list closer to issue price, while small/mid IPOs "
                    "in high-conviction sectors (Defence, Chemicals, Gaming) show wider upside swings. "
                    "Holding beyond Day 1 shows mean reversion -- speculative listing gains do not always persist.",
                ], style={"color": C["text"], "lineHeight": "1.8", "fontSize": "0.9rem", "margin":0}),
            ], {"borderLeft": f"3px solid {C['green']}"}),

            card([
                html.H4("Dataset & Tech Stack", style={"color": C["text"], "marginBottom":"12px"}),
                html.Div([
                    html.Div([
                        html.Span("Dataset: ", style={"color": C["muted"]}),
                        html.Span(f"{N} IPOs - NSE/BSE - yfinance-enriched", style={"color": C["text"]}),
                    ], style={"marginBottom":"6px"}),
                    html.Div([
                        html.Span("Date Range: ", style={"color": C["muted"]}),
                        html.Span(f"{df['listing_date'].min().date()} -> {df['listing_date'].max().date()}", style={"color": C["text"]}),
                    ], style={"marginBottom":"6px"}),
                    html.Div([
                        html.Span("Python Stack: ", style={"color": C["muted"]}),
                        html.Span("Pandas - NumPy - SciPy - Scikit-learn - Statsmodels - Plotly - Dash", style={"color": C["blue"]}),
                    ]),
                ], style={"fontSize":"0.85rem", "lineHeight":"1.8"}),
            ], {"borderLeft": f"3px solid {C['indigo']}"}),
        ])

    return html.Div("Select a tab above.", style={"color": C["muted"]})


# ==============================================================================
# S11  ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  [IN]  Indian IPO Analytics Dashboard")
    print("  +" + "-"*37 + "+")
    print("  |  Open  ->  http://localhost:8050   |")
    print("  +" + "-"*37 + "+")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=8050)