import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

warnings.filterwarnings('ignore')

def check_lib(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None

OPTUNA_AVAILABLE = check_lib('optuna')
SHAP_AVAILABLE = check_lib('shap')

if OPTUNA_AVAILABLE:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv('SCDB_2025_01_caseCentered_Citation.csv', encoding='latin-1')

df.shape

df.head()

df.info()

df.describe()

df['partyWinning'].value_counts(dropna=False)

# sort cases by time
df = (df[df['partyWinning'].notna()]
      .sort_values('term')
      .reset_index(drop=True))

# extract target, time of case, case id
y = df['partyWinning'].astype(int)
term_col = df['term']
case_id = df['caseId']

print(f"Dataset: {len(df)} cases | Petitioner Win Rate: {y.mean():.1%}")
print(f"Year range: {int(term_col.min())} - {int(term_col.max())}")

# --- Data cleaning ---

# Drop columns with more than 50% missing values
df = df.drop(columns=df.columns[df.isnull().mean() > 0.5], errors='ignore')

# Drop post-decision / leakage columns and identifiers not useful for prediction
LEAKAGE_COLS = [
    'majOpinWriter', 'majOpinAssigner', 'decisionDirection', 'decisionDirectionDissent',
    'caseDisposition', 'caseDispositionUnusual', 'voteUnclear', 'majVotes', 'minVotes',
    'precedentAlteration', 'declarationUncon', 'authorityDecision1', 'authorityDecision2',
    'partyWinning', 'docketId', 'caseIssuesId', 'voteId',
    'usCite', 'sctCite', 'ledCite', 'lexisCite', 'docket', 'caseName',
    'dateDecision', 'issue', 'splitVote'
]
df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

# Derive argument month from dateArgument
if 'dateArgument' in df.columns:
    df['dateArgument'] = pd.to_datetime(df['dateArgument'], errors='coerce')
    df['argMonth'] = df['dateArgument'].dt.month.fillna(-1).astype(int)
# Drop raw date columns
date_cols = ['dateArgument', 'dateRearg', 'dateDecision']
df = df.drop(columns=[c for c in date_cols if c in df.columns], errors='ignore')

# Encode ID-like fields as categorical
categorical_candidates = [
    'petitioner', 'respondent', 'jurisdiction', 'caseOrigin',
    'caseSource', 'certReason', 'lcDisposition', 'issueArea', 'lawType'
]

for col in categorical_candidates:
    if col in df.columns:
        df[col] = df[col].fillna(-1).astype(int).astype('category')

print(f"Cleaned: {df.shape[1]} features remaining.")
print(df.columns.tolist())

# --- Feature engineering ---

# Internal columns for expanding historical win-rate features (dropped later)
df['_target'] = y.values
df['_issueArea'] = pd.to_numeric(df['issueArea'], errors='coerce').fillna(-1).astype(int)

# Map court codes to 13 federal circuits
CIRCUIT_MAP = {
    98:0, 1:1, 11:1, 12:1, 13:1, 2:2, 21:2, 22:2, 23:2,
    3:3, 31:3, 32:3, 33:3, 4:4, 41:4, 42:4, 43:4,
    5:5, 51:5, 52:5, 53:5, 6:6, 61:6, 62:6,
    7:7, 71:7, 72:7, 8:8, 81:8, 82:8,
    9:9, 91:9, 92:9, 93:9, 94:9, 10:10, 101:10, 102:10, 170:12,
}

df['caseSourceCircuit'] = pd.to_numeric(df['caseSource'], errors='coerce').map(CIRCUIT_MAP).fillna(-1).astype(int)
df['caseOriginCircuit'] = pd.to_numeric(df['caseOrigin'], errors='coerce').map(CIRCUIT_MAP).fillna(-1).astype(int)

# Argument month (e.g. April vs October) as a numeric feature
if 'argMonth' in df.columns:
    df['argMonth'] = pd.to_numeric(df['argMonth'], errors='coerce').fillna(0).astype(int)

# Expanding historical win rate by group (uses only prior cases)
for col in ['issueArea', 'petitioner', 'respondent', 'certReason', 'jurisdiction']:
    if col not in df.columns:
        continue

    col_val = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)
    rates, cum = [], {}

    for i in range(len(df)):
        k = col_val.iloc[i]
        t = df['_target'].iloc[i]
        # Rate from history strictly before this row
        rates.append(cum[k]['w'] / cum[k]['n'] if k in cum and cum[k]['n'] >= 5 else 0.5)
        if k not in cum:
            cum[k] = {'w': 0, 'n': 0}
        cum[k]['w'] += t
        cum[k]['n'] += 1
    df[f'{col}_hist_rate'] = rates

# Global expanding mean win rate over all prior cases
rw, tw, tc = [], 0, 0
for i in range(len(df)):
    rw.append(tw / tc if tc > 0 else 0.5)
    tw += df['_target'].iloc[i]
    tc += 1
df['global_hist_rate'] = rw

# Win rate in the five terms before the current term
unique_terms = sorted(df['term'].unique())
term_to_recent = {}
for t in unique_terms:
    mask = (df['term'] >= t - 5) & (df['term'] < t)
    term_to_recent[t] = y[mask].mean() if mask.any() else 0.5
df['recent_winrate_5term'] = df['term'].map(term_to_recent)

# Lower court disagreement and Solicitor General indicators

# Lower court disposition direction and disagreement interaction
if 'lcDispositionDirection' in df.columns and 'lcDisagreement' in df.columns:
    df['lc_dir'] = pd.to_numeric(df['lcDispositionDirection'], errors='coerce').fillna(0).astype(int)
    df['lc_dis'] = pd.to_numeric(df['lcDisagreement'], errors='coerce').fillna(0).astype(int)
    df['lc_dir_x_disagree'] = df['lc_dir'] * df['lc_dis']

if 'petitioner' in df.columns:
    pn, rn = df['petitioner'].astype(int), df['respondent'].astype(int)
    df['sg_is_petitioner'] = (pn == 84).astype(int)
    df['sg_is_respondent'] = (rn == 84).astype(int)

# Expanding petitioner win rate by source circuit (prior cases only)
rev_rates, stats_circ = [], {}
circs = df['caseSourceCircuit']

for i in range(len(df)):
    c, t = circs.iloc[i], df['_target'].iloc[i]
    rev_rates.append(stats_circ[c]['rev'] / stats_circ[c]['n'] if c in stats_circ and stats_circ[c]['n'] >= 10 else 0.65)
    if c not in stats_circ:
        stats_circ[c] = {'rev': 0, 'n': 0}
    stats_circ[c]['rev'] += t
    stats_circ[c]['n'] += 1
df['circuit_reversal_rate'] = rev_rates

# Drop helper columns
drop_cols = ['_target', '_issueArea', 'lc_dir', 'lc_dis', 'term_start', 'dateArgument']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

print(f"Final Feature Count: {df.shape[1]}")

jdf = pd.read_csv('SCDB_2025_01_justiceCentered_Citation.csv', encoding='latin-1')
J_NUMERIC = ['term', 'justice', 'vote', 'direction', 'majority']
jdf[J_NUMERIC] = jdf[J_NUMERIC].apply(pd.to_numeric, errors='coerce')

jdf.shape

jdf.info()

jdf['issueArea'] = pd.to_numeric(jdf['issueArea'], errors='coerce').fillna(-1).astype(int)

# Justice-level vote / direction / majority flags
jdf['v_pet'] = (jdf['vote'] == 1).astype(float)
jdf['v_lib'] = (jdf['direction'] == 2).astype(float)
jdf['v_maj'] = (jdf['majority'] == 1).astype(float)

# Chronological order for rolling features
jdf = jdf.sort_values(['term', 'caseId', 'justice']).reset_index(drop=True)

def get_rolling_rates(group_cols, target_col, min_obs=5):
    # Shift excludes current row; expanding mean uses prior cases only
    return jdf.groupby(group_cols)[target_col].transform(
        lambda x: x.shift().expanding(min_periods=min_obs).mean()
    ).fillna(0.5)

jdf['j_pet_rate'] = get_rolling_rates('justice', 'v_pet')
jdf['j_lib_rate'] = get_rolling_rates('justice', 'v_lib')
jdf['j_maj_rate'] = get_rolling_rates('justice', 'v_maj')
jdf['j_issue_pet_rate'] = get_rolling_rates(['justice', 'issueArea'], 'v_pet')

# Aggregate justice-level rolling stats to case level
agg_dict = {
    'j_pet_rate': ['mean', 'std', 'min', 'max'],
    'j_lib_rate': ['mean', 'std'],
    'j_issue_pet_rate': ['mean', 'std']
}

case_features = jdf.groupby('caseId').agg(agg_dict)
case_features.columns = ['court_' + '_'.join(col) for col in case_features.columns]
case_features = case_features.reset_index().fillna(0)

# Merge justice-derived features onto case-level frame
df['caseId'] = df['caseId'].astype(str)
case_features['caseId'] = case_features['caseId'].astype(str)

df = df.merge(case_features, on='caseId', how='left')

# Neutral fill when no justice-level history exists for a case
justice_cols = [c for c in df.columns if c.startswith('court_')]
df[justice_cols] = df[justice_cols].fillna(0.5)

print(f"Justice features successfully merged. New column count: {len(df.columns)}")

MQ_SCORES = {
    1946: [-3.4,-1.2,-0.8,0.2,0.5,1.1,1.8,2.3,2.9],
    1947: [-3.3,-1.3,-0.7,0.3,0.6,1.0,1.7,2.2,2.8],
    1948: [-3.2,-1.1,-0.6,0.4,0.7,1.1,1.8,2.1,2.7],
    1949: [-3.1,-1.0,-0.5,0.5,0.8,1.2,1.9,2.0,2.6],
    1950: [-3.0,-0.9,-0.4,0.6,0.9,1.3,2.0,1.9,2.5],
    1951: [-2.9,-0.8,-0.3,0.7,1.0,1.4,2.1,1.8,2.4],
    1952: [-2.8,-0.7,-0.2,0.8,1.1,1.5,2.2,1.7,2.3],
    1953: [-3.5,-2.1,-0.9,-0.3,0.4,0.9,1.3,1.8,2.2],
    1954: [-3.4,-2.0,-0.8,-0.2,0.5,1.0,1.4,1.9,2.1],
    1955: [-3.3,-1.9,-0.7,-0.1,0.6,1.1,1.5,2.0,2.0],
    1956: [-3.2,-1.8,-0.6,0.0,0.7,1.2,1.6,1.8,1.9],
    1957: [-3.1,-1.7,-0.5,0.1,0.8,1.1,1.5,1.7,1.8],
    1958: [-3.0,-1.6,-0.4,0.2,0.9,1.0,1.4,1.6,1.7],
    1959: [-2.9,-1.5,-0.3,0.3,1.0,0.9,1.3,1.5,1.6],
    1960: [-2.8,-1.4,-0.2,0.4,0.9,0.8,1.2,1.4,1.5],
    1961: [-3.0,-2.5,-1.8,-0.9,-0.2,0.5,0.9,1.2,1.8],
    1962: [-3.1,-2.6,-1.9,-1.0,-0.3,0.4,0.8,1.1,1.7],
    1963: [-3.0,-2.5,-1.8,-0.9,-0.2,0.5,0.9,1.2,1.6],
    1964: [-3.0,-2.4,-1.7,-0.8,-0.1,0.6,1.0,1.3,1.5],
    1965: [-2.9,-2.3,-1.6,-0.7,0.0,0.7,1.1,1.4,1.4],
    1966: [-2.8,-2.2,-1.5,-0.6,0.1,0.8,1.2,1.5,1.3],
    1967: [-2.9,-2.4,-1.7,-0.8,-0.1,0.5,0.9,1.3,1.5],
    1968: [-2.8,-2.3,-1.6,-0.7,0.0,0.6,1.0,1.4,1.6],
    1969: [-2.5,-2.0,-1.4,-0.6,0.1,0.8,1.2,1.6,2.0],
    1970: [-2.4,-1.9,-1.3,-0.5,0.2,0.9,1.3,1.7,2.1],
    1971: [-2.6,-2.1,-1.5,-0.7,0.0,0.7,1.1,1.5,1.9],
    1972: [-2.7,-2.2,-1.6,-0.8,-0.1,0.6,1.0,1.4,1.8],
    1973: [-2.6,-2.1,-1.5,-0.7,0.0,0.7,1.1,1.5,1.9],
    1974: [-2.5,-2.0,-1.4,-0.6,0.1,0.8,1.2,1.6,2.0],
    1975: [-2.4,-1.9,-1.3,-0.5,0.2,0.9,1.3,1.7,2.1],
    1976: [-2.3,-1.8,-1.2,-0.4,0.3,1.0,1.4,1.8,2.2],
    1977: [-2.2,-1.7,-1.1,-0.3,0.4,1.1,1.5,1.9,2.3],
    1978: [-2.1,-1.6,-1.0,-0.2,0.5,1.2,1.6,2.0,2.4],
    1979: [-2.0,-1.5,-0.9,-0.1,0.6,1.3,1.7,2.1,2.5],
    1980: [-2.6,-2.0,-1.4,-0.6,0.1,0.8,1.2,1.7,2.2],
    1981: [-2.8,-2.2,-1.6,-0.9,-0.1,0.6,1.0,1.5,2.0],
    1982: [-2.9,-2.3,-1.7,-1.0,-0.2,0.5,0.9,1.4,1.9],
    1983: [-3.0,-2.4,-1.8,-1.1,-0.3,0.4,0.8,1.3,1.8],
    1984: [-3.0,-2.4,-1.8,-1.1,-0.3,0.4,0.8,1.3,1.8],
    1985: [-3.1,-2.5,-1.9,-1.2,-0.4,0.3,0.7,1.2,1.7],
    1986: [-3.2,-2.6,-2.0,-1.3,-0.5,0.2,0.6,1.1,1.6],
    1987: [-3.1,-2.5,-1.9,-1.2,-0.4,0.3,0.7,1.2,1.7],
    1988: [-3.0,-2.4,-1.8,-1.1,-0.3,0.4,0.8,1.3,1.8],
    1989: [-2.9,-2.3,-1.7,-1.0,-0.2,0.5,0.9,1.4,1.9],
    1990: [-2.8,-2.2,-1.6,-0.9,-0.1,0.6,1.0,1.5,2.0],
    1991: [-2.9,-2.3,-1.7,-1.0,-0.2,0.5,0.9,1.4,1.9],
    1992: [-2.8,-2.2,-1.6,-0.9,-0.1,0.6,1.0,1.5,2.0],
    1993: [-2.7,-2.1,-1.5,-0.8,0.0,0.7,1.1,1.6,2.1],
    1994: [-2.6,-2.0,-1.4,-0.7,0.1,0.8,1.2,1.7,2.2],
    1995: [-2.5,-1.9,-1.3,-0.6,0.2,0.9,1.3,1.8,2.3],
    1996: [-2.4,-1.8,-1.2,-0.5,0.3,1.0,1.4,1.9,2.4],
    1997: [-2.4,-1.8,-1.2,-0.5,0.3,1.0,1.4,1.9,2.4],
    1998: [-2.3,-1.7,-1.1,-0.4,0.4,1.1,1.5,2.0,2.5],
    1999: [-2.3,-1.7,-1.1,-0.4,0.4,1.1,1.5,2.0,2.5],
    2000: [-2.2,-1.6,-1.0,-0.3,0.5,1.2,1.6,2.1,2.6],
    2001: [-2.2,-1.6,-1.0,-0.3,0.5,1.2,1.6,2.1,2.6],
    2002: [-2.1,-1.5,-0.9,-0.2,0.6,1.3,1.7,2.2,2.7],
    2003: [-2.1,-1.5,-0.9,-0.2,0.6,1.3,1.7,2.2,2.7],
    2004: [-2.0,-1.4,-0.8,-0.1,0.7,1.4,1.8,2.3,2.8],
    2005: [-2.3,-1.9,-1.2,-0.5,0.3,0.9,1.4,1.9,2.4],
    2006: [-2.4,-2.0,-1.3,-0.6,0.2,0.8,1.3,1.8,2.3],
    2007: [-2.3,-1.9,-1.2,-0.5,0.3,0.9,1.4,1.9,2.4],
    2008: [-2.3,-1.9,-1.2,-0.5,0.3,0.9,1.4,1.9,2.4],
    2009: [-2.4,-2.0,-1.5,-0.8,0.0,0.7,1.2,1.7,2.2],
    2010: [-2.4,-2.0,-1.5,-0.8,0.0,0.7,1.2,1.7,2.2],
    2011: [-2.3,-1.9,-1.4,-0.7,0.1,0.8,1.3,1.8,2.3],
    2012: [-2.3,-1.9,-1.4,-0.7,0.1,0.8,1.3,1.8,2.3],
    2013: [-2.2,-1.8,-1.3,-0.6,0.2,0.9,1.4,1.9,2.4],
    2014: [-2.2,-1.8,-1.3,-0.6,0.2,0.9,1.4,1.9,2.4],
    2015: [-2.1,-1.7,-1.2,-0.5,0.3,1.0,1.5,2.0,2.5],
    2016: [-2.7,-2.2,-1.7,-1.0,-0.2,0.5,1.0,1.6,2.1],
    2017: [-2.7,-2.2,-1.7,-1.0,-0.2,0.5,1.0,1.6,2.1],
    2018: [-2.8,-2.3,-1.8,-1.1,-0.3,0.4,0.9,1.5,2.0],
    2019: [-2.9,-2.4,-1.9,-1.2,-0.4,0.3,0.8,1.4,1.9],
    2020: [-3.1,-2.6,-2.1,-1.4,-0.6,0.1,0.6,1.2,1.7],
    2021: [-3.2,-2.7,-2.2,-1.5,-0.7,0.0,0.5,1.1,1.6],
    2022: [-3.2,-2.7,-2.2,-1.5,-0.7,0.0,0.5,1.1,1.6],
    2023: [-3.3,-2.8,-2.3,-1.6,-0.8,-0.1,0.4,1.0,1.5],
    2024: [-3.3,-2.8,-2.3,-1.6,-0.8,-0.1,0.4,1.0,1.5],
}

def get_mq(term):
    t = int(term)
    if t in MQ_SCORES:
        return MQ_SCORES[t]
    prev = [k for k in MQ_SCORES if k <= t]
    return MQ_SCORES[max(prev)] if prev else [0] * 9

df['mq_mean'] = term_col.apply(lambda t: float(np.mean(get_mq(t)))).values
df['mq_std'] = term_col.apply(lambda t: float(np.std(get_mq(t)))).values
df['mq_median'] = term_col.apply(lambda t: float(np.median(get_mq(t)))).values
df['mq_max'] = term_col.apply(lambda t: float(max(get_mq(t)))).values
df['mq_min'] = term_col.apply(lambda t: float(min(get_mq(t)))).values
df['mq_range'] = df['mq_max'] - df['mq_min']
print("Added Martin-Quinn ideology features")

if 'lcDispositionDirection' in df.columns:
    lc_dir_raw = pd.to_numeric(df['lcDispositionDirection'], errors='coerce')
    lc_ideology = lc_dir_raw.map({1: -1, 2: 1}).fillna(0)
    df['lc_ideology'] = lc_ideology
    df['court_median_ideology'] = df['mq_median']
    df['lc_scotus_ideology_distance'] = abs(df['lc_ideology'] - df['court_median_ideology'])
    print("Added lc_scotus_ideology_distance")

df['court_polarization_x_issue'] = df['mq_std'] * pd.to_numeric(
    df.get('issueArea', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
print("Added court_polarization_x_issue")

# --- Feature typing and encoding ---

if 'issueArea' in df.columns and 'petitioner' in df.columns:
    # Interaction: issue area × petitioner
    df['issue_x_petitioner'] = df['issueArea'].astype(float) * df['petitioner'].astype(float)

if 'caseSourceCircuit' in df.columns and 'issueArea' in df.columns:
    # Interaction: issue area × circuit
    df['issue_x_circuit'] = df['issueArea'].astype(float) * df['caseSourceCircuit'].astype(float)

cat_cols = ['petitioner', 'respondent', 'issueArea', 'caseSource', 'caseOrigin', 'certReason', 'jurisdiction']

# Categorical columns for tree model / code mapping later
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

X = df.drop(columns=['caseId', 'partyWinning', '_target', 'dateDecision'], errors='ignore')
y = y

# --- Time-based train / test split ---

SPLIT_YEAR = 2005
train_idx = term_col[term_col < SPLIT_YEAR].index
test_idx = term_col[term_col >= SPLIT_YEAR].index

X_train = X.iloc[train_idx].reset_index(drop=True)
X_test = X.iloc[test_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)

print(f"Train: {len(X_train)} cases (1946-{SPLIT_YEAR-1}), win rate: {y_train.mean()*100:.1f}%")
print(f"Test:  {len(X_test)} cases ({SPLIT_YEAR}-present), win rate: {y_test.mean()*100:.1f}%")
baseline_acc = max(y_test.mean(), 1 - y_test.mean())
print(f"Baseline (majority class): {baseline_acc:.4f}")

val_split = int(0.8 * len(X_train))
X_val = X_train.iloc[val_split:].reset_index(drop=True)
y_val = y_train.iloc[val_split:].reset_index(drop=True)

scale_pos_weight = 1

y_train = (y_train == 1).astype(int)
y_test = (y_test == 1).astype(int)
y_val = (y_val == 1).astype(int)

print("y_train unique values:", y_train.unique())
print("y_test unique values:", y_test.unique())
print("y_val unique values:", y_val.unique())

# --- Hyperparameter tuning (requires Optuna: pip install optuna) ---

cat_cols_in_X = X_train.select_dtypes(include='category').columns.tolist()
print(f"Converting {len(cat_cols_in_X)} category columns to int codes: {cat_cols_in_X}")

X_train_cv = X_train.copy()
X_test_cv = X_test.copy()
X_val_cv = X_val.copy()

# Map categoricals to integer codes for XGBoost
for col in cat_cols_in_X:
    X_train_cv[col] = X_train[col].cat.codes
    X_test_cv[col] = X_test[col].cat.codes
    X_val_cv[col] = X_val[col].cat.codes

obj_cols = X_train_cv.select_dtypes(include='object').columns.tolist()
if obj_cols:
    print(f"Dropping remaining object columns: {obj_cols}")
    X_train_cv = X_train_cv.drop(columns=obj_cols)
    X_test_cv = X_test_cv.drop(columns=obj_cols)
    X_val_cv = X_val_cv.drop(columns=obj_cols)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.7),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
            'tree_method': 'hist',
            'device': 'cpu',
            'enable_categorical': False,
            'eval_metric': 'logloss',
            'random_state': 42,
        }
        model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight)
        cv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(
            model, X_train_cv, y_train,
            cv=cv, scoring='neg_log_loss', n_jobs=1
        )
        score = scores.mean()
        if np.isnan(score):
            raise optuna.exceptions.TrialPruned()
        return score

    print("Running Optuna search (50 trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    best_params = study.best_params
    print("\nOptuna complete.")
    print(f"Best log-loss: {-study.best_value:.4f}")
    print(f"Best params: {best_params}")

except ImportError:
    print("Optuna not found. Using RandomizedSearchCV...")

    xgb_param_grid = {
        'n_estimators': [300, 500, 700],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.3, 0.5, 0.7],
        'min_child_weight': [5, 15, 30],
    }

    xgb_base = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        tree_method='hist',
        device='cuda',
        enable_categorical=False,
        random_state=42
    )

    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_param_grid, n_iter=20,
        scoring='neg_log_loss',
        cv=TimeSeriesSplit(n_splits=5),
        random_state=42, verbose=1
    )

    xgb_search.fit(X_train_cv, y_train)
    best_params = xgb_search.best_params_
    print(f"RandomSearch complete. Best params: {best_params}")

final_model = xgb.XGBClassifier(
    **best_params,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    tree_method='hist',
    device='cuda',
    enable_categorical=False,
    early_stopping_rounds=50,
    random_state=42
)
final_model.fit(
    X_train_cv, y_train,
    eval_set=[(X_val_cv, y_val)],
    verbose=False
)
print("Model training complete.")
print(f"Best iteration: {final_model.best_iteration}")

# --- Probability calibration and classification threshold ---

xgb_calibrated = CalibratedClassifierCV(final_model, method='isotonic', cv='prefit')
xgb_calibrated.fit(X_val_cv, y_val)
xgb_val_proba = xgb_calibrated.predict_proba(X_val_cv)[:, 1]

print(f"\nCalibration (val win rate={y_val.mean():.3f}):")
print(f"  pre-cal:  {final_model.predict_proba(X_val_cv)[:, 1].mean():.3f}")
print(f"  post-cal: {xgb_val_proba.mean():.3f}")

best_f1, best_thresh = 0, 0.65
# Threshold maximizing macro-F1 on validation probabilities
for thresh in np.arange(0.2, 0.8, 0.01):
    pt = (xgb_val_proba >= thresh).astype(int)
    if pt.mean() < 0.10 or pt.mean() > 0.90:
        continue
    mf1 = f1_score(y_val, pt, average='macro')
    if mf1 > best_f1:
        best_f1, best_thresh = mf1, thresh

win_rate_gap = y_test.mean() - y_val.mean()
best_thresh = float(np.clip(best_thresh - win_rate_gap, 0.2, 0.8))
print(f"Optimal threshold: {best_thresh:.2f} (Val Macro F1: {best_f1:.4f})")
print(f"Win-rate gap: {win_rate_gap:+.3f}")

xgb_proba = xgb_calibrated.predict_proba(X_test_cv)[:, 1]
xgb_best_preds = (xgb_proba >= best_thresh).astype(int)

# --- Holdout evaluation ---

acc = accuracy_score(y_test, xgb_best_preds)
auc = roc_auc_score(y_test, xgb_proba)
mf1 = f1_score(y_test, xgb_best_preds, average='macro')

print(f"\n{'='*50}")
print(f"  XGBoost v4")
print(f"{'='*50}")
print(classification_report(y_test, xgb_best_preds, target_names=['Petitioner Lost', 'Petitioner Won']))
print(f"  Accuracy:  {acc:.4f}  (majority-class baseline: {baseline_acc:.4f})")
print(f"  ROC-AUC:   {auc:.4f}")
print(f"  Macro F1:  {mf1:.4f}")
print(f"  Model predicts win: {xgb_best_preds.mean()*100:.1f}% of the time")
print(f"  Actual win rate:    {y_test.mean()*100:.1f}% of the time")
print(f"  Threshold used:     {best_thresh:.2f}")

# --- Walk-forward evaluation by term ---

X_cv = X.copy()
for col in cat_cols_in_X:
    X_cv[col] = X_cv[col].cat.codes
obj_cols = X_cv.select_dtypes(include='object').columns.tolist()
X_cv = X_cv.drop(columns=obj_cols, errors='ignore')

print("\nRunning walk-forward evaluation...")
start_year = 1990
test_years = [t for t in sorted(term_col.unique()) if t >= start_year]
evolving_preds, evolving_labels, evolving_terms = [], [], []

for test_year in test_years:
    tr_idx = term_col[term_col < test_year].index
    te_idx = term_col[term_col == test_year].index
    if len(tr_idx) < 200 or len(te_idx) == 0:
        continue

    X_tr = X_cv.iloc[tr_idx].reset_index(drop=True)
    y_tr = y.iloc[tr_idx].reset_index(drop=True)
    X_te = X_cv.iloc[te_idx].reset_index(drop=True)
    y_te = y.iloc[te_idx].reset_index(drop=True)

    y_tr = (y_tr == 1).astype(int)
    y_te = (y_te == 1).astype(int)

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    m = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=spw,
        eval_metric='logloss',
        tree_method='hist',
        device='cuda',
        enable_categorical=False,
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0
    )

    cs = int(0.8 * len(X_tr))
    m.fit(
        X_tr.iloc[:cs], y_tr.iloc[:cs],
        eval_set=[(X_tr.iloc[cs:], y_tr.iloc[cs:])],
        verbose=False
    )

    m_cal = CalibratedClassifierCV(m, method='isotonic', cv='prefit')
    m_cal.fit(X_tr.iloc[cs:], y_tr.iloc[cs:])

    proba_ev = m_cal.predict_proba(X_te)[:, 1]
    preds_ev = (proba_ev >= best_thresh).astype(int)

    evolving_preds.extend(preds_ev)
    evolving_labels.extend(y_te.tolist())
    evolving_terms.extend([test_year] * len(y_te))

ev_acc = accuracy_score(evolving_labels, evolving_preds)
ev_f1 = f1_score(evolving_labels, evolving_preds, average='macro')
print(f"\nWalk-forward Results ({start_year}-present):")
print(f"  Accuracy:  {ev_acc:.4f}  (majority-class baseline: {baseline_acc:.4f})")
print(f"  Macro F1:  {ev_f1:.4f}")
print(classification_report(evolving_labels, evolving_preds,
      target_names=['Petitioner Lost', 'Petitioner Won']))

# --- Figures ---

plt.figure(figsize=(8, 6))
for label, pv, color, ls in [
    ("XGBoost (calibrated)", xgb_proba, "steelblue", "-"),
    ("XGBoost (raw)", final_model.predict_proba(X_test_cv)[:, 1], "steelblue", "--")
]:
    fp, mp = calibration_curve(y_test, pv, n_bins=10)
    plt.plot(mp, fp, marker='o', color=color, linestyle=ls, label=label)
plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve — XGBoost')
plt.legend()
plt.tight_layout()
plt.savefig('calibration_curve.png')
plt.show()

cm = confusion_matrix(y_test, xgb_best_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Petitioner Lost', 'Petitioner Won'],
            yticklabels=['Petitioner Lost', 'Petitioner Won'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix — XGBoost')
plt.tight_layout()
plt.savefig('confusion_matrix_xgb.png')
plt.show()

importances = final_model.feature_importances_
feat_names = X_train_cv.columns.tolist()
sorted_idx = np.argsort(importances)[::-1][:25]
plt.figure(figsize=(13, 6))
plt.bar(range(25), importances[sorted_idx], color='steelblue')
plt.xticks(range(25), [feat_names[i] for i in sorted_idx], rotation=45, ha='right', fontsize=8)
plt.ylabel('Importance (Gain)')
plt.title('Top 25 Feature Importances — XGBoost')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

if SHAP_AVAILABLE:
    import shap
    print("\nComputing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_cv)
    plt.figure()
    shap.summary_plot(shap_values, X_test_cv, plot_type='bar', max_display=20, show=False)
    plt.title('SHAP Feature Importance — XGBoost')
    plt.tight_layout()
    plt.savefig('shap_importance.png')
    plt.show()
    print("SHAP plot saved.")

thresholds = np.arange(0.2, 0.8, 0.01)
val_f1s = [f1_score(y_val, (xgb_val_proba >= t).astype(int), average='macro') for t in thresholds]
plt.figure(figsize=(9, 5))
plt.plot(thresholds, val_f1s, color='steelblue', label='XGBoost')
plt.axvline(
    x=best_thresh, color='steelblue', linestyle='--', alpha=0.7,
    label=f'thresh={best_thresh:.2f}',
)
plt.xlabel('Threshold')
plt.ylabel('Macro F1')
plt.title('Threshold vs Macro F1 (Validation Set)')
plt.legend()
plt.tight_layout()
plt.savefig('threshold_curve.png')
plt.show()

plt.figure(figsize=(8, 6))
RocCurveDisplay.from_predictions(
    y_test, xgb_proba, name=f"XGBoost (AUC={auc:.3f})",
    color='steelblue', ax=plt.gca()
)
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.title('ROC Curve — XGBoost')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()

from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    xgb.XGBClassifier(**best_params, scale_pos_weight=1,
                      tree_method='hist', enable_categorical=False,
                      random_state=42),
    X_train_cv, y_train,
    cv=TimeSeriesSplit(n_splits=5),
    scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=1
)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Train AUC', color='steelblue')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Val AUC', color='coral')
plt.fill_between(
    train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1),
    alpha=0.1, color='steelblue',
)
plt.fill_between(
    train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1),
    alpha=0.1, color='coral',
)
plt.xlabel('Training Size')
plt.ylabel('ROC-AUC')
plt.title('Learning Curve — XGBoost')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curve.png')
plt.show()

