"""(Re)generate notebooks/01_data_understanding.ipynb and notebooks/02_eda.ipynb.
The notebooks only READ reports/ and data/processed/ produced by the pipeline; they contain no
hard-coded results. Run the pipeline first, then open them in Jupyter / VS Code."""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''import os, sys, json
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
ROOT = Path(os.environ.get("ROADRISK_ROOT") or Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd())
sys.path.insert(0, str(ROOT))
from ml.config import get_settings
from ml.feature_config import CLASS_NAMES
cfg = get_settings()
pd.set_option("display.max_columns", 50, "display.width", 140)
def load_report(name):
    p = cfg.reports_dir / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run the pipeline scripts first (see README).")
    return json.loads(p.read_text(encoding="utf-8"))
COLORS = {"Slight": "#5b8def", "Serious": "#f2a33a", "Fatal": "#d64545"}
'''


def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(True)}


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


nb1 = nb([
    md("# 01 - Data understanding (STATS19)\nReads the reports written by `inspect_data.py` and `prepare_data.py`. "
       "Nothing here is hard-coded; every number comes from the real files you downloaded.\n\n"
       "**Questions this notebook answers:** What is the dataset (source, licence, size, period)? Which columns exist? "
       "How much is missing/unknown? Are there duplicates or outliers? How imbalanced is the target? Which columns are excluded to prevent leakage, and why?"),
    code(SETUP),
    md("## 1. Source, licence, size, period"),
    code('''dr = load_report("data_report.json")
d = dr["dataset"]
print("Source  :", d["name"]); print("Licence :", d["licence"])
print("Years requested:", d["years_requested"], "| used:", d["years_used"], "| missing:", d["years_missing"])
print(f"Rows: {dr['n_rows_loaded']:,}  Columns loaded: {dr['n_columns_loaded']}")
pd.DataFrame(d["files"])[["year", "file", "size_mb", "rows_loaded", "columns_in_file"]]'''),
    md("## 2. Raw schema (all columns in the files)"),
    code('''sr = load_report("schema_report.json")
first = sorted(sr["files"])[0]
cols = pd.DataFrame(sr["files"][first]["columns"])[["column", "dtype_inferred", "pct_missing", "n_unique", "sample_values", "used_by_pipeline"]]
print("Columns differing between years:", sr["column_differences_between_years"]["not_in_all_years"])
cols'''),
    md("## 3. Missing values, special codes, duplicates"),
    code('''miss = pd.DataFrame(dr["columns"]).sort_values("pct_missing", ascending=False)
display(miss.head(15))
print("Duplicates:", dr["duplicates"])
pd.DataFrame({c: {"% unknown/missing": v["pct_unknown_or_missing"]} for c, v in dr["special_code_analysis"].items()}).T.sort_values("% unknown/missing", ascending=False)'''),
    md("## 4. Outliers and suspicious values"),
    code('''print("Outliers (IQR rule):"); display(pd.DataFrame(dr["outlier_analysis_iqr"]).T)
print("Handling:", dr["outlier_handling"])
print("Suspicious values:"); display(pd.Series(dr["suspicious_values"], name="count").to_frame())'''),
    md("## 5. Target distribution and imbalance"),
    code('''td = load_report("target_distribution.json")
print("Raw codes observed:", td["raw_values_observed"], "-> mapping", td["mapping_used"])
print("Sanity check:", td["sanity_check_note"])
s = pd.Series(td["class_counts"]).reindex(CLASS_NAMES)
print((100 * s / s.sum()).round(2).to_string(), "\\nimbalance ratio:", td["imbalance_ratio_majority_to_minority"])
ax = s.plot.bar(color=[COLORS[c] for c in s.index], figsize=(5, 3), title="Collision severity (rows)"); ax.set_yscale("log"); plt.show()'''),
    md("## 6. Leakage control: features used vs excluded"),
    code('''print("Excluded features and reasons:")
pd.Series(dr["excluded_features"], name="reason").to_frame()'''),
    code('''lr = cfg.reports_dir / "leakage_report.json"
if lr.exists():
    L = json.loads(lr.read_text()); print("Leakage audit passed:", L["passed"]); print(L["checks"])
    display(pd.Series(L["single_feature_cramers_v_with_target"], name="Cramer's V vs target").sort_values(ascending=False).head(10).to_frame())
else:
    print("leakage_report.json is created by train_models.py (run it first).")'''),
    md("## Your observations\n_Write your own conclusions here after running the notebook on your data._"),
])

nb2 = nb([
    md("# 02 - Exploratory data analysis\nUses `reports/eda_summary.json` (produced by `scripts/run_eda.py` from **train + validation rows only**, so the test set stays unseen) and "
       "the processed data. Charts are tied to a question; interpret them in the last cell.\n\n"
       "**Questions:** How does severity change with speed limit, road type, lighting, weather, surface, urban/rural and time? "
       "Which features are most associated with severity (effect size, not p-value)?"),
    code(SETUP),
    code('''eda = load_report("eda_summary.json")
print("Rows analysed:", f"{eda['rows_analysed']:,}", "| excluded:", eda["excluded_from_eda"])
print("Headline stats:", eda["headline_stats"])
pd.Series(eda["target_distribution"]).to_frame("rows")'''),
    md("## Which features are associated with severity? (Cram\u00e9r's V = effect size)"),
    code('''assoc = pd.Series({k: v["cramers_v"] for k, v in eda["association_with_severity"].items()}).sort_values()
assoc.plot.barh(figsize=(7, 6), color="#4a6fa5", title="Association with severity (Cram\u00e9r's V)"); plt.tight_layout(); plt.show()'''),
    md("## Severity mix by feature (% of collisions within each level)"),
    code('''def plot_rates(feature, order=None):
    t = pd.DataFrame(eda["severity_by_feature"][feature]).T
    t = t.loc[order] if order else t
    ax = t[[f"pct_{c}" for c in CLASS_NAMES]].rename(columns=lambda c: c[4:]).plot.bar(stacked=True, figsize=(8, 3.2),
          color=[COLORS[c] for c in CLASS_NAMES], title=f"Severity mix by {feature}")
    ax.set_ylabel("% of collisions"); ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1)); plt.tight_layout(); plt.show()
    return t[["n"] + [f"pct_{c}" for c in CLASS_NAMES]]
for f in ["speed_limit", "road_type_group", "lighting_group", "weather_group", "surface_group", "urban_or_rural_area", "junction_group"]:
    if f in eda["severity_by_feature"]:
        display(plot_rates(f))'''),
    md("## Severity vs time"),
    code('''for f in ["hour", "weekday", "month", "is_night"]:
    if f in eda["severity_by_feature"]:
        t = pd.DataFrame(eda["severity_by_feature"][f]).T
        t.index = t.index.astype(float).astype(int) if f != "is_night" else t.index
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        t["n"].plot(ax=ax[0], title=f"Collisions by {f}", marker="o")
        t[["pct_Serious", "pct_Fatal"]].plot(ax=ax[1], title=f"Serious / Fatal share by {f} (%)", marker="o", color=["#f2a33a", "#d64545"])
        plt.tight_layout(); plt.show()'''),
    md("## Trend over time"),
    code('''if "monthly_counts" in eda:
    m = pd.DataFrame(eda["monthly_counts"]).T[CLASS_NAMES]; m.plot(figsize=(10, 3), color=[COLORS[c] for c in CLASS_NAMES], title="Monthly collisions by severity"); plt.show()
if "yearly_counts" in eda:
    display(pd.DataFrame(eda["yearly_counts"]).T)'''),
    md("## Numeric distributions"),
    code('pd.DataFrame(eda["numeric_distributions"])'),
    md("## Your interpretation\n_After running on your data, write what the charts show (and what they do NOT show: association is not causation; fatal collisions are rare, so percentages in small groups are noisy)._"),
])

out = ROOT / "notebooks"
out.mkdir(exist_ok=True)
for name, n in (("01_data_understanding.ipynb", nb1), ("02_eda.ipynb", nb2)):
    (out / name).write_text(json.dumps(n, indent=1), encoding="utf-8")
    print("wrote", out / name)
