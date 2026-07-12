from pathlib import Path
import html
import re
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


BASE_DIR = Path(__file__).resolve().parent
DATA_PATHS = [
    BASE_DIR / "masonry_values_filled.csv",
    BASE_DIR.parent / "masonry_values_filled.csv",
]
DATA_PATH = next((path for path in DATA_PATHS if path.exists()), DATA_PATHS[0])
DATASET_VIEWS = {
    "Reported values only": "reported_only",
    "Reported + model-estimated values": "reported_plus_estimated",
}
DATASET_OPTIONS = list(DATASET_VIEWS.keys())


st.set_page_config(
    page_title="Masonry Estimator With Graphs",
    page_icon="",
    layout="centered",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .intro-box {
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        background: #ffffff;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .estimate-box {
        border-left: 5px solid #2563eb;
        background: #eff6ff;
        border-radius: 6px;
        padding: 1rem 1.1rem;
        margin: 1rem 0;
    }
    .estimate-number {
        font-size: 2.2rem;
        font-weight: 750;
        color: #1d4ed8;
        line-height: 1.1;
        margin: 0.25rem 0 0.65rem 0;
    }
    .badge {
        display: inline-block;
        border-radius: 999px;
        background: #dbeafe;
        color: #1e40af;
        padding: 0.32rem 0.68rem;
        font-size: 0.86rem;
        font-weight: 700;
        margin: 0.25rem 0 0.9rem 0;
    }
    .dataset-note {
        border: 1px solid #bfdbfe;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        background: #eff6ff;
        color: #172033;
        padding: 0.9rem 1rem;
        margin: 0.2rem 0 1rem 0;
        line-height: 1.55;
    }
    .dataset-note strong {
        color: #1d4ed8;
    }
    .muted {
        color: #52616b;
        font-size: 0.95rem;
    }
    .simple-chart {
        background: transparent;
        padding: 0.45rem 0 1.6rem 0;
        margin: 0.5rem 0 1.65rem 0;
    }
    .simple-chart-row {
        display: grid;
        grid-template-columns: minmax(150px, 18%) minmax(360px, 1fr);
        gap: 0.55rem;
        align-items: center;
        min-height: 25px;
        margin: 0.08rem 0;
    }
    .simple-chart-label {
        color: #718096;
        font-size: 0.82rem;
        line-height: 1.25;
        overflow-wrap: anywhere;
        text-align: right;
    }
    .simple-chart-track {
        position: relative;
        height: 22px;
        border-radius: 0;
        background-image: repeating-linear-gradient(
            to right,
            #dbe3f1 0,
            #dbe3f1 1px,
            transparent 1px,
            transparent 8.333%
        );
        overflow: visible;
    }
    .simple-chart-bar {
        height: 18px;
        border-radius: 0;
        margin-top: 2px;
    }
    .simple-chart-value {
        position: absolute;
        top: 50%;
        left: min(calc(var(--bar-width) + 5px), calc(100% - 72px));
        transform: translateY(-50%);
        color: #172033;
        font-size: 0.79rem;
        font-weight: 650;
        text-align: left;
        white-space: nowrap;
    }
    .simple-chart-axis {
        display: grid;
        grid-template-columns: minmax(150px, 18%) minmax(360px, 1fr);
        gap: 0.55rem;
        margin-top: 0.5rem;
        color: #718096;
        font-size: 0.78rem;
    }
    .simple-chart-ticks {
        display: flex;
        justify-content: space-between;
        padding-left: 0;
    }
    .simple-chart-axis-title {
        grid-column: 2 / 3;
        text-align: center;
        margin-top: 0.7rem;
        color: #718096;
        font-size: 0.86rem;
    }
    @media (max-width: 640px) {
        .simple-chart-row {
            grid-template-columns: 1fr;
            gap: 0.3rem;
            margin-bottom: 0.85rem;
        }
        .simple-chart-label {
            text-align: left;
        }
        .simple-chart-value {
            left: auto;
            right: 0.25rem;
        }
        .simple-chart-axis {
            display: none;
        }
        .simple-chart-axis-title {
            display: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_column_name(name: str) -> str:
    name = name.strip().lower()
    name = name.replace(" ", "_").replace("/", "_").replace(".", "_")
    return re.sub(r"[^a-z0-9_]", "", name)


def clean_money(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan}),
        errors="coerce",
    )


def money(value: float) -> str:
    return f"${value:,.2f}"


def evidence_label(reported_records: int) -> tuple[str, str]:
    if reported_records >= 10:
        return "Moderate evidence", "This category has 10 or more reported masonry records."
    if reported_records >= 5:
        return "Limited evidence", "This category has 5 to 9 reported masonry records."
    return (
        "Very limited evidence",
        f"This estimate is based on only {reported_records:,} reported masonry record"
        f"{'' if reported_records == 1 else 's'} and may not be reliable.",
    )


@st.cache_data(show_spinner=False)
def load_project_data(dataset_view: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    data = pd.read_csv(DATA_PATH, encoding="cp1252")
    data.columns = [clean_column_name(col) for col in data.columns]

    required = {
        "project_id",
        "project_title",
        "specific_subcategory",
        "project_value",
        "masonry_value",
        "masonry_value_filled",
        "masonry_value_source",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            "masonry_values_filled.csv is missing these columns: "
            + ", ".join(missing)
        )

    data["project_value"] = clean_money(data["project_value"])
    data["masonry_value"] = clean_money(data["masonry_value"])
    data["masonry_value_filled"] = clean_money(data["masonry_value_filled"])
    data["specific_subcategory"] = (
        data["specific_subcategory"].fillna("Unspecified").astype(str).str.strip()
    )
    data["masonry_value_source"] = (
        data["masonry_value_source"].fillna("Unavailable").astype(str).str.strip().str.title()
    )

    if dataset_view == "reported_only":
        data = data[data["masonry_value_source"].eq("Reported")].copy()
        data["masonry_value_filled"] = data["masonry_value"]
        data["masonry_value_source"] = "Reported"

    usable = data[
        data["project_value"].notna()
        & data["masonry_value_filled"].notna()
        & (data["project_value"] > 0)
        & (data["masonry_value_filled"] > 0)
    ].copy()
    usable["masonry_percent"] = usable["masonry_value_filled"] / usable["project_value"]

    reported_mask = usable["masonry_value_source"].eq("Reported")
    predicted_mask = usable["masonry_value_source"].eq("Predicted")

    summary = (
        usable.assign(
            reported_record=reported_mask.astype(int),
            predicted_record=predicted_mask.astype(int),
        )
        .groupby("specific_subcategory", dropna=False)
        .agg(
            available_project_records=("project_id", "count"),
            reported_project_records=("reported_record", "sum"),
            predicted_project_records=("predicted_record", "sum"),
            median_masonry_percent=("masonry_percent", "median"),
            median_masonry_value=("masonry_value_filled", "median"),
        )
        .reset_index()
        .sort_values("specific_subcategory")
    )

    overall_median = float(summary["median_masonry_percent"].median())
    return data, summary, overall_median


def reset_inputs() -> None:
    st.session_state.subcategory = "Apartments"
    st.session_state.project_value = 1_000_000
    st.session_state.show_result = False


def sync_graph_dataset_selector() -> None:
    st.session_state.graph_dataset_label = st.session_state.dataset_label


def sync_top_dataset_selector() -> None:
    st.session_state.dataset_label = st.session_state.graph_dataset_label


def make_comparison_data(
    summary: pd.DataFrame,
    selected: str,
    overall_median: float,
    reference_summary: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    comparison_source = reference_summary if reference_summary is not None else summary
    comparison_categories = (
        comparison_source.sort_values("median_masonry_percent", ascending=False)
        .head(8)["specific_subcategory"]
        .tolist()
    )
    top = summary[summary["specific_subcategory"].isin(comparison_categories)].copy()
    top["comparison_order"] = top["specific_subcategory"].map(
        {category: index for index, category in enumerate(comparison_categories)}
    )
    top = top.sort_values("comparison_order")
    selected_row = summary[summary["specific_subcategory"] == selected].copy()

    selected_label = (
        selected_row["specific_subcategory"].iloc[0] if not selected_row.empty else selected
    )
    selected_row["display_category"] = "Selected category"
    top["display_category"] = top["specific_subcategory"]

    overall = pd.DataFrame(
        {
            "specific_subcategory": ["Overall category median"],
            "display_category": ["Overall category median"],
            "available_project_records": [np.nan],
            "reported_project_records": [np.nan],
            "predicted_project_records": [np.nan],
            "median_masonry_percent": [overall_median],
            "median_masonry_value": [np.nan],
        }
    )

    chart_data = pd.concat([selected_row, overall, top], ignore_index=True)
    chart_data = chart_data.drop_duplicates("display_category")
    chart_data["full_category_name"] = chart_data["specific_subcategory"]
    chart_data["masonry_percent_label"] = (
        chart_data["median_masonry_percent"] * 100
    ).map(lambda value: f"{value:.2f}%")
    chart_data["record_count_label"] = chart_data["reported_project_records"].map(
        lambda value: "" if pd.isna(value) else f"{int(value):,}"
    )
    chart_data["chart_group"] = np.select(
        [
            chart_data["display_category"].eq("Selected category"),
            chart_data["display_category"].eq("Overall category median"),
        ],
        ["Selected category", "Overall median"],
        default="Other categories",
    )
    chart_data["sort_order"] = range(1, len(chart_data) + 1)
    chart_data["chart_label"] = chart_data.apply(
        lambda row: (
            f"{int(row['sort_order'])}. Selected: {row['full_category_name']}"
            if row["display_category"] == "Selected category"
            else f"{int(row['sort_order'])}. {row['full_category_name']}"
        ),
        axis=1,
    )
    return chart_data


def chart_color(chart_group: str) -> str:
    colors = {
        "Selected category": "#2563eb",
        "Overall median": "#64748b",
        "Other categories": "#c7d2fe",
    }
    return colors.get(chart_group, "#c7d2fe")


CHART_COMPONENT_STYLE = """
<style>
body {
    margin: 0;
    background: transparent;
    font-family: "Source Sans Pro", Arial, sans-serif;
}
.simple-chart {
    background: transparent;
    padding: 0.45rem 0 1.2rem 0;
}
.simple-chart-row {
    display: grid;
    grid-template-columns: minmax(150px, 18%) minmax(360px, 1fr);
    gap: 0.55rem;
    align-items: center;
    min-height: 25px;
    margin: 0.08rem 0;
}
.simple-chart-label {
    color: #718096;
    font-size: 0.82rem;
    line-height: 1.25;
    overflow-wrap: anywhere;
    text-align: right;
}
.simple-chart-track {
    position: relative;
    height: 22px;
    border-radius: 0;
    background-image: repeating-linear-gradient(
        to right,
        #dbe3f1 0,
        #dbe3f1 1px,
        transparent 1px,
        transparent 8.333%
    );
    overflow: visible;
}
.simple-chart-bar {
    height: 18px;
    border-radius: 0;
    margin-top: 2px;
}
.simple-chart-value {
    position: absolute;
    top: 50%;
    left: min(calc(var(--bar-width) + 5px), calc(100% - 72px));
    transform: translateY(-50%);
    color: #172033;
    font-size: 0.79rem;
    font-weight: 650;
    text-align: left;
    white-space: nowrap;
}
.simple-chart-axis {
    display: grid;
    grid-template-columns: minmax(150px, 18%) minmax(360px, 1fr);
    gap: 0.55rem;
    margin-top: 0.5rem;
    color: #718096;
    font-size: 0.78rem;
}
.simple-chart-ticks {
    display: flex;
    justify-content: space-between;
}
.simple-chart-axis-title {
    grid-column: 2 / 3;
    text-align: center;
    margin-top: 0.7rem;
    color: #718096;
    font-size: 0.86rem;
}
@media (max-width: 640px) {
    .simple-chart-row {
        grid-template-columns: 1fr;
        gap: 0.3rem;
        margin-bottom: 0.85rem;
    }
    .simple-chart-label {
        text-align: left;
    }
    .simple-chart-value {
        left: auto;
        right: 0.25rem;
    }
    .simple-chart-axis,
    .simple-chart-axis-title {
        display: none;
    }
}
</style>
"""


def render_bar_chart(
    chart_data: pd.DataFrame,
    value_column: str,
    value_label_column: str,
    axis_title: str,
) -> str:
    data = chart_data[chart_data[value_column].notna()].sort_values("sort_order")
    max_value = data[value_column].max()
    if pd.isna(max_value) or max_value <= 0:
        max_value = 1

    if value_column == "median_masonry_percent":
        chart_max = max(0.01, np.ceil(max_value * 100) / 100)
        ticks = [f"{int(round(chart_max * tick * 100))}%" for tick in np.linspace(0, 1, 7)]
    else:
        chart_max = max_value
        ticks = [f"{int(round(chart_max * tick)):,}" for tick in np.linspace(0, 1, 6)]

    rows = []
    for _, row in data.iterrows():
        value = float(row[value_column])
        width = max(1.2, min((value / chart_max) * 100, 100))
        label = html.escape(str(row["chart_label"]))
        value_label = html.escape(str(row[value_label_column]))
        color = chart_color(str(row["chart_group"]))
        record_value = row["available_project_records"]
        record_text = "" if pd.isna(record_value) else f"{int(record_value):,}"
        title = html.escape(
            f"{row['full_category_name']} | Total usable records: "
            f"{record_text}"
        )
        rows.append(
            f'<div class="simple-chart-row" title="{title}">'
            f'<div class="simple-chart-label">{label}</div>'
            f'<div class="simple-chart-track" style="--bar-width: {width:.2f}%;">'
            f'<div class="simple-chart-bar" style="width: {width:.2f}%; background: {color};"></div>'
            f'<div class="simple-chart-value">{value_label}</div>'
            f'</div>'
            f'</div>'
        )
    axis = (
        '<div class="simple-chart-axis">'
        '<div></div>'
        f'<div class="simple-chart-ticks">{"".join(f"<span>{tick}</span>" for tick in ticks)}</div>'
        f'<div class="simple-chart-axis-title">{html.escape(axis_title)}</div>'
        '</div>'
    )
    return CHART_COMPONENT_STYLE + '<div class="simple-chart">' + "".join(rows) + axis + "</div>"


st.title("Masonry Value Estimator")
st.markdown(
    """
    <div class="intro-box">
    This prototype estimates masonry value using the median masonry percentage
    observed among projects in the selected subcategory. It is designed as an
    explainable decision-support estimate, not as a guaranteed project cost.
    </div>
    """,
    unsafe_allow_html=True,
)

if "dataset_label" not in st.session_state:
    st.session_state.dataset_label = DATASET_OPTIONS[1]
if "graph_dataset_label" not in st.session_state:
    st.session_state.graph_dataset_label = st.session_state.dataset_label

dataset_label = st.radio(
    "Dataset view",
    DATASET_OPTIONS,
    horizontal=True,
    key="dataset_label",
    on_change=sync_graph_dataset_selector,
    help=(
        "Use the first view to show only records with reported masonry values. "
        "Use the second view to show the final enriched project output after model filling."
    ),
)
dataset_view = DATASET_VIEWS[dataset_label]
st.markdown(
    """
    <div class="dataset-note">
    <strong>Both views are generated from the final dataset after cleaning and
    machine-learning estimation.</strong><br>
    This app uses the final dataset produced after the team's machine-learning
    estimation process.
    </div>
    """,
    unsafe_allow_html=True,
)
badge_text = (
    "Based on reported masonry values only"
    if dataset_view == "reported_only"
    else "Based on original + predicted masonry values"
)
st.markdown(f'<div class="badge">{badge_text}</div>', unsafe_allow_html=True)

if not DATA_PATH.exists():
    checked = "\n".join(str(path) for path in DATA_PATHS)
    st.error(f"masonry_values_filled.csv was not found. Checked:\n{checked}")
    st.stop()

try:
    data, summary, overall_median = load_project_data(dataset_view)
    _, reference_summary, _ = load_project_data("reported_plus_estimated")
except Exception as exc:
    st.error(str(exc))
    st.stop()

if summary.empty:
    st.error("No usable records were found for the selected dataset view.")
    st.stop()

if "show_result" not in st.session_state:
    st.session_state.show_result = False

usable_records = data[
    data["project_value"].notna()
    & data["masonry_value_filled"].notna()
    & (data["project_value"] > 0)
    & (data["masonry_value_filled"] > 0)
]
reported_records = usable_records["masonry_value_source"].eq("Reported").sum()
predicted_records = usable_records["masonry_value_source"].eq("Predicted").sum()

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Reported values", f"{reported_records:,}")
metric2.metric("Predicted values", f"{predicted_records:,}")
metric3.metric("Total usable records", f"{len(usable_records):,}")
metric4.metric("Categories available", f"{len(summary):,}")

st.divider()

categories = summary["specific_subcategory"].tolist()
default_category = "Apartments" if "Apartments" in categories else categories[0]

if "subcategory" not in st.session_state:
    st.session_state.subcategory = default_category
elif st.session_state.subcategory not in categories:
    st.session_state.subcategory = default_category
if "project_value" not in st.session_state:
    st.session_state.project_value = 1_000_000

st.subheader("1. Enter Project Information")

left, right = st.columns([1.25, 1])
with left:
    st.selectbox("Subcategory", categories, key="subcategory")
with right:
    st.number_input(
        "Project Value",
        min_value=0,
        step=50_000,
        format="%d",
        key="project_value",
    )

button_col, reset_col = st.columns([2, 1])
with button_col:
    if st.button("Estimate Masonry Value", type="primary", use_container_width=True):
        st.session_state.show_result = True
with reset_col:
    st.button("Reset", use_container_width=True, on_click=reset_inputs)

selected = st.session_state.subcategory
project_value = float(st.session_state.project_value)
selected_row = summary[summary["specific_subcategory"] == selected]

if selected_row.empty:
    median_percent = overall_median
    available_count = 0
    reported_count = 0
    predicted_count = 0
else:
    row = selected_row.iloc[0]
    median_percent = float(row["median_masonry_percent"])
    available_count = int(row["available_project_records"])
    reported_count = int(row["reported_project_records"])
    predicted_count = int(row["predicted_project_records"])

estimate = project_value * median_percent
evidence_title, evidence_message = evidence_label(reported_count)

if st.session_state.show_result:
    st.markdown(
        f"""
        <div class="estimate-box">
        <div class="muted">Estimated Masonry Value</div>
        <div class="estimate-number">{money(estimate)}</div>
        <b>Subcategory:</b> {selected}<br>
        <b>Project Value:</b> {money(project_value)}<br>
        <b>Estimation Method:</b> Category median masonry percentage ({median_percent * 100:.2f}%)<br>
        <b>Project records used in the estimate:</b> {available_count:,} total usable records
        ({reported_count:,} reported, {predicted_count:,} predicted).
        </div>
        """,
        unsafe_allow_html=True,
    )

    if reported_count < 5:
        st.warning(f"{evidence_title}: {evidence_message}", icon="⚠")
    else:
        st.info(f"{evidence_title}: {evidence_message}")

st.divider()
st.subheader("2. Graphs That Explain the Estimate")

st.radio(
    "Dataset view",
    DATASET_OPTIONS,
    horizontal=True,
    key="graph_dataset_label",
    on_change=sync_top_dataset_selector,
    help=(
        "Switch here to see how the graphs change when model-estimated masonry "
        "values are included."
    ),
)

chart_data = make_comparison_data(
    summary, selected, overall_median, reference_summary=reference_summary
)

st.write("**How the selected category compares with other categories**")
components.html(
    render_bar_chart(
        chart_data,
        "median_masonry_percent",
        "masonry_percent_label",
        "Median masonry percentage",
    ),
    height=345,
    scrolling=False,
)

st.write("**Number of reported masonry records supporting each category**")
components.html(
    render_bar_chart(
        chart_data,
        "reported_project_records",
        "record_count_label",
        "Reported masonry records",
    ),
    height=345,
    scrolling=False,
)

selected_rank = (
    summary["median_masonry_percent"].rank(method="min", ascending=False)
    [summary["specific_subcategory"] == selected]
)
rank_text = int(selected_rank.iloc[0]) if not selected_rank.empty else "N/A"

explain1, explain2, explain3 = st.columns(3)
explain1.metric("Selected category", selected)
explain2.metric("Overall category median", f"{overall_median * 100:.2f}%")
explain3.metric("Rank", f"{rank_text} of {len(summary)}")

st.markdown("### Short Explanation")
st.write(
    """
    The estimate uses the category median masonry percentage from the selected
    dataset view. The first graph places the selected category first, compares it
    with the overall category median, and then shows high-masonry categories for
    context. The second graph shows how many project records with reported
    masonry values support each category.
    """
)

with st.expander("Formula"):
    st.code(
        "estimated masonry value = project value x category median masonry percentage",
        language="text",
    )

with st.expander("View the category summary table"):
    table = summary.copy()
    table["median_masonry_percent"] = table["median_masonry_percent"] * 100
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "specific_subcategory": st.column_config.TextColumn("Subcategory"),
            "available_project_records": st.column_config.NumberColumn(
                "Total usable records", format="%d"
            ),
            "reported_project_records": st.column_config.NumberColumn(
                "Reported records", format="%d"
            ),
            "predicted_project_records": st.column_config.NumberColumn(
                "Predicted records", format="%d"
            ),
            "median_masonry_percent": st.column_config.NumberColumn(
                "Median masonry %", format="%.2f%%"
            ),
            "median_masonry_value": st.column_config.NumberColumn(
                "Median masonry value", format="$%d"
            ),
        },
    )
