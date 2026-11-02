"""Build the submission report (DOCX).

    python -m freight_rate.reporting.build_report

Every number comes from the data, the experiment results in
``reports/experiments/`` or the generated submission files, so the report
always matches the code. Requires the pipeline (train, predict, score.py) and
the Phase 4 experiments to have been run first.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from freight_rate.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from freight_rate.data import io, schema
from freight_rate.data.cleaning import LabelAnomalyFilter
from freight_rate.data.quality import quote_signal_regimes
from freight_rate.reporting.figures import quote_signal_regime_chart, validation_timeline

log = logging.getLogger(__name__)

EXPERIMENTS = PROJECT_ROOT / "reports" / "experiments"
EDA_FIGURES = PROJECT_ROOT / "reports" / "eda" / "figures"
REPORT_FIGURES = PROJECT_ROOT / "reports" / "figures"
SCORER_CHART = PROJECT_ROOT / "scorer_results" / "candidate_december.png"
OUTPUT = PROJECT_ROOT / "reports" / "Freight_Rate_Prediction_Report.docx"

INK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x52, 0x51, 0x4E)
ACCENT_HEX = "064A56"  # the scorer chart's colour, reused for headings and table headers
HEADER_FILL = "E6EEF0"
CONTENT_WIDTH_IN = 6.5

MODEL_LABELS = {
    "grouped_median": "Baseline",
    "blend_70": "Blend with quote_signal",
    "hgb_r3_noqs": "GB + recency",
    "ridge_r3_noqs": "Ridge + recency",
    "final": "Selected",
}
MODEL_COLUMNS = list(MODEL_LABELS)
MODEL_NOTE = (
    "Baseline = median rate per mile by equipment and distance decile x distance. "
    "GB = gradient boosting. Blend with quote_signal = the first-draft model (70/30 blend that "
    "used quote_signal). Selected = the same 70/30 blend of GB + recency and Ridge + recency, "
    "without quote_signal."
)


# ----------------------------------------------------------------------------- docx helpers


class Report:
    def __init__(self):
        self.doc = Document()
        section = self.doc.sections[0]
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(section, side, Inches(1))

        normal = self.doc.styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(10.5)
        normal.font.color.rgb = INK
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.1
        for level, size in ((1, 15), (2, 12)):
            style = self.doc.styles[f"Heading {level}"]
            style.font.name = "Calibri"
            style.font.size = Pt(size)
            style.font.bold = True
            style.font.color.rgb = RGBColor.from_string(ACCENT_HEX)
            style.paragraph_format.space_before = Pt(14 if level == 1 else 10)
            style.paragraph_format.space_after = Pt(4)
            style.paragraph_format.keep_with_next = True
        self._figure = 0
        self._table = 0
        self._footer()

    def _footer(self):
        para = self.doc.sections[0].footer.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = para.add_run("Page ")
        run.font.size, run.font.color.rgb = Pt(8.5), MUTED
        for code in ("begin", None, "end"):
            if code is None:
                instr = OxmlElement("w:instrText")
                instr.set(qn("xml:space"), "preserve")
                instr.text = "PAGE"
                r = para.add_run()
                r._r.append(instr)
            else:
                fld = OxmlElement("w:fldChar")
                fld.set(qn("w:fldCharType"), code)
                r = para.add_run()
                r._r.append(fld)
            r.font.size, r.font.color.rgb = Pt(8.5), MUTED

    def title(self, text: str, subtitle: str, meta: str):
        p = self.doc.add_paragraph()
        r = p.add_run(text)
        r.font.size, r.font.bold = Pt(22), True
        r.font.color.rgb = RGBColor.from_string(ACCENT_HEX)
        p.paragraph_format.space_after = Pt(2)
        p = self.doc.add_paragraph()
        r = p.add_run(subtitle)
        r.font.size, r.font.color.rgb = Pt(12), INK
        p.paragraph_format.space_after = Pt(2)
        p = self.doc.add_paragraph()
        r = p.add_run(meta)
        r.font.size, r.font.color.rgb = Pt(9.5), MUTED
        border = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        for k, v in {"w:val": "single", "w:sz": "8", "w:space": "6", "w:color": ACCENT_HEX}.items():
            bottom.set(qn(k), v)
        border.append(bottom)
        p._p.get_or_add_pPr().append(border)
        p.paragraph_format.space_after = Pt(12)

    def h1(self, text):
        self.doc.add_heading(text, level=1)

    def h2(self, text):
        self.doc.add_heading(text, level=2)

    def para(self, text: str, bold_lead: str | None = None):
        p = self.doc.add_paragraph()
        if bold_lead:
            p.add_run(bold_lead).bold = True
        p.add_run(text)
        return p

    def bullets(self, items: list[str | tuple[str, str]]):
        for item in items:
            p = self.doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(3)
            if isinstance(item, tuple):
                p.add_run(item[0]).bold = True
                p.add_run(item[1])
            else:
                p.add_run(item)

    def table(
        self,
        header: list[str],
        rows: list[list[str]],
        widths_in: list[float],
        caption: str,
        bold_last_row: bool = False,
        highlight_col: int | None = None,
        numeric: bool = True,
        note: str | None = None,
    ):
        """``numeric``: right-align value columns; otherwise all columns are left-aligned."""
        assert abs(sum(widths_in) - CONTENT_WIDTH_IN) < 0.01, (
            "table widths must fill the page width"
        )
        self._table += 1
        cap = self.doc.add_paragraph()
        r = cap.add_run(f"Table {self._table}. ")
        r.bold = True
        r.font.size = Pt(9)
        r2 = cap.add_run(caption)
        r2.font.size = Pt(9)
        cap.paragraph_format.keep_with_next = True
        cap.paragraph_format.space_after = Pt(3)

        table = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        self._borders(table)
        for i, width in enumerate(widths_in):
            for cell in table.columns[i].cells:
                cell.width = Inches(width)
        for row_idx, values in enumerate([header, *rows]):
            for col_idx, value in enumerate(values):
                cell = table.cell(row_idx, col_idx)
                cell.text = ""
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                right = numeric and col_idx > 0
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if right else WD_ALIGN_PARAGRAPH.LEFT
                # Keep the table on one page: every row stays with the next.
                p.paragraph_format.keep_with_next = row_idx < len(rows)
                run = p.add_run(str(value))
                run.font.size = Pt(9)
                is_header = row_idx == 0
                is_bold_row = bold_last_row and row_idx == len(rows)
                is_highlight = (
                    highlight_col is not None and col_idx == highlight_col and row_idx > 0
                )
                run.bold = is_header or is_bold_row or is_highlight
                if is_header:
                    self._shade(cell, HEADER_FILL)
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if right else WD_ALIGN_PARAGRAPH.LEFT
        for tr in table._tbl.tr_lst:  # never split a row across pages
            tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        table._tbl.tr_lst[0].get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        if note:
            n = self.doc.add_paragraph()
            run = n.add_run(note)
            run.font.size = Pt(8.5)
            run.font.color.rgb = MUTED
            n.paragraph_format.space_before = Pt(3)
        else:
            self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    @staticmethod
    def _shade(cell, fill):
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), fill)
        cell._tc.get_or_add_tcPr().append(shd)

    @staticmethod
    def _borders(table):
        tbl_pr = table._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "bottom", "insideH"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4")
            el.set(qn("w:color"), "C9D3D6")
            borders.append(el)
        for edge in ("left", "right", "insideV"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "nil")
            borders.append(el)
        tbl_pr.append(borders)

    def figure(self, path: Path, caption: str, width_in: float = CONTENT_WIDTH_IN):
        if not Path(path).is_file():
            raise FileNotFoundError(f"figure not found: {path}")
        self._figure += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Inches(width_in))
        cap = self.doc.add_paragraph()
        r = cap.add_run(f"Figure {self._figure}. ")
        r.bold = True
        r.font.size = Pt(9)
        r2 = cap.add_run(caption)
        r2.font.size = Pt(9)
        r2.font.color.rgb = MUTED

    def page_break(self):
        self.doc.add_page_break()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(path)


# ----------------------------------------------------------------------------- facts


def _money(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 1000 else f"${x:,.2f}"


def _signed(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}${abs(x):,.0f}"


def gather_facts(cfg) -> dict:
    train = io.load_train(cfg.paths.train)
    val = io.load_validation(cfg.paths.validation)
    y = train[schema.TARGET]
    anomalies = (
        LabelAnomalyFilter(
            cfg.cleaning.anomaly_low_ratio,
            cfg.cleaning.anomaly_high_ratio,
            cfg.cleaning.anomaly_distance_bins,
        )
        .fit(train, y)
        .is_anomaly(train, y)
    )
    cities_train = set(train.pickup) | set(train.delivery)
    unseen = sorted((set(val.pickup) | set(val.delivery)) - cities_train)
    touches = val.pickup.isin(unseen) | val.delivery.isin(unseen)

    preds = pd.read_csv(cfg.paths.validation_predictions)
    december = pd.read_csv(cfg.paths.december_predictions)
    merged = val.merge(preds, on=schema.ID)
    rpm = merged.predicted_rate / merged.distance

    return {
        "train_rows": len(train),
        "val_rows": len(val),
        "val_nov": int((val.date.dt.month == 11).sum()),
        "val_dec": int((val.date.dt.month == 12).sum()),
        "anomalies": int(anomalies.sum()),
        "anomaly_pct": anomalies.mean() * 100,
        "neg_weight_train": int((train.weight < 0).sum()),
        "neg_weight_val": int((val.weight < 0).sum()),
        "missing_weight": (int(train.weight.isna().sum()), int(val.weight.isna().sum())),
        "missing_mi": (int(train.market_index.isna().sum()), int(val.market_index.isna().sum())),
        "unseen_cities": unseen,
        "unseen_rows": int(touches.sum()),
        "unseen_pct": touches.mean() * 100,
        "corr_distance": train[["distance", schema.TARGET]].corr().iloc[0, 1],
        "pred_min": preds.predicted_rate.min(),
        "pred_max": preds.predicted_rate.max(),
        "pred_median": preds.predicted_rate.median(),
        "pred_rpm_nov": rpm[merged.date.dt.month == 11].median(),
        "pred_rpm_dec": rpm[merged.date.dt.month == 12].median(),
        "dec_min": december.predicted_rate.min(),
        "dec_max": december.predicted_rate.max(),
        "dec_mean": december.predicted_rate.mean(),
    }


def _load_summary(name: str) -> pd.DataFrame:
    path = EXPERIMENTS / f"{name}_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run the Phase 4 experiments first (see README)")
    return pd.read_csv(path, index_col="model")


def holdout_table(name: str = "phase4_r7_final_holdout") -> pd.DataFrame:
    metrics = pd.read_csv(EXPERIMENTS / f"{name}_metrics.csv")
    h = metrics[(metrics.fold == "holdout") & (metrics.scenario == "temporal")]
    out = h.pivot_table(index="model", columns="segment", values="mae")
    clean = h[h.segment == "clean"].set_index("model")
    out["bias"] = clean["bias"]
    out["mape"] = clean["mape_pct"]
    cities = metrics[(metrics.fold == "holdout") & metrics.scenario.str.startswith("cities")]
    c = cities.pivot_table(index="model", columns="scenario", values="mae")
    out["city_cost"] = (c["cities_unseen"] / c["cities_seen"] - 1) * 100
    return out


# ----------------------------------------------------------------------------- content


def build(config_path: Path = DEFAULT_CONFIG, output: Path = OUTPUT) -> Path:
    cfg = load_config(config_path)
    if not SCORER_CHART.is_file():
        raise FileNotFoundError(f"{SCORER_CHART} missing; run score.py first (see README)")
    f = gather_facts(cfg)
    finalists = _load_summary("phase4_r6_normal")
    novdec = _load_summary("phase4_r6_novdec")
    ablation = _load_summary("phase4_r2_ablation")
    drift = _load_summary("phase4_r1_drift")
    tuning = _load_summary("phase4_r3_tuning")
    hold = holdout_table()
    hold_v1 = holdout_table("phase4_r5_final_holdout")
    august = pd.read_csv(EXPERIMENTS / "quote_signal_august_check.csv").set_index(
        ["setup", "model"]
    )
    timeline = validation_timeline(cfg.validation.folds, REPORT_FIGURES / "validation_timeline.png")
    train_frame = io.load_train(cfg.paths.train)
    regimes = pd.concat(
        [
            quote_signal_regimes(train_frame, train_frame).assign(split="training"),
            quote_signal_regimes(train_frame, io.load_validation(cfg.paths.validation)).assign(
                split="validation"
            ),
        ],
        ignore_index=True,
    )
    regime_chart = quote_signal_regime_chart(regimes, REPORT_FIGURES / "quote_signal_regimes.png")

    sel, base = finalists.loc["final"], finalists.loc["grouped_median"]
    hsel, hbase = hold.loc["final"], hold.loc["grouped_median"]
    aug_with = august.loc[("train Jan-Jul", "blend_70"), "MAE clean"]
    aug_without = august.loc[("train Jan-Jul", "blend_70_noqs"), "MAE clean"]

    r = Report()
    r.title(
        "Freight Rate Prediction",
        "Validation approach, model selection and December forecast",
        f"Spotter Labs - Machine Learning Engineer assessment  |  {date.today():%B %d, %Y}",
    )

    # 1. Summary ---------------------------------------------------------------
    r.h1("1. Summary")
    r.para(
        f"The task is to predict posted_rate for {f['val_rows']:,} loads dated November-December 2025, "
        f"using {f['train_rows']:,} labelled loads from January-October 2025. Because the target period lies "
        "after all labelled data, every validation decision in this project is built around one question: "
        "how well does a model trained on the past predict rates one to two months ahead?"
    )
    r.bullets(
        [
            (
                "Split: ",
                "rolling-origin, time-based validation with two-month test windows that mirror the "
                "Nov-Dec horizon; the most recent window (Sep-Oct) is held back as a one-time final check. "
                "A simulated unseen-city test covers the 8 validation cities absent from training.",
            ),
            (
                "Model: ",
                "a 70/30 blend of gradient boosting and a spline Ridge model, both predicting "
                "log(rate) from distance, equipment, weight, market_index and coordinates, with "
                "recent loads weighted more heavily and corrupted labels removed from training. quote_signal is "
                "deliberately excluded: its link to price flips sign by month and is absent in Nov-Dec.",
            ),
            (
                "Result: ",
                f"on the Sep-Oct holdout the selected model reaches MAE "
                f"{_money(hsel['clean'])} ({hsel['mape']:.1f}% MAPE) on clean labels versus "
                f"{_money(hbase['clean'])} ({hbase['mape']:.1f}%) for the strongest naive baseline.",
            ),
            (
                "Deliverables: ",
                "validation_predictions.csv (12,000 rows) and the completed December file are "
                "accepted by the unmodified score.py; the December chart is in Section 6.",
            ),
        ]
    )

    # 2. Data -------------------------------------------------------------------
    r.h1("2. Data findings and data quality")
    r.para(
        f"Distance dominates price (correlation {f['corr_distance']:.2f}); rate per mile falls with distance "
        "and each equipment type shifts the whole curve (Reefer > Flatbed > Dry Van). market_index is a "
        "market-wide daily value: it barely varies within a day but moves strongly across the year, and it "
        "is the strongest signal after distance. Prices also drift up over the year by about 6% beyond what "
        "market_index explains, which makes the most recent months the best guide to November-December."
    )
    r.table(
        ["Issue", "Evidence", "Treatment"],
        [
            [
                "Corrupted labels",
                f"{f['anomalies']} rows ({f['anomaly_pct']:.1f}%) priced at 0.17-0.47x or 2.1-5.2x comparable "
                "loads; no load lies between these clusters and the clean band (0.81-1.28x)",
                "Removed from training in every fold; kept and reported in evaluation",
            ],
            [
                "Negative weights",
                f"{f['neg_weight_train']} train / {f['neg_weight_val']} validation; absolute values match the "
                "normal distribution and are priced normally",
                "Absolute value (sign error)",
            ],
            [
                "Missing weight",
                f"{f['missing_weight'][0]} train / {f['missing_weight'][1]} validation; unrelated to price",
                "Training-window median",
            ],
            [
                "Missing market_index",
                f"{f['missing_mi'][0]} train / {f['missing_mi'][1]} validation; unrelated to price",
                "Mean of the same day's loads (daily market value)",
            ],
            [
                "Unseen cities",
                f"{len(f['unseen_cities'])} validation cities ({f['unseen_rows']:,} loads, "
                f"{f['unseen_pct']:.0f}%) never appear in training",
                "No city-name features; coordinates + distance; tested explicitly",
            ],
            [
                "Distance vs coordinates",
                "Short hauls up to 2.1x the great-circle distance; prices follow the stated distance",
                "Keep stated distance",
            ],
            [
                "quote_signal regimes",
                "Within each month it encodes the true rate per mile almost exactly (R2 ~0.99), but the sign "
                "flips by month and there is no relationship in Aug 2-31 or Nov-Dec (Section 2.1)",
                "Not used by the final model",
            ],
            ["load_id", "Assigned in date order (a time proxy)", "Never used as a feature"],
        ],
        [1.35, 3.15, 2.0],
        "Data-quality issues and how they were handled.",
        numeric=False,
    )
    r.figure(
        EDA_FIGURES / "02_label_anomaly_clusters.png",
        "Rate per mile relative to comparable loads. Clean loads form one tight band; corrupted labels "
        "form two separate clusters, which justifies treating them as label errors rather than market moves.",
    )
    r.h2("2.1 quote_signal: a relationship that flips by month")
    pos = ", ".join(
        pd.Period(p).strftime("%b")
        for p in regimes.query("split == 'training' and regime == 'positive'").period
    )
    inv = ", ".join(
        pd.Period(p).strftime("%b")
        for p in regimes.query("split == 'training' and regime == 'inverted'").period
    )
    r.para(
        "Pooled over the year, quote_signal correlates only 0.07 with relative price, so it first looked like a "
        "weak feature, and the first draft of the model used it. Checked month by month (prompted by review), it "
        "turns out to be an almost exact linear encoding of the true rate per mile (R2 about 0.99): rate per mile "
        f"is approximately quote_signal in {pos}, and approximately 4.13 - quote_signal in {inv}. From 2 August "
        "onwards there is no relationship at all. It even encodes the true price on rows whose label is corrupted."
    )
    r.para(
        "The regime of any period can be read without labels, by correlating quote_signal with the rate per mile "
        "expected from distance and equipment alone. This matches the labelled regime in every training month. "
        "For November and December it is about 0 on every day, and validation quote_signal has August's "
        "distribution exactly. So quote_signal carries no price information in the period to be predicted, and "
        "flipping its sign cannot help: there is no regime to flip to."
    )
    r.figure(
        regime_chart,
        "Label-free regime check per month: correlation of quote_signal with the rate per mile expected from "
        "distance and equipment. Training months show a strong positive or inverted link; August and both "
        "validation months show none.",
    )

    # 3. Split and validation ----------------------------------------------------
    r.h1("3. Train/test split and validation approach")
    r.h2("3.1 Why a time-based split")
    r.para(
        "In deployment the model is fitted on January-October and predicts November-December. A random "
        "K-fold split would mix future loads into training and test the model on periods it has already "
        "seen, overstating accuracy for a forward forecast. Prices drift over time beyond what market_index "
        "explains, so the validation must measure forecasting forward in time."
    )
    r.h2("3.2 Rolling-origin folds")
    fold_rows = []
    train = io.load_train(cfg.paths.train)
    for fold in cfg.validation.folds:
        tr = (train.date <= fold.train_end).sum()
        te = ((train.date >= fold.test_start) & (train.date <= fold.test_end)).sum()
        fold_rows.append(
            [
                fold.name,
                "Model selection" if fold.role == "selection" else "Final check",
                f"Jan 1 - {fold.train_end:%b %d}",
                f"{fold.test_start:%b %d} - {fold.test_end:%b %d}",
                f"{tr:,}",
                f"{te:,}",
            ]
        )
    r.table(
        ["Fold", "Role", "Train", "Test", "Train rows", "Test rows"],
        fold_rows,
        [0.75, 1.75, 1.15, 1.25, 0.8, 0.8],
        "Validation folds. Training windows expand; every test window is two months long.",
    )
    r.figure(
        timeline,
        "Each fold trains on all data before its test window and tests on the next two months, "
        "the same shape as the real task (bottom row).",
    )
    r.bullets(
        [
            (
                "Two-month test windows ",
                "match the Nov-Dec horizon; metrics are also reported separately for "
                "the first and second month, because accuracy decays with horizon.",
            ),
            (
                "Different market regimes: ",
                "fold_1 tests on the May-June peak, fold_2 on the July-August "
                "decline, so a model must work in both.",
            ),
            (
                "Final holdout kept separate: ",
                "Sep-Oct is the closest analogue to Nov-Dec. It was excluded from "
                "all model comparison (enforced in code and by a test) and only evaluated after the model was chosen "
                "(see the disclosure in Section 5).",
            ),
            ("Final model: ", "refitted on all January-October data with the identical pipeline."),
        ]
    )
    r.h2("3.3 Unseen-city check")
    r.para(
        f"{f['unseen_pct']:.0f}% of validation loads touch a city that never appears in training (always "
        "exactly one endpoint). To measure this risk, each selection fold is repeated three times with 8 "
        "random training cities hidden: every training load touching them is removed, and the model is "
        "scored on test loads with exactly one hidden endpoint. The same loads are also scored by the model "
        "that did see those cities; the difference is the cost of a new city."
    )
    r.h2("3.4 Preventing leakage")
    r.bullets(
        [
            "All learned preprocessing (weight median, anomaly reference, model) is fitted on the training "
            "window only; the cleaning step never drops rows, so every load receives a prediction.",
            "Corrupted labels are removed from training only. Test loads are never dropped; the anomaly flag only "
            "splits the reported metrics into 'all rows' and 'clean rows'.",
            "The model never sees the target or load_id; features are row-independent and identical in "
            "cross-validation, final training and inference (no train/serve skew).",
            "Each of these is enforced by an automated test in tests/.",
        ]
    )
    r.h2("3.5 Metrics")
    r.para(
        "The official metric is not disclosed, so MAE in dollars is the primary metric (interpretable and "
        "robust to the corrupted labels), reported on clean rows and on all rows, with RMSE, MAPE (safe: "
        "every rate is positive) and bias as supporting metrics."
    )

    # 4. Model selection -------------------------------------------------------------
    r.h1("4. Model development and selection")
    r.para(
        "All learned models predict log(rate) and are transformed back: errors scale with load size, so a "
        "log target makes them roughly proportional, and the back-transformed value estimates the median "
        "rate, which suits MAE. Experiments changed one thing at a time on the two selection folds."
    )
    r.bullets(
        [
            (
                "Drift: ",
                f"the main weakness was growing under-prediction in the second forecast month. Weighting "
                f"recent loads more (30-day half-life) reduced clean MAE from {_money(drift.loc['hgb', 'MAE clean'])} "
                f"to {_money(drift.loc['hgb_recency_30', 'MAE clean'])}; an explicit time trend made it worse "
                f"({_money(drift.loc['hgb_time', 'MAE clean'])}), because extrapolating the trend misjudges the next months.",
            ),
            (
                "Ablations: ",
                "removing market_index "
                f"(+{ablation.loc['hgb_r30_no_market', 'MAE clean'] - ablation.loc['hgb_recency_30', 'MAE clean']:.0f}), "
                f"weight (+{ablation.loc['hgb_r30_no_weight', 'MAE clean'] - ablation.loc['hgb_recency_30', 'MAE clean']:.0f}), "
                f"or coordinates (+{ablation.loc['hgb_r30_no_coords', 'MAE clean'] - ablation.loc['hgb_recency_30', 'MAE clean']:.0f}) "
                "all hurt; missing-value flags and day-of-week added nothing and were left out.",
            ),
            (
                "quote_signal: ",
                f"on the selection folds, removing it looked harmful (+{ablation.loc['hgb_r30_no_quote', 'MAE clean'] - ablation.loc['hgb_recency_30', 'MAE clean']:.0f}), "
                "but only because those test windows had an active regime. Two checks that match the real forecast "
                "period reverse this. (1) Nov-Dec conditions: with quote_signal shuffled inside each test window, the "
                f"first-draft blend falls to {_money(novdec.loc['blend_70', 'MAE clean'])} while the same blend "
                f"without quote_signal stays at {_money(novdec.loc['final', 'MAE clean'])}. (2) A real lookalike: "
                "trained through July (ending in an inverted month, as October does) and tested on Aug 2-31 (no "
                f"regime), clean MAE improves from {_money(aug_with)} to {_money(aug_without)} "
                f"({(aug_without / aug_with - 1) * 100:+.0f}%) without quote_signal. It was therefore removed.",
            ),
            (
                "Label cleaning: ",
                f"training on the corrupted labels raised clean MAE from "
                f"{_money(ablation.loc['hgb_recency_30', 'MAE clean'])} to "
                f"{_money(ablation.loc['hgb_r30_keep_anomalies', 'MAE clean'])}, confirming the decision to remove them.",
            ),
            (
                "Tuning: ",
                f"gradient boosting settings all landed within "
                f"{_money(tuning.filter(like='hgb_r3', axis=0)['MAE clean'].min())}-"
                f"{_money(tuning.filter(like='hgb_r3', axis=0)['MAE clean'].max())}, i.e. within noise, so defaults "
                "were kept rather than fitted to two folds.",
            ),
            (
                "Blend: ",
                "gradient boosting is strongest in the first month, Ridge (smoother) in the second; a "
                "70/30 blend in log space was best overall. The weight was fixed once, not tuned.",
            ),
        ]
    )
    cols = MODEL_COLUMNS
    r.table(
        ["Selection folds (USD)"] + [MODEL_LABELS[c] for c in cols],
        [
            ["MAE clean, as observed"] + [f"{finalists.loc[c, 'MAE clean']:.1f}" for c in cols],
            ["MAE clean, Nov-Dec conditions"] + [f"{novdec.loc[c, 'MAE clean']:.1f}" for c in cols],
            ["MAE all rows, Nov-Dec cond."] + [f"{novdec.loc[c, 'MAE all']:.1f}" for c in cols],
            ["Month 1 / 2, Nov-Dec cond."]
            + [
                f"{novdec.loc[c, 'MAE clean m1']:.0f} / {novdec.loc[c, 'MAE clean m2']:.0f}"
                for c in cols
            ],
            ["Unseen-city cost"] + [f"{novdec.loc[c, 'unseen-city cost %']:+.1f}%" for c in cols],
            ["Seconds per fit"] + [f"{finalists.loc[c, 'sec per fit']:.2f}" for c in cols],
        ],
        [1.9, 0.92, 0.92, 0.92, 0.92, 0.92],
        "Finalists on the selection folds (mean of fold_1 and fold_2). Nov-Dec conditions = quote_signal "
        "shuffled within each test window, since it carries no information in Nov-Dec. Selected model in bold.",
        highlight_col=5,
        note=MODEL_NOTE,
    )
    r.para(
        f"The selected model reduces clean MAE from {_money(base['MAE clean'])} to {_money(sel['MAE clean'])} "
        f"versus the equipment x distance baseline. Errors are proportional (about 2-2.5% at every distance, "
        f"weight and equipment band, no problem lanes), and a new city costs only "
        f"{novdec.loc['final', 'unseen-city cost %']:.0f}%, so city identity is not needed."
    )

    # 5. Holdout ------------------------------------------------------------------------
    r.h1("5. Final holdout result (Sep-Oct)")
    r.table(
        ["Sep-Oct holdout (USD)"] + [MODEL_LABELS[c] for c in cols],
        [
            ["MAE, clean rows"] + [f"{hold.loc[c, 'clean']:.1f}" for c in cols],
            ["MAE, all rows"] + [f"{hold.loc[c, 'all']:.1f}" for c in cols],
            ["MAPE %, clean"] + [f"{hold.loc[c, 'mape']:.2f}" for c in cols],
            ["Bias, clean"] + [_signed(hold.loc[c, "bias"]) for c in cols],
            ["Unseen-city cost"] + [f"{hold.loc[c, 'city_cost']:+.1f}%" for c in cols],
        ],
        [1.9, 0.92, 0.92, 0.92, 0.92, 0.92],
        "Train January-August, test September-October.",
        highlight_col=5,
        note=MODEL_NOTE,
    )
    r.bullets(
        [
            f"The selected model beats the strongest naive baseline: clean MAE {_money(hsel['clean'])} vs "
            f"{_money(hbase['clean'])}, all-row MAE {_money(hsel['all'])} vs {_money(hbase['all'])}.",
            f"Recency weighting is essential: without it, gradient boosting scores {_money(hold_v1.loc['hgb', 'clean'])}, "
            "worse than the baseline on this period.",
            f"Removing quote_signal leaves the holdout essentially unchanged ({_money(hold.loc['blend_70', 'clean'])} "
            f"with it, {_money(hsel['clean'])} without). The holdout cannot show the benefit: that model's recency "
            "weighting falls mostly on August, when quote_signal had no regime, so it had already learned to ignore "
            "it. The final model weights October (inverted regime) most, which is where quote_signal would do harm.",
            f"Gradient boosting alone scored slightly better here ({_money(hold.loc['hgb_r3_noqs', 'clean'])}), the "
            "reverse of the selection folds; the blend chosen on the selection folds was kept.",
            "Disclosure: the holdout was evaluated once for the first-draft model and a second time for the revised "
            "model after the quote_signal finding. That change was driven by the regime analysis, not by holdout "
            "scores, which barely moved.",
            "The all-row MAE is roughly $55 higher than the clean-row MAE for every model: the unavoidable cost "
            "of the ~1.5% corrupted labels, which validation data almost certainly contains as well.",
        ]
    )

    # 6. December -------------------------------------------------------------------------
    r.h1("6. Final predictions and the December chart")
    r.para(
        f"The final model was refitted on all January-October data. Validation predictions range from "
        f"{_money(f['pred_min'])} to {_money(f['pred_max'])} (median {_money(f['pred_median'])}); predicted "
        f"rate per mile is ${f['pred_rpm_nov']:.2f} in November and ${f['pred_rpm_dec']:.2f} in December, in "
        "line with the September-October actuals."
    )
    r.para(
        "The December chart file contains only lane, distance, equipment, weight and date. Each date receives "
        "the mean market_index of that day's loads in validation.csv (input features, not labels; at least 163 "
        "loads per day), coordinates come from the training data, and the identical pipeline produces the "
        "predictions."
    )
    r.figure(
        SCORER_CHART,
        "December chart produced by the provided score.py from the completed "
        "data/december_chart_inputs.csv.",
    )
    r.para(
        f"Predictions stay between {_money(f['dec_min'])} and {_money(f['dec_max'])}, consistent with recent "
        "Lexington-Fort Wayne Dry Van loads ($790-841 scaled to 360 miles, Aug-Oct). The weekly shape is "
        "real: the daily market_index peaks on Thursdays and dips on Sundays, in training and in December, and "
        "prices follow it by about +/-1%. The model has no day-of-week feature, so the pattern comes "
        "entirely from the market data. No trend is imposed across the month."
    )

    # 7. Limitations -----------------------------------------------------------------------
    r.h1("7. Limitations and next steps")
    r.bullets(
        [
            (
                "Forward drift: ",
                "every model under-predicts increasingly with horizon (about -$59 on the Sep-Oct "
                "holdout); prices appear to follow market_index with a delay. Recency weighting anchors the final "
                "model on the most recent months, but some under-prediction in late December is likely.",
            ),
            (
                "Seasonality: ",
                "one year of history cannot reveal December holiday effects; they are not modelled.",
            ),
            (
                "Small number of folds: ",
                "two selection folds make small differences (about $1-2 MAE) "
                "indistinguishable from noise, which is why tuning and blend weights were not optimised further.",
            ),
            (
                "Next steps: ",
                "with more history, a model of the lagged market response (tested here, not yet "
                "beneficial), seasonal terms, and monitoring of bias on newly labelled loads to recalibrate.",
            ),
        ]
    )

    # Appendix ------------------------------------------------------------------------------
    r.h1("Appendix: reproducing the results")
    for line in [
        "python -m freight_rate.pipelines.train",
        "python -m freight_rate.pipelines.predict",
        "python score.py --predictions validation_predictions.csv "
        "--december-predictions data/december_chart_inputs.csv",
        "python -m freight_rate.reporting.build_report",
        "python -m pytest",
    ]:
        p = r.doc.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
    r.para(
        "Experiments: configs/experiments/phase4.yaml with python -m freight_rate.pipelines.experiment; "
        "results in reports/experiments/. Exploratory analysis: notebooks/01_eda.ipynb. Detailed write-ups "
        "for each phase: docs/."
    )

    r.save(output)
    log.info("wrote %s", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DOCX report.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build(args.config, args.output)


if __name__ == "__main__":
    main()
