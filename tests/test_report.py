from __future__ import annotations

import pytest
from docx import Document

from freight_rate.reporting import build_report as br


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    if (
        not br.SCORER_CHART.is_file()
        or not (br.EXPERIMENTS / "phase4_r4_finalists_summary.csv").is_file()
    ):
        pytest.skip("run the pipeline, score.py and Phase 4 experiments first")
    return Document(br.build(output=tmp_path_factory.mktemp("report") / "report.docx"))


def test_report_has_required_sections(report):
    headings = [p.text for p in report.paragraphs if p.style.name.startswith("Heading")]
    assert "3. Train/test split and validation approach" in headings
    assert "6. Final predictions and the December chart" in headings


def test_report_embeds_figures_including_scorer_chart(report):
    images = report.inline_shapes
    # anomaly clusters, quote_signal regimes, validation timeline, December chart
    assert len(images) == 4
    captions = [p.text for p in report.paragraphs if p.text.startswith("Figure")]
    assert any("score.py" in c for c in captions)


def test_report_numbers_match_experiment_results(report):
    text = "\n".join(p.text for p in report.paragraphs)
    hold = br.holdout_table()
    assert f"${hold.loc['final', 'clean']:,.2f}" in text


def test_report_covers_quote_signal_finding(report):
    cells = [c.text for t in report.tables for row in t.rows for c in row.cells]
    assert "quote_signal regimes" in cells
    headings = [p.text for p in report.paragraphs if p.style.name.startswith("Heading")]
    assert any("quote_signal" in h for h in headings)
