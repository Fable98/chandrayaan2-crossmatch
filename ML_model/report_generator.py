"""ML_model/report_generator.py — Automated ISRO PDF Report Generator.

Compiles pipeline outputs (metadata, metrics, phase diagnostics, spatial
distribution, match visualization, bundle adjustment) into a professional
multi-page A4 PDF using reportlab + matplotlib (Agg backend, headless-safe).

Design guardrails:
  * ``matplotlib.use('Agg')`` is the FIRST matplotlib-related line.
  * NEVER raises from :meth:`ISROReportGenerator.generate_report` on bad /
    missing data; every section is wrapped in try/except and degrades to
    "N/A" or a skipped section.
  * Temporary PNG files are cleaned up after the PDF is built.
"""

import matplotlib

matplotlib.use("Agg")

import datetime
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger("ML_model.report_generator")

SENSOR_GSD = {
    "OHRC": 0.3,
    "TMC": 5.0,
    "TMC-2": 5.0,
    "IIRS": 80.0,
}

_PHASE_ORDER = ["CFOG", "Crater", "Kornia", "SubPixel", "Distribution"]


def _safe(v: Any) -> str:
    """Render a value for the PDF, showing 'N/A' for None/missing/NaN/inf."""
    if v is None:
        return "N/A"
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "N/A"
    if isinstance(v, str) and not v.strip():
        return "N/A"
    return str(v)


def _get(metrics: Optional[dict], *keys: str, default: Any = None) -> Any:
    """Fetch first present key from a dict (supports alias key names)."""
    if not isinstance(metrics, dict):
        return default
    for k in keys:
        if k in metrics and metrics[k] is not None:
            return metrics[k]
    return default


def _rmse_color(rmse: Any):
    """Green if < 0.5px, yellow if 0.5-1.0px, red if > 1.0px."""
    try:
        r = float(rmse)
    except (TypeError, ValueError):
        return colors.black
    if np.isnan(r) or np.isinf(r):
        return colors.black
    if r < 0.5:
        return colors.HexColor("#1B7A2B")
    if r <= 1.0:
        return colors.HexColor("#9A7B00")
    return colors.HexColor("#B00020")


def _phase_status(phases: Any, name: str) -> str:
    """Normalize a phase entry to 'success' / 'partial' / 'failed' / 'skipped'."""
    if not isinstance(phases, dict):
        return "skipped"
    # MasterPipeline style: {"phases_executed": [...], "phases_failed": [...]}
    if "phases_executed" in phases or "phases_failed" in phases:
        executed = phases.get("phases_executed", []) or []
        failed = phases.get("phases_failed", []) or []
        if name in failed:
            return "failed"
        if name in executed:
            return "success"
        return "skipped"
    v = phases.get(name, phases.get(name.lower(), phases.get(name.upper())))
    if v is None:
        return "skipped"
    if isinstance(v, dict):
        s = str(v.get("status", v.get("state", "unknown"))).lower()
    else:
        s = str(v).lower()
    if s in ("success", "ok", "passed", "done", "high", "completed"):
        return "success"
    if s in ("partial", "warning", "warn", "medium", "low"):
        return "partial"
    if s in ("failed", "fail", "error", "crashed", "no_inliers", "n/a"):
        return "failed"
    if s in ("skipped", "skip", "not_run", "none"):
        return "skipped"
    return "partial"


def _phase_matches(phases: Any, name: str) -> str:
    if not isinstance(phases, dict):
        return "N/A"
    v = phases.get(name, phases.get(name.lower(), phases.get(name.upper())))
    if isinstance(v, dict):
        for k in ("matches", "match_count", "inliers", "inlier_count", "count", "num_matches"):
            if v.get(k) is not None:
                return str(v[k])
    if isinstance(v, (int, float)):
        return str(int(v))
    return "N/A"


def _phase_confidence(phases: Any, name: str) -> str:
    if not isinstance(phases, dict):
        return "N/A"
    v = phases.get(name, phases.get(name.lower(), phases.get(name.upper())))
    if isinstance(v, dict):
        for k in ("confidence", "confidence_tier", "quality_tier", "tier"):
            if v.get(k) is not None:
                return str(v[k])
    status = _phase_status(phases, name)
    return {"success": "High", "partial": "Medium"}.get(status, "N/A")


class ISROReportGenerator:
    """Build a multi-page ISRO-style PDF report from pipeline outputs."""

    def __init__(self, output_dir: str = "reports/") -> None:
        self.output_dir = str(output_dir)
        self.logger = logging.getLogger("ML_model.report_generator")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as exc:
            self.logger.warning("Could not create output dir %s: %s", self.output_dir, exc)
            raise
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        matplotlib.rcParams["font.size"] = 9
        matplotlib.rcParams["axes.grid"] = True
        matplotlib.rcParams["grid.alpha"] = 0.3

    # ------------------------------------------------------------------
    def _styles(self) -> Dict[str, ParagraphStyle]:
        base = getSampleStyleSheet()
        styles = {
            "title": ParagraphStyle(
                "ISROTitle",
                parent=base["Title"],
                fontSize=24,
                leading=28,
                alignment=1,
                fontName="Helvetica-Bold",
            ),
            "subtitle": ParagraphStyle(
                "ISROSubtitle",
                parent=base["Normal"],
                fontSize=12,
                leading=15,
                alignment=1,
                textColor=colors.HexColor("#333333"),
            ),
            "heading": ParagraphStyle(
                "ISROHeading",
                parent=base["Heading1"],
                fontSize=14,
                leading=17,
                fontName="Helvetica-Bold",
                spaceBefore=14,
                spaceAfter=8,
            ),
            "body": ParagraphStyle(
                "ISROBody", parent=base["Normal"], fontSize=9, leading=12
            ),
            "cell": ParagraphStyle(
                "ISROCell", parent=base["Normal"], fontSize=8, leading=10
            ),
        }
        return styles

    def _section_table(self, rows, col_widths=None) -> Table:
        tbl = Table(rows, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D66")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F9")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return tbl

    # ------------------------------------------------------------------
    def _create_header_page(self, story, metadata: dict) -> None:
        styles = self._styles()
        team = "Team"
        if isinstance(metadata, dict):
            team = metadata.get("team_name", metadata.get("team", "Team")) or "Team"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Spacer(1, 1.2 * inch))
        story.append(Paragraph("Chandrayaan-2 Multi-Modal Image Registration Report", styles["title"]))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("SIH 2026 | Problem Statement 26166 | ISRO", styles["subtitle"]))
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph(f"Generated: {now}", styles["subtitle"]))
        story.append(Paragraph(f"Team: {_safe(team)}", styles["subtitle"]))
        story.append(Spacer(1, 0.3 * inch))
        info = [
            [Paragraph("<b>Field</b>", styles["cell"]), Paragraph("<b>Value</b>", styles["cell"])],
            [Paragraph("Mission", styles["cell"]), Paragraph("Chandrayaan-2", styles["cell"])],
            [Paragraph("Report Type", styles["cell"]), Paragraph("Automated Registration Summary", styles["cell"])],
            [Paragraph("Team", styles["cell"]), Paragraph(_safe(team), styles["cell"])],
            [Paragraph("Generated", styles["cell"]), Paragraph(now, styles["cell"])],
        ]
        story.append(self._section_table(info, col_widths=[2.2 * inch, 4.3 * inch]))

    # ------------------------------------------------------------------
    def _create_metadata_section(self, story, metadata: dict) -> None:
        styles = self._styles()
        story.append(Paragraph("1. Sensor &amp; Illumination Metadata", styles["heading"]))
        md = metadata if isinstance(metadata, dict) else {}
        sensor = _get(md, "sensor", "sensor_name", default=None)
        gsd = _get(md, "gsd", "gsd_m", "gsd_x", default=None)
        if gsd is None and sensor is not None:
            gsd = SENSOR_GSD.get(str(sensor).upper(), None)
        if gsd is not None:
            try:
                gsd = f"{float(gsd):.2f} m/px"
            except (TypeError, ValueError):
                gsd = str(gsd)
        dims = _get(md, "dimensions", "image_shape", "image_dimensions", default=None)
        if isinstance(dims, (list, tuple)) and len(dims) >= 2:
            dims = f"{dims[1]} x {dims[0]}" if len(dims) == 2 else " x ".join(str(d) for d in dims)
        rows = [
            [Paragraph("<b>Parameter</b>", styles["cell"]), Paragraph("<b>Value</b>", styles["cell"])],
            [Paragraph("Sensor (OHRC / TMC / IIRS)", styles["cell"]), Paragraph(_safe(sensor), styles["cell"])],
            [Paragraph("GSD (Ground Sample Distance)", styles["cell"]), Paragraph(_safe(gsd), styles["cell"])],
            [Paragraph("Sun Azimuth (deg)", styles["cell"]), Paragraph(_safe(_get(md, "sun_az", "sun_azimuth", default=None)), styles["cell"])],
            [Paragraph("Sun Elevation (deg)", styles["cell"]), Paragraph(_safe(_get(md, "sun_el", "sun_elevation", default=None)), styles["cell"])],
            [Paragraph("Capture Timestamp", styles["cell"]), Paragraph(_safe(_get(md, "timestamp", "capture_time", "acquisition_time", default=None)), styles["cell"])],
            [Paragraph("Image Dimensions (W x H)", styles["cell"]), Paragraph(_safe(dims), styles["cell"])],
        ]
        story.append(self._section_table(rows, col_widths=[2.6 * inch, 3.9 * inch]))

    # ------------------------------------------------------------------
    def _create_metrics_section(self, story, metrics: dict) -> None:
        styles = self._styles()
        story.append(Paragraph("2. Registration Metrics", styles["heading"]))
        if not isinstance(metrics, dict):
            story.append(Paragraph("No metrics available.", styles["body"]))
            return
        raw = _get(metrics, "match_count", "total_matches", "raw_matches", "total_raw_matches", default=None)
        inl = _get(metrics, "inlier_count", "inliers", "final_inliers", default=None)
        ratio = _get(metrics, "inlier_ratio", default=None)
        rmse = _get(metrics, "rmse", "fit_rmse_px", "final_rmse_pixels", "final_rmse", default=None)
        ce90 = _get(metrics, "ce90", "CE90", "ce90_px", default=None)
        rmse_ba = _get(metrics, "rmse_after_ba", "rmse_bundle_adjusted", "final_rmse_after_ba",
                       "ba_rmse", "final_rmse_pixels_ba", default=None)
        improvement = _get(metrics, "improvement_pct", "improvement_percent",
                           "ba_improvement_pct", "bundle_adjustment_improvement", default=None)
        if ratio is not None:
            try:
                ratio = f"{float(ratio) * 100.0:.1f}%" if float(ratio) <= 1.0 else f"{float(ratio):.1f}%"
            except (TypeError, ValueError):
                ratio = _safe(ratio)
        rmse_txt, rmse_ba_txt = _safe(rmse), _safe(rmse_ba)
        try:
            if rmse is not None and not (isinstance(rmse, float) and (np.isnan(rmse) or np.isinf(rmse))):
                rmse_txt = f"{float(rmse):.4f} px"
        except (TypeError, ValueError):
            pass
        try:
            if rmse_ba is not None and not (isinstance(rmse_ba, float) and (np.isnan(rmse_ba) or np.isinf(rmse_ba))):
                rmse_ba_txt = f"{float(rmse_ba):.4f} px"
        except (TypeError, ValueError):
            pass
        if ce90 is not None:
            try:
                ce90 = f"{float(ce90):.4f} px"
            except (TypeError, ValueError):
                ce90 = _safe(ce90)
        if improvement is not None:
            try:
                improvement = f"{float(improvement):.1f}%"
            except (TypeError, ValueError):
                improvement = _safe(improvement)
        # Auto-derive BA improvement when both RMSEs are present.
        if (improvement is None or improvement == "N/A") and rmse is not None and rmse_ba is not None:
            try:
                r0, r1 = float(rmse), float(rmse_ba)
                if np.isfinite(r0) and np.isfinite(r1) and r0 > 0:
                    improvement = f"{(r0 - r1) / r0 * 100.0:.1f}%"
                else:
                    improvement = "N/A"
            except (TypeError, ValueError):
                improvement = "N/A"
        rmse_para = Paragraph(f'<font color="{_rmse_color(rmse).hexval()}">{rmse_txt}</font>', styles["cell"])
        rows = [
            [Paragraph("<b>Metric</b>", styles["cell"]), Paragraph("<b>Value</b>", styles["cell"])],
            [Paragraph("Total raw matches", styles["cell"]), Paragraph(_safe(raw), styles["cell"])],
            [Paragraph("RANSAC inliers", styles["cell"]), Paragraph(_safe(inl), styles["cell"])],
            [Paragraph("Inlier ratio", styles["cell"]), Paragraph(_safe(ratio), styles["cell"])],
            [Paragraph("Final RMSE", styles["cell"]), rmse_para],
            [Paragraph("CE90", styles["cell"]), Paragraph(_safe(ce90), styles["cell"])],
            [Paragraph("RMSE after Bundle Adjustment", styles["cell"]), Paragraph(_safe(rmse_ba_txt), styles["cell"])],
            [Paragraph("BA improvement", styles["cell"]), Paragraph(_safe(improvement), styles["cell"])],
        ]
        story.append(self._section_table(rows, col_widths=[2.6 * inch, 3.9 * inch]))
        story.append(Spacer(1, 0.08 * inch))
        story.append(Paragraph("RMSE color code: green &lt; 0.5px (excellent), yellow 0.5-1.0px (acceptable), red &gt; 1.0px (poor).", styles["body"]))

    # ------------------------------------------------------------------
    def _create_phase_diagnostics_section(self, story, phases: dict) -> None:
        styles = self._styles()
        story.append(Paragraph("3. Pipeline Phase Diagnostics", styles["heading"]))
        if not isinstance(phases, dict) or not phases:
            story.append(Paragraph("No phase diagnostics available.", styles["body"]))
            return
        icon = {"success": "✅ Success", "partial": "⚠️ Partial", "failed": "❌ Failed", "skipped": "⏭️ Skipped"}
        header = [
            Paragraph("<b>Phase</b>", styles["cell"]),
            Paragraph("<b>Status</b>", styles["cell"]),
            Paragraph("<b>Matches Found</b>", styles["cell"]),
            Paragraph("<b>Confidence</b>", styles["cell"]),
        ]
        names = [p for p in _PHASE_ORDER if _phase_status(phases, p) != "skipped" or
                 (isinstance(phases, dict) and p in phases)]
        # Include any extra user-supplied phases beyond the canonical five.
        if isinstance(phases, dict) and "phases_executed" not in phases:
            for k in phases:
                if k not in _PHASE_ORDER and k not in ("phases_failed",):
                    names.append(k)
        if not names:
            names = list(_PHASE_ORDER)
        rows = [header]
        for name in names:
            st = _phase_status(phases, name)
            rows.append([
                Paragraph(name, styles["cell"]),
                Paragraph(icon.get(st, st), styles["cell"]),
                Paragraph(_phase_matches(phases, name), styles["cell"]),
                Paragraph(_phase_confidence(phases, name), styles["cell"]),
            ])
        story.append(self._section_table(rows, col_widths=[1.6 * inch, 1.6 * inch, 1.6 * inch, 1.7 * inch]))

    # ------------------------------------------------------------------
    def _create_distribution_chart(self, story, grid_occupancy: np.ndarray,
                                   coverage: float, balance: float, _tmp_files: list) -> None:
        styles = self._styles()
        story.append(Paragraph("4. Spatial Distribution Analysis", styles["heading"]))
        if grid_occupancy is None:
            story.append(Paragraph("No spatial distribution data available.", styles["body"]))
            return
        try:
            grid = np.asarray(grid_occupancy)
        except Exception:
            story.append(Paragraph("No spatial distribution data available.", styles["body"]))
            return
        if grid.size == 0:
            story.append(Paragraph("No spatial distribution data available.", styles["body"]))
            return
        try:
            cov = float(coverage) if coverage is not None else 0.0
        except (TypeError, ValueError):
            cov = 0.0
        try:
            bal = float(balance) if balance is not None else 0.0
        except (TypeError, ValueError):
            bal = 0.0
        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
            ax.set_xlabel("Grid Column")
            ax.set_ylabel("Grid Row")
            ax.set_title(f"Point Distribution Heatmap\nCoverage: {cov:.1%} | Balance: {bal:.2f}")
            plt.colorbar(im, ax=ax, label="Points per cell")
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_path = tmp.name
            tmp.close()
            _tmp_files.append(tmp_path)
            fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
            story.append(RLImage(tmp_path, width=5.5 * inch, height=4.5 * inch))
            story.append(Spacer(1, 0.08 * inch))
            story.append(Paragraph(f"Coverage: {cov:.1%} | Balance: {bal:.2f}", styles["body"]))
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    def _create_match_visualization(self, story, src_img_path: str, ref_img_path: str,
                                    src_pts, ref_pts, max_display: int = 100,
                                    _tmp_files: list = None) -> None:
        styles = self._styles()
        story.append(Paragraph("5. Match Point Visualization", styles["heading"]))
        try:
            if not src_img_path or not ref_img_path:
                raise FileNotFoundError("Missing image path(s).")
            src_img = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
            ref_img = cv2.imread(str(ref_img_path), cv2.IMREAD_COLOR)
            if src_img is None or ref_img is None:
                raise FileNotFoundError(f"Could not load images: {src_img_path}, {ref_img_path}")
            s = np.asarray(src_pts, dtype=np.float64).reshape(-1, 2) if src_pts is not None else np.zeros((0, 2))
            r = np.asarray(ref_pts, dtype=np.float64).reshape(-1, 2) if ref_pts is not None else np.zeros((0, 2))
            n = min(len(s), len(r))
            if n == 0:
                story.append(Paragraph("No match points available for visualization.", styles["body"]))
                return
            n = min(n, int(max_display))
            s, r = s[:n], r[:n]
            # Normalize heights for side-by-side composite.
            h = max(src_img.shape[0], ref_img.shape[0])
            def _resize_to_h(img, h_target: int):
                if img.shape[0] == h_target:
                    return img
                scale = h_target / float(img.shape[0])
                w_new = max(1, int(round(img.shape[1] * scale)))
                return cv2.resize(img, (w_new, h_target), interpolation=cv2.INTER_AREA)
            src_r = _resize_to_h(src_img, h)
            ref_r = _resize_to_h(ref_img, h)
            sy_scale = h / float(src_img.shape[0])
            sx_scale = src_r.shape[1] / float(src_img.shape[1])
            ry_scale = h / float(ref_img.shape[0])
            rx_scale = ref_r.shape[1] / float(ref_img.shape[1])
            composite = np.hstack([src_r, ref_r])
            x_off = src_r.shape[1]
            rng = np.random.RandomState(42)
            palette = (rng.randint(0, 255, size=(max(n, 1), 3))).astype(int)
            for i in range(n):
                color = tuple(int(c) for c in palette[i])
                p1 = (int(round(s[i, 0] * sx_scale)), int(round(s[i, 1] * sy_scale)))
                p2 = (x_off + int(round(r[i, 0] * rx_scale)), int(round(r[i, 1] * ry_scale)))
                cv2.circle(composite, p1, 4, color, 1, cv2.LINE_AA)
                cv2.circle(composite, p2, 4, color, 1, cv2.LINE_AA)
                cv2.line(composite, p1, p2, color, 1, cv2.LINE_AA)
            cv2.putText(composite, f"Top {n} matches", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_path = tmp.name
            tmp.close()
            cv2.imwrite(tmp_path, composite)
            if _tmp_files is not None:
                _tmp_files.append(tmp_path)
            max_w = 6.5 * inch
            scale = min(1.0, max_w / float(composite.shape[1]))
            disp_h = float(composite.shape[0]) * scale
            story.append(RLImage(tmp_path, width=float(composite.shape[1]) * scale,
                                 height=max(1.0, disp_h)))
            story.append(Spacer(1, 0.08 * inch))
            story.append(Paragraph(f"Showing top {n} matches (color-coded).", styles["body"]))
        except FileNotFoundError as exc:
            self.logger.warning("Skipping match visualization: %s", exc)
            story.append(Paragraph("Match visualization unavailable (images could not be loaded).", styles["body"]))
        except Exception as exc:
            self.logger.warning("Match visualization failed: %s", exc)
            story.append(Paragraph("Match visualization unavailable.", styles["body"]))

    # ------------------------------------------------------------------
    def _create_bundle_adjustment_section(self, story, ba_result: dict, _tmp_files: list) -> None:
        styles = self._styles()
        if ba_result is None or not isinstance(ba_result, dict):
            return
        story.append(Paragraph("6. Bundle Adjustment Results", styles["heading"]))
        init_rmse = _get(ba_result, "initial_rmse", "initial_rmse_pixels", "initial_cost_rmse",
                         "rmse_before", "rmse_before_ba", default=None)
        final_rmse = _get(ba_result, "final_rmse", "final_rmse_pixels", "rmse_after",
                          "rmse_after_ba", "final_rmse_after_ba", default=None)
        # Derive RMSE from costs when only squared costs are stored.
        if init_rmse is None and _get(ba_result, "initial_cost", default=None) is not None:
            try:
                n_pts = _get(ba_result, "num_points", "num_residuals", default=None)
                cost = float(_get(ba_result, "initial_cost"))
                init_rmse = float(np.sqrt(cost / float(n_pts))) if n_pts else float(np.sqrt(cost))
            except (TypeError, ValueError):
                pass
        if final_rmse is None and _get(ba_result, "final_cost", default=None) is not None:
            try:
                n_pts = _get(ba_result, "num_points", "num_residuals", default=None)
                cost = float(_get(ba_result, "final_cost"))
                final_rmse = float(np.sqrt(cost / float(n_pts))) if n_pts else float(np.sqrt(cost))
            except (TypeError, ValueError):
                pass
        improvement = _get(ba_result, "improvement_pct", "improvement_percent",
                           "improvement", default=None)
        if improvement is None and init_rmse is not None and final_rmse is not None:
            try:
                r0, r1 = float(init_rmse), float(final_rmse)
                improvement = (r0 - r1) / r0 * 100.0 if r0 > 0 else 0.0
            except (TypeError, ValueError):
                improvement = None
        n_iter = _get(ba_result, "num_iterations", "iterations", "num_function_evals",
                      "nfev", default=None)
        ref_id = _get(ba_result, "reference_id", "reference_image", "ref_id", "reference", default=None)
        rows = [
            [Paragraph("<b>Metric</b>", styles["cell"]), Paragraph("<b>Value</b>", styles["cell"])],
            [Paragraph("Initial RMSE", styles["cell"]),
             Paragraph(f"{float(init_rmse):.4f} px" if init_rmse is not None else "N/A", styles["cell"])],
            [Paragraph("Final RMSE", styles["cell"]),
             Paragraph(f"{float(final_rmse):.4f} px" if final_rmse is not None else "N/A", styles["cell"])],
            [Paragraph("Improvement", styles["cell"]),
             Paragraph(f"{float(improvement):.1f}%" if improvement is not None else "N/A", styles["cell"])],
            [Paragraph("Iterations / Function evals", styles["cell"]), Paragraph(_safe(n_iter), styles["cell"])],
            [Paragraph("Reference image", styles["cell"]), Paragraph(_safe(ref_id), styles["cell"])],
        ]
        story.append(self._section_table(rows, col_widths=[2.6 * inch, 3.9 * inch]))
        # Bar chart: initial vs final RMSE.
        try:
            if init_rmse is not None and final_rmse is not None:
                r0, r1 = float(init_rmse), float(final_rmse)
                if np.isfinite(r0) and np.isfinite(r1):
                    fig, ax = plt.subplots(figsize=(5, 3))
                    try:
                        ax.bar(["Initial RMSE", "Final RMSE"], [r0, r1],
                               color=["#B00020", "#1B7A2B"])
                        ax.set_ylabel("RMSE (px)")
                        ax.set_title("Bundle Adjustment: RMSE Before vs After")
                        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        tmp_path = tmp.name
                        tmp.close()
                        _tmp_files.append(tmp_path)
                        fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
                        story.append(Spacer(1, 0.1 * inch))
                        story.append(RLImage(tmp_path, width=5.0 * inch, height=3.0 * inch))
                    finally:
                        plt.close(fig)
        except Exception as exc:
            self.logger.warning("BA chart failed: %s", exc)

    # ------------------------------------------------------------------
    def generate_report(
        self,
        metadata: dict,
        metrics: dict,
        phases: dict,
        grid_occupancy: np.ndarray,
        coverage: float,
        balance: float,
        src_img_path: str,
        ref_img_path: str,
        src_pts,
        ref_pts,
        ba_result: dict = None,
        team_name: str = "Team",
    ) -> str:
        """Build the full PDF report. NEVER crashes on missing data.

        Returns:
            Absolute file path of the generated PDF (or an error string
            starting with ``"ERROR:"`` when the PDF cannot be written).
        """
        tmp_files: list = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_name = f"ISRO_Registration_Report_{timestamp}.pdf"
        pdf_path = str(Path(self.output_dir).resolve() / pdf_name)
        md = dict(metadata) if isinstance(metadata, dict) else {}
        if team_name and "team_name" not in md and "team" not in md:
            md["team_name"] = team_name

        story: list = []
        try:
            try:
                doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                                        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                                        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                                        title="Chandrayaan-2 Registration Report",
                                        author=str(md.get("team_name", team_name)))
            except PermissionError as exc:
                self.logger.error("PDF write permission denied: %s", exc)
                return f"ERROR: Permission denied writing to {self.output_dir}: {exc}"
            except Exception as exc:
                self.logger.error("Could not create PDF document: %s", exc)
                return f"ERROR: Could not create PDF: {exc}"

            sections = [
                ("header", lambda: self._create_header_page(story, md)),
                ("metadata", lambda: self._create_metadata_section(story, md)),
                ("metrics", lambda: self._create_metrics_section(story, metrics)),
                ("phases", lambda: self._create_phase_diagnostics_section(story, phases)),
                ("distribution", lambda: self._create_distribution_chart(
                    story, grid_occupancy, coverage, balance, tmp_files)),
                ("visualization", lambda: self._create_match_visualization(
                    story, src_img_path, ref_img_path, src_pts, ref_pts,
                    _tmp_files=tmp_files)),
                ("bundle_adjustment", lambda: self._create_bundle_adjustment_section(
                    story, ba_result, tmp_files)),
            ]
            for name, fn in sections:
                try:
                    fn()
                except Exception as exc:
                    self.logger.warning("Report section '%s' failed, continuing: %s", name, exc)
                    continue

            try:
                doc.build(story)
            except PermissionError as exc:
                self.logger.error("PDF write permission denied: %s", exc)
                return f"ERROR: Permission denied writing to {pdf_path}: {exc}"
            except Exception as exc:
                self.logger.error("PDF build failed: %s", exc)
                return f"ERROR: PDF build failed: {exc}"

            try:
                size = os.path.getsize(pdf_path)
                pages = getattr(doc, "page", "?")
                self.logger.info("Report generated: %s (%d bytes, %s pages).", pdf_path, size, pages)
            except Exception:
                pass
            return pdf_path
        finally:
            for p in tmp_files:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass


__all__ = ["ISROReportGenerator"]
