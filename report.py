from __future__ import annotations

import io
import logging
from collections import defaultdict
from dataclasses import dataclass

from Armor.utils.messages import Messages
from asset_management.constants import STATUS_COMPLETED as SCAN_COMPLETED
from asset_management.models import Asset, Port, Scan
from compliance_inspection.models import (
    COMPLIANCE_STATUS_FAIL,
    COMPLIANCE_STATUS_NOT_APPLICABLE,
    COMPLIANCE_STATUS_NEEDS_REVIEW,
    COMPLIANCE_STATUS_PARTIAL,
    COMPLIANCE_STATUS_PASS,
    ComplianceAssessment,
    ComplianceResult,
)
from django.core.files.base import ContentFile
from django.db.models import Case, Count, IntegerField, Prefetch, Q, Value, When
from django.http import FileResponse
from django.utils import timezone
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from risk_assessment.constants import STATUS_COMPLETED as ASSESSMENT_COMPLETED
from risk_assessment.models import AssetRiskProfile, RiskAssessment, Vulnerability

from shared.models import ReportMetadata

logger = logging.getLogger(__name__)


def build_cache_key(
    scan_id: str,
    risk_assessment_id: str | None = None,
    compliance_assessment_id: str | None = None,
) -> str:
    risk_part = risk_assessment_id or "no-risk"
    compliance_part = compliance_assessment_id or "no-compliance"
    return f"{scan_id}:{risk_part}:{compliance_part}"


class ReportGenerationError(Exception):
    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(slots=True)
class ReportSummary:
    total_assets: int
    total_open_ports: int
    risk_assessment_id: str | None
    compliance_assessment_id: str | None
    risk_available: bool
    compliance_available: bool


@dataclass(slots=True)
class ReportContext:
    scan: Scan
    risk_assessment: RiskAssessment | None
    compliance_assessment: ComplianceAssessment | None
    summary: ReportSummary
    assets: list
    asset_risk_profiles: list
    vulnerabilities: list
    compliance_results: list
    risk_note: str | None = None
    compliance_note: str | None = None
    report_metadata: ReportMetadata | None = None


class ReportService:
    def __init__(
        self,
        scan_id=None,
        scan: Scan | None = None,
        risk_assessment: RiskAssessment | None = None,
        compliance_assessment: ComplianceAssessment | None = None,
        report_metadata: ReportMetadata | None = None,
    ):
        if scan is None and scan_id is None:
            raise ValueError("scan_id or scan is required")

        self.scan_id = str(scan_id or scan.id)
        self.scan = scan or self._get_scan()
        self.risk_assessment = (
            risk_assessment
            if risk_assessment is not None
            else self._get_latest_completed_risk_assessment()
        )
        self.compliance_assessment = (
            compliance_assessment
            if compliance_assessment is not None
            else self._get_latest_completed_compliance_assessment()
        )
        self.report_metadata = report_metadata

    def _get_scan(self) -> Scan:
        try:
            scan = Scan.objects.get(id=self.scan_id)
        except Scan.DoesNotExist as exc:
            raise ReportGenerationError(
                Messages.SCAN_NOT_FOUND, status_code=404
            ) from exc

        if scan.status != SCAN_COMPLETED:
            raise ReportGenerationError(
                Messages.REPORT_SCAN_NOT_COMPLETED.format(status=scan.status)
            )
        return scan

    def _get_latest_completed_risk_assessment(self) -> RiskAssessment | None:
        return (
            RiskAssessment.objects.select_related("scan")
            .filter(scan=self.scan, status=ASSESSMENT_COMPLETED)
            .order_by("-completed_at", "-created_at")
            .first()
        )

    def _get_latest_completed_compliance_assessment(
        self,
    ) -> ComplianceAssessment | None:
        if self.risk_assessment is None:
            return None

        return (
            ComplianceAssessment.objects.select_related(
                "risk_assessment", "risk_assessment__scan"
            )
            .filter(
                risk_assessment=self.risk_assessment,
                status=ComplianceAssessment.STATUS_COMPLETED,
            )
            .order_by("-completed_at", "-created_at")
            .first()
        )

    def _get_assets(self):
        ports = Port.objects.order_by("port_number", "protocol")
        return (
            Asset.objects.filter(scan=self.scan)
            .prefetch_related(Prefetch("ports", queryset=ports))
            .annotate(
                total_ports_count=Count("ports", distinct=True),
                open_ports_count=Count(
                    "ports", filter=Q(ports__state="open"), distinct=True
                ),
                filtered_ports_count=Count(
                    "ports", filter=Q(ports__state="filtered"), distinct=True
                ),
            )
            .order_by("ip_address")
        )

    def _get_asset_risk_profiles(self):
        if self.risk_assessment is None:
            return AssetRiskProfile.objects.none()

        vulnerabilities = Vulnerability.objects.select_related(
            "port",
            "cve",
            "asset_risk_profile__asset",
        ).order_by("-armor_risk_score", "-created_at")

        return (
            AssetRiskProfile.objects.filter(assessment=self.risk_assessment)
            .select_related("asset")
            .prefetch_related(Prefetch("vulnerabilities", queryset=vulnerabilities))
            .order_by("-risk_score", "asset__ip_address")
        )

    def _get_vulnerabilities(self):
        if self.risk_assessment is None:
            return Vulnerability.objects.none()

        severity_rank = Case(
            When(severity="critical", then=Value(0)),
            When(severity="high", then=Value(1)),
            When(severity="medium", then=Value(2)),
            When(severity="low", then=Value(3)),
            default=Value(4),
            output_field=IntegerField(),
        )

        return (
            Vulnerability.objects.filter(
                asset_risk_profile__assessment=self.risk_assessment
            )
            .select_related("asset_risk_profile__asset", "port", "cve")
            .annotate(severity_rank=severity_rank)
            .order_by("severity_rank", "-armor_risk_score", "title")
        )

    def _get_compliance_results(self):
        if self.compliance_assessment is None:
            return ComplianceResult.objects.none()

        return ComplianceResult.objects.filter(
            compliance_assessment=self.compliance_assessment
        ).order_by("framework", "status", "control_ref")

    def _risk_note(self) -> str | None:
        if self.risk_assessment is not None:
            return None
        return Messages.REPORT_RISK_ASSESSMENT_NOT_FOUND

    def _compliance_note(self) -> str | None:
        if self.compliance_assessment is not None:
            return None
        return Messages.REPORT_COMPLIANCE_ASSESSMENT_NOT_FOUND

    def build_context(self) -> ReportContext:
        assets = list(self._get_assets())
        return ReportContext(
            scan=self.scan,
            risk_assessment=self.risk_assessment,
            compliance_assessment=self.compliance_assessment,
            summary=ReportSummary(
                total_assets=len(assets),
                total_open_ports=sum(asset.open_ports_count or 0 for asset in assets),
                risk_assessment_id=(
                    str(self.risk_assessment.id) if self.risk_assessment else None
                ),
                compliance_assessment_id=(
                    str(self.compliance_assessment.id)
                    if self.compliance_assessment
                    else None
                ),
                risk_available=self.risk_assessment is not None,
                compliance_available=self.compliance_assessment is not None,
            ),
            assets=assets,
            asset_risk_profiles=list(self._get_asset_risk_profiles()),
            vulnerabilities=list(self._get_vulnerabilities()),
            compliance_results=list(self._get_compliance_results()),
            risk_note=self._risk_note(),
            compliance_note=self._compliance_note(),
            report_metadata=self._get_existing_metadata(),
        )

    def _get_existing_metadata(self) -> ReportMetadata | None:
        if self.report_metadata is not None:
            return self.report_metadata
        return ReportMetadata.objects.filter(cache_key=self._cache_key()).first()

    def _cache_key(self) -> str:
        return build_cache_key(
            self.scan_id,
            str(self.risk_assessment.id) if self.risk_assessment else None,
            str(self.compliance_assessment.id) if self.compliance_assessment else None,
        )

    def _build_filename(self) -> str:
        risk_part = str(self.risk_assessment.id) if self.risk_assessment else "no-risk"
        compliance_part = (
            str(self.compliance_assessment.id)
            if self.compliance_assessment
            else "no-compliance"
        )
        return f"armor_report_{self.scan_id}_{risk_part}_{compliance_part}.pdf"

    def _ensure_metadata(self) -> tuple[ReportMetadata, bool]:
        metadata = self._get_existing_metadata()
        if metadata is not None:
            return metadata, False

        metadata, created = ReportMetadata.objects.get_or_create(
            cache_key=self._cache_key(),
            defaults={
                "scan": self.scan,
                "risk_assessment": self.risk_assessment,
                "compliance_assessment": self.compliance_assessment,
                "generated_at": timezone.now(),
            },
        )
        return metadata, created

    def _format_port_list(self, asset) -> str:
        ports = []
        for port in asset.ports.all():
            if port.state != "open":
                continue
            label = f"{port.port_number}/{port.protocol}"
            if port.service:
                label = f"{label} {port.service}"
            ports.append(label)
        return ", ".join(ports) if ports else "None detected"

    def _format_framework_name(self, framework: str) -> str:
        return {
            "iso27001": "ISO 27001:2022",
            "cis": "CIS Controls v8.1",
            "nist": "NIST SP 800-53 Rev5",
        }.get(framework, framework)

    def _format_last_seen(self, asset) -> str:
        if not getattr(asset, "last_seen", None):
            return "-"
        return asset.last_seen.strftime("%Y-%m-%d %H:%M")

    def _build_asset_inventory_rows(self, assets) -> list[list[str]]:
        rows = [
            [
                "IP Address",
                "Hostname",
                "MAC Address",
                "Device Type",
                "OS Name",
                "Vendor",
                "Severity",
                "Active",
                "Last Seen",
            ]
        ]
        for asset in assets:
            rows.append(
                [
                    asset.ip_address,
                    asset.hostname or "-",
                    asset.mac_address or "-",
                    asset.device_type or "-",
                    asset.os_name or "-",
                    asset.vendor or "-",
                    asset.severity or "-",
                    "Yes" if asset.is_active else "No",
                    self._format_last_seen(asset),
                ]
            )
        return rows

    def _build_asset_exposure_rows(self, assets) -> list[list[str]]:
        rows = [["IP Address", "Open Ports", "Filtered Ports", "Ports"]]
        for asset in assets:
            rows.append(
                [
                    asset.ip_address,
                    str(asset.open_ports_count or 0),
                    str(asset.filtered_ports_count or 0),
                    self._format_port_list(asset),
                ]
            )
        return rows

    # ──────────────────────────────────────────────────────────────────────────
    # Style palette — single source of truth for all colours used in the report
    # ──────────────────────────────────────────────────────────────────────────
    _NAVY = "#0f172a"  # page header / asset card header
    _SLATE = "#1e293b"  # section label text
    _INDIGO = "#4f46e5"  # accent bar on subsection headings
    _INDIGO_LT = "#eef2ff"  # tinted subsection header background
    _BORDER = "#cbd5e1"  # table / card borders
    _ROW_ALT = "#f8fafc"  # table alternating row tint
    _MUTED = "#64748b"  # secondary / caption text
    _GREEN = "#16a34a"  # pass / active badges
    _AMBER = "#d97706"  # medium severity
    _RED = "#dc2626"  # critical / fail
    _ORANGE = "#ea580c"  # high
    _BLUE = "#2563eb"  # low
    _GREY = "#64748b"  # info / none

    def _sev_bg(self, severity: str | None) -> str:
        """Light background tint for severity badge cells."""
        return {
            "critical": "#fef2f2",
            "high": "#fff7ed",
            "medium": "#fffbeb",
            "low": "#eff6ff",
            "info": "#f8fafc",
        }.get((severity or "").lower(), "#f8fafc")

    def _sev_fg(self, severity: str | None) -> str:
        return {
            "critical": self._RED,
            "high": self._ORANGE,
            "medium": self._AMBER,
            "low": self._BLUE,
            "info": self._GREY,
        }.get((severity or "").lower(), self._SLATE)

    def _severity_color(self, severity: str | None) -> colors.Color:
        return colors.HexColor(self._sev_fg(severity))

    def _doc_table_style(self) -> TableStyle:
        return TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(self._NAVY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                ("FONTSIZE", (0, 1), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("LINEBELOW", (0, 0), (-1, 0), 1.5, colors.HexColor(self._INDIGO)),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor(self._BORDER)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor(self._ROW_ALT)],
                ),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )

    def _card_box_style(self) -> TableStyle:
        """Full-width card with a coloured left accent border."""
        return TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(self._INDIGO_LT)),
                ("LINEAFTER", (0, 0), (0, -1), 3, colors.HexColor(self._INDIGO)),
                ("LINEBEFORE", (0, 0), (0, -1), 0.5, colors.HexColor(self._BORDER)),
                ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.HexColor(self._BORDER)),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor(self._BORDER)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Unified per-asset section — asset + risk + compliance all together
    # ──────────────────────────────────────────────────────────────────────────
    def _build_asset_inventory_section(
        self, context: ReportContext, section_style, body_style
    ):
        """
        Render every discovered asset as a self-contained document block.
        Each block includes identity, network exposure, risk profile,
        top-10 vulnerabilities, and only the compliance controls that asset fails.
        No separate Risk or Compliance sections are emitted by this renderer.
        """
        # ── Local style helpers ────────────────────────────────────────────
        W = (
            (595.27 - 40 - 40) / 72 * inch
        )  # exact usable content width (matches _build_pdf_bytes)

        asset_header_style = ParagraphStyle(
            "AssetHeader",
            parent=body_style,
            fontSize=13,
            leading=16,
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )
        asset_index_style = ParagraphStyle(
            "AssetIndex",
            parent=body_style,
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#94a3b8"),
            alignment=TA_RIGHT,
        )
        sub_heading_style = ParagraphStyle(
            "SubHeading",
            parent=body_style,
            fontSize=9,
            leading=11,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor(self._INDIGO),
            spaceBefore=8,
            spaceAfter=3,
        )
        field_label_style = ParagraphStyle(
            "FL",
            parent=body_style,
            fontSize=7.5,
            leading=10,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor(self._MUTED),
        )
        field_value_style = ParagraphStyle(
            "FV",
            parent=body_style,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor(self._SLATE),
        )
        note_style = ParagraphStyle(
            "Note",
            parent=body_style,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor(self._MUTED),
            leftIndent=4,
        )
        badge_pass_style = ParagraphStyle(
            "BadgePass",
            parent=body_style,
            fontSize=8,
            leading=10,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor(self._GREEN),
        )
        badge_fail_style = ParagraphStyle(
            "BadgeFail",
            parent=body_style,
            fontSize=8,
            leading=10,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor(self._RED),
        )

        # ── Pre-compute lookups ────────────────────────────────────────────
        failing_by_asset: dict[str, list] = defaultdict(list)
        for result in context.compliance_results:
            if result.status != COMPLIANCE_STATUS_FAIL:
                continue
            for ae in (result.evidence or {}).get("affected_assets") or []:
                ip = ae.get("ip_address") or ae.get("ip")
                if ip:
                    failing_by_asset[ip].append(result)

        vulns_by_asset = self._build_vulnerability_lookup(context)

        risk_profile_by_asset: dict[str, object] = {}
        for profile in context.asset_risk_profiles:
            risk_profile_by_asset[str(profile.asset.ip_address)] = profile

        # ── Intro paragraph ────────────────────────────────────────────────
        section: list = [Paragraph("Asset Reports", section_style)]
        section.append(
            HRFlowable(
                width="100%",
                thickness=1.5,
                color=colors.HexColor(self._INDIGO),
                spaceAfter=10,
            )
        )
        section.append(
            Paragraph(
                (
                    f"This section presents a consolidated profile for each of the "
                    f"{context.summary.total_assets} assets discovered during the scan. "
                    "Each profile includes identity, network exposure, risk assessment, "
                    "vulnerability findings, and compliance violations — all in one place. "
                    "Assets are ordered by IP address."
                ),
                body_style,
            )
        )
        section.append(Spacer(1, 0.14 * inch))

        total = len(context.assets)
        for asset_idx, asset in enumerate(context.assets, start=1):
            ip = str(asset.ip_address)
            profile = risk_profile_by_asset.get(ip)
            asset_vulns = vulns_by_asset.get(ip, [])

            # Deduplicate failing controls for this asset
            seen_ctrl: set[tuple[str, str]] = set()
            unique_failing: list = []
            for r in failing_by_asset.get(ip, []):
                k = (r.framework, r.control_ref)
                if k not in seen_ctrl:
                    seen_ctrl.add(k)
                    unique_failing.append(r)

            # ── Asset header band ──────────────────────────────────────────
            severity_raw = (asset.severity or "").lower()
            header_bg = colors.HexColor(
                {
                    "critical": "#7f1d1d",
                    "high": "#7c2d12",
                    "medium": "#78350f",
                    "low": "#1e3a8a",
                }.get(severity_raw, self._NAVY)
            )

            header_left = Paragraph(
                f"&#x25A0; &nbsp; {ip}"
                + (f" &nbsp;·&nbsp; {asset.hostname}" if asset.hostname else ""),
                asset_header_style,
            )
            header_right = Paragraph(f"Asset {asset_idx} of {total}", asset_index_style)
            header_table = Table(
                [[header_left, header_right]],
                colWidths=[W * 0.72, W * 0.28],
            )
            header_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), header_bg),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                )
            )
            section.append(header_table)

            # ── Identity + Risk summary — two-column row ───────────────────
            identity_lines = [
                ("MAC Address", asset.mac_address or "—"),
                ("Device Type", asset.device_type or "Unknown"),
                ("Operating System", asset.os_name or "Unknown"),
                ("Vendor", asset.vendor or "Unknown"),
                ("Status", "Active" if asset.is_active else "Inactive"),
                ("Last Seen", self._format_last_seen(asset)),
            ]
            id_rows = [
                [Paragraph(lbl, field_label_style), Paragraph(val, field_value_style)]
                for lbl, val in identity_lines
            ]
            id_table = Table(id_rows, colWidths=[W * 0.28, W * 0.16])
            id_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        (
                            "LINEBELOW",
                            (0, 0),
                            (-1, -2),
                            0.25,
                            colors.HexColor("#e2e8f0"),
                        ),
                    ]
                )
            )

            # Risk summary card (right column)
            if profile is not None:
                risk_level = (profile.risk_level or "").upper()
                rl_color = {
                    "CRITICAL": self._RED,
                    "HIGH": self._ORANGE,
                    "MEDIUM": self._AMBER,
                    "LOW": self._BLUE,
                }.get(risk_level, self._MUTED)
                risk_lines = [
                    ("Risk Score", self._format_risk_value(profile.risk_score)),
                    ("Risk Level", risk_level or "—"),
                    ("Findings", str(profile.vulnerability_count)),
                    ("Critical", str(profile.critical_vuln_count)),
                    ("High", str(profile.high_vuln_count)),
                    ("KEV Affected", "Yes" if profile.is_kev_affected else "No"),
                ]
                risk_rows = []
                for lbl, val in risk_lines:
                    val_style = field_value_style
                    if lbl == "Risk Level" and val != "—":
                        val_style = ParagraphStyle(
                            f"RL_{val}",
                            parent=field_value_style,
                            textColor=colors.HexColor(rl_color),
                            fontName="Helvetica-Bold",
                        )
                    risk_rows.append(
                        [
                            Paragraph(lbl, field_label_style),
                            Paragraph(val, val_style),
                        ]
                    )
                risk_inner = Table(risk_rows, colWidths=[W * 0.22, W * 0.34])
                risk_inner.setStyle(
                    TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                            (
                                "LINEBELOW",
                                (0, 0),
                                (-1, -2),
                                0.25,
                                colors.HexColor("#e2e8f0"),
                            ),
                            (
                                "LINEBEFORE",
                                (0, 0),
                                (0, -1),
                                2.5,
                                colors.HexColor(self._INDIGO),
                            ),
                        ]
                    )
                )
                right_col = risk_inner
            else:
                right_col = Paragraph(
                    "No risk assessment data available for this asset.", note_style
                )

            two_col = Table(
                [[id_table, right_col]],
                colWidths=[W * 0.44, W * 0.56],
            )
            two_col.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                        (
                            "LINEAFTER",
                            (0, 0),
                            (0, -1),
                            0.4,
                            colors.HexColor(self._BORDER),
                        ),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(self._BORDER)),
                    ]
                )
            )
            section.append(two_col)

            # ── AI summary / recommendations ───────────────────────────────
            if profile is not None and profile.ai_summary:
                section.append(Spacer(1, 0.04 * inch))
                summary_box = Table(
                    [
                        [
                            Paragraph(
                                f"<i>{profile.ai_summary}</i>",
                                ParagraphStyle(
                                    "AISummary",
                                    parent=body_style,
                                    fontSize=8,
                                    leading=11,
                                    textColor=colors.HexColor(self._SLATE),
                                    leftIndent=4,
                                ),
                            )
                        ]
                    ],
                    colWidths=[W],
                )
                summary_box.setStyle(
                    TableStyle(
                        [
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, -1),
                                colors.HexColor("#f0f9ff"),
                            ),
                            (
                                "LINEBEFORE",
                                (0, 0),
                                (0, -1),
                                3,
                                colors.HexColor("#0ea5e9"),
                            ),
                            ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#bae6fd")),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ]
                    )
                )
                section.append(summary_box)

            # ── Network exposure ───────────────────────────────────────────
            section.append(Paragraph("NETWORK EXPOSURE", sub_heading_style))
            open_count = asset.open_ports_count or 0
            filtered_count = asset.filtered_ports_count or 0
            port_list = self._format_port_list(asset)
            exposure_data = [
                [
                    Paragraph("Open Ports", field_label_style),
                    Paragraph(str(open_count), field_value_style),
                    Paragraph("Filtered Ports", field_label_style),
                    Paragraph(str(filtered_count), field_value_style),
                ],
                [
                    Paragraph("Services", field_label_style),
                    Paragraph(port_list, field_value_style),
                    Paragraph("", field_label_style),
                    Paragraph("", field_value_style),
                ],
            ]
            exposure_tbl = Table(
                exposure_data,
                colWidths=[W * 0.13, W * 0.25, W * 0.13, W * 0.49],
            )
            exposure_tbl.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(self._BORDER)),
                        (
                            "LINEBELOW",
                            (0, 0),
                            (-1, 0),
                            0.25,
                            colors.HexColor(self._BORDER),
                        ),
                        ("SPAN", (1, 1), (3, 1)),
                    ]
                )
            )
            section.append(exposure_tbl)

            # ── Vulnerabilities ────────────────────────────────────────────
            if context.risk_assessment is not None:
                section.append(Paragraph("VULNERABILITY FINDINGS", sub_heading_style))
                if not asset_vulns:
                    section.append(
                        Paragraph(
                            "No vulnerabilities recorded for this asset.", note_style
                        )
                    )
                else:
                    v_header = [
                        Paragraph(h, field_label_style)
                        for h in ["#", "Title", "Sev", "CVE", "Port", "Score"]
                    ]
                    v_rows = [v_header]
                    for idx, v in enumerate(asset_vulns[:10], 1):
                        cve_id = getattr(getattr(v, "cve", None), "cve_id", None) or "—"
                        port_lbl = (
                            f"{v.port.port_number}/{v.port.protocol}" if v.port else "—"
                        )
                        sev = (v.severity or "").lower()
                        sev_style = ParagraphStyle(
                            f"Sev_{sev}_{idx}",
                            parent=field_value_style,
                            fontSize=7.5,
                            fontName="Helvetica-Bold",
                            textColor=colors.HexColor(self._sev_fg(sev)),
                        )
                        v_rows.append(
                            [
                                Paragraph(str(idx), note_style),
                                Paragraph(v.title or "—", field_value_style),
                                Paragraph(sev.upper(), sev_style),
                                Paragraph(cve_id, note_style),
                                Paragraph(port_lbl, note_style),
                                Paragraph(
                                    self._format_risk_value(v.armor_risk_score),
                                    field_value_style,
                                ),
                            ]
                        )

                    # Severity row background colouring
                    v_style_cmds = list(self._doc_table_style()._cmds)  # type: ignore[attr-defined]
                    for row_idx, v in enumerate(asset_vulns[:10], 1):
                        bg = self._sev_bg(v.severity)
                        if bg != "#f8fafc":
                            v_style_cmds.append(
                                (
                                    "BACKGROUND",
                                    (2, row_idx),
                                    (2, row_idx),
                                    colors.HexColor(bg),
                                )
                            )

                    v_table = Table(
                        v_rows,
                        repeatRows=1,
                        colWidths=[
                            W * 0.04,
                            W * 0.36,
                            W * 0.10,
                            W * 0.16,
                            W * 0.10,
                            W * 0.10,
                        ],
                        hAlign="LEFT",
                    )
                    v_table.setStyle(TableStyle(v_style_cmds))
                    section.append(v_table)

                    if len(asset_vulns) > 10:
                        section.append(
                            Paragraph(
                                f"Showing the top 10 of {len(asset_vulns)} vulnerabilities "
                                "ordered by risk score.",
                                note_style,
                            )
                        )

                    # Recommendations inline
                    if profile is not None and profile.recommendations:
                        recs = "; ".join(
                            str(r)
                            for r in profile.recommendations[:3]
                            if str(r).strip()
                        )
                        if recs:
                            section.append(Spacer(1, 0.04 * inch))
                            rec_box = Table(
                                [
                                    [
                                        Paragraph(
                                            f"<b>Recommendations:</b> {recs}",
                                            ParagraphStyle(
                                                "Rec",
                                                parent=body_style,
                                                fontSize=8,
                                                leading=11,
                                                textColor=colors.HexColor(self._SLATE),
                                            ),
                                        )
                                    ]
                                ],
                                colWidths=[W],
                            )
                            rec_box.setStyle(
                                TableStyle(
                                    [
                                        (
                                            "BACKGROUND",
                                            (0, 0),
                                            (-1, -1),
                                            colors.HexColor("#fefce8"),
                                        ),
                                        (
                                            "LINEBEFORE",
                                            (0, 0),
                                            (0, -1),
                                            3,
                                            colors.HexColor(self._AMBER),
                                        ),
                                        (
                                            "BOX",
                                            (0, 0),
                                            (-1, -1),
                                            0.4,
                                            colors.HexColor("#fde68a"),
                                        ),
                                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                                    ]
                                )
                            )
                            section.append(rec_box)

            # ── Compliance violations ──────────────────────────────────────
            if context.compliance_assessment is not None:
                section.append(Paragraph("COMPLIANCE VIOLATIONS", sub_heading_style))
                if not unique_failing:
                    ok_box = Table(
                        [
                            [
                                Paragraph("✓", badge_pass_style),
                                Paragraph(
                                    "No failing compliance controls for this asset.",
                                    field_value_style,
                                ),
                            ]
                        ],
                        colWidths=[0.22 * inch, W - 0.22 * inch],
                    )
                    ok_box.setStyle(
                        TableStyle(
                            [
                                (
                                    "BACKGROUND",
                                    (0, 0),
                                    (-1, -1),
                                    colors.HexColor("#f0fdf4"),
                                ),
                                (
                                    "BOX",
                                    (0, 0),
                                    (-1, -1),
                                    0.4,
                                    colors.HexColor("#bbf7d0"),
                                ),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                ("TOPPADDING", (0, 0), (-1, -1), 4),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ]
                        )
                    )
                    section.append(ok_box)
                else:
                    # Group failing controls by framework
                    framework_map = {
                        "iso27001": ("ISO 27001:2022", "#3b0764", "#f5f3ff"),
                        "nist": ("NIST SP 800-53 Rev5", "#0c2340", "#eff6ff"),
                        "cis": ("CIS Controls v8.1", "#052e16", "#f0fdf4"),
                    }
                    framework_order = ["iso27001", "nist", "cis"]
                    results_by_framework: dict = {fw: [] for fw in framework_order}
                    for result in unique_failing:
                        fw_key = result.framework
                        if fw_key in results_by_framework:
                            results_by_framework[fw_key].append(result)

                    c_header_cols = [
                        "Control",
                        "Category",
                        "Severity Impact",
                        "Root Cause",
                    ]

                    for fw_key in framework_order:
                        fw_results = results_by_framework[fw_key]
                        if not fw_results:
                            continue

                        fw_display_name, fw_label_color, fw_bg_color = framework_map[
                            fw_key
                        ]

                        # Framework sub-label
                        fw_label_style = ParagraphStyle(
                            f"FwLabel_{fw_key}",
                            parent=body_style,
                            fontSize=9,
                            leading=12,
                            fontName="Helvetica-Bold",
                            textColor=colors.HexColor(fw_label_color),
                            spaceBefore=6,
                            spaceAfter=3,
                        )
                        section.append(Paragraph(fw_display_name, fw_label_style))

                        c_header = [
                            Paragraph(h, field_label_style) for h in c_header_cols
                        ]
                        c_rows = [c_header]
                        for result in fw_results:
                            ev = result.evidence or {}
                            root_cause = self._compact_text(
                                ev.get("note")
                                or ev.get("rationale")
                                or ev.get("reason")
                            )
                            c_rows.append(
                                [
                                    Paragraph(
                                        result.control_ref or "—", field_value_style
                                    ),
                                    Paragraph(result.category or "—", note_style),
                                    Paragraph(
                                        result.severity_impact or "—", field_value_style
                                    ),
                                    Paragraph(
                                        root_cause if root_cause != "-" else "—",
                                        note_style,
                                    ),
                                ]
                            )

                        c_style_cmds = list(self._doc_table_style()._cmds)  # type: ignore[attr-defined]
                        # Header background matches framework colour
                        c_style_cmds.append(
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                colors.HexColor(fw_bg_color),
                            )
                        )
                        # Tint every data row with a very light red
                        for ri in range(1, len(c_rows)):
                            c_style_cmds.append(
                                (
                                    "BACKGROUND",
                                    (0, ri),
                                    (1, ri),
                                    colors.HexColor("#fff5f5"),
                                )
                            )

                        c_table = Table(
                            c_rows,
                            repeatRows=1,
                            colWidths=[W * 0.14, W * 0.24, W * 0.14, W * 0.48],
                            hAlign="LEFT",
                        )
                        c_table.setStyle(TableStyle(c_style_cmds))
                        section.append(c_table)
                        section.append(Spacer(1, 0.04 * inch))

            # ── Separator between assets ───────────────────────────────────
            section.append(Spacer(1, 0.18 * inch))

        return section

    def _format_risk_value(self, value) -> str:
        if value in (None, ""):
            return "-"
        return f"{value:.1f}" if isinstance(value, (int, float)) else str(value)

    def _build_risk_overview_rows(self, context: ReportContext) -> list[list[str]]:
        assessment = context.risk_assessment
        rows = [["Metric", "Value"]]
        if assessment is None:
            rows.append(["Risk assessment", "Not available"])
            return rows

        rows.extend(
            [
                [
                    "Overall score",
                    self._format_risk_value(assessment.overall_risk_score),
                ],
                ["Risk level", assessment.overall_risk_level or "-"],
                ["Total vulnerabilities", str(assessment.total_vulnerabilities)],
                ["Critical", str(assessment.critical_count)],
                ["High", str(assessment.high_count)],
                ["Medium", str(assessment.medium_count)],
                ["Low", str(assessment.low_count)],
                ["Info", str(assessment.info_count)],
                ["NVD covered assets", str(assessment.nvd_covered_assets)],
                ["Rule-only assets", str(assessment.rule_only_assets)],
            ]
        )
        return rows

    def _build_severity_breakdown_rows(self, context: ReportContext) -> list[list[str]]:
        rows = [["Severity", "Assessment Count", "Rendered Count"]]
        assessment = context.risk_assessment
        assessment_counts = {
            "critical": getattr(assessment, "critical_count", 0) if assessment else 0,
            "high": getattr(assessment, "high_count", 0) if assessment else 0,
            "medium": getattr(assessment, "medium_count", 0) if assessment else 0,
            "low": getattr(assessment, "low_count", 0) if assessment else 0,
            "info": getattr(assessment, "info_count", 0) if assessment else 0,
        }
        rendered_counts = defaultdict(int)
        for vulnerability in context.vulnerabilities:
            rendered_counts[vulnerability.severity] += 1

        for severity in ["critical", "high", "medium", "low", "info"]:
            rows.append(
                [
                    severity.title(),
                    str(assessment_counts[severity]),
                    str(rendered_counts[severity]),
                ]
            )
        return rows

    def _aggregate_top_vulnerabilities(self, context: ReportContext) -> list[dict]:
        aggregated: dict[tuple, dict] = {}
        for vulnerability in context.vulnerabilities:
            cve_id = getattr(getattr(vulnerability, "cve", None), "cve_id", None)
            port_label = (
                f"{vulnerability.port.port_number}/{vulnerability.port.protocol}"
                if vulnerability.port
                else "-"
            )
            canonical_key = (
                cve_id or vulnerability.title,
                vulnerability.severity,
                vulnerability.vuln_type,
                port_label,
                vulnerability.remediation or "",
            )
            entry = aggregated.setdefault(
                canonical_key,
                {
                    "title": vulnerability.title,
                    "severity": vulnerability.severity,
                    "cve_id": cve_id,
                    "port_label": port_label,
                    "vuln_type": vulnerability.vuln_type,
                    "armor_risk_score": vulnerability.armor_risk_score,
                    "affected_assets": set(),
                    "remediation": vulnerability.remediation or "",
                },
            )
            entry["affected_assets"].add(
                vulnerability.asset_risk_profile.asset.ip_address
            )
            entry["armor_risk_score"] = max(
                entry["armor_risk_score"], vulnerability.armor_risk_score
            )

        rows = sorted(
            aggregated.values(),
            key=lambda item: (
                -float(item["armor_risk_score"] or 0.0),
                -len(item["affected_assets"]),
                item["title"],
            ),
        )
        return rows

    def _build_top_vulnerability_rows(self, context: ReportContext) -> list[list[str]]:
        rows = [["Title", "Severity", "CVE", "Affected Assets", "Port", "Risk Score"]]
        for vulnerability in self._aggregate_top_vulnerabilities(context)[:10]:
            rows.append(
                [
                    vulnerability["title"],
                    vulnerability["severity"],
                    vulnerability["cve_id"] or "-",
                    str(len(vulnerability["affected_assets"])),
                    vulnerability["port_label"],
                    self._format_risk_value(vulnerability["armor_risk_score"]),
                ]
            )
        return rows

    def _build_asset_risk_profile_rows(self, context: ReportContext) -> list[list[str]]:
        rows = [
            ["IP Address", "Risk Score", "Level", "Vulns", "Critical", "High", "KEV"]
        ]
        for profile in context.asset_risk_profiles:
            rows.append(
                [
                    profile.asset.ip_address,
                    self._format_risk_value(profile.risk_score),
                    profile.risk_level or "-",
                    str(profile.vulnerability_count),
                    str(profile.critical_vuln_count),
                    str(profile.high_vuln_count),
                    "Yes" if profile.is_kev_affected else "No",
                ]
            )
        return rows

    def _build_asset_vulnerability_sections(
        self, context: ReportContext, section_style, body_style
    ):
        if not context.asset_risk_profiles:
            return []

        subsection_style = ParagraphStyle(
            "SharedReportSubsection",
            parent=body_style,
            fontSize=11,
            leading=14,
            textColor=colors.HexColor(self._SLATE),
            fontName="Helvetica-Bold",
            spaceBefore=8,
            spaceAfter=4,
        )

        vulnerabilities_by_asset = defaultdict(list)
        for vulnerability in context.vulnerabilities:
            vulnerabilities_by_asset[vulnerability.asset_risk_profile.asset_id].append(
                vulnerability
            )

        sections = [Paragraph("Per-Asset Risk Detail", section_style)]
        sections.append(
            HRFlowable(
                width="100%",
                thickness=1.5,
                color=colors.HexColor(self._INDIGO),
                spaceAfter=10,
            )
        )
        for profile in context.asset_risk_profiles:
            asset = profile.asset
            asset_vulnerabilities = sorted(
                vulnerabilities_by_asset.get(profile.asset_id, []),
                key=lambda item: (-item.armor_risk_score, item.title),
            )

            sections.append(
                Paragraph(
                    (
                        f"<b>{asset.ip_address}</b> — risk score {self._format_risk_value(profile.risk_score)}, "
                        f"level {profile.risk_level or '-'}, findings {profile.vulnerability_count}."
                    ),
                    subsection_style,
                )
            )

            if profile.ai_summary:
                sections.append(Paragraph(profile.ai_summary, body_style))

            if profile.recommendations:
                recommendations = "; ".join(
                    str(item)
                    for item in profile.recommendations[:3]
                    if str(item).strip()
                )
                if recommendations:
                    sections.append(
                        Paragraph(f"Recommendations: {recommendations}", body_style)
                    )

            if not asset_vulnerabilities:
                sections.append(Paragraph("No vulnerabilities recorded.", body_style))
                continue

            rows = [["Title", "Severity", "CVE", "Port", "Risk Score", "Remediation"]]
            for vulnerability in asset_vulnerabilities[:10]:
                rows.append(
                    [
                        Paragraph(vulnerability.title, body_style),
                        vulnerability.severity,
                        getattr(getattr(vulnerability, "cve", None), "cve_id", None)
                        or "-",
                        (
                            f"{vulnerability.port.port_number}/{vulnerability.port.protocol}"
                            if vulnerability.port
                            else "-"
                        ),
                        self._format_risk_value(vulnerability.armor_risk_score),
                        Paragraph(
                            vulnerability.remediation or "-",
                            body_style,
                        ),
                    ]
                )

            W_inner = (595.27 - 40 - 40) / 72 * inch
            table = Table(
                rows,
                repeatRows=1,
                colWidths=[
                    W_inner * 0.27,
                    W_inner * 0.10,
                    W_inner * 0.12,
                    W_inner * 0.11,
                    W_inner * 0.10,
                    W_inner * 0.30,
                ],
            )
            table.setStyle(self._doc_table_style())
            sections.append(table)

            if len(asset_vulnerabilities) > 10:
                sections.append(
                    Paragraph(
                        f"Showing 10 of {len(asset_vulnerabilities)} vulnerabilities for this asset.",
                        body_style,
                    )
                )

        return sections

    def _format_compliance_status(self, status: str | None) -> str:
        return {
            COMPLIANCE_STATUS_PASS: "Pass",
            COMPLIANCE_STATUS_FAIL: "Fail",
            COMPLIANCE_STATUS_PARTIAL: "Partial",
            COMPLIANCE_STATUS_NOT_APPLICABLE: "Not applicable",
            COMPLIANCE_STATUS_NEEDS_REVIEW: "Needs review",
        }.get(status or "", status or "-")

    def _compact_text(self, value) -> str:
        if value in (None, "", [], {}, ()):
            return "-"
        if isinstance(value, dict):
            return ", ".join(
                f"{key}={self._compact_text(val)}" for key, val in value.items()
            )
        if isinstance(value, (list, tuple, set)):
            return "; ".join(self._compact_text(item) for item in value)
        return str(value)

    def _build_framework_summary_rows(self, context: ReportContext) -> list[list[str]]:
        rows = [["Framework", "Pass", "Fail", "Partial", "N/A", "Review", "Score"]]
        if context.compliance_assessment is None:
            rows.append(["Compliance assessment", "-", "-", "-", "-", "-", "-"])
            return rows

        for framework in context.compliance_assessment.frameworks:
            summary = context.compliance_assessment.get_framework_summary(framework)
            if summary is None:
                continue
            rows.append(
                [
                    self._format_framework_name(framework),
                    str(summary["controls_pass"]),
                    str(summary["controls_fail"]),
                    str(summary["controls_partial"]),
                    str(summary["controls_not_applicable"]),
                    str(summary["controls_needs_review"]),
                    (
                        f"{summary['score']:.1f}"
                        if summary["score"] is not None
                        else "N/A"
                    ),
                ]
            )
        return rows

    def _build_compliance_control_rows(self, context: ReportContext) -> list[list[str]]:
        rows = [["Framework", "Control", "Status", "Category", "Severity", "Evidence"]]
        for result in context.compliance_results:
            evidence = result.evidence or {}
            evidence_summary = [
                self._compact_text(evidence.get("rationale")),
                self._compact_text(evidence.get("note")),
                self._compact_text(evidence.get("reason")),
            ]
            evidence_summary = " | ".join(
                item for item in evidence_summary if item != "-"
            )
            rows.append(
                [
                    self._format_framework_name(result.framework),
                    result.control_ref,
                    self._format_compliance_status(result.status),
                    result.category,
                    result.severity_impact or "-",
                    evidence_summary or "-",
                ]
            )
        return rows

    def _build_failed_control_rows(self, context: ReportContext) -> list[list[str]]:
        rows = [
            [
                "Framework",
                "Control",
                "Category",
                "Severity",
                "Affected Assets",
                "Root Cause",
            ]
        ]
        for result in context.compliance_results:
            if result.status != COMPLIANCE_STATUS_FAIL:
                continue
            evidence = result.evidence or {}
            affected_assets = evidence.get("affected_assets") or []
            root_cause = self._compact_text(
                evidence.get("note")
                or evidence.get("rationale")
                or evidence.get("reason")
            )
            if affected_assets:
                for asset in affected_assets[:10]:
                    rows.append(
                        [
                            self._format_framework_name(result.framework),
                            result.control_ref,
                            result.category,
                            result.severity_impact or "-",
                            asset.get("ip_address") or asset.get("ip") or "-",
                            root_cause,
                        ]
                    )
            else:
                rows.append(
                    [
                        self._format_framework_name(result.framework),
                        result.control_ref,
                        result.category,
                        result.severity_impact or "-",
                        "-",
                        root_cause,
                    ]
                )
        return rows

    def _build_violating_asset_rows(self, context: ReportContext) -> list[list[str]]:
        assets: dict[str, dict] = {}
        for result in context.compliance_results:
            if result.status != COMPLIANCE_STATUS_FAIL:
                continue
            evidence = result.evidence or {}
            affected_assets = evidence.get("affected_assets") or []
            for asset in affected_assets:
                ip_address = asset.get("ip_address") or asset.get("ip")
                if not ip_address:
                    continue
                asset_entry = assets.setdefault(
                    ip_address,
                    {
                        "hostname": asset.get("hostname") or "-",
                        "device_type": asset.get("device_type") or "-",
                        "os_name": asset.get("os_name") or "-",
                        "controls": set(),
                        "root_causes": set(),
                    },
                )
                asset_entry["controls"].add(result.control_ref)
                root_cause = self._compact_text(
                    evidence.get("note")
                    or evidence.get("rationale")
                    or evidence.get("reason")
                )
                if root_cause != "-":
                    asset_entry["root_causes"].add(root_cause)

        rows = [
            [
                "IP Address",
                "Hostname",
                "Device Type",
                "OS",
                "Controls Violated",
                "Root Causes",
            ]
        ]
        for ip_address, asset_entry in sorted(assets.items()):
            rows.append(
                [
                    ip_address,
                    asset_entry["hostname"],
                    asset_entry["device_type"],
                    asset_entry["os_name"],
                    ", ".join(sorted(asset_entry["controls"])) or "-",
                    " | ".join(sorted(asset_entry["root_causes"])) or "-",
                ]
            )
        return rows

    def _build_vulnerability_lookup(
        self, context: ReportContext
    ) -> dict[str, list[Vulnerability]]:
        lookup: dict[str, list[Vulnerability]] = defaultdict(list)
        for vulnerability in context.vulnerabilities:
            asset = vulnerability.asset_risk_profile.asset
            lookup[asset.ip_address].append(vulnerability)

        for vulnerabilities in lookup.values():
            vulnerabilities.sort(
                key=lambda item: (-float(item.armor_risk_score or 0.0), item.title)
            )
        return lookup

    def _format_asset_label(self, asset: dict) -> str:
        ip_address = asset.get("ip_address") or asset.get("ip") or "-"
        hostname = self._compact_text(asset.get("hostname"))
        if hostname != "-" and hostname != ip_address:
            return f"{ip_address} ({hostname})"
        return ip_address

    def _format_vulnerability_reference(self, vulnerability: Vulnerability) -> str:
        parts = [vulnerability.title]
        cve_id = getattr(getattr(vulnerability, "cve", None), "cve_id", None)
        if cve_id:
            parts.append(cve_id)
        if vulnerability.port:
            parts.append(
                f"{vulnerability.port.port_number}/{vulnerability.port.protocol}"
            )
        if vulnerability.is_kev:
            parts.append("KEV")
        return " | ".join(parts)

    def _format_finding_reference(self, finding: dict) -> str:
        parts = [self._compact_text(finding.get("vuln_type") or "finding")]
        if finding.get("port") is not None:
            port = str(finding["port"])
            service = self._compact_text(finding.get("service"))
            parts.append(f"port {port}" if service == "-" else f"port {port} {service}")
        if finding.get("cvss_score") is not None:
            parts.append(f"CVSS {float(finding['cvss_score']):.1f}")
        if finding.get("is_kev"):
            parts.append("KEV")
        return " | ".join(parts)

    def _match_vulnerability_for_finding(
        self, vulnerabilities: list[Vulnerability], finding: dict
    ) -> Vulnerability | None:
        target_type = finding.get("vuln_type")
        target_port = finding.get("port")
        target_service = (finding.get("service") or "").strip().lower()
        kev_only = bool(finding.get("is_kev"))

        candidates = []
        for vulnerability in vulnerabilities:
            if target_type and vulnerability.vuln_type != target_type:
                continue
            if kev_only and not vulnerability.is_kev:
                continue
            if target_port is not None and vulnerability.port is not None:
                if str(vulnerability.port.port_number) != str(target_port):
                    continue
            elif target_port is not None and vulnerability.port is None:
                continue
            if target_service and vulnerability.port is not None:
                vulnerability_service = (vulnerability.port.service or "").lower()
                if vulnerability_service and vulnerability_service != target_service:
                    continue
            candidates.append(vulnerability)

        return candidates[0] if candidates else None

    def _build_finding_summary(self, result: ComplianceResult, finding: dict) -> str:
        parts = [self._format_finding_reference(finding)]
        if result.severity_impact:
            parts.append(f"impact {result.severity_impact}")
        return " | ".join(parts)

    def _build_correlation_rows(self, context: ReportContext) -> list[list[str]]:
        rows = [
            ["Asset", "Vulnerability", "Compliance Failure", "Root Cause", "Reference"]
        ]
        seen: set[tuple[str, str, str, str]] = set()
        vulnerabilities_by_asset = self._build_vulnerability_lookup(context)

        for result in context.compliance_results:
            if result.status != COMPLIANCE_STATUS_FAIL:
                continue
            evidence = result.evidence or {}
            affected_assets = evidence.get("affected_assets") or []
            root_cause = self._compact_text(
                evidence.get("note")
                or evidence.get("rationale")
                or evidence.get("reason")
            )
            for asset in affected_assets:
                asset_ip = asset.get("ip_address") or asset.get("ip") or "-"
                asset_label = self._format_asset_label(asset)
                asset_vulnerabilities = vulnerabilities_by_asset.get(asset_ip, [])
                findings = asset.get("findings") or []

                if not findings:
                    key = (asset_ip, "-", result.control_ref, root_cause)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        [
                            asset_label,
                            "-",
                            f"{result.control_ref} ({self._format_framework_name(result.framework)})",
                            root_cause,
                            f"Evidence for {result.control_ref}",
                        ]
                    )
                    continue

                for finding in findings:
                    vulnerability = self._match_vulnerability_for_finding(
                        asset_vulnerabilities, finding
                    )
                    vulnerability_reference = (
                        self._format_vulnerability_reference(vulnerability)
                        if vulnerability is not None
                        else self._build_finding_summary(result, finding)
                    )
                    reference = (
                        vulnerability_reference
                        if vulnerability is not None
                        else f"{result.control_ref} finding"
                    )
                    key = (
                        asset_ip,
                        vulnerability_reference,
                        result.control_ref,
                        root_cause,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        [
                            asset_label,
                            vulnerability_reference,
                            f"{result.control_ref} ({self._format_framework_name(result.framework)})",
                            root_cause,
                            reference,
                        ]
                    )

        if len(rows) == 1:
            rows.append(
                [
                    "-",
                    "No asset-specific compliance correlations found.",
                    "-",
                    "-",
                    "-",
                ]
            )
        return rows

    def _build_evidence_appendix_rows(self, context: ReportContext) -> list[list[str]]:
        rows = [
            [
                "Control",
                "Status",
                "Evidence Summary",
                "Affected Assets",
                "Finding References",
            ]
        ]
        vulnerabilities_by_asset = self._build_vulnerability_lookup(context)
        for result in context.compliance_results:
            evidence = result.evidence or {}
            affected_assets = evidence.get("affected_assets") or []
            findings_summary: list[str] = []
            affected_asset_summary: list[str] = []
            for asset in affected_assets:
                asset_ip = asset.get("ip_address") or asset.get("ip") or "-"
                affected_asset_summary.append(self._format_asset_label(asset))
                for finding in (asset.get("findings") or [])[:2]:
                    vulnerability = self._match_vulnerability_for_finding(
                        vulnerabilities_by_asset.get(asset_ip, []), finding
                    )
                    findings_summary.append(
                        self._format_vulnerability_reference(vulnerability)
                        if vulnerability is not None
                        else self._format_finding_reference(finding)
                    )

            evidence_summary_parts = [
                self._compact_text(
                    evidence.get("rationale")
                    or evidence.get("note")
                    or evidence.get("reason")
                ),
            ]
            if evidence.get("severity_impact"):
                evidence_summary_parts.append(f"impact={evidence['severity_impact']}")
            if evidence.get("affected_asset_count") is not None:
                evidence_summary_parts.append(
                    f"assets={evidence['affected_asset_count']}"
                )
            if evidence.get("total_findings") is not None:
                evidence_summary_parts.append(f"findings={evidence['total_findings']}")
            rows.append(
                [
                    result.control_ref,
                    self._format_compliance_status(result.status),
                    " | ".join(item for item in evidence_summary_parts if item != "-"),
                    "; ".join(affected_asset_summary[:5]) or "0",
                    "; ".join(findings_summary[:5]) or "-",
                ]
            )
        return rows

    def _table_style(self) -> TableStyle:
        return TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(self._NAVY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                ("FONTSIZE", (0, 1), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("LINEBELOW", (0, 0), (-1, 0), 1.5, colors.HexColor(self._INDIGO)),
                ("GRID", (0, 1), (-1, -1), 0.35, colors.HexColor(self._BORDER)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor(self._ROW_ALT)],
                ),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )

    def _build_pdf_bytes(self, context: ReportContext) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=44,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()
        # A4 = 595.27pt wide; leftMargin=40, rightMargin=40 → usable = 515.27pt ≈ 7.157 inch
        W = (595.27 - 40 - 40) / 72 * inch  # exact usable content width

        # ── Typography ─────────────────────────────────────────────────────
        title_style = ParagraphStyle(
            "RTitle",
            parent=styles["Heading1"],
            fontSize=28,
            leading=34,
            alignment=TA_CENTER,
            textColor=colors.white,
            fontName="Helvetica-Bold",
            spaceAfter=2,
        )
        tagline_style = ParagraphStyle(
            "RTagline",
            parent=styles["BodyText"],
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#94a3b8"),
            spaceAfter=0,
        )
        meta_label_style = ParagraphStyle(
            "RMetaLabel",
            parent=styles["BodyText"],
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#64748b"),
            fontName="Helvetica-Bold",
            spaceAfter=1,
        )
        meta_value_style = ParagraphStyle(
            "RMetaValue",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#e2e8f0"),
        )
        section_style = ParagraphStyle(
            "RSection",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor(self._SLATE),
            spaceBefore=18,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        )
        body_style = ParagraphStyle(
            "RBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor(self._SLATE),
        )
        caption_style = ParagraphStyle(
            "RCaption",
            parent=styles["BodyText"],
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor(self._MUTED),
        )

        story: list = []

        # ══════════════════════════════════════════════════════════════════
        # PAGE 1 — BRANDING COVER (full page, no header/footer)
        # ══════════════════════════════════════════════════════════════════
        brand_logo_style = ParagraphStyle(
            "BrandLogo",
            parent=styles["BodyText"],
            fontSize=52,
            leading=60,
            alignment=TA_CENTER,
            textColor=colors.white,
            fontName="Helvetica-Bold",
            spaceAfter=0,
        )
        brand_tagline_style = ParagraphStyle(
            "BrandTagline",
            parent=styles["BodyText"],
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#94a3b8"),
            spaceAfter=0,
        )
        brand_sub_style = ParagraphStyle(
            "BrandSub",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#64748b"),
        )
        brand_report_title_style = ParagraphStyle(
            "BrandReportTitle",
            parent=styles["BodyText"],
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.white,
            fontName="Helvetica-Bold",
        )
        brand_meta_label_style = ParagraphStyle(
            "BrandMetaLabel",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#64748b"),
            fontName="Helvetica-Bold",
        )
        brand_meta_value_style = ParagraphStyle(
            "BrandMetaValue",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#cbd5e1"),
        )

        # Full-page dark cover table
        A4_H = 841.89  # A4 height in points
        cover_page_table = Table(
            [
                # Top spacer
                [Spacer(1, 1.6 * inch)],
                # Logo / product name
                [Paragraph("ARMOR", brand_logo_style)],
                [Spacer(1, 0.08 * inch)],
                # Accent rule
                [
                    HRFlowable(
                        width="30%",
                        thickness=2,
                        color=colors.HexColor(self._INDIGO),
                        hAlign="CENTER",
                        spaceAfter=6,
                    )
                ],
                [Spacer(1, 0.04 * inch)],
                # Tagline
                [
                    Paragraph(
                        "Cybersecurity Platform for Asset Management,<br/>Risk Assessment &amp; Compliance Inspection",
                        brand_tagline_style,
                    )
                ],
                [Spacer(1, 1.1 * inch)],
                # Divider
                [
                    HRFlowable(
                        width="60%",
                        thickness=0.5,
                        color=colors.HexColor("#334155"),
                        hAlign="CENTER",
                        spaceAfter=10,
                    )
                ],
                [Spacer(1, 0.18 * inch)],
                # Report type
                [Paragraph("Consolidated Security Report", brand_report_title_style)],
                [Spacer(1, 0.35 * inch)],
                # Meta grid row
                [
                    Table(
                        [
                            [
                                Paragraph("TARGET", brand_meta_label_style),
                                Paragraph("SCAN TYPE", brand_meta_label_style),
                                Paragraph("DATE GENERATED", brand_meta_label_style),
                            ],
                            [
                                Paragraph(
                                    context.scan.ip_range or "—", brand_meta_value_style
                                ),
                                Paragraph(
                                    context.scan.scan_type or "—",
                                    brand_meta_value_style,
                                ),
                                Paragraph(
                                    f"{timezone.now():%d %B %Y}", brand_meta_value_style
                                ),
                            ],
                        ],
                        colWidths=[W / 3] * 3,
                        style=TableStyle(
                            [
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("TOPPADDING", (0, 0), (-1, -1), 3),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                                (
                                    "LINEAFTER",
                                    (0, 0),
                                    (1, -1),
                                    0.5,
                                    colors.HexColor("#334155"),
                                ),
                            ]
                        ),
                    )
                ],
                [Spacer(1, 0.6 * inch)],
                # Bottom branding note
                [
                    Paragraph(
                        "CONFIDENTIAL — For authorised recipients only",
                        brand_sub_style,
                    )
                ],
                [Spacer(1, 0.3 * inch)],
            ],
            colWidths=[W],
        )
        cover_page_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(self._NAVY)),
                    ("LEFTPADDING", (0, 0), (-1, -1), 20),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 20),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(cover_page_table)
        story.append(PageBreak())

        # ══════════════════════════════════════════════════════════════════
        # PAGE 2 onwards — documentation starts here (page number = 2)
        # ══════════════════════════════════════════════════════════════════

        # Inline page template override: start numbering from 2 on the
        # second physical page by using a custom onPage callback.
        def _add_page_number(canvas, doc):
            # Page 1 is the branding cover — no number printed.
            if canvas.getPageNumber() == 1:
                return
            display_no = canvas.getPageNumber()
            canvas.saveState()
            # Footer left: product name
            canvas.setFont("Helvetica-Bold", 7)
            canvas.setFillColor(colors.HexColor(self._INDIGO))
            canvas.drawString(40, 22, "ARMOR")
            # Footer centre: scan range
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(colors.HexColor(self._MUTED))
            canvas.drawCentredString(
                A4[0] / 2,
                22,
                f"Consolidated Security Report — {context.scan.ip_range or ''}",
            )
            # Footer right: page number
            canvas.drawRightString(A4[0] - 40, 22, f"Page {display_no}")
            # Footer top hairline
            canvas.setStrokeColor(colors.HexColor(self._BORDER))
            canvas.setLineWidth(0.4)
            canvas.line(40, 34, A4[0] - 40, 34)
            canvas.restoreState()

        # Compact inline cover block reused as a slim running header on content pages
        cover_inner = [
            [Paragraph("ARMOR", title_style)],
            [Paragraph("Consolidated Security Report", tagline_style)],
            [Spacer(1, 0.06 * inch)],
            [
                HRFlowable(
                    width="100%",
                    thickness=1.5,
                    color=colors.HexColor(self._INDIGO),
                    hAlign="CENTER",
                    spaceAfter=10,
                )
            ],
        ]
        cover_table = Table(cover_inner, colWidths=[W])
        cover_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(self._NAVY)),
                    ("LEFTPADDING", (0, 0), (-1, -1), 16),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 16),
                    ("TOPPADDING", (0, 0), (0, 0), 20),
                    ("BOTTOMPADDING", (0, -1), (-1, -1), 16),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(cover_table)

        # Meta row — scan / generated / assessments
        meta_rows = [
            [
                Paragraph("IP RANGE", meta_label_style),
                Paragraph("SCAN TYPE", meta_label_style),
                Paragraph("GENERATED", meta_label_style),
            ],
            [
                Paragraph(context.scan.ip_range or "—", meta_value_style),
                Paragraph(context.scan.scan_type or "—", meta_value_style),
                Paragraph(f"{timezone.now():%Y-%m-%d %H:%M UTC}", meta_value_style),
            ],
        ]
        meta_table = Table(meta_rows, colWidths=[W / 3] * 3)
        meta_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1e293b")),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#334155")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEAFTER", (0, 0), (1, -1), 0.4, colors.HexColor("#334155")),
                ]
            )
        )
        story.append(meta_table)
        story.append(Spacer(1, 0.18 * inch))

        # ══════════════════════════════════════════════════════════════════
        # EXECUTIVE SUMMARY
        # ══════════════════════════════════════════════════════════════════
        story.append(Paragraph("Executive Summary", section_style))
        story.append(
            HRFlowable(
                width="100%",
                thickness=1.5,
                color=colors.HexColor(self._INDIGO),
                spaceAfter=10,
            )
        )

        # Stat cards — 4 across
        ra = context.risk_assessment
        ca = context.compliance_assessment

        total_vulns = ra.total_vulnerabilities if ra else 0
        critical_count = ra.critical_count if ra else 0
        risk_level = (ra.overall_risk_level or "N/A").upper() if ra else "N/A"
        risk_score = self._format_risk_value(ra.overall_risk_score) if ra else "N/A"

        fail_count = sum(
            1 for r in context.compliance_results if r.status == COMPLIANCE_STATUS_FAIL
        )
        pass_count = sum(
            1 for r in context.compliance_results if r.status == COMPLIANCE_STATUS_PASS
        )

        def _stat_card(label: str, value: str, sub: str, accent: str) -> list:
            return [
                Table(
                    [
                        [
                            Paragraph(
                                value,
                                ParagraphStyle(
                                    f"StatVal_{label}",
                                    parent=styles["BodyText"],
                                    fontSize=26,
                                    leading=30,
                                    fontName="Helvetica-Bold",
                                    textColor=colors.HexColor(accent),
                                    alignment=TA_CENTER,
                                ),
                            )
                        ],
                        [
                            Paragraph(
                                label,
                                ParagraphStyle(
                                    f"StatLbl_{label}",
                                    parent=styles["BodyText"],
                                    fontSize=8,
                                    leading=11,
                                    fontName="Helvetica-Bold",
                                    textColor=colors.HexColor(self._MUTED),
                                    alignment=TA_CENTER,
                                ),
                            )
                        ],
                        [
                            Paragraph(
                                sub,
                                ParagraphStyle(
                                    f"StatSub_{label}",
                                    parent=styles["BodyText"],
                                    fontSize=7,
                                    leading=9,
                                    textColor=colors.HexColor(self._MUTED),
                                    alignment=TA_CENTER,
                                ),
                            )
                        ],
                    ],
                    colWidths=[(W / 4) - 0.1 * inch],
                )
            ]

        card_data = [
            _stat_card(
                "ASSETS DISCOVERED",
                str(context.summary.total_assets),
                f"{context.summary.total_open_ports} open ports",
                self._INDIGO,
            ),
            _stat_card(
                "OVERALL RISK",
                risk_score,
                risk_level,
                {
                    "CRITICAL": self._RED,
                    "HIGH": self._ORANGE,
                    "MEDIUM": self._AMBER,
                    "LOW": self._BLUE,
                }.get(risk_level, self._MUTED),
            ),
            _stat_card(
                "VULNERABILITIES",
                str(total_vulns),
                f"{critical_count} critical",
                self._RED if critical_count else self._MUTED,
            ),
            _stat_card(
                "COMPLIANCE",
                f"{fail_count} fail",
                f"{pass_count} pass",
                self._RED if fail_count else self._GREEN,
            ),
        ]

        stat_row_table = Table(
            [card_data],
            colWidths=[W / 4] * 4,
        )
        stat_row_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor(self._BORDER)),
                    ("LINEAFTER", (0, 0), (2, -1), 0.5, colors.HexColor(self._BORDER)),
                    ("TOPPADDING", (0, 0), (-1, -1), 14),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(stat_row_table)
        story.append(Spacer(1, 0.1 * inch))

        # Narrative
        risk_phrase = (
            f"The risk assessment assigned an overall score of "
            f"<b>{risk_score}</b> at level <b>{risk_level}</b>, "
            f"with <b>{total_vulns}</b> vulnerabilities identified including "
            f"<b>{critical_count}</b> critical."
            if ra
            else "No completed risk assessment was found for this scan."
        )
        compliance_phrase = (
            f"The compliance assessment evaluated "
            f"<b>{', '.join(ca.frameworks) or 'no'}</b> framework(s), "
            f"producing <b>{fail_count}</b> failing and <b>{pass_count}</b> passing controls."
            if ca
            else "No completed compliance assessment was found."
        )
        story.append(
            Paragraph(
                f"The scan of <b>{context.scan.ip_range}</b> discovered "
                f"<b>{context.summary.total_assets}</b> assets with "
                f"<b>{context.summary.total_open_ports}</b> open ports in total. "
                f"{risk_phrase} {compliance_phrase} "
                "Full per-asset details follow in the Asset Reports section below.",
                body_style,
            )
        )

        # Assessment status mini-table
        story.append(Spacer(1, 0.1 * inch))
        status_rows = [
            [
                Paragraph("Assessment", meta_label_style),
                Paragraph("Status", meta_label_style),
            ]
        ]
        for label, available, note, ident in [
            (
                "Risk Assessment",
                context.summary.risk_available,
                context.risk_note,
                context.summary.risk_assessment_id,
            ),
            (
                "Compliance Assessment",
                context.summary.compliance_available,
                context.compliance_note,
                context.summary.compliance_assessment_id,
            ),
        ]:
            status_text = "Completed" if available else (note or "Not available")
            status_rows.append(
                [
                    Paragraph(label, body_style),
                    Paragraph(status_text, body_style),
                ]
            )

        status_tbl = Table(status_rows, colWidths=[W * 0.35, W * 0.65])
        status_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(self._NAVY)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ("LEADING", (0, 0), (-1, -1), 13),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(self._BORDER)),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor(self._ROW_ALT)],
                    ),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(status_tbl)

        if context.risk_note:
            story.append(Spacer(1, 0.05 * inch))
            story.append(Paragraph(context.risk_note, caption_style))
        if context.compliance_note:
            story.append(Spacer(1, 0.05 * inch))
            story.append(Paragraph(context.compliance_note, caption_style))

        # ══════════════════════════════════════════════════════════════════
        # ASSET REPORTS — one block per asset, all data unified
        # ══════════════════════════════════════════════════════════════════
        story.append(PageBreak())
        story.extend(
            self._build_asset_inventory_section(context, section_style, body_style)
        )

        doc.build(story, onLaterPages=_add_page_number, onFirstPage=_add_page_number)
        buffer.seek(0)
        return buffer.getvalue()

    def build_pdf_artifact(self) -> tuple[bytes, str]:
        context = self.build_context()
        return self._build_pdf_bytes(context), self._build_filename()

    def persist_report(self) -> ReportMetadata:
        metadata = self._get_existing_metadata()
        if metadata is None:
            metadata = ReportMetadata.objects.create(
                cache_key=self._cache_key(),
                scan=self.scan,
                risk_assessment=self.risk_assessment,
                compliance_assessment=self.compliance_assessment,
                generated_at=timezone.now(),
            )

        pdf_bytes, filename = self.build_pdf_artifact()
        metadata.generated_at = timezone.now()
        metadata.pdf_file.save(filename, ContentFile(pdf_bytes), save=True)
        return metadata

    def generate_response(self) -> FileResponse:
        metadata, created = self._ensure_metadata()
        needs_regeneration = (
            created or not metadata.pdf_file or not metadata.pdf_file.name
        )

        if needs_regeneration:
            metadata = self.persist_report()

        try:
            file_handle = metadata.pdf_file.open("rb")
        except FileNotFoundError:
            metadata = self.persist_report()
            file_handle = metadata.pdf_file.open("rb")
            needs_regeneration = True

        response = FileResponse(file_handle, content_type="application/pdf")
        response["Content-Disposition"] = (
            f'attachment; filename="{metadata.pdf_file.name.split("/")[-1]}"'
        )
        response["X-Report-Id"] = str(metadata.id)
        response["X-Report-Reused"] = "false" if needs_regeneration else "true"
        response["X-Scan-Id"] = str(metadata.scan_id)
        response["X-Risk-Assessment-Id"] = str(metadata.risk_assessment_id)
        response["X-Compliance-Assessment-Id"] = str(metadata.compliance_assessment_id)
        return response
    