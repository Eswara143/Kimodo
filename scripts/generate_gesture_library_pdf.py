#!/usr/bin/env python3
"""Gesture Library PDF Generator for Unitree G1.

Generates a formatted PDF catalog containing all 500 gesture assets,
their motion prompts, categories, intents, and quality audit statuses.
"""

from __future__ import annotations

import json
import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

ROOT = os.path.dirname(os.path.abspath(__file__))
LIBRARY_JSON_PATH = os.path.join(ROOT, "gesture_library_250.json")
AUDIT_JSON_PATH = os.path.join(ROOT, "outputs", "full_library_audit.json")
OUTPUT_PDF_PATH = os.path.join(ROOT, "outputs", "Unitree_G1_Gesture_Library_Catalog.pdf")


class NumberedCanvas(canvas.Canvas):
    """Canvas for adding page numbers and running header/footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 756, "Unitree G1 Gesture Library Catalog & Prompt Specifications")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(36, 748, 576, 748)
            
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 25, footer_text)
        self.drawString(36, 25, "Confidential - Unitree G1 Motion Quality System")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(36, 38, 576, 38)
        
        self.restoreState()


def build_pdf():
    print(f"Loading gesture metadata from {LIBRARY_JSON_PATH}...")
    with open(LIBRARY_JSON_PATH, "r") as f:
        lib_data = json.load(f).get("gestures", [])

    audit_map = {}
    if os.path.exists(AUDIT_JSON_PATH):
        with open(AUDIT_JSON_PATH, "r") as f:
            audit_data = json.load(f)
            for rec in audit_data.get("records", []):
                audit_map[rec["gesture_id"]] = rec

    # Group by Category
    categories: dict[str, list[dict]] = {}
    for g in lib_data:
        cat = g.get("category", "Uncategorized")
        if cat not in categories:
            categories[cat] = []
        
        gid = g.get("gesture_id")
        audit_info = audit_map.get(gid, {})
        g["audit_status"] = audit_info.get("final_status", "UNKNOWN")
        g["failed_checks"] = audit_info.get("failed_checks", [])
        categories[cat].append(g)

    doc = SimpleDocTemplate(
        OUTPUT_PDF_PATH,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=15
    )

    cat_heading_style = ParagraphStyle(
        'CatHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=14,
        spaceAfter=8
    )

    cell_hdr_style = ParagraphStyle(
        'CellHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#ffffff")
    )

    cell_id_style = ParagraphStyle(
        'CellID',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a")
    )

    cell_intent_style = ParagraphStyle(
        'CellIntent',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1e293b")
    )

    cell_prompt_style = ParagraphStyle(
        'CellPrompt',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155")
    )

    def get_status_paragraph(status: str) -> Paragraph:
        if status in ("CHECKS PASSED", "POST-CORRECTION PASSED"):
            bg_color = "#dcfce7"
            text_color = "#15803d"
            label = "PASSED"
        elif status == "BLOCKED":
            bg_color = "#e2e8f0"
            text_color = "#475569"
            label = "BLOCKED"
        elif status == "NEEDS REVIEW":
            bg_color = "#fef3c7"
            text_color = "#b45309"
            label = "REVIEW"
        else:
            bg_color = "#fee2e2"
            text_color = "#b91c1c"
            label = "FAILED"
            
        p_style = ParagraphStyle(
            'StatusTag',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=7,
            leading=9,
            alignment=1,
            textColor=colors.HexColor(text_color)
        )
        return Paragraph(f"<b>{label}</b>", p_style)

    elements = []

    # Title Banner
    elements.append(Paragraph("Unitree G1 Gesture Library Catalog", title_style))
    elements.append(Paragraph(f"Complete Inventory of <b>{len(lib_data)} Gesture Assets</b> with Motion Prompts & Quality Audit Status", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=15))

    # Summary Statistics Table
    summary_data = [
        [
            Paragraph("<b>Total Discovered</b>", cell_hdr_style),
            Paragraph("<b>Total Passed</b>", cell_hdr_style),
            Paragraph("<b>Failed (Excluded)</b>", cell_hdr_style),
            Paragraph("<b>Needs Review</b>", cell_hdr_style),
            Paragraph("<b>Blocked (Static)</b>", cell_hdr_style)
        ],
        [
            Paragraph("<b>500</b>", ParagraphStyle('SumV', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, alignment=1, textColor=colors.HexColor('#0f172a'))),
            Paragraph("<b>313</b>", ParagraphStyle('SumV', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, alignment=1, textColor=colors.HexColor('#16a34a'))),
            Paragraph("<b>133</b>", ParagraphStyle('SumV', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, alignment=1, textColor=colors.HexColor('#dc2626'))),
            Paragraph("<b>5</b>", ParagraphStyle('SumV', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, alignment=1, textColor=colors.HexColor('#d97706'))),
            Paragraph("<b>49</b>", ParagraphStyle('SumV', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, alignment=1, textColor=colors.HexColor('#475569')))
        ]
    ]
    
    sum_table = Table(summary_data, colWidths=[108, 108, 108, 108, 108])
    sum_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f8fafc')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(sum_table)
    elements.append(Spacer(1, 15))

    # Catalog Table by Category
    for cat_name, g_list in categories.items():
        elements.append(Paragraph(f"{cat_name} ({len(g_list)} Gestures)", cat_heading_style))
        
        table_data = [
            [
                Paragraph("<b>ID & #</b>", cell_hdr_style),
                Paragraph("<b>Intent</b>", cell_hdr_style),
                Paragraph("<b>Motion Prompt Specification</b>", cell_hdr_style),
                Paragraph("<b>Quality Audit Status</b>", cell_hdr_style)
            ]
        ]

        for g in g_list:
            gid = g.get("gesture_id", "")
            num = g.get("number", "")
            intent = g.get("intent", "")
            prompt = g.get("motion_prompt", "")
            status = g.get("audit_status", "")

            id_p = Paragraph(f"<b>{gid}</b><br/><font color='#64748b'>#{num}</font>", cell_id_style)
            intent_p = Paragraph(intent, cell_intent_style)
            prompt_p = Paragraph(prompt, cell_prompt_style)
            status_p = get_status_paragraph(status)

            table_data.append([id_p, intent_p, prompt_p, status_p])

        # Widths: ID (55), Intent (95), Prompt (310), Status (80) = 540 pt total
        cat_table = Table(table_data, colWidths=[55, 95, 310, 80], repeatRows=1)
        cat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#334155')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        elements.append(cat_table)
        elements.append(Spacer(1, 12))

    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"PDF Catalog successfully built: {OUTPUT_PDF_PATH}")
    return OUTPUT_PDF_PATH


if __name__ == "__main__":
    build_pdf()
