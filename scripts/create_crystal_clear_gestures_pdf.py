#!/usr/bin/env python3
"""Crystal Clear Gesture Names & Motion Prompts PDF Generator for Unitree G1.

Creates an ultra-readable, beautifully structured PDF catalog focused exclusively on:
  - Gesture Number & Name (e.g., #1. Hello)
  - Category
  - What the Robot Does (Full Motion Prompt Description in plain text)
"""

from __future__ import annotations

import json
import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

ROOT = os.path.dirname(os.path.abspath(__file__))
LIBRARY_JSON_PATH = os.path.join(ROOT, "gesture_library_250.json")
OUTPUT_PDF_PATH = os.path.join(ROOT, "outputs", "Gesture_Prompts_Clear_Catalog.pdf")
SHORT_PDF_PATH = os.path.join(ROOT, "outputs", "Gesture_Prompts_List.pdf")


class CrystalClearCanvas(canvas.Canvas):

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
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 9)
            self.setFillColor(colors.HexColor("#2563eb"))
            self.drawString(36, 756, "UNITREE G1 ROBOT")
            self.setFont("Helvetica", 9)
            self.setFillColor(colors.HexColor("#475569"))
            self.drawString(135, 756, "— Gesture Names & Motion Prompt Reference")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.75)
            self.line(36, 748, 576, 748)
            
        # Footer (All Pages)
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(36, 25, "Unitree G1 Gesture Library Catalog (500 Gestures)")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 25, page_str)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.75)
        self.line(36, 38, 576, 38)
        
        self.restoreState()


def build_clear_pdf():
    print(f"Loading gesture data from {LIBRARY_JSON_PATH}...")
    with open(LIBRARY_JSON_PATH, "r") as f:
        lib_data = json.load(f).get("gestures", [])

    # Group gestures by category
    categories: dict[str, list[dict]] = {}
    for g in lib_data:
        cat = g.get("category", "General Gestures")
        if cat not in categories:
            categories[cat] = []
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

    # Typography & Styles
    doc_title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4
    )

    doc_sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14
    )

    cat_title_style = ParagraphStyle(
        'CatTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=14,
        spaceAfter=8
    )

    cell_hdr_style = ParagraphStyle(
        'CellHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor("#ffffff")
    )

    g_num_style = ParagraphStyle(
        'GNum',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#1e293b")
    )

    g_name_style = ParagraphStyle(
        'GName',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0f172a")
    )

    g_prompt_style = ParagraphStyle(
        'GPrompt',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b")
    )

    elements = []

    # Main Title
    elements.append(Paragraph("Unitree G1 Robot Gesture Catalog", doc_title_style))
    elements.append(Paragraph("Clear reference list matching every <b>Gesture Name</b> directly with its <b>Motion Prompt</b>", doc_sub_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2563eb"), spaceAfter=14))

    # Render Category Tables
    global_idx = 0
    for cat_name, g_list in categories.items():
        elements.append(Paragraph(f"📁 Category: {cat_name} ({len(g_list)} Gestures)", cat_title_style))
        
        table_data = [
            [
                Paragraph("<b>#</b>", cell_hdr_style),
                Paragraph("<b>Gesture Name (Intent)</b>", cell_hdr_style),
                Paragraph("<b>Motion Prompt (What the Robot Does)</b>", cell_hdr_style)
            ]
        ]

        for g in g_list:
            global_idx += 1
            num = g.get("number", global_idx)
            gid = g.get("gesture_id", "")
            intent = g.get("intent", "")
            prompt = g.get("motion_prompt", "")

            num_cell = Paragraph(f"<b>#{num}</b><br/><font size=7 color='#64748b'>{gid}</font>", g_num_style)
            name_cell = Paragraph(f"<b>{intent}</b>", g_name_style)
            prompt_cell = Paragraph(f"{prompt}", g_prompt_style)

            table_data.append([num_cell, name_cell, prompt_cell])

        # Widths: # (45), Name (135), Prompt (360) = 540 pt total
        cat_table = Table(table_data, colWidths=[45, 135, 360], repeatRows=1)
        cat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))

        elements.append(cat_table)
        elements.append(Spacer(1, 14))

    doc.build(elements, canvasmaker=CrystalClearCanvas)
    
    # Also save to Gesture_Prompts_List.pdf so both paths match
    import shutil
    shutil.copyfile(OUTPUT_PDF_PATH, SHORT_PDF_PATH)
    
    print(f"Crystal Clear PDF built successfully:\n  - {OUTPUT_PDF_PATH}\n  - {SHORT_PDF_PATH}")
    return OUTPUT_PDF_PATH


if __name__ == "__main__":
    build_clear_pdf()
