#!/usr/bin/env python3
"""Gesture Prompts List PDF Generator.

Generates a clean PDF catalog containing every gesture name/intent (e.g., Hello, Welcome),
its ID, category, and full motion prompt description.
"""

from __future__ import annotations

import json
import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

ROOT = os.path.dirname(os.path.abspath(__file__))
LIBRARY_JSON_PATH = os.path.join(ROOT, "gesture_library_250.json")
OUTPUT_PDF_PATH = os.path.join(ROOT, "outputs", "Gesture_Prompts_List.pdf")


class NumberedCanvas(canvas.Canvas):

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
            self.drawString(36, 756, "Unitree G1 Gesture Names & Motion Prompts List")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 748, 576, 748)
            
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 25, footer_text)
        self.drawString(36, 25, "Unitree G1 Gesture Library - Complete Prompt Reference")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 38, 576, 38)
        
        self.restoreState()


def build_prompts_pdf():
    print(f"Loading gestures from {LIBRARY_JSON_PATH}...")
    with open(LIBRARY_JSON_PATH, "r") as f:
        lib_data = json.load(f).get("gestures", [])

    # Group by category
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

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )

    cat_heading_style = ParagraphStyle(
        'CatHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6
    )

    cell_hdr_style = ParagraphStyle(
        'CellHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#ffffff")
    )

    cell_num_style = ParagraphStyle(
        'CellNum',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        alignment=1,
        textColor=colors.HexColor("#475569")
    )

    cell_name_style = ParagraphStyle(
        'CellName',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )

    cell_prompt_style = ParagraphStyle(
        'CellPrompt',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155")
    )

    elements = []

    # Document Header
    elements.append(Paragraph("Unitree G1 Gesture Names & Prompts List", title_style))
    elements.append(Paragraph(f"Complete Reference Catalog of all <b>{len(lib_data)} Gestures</b> with their corresponding Motion Prompts", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=14))

    # Build Tables for Each Category
    total_count = 0
    for cat_name, g_list in categories.items():
        elements.append(Paragraph(f"{cat_name} ({len(g_list)} Gestures)", cat_heading_style))
        
        table_data = [
            [
                Paragraph("<b>#</b>", cell_hdr_style),
                Paragraph("<b>ID</b>", cell_hdr_style),
                Paragraph("<b>Gesture Name / Intent</b>", cell_hdr_style),
                Paragraph("<b>Motion Prompt Description</b>", cell_hdr_style)
            ]
        ]

        for g in g_list:
            total_count += 1
            num = g.get("number", total_count)
            gid = g.get("gesture_id", "")
            intent = g.get("intent", "")
            prompt = g.get("motion_prompt", "")

            num_p = Paragraph(f"{num}", cell_num_style)
            id_p = Paragraph(f"<b>{gid}</b>", ParagraphStyle('CID', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, textColor=colors.HexColor('#2563eb')))
            name_p = Paragraph(f"<b>{intent}</b>", cell_name_style)
            prompt_p = Paragraph(prompt, cell_prompt_style)

            table_data.append([num_p, id_p, name_p, prompt_p])

        # Widths: # (30), ID (55), Name (115), Prompt (340) = 540 pt
        cat_table = Table(table_data, colWidths=[30, 55, 115, 340], repeatRows=1)
        cat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        elements.append(cat_table)
        elements.append(Spacer(1, 10))

    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"Prompts PDF List built successfully: {OUTPUT_PDF_PATH}")
    return OUTPUT_PDF_PATH


if __name__ == "__main__":
    build_prompts_pdf()
