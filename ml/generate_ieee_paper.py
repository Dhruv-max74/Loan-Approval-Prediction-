#!/usr/bin/env python3
"""
Python-docx generator for CreditWise IEEE Conference Paper (.docx).
Reads src/data/trainedModels.json and produces a two-column IEEE conference manuscript
for the 10,000-record hybrid consumer loan dataset.
"""

import json
import os
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ARTIFACT_PATH = "src/data/trainedModels.json"
OUT_DOCX = "public/CreditWise_IEEE_Conference_Paper.docx"


def set_two_columns(section, space_inches=0.25):
    sectPr = section._sectPr
    cols = sectPr.xpath("./w:cols")
    cols_el = cols[0] if cols else OxmlElement("w:cols")
    cols_el.set(qn("w:num"), "2")
    cols_el.set(qn("w:space"), str(int(space_inches * 1440)))
    cols_el.set(qn("w:equalWidth"), "1")
    if not cols:
        sectPr.append(cols_el)


def add_ieee_heading(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)


def add_ieee_body(doc, text, indent=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.18)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)


def main():
    with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    ds = data["datasetSummary"]
    dt = data["models"]["decisionTree"]
    lr = data["models"]["logisticRegression"]

    doc = Document()
    sec1 = doc.sections[0]
    sec1.page_width = Inches(8.5)
    sec1.page_height = Inches(11.0)
    sec1.top_margin = Inches(0.75)
    sec1.bottom_margin = Inches(1.0)
    sec1.left_margin = Inches(0.625)
    sec1.right_margin = Inches(0.625)

    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rt = t.add_run(
        "CreditWise: Comparative Credit Risk Assessment Using Interpretable Decision Tree and Logistic Regression Classifiers"
    )
    rt.bold = True
    rt.font.name = "Times New Roman"
    rt.font.size = Pt(22)

    a = doc.add_paragraph()
    a.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ra = a.add_run("CreditWise Analytics Research Group\nDepartment of Computer Science & Financial Engineering")
    ra.italic = True
    ra.font.name = "Times New Roman"
    ra.font.size = Pt(10)

    sec2 = doc.add_section(WD_SECTION.CONTINUOUS)
    sec2.top_margin = Inches(0.75)
    sec2.bottom_margin = Inches(1.0)
    sec2.left_margin = Inches(0.625)
    sec2.right_margin = Inches(0.625)
    set_two_columns(sec2, 0.25)

    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r1 = p_abs.add_run("Abstract—")
    r1.bold = True
    r1.italic = True
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(9)
    r2 = p_abs.add_run(
        f"This study evaluates CART Decision Tree ({dt['metrics']['accuracy']*100:.2f}% accuracy, "
        f"{dt['metrics']['rocAuc']:.4f} ROC-AUC) and L2-regularized Logistic Regression "
        f"({lr['metrics']['accuracy']*100:.2f}% accuracy, {lr['metrics']['rocAuc']:.4f} ROC-AUC) "
        f"across {ds['totalRecords']} hybrid consumer loan applications ({ds['approvedCount']} approved, {ds['rejectedCount']} rejected)."
    )
    r2.bold = True
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(9)

    add_ieee_heading(doc, "I. INTRODUCTION")
    add_ieee_body(doc, "Automated credit underwriting requires accurate and transparent machine learning models.", indent=False)
    add_ieee_heading(doc, "II. EXPERIMENTAL RESULTS")
    add_ieee_body(doc, f"Decision Tree achieved {dt['metrics']['accuracy']*100:.2f}% accuracy vs. {lr['metrics']['accuracy']*100:.2f}% for Logistic Regression.", indent=False)

    os.makedirs(os.path.dirname(OUT_DOCX), exist_ok=True)
    doc.save(OUT_DOCX)
    print("Saved", OUT_DOCX)


if __name__ == "__main__":
    main()
