"""ReportLab PDF builders for diagnosis and full patient reports."""
import os
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, HRFlowable


def _abs_path(app_root, rel_path):
    if not rel_path:
        return None
    return os.path.normpath(os.path.join(app_root, rel_path.replace("/", os.sep)))


def _para(text, style, escape_text=True):
    s = str(text if text is not None else "")
    if escape_text:
        # Escape for XML but keep existing intended tags if we are careful
        # However, it's safer to just escape and then manually replace tags we trust
        # OR just not escape if we control the input.
        s = escape(s)
    
    s = s.replace("\n", "<br/>")
    return Paragraph(s, style)


def _get_header(app_root, styles, title="Diagnosis Report"):
    logo_path = os.path.join(app_root, "image.png")
    logo = None
    if os.path.exists(logo_path):
        try:
            logo = RLImage(logo_path, width=1.2 * inch, height=1.2 * inch)
        except:
            logo = None

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor("#2c3e50"),
        alignment=0, # Left
        spaceAfter=10
    )
    
    header_title = _para(title, title_style)
    # Don't escape here because we have tags
    clinic_info = _para("<b>EyeCare AI Diagnostics</b><br/>Advanced Retinal Analysis System", styles["Normal"], escape_text=False)
    
    data = []
    if logo:
        data.append([logo, [header_title, clinic_info]])
    else:
        data.append([[header_title, clinic_info]])
        
    header_table = Table(data, colWidths=[1.5 * inch, 5 * inch] if logo else [6.5 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ]))
    
    return [header_table, HRFlowable(width="100%", thickness=1, color=colors.HexColor("#bdc3c7"), spaceAfter=20)]


def _patient_info_section(patient, styles):
    data = [
        [Paragraph("<b>Patient Details</b>", styles["Heading3"]), ""],
        ["Name:", patient.name],
        ["Age:", str(patient.age)],
        ["Gender:", patient.gender],
        ["Blood Group:", patient.blood_group],
        ["Location:", patient.place],
    ]
    
    t = Table(data, colWidths=[1.5 * inch, 4.5 * inch])
    t.setStyle(TableStyle([
        ('SPAN', (0, 0), (1, 0)),
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#f8f9fa")),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.HexColor("#2c3e50")),
        ('BOTTOMPADDING', (0, 0), (1, 0), 8),
        ('TOPPADDING', (0, 0), (1, 0), 8),
        ('GRID', (0, 1), (-1, -1), 0.5, colors.HexColor("#ecf0f1")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    return [t, Spacer(1, 0.25 * inch)]


def _add_image_flowable(story, abs_path, label, styles):
    story.append(Paragraph(f"<b>{label}</b>", styles["Normal"]))
    story.append(Spacer(1, 0.05 * inch))
    
    if not abs_path or not os.path.isfile(abs_path):
        story.append(_para("(Image file not found)", styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))
        return

    try:
        from PIL import Image as PILImage
        with PILImage.open(abs_path) as im:
            w, h = im.size
        
        max_w, max_h = 2.8 * inch, 2.8 * inch
        scale = min(max_w/w, max_h/h, 1.0)
        story.append(RLImage(abs_path, width=w*scale, height=h*scale))
    except:
        story.append(_para("(Could not embed image)", styles["Normal"]))
    story.append(Spacer(1, 0.1 * inch))


def _diagnosis_section(diagnosis, app_root, styles):
    story = []
    story.append(Paragraph("<b>Medical Findings & AI Analysis</b>", styles["Heading3"]))
    story.append(Spacer(1, 0.1 * inch))

    # Format probability - if it's > 1 assume it's already percentage
    prob = diagnosis.probability or 0
    if prob > 1:
        conf_val = prob
    else:
        conf_val = prob * 100
    
    date_s = diagnosis.date.strftime("%Y-%m-%d %H:%M") if diagnosis.date else "N/A"
    doc_name = diagnosis.doctor.username if diagnosis.doctor else "N/A"

    summary_data = [
        ["Date of Analysis:", date_s],
        ["Primary Diagnosis:", Paragraph(f"<b>{diagnosis.disease}</b>", styles["Normal"])],
        ["AI Confidence Score:", f"{round(conf_val, 2)}%"],
        ["Consulting Doctor:", f"Dr. {doc_name if doc_name != 'admin' else 'System'}"],
    ]
    
    st = Table(summary_data, colWidths=[2 * inch, 4 * inch])
    st.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('BACKGROUND', (0, 1), (0, 1), colors.HexColor("#fff4f4") if "proliferative" in diagnosis.disease.lower() or "severe" in diagnosis.disease.lower() else colors.white),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(st)
    story.append(Spacer(1, 0.2 * inch))

    if diagnosis.notes:
        story.append(Paragraph("<b>Doctor's Clinical Notes:</b>", styles["Normal"]))
        story.append(_para(diagnosis.notes, styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

    # Side-by-side images
    img_data = []
    
    orig_path = _abs_path(app_root, diagnosis.image_path)
    mask_path = _abs_path(app_root, diagnosis.mask_path)
    
    # We build mini-stories for images to use in Table
    orig_cell = []
    _add_image_flowable(orig_cell, orig_path, "Fundus Image", styles)
    
    mask_cell = []
    _add_image_flowable(mask_cell, mask_path, "AI Attention Map (G-CAM)", styles)
    
    img_data = [[orig_cell, mask_cell]]
    img_table = Table(img_data, colWidths=[3.25 * inch, 3.25 * inch])
    img_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(img_table)
    
    # Add a disclaimer
    story.append(Spacer(1, 0.3 * inch))
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
    story.append(Paragraph("<i>Disclaimer: This report is generated by an Artificial Intelligence system. It is intended for clinical decision support and should be reviewed by a qualified ophthalmologist before any treatment plan is initiated.</i>", disclaimer_style))
    
    return story


def build_single_diagnosis_pdf(patient, diagnosis, app_root):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    
    story = []
    story.extend(_get_header(app_root, styles))
    story.extend(_patient_info_section(patient, styles))
    story.extend(_diagnosis_section(diagnosis, app_root, styles))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


def build_full_patient_report_pdf(patient, diagnoses_desc, app_root):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    
    story = []
    story.extend(_get_header(app_root, styles, "Full Medical History"))
    story.extend(_patient_info_section(patient, styles))
    
    if not diagnoses_desc:
        story.append(_para("No medical records found for this patient.", styles["Normal"]))
    else:
        for i, d in enumerate(diagnoses_desc):
            if i > 0:
                story.append(PageBreak())
                story.extend(_get_header(app_root, styles, f"Diagnosis History (Record {i+1})"))
            story.extend(_diagnosis_section(d, app_root, styles))
            
    doc.build(story)
    buffer.seek(0)
    return buffer
