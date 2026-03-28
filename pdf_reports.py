"""ReportLab PDF builders for diagnosis and full patient reports."""
import os
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer


def _abs_path(app_root, rel_path):
    if not rel_path:
        return None
    return os.path.normpath(os.path.join(app_root, rel_path.replace("/", os.sep)))


def _para(text, style):
    s = escape(str(text if text is not None else ""))
    s = s.replace("\n", "<br/>")
    return Paragraph(s, style)


def _add_image_flowable(story, abs_path, styles, max_w=3.8 * inch, max_h=2.8 * inch):
    if not abs_path or not os.path.isfile(abs_path):
        story.append(_para("(Image file not found)", styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))
        return
    try:
        from PIL import Image as PILImage

        with PILImage.open(abs_path) as im:
            w, h = im.size
        if w <= 0 or h <= 0:
            raise ValueError("invalid image size")
        scale = min(max_w / w, max_h / h, 1.0)
        rw, rh = w * scale, h * scale
        story.append(RLImage(abs_path, width=rw, height=rh))
    except Exception:
        story.append(_para("(Could not embed image)", styles["Normal"]))
    story.append(Spacer(1, 0.1 * inch))


def _patient_info_story(patient, styles):
    blocks = [
        _para(f"Name: {patient.name}", styles["Normal"]),
        _para(f"Age: {patient.age}", styles["Normal"]),
        _para(f"Gender: {patient.gender}", styles["Normal"]),
        _para(f"Blood Group: {patient.blood_group}", styles["Normal"]),
        _para(f"Place: {patient.place}", styles["Normal"]),
        Spacer(1, 0.15 * inch),
    ]
    return blocks


def _diagnosis_block_story(diagnosis, app_root, styles):
    story = []
    date_s = diagnosis.date.strftime("%Y-%m-%d %H:%M") if diagnosis.date else "N/A"
    story.append(Paragraph(f"<b>{escape('Date: ' + date_s)}</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(_para(f"Disease: {diagnosis.disease}", styles["Normal"]))
    conf = round((diagnosis.probability or 0) * 100, 2)
    story.append(_para(f"Confidence: {conf}%", styles["Normal"]))
    doc_name = diagnosis.doctor.username if diagnosis.doctor else "N/A"
    story.append(_para(f"Doctor: {doc_name}", styles["Normal"]))
    if diagnosis.notes:
        story.append(Spacer(1, 0.06 * inch))
        story.append(_para("Doctor notes:", styles["Normal"]))
        story.append(_para(diagnosis.notes, styles["Normal"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(_para("Original image:", styles["Normal"]))
    _add_image_flowable(story, _abs_path(app_root, diagnosis.image_path), styles)
    story.append(_para("Segmentation mask:", styles["Normal"]))
    _add_image_flowable(story, _abs_path(app_root, diagnosis.mask_path), styles)
    return story


def build_single_diagnosis_pdf(patient, diagnosis, app_root):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Diagnosis Report", styles["Title"]),
        Spacer(1, 0.2 * inch),
        Paragraph("Patient Information", styles["Heading1"]),
        Spacer(1, 0.1 * inch),
    ]
    story.extend(_patient_info_story(patient, styles))
    story.append(Paragraph("AI Diagnosis", styles["Heading1"]))
    story.extend(_diagnosis_block_story(diagnosis, app_root, styles))
    doc.build(story)
    buffer.seek(0)
    return buffer


def build_full_patient_report_pdf(patient, diagnoses_desc, app_root):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Complete Patient Medical Report", styles["Title"]),
        Spacer(1, 0.25 * inch),
        Paragraph("Section 1: Patient Information", styles["Heading1"]),
        Spacer(1, 0.1 * inch),
    ]
    story.extend(_patient_info_story(patient, styles))
    story.append(Paragraph("Section 2: Diagnosis History", styles["Heading1"]))
    story.append(Spacer(1, 0.1 * inch))
    if not diagnoses_desc:
        story.append(_para("No diagnosis records on file.", styles["Normal"]))
    for i, d in enumerate(diagnoses_desc):
        if i:
            story.append(PageBreak())
        story.extend(_diagnosis_block_story(d, app_root, styles))
    doc.build(story)
    buffer.seek(0)
    return buffer
