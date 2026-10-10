from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

def generate_pdf_report(record, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=20))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12))
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = [Paragraph("MindProof Analysis Report", styles["TitleCenter"]), Spacer(1, 12)]
    story.append(Paragraph(f"<b>Topic:</b> {escape(str(record.get('topic','')))}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    for person_key, label in [("person1", "Person 1"), ("person2", "Person 2")]:
        p = record.get(person_key, {})
        story.append(Paragraph(label, styles["Heading2"]))
        story.append(Paragraph(f"<b>Argument:</b> {escape(str(p.get('argument','')))}", styles["BodyText"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"<b>Quality:</b> {p.get('quality','—')}/100 &nbsp;&nbsp; "
            f"<b>Stance:</b> {p.get('stance_label','—')}", styles["BodyText"]))
        story.append(Spacer(1, 6))

        prof = p.get("profile", {})
        if prof:
            data = [["Profile signal", "Value"]] + [[str(k), str(v)] for k,v in prof.items()]
            table = Table(data, colWidths=[220, 220])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#222222")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
                ("FONTSIZE", (0,0), (-1,-1), 8),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
            ]))
            story += [table, Spacer(1, 8)]

        story.append(Paragraph("<b>Explainability</b>", styles["Heading3"]))
        for x in p.get("explainability", []):
            story.append(Paragraph("• " + escape(str(x)), styles["Small"]))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Possible fallacy indicators</b>", styles["Heading3"]))
        fallacies = p.get("fallacies", [])
        if fallacies:
            for f in fallacies:
                story.append(Paragraph(f"• {escape(str(f['name']))} — {escape(str(f['confidence']))}", styles["Small"]))
        else:
            story.append(Paragraph("No rule-based indicator was detected.", styles["Small"]))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Improvement suggestions</b>", styles["Heading3"]))
        for s in p.get("suggestions", []):
            story.append(Paragraph("• " + escape(str(s)), styles["Small"]))
        story.append(Spacer(1, 14))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Neutral interpretation:</b> MindProof reports model predictions and linguistic indicators. "
        "A higher quality score does not mean an argument is factually true, and MindProof does not declare a winner.",
        styles["Small"]))
    doc.build(story)
    return str(output_path)
