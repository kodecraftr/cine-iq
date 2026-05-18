from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "reports" / "cine_iq_report.md"
REPORT_PDF = ROOT / "reports" / "cine_iq_report.pdf"


def clean_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def parse_markdown(md: str):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyTight",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=12.5,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Code"],
            fontSize=8,
            leading=10,
            backColor=colors.HexColor("#f4f6f8"),
            borderPadding=5,
            spaceAfter=8,
        )
    )

    story = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue

        if line.startswith("# "):
            story.append(Paragraph(clean_inline(line[2:]), styles["Title"]))
            story.append(Spacer(1, 0.08 * inch))
            i += 1
            continue

        if line.startswith("## "):
            story.append(Spacer(1, 0.06 * inch))
            story.append(Paragraph(clean_inline(line[3:]), styles["Heading2"]))
            i += 1
            continue

        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            story.append(Paragraph("<br/>".join(clean_inline(x) for x in code), styles["CodeBlock"]))
            i += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|\s*-", lines[i]):
                    table_lines.append(lines[i])
                i += 1
            data = []
            for row in table_lines:
                cells = [Paragraph(clean_inline(c.strip()), styles["BodyTight"]) for c in row.strip("|").split("|")]
                data.append(cells)
            if data:
                table = Table(data, repeatRows=1, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3")),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#aab4c0")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 5),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 0.08 * inch))
            continue

        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "```")):
            paragraph.append(lines[i].strip())
            i += 1
        story.append(Paragraph(clean_inline(" ".join(paragraph)), styles["BodyTight"]))

    return story


def main():
    REPORT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title="Cine IQ Project Report",
    )
    story = parse_markdown(REPORT_MD.read_text(encoding="utf-8"))
    doc.build(story)
    print(REPORT_PDF)


if __name__ == "__main__":
    main()

