"""Render editable Markdown as an academic A4 PDF with ReportLab.

Supported blocks: headings, paragraphs, Markdown tables, lists, local figures,
quotes, fenced/indented formulas, and explicit <!-- pagebreak --> markers.
This script does not invent report content or calculate research results.
Use the bundled Python runtime providing reportlab, pypdf, and Pillow.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    HRFlowable,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)

BLUE = colors.HexColor("#344B63")
INK = colors.HexColor("#202B35")
MUTED = colors.HexColor("#5C6976")
LIGHT = colors.HexColor("#EAF0F5")
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGINS = {"leftMargin": 50, "rightMargin": 50, "topMargin": 59, "bottomMargin": 51}
# SimpleDocTemplate's frame reserves six points of internal padding on each side.
CONTENT_WIDTH = PAGE_WIDTH - MARGINS["leftMargin"] - MARGINS["rightMargin"] - 12


@dataclass
class Block:
    kind: str
    text: str = ""
    level: int = 0
    rows: list[list[str]] = field(default_factory=list)
    alignments: list[str] = field(default_factory=list)
    marker: str = ""


def table_cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells, current, quoted, escaped = [], [], False, False
    for character in line:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "`":
            quoted = not quoted
            current.append(character)
        elif character == "|" and not quoted:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def table_separator(line: str) -> bool:
    return "|" in line and all(re.fullmatch(r":?-{3,}:?", item) for item in table_cells(line))


def parse_markdown(source: str) -> list[Block]:
    lines = source.replace("\r\n", "\n").split("\n")
    blocks, index = [], 0
    heading = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    list_item = re.compile(r"^ {0,3}([-+*]|\d+[.)])\s+(.+)$")

    def starts_block(position):
        line = lines[position]
        stripped = line.strip()
        return (
            not stripped
            or heading.match(line)
            or list_item.match(line)
            or stripped.startswith(("```", "~~~", "![", ">", "<!-- pagebreak"))
            or line.startswith(("    ", "\t"))
            or re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", stripped)
            or (position + 1 < len(lines) and table_separator(lines[position + 1]))
        )

    while index < len(lines):
        line, stripped = lines[index], lines[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped == "<!-- pagebreak -->":
            blocks.append(Block("pagebreak"))
            index += 1
        elif re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", stripped):
            blocks.append(Block("rule"))
            index += 1
        elif stripped.startswith(("```", "~~~")):
            fence, text = stripped[:3], []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(fence):
                text.append(lines[index])
                index += 1
            if index == len(lines):
                raise ValueError("Unclosed fenced formula/code block")
            blocks.append(Block("code", "\n".join(text)))
            index += 1
        elif line.startswith(("    ", "\t")):
            text = []
            while index < len(lines):
                current = lines[index]
                if current.startswith("    "):
                    text.append(current[4:])
                elif current.startswith("\t"):
                    text.append(current[1:])
                elif (
                    not current.strip()
                    and index + 1 < len(lines)
                    and lines[index + 1].startswith(("    ", "\t"))
                ):
                    text.append("")
                else:
                    break
                index += 1
            blocks.append(Block("code", "\n".join(text)))
        elif heading.match(line):
            match = heading.match(line)
            blocks.append(Block("heading", match[2], len(match[1])))
            index += 1
        elif index + 1 < len(lines) and table_separator(lines[index + 1]):
            header, separators = table_cells(line), table_cells(lines[index + 1])
            if len(header) != len(separators):
                raise ValueError(f"Table header/separator width mismatch near line {index + 1}")
            rows = [header]
            index += 2
            while index < len(lines) and lines[index].strip() and "|" in lines[index]:
                row = table_cells(lines[index])
                if len(row) != len(header):
                    raise ValueError(f"Table column count mismatch at line {index + 1}")
                rows.append(row)
                index += 1
            alignments = [
                "center"
                if item.startswith(":") and item.endswith(":")
                else "right"
                if item.endswith(":")
                else "left"
                for item in separators
            ]
            blocks.append(Block("table", rows=rows, alignments=alignments))
        elif stripped.startswith("!["):
            match = re.fullmatch(r"!\[([^]]*)\]\((.+)\)", stripped)
            if not match:
                raise ValueError(f"Figure must occupy its own Markdown line: {stripped}")
            blocks.append(Block("image", match[2], marker=match[1]))
            index += 1
        elif list_item.match(line):
            match = list_item.match(line)
            marker, pieces = match[1], [match[2]]
            index += 1
            while index < len(lines) and lines[index].strip() and not starts_block(index):
                pieces.append(lines[index].strip())
                index += 1
            blocks.append(Block("list", " ".join(pieces), marker=marker))
        elif stripped.startswith(">"):
            pieces = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                pieces.append(lines[index].strip()[1:].strip())
                index += 1
            blocks.append(Block("quote", " ".join(pieces)))
        else:
            pieces = [stripped]
            index += 1
            while index < len(lines) and not starts_block(index):
                pieces.append(lines[index].strip())
                index += 1
            blocks.append(Block("paragraph", " ".join(pieces)))
    return blocks


def register_fonts():
    regular, bold, italic, bold_italic = (
        "Times-Roman",
        "Times-Bold",
        "Times-Italic",
        "Times-BoldItalic",
    )
    directory = Path("C:/Windows/Fonts")
    files = ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]
    names = ["CourseTimes", "CourseTimesBold", "CourseTimesItalic", "CourseTimesBoldItalic"]
    if all((directory / filename).is_file() for filename in files):
        for name, filename in zip(names, files, strict=True):
            pdfmetrics.registerFont(TTFont(name, str(directory / filename)))
        regular, bold, italic, bold_italic = names
        pdfmetrics.registerFontFamily(
            regular, normal=regular, bold=bold, italic=italic, boldItalic=bold_italic
        )
    cjk = directory / "msyh.ttc"
    if cjk.is_file():
        pdfmetrics.registerFont(TTFont("CourseCJK", str(cjk), subfontIndex=0))
        pdfmetrics.registerFontFamily(
            "CourseCJK",
            normal="CourseCJK",
            bold="CourseCJK",
            italic="CourseCJK",
            boldItalic="CourseCJK",
        )
    return {"regular": regular, "bold": bold, "italic": italic, "bold_italic": bold_italic}


INLINE = re.compile(
    r"(`+)(.+?)\1|\[([^]]+)\]\(([^)]+)\)|\*\*(.+?)\*\*|__(.+?)__|"
    r"(?<!\*)\*([^*]+)\*(?!\*)|(?<!\w)_([^_]+)_(?!\w)"
)


def escaped_text(text):
    escaped = html.escape(text, quote=False)
    if "CourseCJK" in pdfmetrics.getRegisteredFontNames():
        escaped = re.sub(
            r"[\u2e80-\u9fff\uf900-\ufaff]+",
            lambda match: f'<font name="CourseCJK">{match[0]}</font>',
            escaped,
        )
    return escaped


def inline(text):
    result, position = [], 0
    for match in INLINE.finditer(text):
        result.append(escaped_text(text[position : match.start()]))
        if match[1]:
            result.append(f'<font name="Courier" size="9">{escaped_text(match[2])}</font>')
        elif match[3] is not None:
            label, target = inline(match[3]), match[4].strip()
            if target.startswith(("https://", "http://", "mailto:", "#")):
                result.append(
                    f'<link href="{html.escape(target, quote=True)}" color="#345D81">{label}</link>'
                )
            else:
                result.append(label)
        elif match[5] is not None or match[6] is not None:
            result.append(f"<b>{inline(match[5] or match[6])}</b>")
        else:
            result.append(f"<i>{inline(match[7] or match[8])}</i>")
        position = match.end()
    result.append(escaped_text(text[position:]))
    return "".join(result)


def plain(text):
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    return text.replace("**", "").replace("`", "").replace("__", "")


def make_styles(fonts):
    body = ParagraphStyle(
        "body",
        fontName=fonts["regular"],
        fontSize=10.7,
        leading=14.1,
        textColor=INK,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        allowWidows=0,
        allowOrphans=0,
        splitLongWords=1,
    )
    styles = {"body": body}
    for level, size in [(1, 18), (2, 14), (3, 11.6), (4, 10.8), (5, 10.7), (6, 10.7)]:
        styles[f"h{level}"] = ParagraphStyle(
            f"h{level}",
            parent=body,
            fontName=fonts["bold"],
            fontSize=size,
            leading=size * 1.2,
            textColor=BLUE,
            alignment=TA_LEFT,
            spaceBefore=12 if level > 1 else 5,
            spaceAfter=7,
            keepWithNext=True,
        )
    styles["caption"] = ParagraphStyle(
        "caption",
        parent=body,
        fontName=fonts["italic"],
        fontSize=9.2,
        leading=12,
        textColor=MUTED,
        alignment=TA_LEFT,
        spaceBefore=5,
        spaceAfter=10,
    )
    styles["table_caption"] = ParagraphStyle(
        "table_caption",
        parent=styles["caption"],
        keepWithNext=False,
        spaceAfter=5,
    )
    styles["list"] = ParagraphStyle(
        "list",
        parent=body,
        alignment=TA_LEFT,
        leftIndent=15,
        firstLineIndent=0,
        bulletIndent=0,
        bulletFontName=fonts["regular"],
        bulletFontSize=10.2,
        spaceAfter=4,
    )
    styles["quote"] = ParagraphStyle(
        "quote",
        parent=body,
        alignment=TA_LEFT,
        leftIndent=12,
        rightIndent=10,
        textColor=MUTED,
        fontName=fonts["italic"],
        spaceBefore=3,
    )
    styles["code"] = ParagraphStyle(
        "code",
        fontName="Courier",
        fontSize=8.8,
        leading=12.2,
        leftIndent=14,
        rightIndent=8,
        textColor=INK,
        spaceBefore=4,
        spaceAfter=9,
    )
    return styles


def markdown_table(block, fonts):
    columns = len(block.rows[0])
    if columns > 12:
        raise ValueError("Table has more than 12 columns; split it in the editable Markdown")
    size = 9.0 if columns <= 6 else 8.3
    weights = [
        max(8, min(45, max(len(plain(row[column])) for row in block.rows)))
        for column in range(columns)
    ]
    minimums = []
    for column in range(columns):
        longest = max(
            (
                pdfmetrics.stringWidth(
                    word, fonts["bold"] if index == 0 else fonts["regular"], size
                )
                for index, row in enumerate(block.rows)
                for word in plain(row[column]).split()
            ),
            default=20,
        )
        minimums.append(max(30, min(100, longest + 10)))
    if sum(minimums) >= CONTENT_WIDTH:
        widths = [CONTENT_WIDTH * minimum / sum(minimums) for minimum in minimums]
    else:
        remaining = CONTENT_WIDTH - sum(minimums)
        widths = [
            minimum + remaining * weight / sum(weights)
            for minimum, weight in zip(minimums, weights, strict=True)
        ]
    rows = []
    for index, cells in enumerate(block.rows):
        row = []
        for column, cell in enumerate(cells):
            alignment = {"left": TA_LEFT, "right": TA_RIGHT, "center": TA_CENTER}[
                block.alignments[column]
            ]
            style = ParagraphStyle(
                f"cell-{index}-{column}",
                fontName=fonts["bold"] if index == 0 else fonts["regular"],
                fontSize=size,
                leading=size * 1.28,
                textColor=BLUE if index == 0 else INK,
                alignment=alignment,
                splitLongWords=1,
                allowWidows=1,
                allowOrphans=1,
            )
            row.append(Paragraph(inline(cell), style))
        rows.append(row)
    table = LongTable(
        rows,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
        splitInRow=1,
        spaceBefore=3,
        spaceAfter=10,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
                ("LINEABOVE", (0, 0), (-1, 0), 0.65, BLUE),
                ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.HexColor("#ADBBC8")),
                ("LINEBELOW", (0, -1), (-1, -1), 0.45, colors.HexColor("#ADBBC8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


class ReportDocument(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, "bookmark_name"):
            self.canv.bookmarkPage(flowable.bookmark_name)
            previous = getattr(self, "outline_level", -1)
            level = min(flowable.heading_level - 1, previous + 1)
            self.canv.addOutlineEntry(
                flowable.heading_text, flowable.bookmark_name, level=max(0, level), closed=False
            )
            self.outline_level = max(0, level)


def render_report(source: Path, output: Path, *, title=None, author="", header=None, cover=False):
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Source Markdown and output PDF must be different files")
    source_bytes = source.read_bytes()
    markdown = source_bytes.decode("utf-8")
    blocks = parse_markdown(markdown)
    if not blocks:
        raise ValueError("Cannot render an empty report")
    fonts = register_fonts()
    styles = make_styles(fonts)
    title = title or next(
        (plain(block.text) for block in blocks if block.kind == "heading" and block.level == 1),
        source.stem,
    )
    header = header or title
    story, anchors, figures = [], {}, []
    if cover:
        story.extend([Spacer(1, 105), Paragraph(inline(title), styles["h1"]), Spacer(1, 18)])
        if author:
            story.append(Paragraph(inline(author), styles["body"]))
        story.extend(
            [Spacer(1, 12), HRFlowable(width="100%", thickness=0.8, color=BLUE), PageBreak()]
        )
    consumed_captions = set()
    for position, block in enumerate(blocks):
        if position in consumed_captions:
            continue
        if block.kind == "heading":
            upcoming = blocks[position + 1 : position + 3]
            table_follows = bool(upcoming) and (
                upcoming[0].kind == "table"
                or (
                    len(upcoming) == 2
                    and upcoming[1].kind == "table"
                    and upcoming[0].kind == "paragraph"
                    and re.match(r"^Table\s+\d+[.:]", plain(upcoming[0].text).strip("*_ "))
                )
            )
            heading_style = styles[f"h{block.level}"]
            if table_follows:
                # Reserve a short table lead, not an entire possibly multi-page table.
                story.append(CondPageBreak(130))
                heading_style = ParagraphStyle(
                    "table_lead_heading", parent=heading_style, keepWithNext=False
                )
            name = re.sub(r"[^\w-]+", "-", plain(block.text).lower()).strip("-") or "section"
            anchors[name] = anchors.get(name, 0) + 1
            bookmark = name if anchors[name] == 1 else f"{name}-{anchors[name]}"
            paragraph = Paragraph(
                f'<a name="{html.escape(bookmark)}"/>{inline(block.text)}',
                heading_style,
            )
            paragraph.bookmark_name, paragraph.heading_level = bookmark, block.level
            paragraph.heading_text = plain(block.text)
            story.append(paragraph)
        elif block.kind == "paragraph":
            follows_with_table = (
                position + 1 < len(blocks) and blocks[position + 1].kind == "table"
            )
            emphasized_table_caption = follows_with_table and bool(
                re.fullmatch(r"\*[^*]+\*|_[^_]+_", block.text.strip())
            )
            style = (
                styles["table_caption"]
                if re.match(r"^Table\s+\d+[.:]", plain(block.text).strip("*_ "))
                or emphasized_table_caption
                else styles["body"]
            )
            if style is styles["table_caption"]:
                story.append(CondPageBreak(100))
            story.append(Paragraph(inline(block.text), style))
        elif block.kind == "table":
            story.append(markdown_table(block, fonts))
        elif block.kind == "code":
            maximum = int((CONTENT_WIDTH - 26) / pdfmetrics.stringWidth("M", "Courier", 8.8))
            wrapped = "\n".join(
                "\n".join(
                    textwrap.wrap(
                        line,
                        width=maximum,
                        subsequent_indent="    ",
                        replace_whitespace=False,
                        drop_whitespace=False,
                        break_long_words=True,
                        break_on_hyphens=False,
                    )
                )
                if line
                else ""
                for line in block.text.split("\n")
            )
            story.append(Preformatted(wrapped, styles["code"]))
        elif block.kind == "list":
            marker = block.marker if block.marker[0].isdigit() else "•"
            story.append(Paragraph(inline(block.text), styles["list"], bulletText=marker))
        elif block.kind == "quote":
            story.append(Paragraph(inline(block.text), styles["quote"]))
        elif block.kind == "image":
            target = block.text.strip().strip("<>")
            if re.match(r"^[a-zA-Z]+://", target):
                raise ValueError("Figures must be local files; remote images are not downloaded")
            image_path = Path(unquote(target))
            if not image_path.is_absolute():
                image_path = source.parent / image_path
            image_path = image_path.resolve(strict=True)
            with PILImage.open(image_path) as bitmap:
                width, height = bitmap.size
            scale = min(CONTENT_WIDTH / width, 360 / height)
            figure = Image(str(image_path), width=width * scale, height=height * scale)
            figure.hAlign = "CENTER"
            caption = block.marker or f"Figure {len(figures) + 1}"
            if block.marker and not re.match(r"^Figure\s+\d+", block.marker):
                caption = f"Figure {len(figures) + 1}. {block.marker}"
            if (
                position + 1 < len(blocks)
                and blocks[position + 1].kind == "paragraph"
                and re.match(r"^Figure\s+\d+[.:]", plain(blocks[position + 1].text).strip("*_ "))
            ):
                # A full Markdown caption supersedes the short image alt text.
                # Keep the full caption with its figure even at a page boundary.
                caption = blocks[position + 1].text
                consumed_captions.add(position + 1)
            story.append(KeepTogether([figure, Paragraph(inline(caption), styles["caption"])]))
            figures.append(str(image_path))
        elif block.kind == "pagebreak":
            story.append(PageBreak())
        elif block.kind == "rule":
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.4,
                    color=colors.HexColor("#C6D0D9"),
                    spaceBefore=5,
                    spaceAfter=10,
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    document = ReportDocument(
        str(output), pagesize=A4, title=title, author=author, pageCompression=1, **MARGINS
    )

    def page_furniture(canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            label = header
            while label and pdfmetrics.stringWidth(label, fonts["regular"], 8.2) > CONTENT_WIDTH:
                label = label[:-1]
            canvas.setFont(fonts["regular"], 8.2)
            canvas.setFillColor(MUTED)
            canvas.drawString(MARGINS["leftMargin"], PAGE_HEIGHT - 32, label)
            canvas.setStrokeColor(colors.HexColor("#CFD7DF"))
            canvas.setLineWidth(0.35)
            canvas.line(
                MARGINS["leftMargin"],
                PAGE_HEIGHT - 40,
                PAGE_WIDTH - MARGINS["rightMargin"],
                PAGE_HEIGHT - 40,
            )
        canvas.setFont(fonts["regular"], 8.2)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(PAGE_WIDTH - MARGINS["rightMargin"], 29, f"{doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=page_furniture, onLaterPages=page_furniture)
    reader = PdfReader(output)
    if not reader.pages or not all(page.extract_text().strip() for page in reader.pages):
        raise ValueError("Generated PDF contains no pages or an empty page")
    return {
        "pdf": str(output),
        "source": str(source),
        "pages": len(reader.pages),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "figures": figures,
        "tables": sum(block.kind == "table" for block in blocks),
        "fonts": fonts,
        "visual_review_required": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("docs/report/final-report.md"))
    parser.add_argument(
        "--output", type=Path, default=Path("output/pdf/COMP5331_Group14_Report.pdf")
    )
    parser.add_argument("--title")
    parser.add_argument("--author", default="")
    parser.add_argument("--header")
    parser.add_argument("--cover", action="store_true")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    result = render_report(
        args.source,
        args.output,
        title=args.title,
        author=args.author,
        header=args.header,
        cover=args.cover,
    )
    if args.preview_dir:
        converter = shutil.which("pdftoppm")
        if not converter:
            raise FileNotFoundError("pdftoppm is required for page previews")
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.preview_dir / args.output.stem
        subprocess.run(
            [converter, "-r", "110", "-png", str(args.output.resolve()), str(prefix.resolve())],
            check=True,
        )
        result["preview_prefix"] = str(prefix.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
