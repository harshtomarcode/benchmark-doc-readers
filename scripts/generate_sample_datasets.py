#!/usr/bin/env python3
"""Regenerate the bundled sample datasets under datasets/."""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = f"/usr/share/fonts/truetype/dejavu/{name}"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def gen_text() -> None:
    d = DATA / "text"
    d.mkdir(parents=True, exist_ok=True)
    (d / "q1_report.txt").write_text(
        "Acme Corp Quarterly Report — Q1 2024\n\n"
        "Revenue for the first quarter reached $12.4 million, up 18% year over year.\n"
        "Operating expenses totaled $8.1 million. Net income was $3.2 million.\n"
        "The company hired 45 new employees and opened an office in Austin, Texas.\n"
        "CEO Maya Chen stated that product adoption in the enterprise segment grew by 27%.\n",
        encoding="utf-8",
    )
    (d / "minutes.txt").write_text(
        "City Council Meeting Minutes — March 12, 2025\n\n"
        "Attendance: Mayor Lopez, Councilors Reed, Okonkwo, Patel, and Nguyen.\n"
        "The council approved Ordinance 441 to expand the downtown bike lane network.\n"
        "Budget amendment BA-19 allocated $250,000 for park renovations at Riverside Green.\n"
        "The next meeting is scheduled for March 26, 2025 at 7:00 PM.\n"
        "Public comments closed at 8:15 PM.\n",
        encoding="utf-8",
    )
    write_jsonl(
        d / "samples.jsonl",
        [
            {
                "id": "text-001",
                "document": "q1_report.txt",
                "question": "What was Acme Corp's Q1 2024 revenue?",
                "answer": "$12.4 million",
            },
            {
                "id": "text-002",
                "document": "q1_report.txt",
                "question": "Where did Acme open a new office?",
                "answer": "Austin, Texas",
            },
            {
                "id": "text-003",
                "document": "q1_report.txt",
                "question": "Who is the CEO of Acme Corp?",
                "answer": "Maya Chen",
            },
            {
                "id": "text-004",
                "document": "minutes.txt",
                "question": "Which ordinance was approved?",
                "answer": "Ordinance 441",
            },
            {
                "id": "text-005",
                "document": "minutes.txt",
                "question": "How much was allocated for Riverside Green renovations?",
                "answer": "$250,000",
            },
        ],
    )


def gen_tables() -> None:
    d = DATA / "tables"
    d.mkdir(parents=True, exist_ok=True)
    (d / "sales.csv").write_text(
        "Product,Region,Units,Revenue\n"
        "Widget A,North,120,4800\n"
        "Widget A,South,90,3600\n"
        "Widget B,North,200,10000\n"
        "Widget B,South,150,7500\n"
        "Widget C,North,50,5000\n"
        "Widget C,South,40,4000\n",
        encoding="utf-8",
    )
    (d / "employees.md").write_text(
        "| Employee | Department | Salary |\n"
        "|----------|------------|--------|\n"
        "| Ana Ruiz | Engineering | 145000 |\n"
        "| Ben Cho  | Sales      | 98000  |\n"
        "| Cara Li  | Engineering | 152000 |\n"
        "| Dan Park | Marketing   | 87000  |\n",
        encoding="utf-8",
    )
    img = Image.new("RGB", (520, 220), "white")
    draw = ImageDraw.Draw(img)
    font, font_b = _font(16), _font(16, bold=True)
    rows = [
        ["Item", "Q1", "Q2", "Q3"],
        ["Apples", "120", "140", "160"],
        ["Oranges", "80", "95", "110"],
        ["Bananas", "200", "180", "210"],
    ]
    x0, y0, cw, rh = 20, 20, 120, 40
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x, y = x0 + c * cw, y0 + r * rh
            draw.rectangle([x, y, x + cw, y + rh], outline="black", width=1)
            draw.text((x + 10, y + 10), cell, fill="black", font=font_b if r == 0 else font)
    img.save(d / "fruit_table.png")
    write_jsonl(
        d / "samples.jsonl",
        [
            {
                "id": "tbl-001",
                "document": "sales.csv",
                "question": "What was the revenue for Widget B in the North region?",
                "answer": "10000",
            },
            {
                "id": "tbl-002",
                "document": "sales.csv",
                "question": "How many units of Widget A were sold in the South?",
                "answer": "90",
            },
            {
                "id": "tbl-003",
                "document": "employees.md",
                "question": "What is Cara Li's salary?",
                "answer": "152000",
            },
            {
                "id": "tbl-004",
                "document": "fruit_table.png",
                "question": "How many bananas were sold in Q3?",
                "answer": "210",
            },
        ],
    )


def draw_bar_chart(path: Path, title: str, labels: list[str], values: list[int]) -> None:
    w, h = 640, 400
    img = Image.new("RGB", (w, h), "#fafafa")
    draw = ImageDraw.Draw(img)
    font, font_t = _font(14), _font(18, bold=True)
    draw.text((20, 15), title, fill="#111", font=font_t)
    margin_l, margin_b, margin_t, margin_r = 60, 50, 60, 20
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b
    draw.line(
        [(margin_l, margin_t), (margin_l, h - margin_b), (w - margin_r, h - margin_b)],
        fill="#333",
        width=2,
    )
    max_v = max(values) or 1
    bar_w = plot_w / (len(values) * 1.5)
    colors = ["#2a6f97", "#e76f51", "#2a9d8f", "#e9c46a"]
    for i, (lab, val) in enumerate(zip(labels, values)):
        bh = (val / max_v) * (plot_h - 10)
        x = margin_l + 20 + i * (bar_w * 1.5)
        y = h - margin_b - bh
        draw.rectangle([x, y, x + bar_w, h - margin_b], fill=colors[i % len(colors)])
        draw.text((x, h - margin_b + 8), lab, fill="#111", font=font)
        draw.text((x, y - 18), str(val), fill="#111", font=font)
    img.save(path)


def gen_charts() -> None:
    d = DATA / "charts"
    d.mkdir(parents=True, exist_ok=True)
    draw_bar_chart(
        d / "revenue_by_region.png",
        "Revenue by Region ($M)",
        ["North", "South", "East", "West"],
        [12, 9, 15, 7],
    )
    draw_bar_chart(
        d / "monthly_users.png",
        "Monthly Active Users (k)",
        ["Jan", "Feb", "Mar", "Apr"],
        [40, 48, 55, 62],
    )
    write_jsonl(
        d / "samples.jsonl",
        [
            {
                "id": "chart-001",
                "image": "revenue_by_region.png",
                "question": "Which region has the highest revenue?",
                "answer": "East",
            },
            {
                "id": "chart-002",
                "image": "revenue_by_region.png",
                "question": "What is the revenue for the West region in millions?",
                "answer": "7",
            },
            {
                "id": "chart-003",
                "image": "monthly_users.png",
                "question": "How many monthly active users (in thousands) were there in March?",
                "answer": "55",
            },
            {
                "id": "chart-004",
                "image": "monthly_users.png",
                "question": "Did user count increase from January to April?",
                "answer": "yes",
            },
        ],
    )


def gen_infographics() -> None:
    d = DATA / "infographics"
    d.mkdir(parents=True, exist_ok=True)
    font, font_t, font_b = _font(18), _font(28, bold=True), _font(36, bold=True)

    img = Image.new("RGB", (700, 500), "#0b3d4a")
    draw = ImageDraw.Draw(img)
    draw.text((30, 25), "Global Water Use 2023", fill="white", font=font_t)
    for i, (num, label) in enumerate(
        [("70%", "Agriculture"), ("20%", "Industry"), ("10%", "Domestic")]
    ):
        x = 40 + i * 220
        draw.ellipse([x, 120, x + 160, 280], fill="#2a9d8f", outline="white", width=3)
        draw.text((x + 35, 165), num, fill="white", font=font_b)
        draw.text((x + 30, 300), label, fill="#e9c46a", font=font)
    draw.text(
        (30, 400),
        "Source: Example Water Institute  |  Population covered: 5.2B",
        fill="#aad",
        font=font,
    )
    img.save(d / "water_use.png")

    img2 = Image.new("RGB", (700, 420), "#1a1a2e")
    draw = ImageDraw.Draw(img2)
    draw.text((30, 20), "EV Adoption Snapshot", fill="#eaeaea", font=font_t)
    for i, (country, pct) in enumerate(
        [("Norway", "82%"), ("China", "29%"), ("USA", "8%"), ("Japan", "3%")]
    ):
        y = 100 + i * 70
        draw.rectangle([30, y, 670, y + 50], fill="#16213e", outline="#0f3460")
        draw.text((50, y + 12), country, fill="#eaeaea", font=font)
        draw.text((520, y + 8), pct, fill="#e94560", font=font_b)
    draw.text((30, 380), "Share of new car sales that are electric", fill="#888", font=font)
    img2.save(d / "ev_adoption.png")

    write_jsonl(
        d / "samples.jsonl",
        [
            {
                "id": "info-001",
                "image": "water_use.png",
                "question": "What percentage of global water use is for agriculture?",
                "answer": "70%",
            },
            {
                "id": "info-002",
                "image": "water_use.png",
                "question": "What population was covered according to the infographic?",
                "answer": "5.2B",
            },
            {
                "id": "info-003",
                "image": "ev_adoption.png",
                "question": "Which country has the highest EV share of new car sales?",
                "answer": "Norway",
            },
            {
                "id": "info-004",
                "image": "ev_adoption.png",
                "question": "What is the EV adoption percentage in the USA?",
                "answer": "8%",
            },
        ],
    )


def make_scan(path: Path, lines: list[str]) -> None:
    w, h = 800, 500
    img = Image.new("RGB", (w, h), "#e8e4d9")
    draw = ImageDraw.Draw(img)
    font = _font(22)
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="#1a1a1a", font=font)
        y += 40
    px = img.load()
    for _ in range(8000):
        x, yy = random.randint(0, w - 1), random.randint(0, h - 1)
        c = px[x, yy]
        px[x, yy] = tuple(max(0, min(255, ch + random.randint(-25, 25))) for ch in c)
    img.save(path)


def gen_ocr() -> None:
    d = DATA / "ocr"
    d.mkdir(parents=True, exist_ok=True)
    make_scan(
        d / "invoice_scan.png",
        [
            "INVOICE #INV-2048",
            "Date: 2024-11-03",
            "Bill To: Orion Labs Inc.",
            "Item: Cloud Compute Credits",
            "Amount Due: $1,275.50",
            "Payment Terms: Net 30",
        ],
    )
    make_scan(
        d / "form_scan.png",
        [
            "PATIENT INTAKE FORM",
            "Name: Jordan Ellis",
            "DOB: 1988-04-17",
            "MRN: 559021",
            "Allergies: Penicillin",
            "Physician: Dr. Samira Haddad",
        ],
    )
    write_jsonl(
        d / "samples.jsonl",
        [
            {
                "id": "ocr-001",
                "image": "invoice_scan.png",
                "question": "What is the invoice number?",
                "answer": "INV-2048",
            },
            {
                "id": "ocr-002",
                "image": "invoice_scan.png",
                "question": "What is the amount due?",
                "answer": "$1,275.50",
            },
            {
                "id": "ocr-003",
                "image": "form_scan.png",
                "question": "What is the patient's name?",
                "answer": "Jordan Ellis",
            },
            {
                "id": "ocr-004",
                "image": "form_scan.png",
                "question": "What allergies are listed?",
                "answer": "Penicillin",
            },
        ],
    )


def main() -> None:
    random.seed(42)
    gen_text()
    gen_tables()
    gen_charts()
    gen_infographics()
    gen_ocr()
    print(f"Sample datasets written under {DATA}")


if __name__ == "__main__":
    main()