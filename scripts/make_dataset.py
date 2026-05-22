from __future__ import annotations
import argparse, random, json, shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CLASSES = ["invoice", "resume", "academic_report", "business_letter", "memo"]

TEMPLATES = {
    "invoice": ["INVOICE #{n}", "Bill To: {name}", "Date: 2025-{m:02d}-{d:02d}", "Description     Qty     Total", "Consulting services  1   USD {amount}", "Payment due in 30 days"],
    "resume": ["{name}", "Email: {email}", "Professional Summary", "Data analyst with Python, SQL, and AI project experience.", "Skills: machine learning, dashboards, communication", "Education: BSc AI & Big Data"],
    "academic_report": ["Academic Report", "Title: Multi-Agent AI Systems", "Abstract: This report studies agent collaboration.", "Methodology: dataset, model, experiments", "Results: accuracy and confusion matrix", "Conclusion and limitations"],
    "business_letter": ["{company}", "Dear Sir or Madam,", "We are writing regarding the proposed collaboration.", "Please find attached the requested information.", "Sincerely,", "Management Office"],
    "memo": ["MEMORANDUM", "To: Project Team", "From: Coordinator", "Subject: Weekly Update", "Action items and deadlines are listed below.", "Please confirm receipt by Friday."],
}

NAMES = ["Amina El Idrissi", "Youssef Benali", "Sara Haddad", "Omar Alaoui", "Nora Mansouri"]
COMPANIES = ["Atlas Analytics", "Rabat Consulting", "Maghreb Digital", "UIR Innovation"]

def render(lines, path: Path, seed: int):
    rng = random.Random(seed)
    img = Image.new("L", (512, 720), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", rng.choice([12, 14, 16]))
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default(); title_font = font
    y = rng.randint(18, 35)
    x0 = rng.randint(20, 45)
    for i, line in enumerate(lines):
        draw.text((x0, y), line, fill=rng.randint(0, 30), font=title_font if i == 0 else font)
        y += rng.randint(28, 42)
        if rng.random() < 0.35:
            draw.line((x0, y, 480, y), fill=180, width=1); y += 12
    # simple noise lines/boxes for visual variety
    # Class-specific visual layout cues that mimic real document structure.
    joined = "\n".join(lines).lower()
    if "invoice" in joined:
        for yy in range(240, 420, 45):
            draw.line((40, yy, 470, yy), fill=90, width=2)
        for xx in [210, 310, 410]:
            draw.line((xx, 220, xx, 430), fill=100, width=2)
    elif "professional summary" in joined:
        draw.rectangle((25, 70, 150, 680), fill=230, outline=80)
        for yy in range(120, 600, 70): draw.line((175, yy, 470, yy), fill=160, width=1)
    elif "academic report" in joined:
        draw.line((256, 150, 256, 670), fill=110, width=2)
        for yy in range(180, 650, 35):
            draw.line((40, yy, 235, yy), fill=180, width=1); draw.line((275, yy, 470, yy), fill=180, width=1)
    elif "dear sir" in joined:
        draw.rectangle((25, 20, 490, 85), outline=70, width=3)
        draw.line((60, 610, 250, 610), fill=70, width=2)
    elif "memorandum" in joined:
        draw.rectangle((25, 25, 490, 145), outline=50, width=4)
        for yy in [210, 260, 310, 360]: draw.ellipse((55, yy, 70, yy+15), fill=80)
    for _ in range(rng.randint(1, 3)):
        x1, y1 = rng.randint(20, 420), rng.randint(80, 650)
        draw.rectangle((x1, y1, x1+rng.randint(30, 80), y1+rng.randint(8, 25)), outline=200)
    img = img.resize((128, 128))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

def make_line(cls, n, rng):
    data = dict(n=n, name=rng.choice(NAMES), email=f"user{n}@example.com", company=rng.choice(COMPANIES), amount=rng.randint(100, 9000), m=rng.randint(1,12), d=rng.randint(1,28))
    lines = [line.format(**data) for line in TEMPLATES[cls]]
    if rng.random() < 0.5:
        lines.append(rng.choice(["Reference: SDA-2025", "Approved by supervisor", "Confidential", "Page 1 of 1"]))
    return lines

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--samples-per-class', type=int, default=80); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--out', default='data/synthetic')
    args=ap.parse_args(); root=Path(args.out)
    if root.exists(): shutil.rmtree(root)
    rng=random.Random(args.seed)
    splits=(['train']*70)+(['val']*15)+(['test']*15)
    for cls in CLASSES:
        for i in range(args.samples_per_class):
            split=splits[int(i*100/args.samples_per_class)] if args.samples_per_class>=100 else ('train' if i < args.samples_per_class*0.7 else 'val' if i < args.samples_per_class*0.85 else 'test')
            lines=make_line(cls, i, rng)
            render(lines, root/split/cls/f'{cls}_{i:04d}.png', args.seed+i+hash(cls)%1000)
    Path('models').mkdir(exist_ok=True)
    Path('models/classes.json').write_text(json.dumps(CLASSES, indent=2), encoding='utf-8')
    print(f'Generated synthetic dataset at {root} with classes {CLASSES}')
if __name__=='__main__': main()
