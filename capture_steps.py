"""Capture real browser views of saved project evidence, with provenance labels."""

import html
import json
import platform
import sys
from pathlib import Path

import torch
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent.resolve()


def read(name):
    p = ROOT / name
    return (
        p.read_text(encoding="utf-8", errors="replace")
        if p.exists()
        else "Not available yet"
    )


def capture(stage):
    out = ROOT / "screenshots"
    out.mkdir(exist_ok=True)
    sections = []
    if stage == "01_setup":
        data = dict(
            python=platform.python_version(),
            pytorch=torch.__version__,
            cuda=torch.cuda.is_available(),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
        sections = [
            ("Runtime verified", json.dumps(data, indent=2)),
            ("Cell validation", read("logs/cell_checks.txt")),
        ]
    elif stage == "02_data":
        sections = [
            ("Dataset download provenance", read("results/dataset_provenance.json")),
            ("Dataset and fixed splits", read("results/data_summary.json")),
        ]
    elif stage == "03_models":
        sections = [
            ("Equation, gradient and padding checks", read("logs/cell_checks.txt")),
            ("Manual cell implementation", read("models.py")),
        ]
    elif stage == "04_training":
        sections = [("Actual training output", read("logs/training.txt")[-14000:])]
    elif stage == "05_results":
        sections = [
            ("Held-out test comparison", read("results/comparison.csv")),
            ("Majority baseline", read("results/baseline.json")),
        ]
    elif stage.startswith("run_"):
        index = int(stage.split("_")[1]) - 1
        experiments = json.loads(read("results/experiments.json"))
        sections = [
            (
                "Measured experiment configuration and epoch history",
                json.dumps(experiments[index], indent=2),
            )
        ]
    content = "".join(
        "<h2>" + html.escape(title) + "</h2><pre>" + html.escape(text) + "</pre>"
        for title, text in sections
    )
    images = {
        "02_data": ["eda.png"],
        "05_results": [
            "learning_curves.png",
            "vanilla_confusion.png",
            "cifg_confusion.png",
            "wmc_confusion.png",
        ],
    }.get(stage, [])
    content += "".join(f'<img src="../results/{im}">' for im in images)
    page_html = f"""<!doctype html><meta charset="utf-8"><style>body{{font:18px Arial;background:#f5f7fb;color:#18314e;margin:40px;max-width:1160px}}h1{{font-size:34px}}h2{{font-size:24px;color:#216a8d}}pre{{background:white;border-left:4px solid #216a8d;padding:22px;white-space:pre-wrap;font:15px Consolas;line-height:1.5}}img{{max-width:100%;margin:20px 0}}footer{{color:#52677d;font-size:14px}}</style><h1>RNN lab project / {stage.replace("_", " ")}</h1><p>Zouhair Eouichak / TP B / UIR 2026-2027</p>{content}<footer>Browser screenshot of actual saved code, validation logs and experiment outputs. This page is an evidence viewer.</footer>"""
    page_file = out / (stage + ".html")
    page_file.write_text(page_html, encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(
            viewport={"width": 1280, "height": 900}, device_scale_factor=1
        )
        page.goto(page_file.as_uri())
        page.screenshot(path=str(out / (stage + ".png")), full_page=True)
        browser.close()
    print("Captured", stage)


if __name__ == "__main__":
    for stage in sys.argv[1:] or [
        "01_setup",
        "02_data",
        "03_models",
        "04_training",
        "05_results",
    ]:
        capture(stage)
