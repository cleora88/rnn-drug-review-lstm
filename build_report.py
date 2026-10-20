"""Build a five-page report using measured results only."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

pdfmetrics.registerFont(TTFont("ReportArial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("ReportArialBold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFontFamily(
    "ReportArial",
    normal="ReportArial",
    bold="ReportArialBold",
    italic="ReportArial",
    boldItalic="ReportArialBold",
)
R = Path(__file__).parent.resolve()
OUTPUT_DIR = R / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load(n):
    return json.loads((R / "results" / f"{n}.json").read_text())


data = load("data_summary")
runs = load("experiments")
final = load("final_results")
baseline = load("baseline")
env = load("environment")
styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="Body",
        fontName="ReportArial",
        fontSize=10,
        leading=14.5,
        spaceAfter=10,
        textColor=colors.HexColor("#18364B"),
    )
)
styles["Heading2"].fontName = "ReportArialBold"
styles["Heading2"].fontSize = 18
styles["Heading2"].leading = 23
styles["Heading2"].textColor = colors.HexColor("#007F7A")
styles["Heading2"].spaceAfter = 13
styles.add(
    ParagraphStyle(
        name="Cover",
        fontName="ReportArialBold",
        fontSize=31,
        leading=37,
        alignment=0,
        textColor=colors.HexColor("#102C40"),
        spaceAfter=18,
    )
)
styles.add(
    ParagraphStyle(
        name="CenterSmall",
        fontName="ReportArial",
        fontSize=11,
        leading=17,
        alignment=0,
        textColor=colors.HexColor("#526878"),
    )
)
story = []


def p(s):
    story.append(Paragraph(s, styles["Body"]))


def h(s):
    story.append(Paragraph(s, styles["Heading2"]))


def img(n, w=475):
    from PIL import Image as PI

    path = R / "results" / n
    with PI.open(path) as im:
        ratio = im.height / im.width
    story.append(Image(str(path), width=w, height=w * ratio))


def table(rows, widths=None):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "ReportArial"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102C40")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "ReportArialBold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.HexColor("#EFF3EF"), colors.white],
                ),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 13))


def page():
    story.append(PageBreak())


logo = R / "assets/UIR_logo.png"
from PIL import Image as PI

with PI.open(logo) as im:
    ratio = im.height / im.width
story.append(Image(str(logo), width=225, height=225 * ratio, hAlign="LEFT"))
story.append(Spacer(1, 42))
story.append(
    Paragraph(
        "LAB PROJECT / RECURRENT NEURAL NETWORKS",
        ParagraphStyle(
            name="Kicker",
            fontName="ReportArialBold",
            fontSize=10,
            textColor=colors.HexColor("#007F7A"),
            spaceAfter=15,
        ),
    )
)
story.append(
    Paragraph("Drug Review Sentiment<br/>with Manual LSTM Cells", styles["Cover"])
)
story.append(
    Paragraph(
        "Advanced Natural Language Processing<br/>Lab Project on Recurrent Neural Networks<br/>University year 2026/2027",
        styles["CenterSmall"],
    )
)
story.append(Spacer(1, 25))
story.append(
    Paragraph(
        "<b>Zouhair Eouichak</b> &nbsp; / &nbsp; Aymane El Youssi<br/>Malak Bartal &nbsp; / &nbsp; Sara Baalla<br/>Group TP B &nbsp; / &nbsp; 20 October 2026",
        styles["CenterSmall"],
    )
)
story.append(Spacer(1, 35))
h("Task and study design")
p(
    "This project predicts rating-derived sentiment from patient review text and compares a vanilla LSTM with two manually implemented variants: Coupled Input-Forget Gate (CIFG) and Working Memory Connections (WMC). The comparison uses identical data partitions and a common evaluation protocol."
)
p(
    "The study evaluates learning rate and batch size across nine training runs. Each architecture uses the configuration and checkpoint with the highest validation macro-F1. Accuracy and macro-F1 on a held-out test subset measure final performance."
)
page()
h("1. Dataset and exploratory analysis")
p(
    f"The Drugs.com dataset supplied in the project brief contains {data['original']['train']:,} training reviews and {data['original']['test']:,} test reviews in the downloaded mirror. Input features contain review text only. Ratings define negative (1-4), neutral (5-6), and positive (7-10) labels. These thresholds are a modeling choice, and the resulting labels are a proxy for sentiment."
)
p(
    f"Text normalization decodes HTML, lowercases text and retains word tokens, numbers, exclamation marks and question marks. Exact normalized duplicates are removed within each partition. Removing test reviews found in training excludes {data['removed_test_overlap']:,} additional test rows. This reduces direct text leakage but does not guarantee independence of near-duplicate or same-drug reviews."
)
table(
    [["Partition", "Reviews", "Negative", "Neutral", "Positive"]]
    + [
        [n, str(d["size"])] + [str(d["classes"].get(str(k), 0)) for k in range(3)]
        for n, d in data["splits"].items()
    ],
    [100, 80, 90, 90, 90],
)
img("eda.png")
p(
    f"The original training partition is split 85/15 with stratification and seed 42, followed by fixed stratified subsets for practical manual-cell training. The training-only vocabulary contains {data['vocabulary']:,} tokens including padding and unknown tokens. A 96-token limit truncates {data['train_truncation_fraction']:.1%} of training reviews. Class imbalance motivates class-weighted loss and macro-F1."
)
page()
h("2. Models and implementation")
p(
    "Every recurrent cell uses PyTorch nn.Linear for its affine transformations. The implementation contains no nn.LSTM. Embeddings have dimension 64, recurrent states have dimension 64, dropout is 0.2, and a linear head maps the final non-padding hidden state to three logits. Padding leaves both hidden and memory states unchanged."
)
table(
    [
        ["Cell", "Main change"],
        ["Vanilla", "Independent input, forget and output gates"],
        ["CIFG", "Input gate i_t = 1 - f_t"],
        ["WMC", "Memory-to-gate connections through tanh"],
    ],
    [85, 365],
)
p(
    "<b>Vanilla:</b> sigmoid gates control the candidate tanh update, memory retention and hidden output. The memory update is c_t = f_t * c_(t-1) + i_t * g_t. The output is h_t = o_t * tanh(c_t)."
)
p(
    "<b>CIFG:</b> the input gate complements the forget gate, eliminating one learned gate. This reduces recurrent parameters while forcing a direct trade-off between retaining old memory and accepting new content."
)
p(
    "<b>WMC:</b> input and forget gates add tanh-transformed connections from the previous memory state. The output gate adds a tanh-transformed connection from the newly computed memory state c_t, following the appendix ordering. Each connection uses a full learned linear matrix without an additional bias."
)
table(
    [["Architecture", "Total trainable parameters"]]
    + [[f["model"].upper(), f"{f['parameters']:,}"] for f in final],
    [225, 225],
)
p(
    "Totals include embeddings and the classifier, so the shared embedding can dominate the overall parameter count. Cell validation checks equations against explicit calculations, verifies finite gradients and tests invariance to trailing padding for all three variants."
)
h("Training protocol")
p(
    f"The recorded runtime uses Python {env['python']}, PyTorch {escape(env['torch'])}, and {escape(str(env['gpu']))}. Training uses Adam, class-weighted cross entropy, gradient clipping at norm 1.0, four epochs and seed 42. All architectures share vocabulary, split files, sequence length and loss weighting."
)
page()
h("3. Hyperparameter experiments and evaluation")
p(
    "The three settings are learning rate 0.001 with batch size 128, learning rate 0.003 with batch size 128, and learning rate 0.001 with batch size 64. This provides controlled comparisons for learning rate and batch size, but does not estimate their full interaction. Default PyTorch initialization remains fixed."
)
table(
    [["Cell", "Learning rate", "Batch", "Best validation F1"]]
    + [
        [
            r["model"],
            str(r["learning_rate"]),
            str(r["batch_size"]),
            f"{r['best_validation_macro_f1']:.4f}",
        ]
        for r in runs
    ],
    [90, 100, 70, 190],
)
img("learning_curves.png", 450)
p(
    "Macro-F1 averages the F1 scores of all three classes equally. The validation score selects settings and epoch checkpoints. The held-out test set is evaluated only after that selection. A single random seed limits claims about the stability of differences between cells."
)
page()
h("4. Final results and discussion")
table(
    [["Cell", "LR", "Batch", "Test accuracy", "Test macro-F1"]]
    + [
        [
            f["model"].upper(),
            str(f["learning_rate"]),
            str(f["batch_size"]),
            f"{f['accuracy']:.4f}",
            f"{f['macro_f1']:.4f}",
        ]
        for f in final
    ],
    [80, 70, 60, 110, 130],
)
p(
    f"A majority-class predictor scores {baseline['majority_accuracy']:.4f} accuracy and {baseline['majority_macro_f1']:.4f} macro-F1 on this test subset. This baseline makes it possible to assess whether learned models improve beyond class frequency alone. Architecture conclusions must consider macro-F1, model size and run-to-run uncertainty together."
)
chosen = max(runs, key=lambda r: r["best_validation_macro_f1"])
img(chosen["model"] + "_confusion.png", 280)
report = load(chosen["model"] + "_classification")
p(
    f"The configuration with the highest validation macro-F1 belongs to {chosen['model'].upper()}. Its per-class test F1 scores are negative {report['negative']['f1-score']:.3f}, neutral {report['neutral']['f1-score']:.3f}, and positive {report['positive']['f1-score']:.3f}. The confusion matrix above reports its measured test errors. The saved prediction CSVs support further qualitative error analysis."
)
p(
    "<b>Limitations and improvements:</b> use several seeds, train longer with early stopping, increase sequence length and data size, and compare pretrained embeddings. Inspect ambiguous ratings, negation and mixed sentiment. Consider grouping by drug or stronger duplicate detection when testing generalization. The present experiment uses rating labels and a limited training budget, so it cannot establish broad superiority of any cell."
)
p(
    "<b>References:</b> Project Guidelines_RNN_2627, appendix equations. UCI Drug Review Dataset (Drugs.com), dataset 462, Kallumadi and Grasser (2018). Download mirror: github.com/semnan-university-ai/Drug-Review-Dataset. CIFG reading: arxiv.org/abs/1503.04069. WMC reading: doi/source identifier S0893608021003439."
)


def footer(c, doc):
    c.setStrokeColor(colors.HexColor("#C7D5D2"))
    c.setLineWidth(0.5)
    c.line(48, 39, A4[0] - 48, 39)
    c.setFont("ReportArial", 8)
    c.setFillColor(colors.HexColor("#526878"))
    c.drawString(48, 25, "UIR / ADVANCED NLP / TP B")
    c.drawRightString(A4[0] - 48, 25, f"{doc.page:02d} / 05")
    if doc.page > 1:
        c.setFont("ReportArialBold", 8)
        c.setFillColor(colors.HexColor("#007F7A"))
        c.drawString(48, A4[1] - 30, "DRUG REVIEW SENTIMENT / MANUAL LSTM COMPARISON")


SimpleDocTemplate(
    str(OUTPUT_DIR / "RNN_Report_Zouhair_Eouichak.pdf"),
    title="Drug Review Sentiment with Manual LSTM Cells",
    author="Zouhair Eouichak; Aymane El Youssi; Malak Bartal; Sara Baalla",
    pagesize=A4,
    rightMargin=48,
    leftMargin=48,
    topMargin=54,
    bottomMargin=50,
).build(story, onFirstPage=footer, onLaterPages=footer)
print("Report created")
