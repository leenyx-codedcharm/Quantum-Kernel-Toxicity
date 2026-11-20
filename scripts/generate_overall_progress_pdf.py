from pathlib import Path
from datetime import datetime
import csv
import json
import subprocess
import sys
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether, HRFlowable
)
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "paper" / "progress_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "Quantum_Kernel_Toxicity_Overall_Progress_Report.pdf"

NAVY = colors.HexColor("#20364F")
TEAL = colors.HexColor("#287C78")
GREY = colors.HexColor("#647180")
LIGHT = colors.HexColor("#EAF0F4")
PALE = colors.HexColor("#F5F7F9")
DARK = colors.HexColor("#26323D")
BORDER = colors.HexColor("#D5DEE6")

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="ReportTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=27,
    leading=33,
    textColor=NAVY,
    alignment=TA_LEFT,
    spaceAfter=15
))

styles.add(ParagraphStyle(
    name="SubtitleCustom",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=13,
    leading=19,
    textColor=TEAL,
    spaceAfter=15
))

styles.add(ParagraphStyle(
    name="SectionCustom",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=20,
    textColor=NAVY,
    spaceBefore=15,
    spaceAfter=9,
    keepWithNext=True
))

styles.add(ParagraphStyle(
    name="SubsectionCustom",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=11.5,
    leading=15,
    textColor=TEAL,
    spaceBefore=10,
    spaceAfter=5,
    keepWithNext=True
))

styles.add(ParagraphStyle(
    name="BodyCustom",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9.3,
    leading=14,
    textColor=DARK,
    spaceAfter=7
))

styles.add(ParagraphStyle(
    name="SmallCustom",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=7.4,
    leading=10,
    textColor=DARK
))

styles.add(ParagraphStyle(
    name="CaptionCustom",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=8,
    leading=11,
    textColor=GREY,
    alignment=TA_CENTER,
    spaceBefore=4,
    spaceAfter=12
))

styles.add(ParagraphStyle(
    name="NoteCustom",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9,
    leading=13,
    textColor=NAVY,
    backColor=LIGHT,
    borderPadding=8,
    spaceBefore=5,
    spaceAfter=10
))

story = []

def para(text, style="BodyCustom"):
    return Paragraph(escape(str(text)), styles[style])

def heading(text, level=1):
    return Paragraph(
        escape(text),
        styles["SectionCustom" if level == 1 else "SubsectionCustom"]
    )

def bullet(text):
    return Paragraph(
        "&bull;&nbsp;&nbsp;" + escape(str(text)),
        styles["BodyCustom"]
    )

def note(title, body):
    return Paragraph(
        "<b>" + escape(title) + "</b><br/>" + escape(body),
        styles["NoteCustom"]
    )

def add_table(headers, rows, widths=None):
    data = [
        [Paragraph("<b>" + escape(str(x)) + "</b>", styles["SmallCustom"])
         for x in headers]
    ]

    for row in rows:
        data.append([
            Paragraph(escape(str(x)), styles["SmallCustom"])
            for x in row
        ])

    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")

    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(t)
    story.append(Spacer(1, 9))

def read_csv(path):
    if not path or not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []

def find_file(filename, roots=None):
    if roots is None:
        roots = [ROOT / "results", ROOT / "paper", ROOT / "reports"]

    for root in roots:
        if root.exists():
            found = list(root.rglob(filename))
            if found:
                return found[0]
    return None

def run_git(args):
    try:
        r = subprocess.run(
            ["git"] + args,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15
        )
        return (r.stdout or r.stderr).strip()
    except Exception as e:
        return str(e)

def fmt(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "Not reported"

def add_image(path, caption):
    try:
        reader = ImageReader(str(path))
        iw, ih = reader.getSize()

        max_w = 6.6 * inch
        max_h = 4.35 * inch
        scale = min(max_w / iw, max_h / ih)

        img = Image(
            str(path),
            width=iw * scale,
            height=ih * scale
        )

        story.append(Spacer(1, 5))
        story.append(img)
        story.append(para(caption, "CaptionCustom"))
    except Exception as e:
        story.append(para(
            f"Figure could not be embedded: {path.name} ({e})"
        ))

def collect_figures():
    roots = [
        ROOT / "results" / "visualizations",
        ROOT / "results" / "quantum_benchmarks",
        ROOT / "paper" / "quantum_benchmark",
    ]

    files = []

    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.png"))

    # Prefer full-run figures over pilot copies with the same name.
    selected = {}

    for p in files:
        name = p.stem.lower()
        parts = [x.lower() for x in p.parts]

        score = 0
        if "full" in parts:
            score += 10
        if "pilot" in parts:
            score -= 5
        if "quantum" in str(p).lower():
            score += 2

        if name not in selected or score > selected[name][0]:
            selected[name] = (score, p)

    return [
        x[1] for x in sorted(
            selected.values(),
            key=lambda x: x[1].stem.lower()
        )
    ]

def get_metric_rows():
    path = find_file("complete_metric_comparison.csv")
    rows = read_csv(path)

    rows = [
        r for r in rows
        if r.get("experiment", "").lower() == "full"
    ]

    if rows:
        return rows, path

    # Verified values from the supplied full-run terminal output.
    fallback = [
        {"model":"LightGBM descriptors","f1_positive_mean":"0.7022364","f1_macro_mean":"0.6995852","auroc_mean":"0.7872972","auprc_mean":"0.7891804"},
        {"model":"LightGBM fingerprints","f1_positive_mean":"0.7330797","f1_macro_mean":"0.7498300","auroc_mean":"0.8454548","auprc_mean":"0.8490820"},
        {"model":"LightGBM combined","f1_positive_mean":"0.7666320","f1_macro_mean":"0.7771866","auroc_mean":"0.8615599","auprc_mean":"0.8631747"},
        {"model":"Random Forest descriptors","f1_positive_mean":"0.7310735","f1_macro_mean":"0.7205110","auroc_mean":"0.8089529","auprc_mean":"0.8083028"},
        {"model":"Random Forest fingerprints","f1_positive_mean":"0.7736154","f1_macro_mean":"0.7756497","auroc_mean":"0.8562497","auprc_mean":"0.8574293"},
        {"model":"Random Forest combined","f1_positive_mean":"0.7832676","f1_macro_mean":"0.7938145","auroc_mean":"0.8758503","auprc_mean":"0.8765896"},
        {"model":"Tanimoto-SVM fingerprints","f1_positive_mean":"0.7630921","f1_macro_mean":"0.7768473","auroc_mean":"0.8604936","auprc_mean":"0.8637065"},
    ]
    return fallback, None

def page_footer(canvas, doc):
    canvas.saveState()
    w, h = A4

    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(18*mm, 15*mm, w-18*mm, 15*mm)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(18*mm, 10*mm, "Quantum Kernel Toxicity | Research Progress")
    canvas.drawRightString(w-18*mm, 10*mm, f"Page {doc.page}")

    canvas.restoreState()

# =========================================================
# COVER
# =========================================================

story.append(Spacer(1, 60))
story.append(para("QUANTUM KERNEL TOXICITY", "ReportTitle"))
story.append(para(
    "Overall Research Progress, Experimental Results and Benchmark Report",
    "SubtitleCustom"
))
story.append(HRFlowable(
    width="100%", thickness=2, color=TEAL, spaceAfter=22
))
story.append(para(
    "Comparative molecular toxicity prediction using classical "
    "machine learning, molecular representations and quantum models."
))
story.append(Spacer(1, 30))

cover_rows = [
    ["Report type", "Comprehensive research progress report"],
    ["Prepared", datetime.now().strftime("%d %B %Y")],
    ["Primary endpoints", "hERG inhibition and drug-induced liver injury"],
    ["Primary repository", "Quantum-Kernel-Toxicity"],
    ["Main experimental dataset", "hERG_Karim"],
]

add_table(
    ["Item", "Details"],
    cover_rows,
    [1.65*inch, 4.8*inch]
)

story.append(Spacer(1, 35))
story.append(note(
    "Report scope",
    "This report records work completed and results supplied through "
    "18 November 2026. It distinguishes completed experiments, "
    "preliminary pilot results, provisional data mappings and "
    "unfinished work. Missing results are not estimated."
))

story.append(PageBreak())

# =========================================================
# CONTENTS
# =========================================================

story.append(heading("Contents"))
contents = [
    "Executive summary",
    "1. Research objective and scope",
    "2. Dataset collection and audit",
    "3. Molecular curation and identity verification",
    "4. Feature generation",
    "5. Classical model benchmarking",
    "6. Quantum model development",
    "7. Quantum pilot results",
    "8. Scientific visualisations",
    "9. Software environment and testing",
    "10. Repository and reproducibility",
    "11. Remaining work and research risks",
    "12. Recommended next-stage plan",
    "Appendix: Key files and commands",
]
for x in contents:
    story.append(bullet(x))

story.append(Spacer(1, 10))
story.append(heading("Executive summary"))
story.append(para(
    "Quantum Kernel Toxicity investigates whether quantum-kernel and "
    "variational quantum approaches can improve molecular toxicity "
    "prediction for structurally unfamiliar compounds. The research "
    "currently focuses on hERG inhibition and drug-induced liver injury "
    "(DILI). The intended comparison includes classical machine-learning "
    "baselines, molecular foundation models, graph learning and quantum "
    "architectures."
))
story.append(para(
    "The hERG_Karim pipeline is the most mature. The raw dataset contains "
    "13,445 records and the curated feature dataset contains 10,824 "
    "molecules. Morgan fingerprints, RDKit descriptors and model-input "
    "files have been generated. Five frozen scaffold splits are available. "
    "LightGBM, Random Forest and Tanimoto-SVM have been evaluated across "
    "those five repeats."
))
story.append(para(
    "Random Forest using combined fingerprints and descriptors currently "
    "leads the reported classical comparison, with mean positive-class "
    "F1 of 0.7833, macro-F1 of 0.7938, AUROC of 0.8759 and AUPRC of 0.8766."
))
story.append(para(
    "The first four-qubit Fidelity Quantum Kernel pilot has also completed. "
    "Its test AUROC was 0.5704, below the matched Tanimoto-SVM control at "
    "0.6590. Additional quantum pilot work reported AUROC of 0.4598 for "
    "VQC with COBYLA, 0.5006 for VQC with SPSA, and 0.5704 for a trainable "
    "fidelity-kernel pilot. These are small pilot results and not final "
    "five-split benchmarks."
))
story.append(para(
    "The next research priorities are to complete the remaining quantum "
    "architectures, finish matched five-split evaluation, resolve DILI "
    "structure mapping and cross-source overlap, and complete calibration, "
    "conformal prediction and final statistical analysis."
))

# =========================================================
# 1. OBJECTIVE
# =========================================================

story.append(heading("1. Research objective and scope"))
story.append(heading("1.1 Main research question", 2))
story.append(para(
    "Can quantum-kernel and variational quantum models improve molecular "
    "toxicity prediction on structurally unfamiliar compounds compared "
    "with strong classical molecular machine-learning methods?"
))
story.append(heading("1.2 Research endpoints", 2))
add_table(
    ["Endpoint", "Description", "Current status"],
    [
        ["hERG inhibition", "Binary prediction of hERG channel inhibition.", "Primary experimental track; classical and quantum pilots completed."],
        ["DILI", "Binary prediction of drug-induced liver injury concern.", "Data collection and identity verification in progress."],
    ],
    [1.25*inch, 2.9*inch, 2.3*inch]
)
story.append(heading("1.3 Planned model families", 2))
add_table(
    ["Family", "Models"],
    [
        ["Classical ML", "LightGBM, Random Forest, Tanimoto-SVM"],
        ["Tabular foundation", "TabPFN v2"],
        ["Molecular graph", "Chemprop / D-MPNN"],
        ["Molecular foundation", "MoLFormer"],
        ["Quantum kernels", "Fixed Fidelity, Trainable Fidelity, Projected, Hybrid Tanimoto-Quantum"],
        ["Variational quantum", "VQC with SLSQP, COBYLA and SPSA; Data Re-uploading VQC"],
        ["Quantum feature learning", "Quantum Extreme Learning Machine"],
    ],
    [1.65*inch, 4.8*inch]
)

# =========================================================
# 2. DATASETS
# =========================================================

story.append(heading("2. Dataset collection and audit"))
add_table(
    ["Dataset", "Records", "Current role and status"],
    [
        ["TDC hERG", "655", "Downloaded; initial audit found duplicate IDs, duplicate structures and conflicting labels."],
        ["TDC hERG_Karim", "13,445 raw; 10,824 curated", "Primary hERG dataset; features and five scaffold splits available."],
        ["TDC DILI", "475", "Downloaded; 239 negative and 236 positive; initial audit passed."],
        ["DILIst", "1,279", "Audited and provisionally mapped; 100 records remain unresolved."],
        ["FDA DILIrank 2.0", "1,336", "Workbook downloaded; structure extraction still unresolved."],
        ["hERGCentral", "Not downloaded", "Optional expansion; not yet included."],
    ],
    [1.3*inch, 1.2*inch, 3.95*inch]
)

story.append(heading("2.1 TDC hERG audit", 2))
story.append(para(
    "The initial audit reported 451 positive and 204 negative records. "
    "It found 39 duplicate Drug_ID values, seven duplicate SMILES, seven "
    "canonical duplicate molecules and three conflicting label groups. "
    "The audit counted 648 unique valid canonical structures. These "
    "findings require a documented duplicate and label-conflict policy "
    "before this dataset is used in final benchmarking."
))

story.append(heading("2.2 hERG_Karim audit", 2))
story.append(para(
    "The raw hERG_Karim dataset contains 13,445 records, with 6,718 "
    "positive and 6,727 negative labels. Curation produced 10,824 "
    "molecules. The curation manifest reports 2,419 raw rows merged, "
    "2,020 same-label parent groups merged, 202 raw rows excluded for "
    "label conflicts, and 83 conflicting parent groups excluded."
))
story.append(para(
    "The subsequent integrity gate reports that all five frozen scaffold "
    "splits are disjoint, cover the curated rows and have zero cross-partition "
    "overlap in canonical SMILES, InChIKey and parent key. The 285 canonical "
    "duplicate groups in the earlier manifest refer to the pre-curation "
    "source audit; the curated data passed the reported identity checks. "
    "Three standardisation warnings remain documented: two SMILES "
    "serialization differences and one stored InChIKey/parent-key mismatch. "
    "A separate SMILES consistency warning was reported in the smaller "
    "hERG dataset."
))

story.append(heading("2.3 TDC DILI audit", 2))
story.append(para(
    "The TDC DILI dataset contains 475 records: 239 negative and 236 "
    "positive. The initial audit reported no missing values, duplicate "
    "IDs, duplicate SMILES, invalid SMILES, canonical duplicates or "
    "conflicting canonical-label groups."
))

story.append(heading("2.4 DILIst audit and mapping", 2))
story.append(para(
    "The DILIst supplementary workbook contains 1,279 records. It has "
    "768 positive and 511 negative labels. There are no missing compound "
    "names or labels, no duplicate IDs and no duplicate compound names. "
    "The administration-route field is missing for 266 records; this "
    "field is retained as metadata and is not intended as an input to "
    "the primary structure-only models."
))
add_table(
    ["Mapping result", "Records"],
    [
        ["PubChem candidate matches with RDKit-valid structures", "1,179"],
        ["Not found", "91"],
        ["Multiple CID candidates", "9"],
        ["Unique canonical structures reported", "1,176"],
        ["Duplicate-structure rows", "6"],
        ["Conflicting canonical-label groups", "0"],
    ],
    [4.6*inch, 1.8*inch]
)
story.append(para(
    "The six duplicate rows form three pairs: triacetyldiphenolisatin "
    "and Phenisatin; epoetin alfa and erythropoietin; bendazac and "
    "bendazolic acid. Identity review is still required. The epoetin/"
    "erythropoietin pair is a biologic and should not be treated as an "
    "ordinary small-molecule QSAR record without an explicit rationale. "
    "The 1,179 mapped records remain provisional."
))

story.append(heading("2.5 FDA DILIrank 2.0", 2))
story.append(para(
    "The FDA workbook contains 1,336 records, including most-concern, "
    "less-concern, no-concern and ambiguous-concern categories. The "
    "proposed study mapping treats most/less concern as positive, "
    "no concern as negative and ambiguous records as excluded. This "
    "is a study decision that must be documented. The last confirmed "
    "mapping run produced zero model-ready rows because the property "
    "extraction logic did not correctly use the available PubChem "
    "SMILES fields. This dataset is not yet ready for modelling."
))

# =========================================================
# 3. FEATURE GENERATION
# =========================================================

story.append(heading("3. Molecular curation and feature generation"))
story.append(para(
    "RDKit was used to parse molecular SMILES, generate canonical "
    "isomeric representations and calculate molecular descriptors. "
    "The feature pipeline preserves molecule identifiers and split "
    "assignments so that all model families can be compared on aligned "
    "molecular observations."
))
add_table(
    ["Feature representation", "Configuration", "Intended models"],
    [
        ["Morgan / ECFP4", "Radius 2, 2,048 bits, chirality enabled.", "Random Forest, LightGBM, Tanimoto-SVM and quantum experiments."],
        ["RDKit descriptors", "14 physicochemical and structural descriptors.", "LightGBM, TabPFN and feature ablations."],
        ["Chemprop input", "Molecule ID, SMILES, label and split assignment.", "Molecular graph learning."],
        ["MoLFormer input", "Clean SMILES with IDs, labels and split assignments.", "Pretrained molecular embeddings."],
    ],
    [1.35*inch, 2.9*inch, 2.15*inch]
)
story.append(para(
    "The feature-generation manifest records 10,824 molecules, 2,048-bit "
    "fingerprints and 14 descriptors. It records software versions and "
    "input checksums. Feature-generation validation reported seven passing "
    "tests at that stage. Later project validation reported 27 tests passing, "
    "followed by 32 passing and one skipped in the original environment "
    "and 10 passing in the quantum environment."
))

# =========================================================
# 4. CLASSICAL BENCHMARK
# =========================================================

story.append(PageBreak())
story.append(heading("4. Classical model benchmarking"))
story.append(para(
    "Three classical model families were evaluated using five repeated "
    "frozen scaffold splits. LightGBM and Random Forest were evaluated "
    "with descriptor-only, fingerprint-only and combined representations. "
    "Tanimoto-SVM used Morgan fingerprints."
))

metric_rows, metric_path = get_metric_rows()
table_rows = []

for r in metric_rows:
    table_rows.append([
        r.get("model", "Unknown"),
        fmt(r.get("f1_positive_mean")),
        fmt(r.get("f1_macro_mean")),
        fmt(r.get("auroc_mean")),
        fmt(r.get("auprc_mean")),
    ])

add_table(
    ["Model configuration", "Positive F1", "Macro-F1", "AUROC", "AUPRC"],
    table_rows,
    [2.25*inch, 1.0*inch, 0.95*inch, 0.95*inch, 0.95*inch]
)

story.append(note(
    "Current classical leader",
    "Random Forest with combined molecular fingerprints and RDKit "
    "descriptors has the highest reported mean positive-class F1 "
    "(0.7833), macro-F1 (0.7938), AUROC (0.8759) and AUPRC (0.8766) "
    "among the listed classical configurations."
))
story.append(para(
    "The combined representation improved positive-class F1 for both "
    "LightGBM and Random Forest relative to descriptor-only configurations. "
    "Tanimoto-SVM remained competitive and is especially important as "
    "the classical molecular-kernel control for quantum experiments."
))
story.append(para(
    "The full experiment reports 10,866 held-out predictions across "
    "five repeats. Repeated splits reuse molecules across different "
    "repetitions; therefore, intervals across repeats are descriptive "
    "variability summaries rather than independent-sample confidence "
    "intervals. Pilot results must remain separate from the full benchmark."
))

# =========================================================
# 5. QUANTUM METHODS
# =========================================================

story.append(heading("5. Quantum model development"))
story.append(heading("5.1 Fixed Fidelity Quantum Kernel", 2))
story.append(para(
    "The first quantum experiment used Qiskit and Qiskit Machine Learning. "
    "The molecular input combined ECFP4 fingerprints and 14 RDKit "
    "descriptors. Scaling and four-component PCA were fitted on training "
    "molecules only. Four qubits encoded the four-dimensional representation "
    "using a ZZ feature map with one repetition and linear entanglement. "
    "Exact statevector simulation was used with shots=None."
))
story.append(heading("5.2 Trainable Fidelity Kernel", 2))
story.append(para(
    "A trainable fidelity-kernel pilot was evaluated and reported AUROC "
    "of 0.5704. This equals the fixed fidelity pilot AUROC in the supplied "
    "summary. The optimization history and parameter changes should be "
    "inspected before concluding that training had no effect."
))
story.append(heading("5.3 Variational Quantum Classifier", 2))
add_table(
    ["VQC optimizer", "Pilot AUROC", "Status"],
    [
        ["COBYLA", "0.4598", "Pilot evaluated"],
        ["SPSA", "0.5006", "Pilot evaluated"],
        ["SLSQP", "Not available", "Matched rerun interrupted; no partial artifacts."],
    ],
    [2.0*inch, 1.3*inch, 2.8*inch]
)
story.append(heading("5.4 Additional quantum architectures", 2))
add_table(
    ["Architecture", "Implementation status"],
    [
        ["Data Re-uploading VQC", "Helpers added; predictive benchmark not yet reported."],
        ["Projected Quantum Kernel", "Helpers added; predictive benchmark not yet reported."],
        ["Hybrid Tanimoto-Quantum Kernel", "Not confirmed as trained."],
        ["Quantum Extreme Learning Machine", "Not confirmed as trained."],
    ],
    [2.5*inch, 3.9*inch]
)

# =========================================================
# 6. QUANTUM PILOT RESULTS
# =========================================================

story.append(heading("6. Quantum pilot results"))
story.append(para(
    "The authoritative fixed-fidelity pilot used 64 training, 128 "
    "validation and 128 test molecules from their respective frozen "
    "scaffold_r0 partitions. Positive rates were 51.56% in training, "
    "46.09% in validation and 38.28% in testing."
))
story.append(para(
    "Each SVM received 12 validation candidates. Both selected C=10 "
    "without class weighting. Validation AUROC was 0.5733 for fidelity "
    "and 0.6233 for Tanimoto. The selected thresholds were 0.519831 "
    "and 0.5 respectively. The configuration lock was verified before "
    "test evaluation."
))
add_table(
    ["Test metric", "Fidelity SVM", "Tanimoto-SVM"],
    [
        ["AUROC", "0.5704", "0.6590"],
        ["AUPRC", "0.4499", "0.5752"],
        ["Positive F1", "0.4082", "0.5225"],
        ["Macro-F1", "0.5205", "0.5785"],
        ["Balanced accuracy", "0.5205", "0.5871"],
        ["MCC", "0.0411", "0.1693"],
        ["Brier score", "0.2518", "0.2361"],
        ["Log loss", "0.6967", "0.6717"],
    ],
    [2.7*inch, 1.85*inch, 1.85*inch]
)
story.append(heading("6.1 Runtime comparison", 2))
add_table(
    ["Runtime measure", "Fidelity SVM", "Tanimoto-SVM"],
    [
        ["Kernel construction", "1.0992 s", "0.0270 s"],
        ["Final training", "0.0081 s", "0.0028 s"],
        ["Test prediction", "0.0006 s", "0.0011 s"],
    ],
    [2.7*inch, 1.85*inch, 1.85*inch]
)
story.append(para(
    "The Tanimoto control outperformed the fixed fidelity kernel on "
    "this pilot subset. This is a single-split result and does not establish "
    "the final ranking across the five scaffold repeats. Kernel matrices "
    "passed finite-value, ID-alignment, symmetry, unit-diagonal and PSD "
    "checks as reported in the experiment summary."
))
story.append(note(
    "Interpretation",
    "The current pilot does not demonstrate a quantum advantage. The "
    "quantum results must be improved or better understood through "
    "controlled architecture and optimizer experiments, then evaluated "
    "across all five frozen scaffold splits."
))

# =========================================================
# 7. VISUALIZATIONS
# =========================================================

story.append(PageBreak())
story.append(heading("7. Scientific visualisations and figures"))
story.append(para(
    "The local visualisation pipeline generated dataset and model "
    "comparison figures from saved predictions. It includes class "
    "distribution, descriptor distributions, scaffold distribution, "
    "molecular similarity, ROC curves, precision-recall curves, AUROC, "
    "AUPRC, F1, MCC, confusion matrices and split-level performance."
))
story.append(para(
    "The report generator searches the local results directory and embeds "
    "the actual PNG figures found there. Pilot and full versions are "
    "distinguished where separate files are available."
))

figures = collect_figures()

if figures:
    story.append(para(
        f"Actual local figures found and selected for embedding: {len(figures)}."
    ))

    for i, p in enumerate(figures, 1):
        caption = f"Figure {i}. {p.stem.replace('_', ' ').title()}."
        add_image(p, caption)
else:
    story.append(note(
        "No figures found",
        "No PNG figures were found in the expected results/visualizations, "
        "results/quantum_benchmarks or paper/quantum_benchmark directories. "
        "The report includes numerical benchmark tables, but the figure "
        "files must be generated or copied into those locations."
    ))

# =========================================================
# 8. SOFTWARE AND TESTING
# =========================================================

story.append(PageBreak())
story.append(heading("8. Software environment and testing"))
add_table(
    ["Environment", "Reported configuration"],
    [
        ["Original RDKit environment", "Python 3.11.9; NumPy 1.26.4; Pandas 2.2.3; RDKit 2022.09.5; scikit-learn 1.9.1."],
        ["Quantum environment", "Python 3.11.9; Qiskit 2.6.2; Qiskit Machine Learning 0.9.1; NumPy 2.4.6; Pandas 3.1.1; scikit-learn 1.9.1."],
        ["Original environment tests", "32 passed, 1 skipped."],
        ["Quantum environment tests", "10 passed."],
        ["Dependency check", "No broken requirements reported in the respective environments."],
    ],
    [1.8*inch, 4.6*inch]
)
story.append(para(
    "The original RDKit environment previously experienced a NumPy 2.x "
    "binary compatibility issue. It was repaired using NumPy 1.26.4 and "
    "Pandas 2.2.3, and RDKit imports were verified. Quantum packages were "
    "installed in a separate environment to avoid disturbing the RDKit "
    "stack."
))
story.append(para(
    "Full default pytest collection was reported as blocked by an "
    "untracked quantum smoke-test script importing Qiskit under the "
    "original RDKit environment. Tests run separately in the original "
    "and quantum environments passed as reported. This environment "
    "separation should be retained."
))
story.append(para(
    "No LaTeX compiler was available during the latest quantum development "
    "update. The LaTeX source exists, but PDF compilation was not completed."
))

# =========================================================
# 9. REPOSITORY
# =========================================================

story.append(heading("9. Repository and reproducibility"))
branch = run_git(["branch", "--show-current"])
commit = run_git(["log", "-1", "--oneline"])
status = run_git(["status", "--short"])
remotes = run_git(["remote", "-v"])

story.append(para(
    "The project repository is Quantum-Kernel-Toxicity. Git output "
    "previously confirmed a successful merge of the scientific "
    "visualisation milestone into main, with merge commit 9b610fa. "
    "Later quantum work added further scripts, reports and LaTeX source."
))
add_table(
    ["Git item", "Current local value"],
    [
        ["Current branch", branch or "Not available"],
        ["Latest commit", commit or "Not available"],
        ["Working-tree status", status if status else "Clean"],
        ["Remotes", remotes or "Not available"],
    ],
    [1.55*inch, 4.85*inch]
)
story.append(para(
    "The current report records local repository state at generation "
    "time. It does not assume that the latest quantum changes have "
    "already been committed or pushed."
))

# =========================================================
# 10. REMAINING WORK
# =========================================================

story.append(heading("10. Remaining work and research risks"))
add_table(
    ["Workstream", "Current status", "Next action"],
    [
        ["hERG data integrity", "Five-split identity checks passed; three standardisation warnings remain.", "Retain warnings and complete documented review."],
        ["DILIst", "1,179 provisional structures; 100 unresolved.", "Verify identities, duplicates, biologics and source overlap."],
        ["FDA DILIrank", "Zero model-ready rows in last confirmed run.", "Fix SMILES extraction and rerun from cache."],
        ["Quantum SLSQP VQC", "Interrupted; no final result.", "Rerun a small matched pilot."],
        ["Trainable Fidelity", "Pilot AUROC 0.5704.", "Inspect parameter optimization and kernel changes."],
        ["Data Re-uploading", "Helper implementation only.", "Train and evaluate matched pilot."],
        ["Projected kernel", "Helper implementation only.", "Train and evaluate matched pilot."],
        ["Hybrid kernel and QELM", "Not confirmed as trained.", "Implement only after core pilots are stable."],
        ["Five-split quantum benchmark", "Incomplete.", "Evaluate selected models across all frozen splits."],
        ["TabPFN, Chemprop, MoLFormer", "Not benchmarked in supplied progress update.", "Implement after core quantum comparison is stable."],
        ["Calibration and conformal prediction", "Incomplete.", "Use proper validation/calibration partitions."],
        ["LaTeX PDF", "Source exists; compiler unavailable.", "Install or configure TeX toolchain and compile."],
    ],
    [1.55*inch, 2.25*inch, 2.6*inch]
)

story.append(heading("11. Recommended next-stage plan"))
for item in [
    "Inspect the matched quantum pilot report, CSV and optimization histories.",
    "Complete the interrupted VQC SLSQP pilot.",
    "Verify whether trainable fidelity optimization changed the kernel parameters.",
    "Train Data Re-uploading VQC and Projected Quantum Kernel on matched molecules.",
    "Implement hybrid Tanimoto-Quantum and Quantum Extreme Learning Machine if feasible.",
    "Select stable configurations using validation data only.",
    "Run selected configurations across all five frozen scaffold splits.",
    "Generate unified F1, AUROC, AUPRC, MCC, runtime and circuit-complexity comparisons.",
    "Complete DILIst and DILIrank identity and overlap work before DILI model claims.",
    "Perform calibration, conformal prediction, novelty analysis and statistical comparisons.",
    "Compile the complete LaTeX manuscript after all benchmark results are locked.",
]:
    story.append(bullet(item))

# =========================================================
# APPENDIX
# =========================================================

story.append(PageBreak())
story.append(heading("Appendix A. Important project files"))
add_table(
    ["Artifact", "Location"],
    [
        ["Feature-generation script", "scripts/07_generate_features.py"],
        ["Classical baseline script", "scripts/08_train_classical_baselines.py"],
        ["Scientific visualisation script", "scripts/09_generate_scientific_visualizations.py"],
        ["DILIst mapper", "scripts/06_map_dilist.py"],
        ["hERG feature manifest", "data/processed/features/hERG_Karim/feature_manifest.json"],
        ["Classical metric table", "results/reports/herg_karim/complete_metric_comparison.csv"],
        ["Classical per-split metrics", "results/reports/herg_karim/per_split_metrics.csv"],
        ["Matched quantum pilot report", "matched_pilot_report.md (search repository)"],
        ["Matched quantum pilot results", "matched_pilot_results.csv (search repository)"],
        ["LaTeX source", "main.tex (search paper directory)"],
    ],
    [2.0*inch, 4.4*inch]
)

story.append(heading("Appendix B. Interpretation and reporting rules"))
for item in [
    "A valid RDKit SMILES parse does not prove that a PubChem structure matches the intended drug identity.",
    "DILIst and DILIrank must not be treated as independent without source and molecular overlap analysis.",
    "Pilot results must be distinguished from five-split benchmark results.",
    "Repeated scaffold splits reuse molecules across repeats; intervals across repeats are descriptive.",
    "Quantum simulation does not demonstrate computational quantum advantage.",
    "Test data must not be used for feature fitting, hyperparameter selection, threshold selection or conformal calibration.",
    "Report all negative and inconclusive findings, not only the best-performing model.",
]:
    story.append(bullet(item))

story.append(heading("Basis of preparation"))
story.append(para(
    "This report was generated from the progress summaries, terminal "
    "outputs, dataset audit results, feature manifest, baseline metric "
    "table, quantum pilot results, test summaries and repository state "
    "provided during the research workflow. Where local figures or "
    "result files were available, they were embedded or read directly. "
    "Where results were not supplied, the report marks them as pending "
    "rather than estimating them."
))

doc = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    rightMargin=18*mm,
    leftMargin=18*mm,
    topMargin=18*mm,
    bottomMargin=22*mm,
    title="Quantum Kernel Toxicity - Overall Research Progress Report",
    author="Quantum Kernel Toxicity Research Project",
    subject="Molecular toxicity prediction and quantum machine learning"
)

doc.build(
    story,
    onFirstPage=page_footer,
    onLaterPages=page_footer
)

print("\n" + "="*65)
print("QUANTUM KERNEL TOXICITY REPORT GENERATED")
print("="*65)
print("PDF:", OUT)
print("Size:", f"{OUT.stat().st_size:,}", "bytes")
print("Figures embedded:", len(figures))
print("Classical metric source:", metric_path if metric_path else "Verified supplied results")
print("Current branch:", branch)
print("Latest commit:", commit)
print("\nReport generation completed.")
