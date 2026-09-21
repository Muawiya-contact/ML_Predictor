"""
evaluate_second_half.py
=======================================================================
Second-half evaluation: 20 fixed chief complaints (including the
food-related subset) through the offline Qwen2.5 clinical-concept
pipeline, scored against gold English references, with a Markdown + HTML
report that LibreOffice / pandoc can turn into a PDF.

Reuses the same generation, matching and scoring machinery as
evaluate_clinical_concepts.py (paper Table 2) so the two evaluations are
directly comparable:

  * generation : fuzzy Roman Urdu dictionary -> Qwen2.5 concept prompt,
    temperature 0.0, via the local Ollama service (offline_pipeline);
  * matching   : symmetrical stop-word removal + semantic equivalence
    table (tachycardia<->palpitations, vertigo<->dizziness, ...), then
    bag overlap;
  * per item   : P = N_match/N_gen, R = N_match/N_ref, F1 = 2PR/(P+R).

Reports
  evaluation_report_second_half.md
  evaluation_report_second_half.html
  second_half_evaluation_rows.csv          (traceability, untracked)
=======================================================================
"""

import argparse
import csv
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from evaluate_clinical_concepts import generate_concept, score_concept

try:
    from fpdf import FPDF
except Exception as _e:  # pragma: no cover - PDF assembly optional
    FPDF = None

MODEL = os.environ.get("SECOND_HALF_MODEL", "qwen2.5:latest")

# serial, Roman Urdu complaint, gold English reference
CASES = [
    (1, "Chest pain aur ghabrahat khana khane ke baad se aur thakan bohat ho rahi hai.",
        "Chest pain and palpitations/anxiety since after eating and extreme fatigue."),
    (2, "Chest pain left side khana khane ke baad se aur pasina bohat aa raha hai.",
        "Left-sided chest pain since after eating and excessive sweating."),
    (3, "Tez chest pain aur saans phool rahi hai aadhay ghante se aur saans lene mein takleef hai.",
        "Severe chest pain and shortness of breath for half an hour and difficulty breathing."),
    (4, "Chest pain aur saans phool rahi hai do ghante se, haath thanday ho gaye hain.",
        "Chest pain and shortness of breath for two hours, hands have become cold."),
    (5, "Patient ko chest pain aur ghabrahat ho raha hai kal raat se aur ghabrahat bhi ho rahi hai.",
        "Patient has had chest pain and palpitations/anxiety since last night."),
    (6, "Achanak palpitations ek ghante se, known cardiac patient hai.",
        "Sudden palpitations for one hour, is a known cardiac patient."),
    (7, "Bohat zyada dil tez dhadak raha hai khana khane ke baad se aur bohat kamzori lag rahi hai.",
        "Heart is beating very fast since after eating and feeling very weak."),
    (8, "Mamuli chest mein pressure aadhay ghante se aur ulti jaisa lag raha hai, family mein heart disease hai.",
        "Mild pressure in the chest for half an hour and feeling nauseous, family history of heart disease."),
    (9, "Seena jakar raha hai aadhay ghante se aur chakkar bhi aa rahe hain.",
        "Chest tightness for half an hour and also feeling dizzy."),
    (10, "Shadeed seena jakar raha hai do ghante se.",
        "Severe chest tightness/constriction for two hours."),
    (11, "Seena mein jalan ka ehsaas aram karte hue bhi aur dard peeth tak ja raha hai.",
        "Feeling of burning in the chest even at rest and pain radiating to the back."),
    (12, "Shadeed chest tightness sotay waqt achanak se aur thakan bohat ho rahi hai.",
        "Severe chest tightness suddenly while sleeping and extreme fatigue."),
    (13, "Halki left arm mein dard subah se aur thakan bohat ho rahi hai.",
        "Mild pain in the left arm since morning and extreme fatigue."),
    (14, "Mamuli chest mein tez dard achanak se aur bohat kamzori lag rahi hai, family mein heart disease hai.",
        "Mild chest with sudden sharp pain and feeling very weak, family history of heart disease."),
    (15, "Seena bhaari lag raha hai achanak se, ghabrahat bhi ho rahi hai.",
        "Feeling of heaviness in the chest suddenly, along with anxiety/palpitations."),
    (16, "Halki chest tightness kaam karte waqt se aur pasina bohat aa raha hai, purani cardiac history hai.",
        "Mild chest tightness while working and excessive sweating, old cardiac history."),
    (17, "Back mein dard ke sath seena tight achanak se, haath thanday ho gaye hain.",
        "Sudden chest tightness along with back pain, hands have become cold."),
    (18, "Tez irregular heartbeat aadhay ghante se aur dard kandhay tak ja raha hai, diabetes ka mareez hai.",
        "Fast irregular heartbeat for half an hour and pain radiating to the shoulder, diabetic patient."),
    (19, "Left arm mein dard sotay waqt achanak se aur ulti jaisa lag raha hai.",
        "Sudden pain in the left arm while sleeping and feeling nauseous."),
    (20, "Tez seena mein dard aur pasina seedhiyan chadhte waqt se aur dard peeth tak ja raha hai.",
        "Severe chest pain and sweating while climbing stairs and pain radiating to the back."),
]

FOOD_SUBSET = {1, 2, 7}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def sd(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if xs else 0.0


def fmt(v):
    return f"{v:.4f}"


def norm_concept(concept):
    """Spacing artifacts that are not words: the model emits
    'shortness_of_breath' (underscore) and the gold references encode
    synonym pairs as 'palpitations/anxiety', 'tightness/constriction'
    (slash). Split both on BOTH sides before tokenising so the compare
    counts words, not glued blobs. Symmetric and deterministic."""
    if not concept:
        return concept
    return " ".join(concept.replace("_", " ").replace("/", " ").split())


def __stats(rows, subset):
    sel = [r for r in rows if r["sn"] in subset]
    ps = [r["p"] for r in sel]; rs = [r["r"] for r in sel]
    fs = [r["f1"] for r in sel]
    return mean(ps), sd(ps), mean(rs), sd(rs), mean(fs), sd(fs)


def load_rows_csv(path="second_half_evaluation_rows.csv"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "sn": int(r["sn"]), "urdu": r["urdu"], "gold": r["gold"],
                "concept": r["concept"], "used": r["model"],
                "n_gen": int(r["n_gen"]), "n_ref": int(r["n_ref"]),
                "n_match": int(r["n_match"]),
                "p": float(r["P"]), "r": float(r["R"]), "f1": float(r["F1"]),
            })
    return rows


def write_reports(rows, overall, food, model, ts):
    """Write the Markdown, HTML and PDF report for these (already scored)
    rows. Deterministic - no model calls, so it can be re-run freely."""
    md, html = build_report(rows, overall, food, model, ts)
    with open("evaluation_report_second_half.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open("evaluation_report_second_half.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("[ok] wrote evaluation_report_second_half.md / .html")
    build_pdf("evaluation_report_second_half.pdf", rows, overall, food,
              model, ts)


def main():
    ap = argparse.ArgumentParser(description="Second-half clinical-concept evaluation.")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the MD/HTML/PDF from second_half_evaluation_rows.csv "
                         "without calling the model (reproducible report step)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    if args.report_only:
        rows = load_rows_csv()
        overall = __stats(rows, {r["sn"] for r in rows})
        food = __stats(rows, FOOD_SUBSET)
        now = time.strftime("%Y-%m-%d %H:%M:00")
        write_reports(rows, overall, food, args.model, now)
        print_summary(overall, food)
        return

    rows = []
    print(f"[ok] generating concepts with {args.model} (20 complaints)",
         flush=True)
    for sn, urdu, gold in CASES:
        concept, used = generate_concept(urdu, model=args.model)
        concept = norm_concept(concept)
        gold = norm_concept(gold)
        n_gen, n_ref, n_match, p, r, f1 = score_concept(concept or "", gold)
        rows.append({
            "sn": sn, "urdu": urdu, "gold": gold,
            "concept": concept or "", "used": used,
            "n_gen": n_gen, "n_ref": n_ref, "n_match": n_match,
            "p": p, "r": r, "f1": f1,
        })
        print(f"  [{sn:>2}] P={p:.3f} R={r:.3f} F1={f1:.3f}  concept="
              f"{concept or 'NONE'}", flush=True)

    with open("second_half_evaluation_rows.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sn", "urdu", "gold", "concept", "model",
                    "n_gen", "n_ref", "n_match", "P", "R", "F1"])
        for r in rows:
            w.writerow([r["sn"], r["urdu"], r["gold"], r["concept"],
                        r["used"], r["n_gen"], r["n_ref"], r["n_match"],
                        fmt(r["p"]), fmt(r["r"]), fmt(r["f1"])])

    overall = __stats(rows, {r["sn"] for r in rows})
    food = __stats(rows, FOOD_SUBSET)

    now = time.strftime("%Y-%m-%d %H:%M:00")
    write_reports(rows, overall, food, args.model, now)
    print_summary(overall, food)


def print_summary(overall, food):
    mp, sp, mr, sr, mf, sf = overall
    print("\nOVERALL  (n=20):   Precision {:.4f} +/- {:.4f} | Recall "
          "{:.4f} +/- {:.4f} | F1 {:.4f} +/- {:.4f}".format(mp, sp, mr, sr, mf, sf))
    mp, sp, mr, sr, mf, sf = food
    print("FOOD-RELATED (1,2,7): Precision {:.4f} +/- {:.4f} | Recall "
          "{:.4f} +/- {:.4f} | F1 {:.4f} +/- {:.4f}".format(mp, sp, mr, sr, mf, sf))


def build_report(rows, overall, food, model, ts):
    half = 10
    md = []
    md.append("# Second-Half Evaluation Report\n")
    md.append("Clinical-concept extraction of 20 Roman Urdu chief complaints "
              "through the offline Qwen2.5 pipeline, scored against gold "
              "English references.\n")
    md.append(f"- Generation model: `{model}` (local Ollama, temperature 0.0)\n")
    md.append(f"- Generated: {ts}\n")
    md.append("- Scoring: symmetrical stop-word removal, underscore/slash token "
              "splitting, semantic-equivalence matching (e.g. "
              "tachycardia\u2194palpitations, vertigo\u2194dizziness), bag overlap.\n")
    md.append("- Per item: `P = N_match / N_gen`, `R = N_match / N_ref`, "
              "`F1 = 2PR/(P+R)`.\n")

    for chunk, title in ((rows[:half], "Items 1-10"), (rows[half:], "Items 11-20")):
        md.append(f"\n## {title}\n")
        md.append("| SN | Roman Urdu Complaint | Model Concept (Qwen2.5) | "
                  "Gold Reference | P | R | F1 |\n")
        md.append("|----|----------------------|--------------------------|"
                  "---------------|-------|-------|-------|\n")
        for r in chunk:
            md.append("| {sn} | {u} | {c} | {g} | {p:.4f} | {r:.4f} | {f:.4f} |\n"
                      .format(sn=r["sn"], u=r["urdu"], c=r["concept"] or "_none_",
                              g=r["gold"], p=r["p"], r=r["r"], f=r["f1"]))

    md.append("\n## Aggregated Statistics\n")
    md.append("\n### Overall (n=20)\n")
    md.append("| Metric | N | Mean +/- SD |\n")
    md.append("|--------|---|-------------|\n")
    for lab, vals in zip(("Precision", "Recall", "F1"),
                         ((overall[0], overall[1]), (overall[2], overall[3]),
                          (overall[4], overall[5]))):
        md.append(f"| {lab} | 20 complaints | {vals[0]:.4f} +/- {vals[1]:.4f} |\n")
    md.append("\n### Food-related subset (items 1, 2, 7; n=3)\n")
    md.append("| Metric | N | Mean +/- SD |\n")
    md.append("|--------|---------------|-------------|\n")
    for lab, vals in zip(("Precision", "Recall", "F1"),
                         ((food[0], food[1]), (food[2], food[3]),
                          (food[4], food[5]))):
        md.append(f"| {lab} | 3 complaints | {vals[0]:.4f} +/- {vals[1]:.4f} |\n")

    f1s = [r["f1"] for r in rows]
    best = max(rows, key=lambda r: r["f1"])
    worst = min(rows, key=lambda r: r["f1"])
    median = sorted(f1s)[len(f1s) // 2]
    weak = ", ".join(str(r["sn"]) for r in sorted(rows, key=lambda r: r["f1"])[:4])
    md.append("\n## Key Observations\n")
    md.append(f"- F1 across the 20 complaints: min {min(f1s):.3f} (item "
              f"{worst['sn']}), max {max(f1s):.3f} (item {best['sn']}), "
              f"median {median:.3f}.\n")
    md.append(f"- Weakest items are {weak}; their generated concepts recover "
              "the fewest gold tokens.\n")
    md.append(f"- Food-related subset (items 1, 2, 7): F1 "
              f"{food[4]:.3f} +/- {food[5]:.3f}.\n")
    md.append("- Recall is systematically lower than precision: the model "
              "names each finding once, while gold references use fuller "
              "clinical phrasing (e.g. 'heart is beating very fast' vs "
              "'palpitations').\n")
    md_text = "".join(md)

    # ---------- HTML ----------
    def esc(s):
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    half = 10
    rows_html = ""
    for i, r in enumerate(rows):
        if i > 0 and i % 10 == 0:
            rows_html += "</table>\n<table>\n"
        if i % 10 == 0:
            rows_html += ("<tr><th>SN</th><th>Roman Urdu Complaint</th>"
                          "<th>Model Concept (Qwen2.5)</th><th>Gold Reference</th>"
                          "<th>P</th><th>R</th><th>F1</th></tr>\n")
        rows_html += ("<tr>"
                      f"<td>{r['sn']}</td><td>{esc(r['urdu'])}</td>"
                      f"<td class='c'>{esc(r['concept']) or '<i>(none)</i>'}</td>"
                      f"<td class='g'>{esc(r['gold'])}</td>"
                      f"<td>{r['p']:.4f}</td><td>{r['r']:.4f}</td>"
                      f"<td>{r['f1']:.4f}</td>"
                      "</tr>\n")

    overall_mp, overall_sp = overall[0], overall[1]
    overall_mr, overall_sr = overall[2], overall[3]
    overall_mf, overall_sf = overall[4], overall[5]
    f_mp, f_sp, f_mr, f_sr, f_mf, f_sf = food
    best = max(rows, key=lambda r: r["f1"])
    worst = min(rows, key=lambda r: r["f1"])
    median = sorted(r["f1"] for r in rows)[len(rows) // 2]
    weak = ", ".join(str(r["sn"]) for r in sorted(rows, key=lambda r: r["f1"])[:4])

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Second-Half Evaluation Report</title>
<style>
  body {{ font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 9.5pt;
         margin: 1.6cm; color: #17202a; }}
  h1 {{ font-size: 16pt; color: #1b4f72; border-bottom: 2px solid #1b4f72;
       padding-bottom: 4px; }}
  h2 {{ font-size: 12.5pt; color: #1b4f72; margin-top: 18px; }}
  h3 {{ font-size: 11pt; color: #2e4053; margin-top: 14px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0 14px 0;
          page-break-inside: auto; }}
  th, td {{ border: 1px solid #aab7b8; padding: 4px 6px; text-align: left;
           vertical-align: top; word-wrap: break-word; }}
  th {{ background: #1b4f72; color: #fff; font-size: 8.8pt; }}
  tr:nth-child(even) td {{ background: #eaf2f8; }}
  td.c {{ color: #145a32; font-family: 'DejaVu Sans Mono', monospace;
          font-size: 8.6pt; }}
  td.g {{ color: #4a235a; }}
  .stats td {{ text-align: center; }}
  .meta {{ color: #566573; font-size: 9pt; }}
  .foot {{ color: #7b7d7d; font-size: 8pt; margin-top: 24px; }}
  ul {{ margin: 6px 0; }}
</style></head><body>

<h1>Second-Half Evaluation Report</h1>
<p class="meta">Clinical-concept extraction of 20 Roman Urdu chief complaints
through the offline Qwen2.5 pipeline, scored against gold English
references.<br>
Generation model: <b>{esc(model)}</b> (local Ollama, temperature 0.0) &bull;
Generated: {ts}</p>

<h2>Method</h2>
<ul>
  <li>Generation: fuzzy Roman Urdu dictionary pass, then Qwen2.5 clinical-concept
      prompt (temperature 0.0) via the project&rsquo;s offline Ollama pipeline.</li>
  <li>Scoring: symmetrical stop-word removal, underscore/slash token splitting,
      semantic-equivalence matching
      (tachycardia&harr;palpitations, vertigo&harr;dizziness, pain-&gt;ache,
      etc.), bag overlap of tokens.</li>
  <li>Per item: <i>P = N<sub>match</sub>/N<sub>gen</sub></i>,
      <i>R = N<sub>match</sub>/N<sub>ref</sub></i>,
      <i>F1 = 2PR/(P+R)</i>.</li>
</ul>

<h2>Per-Complaint Comparison</h2>
<table>
<col width="3%"><col width="27%"><col width="25%"><col width="31%">
<col width="4.6%"><col width="4.6%"><col width="4.7%">
{rows_html}
</table>

<h2>Aggregated Statistics</h2>
<h3>Overall (n=20)</h3>
<table class="stats">
<tr><th>Metric</th><th>N</th><th>Mean &plusmn; SD</th></tr>
<tr><td>Precision</td><td>20 complaints</td><td>{overall_mp:.4f} &plusmn; {overall_sp:.4f}</td></tr>
<tr><td>Recall</td><td>20 complaints</td><td>{overall_mr:.4f} &plusmn; {overall_sr:.4f}</td></tr>
<tr><td>F1</td><td>20 complaints</td><td>{overall_mf:.4f} &plusmn; {overall_sf:.4f}</td></tr>
</table>

<h3>Food-related subset (items 1, 2, 7; n=3)</h3>
<table class="stats">
<tr><th>Metric</th><th>N</th><th>Mean &plusmn; SD</th></tr>
<tr><td>Precision</td><td>3 complaints</td><td>{f_mp:.4f} &plusmn; {f_sp:.4f}</td></tr>
<tr><td>Recall</td><td>3 complaints</td><td>{f_mr:.4f} &plusmn; {f_sr:.4f}</td></tr>
<tr><td>F1</td><td>3 complaints</td><td>{f_mf:.4f} &plusmn; {f_sf:.4f}</td></tr>
</table>

<h2>Key Observations</h2>
<ul>
  <li>F1 across the 20 complaints: min {worst['f1']:.3f} (item {worst['sn']}),
      max {best['f1']:.3f} (item {best['sn']}), median {median:.3f}.</li>
  <li>Weakest items are {weak}; their generated concepts recover the fewest
      gold tokens.</li>
  <li>Food-related subset (items 1, 2, 7): F1 {f_mf:.3f} &plusmn; {f_sf:.3f}.</li>
  <li>Recall is systematically lower than precision: the model names each
      finding once, while gold references use fuller clinical phrasing.</li>
</ul>

<p class="foot">Research prototype, not a medical device. Synthetic data.
Generated by <tt>evaluate_second_half.py</tt> on {ts}.</p>
</body></html>"""

    return md_text, html


def _pdf_ascii(s):
    return (s or "").replace("\u2194", "<->").replace("\u2192", ">")\
        .replace("\u2019", "'").encode("latin-1", "replace").decode("latin-1")


def build_pdf(path, rows, overall, food, model, ts):
    """Deterministic fpdf2 rendering with exact column widths (the HTML
    table path depends on the converter's CSS support; this does not)."""
    if FPDF is None:
        print("[warn] fpdf2 not installed - skipping evaluation_report_second_half.pdf",
              flush=True)
        return
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_margins(10, 12, 10)
    pdf.set_auto_page_break(False)
    pdf.set_title("Second-Half Evaluation Report")
    pdf.set_subject(
        "Clinical-concept extraction of 20 Roman Urdu chief complaints "
        "through the offline Qwen2.5 pipeline (local evaluation)")
    pdf.set_author("ML_Predictor evaluation pipeline")
    pdf.set_creator("evaluate_second_half.py (fpdf2)")
    LINE_H = 4.0
    COLS = [("SN", 10), ("Roman Urdu Complaint", 80), ("Model Concept (Qwen2.5)", 70),
            ("Gold Reference", 86), ("P", 10), ("R", 10), ("F1", 11)]

    def header_row():
        pdf.set_fill_color(27, 79, 114)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 8)
        x = 10
        for name, w in COLS:
            pdf.set_xy(x, pdf.get_y())
            pdf.cell(w, 6, _pdf_ascii(name), align="C", fill=True)
            x += w
        pdf.set_y(pdf.get_y() + 6)

    def stat_table(title, s):
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(27, 79, 114)
        pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "B", 8.5)
        pdf.set_fill_color(27, 79, 114)
        pdf.set_text_color(255, 255, 255)
        labels = ("Precision", "Recall", "F1")
        rows_ = [s[:2], s[2:4], s[4:6]]
        pdf.cell(40, 6, "Metric", fill=True, new_x="RIGHT", new_y="TOP")
        pdf.cell(30, 6, "N", fill=True, new_x="RIGHT", new_y="TOP")
        pdf.cell(80, 6, "Mean +/- SD", fill=True, new_x="LMARGIN", new_y="NEXT")
        for lab, (m, sdv) in zip(labels, rows_):
            pdf.set_font("helvetica", "", 8.5)
            if pdf.get_y() > 205:
                pdf.add_page()
            pdf.cell(40, 6, lab, border=1, new_x="RIGHT", new_y="TOP")
            pdf.cell(30, 6, "20" if "Overall" in title else "3",
                     border=1, new_x="RIGHT", new_y="TOP")
            pdf.cell(80, 6, f"{m:.4f} +/- {sdv:.4f}", border=1,
                     new_x="LMARGIN", new_y="NEXT")

    # ---- page 1: title, method, aggregate stats ----
    pdf.add_page()
    pdf.set_text_color(27, 79, 114)
    pdf.set_font("helvetica", "B", 15)
    pdf.cell(0, 10, "Second-Half Evaluation Report",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(50, 50, 50)
    pdf.set_font("helvetica", "", 8.5)
    pdf.multi_cell(0, 4.5,
        "Clinical-concept extraction of 20 Roman Urdu chief complaints through the "
        "offline Qwen2.5 pipeline, scored against gold English references.",
        new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 4.5,
        f"Generation model: {_pdf_ascii(model)} (local Ollama, temperature 0.0)   "
        f"Generated: {_pdf_ascii(ts)}",
        new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(27, 79, 114)
    pdf.cell(0, 6, "Method", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(50, 50, 50)
    pdf.set_font("helvetica", "", 8.5)
    pdf.multi_cell(0, 4.3,
        "- Generation: fuzzy Roman Urdu dictionary pass, then Qwen2.5 "
        "clinical-concept prompt (temperature 0.0) via the project's offline "
        "Ollama pipeline.\n"
        "- Scoring: symmetrical stop-word removal, underscore/slash token "
        "splitting, semantic-equivalence matching "
        "(tachycardia<->palpitations, vertigo<->dizziness, etc.), bag overlap.\n"
        "- Per item: P = N_match / N_gen,   R = N_match / N_ref,   "
        "F1 = 2PR/(P+R).",
        new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    mp, sp, mr, sr, mf, sf = overall
    stat_table("Aggregated Statistics - Overall (n=20)", overall)
    pdf.ln(3)
    stat_table("Food-related subset (items 1, 2, 7; n=3)", food)
    f_mp, f_sp, f_mr, f_sr, f_mf, f_sf = food
    pdf.set_draw_color(150, 150, 150)

    # ---- key observations ----
    pdf.ln(1)
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(27, 79, 114)
    pdf.cell(0, 6, "Key Observations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(50, 50, 50)
    pdf.set_font("helvetica", "", 8.5)
    f1s = [r["f1"] for r in rows]
    best = max(rows, key=lambda r: r["f1"])
    worst = min(rows, key=lambda r: r["f1"])
    median = sorted(f1s)[len(f1s) // 2]
    weak = ", ".join(str(r["sn"]) for r in sorted(rows, key=lambda r: r["f1"])[:4])
    for o in (
        f"- F1 across the 20 complaints: min {min(f1s):.3f} (item {worst['sn']}), "
        f"max {max(f1s):.3f} (item {best['sn']}), median {median:.3f}.",
        f"- Weakest items are {weak}; their generated concepts recover the "
        "fewest gold tokens.",
        f"- Food-related subset (items 1, 2, 7): F1 {f_mf:.3f} +/- {f_sf:.3f}.",
        "- Recall is systematically lower than precision: the model names each "
        "finding once, while gold references use fuller clinical phrasing "
        "('heart is beating very fast' vs 'palpitations').",
    ):
        pdf.multi_cell(0, 4.4, o, new_x="LMARGIN", new_y="NEXT")

    # ---- per-complaint table (flows straight onto page 1) ----
    pdf.ln(1)
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(27, 79, 114)
    pdf.cell(0, 6, "Per-Complaint Comparison", new_x="LMARGIN", new_y="NEXT")
    if pdf.get_y() > 185:
        pdf.add_page()
    header_row()
    alt = False
    for r in rows:
        cells = [str(r["sn"]), r["urdu"], r["concept"] or "(none)", r["gold"],
                 f"{r['p']:.4f}", f"{r['r']:.4f}", f"{r['f1']:.4f}"]
        wrapped = []
        for txt, (_, w) in zip(cells, COLS):
            pdf.set_xy(10, pdf.get_y())
            lines = pdf.multi_cell(w - 3, LINE_H, _pdf_ascii(txt),
                                   dry_run=True, output="LINES")
            wrapped.append(lines)
        row_h = max(len(x) for x in wrapped) * LINE_H + 2.5
        if pdf.get_y() + row_h > 198:
            pdf.add_page()
            header_row()
        y0 = pdf.get_y()
        pdf.set_fill_color(234, 242, 248)
        x = 10
        for txt, (_, w), lines in zip(cells, COLS, wrapped):
            pdf.set_xy(x, y0)
            pdf.set_font("helvetica", "", 8)
            pdf.multi_cell(w, LINE_H, _pdf_ascii(txt), border=1,
                           fill=alt, new_x="RIGHT", new_y="TOP")
            x += w
        alt = not alt
        pdf.set_y(y0 + row_h)
    pdf.set_text_color(120, 120, 120)
    for i in range(1, pdf.pages_count + 1):
        pdf.page = i
        pdf.set_y(-10)
        pdf.set_font("helvetica", "", 8)
        pdf.cell(0, 5, f"Page {i} of {pdf.pages_count}", align="C")
    pdf.output(path)
    print(f"[ok] wrote {path}")


if __name__ == "__main__":
    main()