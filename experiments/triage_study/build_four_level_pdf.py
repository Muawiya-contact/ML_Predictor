"""Eight-page four-level report: all baseline conditions plus CV-selected model."""
import argparse, json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
NAMES = {'logreg': 'Logistic Regression', 'hgb': 'Hist Gradient Boosting', 'rf': 'Random Forest'}

def main(results, output):
    results = Path(results)
    output = Path(output)
    rows = json.loads((results / 'results.json').read_text())
    plan = json.loads((results / 'comparison_plan.json').read_text())
    expected = {f'{v}_{pc}_{c}' for v in ['text', 'fused'] for pc in [768, 64] for c in NAMES}
    assert len(rows) == 12 and {r['id'] for r in rows} == expected, 'All twelve comparisons must be complete.'
    byid = {r['id']: r for r in rows}
    assert plan['labels'] == [0, 1, 2, 3]
    selected = json.loads((results / 'selected_results.json').read_text())
    verification = json.loads((results / 'verification.json').read_text())
    assert verification['status'] == 'passed'
    cv = pd.read_csv(results / 'cv_summary.csv')
    audit = json.loads((results / 'data_audit.json').read_text())
    figures = results / 'figures'
    figures.mkdir(exist_ok=True)
    ntest = plan['test_rows']
    ntrain = plan['development_rows']
    encoder = 'SapBERT' if plan['encoder']['name'] == 'sapbert_concept' else 'MPNet (SBERT)'
    fonts = Path(reportlab.__file__).parent / 'fonts'
    for name, file in [('Vera', 'Vera.ttf'), ('VeraBold', 'VeraBd.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    styles = getSampleStyleSheet()
    for s in styles.byName.values():
        s.fontName = 'VeraBold' if s.name in ['Title', 'Heading2'] else 'Vera'
        s.textColor = colors.black
    styles['Title'].fontSize = 17
    styles['Title'].leading = 21
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 12
    styles.add(styles['BodyText'].clone('Small'))
    styles['Small'].fontSize = 8
    styles['Small'].leading = 10.5
    story = []

    def p(text, style='BodyText'):
        story.extend([Paragraph(text, styles[style]), Spacer(1, 7)])

    def page(title):
        if story:
            story.append(PageBreak())
        p(title, 'Title')

    def table(data, widths, size=8.5, padding=5):
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, -1), 'Vera'), ('FONTNAME', (0, 0), (-1, 0), 'VeraBold'), ('FONTSIZE', (0, 0), (-1, -1), size), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), 0.7, colors.black), ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.black), ('TOPPADDING', (0, 0), (-1, -1), padding), ('BOTTOMPADDING', (0, 0), (-1, -1), padding), ('ALIGN', (2, 1), (-1, -1), 'RIGHT')]))
        story.extend([t, Spacer(1, 10)])

    def export_table(data, name, note):
        fig, ax = plt.subplots(figsize=(10, 0.36 * len(data) + 0.8))
        ax.axis('off')
        widths = {6: [0.12, 0.32, 0.14, 0.14, 0.14, 0.14], 5: [0.32, 0.17, 0.17, 0.17, 0.17], 4: [0.58, 0.14, 0.14, 0.14]}[len(data[0])]
        tab = ax.table(cellText=data[1:], colLabels=data[0], colWidths=widths, loc='center', cellLoc='left')
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1, 1.5)
        for (row, col), cell in tab.get_celld().items():
            cell.visible_edges = 'B' if row in (0, len(data)-1) else ''
            if row == 0: cell.set_text_props(weight='bold')
        fig.text(0.06, 0.025, note, fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(figures / (name + '.png'), dpi=220)
        plt.close(fig)

    def pct(v):
        return f'{100 * v:.2f}%'
    page(f'{encoder} Four-Level Comparison')
    p('Sections 3.4 and 3.5 | Full 768-D versus PCA-64', 'Heading2')
    p(f'Encoder: {encoder}. Frozen, normalized 768-dimensional clinical-concept embeddings. PCA reduces the same embeddings to 64 dimensions. Each classifier is evaluated on both text-only and text-plus-patient inputs, producing all 12 comparisons.')
    p(f'Dataset: {ntrain + ntest:,} supplied cardiac records. Fixed grouped split: {ntrain:,} development and {ntest:,} test rows; labels 0-3 (Emergency, Urgent, Standard, Non-urgent). Every comparison uses the same rows. PCA and patient-feature preprocessing are fitted on development rows only. Classifier settings are fixed across representations.')
    p('Class totals: ' + '; '.join(f"L{k} {name}: {audit['class_counts'][str(k)]:,}" for k, name in enumerate(plan['label_names'])) + '.', 'Small')
    for view, title in [('text', 'Text-only comparison'), ('fused', 'Text plus patient features')]:
        p(title, 'Heading2')
        data = [['Text size', 'Classifier', 'Accuracy', 'Precision*', 'Recall*', 'F1*']]
        for pc in [768, 64]:
            for c in NAMES:
                r = byid[f'{view}_{pc}_{c}']
                data.append(['768-D' if pc == 768 else 'PCA-64', NAMES[c]] + [pct(r['metrics'][k]) for k in ['accuracy', 'precision_macro', 'recall_macro', 'macro_f1']])
        export_table(data, view + '_results_table', '*Precision, recall and F1 are macro-averaged; identical test rows for every model.')
        table(data, [53, 146, 70, 70, 70, 65], 7.6)
    variance = byid['text_64_logreg']['pca_retained_variance']
    dims = byid['fused_768_logreg']['feature_count'] - 768
    p(f'*Precision, recall and F1 are macro-averaged. Fusion adds {dims} patient features: 768 + {dims} = {768 + dims}, or 64 + {dims} = {64 + dims} total inputs. PCA retains {variance:.2%} of development embedding variance.', 'Small')
    p('Interpretation', 'Heading2')
    p('All classifiers are newly fitted to the four-level targets. The encoder is frozen, not fine-tuned. Five-fold development CV chooses the GUI model before holdout evaluation (page 8). Scores measure agreement with dataset labels, not clinical validation.', 'Small')
    for view, title in [('text', 'Text-only performance'), ('fused', 'Performance with patient features')]:
        page(title)
        for metric, ylab in [('accuracy', 'Accuracy'), ('precision_macro', 'Macro precision')]:
            fig, ax = plt.subplots(figsize=(8.2, 4.1))
            for j, (pc, color, lab) in enumerate([(768, '#1f77b4', 'Full 768-D'), (64, '#ff7f0e', 'PCA-64')]):
                values = [byid[f'{view}_{pc}_{c}']['metrics'][metric] for c in NAMES]
                bars = ax.bar(np.arange(3) + (j - 0.5) * 0.34, values, width=0.34, label=lab, color=color)
                for bar, v in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.016, f'{v:.1%}', ha='center', fontsize=8)
            ax.set_xticks(range(3), ['Logistic\nRegression', 'Hist Gradient\nBoosting', 'Random\nForest'])
            ax.set_ylabel(ylab)
            ax.set_xlabel('Classifier')
            ax.set_ylim(0, 1.12)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.set_title('Identical test rows: full embeddings versus PCA-64', fontsize=10)
            ax.legend(fontsize=8, loc='lower right')
            fig.tight_layout()
            path = figures / f'{view}_{metric}.png'
            fig.savefig(path, dpi=200)
            plt.close(fig)
            story.extend([Image(str(path), width=490, height=245), Spacer(1, 14)])
        p('Blue bars: original 768-dimensional embeddings. Orange bars: the same embeddings reduced to 64 dimensions. Scores are from the fixed test split, not mixed cross-validation results.', 'Small')
    for c in NAMES:
        page(NAMES[c] + ': confusion matrices')
        p(f'Rows show reference triage levels; columns show predictions. Every matrix contains the same {ntest:,} test records. Top row: text only. Bottom row: text plus patient features. Left: full 768-D. Right: PCA-64.')
        fig, axes = plt.subplots(2, 2, figsize=(8, 7))
        for ax, (view, pc, caption) in zip(axes.flat, [('text', 768, '(a) Text only - 768-D'), ('text', 64, '(b) Text only - PCA-64'), ('fused', 768, '(c) Text + patient features - 768-D'), ('fused', 64, '(d) Text + patient features - PCA-64')]):
            matrix = np.array(byid[f'{view}_{pc}_{c}']['confusion'])
            assert matrix.sum() == ntest
            single, sax = plt.subplots(figsize=(4, 3.5))
            sax.imshow(matrix, cmap='Blues', vmin=0, vmax=ntest * 0.5)
            for si in range(4):
                for sj in range(4):
                    sax.text(sj, si, str(matrix[si, sj]), ha='center', va='center', color='white' if matrix[si, sj] > ntest * 0.25 else '#17355b')
            sax.set_xticks([0, 1, 2, 3], ['0', '1', '2', '3'])
            sax.set_yticks([0, 1, 2, 3], ['0', '1', '2', '3'])
            sax.set_xlabel('Predicted level')
            sax.set_ylabel('Reference level')
            sax.set_title(NAMES[c] + ' / ' + view + ' / ' + str(pc) + 'D', fontsize=9)
            single.tight_layout()
            single.savefig(figures / f'{view}_{pc}_{c}_confusion.png', dpi=220)
            plt.close(single)
            ax.imshow(matrix, cmap='Blues', vmin=0, vmax=ntest * 0.5)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=10, color='white' if matrix[i, j] > ntest * 0.25 else '#17355b')
            ax.set_xticks([0, 1, 2, 3], ['0', '1', '2', '3'])
            ax.set_yticks([0, 1, 2, 3], ['0', '1', '2', '3'])
            ax.set_xlabel('Predicted level')
            ax.set_ylabel('Reference level')
            ax.set_title(caption, fontsize=9)
        fig.tight_layout(pad=1.5)
        path = figures / f'{c}_confusion_grid.png'
        fig.savefig(path, dpi=220)
        plt.close(fig)
        story.append(Image(str(path), width=490, height=429))
        p('Figure: ' + NAMES[c] + ' under the four input conditions. Only the representation/input condition changes; the classifier settings and split stay fixed. Each matrix contains all four levels: 0 Emergency, 1 Urgent, 2 Standard and 3 Non-urgent.', 'Small')
    page('Section 3.6 | Comparison and reproducibility')
    p('Performance comparison of ED triage models', 'Heading2')
    data = [['Model', 'Accuracy', 'Recall', 'F1'], ['TF-IDF + Logistic Regression [1]', '0.750', '0.988', '0.544'], ['TF-IDF + Random Forest [1]', '0.751', '0.964', '0.582'], ['BiLSTM [1]', '0.746', '0.846', '0.670'], ['CNN [1]', '0.723', '0.787', '0.667']]
    for c in NAMES:
        r = byid[f'fused_64_{c}']
        data.append([f'Current {NAMES[c]} + PCA-64'] + [f"{r['metrics'][k]:.4f}" for k in ['accuracy', 'recall_macro', 'macro_f1']])
    data.append(['Selected ' + NAMES[selected['config']['classifier']] + ' + PCA-64'] + [f"{selected['metrics'][k]:.4f}" for k in ['accuracy', 'recall_macro', 'macro_f1']])
    export_table(data, 'literature_comparison_table', 'Different datasets/tasks: contextual comparison only. Current recall and F1 are macro-averaged.')
    table(data, [280, 75, 75, 65], 8)
    p('Table layout follows the supplied screenshot. Current rows use combined inputs and macro recall/F1. Literature rows use a different binary KTAS task and the published metric definitions, so this is context, not a like-for-like superiority benchmark.', 'Small')
    p('Fixed classifier settings', 'Heading2')
    p('Logistic Regression: C = 1, balanced training weights, 2,000 maximum iterations. Hist Gradient Boosting: 300 iterations, learning rate 0.1, other settings at the uploaded baseline defaults. Random Forest: 300 trees, minimum leaf size 3, max_features = 0.7, balanced training weights. Random seed is 42. These settings define the 12 baseline comparisons. A separate, prespecified 10-configuration search selects the serving model on development CV only; no post-test tuning or probability adjustment is used.', 'Small')
    p('Data and representation controls', 'Heading2')
    p('Groups connect repeated normalized complaints or clinical concepts. Numeric imputation/scaling and categorical encoding are fitted on development rows. The full SVD PCA solver is fitted once per input view using development embeddings. The same encoder, labels, rows and classifier settings are retained across the four conditions.', 'Small')
    p('The workbook contains 10,000 records. Its 4,290 missing concepts were recovered only after all 11 complaint/patient inputs and populated concepts matched the previous supplied file row-for-row; new labels were preserved. Triage_Label, Category and processing metadata are excluded. Source checksums, splits and all predictions are saved locally.', 'Small')
    p('[1] Seo et al. (2025). Artificial intelligence for severity triage based on conversations in an emergency department in Korea. Scientific Reports 15, 16870. DOI: <link href="https://doi.org/10.1038/s41598-025-99874-0">10.1038/s41598-025-99874-0</link>, Table 2. The publication uses binary KTAS and ten-fold cross-validation.', 'Small')

    p('The provider described labeling as “by using all”; the exact process and independent per-record review are not documented. Evaluation uses supplied clinical concepts; live Ollama translation performance has not been established. The GUI retains the anatomical gate. A missing complaint displays 50% as a placeholder and no triage level.', 'Small')
    page('Selected model | Validation and class performance')
    p('Five-fold grouped development cross-validation', 'Heading2')
    data = [['Candidate', 'Accuracy', 'Macro F1 +/- SD', 'L0 recall', 'Under-triage']]
    for _, r in cv.iterrows():
        data.append([r['candidate'], pct(r.accuracy_mean),
                     f'{100*r.macro_f1_mean:.2f} +/- {100*r.macro_f1_std:.2f}',
                     pct(r.emergency_recall_mean), pct(r.under_triage_mean)])
    table(data, [150, 75, 112, 75, 82], 7.5, padding=3)
    p('Selection: highest mean macro F1, with emergency recall and lower under-triage as tie breakers. PCA and patient preprocessing are fitted separately inside every training fold. SD is across folds; it is not a confidence interval.', 'Small')
    p('GUI model: ' + NAMES[selected['config']['classifier']] + ' + PCA-64', 'Heading2')
    p('Selected settings: ' + ', '.join(f'{k}={v}' for k, v in selected['config']['params'].items()) + '. Classifier seed: 42.', 'Small')
    m = selected['metrics']
    p(f"Held-out accuracy {pct(m['accuracy'])}; macro precision {pct(m['precision_macro'])}; macro recall {pct(m['recall_macro'])}; macro F1 {pct(m['macro_f1'])}. Under-triage {pct(m['under_triage_rate'])}; over-triage {pct(m['over_triage_rate'])}. Selected candidate: {selected['candidate']}.", 'Small')
    ci = verification['selected_group_bootstrap_95_ci']
    p(f"Group-bootstrap 95% intervals: accuracy {pct(ci['accuracy'][0])} to {pct(ci['accuracy'][1])}; macro F1 {pct(ci['macro_f1'][0])} to {pct(ci['macro_f1'][1])} (1,000 resamples of held-out groups).", 'Small')
    data = [['Level', 'Precision', 'Recall', 'F1', 'Support']]
    for level, name in enumerate(plan['label_names']):
        r = selected['report'][str(level)]
        data.append([f'{level} {name}'] + [pct(r[k]) for k in ['precision', 'recall', 'f1-score']] + [str(int(r['support']))])
    export_table(data, 'selected_class_metrics', 'Selected on development cross-validation; values measured on the fixed holdout.')
    table(data, [160, 85, 85, 85, 75], 8, padding=4)
    matrix = np.array(selected['confusion'])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.imshow(matrix, cmap='Blues')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matrix[i,j]), ha='center', va='center',
                    color='white' if matrix[i,j] > matrix.max()/2 else '#17355b')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xlabel('Predicted level'); ax.set_ylabel('Reference level')
    ax.set_title('Selected model: held-out confusion matrix', fontsize=9)
    fig.tight_layout()
    path = figures / 'selected_confusion.png'
    fig.savefig(path, dpi=220); plt.close(fig)
    story.append(Image(str(path), width=235, height=179))

    # Additional publication figures accompany the PDF as standalone PNGs.
    for view in ['text', 'fused']:
        for metric in ['recall_macro', 'macro_f1']:
            fig, ax = plt.subplots(figsize=(8.2, 4.1))
            for j, (pc, color, label) in enumerate([(768, '#1f77b4', 'Full 768-D'), (64, '#ff7f0e', 'PCA-64')]):
                values = [byid[f'{view}_{pc}_{c}']['metrics'][metric] for c in NAMES]
                bars = ax.bar(np.arange(3) + (j - .5) * .34, values, width=.34, color=color, label=label)
                ax.bar_label(bars, labels=[pct(v) for v in values], padding=3, fontsize=8)
            ax.set_xticks(range(3), ['Logistic Regression', 'Hist Gradient Boosting', 'Random Forest'])
            ax.set_ylim(0, 1.1); ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(view.title() + ' inputs: identical held-out rows')
            ax.legend(loc='lower right'); fig.tight_layout()
            fig.savefig(figures / f'{view}_{metric}.png', dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(cv.candidate.iloc[::-1], cv.macro_f1_mean.iloc[::-1],
            xerr=cv.macro_f1_std.iloc[::-1], color='#1f77b4', capsize=3)
    ax.set_xlim(0, 1); ax.set_xlabel('Development macro F1 (mean +/- fold SD)')
    ax.set_title('Five-fold grouped CV: model selected before holdout evaluation')
    fig.tight_layout(); fig.savefig(figures / 'cv_macro_f1.png', dpi=220); plt.close(fig)

    def footer(canvas, doc):
        canvas.setFont('Vera', 8)
        canvas.drawString(42, 25, f'ML_Predictor | {encoder} research comparison')
        canvas.drawRightString(A4[0] - 42, 25, str(doc.page))
    output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(output), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=38, bottomMargin=42, title=f'{encoder} Full 768-D versus PCA-64 Comparison').build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    main(a.results, a.output)
