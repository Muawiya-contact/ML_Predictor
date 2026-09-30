"""Render all twelve fixed comparisons in the layout requested in the video."""
import argparse, json
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

    def table(data, widths, size=8.5):
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, -1), 'Vera'), ('FONTNAME', (0, 0), (-1, 0), 'VeraBold'), ('FONTSIZE', (0, 0), (-1, -1), size), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), 0.7, colors.black), ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.black), ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5), ('ALIGN', (2, 1), (-1, -1), 'RIGHT')]))
        story.extend([t, Spacer(1, 10)])

    def export_table(data, name, note):
        fig, ax = plt.subplots(figsize=(10, 0.36 * len(data) + 0.8))
        ax.axis('off')
        widths = [0.15, 0.36, 0.17, 0.17, 0.15] if len(data[0]) == 5 else [0.58, 0.14, 0.14, 0.14]
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
    page(f'{encoder} Classifier Comparison')
    p('Sections 3.4 and 3.5 | Full 768-D versus PCA-64', 'Heading2')
    p(f'Encoder: {encoder}. Frozen, normalized 768-dimensional clinical-concept embeddings. PCA reduces the same embeddings to 64 dimensions. Each classifier is evaluated on both text-only and text-plus-patient inputs, producing all 12 comparisons.')
    p(f'Dataset: {ntrain + ntest:,} supplied cardiac records. Fixed grouped split: {ntrain:,} development and {ntest:,} test rows; labels 1-3. Every comparison uses the same rows. PCA and patient-feature preprocessing are fitted on development rows only. Classifier settings are fixed across representations.')
    for view, title in [('text', 'Text-only comparison'), ('fused', 'Text plus patient features')]:
        p(title, 'Heading2')
        data = [['Text size', 'Classifier', 'Accuracy', 'Precision*', 'F1*']]
        for pc in [768, 64]:
            for c in NAMES:
                r = byid[f'{view}_{pc}_{c}']
                data.append(['768-D' if pc == 768 else 'PCA-64', NAMES[c]] + [pct(r['metrics'][k]) for k in ['accuracy', 'precision_macro', 'macro_f1']])
        export_table(data, view + '_results_table', '*Precision and F1 are macro-averaged; identical test rows for every model.')
        table(data, [60, 165, 82, 82, 70])
    variance = byid['text_64_logreg']['pca_retained_variance']
    dims = byid['fused_768_logreg']['feature_count'] - 768
    p(f'*Precision and F1 are macro-averaged. Fusion adds {dims} patient features: 768 + {dims} = {768 + dims}, or 64 + {dims} = {64 + dims} total inputs. PCA retains {variance:.2%} of development embedding variance.', 'Small')
    p('Interpretation', 'Heading2')
    p('This is a fixed follow-up comparison on the existing test split, not new independent validation or test-based model selection. The dataset provider confirmed AI-assigned labels. The application model and safety checks are unchanged.', 'Small')
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
            for si in range(3):
                for sj in range(3):
                    sax.text(sj, si, str(matrix[si, sj]), ha='center', va='center', color='white' if matrix[si, sj] > ntest * 0.25 else '#17355b')
            sax.set_xticks([0, 1, 2], ['1', '2', '3'])
            sax.set_yticks([0, 1, 2], ['1', '2', '3'])
            sax.set_xlabel('Predicted level')
            sax.set_ylabel('Reference level')
            sax.set_title(NAMES[c] + ' / ' + view + ' / ' + str(pc) + 'D', fontsize=9)
            single.tight_layout()
            single.savefig(figures / f'{view}_{pc}_{c}_confusion.png', dpi=220)
            plt.close(single)
            ax.imshow(matrix, cmap='Blues', vmin=0, vmax=ntest * 0.5)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=10, color='white' if matrix[i, j] > ntest * 0.25 else '#17355b')
            ax.set_xticks([0, 1, 2], ['1', '2', '3'])
            ax.set_yticks([0, 1, 2], ['1', '2', '3'])
            ax.set_xlabel('Predicted level')
            ax.set_ylabel('Reference level')
            ax.set_title(caption, fontsize=9)
        fig.tight_layout(pad=1.5)
        path = figures / f'{c}_confusion_grid.png'
        fig.savefig(path, dpi=220)
        plt.close(fig)
        story.append(Image(str(path), width=490, height=429))
        p('Figure: ' + NAMES[c] + ' under the four input conditions. Only the representation/input condition changes; the classifier settings and split stay fixed. Matrices have three levels because the supplied dataset contains no level-4 records.', 'Small')
    page('Section 3.6 | Comparison and reproducibility')
    p('Performance comparison of ED triage models', 'Heading2')
    data = [['Model', 'Accuracy', 'Recall', 'F1'], ['TF-IDF + Logistic Regression [1]', '0.750', '0.988', '0.544'], ['TF-IDF + Random Forest [1]', '0.751', '0.964', '0.582'], ['BiLSTM [1]', '0.746', '0.846', '0.670'], ['CNN [1]', '0.723', '0.787', '0.667']]
    for c in NAMES:
        r = byid[f'fused_64_{c}']
        data.append([f'Current {NAMES[c]} + PCA-64'] + [f"{r['metrics'][k]:.4f}" for k in ['accuracy', 'recall_macro', 'macro_f1']])
    export_table(data, 'literature_comparison_table', 'Different datasets/tasks: contextual comparison only. Current recall and F1 are macro-averaged.')
    table(data, [280, 75, 75, 65], 8)
    p('Table layout follows the supplied screenshot. Current rows use combined inputs and macro recall/F1. Literature rows use a different binary KTAS task and the published metric definitions, so this is context, not a like-for-like superiority benchmark.', 'Small')
    p('Fixed classifier settings', 'Heading2')
    p('Logistic Regression: C = 1, balanced training weights, 2,000 maximum iterations. Hist Gradient Boosting: 300 iterations, learning rate 0.1, other settings at the uploaded baseline defaults. Random Forest: 300 trees, minimum leaf size 3, max_features = 0.7, balanced training weights. Random seed is 42. No post-test parameter tuning or probability adjustment is used.', 'Small')
    p('Data and representation controls', 'Heading2')
    p('Groups connect repeated normalized complaints or clinical concepts. Numeric imputation/scaling and categorical encoding are fitted on development rows. The full SVD PCA solver is fitted once per input view using development embeddings. The same encoder, labels, rows and classifier settings are retained across the four conditions.', 'Small')
    p('The complete result set contains two input views, two text representations and three classifiers. All twelve configurations and their predictions are saved with input checksums for reproducibility.', 'Small')
    p('[1] Seo et al. (2025). Artificial intelligence for severity triage based on conversations in an emergency department in Korea. Scientific Reports 15, 16870. DOI: 10.1038/s41598-025-99874-0, Table 2. The publication uses binary KTAS and ten-fold cross-validation.', 'Small')

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
