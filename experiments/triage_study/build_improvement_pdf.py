"""Single verified report with original comparisons and optional follow-up audits."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

NAMES = {'logreg': 'Logistic Regression', 'hgb': 'HistGradientBoosting', 'rf': 'Random Forest',
         'catboost': 'CatBoost', 'xgboost': 'XGBoost', 'svc': 'RBF SVM', 'ordinal': 'Ordinal Logistic', 'mlp': 'Neural classifier', 'soft_vote': 'Probability ensemble'}

def build(source, original, output, audit=None, investigation=None, expanded=None, geometry=None, followup=None):
    if json.loads((source / 'verification.json').read_text())['status'] != 'passed':
        raise ValueError('Verify the comparison before generating the report')
    protocol = json.loads((source / 'protocol.json').read_text())
    selection = json.loads((source / 'selection.json').read_text())
    results = json.loads((source / 'retrospective_results.json').read_text())
    winner = next((r for r in results if r['candidate'] == selection['candidate']))
    baseline = results[0]
    previous_path = source / 'previous_round_result.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else baseline
    cv = pd.read_csv(source / 'cv_summary.csv')
    family_path = source / 'family_results.json'
    families = json.loads(family_path.read_text()) if family_path.exists() else []
    old = json.loads((original / 'results.json').read_text())
    figures = source / 'figures'
    figures.mkdir(exist_ok=True)
    fonts = Path(reportlab.__file__).parent / 'fonts'
    pdfmetrics.registerFont(TTFont('Vera', str(fonts / 'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('VeraBold', str(fonts / 'VeraBd.ttf')))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.textColor = colors.black
        style.fontName = 'VeraBold' if style.name in ['Title', 'Heading1', 'Heading2', 'Heading3'] else 'Vera'
    styles['Title'].fontSize = 17
    styles['Title'].leading = 21
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 12
    story = []

    def p(text, style='BodyText'):
        story.extend([Paragraph(text, styles[style]), Spacer(1, 7)])

    def page(title):
        if story:
            story.append(PageBreak())
        p(title, 'Title')

    def table(rows, widths=None, size=8, padding=5):
        t = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, -1), 'Vera'), ('FONTNAME', (0, 0), (-1, 0), 'VeraBold'), ('FONTSIZE', (0, 0), (-1, -1), size), ('LEADING', (0, 0), (-1, -1), size + 2), ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f4d35e')), ('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.black), ('BOTTOMPADDING', (0, 0), (-1, -1), padding), ('TOPPADDING', (0, 0), (-1, -1), padding), ('VALIGN', (0, 0), (-1, -1), 'TOP')]))
        story.extend([t, Spacer(1, 10)])

    def pic(path, width=495, height=290):
        story.append(Image(str(path), width=width, height=height, kind='proportional'))

    def pct(x):
        return f'{100 * x:.2f}'

    def matrix(ax, cm, title):
        ax.imshow(cm, cmap='Blues')
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xlabel('Predicted level')
        ax.set_ylabel('Reference level')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=12, color='white' if cm[i][j] > np.max(cm) / 2 else 'black')
    for result in families:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        matrix(ax, result['confusion'], 'Tuned ' + result['config']['classifier'])
        fig.tight_layout()
        fig.savefig(figures / ('tuned_' + result['config']['classifier'] + '_confusion.png'), dpi=190)
        plt.close(fig)
    if families:
        pd.DataFrame([dict(candidate=r['candidate'], classifier=r['config']['classifier'], **r['metrics']) for r in families]).to_csv(source / 'family_metrics.csv', index=False)
    page('Four-Level SapBERT: Model Comparison')
    p('Revised comparison | Levels 0 Emergency, 1 Urgent, 2 Standard, 3 Non-urgent', 'Heading2')
    p('This improvement study uses the same 10,000 supplied records, frozen SapBERT embeddings and original grouped partitions. It expands the search to PCA-64, PCA-128 and PCA-256, stronger/weaker regularization, class balancing, PCA whitening and structured-only controls. No labels are changed.')
    classifier_name = NAMES[selection['config']['classifier']]
    p(f"Selected: SapBERT + PCA-{selection['config']['features']['pca']} + {classifier_name}", 'Heading2')
    if expanded is not None and selection['candidate'] == previous['candidate']:
        p('The new CatBoost, XGBoost and SVM trials did not beat this configuration. The existing model is retained; the 90% target across all four aggregate metrics remains unmet.')
    if selection['config']['features'].get('encoder') == 'sapbert_pair':
        p('SapBERT now receives the supplied clinical concept plus the original complaint, separated by [SEP], with a 128-token limit. The live app uses locally translated English plus the original complaint.')
    if selection['config'].get('components'):
        parts = [f"{100*c['weight']:.0f}% {NAMES[c['config']['classifier']]}" for c in selection['config']['components']]
        p('Fixed probability weights: ' + ', '.join(parts) + '. Each component is fitted on development rows only.')
    p('Training settings: C=' + str(selection['config']['params'].get('C', 'see protocol')) + '; balanced class weights.' if selection['config']['params'].get('balance') else 'Training settings are listed in the protocol.')
    if selection['config']['features'].get('polynomial'):
        p('Patient preprocessing includes quadratic numeric terms. SapBERT remains frozen.')
    if selection['config'].get('text_details'):
        p('An additional 19 explicit detail features preserve severity, onset, duration and other mentions from the original complaint alongside the SapBERT concept embedding.')
    p(f"Selected total input size: {winner.get('feature_count', 'see manifest')} features. PCA retained variance: {(pct(winner['pca_retained_variance']) + '%' if winner.get('pca_retained_variance') is not None else 'not applicable')}.")
    table([['Retrospective metric (%)', 'Initial', 'Prior round', 'Selected']] + [[label, pct(baseline['metrics'][key]), pct(previous['metrics'][key]), pct(winner['metrics'][key])] for key, label in [('accuracy', 'Accuracy'), ('precision_macro', 'Macro precision'), ('recall_macro', 'Macro recall'), ('macro_f1', 'Macro F1'), ('emergency_recall', 'Emergency recall'), ('under_triage_rate', 'Under-triage')]], [225, 90, 90, 90])
    p(f"Selected quadratic weighted kappa: {winner['metrics']['qwk']:.4f}; mean absolute level error: {winner['metrics']['mae']:.4f}; over-triage: {pct(winner['metrics']['over_triage_rate'])}% (initial {pct(baseline['metrics']['over_triage_rate'])}%).")
    probability_diagnostics = json.loads((source / 'verification.json').read_text()).get('selected_probability_diagnostics', {})
    if probability_diagnostics:
        d = probability_diagnostics['retrospective_test']
        p(f"Selected retrospective MCC: {d['mcc']:.4f}; log loss: {d['log_loss']:.4f}; multiclass Brier score: {d['multiclass_brier']:.4f}.")
    p(f"Mean development CV macro F1 change versus initial: {selection['cv_gain'] * 100:+.2f} percentage points. Selection uses all five development folds and an emergency-recall constraint; the test results above do not select the winner.")
    uncertainty=json.loads((source/'verification.json').read_text())['development_oof_paired_group_bootstrap']
    lo,hi=uncertainty['ci95']
    p(f"Paired development bootstrap versus initial: F1 gain 95% interval {100*lo:+.2f} to {100*hi:+.2f} percentage points. {'It includes zero; the gain is not yet statistically established.' if lo <= 0 <= hi else 'This conditional interval excludes zero.'} This diagnostic excludes model-selection uncertainty.")
    p('<b>Evaluation scope:</b> These 1,999 test rows were already examined in the previous study. The new scores are retrospective comparisons, not a fresh independent estimate. New labelled records are needed to confirm generalization. Saved concepts are evaluated here; live translation accuracy is not measured.')
    p('Trade-offs remain visible: compare emergency recall and under-triage as well as aggregate scores. SapBERT is not fine-tuned; these gains do not establish clinical validity.')
    fig, ax = plt.subplots(figsize=(9, 4))
    keys = ['accuracy', 'precision_macro', 'recall_macro', 'macro_f1', 'emergency_recall']
    for offset, result, label, color in [(-0.18, baseline, 'Initial baseline', '#2874ad'), (0.18, winner, 'Selected', '#efa928')]:
        ax.bar(np.arange(len(keys)) + offset, [result['metrics'][k] * 100 for k in keys], 0.36, label=label, color=color)
    ax.set_xticks(range(len(keys)), ['Accuracy', 'Macro precision', 'Macro recall', 'Macro F1', 'Emergency recall'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Retrospective score (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'previous_selected_metrics.png', dpi=190)
    plt.close(fig)
    page('Complete Development Cross-Validation')
    trainable = sum(c['classifier'] != 'soft_vote' for c in protocol['candidates'].values())
    p(f'All {len(cv)} configurations; {trainable * 5} training fits and {(len(cv)-trainable)*5} reused-probability evaluations across five grouped folds. F1, accuracy and emergency recall are percentages. SD is the fold-to-fold F1 standard deviation in percentage points. The original reference is lr_pca64_c10_balanced1.')
    chunks = [cv.iloc[i:i+45] for i in range(0, len(cv), 45)]
    for index, chunk in enumerate(chunks):
        if index:
            page('Development Cross-Validation: Continued')
        table([['Configuration', 'F1', 'SD', 'Accuracy', 'Emergency recall']] + [[r.candidate, pct(r.macro_f1), pct(r.f1_std), pct(r.accuracy), pct(r.emergency_recall)] for r in chunk.itertuples()], [215, 60, 50, 70, 100], 8, 0.7 if len(chunk) > 50 else (1.2 if len(chunk) > 40 else 2))
    p('Selection: highest mean macro F1 among candidates whose mean emergency recall is within one percentage point of the original reference; accuracy resolves ties. A higher F1 does not qualify a candidate whose emergency recall falls below the threshold. This is not a clinical safety guarantee.')
    page('Original 768-D versus PCA-64 Baselines')
    p('All original fixed comparisons are retained below. They use the same 1,999 previously examined test rows. Values are percentages; precision, recall and F1 are macro-averaged. These baseline runs are from the first round and are not new experiments.')
    for view, label in [('text', 'Text only'), ('fused', 'Text plus patient features')]:
        p(label, 'Heading2')
        names = {'logreg': 'Logistic Regression', 'hgb': 'HistGradientBoosting', 'rf': 'Random Forest'}
        rows = [['Text size', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1']]
        for r in old:
            if r['config']['features']['view'] == view:
                rows.append([str(r['config']['features']['pca'] or 768), names[r['config']['classifier']]] + [pct(r['metrics'][k]) for k in ['accuracy', 'precision_macro', 'recall_macro', 'macro_f1']])
        table(rows, [55, 155, 72, 72, 70, 70], 7.5)
    p('The original fused PCA-64 input has 86 features: 64 text components and 22 encoded patient features. The full fused input has 790 features. All training transformations are fitted without validation/test rows.')
    page('Performance Graphs')
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for col, view in enumerate(['text', 'fused']):
        for row, metric in enumerate(['accuracy', 'precision_macro']):
            ax = axes[row, col]
            for j, pc in enumerate([0, 64]):
                rows = [r for r in old if r['config']['features']['view'] == view and r['config']['features']['pca'] == pc]
                ax.bar(np.arange(3) + (j - 0.5) * 0.35, [r['metrics'][metric] * 100 for r in rows], width=0.35, label='768-D' if pc == 0 else 'PCA-64', color=['#2874ad', '#efa928'][j])
            ax.set_xticks(range(3), ['LR', 'HGB', 'RF'])
            ax.set_ylim(0, 100)
            ax.set_ylabel(('Accuracy' if row == 0 else 'Macro precision') + ' (%)')
            ax.set_title('Text only' if view == 'text' else 'Text plus patient features')
            ax.legend()
    fig.tight_layout()
    path = figures / 'accuracy_precision_comparison.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    pic(path, height=470 if len(families) > 3 else (310 if families else 470))
    if families:
        if len(families) > 3:
            page('Expanded Classifier Comparison')
        p('Tuned classifiers: retrospective comparison', 'Heading2')
        fig, ax = plt.subplots(figsize=(10, 4 if len(families) > 3 else 2.2))
        for offset, key, label, color in [(-.18, 'accuracy', 'Accuracy', '#2874ad'), (.18, 'macro_f1', 'Macro F1', '#efa928')]:
            bars = ax.bar(np.arange(len(families)) + offset, [r['metrics'][key] * 100 for r in families], .36, label=label, color=color)
            ax.bar_label(bars, fmt='%.2f', fontsize=8)
        ax.set_xticks(range(len(families)), [NAMES[r['config']['classifier']] for r in families], rotation=35 if len(families) > 3 else 0, ha='right' if len(families) > 3 else 'center')
        ax.set_ylim(0, 110)
        ax.set_ylabel('Score (%)')
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        latest_plot = figures / 'tuned_family_metrics.png'
        fig.savefig(latest_plot, dpi=180)
        plt.close(fig)
        pic(latest_plot, height=235 if len(families) > 3 else 115)

        table([['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1']] + [[NAMES[r['config']['classifier']]] + [pct(r['metrics'][k]) for k in ['accuracy', 'precision_macro', 'recall_macro', 'macro_f1']] for r in families], [175, 80, 80, 80, 80], 8)
        p('The family representatives can use different text representations; this compares their selected pipelines, not a controlled classifier-only effect. Each family uses its highest fused development CV F1 setting. These descriptive choices do not override the emergency-recall constraint for deployment. Scores are percentages; precision, recall and F1 are macro averages.')
    p('Blue and yellow bars preserve the original comparison style. The selected improved result is shown separately on page 1 because it follows a larger development search.')
    for classifiers, title, filename in [(['logreg', 'hgb'], 'Baseline Confusion Matrices: LR and HGB', 'lr_hgb_matrices'), (['rf'], 'Baseline and Selected Confusion Matrices', 'rf_selected_matrices')]:
        page(title)
        rows = [r for r in old if r['config']['classifier'] in classifiers]
        if classifiers == ['rf']:
            rows.append(dict(id='Selected improved model', confusion=winner['confusion']))
            for r in families if len(families) <= 3 else []:
                rows.append(dict(id='Tuned ' + r['config']['classifier'], confusion=r['confusion']))
        nrows = (len(rows) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(9, 3 * nrows), squeeze=False)
        for ax, r in zip(axes.flat, rows):
            matrix(ax, r['confusion'], r['id'])
        for ax in list(axes.flat)[len(rows):]:
            ax.axis('off')
        fig.tight_layout()
        path = figures / (filename + '.png')
        fig.savefig(path, dpi=190)
        plt.close(fig)
        pic(path, height=600)
        p('Rows are reference labels; columns are predictions. Level order: 0 Emergency, 1 Urgent, 2 Standard, 3 Non-urgent. Baseline matrices are unchanged from round one.', 'BodyText')
    if len(families) > 3:
        for start in range(0,len(families),6):
            group=families[start:start+6]
            page('Tuned Classifier Confusion Matrices' + (f' ({start//6+1})' if len(families)>6 else ''))
            nrows=(len(group)+1)//2
            fig, axes = plt.subplots(nrows, 2, figsize=(9, 3.5*nrows), squeeze=False)
            for ax, result in zip(axes.flat, group):
                matrix(ax, result['confusion'], NAMES[result['config']['classifier']])
            for ax in list(axes.flat)[len(group):]:
                ax.axis('off')
            fig.tight_layout()
            path = figures / f'expanded_family_matrices_{start//6+1}.png'
            fig.savefig(path, dpi=190)
            plt.close(fig)
            pic(path, height=595 if nrows==3 else 450)
            p('All matrices contain the same 1,999 records. Rows are reference labels and columns are predictions, in level order 0, 1, 2, 3. Family representatives are selected by development scores.')
    page('Class Results and Error Review')
    table([['Level', 'Precision (%)', 'Recall (%)', 'F1 (%)', 'Support']] + [[f'{i} ' + name] + [pct(winner['report'][str(i)][k]) for k in ['precision', 'recall', 'f1-score']] + [str(int(winner['report'][str(i)]['support']))] for i, name in enumerate(['Emergency', 'Urgent', 'Standard', 'Non-urgent'])], [140, 95, 90, 90, 80])
    oof = np.load(source / 'development_oof.npz')
    cm = np.zeros((4, 4), dtype=int)
    for y, pred in zip(oof['reference'], oof['selected'].argmax(1)):
        cm[y, pred] += 1
    fig, ax = plt.subplots(figsize=(6, 4.5))
    matrix(ax, cm, 'Selected model: development out-of-fold errors')
    fig.tight_layout()
    path = figures / 'development_confusion.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    p('Development out-of-fold confusion matrix', 'Heading2')
    table([['Reference / predicted', '0', '1', '2', '3']] + [[str(i)] + [str(v) for v in row] for i, row in enumerate(cm)], [175, 80, 80, 80, 80], 8)
    p('An input audit found no identical development input combinations with conflicting targets. This does not establish that labels are clinically correct. A local review CSV lists development errors and model confidence; it is not committed because it contains individual records.')
    p('Use the review list to investigate unclear label boundaries, missing symptom severity/duration/history and data entry issues. Do not replace reference labels with model predictions merely to improve agreement.')
    p('Literature context (different task)', 'Heading2')
    literature = json.loads((original / 'literature_sources.json').read_text())
    table([['Model', 'Accuracy (%)', 'Recall (%)', 'F1 (%)']] + [[name, pct(m['accuracy']), pct(m['recall']), pct(m['f1'])] for name, m in literature['models'].items()] + [['Current four-level SapBERT', pct(winner['metrics']['accuracy']), pct(winner['metrics']['recall_macro']), pct(winner['metrics']['macro_f1'])]], [215, 94, 93, 93], 7.5)
    p('Seo et al., Scientific Reports (2025), Table 2. DOI: 10.1038/s41598-025-99874-0. Different binary KTAS task; publication metric definitions differ. Context only, not evidence of superiority over these models.')
    page('Methods, Reproducibility and Limits')
    p('Data and preprocessing', 'Heading2')
    p('The original 8,001 development / 1,999 test partition and five grouped folds are reused. Groups do not overlap across training and validation/test boundaries. Input and embedding hashes are checked before and after the run. Median imputation, categorical encoding, scaling and PCA are learned inside each development fold. No label-derived columns are model inputs.')
    p('Encoder and search', 'Heading2')
    p('SapBERT-from-PubMedBERT-fulltext uses frozen 768-D CLS embeddings, L2 normalization, the original pinned revision and a 64-token limit for the original concept-only experiments. The selected paired-text model uses a 128-token limit. Logistic Regression varies C=10, 100 and 1000 with and without balancing, plus whitened PCA with C=0.1. Six additional LR configurations use quadratic numeric terms, with C=1/10/100 and optional balancing. HGB tests 7/15 leaves with regularization; Random Forest tests structured and fused inputs. Every candidate is evaluated on all five folds.')
    if len(cv) > 35:
        p('The refinement adds 14 configurations: balanced LR C=30/300 at PCA-64/128; quadratic LR C=100/300 at PCA-128/256; HGB with 7/31 leaves and 800/400 iterations; and RF with 500 trees, square-root feature sampling and leaf sizes 1/3. These historical refinement settings retained the original three classifier families.')
    if selection['config'].get('text_details'):
        p('Six additional settings compare 19 explicit details extracted from English concepts or original complaints, using each family’s prior best setting. The separate learning-curve audit uses nested whole-group subsets; labels remain unchanged.')
    p('Selection and deployment', 'Heading2')
    if expanded is not None:
        p('The expanded comparison adds eleven settings: CatBoost depth 4/6 with 800 iterations and XGBoost depth 4/6 with 600 trees, each at PCA-64/128; RBF SVM C=1/10/100 at PCA-128. All retain original-complaint detail features. New dependencies and parameters are recorded with the experiment.')
    p('The incumbent is included in the same search. Selection is recorded before the retrospective test evaluation. The exported model must reproduce saved predictions through the shared GUI/CLI preprocessing. Model artefacts and the incumbent are retained separately. Quadratic terms, if selected, are applied by the fitted numeric preprocessing pipeline. No label correction or encoder fine-tuning is claimed.')
    p('Research interpretation', 'Heading2')
    p('The larger search can overfit development cross-validation. The old test set is already exposed, so its scores must not be presented as untouched confirmation. A new independently labelled dataset is needed for confirmation. Supplied/recovered concepts bypass live translation during this evaluation. The provider could not supply label-assignment rules; independent per-record review is not documented. These are research comparisons, not clinical validation.')
    p('Files', 'Heading2')
    p('protocol.json freezes candidates and source identities; cross_validation.csv contains every fit; cv_summary.csv reports every candidate; selection.json records the decision; retrospective_results.json contains class scores and matrices. Source data, embeddings and per-record error files remain local.')
    if audit is not None:
        audit_check = json.loads((audit / 'verification.json').read_text())
        if audit_check['status'] != 'passed':
            raise ValueError('Verify the development audit first')
        detail = json.loads((audit / 'input_error_audit.json').read_text())
        learning = pd.read_csv(audit / 'summary.csv')
        page('Complaint Audit, Learning Curves and Next Steps')
        p(f"The development audit flags {detail['explicit_duration_disagreements']} explicit duration disagreements between original complaints and supplied concepts, {detail['explicit_duration_missing_from_concept']} durations not recovered from concepts and {detail['family_word_missing_from_concept']} family-word omissions. These lexical flags require review; they do not establish incorrect clinical labels. Labels remain unchanged.")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        names = {'logreg': 'Logistic Regression', 'hgb': 'HistGradientBoosting', 'rf': 'Random Forest'}
        for ax, (family, name) in zip(axes, names.items()):
            rows = learning[learning.classifier.eq(family) & learning.variant.eq('baseline')].sort_values('fraction')
            ax.plot(rows.train_rows, 100 * rows.train_f1, 'o-', label='Training', color='#2874ad')
            ax.errorbar(rows.train_rows, 100 * rows.macro_f1, yerr=100 * rows.f1_std, fmt='o-', label='Validation', color='#d49b00', capsize=3)
            ax.set_title(name, fontsize=10); ax.set_xlabel('Training rows per fold'); ax.set_ylim(65, 101); ax.grid(alpha=.2)
        axes[0].set_ylabel('Macro F1 (%)'); axes[-1].legend(fontsize=8)
        fig.tight_layout(); path = figures / 'learning_curves.png'; fig.savefig(path, dpi=200); plt.close(fig)
        pic(path, height=145)
        p('Nested whole-group subsets use 25%, 50%, 75% and 100% of each training fold with unchanged validation rows. Error bars show fold standard deviation. Preprocessing is fitted inside each subset. The audit contains 90 fits, including reproduced baselines; no score at 20,000 rows is extrapolated.')
        rows = [['Full-size development condition', 'Accuracy', 'Macro F1', 'Emergency recall']]
        variants = {'baseline': 'Prior features', 'concept_details': 'English details', 'complaint_details': 'Original details'}
        for family, name in names.items():
            for variant, label in variants.items():
                r = learning[learning.classifier.eq(family) & learning.variant.eq(variant) & learning.fraction.eq(1)].iloc[0]
                rows.append([name + ' / ' + label, pct(r.accuracy), pct(r.macro_f1), pct(r.emergency_recall)])
        table(rows, [250, 75, 75, 95], 7.2, 3)
        p('Recommended next step', 'Heading2')
        p('Review the local error queue and duration mismatches, agree a consistent four-level rubric with qualified reviewers, and collect distinct new examples. More rows may help, but duplicated templates or unclear labels will not reliably solve the remaining Urgent/Standard errors. Reserve new independently reviewed records for confirmation before further selection.')
        p('The GUI retains the original complaint for the 19 detail features and includes it alongside checked English when the paired-text encoder configuration is selected. Similarity and cluster views use full 768-D vectors; stop-word removal is inactive. The report measures supplied concepts plus original details, not end-to-end live translation accuracy.')
    if investigation is not None:
        diag=json.loads((investigation/'diagnostics.json').read_text())
        page('Further Four-Level Error Investigation')
        p('This diagnostic uses only the 8,001 development records and the pre-refinement model. Supplied labels 0, 1, 2 and 3 remain unchanged. It is a diagnostic of model errors, not an independent assessment of clinical label correctness.')
        pic(investigation/'figures/current_error_confidence.png',height=160)
        table([['Pre-refinement development diagnostic','Value'],
               ['Out-of-fold errors',str(diag['errors'])],
               ['Errors between neighbouring levels',str(diag['adjacent_errors'])],
               ['Urgent / Standard confusions',str(diag['urgent_standard_errors'])],
               ['Errors with confidence at least 90%',str(diag['high_confidence_errors_90'])],
               ['All three classifiers agree on the wrong level',str(diag.get('all_three_agree_on_wrong_level','not measured'))],
               ['Matthews correlation coefficient',f"{diag['multiclass_mcc']:.4f}"],
               ['Multiclass log loss',f"{diag['multiclass_log_loss']:.4f}"],
               ['Multiclass Brier score',f"{diag['multiclass_brier']:.4f}"],
               ['Confidence calibration gap (10 bins)',f"{diag['confidence_ece_10_bins']:.4f}"]], [365,130],8,3)
        p('Higher MCC is better; lower log loss, Brier score and calibration gap are better. Confidence bins compare the mean maximum predicted probability with observed accuracy. These conditional development diagnostics do not measure end-to-end translation accuracy.')
        p('The follow-up freezes 13 settings before evaluation: LR regularization and balancing at PCA-128, PCA-64/256 alternatives, HGB leaf/regularization settings and RF feature sampling. All retain the 19 original-complaint details. Every setting uses five grouped folds; the original emergency-recall selection constraint remains fixed.')
        if probability_diagnostics:
            d = probability_diagnostics['development']
            p(f"Selected development diagnostics: MCC {d['mcc']:.4f}; log loss {d['log_loss']:.4f}; Brier score {d['multiclass_brier']:.4f}; calibration gap {d['ece_10_bins']:.4f}.")
        p('The older three-level scores concern a different labelling task. No labels are replaced, no difficult records are dropped and no test-based selection is used.')
        refinement_file=investigation/'refinement_verification.json'
        if refinement_file.exists():
            refinement=json.loads(refinement_file.read_text())
            interval=refinement.get('paired_bootstrap_vs_incumbent')
            if interval:
                lower,upper=interval['ci95']
                p(f"Historical 13-setting refinement versus its prior model: pooled development F1 gain {100*interval['pooled_macro_f1_gain']:+.2f} percentage points; conditional paired group interval {100*lower:+.2f} to {100*upper:+.2f}. {'The interval includes zero, so a reliable gain is not established.' if lower <= 0 <= upper else 'The conditional interval excludes zero.'} Repeated-selection uncertainty is not included.")

    if followup is not None:
        page('Further Methods and Final Selection')
        p('The follow-up evaluates 24 trainable settings covering cubic patient features, cumulative ordinal logistic classifiers and supervised neural classifiers. Six further settings preserve the original complaint alongside the clinical concept in a frozen SapBERT input, using a 128-token limit. SapBERT itself is not fine-tuned.')
        p('An initial fixed-weight comparison is followed by a declared weight refinement that includes the ordered classifier. Across both stages, 24 distinct blends reuse existing out-of-fold probabilities. These are probability evaluations, not additional component training runs. No fitted stacking model sees validation labels.')
        groups=[('Cubic Logistic Regression',lambda n:n.startswith('lr_cubic')),('Ordered Logistic Regression',lambda n:n.startswith('ordinal_')),('Neural classifier',lambda n:n.startswith('mlp_')),('Concept + original complaint',lambda n:n.startswith('lr_pair')),('Fixed probability combinations',lambda n:n.startswith('blend_'))]
        rows=[['Development condition','Accuracy','Precision','Recall','F1','ER recall']]
        for label,match in groups:
            subset=cv[cv.candidate.map(match)].sort_values(['macro_f1','accuracy','candidate'],ascending=[False,False,True])
            if len(subset):
                row=subset.iloc[0]
                rows.append([label]+[pct(row[k]) for k in ['accuracy','precision','recall','macro_f1','emergency_recall']])
        table(rows,[195,60,60,60,60,60],7.3,5)
        p('Rows summarize the highest development F1 setting in each condition. The final deployment additionally applies the unchanged emergency-recall constraint. The complete table earlier in this report includes unsuccessful settings; per-setting precision and recall are also saved in cv_summary.csv.')
        selected=cv[cv.candidate.eq(selection['candidate'])].iloc[0]
        table([['Final model metric','Development CV (%)','Retrospective (%)']]+[[label,pct(selected[a]),pct(winner['metrics'][b])] for label,a,b in [('Accuracy','accuracy','accuracy'),('Macro precision','precision','precision_macro'),('Macro recall','recall','recall_macro'),('Macro F1','macro_f1','macro_f1'),('Emergency recall','emergency_recall','emergency_recall')]], [195,150,150],8)
        p('The 90% objective concerns four aggregate metrics. Individual Urgent and Standard class scores remain below 90%; inspect the separate class table. The old test set remains previously exposed. These observations do not establish that future or independently labelled cases will exceed 90%.')
        uncertainty_file = followup / 'triage_paired_classifiers' / 'verification.json'
        if uncertainty_file.exists() and selection['config']['features'].get('encoder') == 'sapbert_pair':
            checked = json.loads(uncertainty_file.read_text())
            delta = checked.get('paired_bootstrap_vs_incumbent')
            if delta:
                lo,hi=delta['ci95']
                p(f"Paired development F1 change versus the previous model: {100*delta['pooled_macro_f1_gain']:+.2f} percentage points; conditional group-bootstrap 95% interval {100*lo:+.2f} to {100*hi:+.2f}. This interval excludes model-selection uncertainty.")
        p('The selected model preserves original complaint wording in its encoder input. The GUI, batch export and embedding views construct that same paired text. Probability combinations remain comparison candidates; their component mappings are checked before export. Every selected probability must match the live GUI/CLI adapter before promotion.')

    if geometry is not None:
        diag = json.loads((geometry / 'diagnostics.json').read_text())
        page('SapBERT Embedding Geometry')
        p(f"A fixed stratified sample contains {diag['sample_rows']:,} development records, one per complaint group. This diagnostic compares triage-label similarity in frozen full-768 embeddings and the fitted PCA projection. No test rows participate.")
        pic(geometry / 'triage_embedding_projection.png', height=305)
        table([['Representation', 'Within-label', 'Between-label', 'Difference', 'Silhouette']] + [[r['representation']] + [f"{r[k]:.4f}" for k in ['within_label_cosine', 'between_label_cosine', 'separation', 'cosine_silhouette']] for r in diag['metrics']], [140, 90, 90, 80, 95], 8)
        p('Within-label and between-label values are mean pairwise cosine similarities. Their difference describes separation by the supplied triage labels. Silhouette near zero indicates overlapping labels in that representation. PCA centres the vectors, so absolute cosine values across raw and projected spaces are not directly interchangeable.')
        p('The triage-label overlap helps explain why changing only the classifier may give limited gains. SapBERT encodes medical meaning; similar medical complaints can receive different triage levels when patient measurements and symptom details differ. The deployed classifier uses those additional features.')
        p('The plot shows only two principal components. The diagnostic uses a fixed sample and a PCA fitted on all development rows; it is descriptive, not out-of-fold predictive performance or evidence of a maximum achievable accuracy. It does not establish that any supplied label is wrong.')

    if expanded is not None:
        check = json.loads((expanded / 'verification.json').read_text())
        if check['status'] != 'passed':
            raise ValueError('Verify expanded classifiers before reporting')
        expanded_cv = pd.read_csv(expanded / 'cross_validation.csv')
        mean = expanded_cv.groupby('candidate').agg(accuracy=('accuracy', 'mean'), precision=('precision_macro', 'mean'), recall=('recall_macro', 'mean'), f1=('macro_f1', 'mean'), emergency=('emergency_recall', 'mean')).reset_index()
        page('Earlier Classifier Expansion and Current Target' if followup is not None else 'Additional Classifiers and the 90% Target')
        p('This earlier stage tested eleven new settings declared before fitting and evaluated across the same five grouped development folds: 55 additional fits. CatBoost, XGBoost and RBF SVM were compared with the previous 68 configurations. Dataset, labels, encoder and test membership remain unchanged.')
        table([['New configuration', 'Accuracy', 'Precision', 'Recall', 'F1', 'ER recall']] + [[r.candidate] + [pct(v) for v in [r.accuracy, r.precision, r.recall, r.f1, r.emergency]] for r in mean.itertuples()], [180, 63, 63, 63, 63, 63], 7, 4)
        selected_cv = cv[cv.candidate.eq(selection['candidate'])].iloc[0]
        target_rows = [['Metric', 'CV mean (%)', 'Retrospective (%)', 'Target (%)']]
        achieved = True
        for label, cv_key, result_key in [('Accuracy','accuracy','accuracy'), ('Macro precision','precision','precision_macro'), ('Macro recall','recall','recall_macro'), ('Macro F1','macro_f1','macro_f1')]:
            target_rows.append([label, pct(selected_cv[cv_key]), pct(winner['metrics'][result_key]), '90.00'])
            achieved = achieved and selected_cv[cv_key] >= .90 and winner['metrics'][result_key] >= .90
        table(target_rows, [180, 110, 120, 85], 8)
        p('All four aggregate targets were met in these comparisons.' if achieved else '<b>The 90% target was not reached across all four aggregate metrics.</b> Adding these classifier families did not establish the requested result. The best eligible development model is retained; no result is rounded up to imply success.')
        p('The constraint requires mean emergency recall of at least ' + pct(check['minimum_emergency_recall']) + '%. Selection ranks eligible settings by macro F1, then accuracy. The desired 90% aggregate target does not override this rule.')
        if selection['candidate'] == previous['candidate']:
            p('The selected configuration is unchanged from the previous round. Therefore, the deployed model and its measured scores remain unchanged; the new work expands the evidence rather than claiming a gain.')
        p('These scores measure agreement with the supplied four-level labels. They do not establish a maximum achievable score or prove that a label is wrong. The three-level study had different targets, so its higher scores are not a valid requirement for this four-level experiment.')

    output.parent.mkdir(parents=True, exist_ok=True)

    def footer(canvas, doc):
        canvas.setFont('Vera', 8)
        canvas.drawString(40, 25, 'Four-level SapBERT | Research comparison')
        canvas.drawRightString(A4[0] - 40, 25, str(doc.page))
    SimpleDocTemplate(str(output), title='SapBERT Four-Level Improved Comparison', pagesize=A4, rightMargin=40, leftMargin=40, topMargin=35, bottomMargin=40).build(story, onFirstPage=footer, onLaterPages=footer)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit', type=Path)
    p.add_argument('--investigation', type=Path)
    p.add_argument('--expanded', type=Path)
    p.add_argument('--geometry', type=Path)
    p.add_argument('--followup', type=Path)
    a = p.parse_args()
    build(a.source, a.original, a.output, a.audit, a.investigation, a.expanded, a.geometry, a.followup)
