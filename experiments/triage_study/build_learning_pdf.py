"""Publish aggregate development-only learning curves and complaint audit."""
import argparse
import json
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


def build(source, comparison, output):
    audit=json.loads((source/'input_error_audit.json').read_text())
    verification=json.loads((source/'verification.json').read_text())
    assert verification['status']=='passed'
    summary=pd.read_csv(source/'summary.csv')
    selection=json.loads((source/'selection.json').read_text())
    chosen=json.loads((comparison/'selection.json').read_text())
    results=json.loads((comparison/'retrospective_results.json').read_text())
    winner=next(r for r in results if r['candidate']==chosen['candidate'])
    previous=json.loads((comparison/'previous_round_result.json').read_text())
    figures=source/'figures';figures.mkdir(exist_ok=True)
    names={'logreg':'Logistic Regression','hgb':'HistGradientBoosting','rf':'Random Forest'}
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,(family,name) in zip(axes,names.items()):
        rows=summary[summary.classifier.eq(family)&summary.variant.eq('baseline')].sort_values('fraction')
        ax.plot(rows.train_rows,100*rows.train_f1,'o-',label='Training',color='#2979b8')
        ax.errorbar(rows.train_rows,100*rows.macro_f1,yerr=100*rows.f1_std,fmt='o-',label='Validation',color='#d49b00',capsize=3)
        ax.set_title(name,fontsize=11);ax.set_xlabel('Mean training rows per fold');ax.grid(alpha=.2);ax.set_ylim(65,101)
    axes[0].set_ylabel('Macro F1 (%)');axes[-1].legend(fontsize=9)
    fig.tight_layout();plot=figures/'learning_curves.png';fig.savefig(plot,dpi=200);plt.close(fig)
    fonts=Path(reportlab.__file__).parent/'fonts'
    pdfmetrics.registerFont(TTFont('Vera',str(fonts/'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('VeraBold',str(fonts/'VeraBd.ttf')))
    styles=getSampleStyleSheet()
    for style in styles.byName.values():
        style.textColor=colors.black
        style.fontName='VeraBold' if style.name in ('Title','Heading1','Heading2') else 'Vera'
    styles['BodyText'].fontSize=10;styles['BodyText'].leading=14
    story=[]
    def p(text,style='BodyText'):story.extend([Paragraph(text,styles[style]),Spacer(1,9)])
    def page(title):
        if story:story.append(PageBreak())
        p(title,'Title')
    def table(rows,widths=None):
        t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Vera'),('FONTNAME',(0,0),(-1,0),'VeraBold'),('FONTSIZE',(0,0),(-1,-1),9),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f4d35e')),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.6,colors.black)]))
        story.extend([t,Spacer(1,12)])
    def pct(value):return f'{100*value:.2f}'
    page('SapBERT: Data Quality and Learning Curves')
    p('Four-level development audit | 0 Emergency, 1 Urgent, 2 Standard, 3 Non-urgent','Heading2')
    p('We kept the same labels, frozen 768-dimensional SapBERT encoder and three classifier families. This audit tested whether useful information in the original complaint was lost in the supplied English concept, and whether adding more training rows is likely to help.')
    table([['Development audit','Rows'],['Development records',str(audit['development_rows'])],['Previous out-of-fold errors',str(audit['oof_errors'])],['Explicit duration disagreements',str(audit['explicit_duration_disagreements'])],['Raw duration not recovered from concept',str(audit['explicit_duration_missing_from_concept'])],['Family word present only in complaint',str(audit['family_word_missing_from_concept'])],['Local review queue (overlapping flags deduplicated)',str(audit['review_rows'])]],[395,100])
    p('For example, a complaint describing half an hour can have a supplied concept describing an hour. This motivates preserving duration information directly from the original complaint. Missing-word flags are screening signals: synonyms, negation and wording differences can create flags without an actual clinical information error.')
    p('The tested improvement adds 19 explicit features: 16 bilingual word/phrase indicators and three duration indicators. These are mentions, not clinical diagnoses. They cover severity words, onset, exertion, radiation, breathing, fainting, sweating, nausea, history, family, hypertension, diabetes, negation and time of day. The original SapBERT representation remains in place.')
    p('No labels were changed. Individual records and the review queue remain local. The model cannot determine whether the supplied four-level labels are clinically correct; that requires independent review.')
    page('Does More Training Data Help?')
    story.append(Image(str(plot),width=495,height=165))
    p('Each point uses the same validation fold. Training subsets contain whole groups and are nested at 25%, 50%, 75% and 100%. Preprocessing is fitted again within each subset. Bars show the standard deviation across five folds, not a confidence interval.')
    rows=[['Classifier','25% F1','50% F1','75% F1','100% F1']]
    for family,name in names.items():
        values=summary[summary.classifier.eq(family)&summary.variant.eq('baseline')].sort_values('fraction')
        rows.append([name]+[pct(v) for v in values.macro_f1])
    table(rows,[175,80,80,80,80])
    lr=summary[summary.classifier.eq('logreg')&summary.variant.eq('baseline')].sort_values('fraction')
    gain=100*(lr.iloc[-1].macro_f1-lr.iloc[-2].macro_f1)
    p(f'The last increase from 75% to 100% training data changes Logistic Regression validation F1 by {gain:+.2f} percentage points. This measures the observed range only. It does not predict a score at 20,000 rows.')
    p('A larger dataset is useful when it adds distinct complaint patterns and consistent, well-defined labels. Duplicating records or repeating the same templates is unlikely to resolve an unclear Urgent/Standard boundary. The next dataset should be independently reviewed and reserved for confirmation before further model selection.')
    p('The learning curves use the prior best setting within each family. They do not retune parameters separately at every size. The original-detail variants were tested at full development size and are compared on the following page.')
    page('Preserving Original Complaint Details')
    rows=[['Classifier / feature variant','Accuracy','F1','Emergency recall']]
    labels={'baseline':'Baseline','concept_details':'English details','complaint_details':'Original details'}
    for family,name in names.items():
        for variant in labels:
            r=summary[summary.classifier.eq(family)&summary.variant.eq(variant)&summary.fraction.eq(1)].iloc[0]
            rows.append([name+' / '+labels[variant],pct(r.accuracy),pct(r.macro_f1),pct(r.emergency_recall)])
    table(rows,[260,70,70,95])
    p('All values are five-fold development means (%). The full audit contains 90 fits: 60 learning-curve fits, including 15 reproduced full-size baselines, and 30 additional detail-feature fits. The six additional feature settings bring the combined full-size search to 55 settings and 275 fold fits.')
    p(f"Selected development condition: {names[selection['classifier']]} with {labels[selection['variant']].lower()}. Macro F1 is {pct(selection['cv_macro_f1'])}%, a gain of {100*selection['cv_gain']:.2f} percentage points over the prior deployed setting. Selection also retains the original emergency-recall constraint.")
    p('The same encoder and classifier families are used throughout. This is feature preservation and classifier retraining, not SapBERT fine-tuning. Candidate comparisons reuse development data; they require independent confirmation.')
    page('Retrospective Result and Next Step')
    table([['Metric (%)','Prior model','Selected model']]+[[label,pct(previous['metrics'][key]),pct(winner['metrics'][key])] for key,label in [('accuracy','Accuracy'),('precision_macro','Macro precision'),('recall_macro','Macro recall'),('macro_f1','Macro F1'),('emergency_recall','Emergency recall'),('under_triage_rate','Under-triage')]], [255,120,120])
    p('These results use the existing 1,999 test records, which have already been examined during earlier iterations. They are retrospective comparisons rather than an untouched test estimate. The accompanying full comparison PDF includes all baseline full-768/PCA-64 results, tuned-family scores, class reports and confusion matrices.')
    p('The deployed preprocessing must retain the original complaint for detail extraction and use the supplied or locally translated English complaint for SapBERT. The adapter checks artifact identities and required inputs. Missing complaints retain the existing no-level placeholder behavior; the displayed 50% is not a model-derived risk estimate.')
    p('Recommended next step','Heading2')
    p('Review ambiguous Urgent/Standard examples and the duration discrepancies using the local review queue. Define a consistent four-level labelling rubric with qualified clinical reviewers. Collect genuinely new examples, preserve original complaint details, then measure the frozen pipeline on that independently reviewed dataset. No target accuracy is guaranteed by increasing row count alone.')
    p('Saved English concepts were used for the embedding experiments. End-to-end live Ollama translation accuracy was not measured in this audit. This research prototype has not been clinically validated.')
    output.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont('Vera',8);canvas.drawString(40,24,'SapBERT | Development audit and retrospective comparison');canvas.drawRightString(A4[0]-40,24,str(doc.page))
    SimpleDocTemplate(str(output),title="SapBERT Data Quality and Learning Curves",pagesize=A4,leftMargin=40,rightMargin=40,topMargin=35,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.source,a.comparison,a.output)
