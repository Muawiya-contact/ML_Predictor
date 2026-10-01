"""Eight-page second-round report with all original baseline comparisons."""
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

def build(source, original, output):
    selection=json.loads((source/'selection.json').read_text());results=json.loads((source/'retrospective_results.json').read_text())
    winner=next(r for r in results if r['candidate']==selection['candidate']);baseline=results[0]
    cv=pd.read_csv(source/'cv_summary.csv');old=json.loads((original/'results.json').read_text())
    figures=source/'figures';figures.mkdir(exist_ok=True)
    styles=getSampleStyleSheet()
    for style in styles.byName.values():style.textColor=colors.black
    styles['Title'].fontSize=17;styles['Title'].leading=21
    styles['BodyText'].fontSize=9;styles['BodyText'].leading=12
    story=[]
    def p(text,style='BodyText'):story.extend([Paragraph(text,styles[style]),Spacer(1,7)])
    def page(title):
        if story:story.append(PageBreak())
        p(title,'Title')
    def table(rows,widths=None,size=8,padding=5):
        t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),size),
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f4d35e')),('LINEBELOW',(0,0),(-1,0),.6,colors.black),
            ('BOTTOMPADDING',(0,0),(-1,-1),padding),('TOPPADDING',(0,0),(-1,-1),padding),('VALIGN',(0,0),(-1,-1),'TOP')]))
        story.extend([t,Spacer(1,10)])
    def pic(path,width=495,height=290):story.append(Image(str(path),width=width,height=height,kind='proportional'))
    def pct(x):return f'{100*x:.2f}'
    def matrix(ax,cm,title):
        ax.imshow(cm,cmap='Blues');ax.set_title(title,fontsize=10);ax.set_xticks(range(4));ax.set_yticks(range(4));ax.set_xlabel('Predicted level');ax.set_ylabel('Reference level')
        for i in range(4):
            for j in range(4):ax.text(j,i,str(cm[i][j]),ha='center',va='center',fontsize=9,color='white' if cm[i][j]>np.max(cm)/2 else 'black')
    page('Four-Level SapBERT: Improvement Study')
    p('Revised comparison | Levels 0 Emergency, 1 Urgent, 2 Standard, 3 Non-urgent','Heading2')
    p('This second round uses the same 10,000 supplied records, frozen SapBERT embeddings and original grouped partitions. It expands the search to PCA-64, PCA-128 and PCA-256, stronger/weaker regularization, class balancing, PCA whitening and structured-only controls. No labels are changed.')
    p('Selected configuration: '+selection['candidate']+'.','Heading2')
    p(f"Selected total input size: {winner.get('feature_count', 'see manifest')} features. PCA retained variance: {pct(winner['pca_retained_variance']) + '%' if winner.get('pca_retained_variance') is not None else 'not applicable'}.")
    table([['Retrospective metric (%)','Previous model','Selected model']]+[[label,pct(baseline['metrics'][key]),pct(winner['metrics'][key])] for key,label in [('accuracy','Accuracy'),('precision_macro','Macro precision'),('recall_macro','Macro recall'),('macro_f1','Macro F1'),('emergency_recall','Emergency recall'),('under_triage_rate','Under-triage')]], [245,120,130])
    p(f"Mean development CV macro F1 change: {selection['cv_gain']*100:+.2f} percentage points. Selection uses all five development folds and an emergency-recall constraint; the test results above do not select the winner.")
    p('<b>Evaluation scope:</b> These 1,999 test rows were already examined in the previous study. The new scores are retrospective comparisons, not a fresh independent estimate. New labelled records are needed to confirm generalization. Saved concepts are evaluated here; live translation accuracy is not measured.')
    p('The previous model and its report remain available for traceability. SapBERT is not fine-tuned in this round. Higher scores alone do not establish clinical validity.')
    page('Complete Development Cross-Validation')
    p('All 35 configurations; 175 fits across five grouped folds. F1, accuracy and emergency recall are percentages. SD is the fold-to-fold F1 standard deviation in percentage points. The incumbent is lr_pca64_c10_balanced1.')
    table([['Configuration','F1','SD','Accuracy','Emergency recall']]+[[r.candidate,pct(r.macro_f1),pct(r.f1_std),pct(r.accuracy),pct(r.emergency_recall)] for r in cv.itertuples()], [215,60,50,70,100],7.1,3)
    p('Selection: highest mean macro F1 among candidates whose mean emergency recall is within one percentage point of the incumbent; accuracy resolves ties. This constraint is an experimental trade-off, not a clinical safety guarantee.')
    page('Original 768-D versus PCA-64 Baselines')
    p('All original fixed comparisons are retained below. They use the same 1,999 previously examined test rows. Values are percentages; precision, recall and F1 are macro-averaged. These baseline runs are from the first round and are not new experiments.')
    for view,label in [('text','Text only'),('fused','Text plus patient features')]:
        p(label,'Heading2')
        names={'logreg':'Logistic Regression','hgb':'HistGradientBoosting','rf':'Random Forest'}
        rows=[['Text size','Classifier','Accuracy','Precision','Recall','F1']]
        for r in old:
            if r['config']['features']['view']==view:rows.append([str(r['config']['features']['pca'] or 768),names[r['config']['classifier']]]+[pct(r['metrics'][k]) for k in ['accuracy','precision_macro','recall_macro','macro_f1']])
        table(rows,[55,155,72,72,70,70],7.5)
    p('The original fused PCA-64 input has 86 features: 64 text components and 22 encoded patient features. The full fused input has 790 features. All training transformations are fitted without validation/test rows.')
    page('Performance Graphs')
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for col,view in enumerate(['text','fused']):
        for row,metric in enumerate(['accuracy','precision_macro']):
            ax=axes[row,col]
            for j,pc in enumerate([0,64]):
                rows=[r for r in old if r['config']['features']['view']==view and r['config']['features']['pca']==pc]
                ax.bar(np.arange(3)+(j-.5)*.35,[r['metrics'][metric]*100 for r in rows],width=.35,label='768-D' if pc==0 else 'PCA-64',color=['#2874ad','#efa928'][j])
            ax.set_xticks(range(3),['LR','HGB','RF']);ax.set_ylim(0,100)
            ax.set_ylabel(('Accuracy' if row==0 else 'Macro precision')+' (%)')
            ax.set_title('Text only' if view=='text' else 'Text plus patient features');ax.legend()
    fig.tight_layout();path=figures/'accuracy_precision_comparison.png';fig.savefig(path,dpi=180);plt.close(fig);pic(path,height=470)
    p('Blue and yellow bars preserve the original comparison style. The selected second-round result is shown separately on page 1 because it follows a larger development search.')
    for classifiers,title,filename in [(['logreg','hgb'],'Baseline Confusion Matrices: LR and HGB','lr_hgb_matrices'),(['rf'],'Baseline and Selected Confusion Matrices','rf_selected_matrices')]:
        page(title)
        rows=[r for r in old if r['config']['classifier'] in classifiers]
        if classifiers==['rf']:rows.append(dict(id='Selected second-round model',confusion=winner['confusion']))
        nrows=(len(rows)+1)//2;fig,axes=plt.subplots(nrows,2,figsize=(9,3*nrows),squeeze=False)
        for ax,r in zip(axes.flat,rows):matrix(ax,r['confusion'],r['id'])
        for ax in list(axes.flat)[len(rows):]:ax.axis('off')
        fig.tight_layout();path=figures/(filename+'.png');fig.savefig(path,dpi=190);plt.close(fig);pic(path,height=600)
        p('Rows are reference labels; columns are predictions. Level order: 0 Emergency, 1 Urgent, 2 Standard, 3 Non-urgent. Baseline matrices are unchanged from round one.', 'BodyText')
    page('Class Results and Error Review')
    table([['Level','Precision (%)','Recall (%)','F1 (%)','Support']]+[[f'{i} '+name]+[pct(winner['report'][str(i)][k]) for k in ['precision','recall','f1-score']]+[str(int(winner['report'][str(i)]['support']))] for i,name in enumerate(['Emergency','Urgent','Standard','Non-urgent'])],[140,95,90,90,80])
    oof=np.load(source/'development_oof.npz');cm=np.zeros((4,4),dtype=int)
    for y,pred in zip(oof['reference'],oof['selected'].argmax(1)):cm[y,pred]+=1
    fig,ax=plt.subplots(figsize=(6,4.5));matrix(ax,cm,'Selected model: development out-of-fold errors');fig.tight_layout();path=figures/'development_confusion.png';fig.savefig(path,dpi=180);plt.close(fig);p('Development out-of-fold confusion matrix', 'Heading2');table([['Reference / predicted','0','1','2','3']]+[[str(i)]+[str(v) for v in row] for i,row in enumerate(cm)], [175,80,80,80,80],8)
    p('An input audit found no identical development input combinations with conflicting targets. This does not establish that labels are clinically correct. A local review CSV lists development errors and model confidence; it is not committed because it contains individual records.')
    p('Use the review list to investigate unclear label boundaries, missing symptom severity/duration/history and data entry issues. Do not replace reference labels with model predictions merely to improve agreement.')
    p('Literature context (different task)', 'Heading2')
    literature=json.loads((original/'literature_sources.json').read_text())
    table([['Model','Accuracy (%)','Recall (%)','F1 (%)']]+[[name,pct(m['accuracy']),pct(m['recall']),pct(m['f1'])] for name,m in literature['models'].items()], [195,100,100,100],7.5)
    p('Seo et al., Scientific Reports (2025), Table 2. DOI: 10.1038/s41598-025-99874-0. Different binary KTAS task; publication metric definitions differ. Context only, not evidence of superiority over these models.')
    page('Methods, Reproducibility and Limits')
    p('Data and preprocessing','Heading2')
    p('The original 8,001 development / 1,999 test partition and five grouped folds are reused. Groups do not overlap across training and validation/test boundaries. Input and embedding hashes are checked before and after the run. Median imputation, categorical encoding, scaling and PCA are learned inside each development fold. No label-derived columns are model inputs.')
    p('Encoder and search','Heading2')
    p('SapBERT-from-PubMedBERT-fulltext uses frozen 768-D CLS embeddings, L2 normalization, a 64-token limit and the original pinned revision. Logistic Regression varies C=10, 100 and 1000 with and without balancing, plus whitened PCA with C=0.1. Six additional LR configurations use quadratic numeric terms, with C=1/10/100 and optional balancing. HGB tests 7/15 leaves with regularization; Random Forest tests structured and fused inputs. Every candidate is evaluated on all five folds.')
    p('Selection and deployment','Heading2')
    p('The incumbent is included in the same search. Selection is recorded before the retrospective test evaluation. The exported model must reproduce saved predictions through the shared GUI/CLI preprocessing. Model artefacts and the incumbent are retained separately. Quadratic terms, if selected, are applied by the fitted numeric preprocessing pipeline. No label correction or encoder fine-tuning is claimed.')
    p('Research interpretation','Heading2')
    p('The larger search can overfit development cross-validation. The old test set is already exposed, so its scores must not be presented as untouched confirmation. A new independently labelled dataset is needed for confirmation. Supplied/recovered concepts bypass live translation during this evaluation. The provider could not supply label-assignment rules; independent per-record review is not documented. These are research comparisons, not clinical validation.')
    p('Files','Heading2')
    p('protocol.json freezes candidates and source identities; cross_validation.csv contains every fit; cv_summary.csv reports every candidate; selection.json records the decision; retrospective_results.json contains class scores and matrices. Source data, embeddings and per-record error files remain local.')
    output.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):canvas.setFont('Helvetica',8);canvas.drawString(40,25,'Four-level SapBERT | Second-round research comparison');canvas.drawRightString(A4[0]-40,25,str(doc.page))
    SimpleDocTemplate(str(output),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=35,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.source,a.original,a.output)
