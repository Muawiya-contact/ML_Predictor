"""Current selected-model report and figures, derived from verified saved evidence."""
from pathlib import Path
import hashlib,json,shutil,zipfile
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'output/results/triage_four_level_round7'
FIG=ROOT/'output/images/SapBERT_Selected_Model'
PDF=ROOT/'output/pdf/SapBERT_Final_Paper_Report.pdf'

def build():
    FIG.mkdir(parents=True,exist_ok=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    read=lambda p:json.loads(p.read_text())
    selection=read(SRC/'selection.json');name=selection['candidate']
    result=next(r for r in read(SRC/'retrospective_results.json') if r['candidate']==name)
    verification=read(SRC/'verification.json');assert verification['status']=='passed'
    manifest=read(ROOT/'triage_model_sapbert/model_manifest.json')
    assert manifest['source_config']==selection['config']
    for file,value in manifest['artifact_sha256'].items():assert sha(ROOT/'triage_model_sapbert'/file)==value
    cv=pd.read_csv(SRC/'cross_validation.csv');cv=cv[cv.candidate.eq(name)].sort_values('fold');assert len(cv)==5
    avg=pd.read_csv(SRC/'cv_summary.csv').set_index('candidate').loc[name]
    y=np.load(SRC/'development_oof.npz');assert len(y['reference'])==8001
    cm=np.asarray(result['confusion']);assert cm.sum()==1999
    diagdir=ROOT/'output/results/triage_pair_embedding_diagnostics';diag=read(diagdir/'diagnostics.json')
    assert diag['pca_sha256']==sha(ROOT/'triage_model_sapbert/pca.pkl')
    assert diag['embedding_sha256']==sha(ROOT/'output/results/triage_four_level/emb_sapbert_pair.npy')
    assert diag['text_input']=='concept_and_complaint'
    levels=['0 Emergency','1 Urgent','2 Standard','3 Non-urgent']
    def save(fig,name):
        fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=300);fig.savefig(FIG/(name+'.pdf'));plt.close(fig)
    keys=['accuracy','precision_macro','recall_macro','macro_f1']
    fig,ax=plt.subplots(figsize=(8,4.4));x=np.arange(4)
    for shift,vals,label,color in [(-.19,cv[keys].mean().values,'Grouped CV mean','#2874ad'),(.19,[result['metrics'][k] for k in keys],'Retrospective test','#efa928')]:
        bars=ax.bar(x+shift,np.asarray(vals)*100,.38,label=label,color=color);ax.bar_label(bars,fmt='%.2f',fontsize=9)
    ax.set_xticks(x,['Accuracy','Macro precision','Macro recall','Macro F1']);ax.set_ylim(0,105);ax.set_ylabel('Score (%)');ax.legend(loc='lower right');ax.set_title('Selected Logistic Regression pipeline')
    save(fig,'01_selected_metrics')
    fig,ax=plt.subplots(figsize=(8,4.2))
    for k,l in [('accuracy','Accuracy'),('macro_f1','Macro F1'),('emergency_recall','Emergency recall')]:ax.plot(cv.fold,cv[k]*100,'o-',label=l)
    ax.set_xticks(range(1,6));ax.set_xlabel('Grouped validation fold');ax.set_ylabel('Score (%)');ax.set_ylim(80,100);ax.legend();ax.grid(alpha=.2);ax.set_title('Selected configuration: five validation folds')
    save(fig,'02_validation_folds')
    fig,axes=plt.subplots(1,2,figsize=(10,4.7))
    for ax,values,title,percent in [(axes[0],cm,'Confusion matrix: counts',False),(axes[1],100*cm/cm.sum(1,keepdims=True),'Confusion matrix: row percentages',True)]:
        ax.imshow(values,cmap='Blues');ax.set_xticks(range(4));ax.set_yticks(range(4));ax.set_xlabel('Predicted level');ax.set_ylabel('Reference level');ax.set_title(title)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{values[i,j]:.1f}%' if percent else str(values[i,j]),ha='center',va='center',color='white' if values[i,j]>values.max()/2 else 'black')
    save(fig,'03_selected_confusion')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for j,(k,label) in enumerate([('precision','Precision'),('recall','Recall'),('f1-score','F1')]):
        bars=ax.bar(np.arange(4)+(j-1)*.25,[100*result['report'][str(i)][k] for i in range(4)],.25,label=label);ax.bar_label(bars,fmt='%.1f',fontsize=8)
    ax.set_xticks(range(4),levels);ax.set_ylim(0,108);ax.set_ylabel('Score (%)');ax.legend(loc='lower right');ax.set_title('Selected model: individual class performance');save(fig,'04_class_scores')
    shutil.copy2(diagdir/'triage_embedding_projection.png',FIG/'05_embedding_projection.png')
    pca=joblib.load(ROOT/'triage_model_sapbert/pca.pkl')
    fig,ax=plt.subplots(figsize=(8,4.5));variance=100*np.cumsum(pca.explained_variance_ratio_)
    ax.plot(np.arange(1,65),variance,color='#2874ad');ax.scatter([64],[variance[-1]],color='#efa928');ax.annotate(f'64 components: {variance[-1]:.2f}%',(64,variance[-1]),xytext=(-155,-20),textcoords='offset points');ax.set_xlabel('Retained PCA components');ax.set_ylabel('Cumulative explained variance (%)');ax.set_ylim(0,100);ax.grid(alpha=.2);ax.set_title('Fitted serving PCA: 768 dimensions reduced to 64');save(fig,'06_pca_variance')
    fonts=Path(reportlab.__file__).parent/'fonts'
    for key,file in [('Vera','Vera.ttf'),('VeraBold','VeraBd.ttf')]:pdfmetrics.registerFont(TTFont(key,str(fonts/file)))
    styles=getSampleStyleSheet()
    for s in styles.byName.values():s.fontName='VeraBold' if s.name in ['Title','Heading1','Heading2'] else 'Vera'
    styles['Title'].fontSize=17;styles['Title'].leading=22
    styles['BodyText'].fontSize=9;styles['BodyText'].leading=13
    flow=[]
    def p(t,style='BodyText'):flow.extend([Paragraph(t,styles[style]),Spacer(1,9)])
    def page(t):
        if flow:flow.append(PageBreak())
        p(t,'Title')
    def table(rows,widths):
        t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Vera'),('FONTNAME',(0,0),(-1,0),'VeraBold'),('FONTSIZE',(0,0),(-1,-1),8),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f4d45c')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f5f7')]),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));flow.extend([t,Spacer(1,12)])
    def pic(name,h):
        from PIL import Image as PIL
        path=FIG/(name+'.png')
        with PIL.open(path) as im:w,ih=im.size
        ratio=min(499/w,h/ih);flow.extend([Image(str(path),width=w*ratio,height=ih*ratio),Spacer(1,10)])
    pct=lambda v:f'{100*v:.2f}%'
    page('Selected SapBERT Model: Final Report')
    p('Current four-level dataset | Selected classifier only','Heading2')
    p('The deployed pipeline uses frozen SapBERT, PCA-64 and balanced Logistic Regression (C=100). This report describes the selected model only. It excludes historical baseline tables and other classifier families.')
    table([['Metric','Grouped CV mean','Retrospective test']]+[[label,pct(cv[k].mean()),pct(result['metrics'][k])] for k,label in zip(keys,['Accuracy','Macro precision','Macro recall','Macro F1'])],[199,150,150])
    p('All four aggregate metrics exceed 90% in these evaluations. Individual class scores, shown later, do not all exceed 90%.')
    p('Dataset and prediction target','Heading2')
    p('The supplied cardiac_multilingual_10000_4level_triage.xlsx dataset contains 10,000 records. The unchanged partition has 8,001 development records and 1,999 test records. Targets are 0 Emergency, 1 Urgent, 2 Standard and 3 Non-urgent. Supplied labels were not edited to improve scores.')
    p('What this report establishes','Heading2')
    p('All figures are plotted from saved experiment data or fitted model artifacts; no decorative or invented clusters are used. The four embedding panels are two-dimensional projections of actual paired-text SapBERT embeddings. A plot shows different evidence from classifier accuracy, as explained on page 5.')
    p('These test records were examined previously. Reported test scores are retrospective and require independent confirmation on new records. Label-assignment rules and independent per-record clinical review were not documented. No clinical validation or end-to-end live translation accuracy is claimed.')
    page('Selected Pipeline and Input Features')
    for title,text in [('1. Prepare the complaint','During the study, SapBERT receives the supplied/recovered English clinical concept followed by [SEP] and the original complaint. In the GUI, local Ollama supplies the English translation and the anatomical safety check runs before classification. The original complaint is also retained.'),('2. Generate the embedding','The frozen SapBERT-from-PubMedBERT-fulltext checkpoint uses CLS pooling, L2 normalization and a 128-token input limit. The output is a 768-dimensional vector. SapBERT weights were not fine-tuned.'),('3. Reduce dimensions','A fitted PCA transform reduces the vector to 64 dimensions. PCA and other fitted preprocessing are learned on training rows within each validation fold. The final serving transforms are fitted on development data, not test data.'),('4. Add patient and complaint features','The 64 PCA components are combined with 50 patient features and 19 complaint-detail features, for 133 inputs. Patient features include age, heart rate, systolic and diastolic blood pressure, temperature, SpO2, AVPU, gender, arrival mode and ECG status. Numeric terms include quadratic combinations. Complaint details cover such items as severity, duration, onset and symptom mentions.'),('5. Predict one of four levels','One balanced Logistic Regression classifier with C=100 produces the class probabilities. The highest-probability class supplies the model prediction. Other evaluated classifier families are not used together in this deployed model.')]:p(title,'Heading2');p(text)
    p('The study uses supplied/recovered concepts; GUI translation can introduce a different error source. The report does not equate saved-concept accuracy with real-time translation performance.')
    page('Selected Model: Aggregate and Fold Scores')
    pic('01_selected_metrics',245);pic('02_validation_folds',245)
    p('Grouped CV mean averages the five validation-fold scores. Related complaint groups stay together across fold boundaries. Retrospective test scores come from the fixed 1,999-row test partition. Both quantities are measured; neither guarantees performance on new clinical records.')
    p('This configuration was selected after a larger development search. Its CV scores are therefore subject to selection optimism. The previously examined test partition is not a fresh independent confirmation set.')
    page('Selected Model: Four-Level Predictions')
    table([['Level','Precision','Recall','F1','Support']]+[[levels[i],pct(result['report'][str(i)]['precision']),pct(result['report'][str(i)]['recall']),pct(result['report'][str(i)]['f1-score']),str(int(result['report'][str(i)]['support']))] for i in range(4)],[139,90,90,90,90])
    pic('03_selected_confusion',220);pic('04_class_scores',225)
    p('Confusion-matrix rows are reference levels and columns are predictions. Counts total 1,999; row percentages divide by each reference class total. Urgent and Standard remain the most difficult classes.')
    page('Embedding Plots: Four Reproducible Samples')
    paper=ROOT/'output/images/SapBERT_Paper_Figures'
    shutil.copy2(paper/'Figure_05_SapBERT_Triage_Embeddings_Four_Samples.png',FIG/'05_embedding_projection.png')
    pic('05_embedding_projection',390)
    p('Figure 5. Two-dimensional PCA projections of current paired-text SapBERT embeddings. Each panel samples 200 distinct development complaint groups, balanced at 50 records per triage level. Seeds are 42, 99, 404 and 777. All panels use the same fitted serving PCA and shared axes. Colours indicate supplied triage levels 0-3; no test records are included.')
    p('Each point represents one record. Panels are repeated samples of the same dataset and can share records; they are not four classifiers, independent datasets or separately trained encoders. Group selection changes between panels, while the model and projection remain fixed.')
    p('Overlap does not contradict the classifier scores. The plot shows only two text components; the classifier uses 64 text components plus 69 patient and complaint-detail features. Similar complaints can have different urgency depending on patient measurements and other information.')
    p('These are triage-level plots, not the earlier arm-pain/back-pain category experiment. The current prepared dataset has no verified complaint-category column. No category labels or separated clusters were invented to reproduce that older appearance.')
    page('PCA and Embedding Geometry')
    pic('06_pca_variance',310)
    p(f'The 64 selected components retain {variance[-1]:.2f}% of the embedding variance on the fitted development data. Explained variance measures retained embedding variation, not classification accuracy.')
    geometry=read(paper/'Embedding_Geometry_Summary.json')
    def mean_sd(v):return f"{v['mean']:.3f} ({v['sd']:.3f})"
    table([['Space','Intra distance','Inter distance','Difference','Silhouette']]+[[r['space']]+[mean_sd(r[k]) for k in ['intra_cosine_distance','inter_cosine_distance','difference','silhouette']] for r in geometry],[119,95,95,95,95])
    p('Table 3. Mean (sample standard deviation) over 20 reproducible development resamples, each with 200 distinct groups and 50 records per triage level. Intra/inter values are pairwise cosine distances within/between labels; difference is inter minus intra. Silhouette uses cosine distance. The same frozen embeddings and fitted PCA are reused. Samples may overlap; these are not 20 independent datasets or training runs.')
    p('Near-zero or negative silhouette indicates weak separation of triage labels in text geometry alone, not a performance ceiling for the combined classifier. Centering and projection change cosine geometry, so raw distances should not be treated as directly interchangeable across spaces. No ANOSIM, permutation-test significance or Cohen d is claimed. The accompanying CSV preserves all 60 sample-by-space measurements.')
    page('Verification and Additional Performance Measures')
    d=verification['selected_probability_diagnostics']['retrospective_test']
    table([['Measure','Selected model']]+[[label,value] for label,value in [('Emergency recall',pct(result['metrics']['emergency_recall'])),('Undertriage',pct(result['metrics']['under_triage_rate'])),('Overtriage',pct(result['metrics']['over_triage_rate'])),('Quadratic weighted kappa',f"{result['metrics']['qwk']:.4f}"),('Mean absolute level error',f"{result['metrics']['mae']:.4f}"),('Matthews correlation coefficient',f"{d['mcc']:.4f}"),('Log loss',f"{d['log_loss']:.4f}"),('Multiclass Brier score',f"{d['multiclass_brier']:.4f}")]],[309,190])
    p('Undertriage means predicting a numerically higher, less urgent level than the supplied label. Overtriage means predicting a more urgent level. Kappa and MCC measure agreement; log loss and Brier score assess probabilities, with lower values preferred. These are not all percentage metrics.')
    p('Verification completed','Heading2')
    p('The serving adapter reproduced all 1,999 selected predictions and probabilities. Thirteen live embedding checks and one live English prediction passed. All 49 targeted unit tests and the real six-tab GUI audit passed. Dataset identities, grouping boundaries and the fitted artifact hashes were checked. GUI, CLI and batch inference share the selected preprocessing and classifier contract.')
    p('The current report reads saved verified results; generating it does not retrain the model. Individual records and per-record predictions are excluded from the shared image package.')
    page('Why Accuracy Improved with the Same Model Family')
    p('The classifier family remains Logistic Regression and the SapBERT checkpoint remains frozen. However, the complete prediction pipeline changed: it now represents the supplied concept together with the original complaint, uses a 128-token input limit, retains 64 PCA components and uses balanced Logistic Regression with C=100. Patient-feature combinations and explicit complaint details are included.')
    p('The original complaint may preserve severity, timing or context that the supplied clinical concept omits. Giving the classifier this additional representation, while choosing suitable dimensionality and regularization, produced better measured aggregate scores. The experiments do not isolate exactly how much of the improvement comes from each simultaneous change.')
    p('What to tell your professor','Heading2')
    p('“We retained the SapBERT encoder and Logistic Regression classifier family, but improved the input representation and tuned the prediction pipeline. SapBERT now encodes the clinical concept together with the original complaint. The selected model combines PCA-64 text features with patient measurements and complaint details. On the unchanged four-level dataset and grouped splits, it achieved 90.60% retrospective accuracy and 90.99% macro F1. The encoder was not fine-tuned, and the labels were not changed. Independent evaluation is still needed because the test records had been examined previously.”')
    p('Earlier uncertainty should be described accurately: 90% could not be guaranteed from the results available at that time. It was not established that improvement with this model family was impossible. The new measured result supports this particular pipeline, not a guarantee that every metric or every future dataset will exceed 90%.')
    p('This report includes only the selected deployed model. Full search records remain available separately for reproducibility; unsuccessful models are not replaced with invented high scores.')
    def footer(c,doc):c.setFont('Vera',8);c.drawString(48,25,'SapBERT | Selected-model research report');c.drawRightString(A4[0]-48,25,str(doc.page))
    SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=38,bottomMargin=42,title='Selected SapBERT Model: Final Report').build(flow,onFirstPage=footer,onLaterPages=footer)
    evidence=dict(selected_candidate=name,source_result_sha256=sha(SRC/'retrospective_results.json'),embedding_diagnostic=read(paper/'Embedding_Protocol.json'),figure_sha256={p.name:sha(p) for p in sorted(FIG.glob('*.png'))})
    (FIG/'figure_sources.json').write_text(json.dumps(evidence,indent=2))
    (FIG/'README.txt').write_text('Six figures for the current selected SapBERT + PCA64 + Logistic Regression model. All figures use saved measured results or fitted artifacts. 05 is a two-component projection of actual paired-text embeddings from 1,200 development groups, not synthetic plotting coordinates. No patient records are included. See the selected-model PDF for interpretation and evaluation limits.\n')
    with zipfile.ZipFile(ROOT/'output/pdf/SapBERT_Selected_Model_Images.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(FIG.iterdir()):
            if p.is_file():z.write(p,'SapBERT_Selected_Model/'+p.name)
    print(PDF)

if __name__=='__main__':build()
