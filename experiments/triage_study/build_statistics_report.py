"""Refresh the single current PDF with measured semantic and model comparisons."""
from pathlib import Path
import json,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image
from pypdf import PdfReader,PdfWriter
ROOT=Path(__file__).resolve().parents[2]
FIG=ROOT/'output/images/SapBERT_Method_Update'
OUT=ROOT/'output/pdf/SapBERT_Method_Improvement_and_Semantic_Concepts.pdf'
def build():
 FIG.mkdir(parents=True,exist_ok=True);(ROOT/'tmp/pdfs').mkdir(parents=True,exist_ok=True)
 stat=ROOT/'output/results/semantic_statistics'
 summary=json.loads((stat/'summary.json').read_text());runs=pd.read_csv(stat/'per_run_metrics.csv')
 assert len(runs)==80 and runs.groupby('space').size().eq(20).all()
 assert runs.silhouette.between(-1,1).all()
 r7=ROOT/'output/results/triage_four_level_round7'
 fam=json.loads((r7/'family_results.json').read_text())
 wanted=['lr_pair64_c100','hgb__complaint_details','rf_detail64_fraction0.3']
 selected=[next(r for r in fam if r['candidate']==n) for n in wanted]
 names=['Logistic Regression','HistGradientBoosting','Random Forest']
 current=selected[0]['metrics'];assert round(current['accuracy']*100,2)==90.60
 def save(fig,name):
  fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=300)
  plt.close(fig)
 labels=[r['space'] for r in summary]
 fig,ax=plt.subplots(figsize=(8,3.7));ax.bar(labels,[r['silhouette'] for r in summary],yerr=[r['silhouette_sd'] for r in summary],capsize=4,color=['#327daf','#e7b831','#4c9c73','#999999']);ax.axhline(0,color='black',lw=.8);ax.set_ylim(-1,1);ax.set_ylabel('Mean silhouette (cosine distance)');ax.set_title('Semantic mentions: 20 fixed development resamples')
 save(fig,'Semantic_Silhouette_20_Resamples')
 fig,ax=plt.subplots(figsize=(8,3.7));x=np.arange(4)
 for shift,k,label,col in [(-.18,'intra','Within mention group','#337daf'),(.18,'inter','Between mention groups','#e7b831')]:ax.bar(x+shift,[r[k] for r in summary],.36,yerr=[r[k+'_sd'] for r in summary],capsize=3,label=label,color=col)
 ax.set_xticks(x,labels);ax.set_ylabel('Mean cosine distance');ax.set_ylim(0,1.5);ax.legend();ax.set_title('Distances: lower within-group than between-group means');save(fig,'Semantic_Intra_Inter_20_Resamples')
 fig,ax=plt.subplots(figsize=(8,3.6));x=np.arange(3)
 for j,(k,label) in enumerate([('accuracy','Accuracy'),('precision_macro','Macro precision'),('recall_macro','Macro recall'),('macro_f1','Macro F1')]):
  bars=ax.bar(x+(j-1.5)*.2,[r['metrics'][k]*100 for r in selected],.2,label=label);ax.bar_label(bars,fmt='%.1f',fontsize=7)
 ax.set_xticks(x,names);ax.set_ylim(0,110);ax.set_ylabel('Score (%)');ax.legend(loc='lower right',fontsize=8);ax.set_title('Current tuned family representatives (retrospective)');save(fig,'Current_Three_Classifier_Metrics')
 fig,axes=plt.subplots(1,3,figsize=(10,3.4))
 for ax,r,name in zip(axes,selected,names):
  cm=np.asarray(r['confusion']);assert cm.sum()==1999
  ax.imshow(cm,cmap='Blues');ax.set_title(name,fontsize=10);ax.set_xticks(range(4));ax.set_yticks(range(4));ax.set_xlabel('Predicted level');ax.set_ylabel('Reference level')
  for i in range(4):
   for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=9,color='white' if cm[i,j]>cm.max()/2 else 'black')
 save(fig,'Current_Three_Classifier_Confusion_Matrices')
 fonts=Path(reportlab.__file__).parent/'fonts'
 for n,f in [('Vera','Vera.ttf'),('VeraBold','VeraBd.ttf')]:pdfmetrics.registerFont(TTFont(n,str(fonts/f)))
 styles=getSampleStyleSheet()
 for s in styles.byName.values():s.fontName='VeraBold' if s.name in ['Title','Heading1','Heading2'] else 'Vera'
 styles['Title'].fontSize=17;styles['Title'].leading=22;styles['BodyText'].fontSize=9;styles['BodyText'].leading=13
 styles['Heading2'].fontSize=11;styles['Heading2'].leading=15
 styles.add(ParagraphStyle('Small',parent=styles['BodyText'],fontName='Vera',fontSize=7.5,leading=10))
 flow=[]
 def p(t,style='BodyText'):flow.extend([Paragraph(t,styles[style]),Spacer(1,8)])
 def page(t):
  if flow:flow.append(PageBreak())
  p(t,'Title')
 def table(rows,widths):
  t=Table([[Paragraph(str(v),styles['Small']) for v in row] for row in rows],colWidths=widths,repeatRows=1)
  t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.8,colors.black),('LINEBELOW',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,-1),(-1,-1),.8,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));flow.extend([t,Spacer(1,10)])
 def pic(name,h):
  from PIL import Image as PIL
  path=FIG/(name+'.png')
  with PIL.open(path) as im:w,ih=im.size
  ratio=min(499/w,h/ih);flow.extend([Image(str(path),width=w*ratio,height=ih*ratio),Spacer(1,8)])
 page('Semantic Separation: Actual Silhouette Values')
 p('Silhouette is bounded by -1 and +1. A value above +1 is invalid. Positive values indicate closer average similarity to the assigned group than to the nearest alternative group; values near zero indicate overlap. A negative value can occur when text-based distances do not agree with the chosen labels. It is not a classification accuracy percentage.')
 p('<b>Table 4. Semantic-mention geometry.</b> Twenty fixed samples of 50 development complaints (10 per mention group). Intra/inter values are means +/- sample SD across resamples; remaining columns are means. Colours/groups refer to arm, back, jaw, palpitations and shoulder mentions, not urgency.')
 rows=[['Space','Intra','Inter','Diff.','d','R','Silhouette','Runs p&lt;.05*']]
 for r in summary:rows.append([r['space'],f"{r['intra']:.3f} +/- {r['intra_sd']:.3f}",f"{r['inter']:.3f} +/- {r['inter_sd']:.3f}",f"{r['difference']:.3f}",f"{r['cohen_d']:.2f}",f"{r['anosim_R']:.3f}",f"{r['silhouette']:.3f}",f"{r['holm_significant']}/20"])
 table(rows,[77,82,82,42,35,40,55,86])
 p('*ANOSIM label-permutation p-values, Holm-adjusted across all 80 run/space tests. All values are recomputed from current paired-text embeddings. d is a descriptive standardized pair-distance difference and can exceed 1; silhouette cannot. R is ANOSIM rank separation. These are different statistics.','Small')
 pic('Semantic_Silhouette_20_Resamples',250)
 p('The four 100-record samples on page 9 previously gave mean silhouettes of 0.246 (768-D), 0.291 (64-D) and 0.289 (2-D diagnostic). This table uses 20 different 50-record samples under the repeated-sampling protocol; sampling explains the small numerical changes. No negative score was replaced with a positive one.')
 p('The 2-D serving view remains negative. The separate diagnostic projection, fitted without labels to the eligible development pool, answers a different display question. The deployed model and accuracy remain unchanged. Original triage-label geometry on page 6 remains an auxiliary result.')
 page('Within- and Between-Group Comparison')
 p('<b>Table 5. Repeated-sample comparison.</b> Each resample contributes one within-group mean and one between-group mean. Diff. = inter minus intra. A positive difference indicates larger average distances between groups. Samples overlap, so they are not 20 independent datasets.')
 rows=[['Space','Mean diff.','SD diff.','Positive runs','Raw p range','Holm p range']]
 for r in summary:rows.append([r['space'],f"{r['difference']:.3f}",f"{r['difference_sd']:.3f}",f"{r['positive_difference_runs']}/20",f"{r['min_p']:.4f}-{r['max_p']:.4f}",f"{r['min_holm']:.3f}-{r['max_holm']:.3f}"])
 table(rows,[89,65,65,70,105,105]);pic('Semantic_Intra_Inter_20_Resamples',185)
 p('Choice of statistical tests','Heading2')
 p('Repeated random samples reuse development records, and pairwise distances within each sample share records. Treating these observations as independent would exaggerate statistical precision. The report therefore gives descriptive across-sample differences and per-sample record-label permutation tests, rather than presenting an across-run t-test or Wilcoxon result as independent clinical evidence.')
 p('Statistical protocol','Heading2')
 p('Seeds 20261005-20261024 were fixed before computation; every run is retained. Each sample has 50 distinct complaint groups, drawn without replacement within the sample. ANOSIM uses 4,999 shuffled label assignments plus the observed arrangement, with p=(exceedances+1)/5,000. Holm correction covers 80 tests. Minimum resolvable raw p is 0.0002; no p=0 is reported.')
 p('Cohen d divides the inter-minus-intra distance difference by the pooled sample SD of pair distances. It is descriptive, with no independent-pair confidence interval. Cosine distances can range from 0 to 2; centering and projection alter their scale. Significant average separation can coexist with negative silhouette because silhouette compares each record with its nearest alternative group.')
 p('Mention labels come from the same text that is encoded. These tests describe lexical/semantic association, not an independently labelled semantic benchmark, clinical validity or triage accuracy. All fitted classifier artifacts and supplied labels are unchanged.','Small')
 page('ED Triage Model Comparison')
 p('<b>Table 6. Literature context.</b> Scores are fractions (0.906 is 90.6%). Literature rows describe a different binary KTAS task and tenfold CV; our row is retrospective four-level evaluation. Metric definitions and inputs differ, so these rows do not establish superiority.')
 table([['Model / evaluation','Accuracy','Recall','F1'],['TF-IDF + Logistic Regression [1]','0.750','0.988','0.544'],['TF-IDF + Random Forest [1]','0.751','0.964','0.582'],['BiLSTM [1]','0.746','0.846','0.670'],['Current paired SapBERT + PCA-64 + LR',f"{current['accuracy']:.3f}",f"{current['recall_macro']:.3f}",f"{current['macro_f1']:.3f}"]],[274,75,75,75])
 p('[1] Seo et al. (2025), <i>Artificial intelligence for severity triage based on conversations in an emergency department in Korea</i>, Scientific Reports 15:16870, Table 2. DOI: <link href="https://doi.org/10.1038/s41598-025-99874-0">10.1038/s41598-025-99874-0</link>. Published values are transcribed as reported, not recomputed. The 0.746/0.846/0.670 row is BiLSTM, not a combined BiLSTM-CNN-RNN hybrid.','Small')
 p('The measured current four-level accuracy is 0.905953 (90.60%), rather than 0.990. The literature models above were not trained as part of our current comparison.')
 p('<b>Table 7. Actual current three-family comparison.</b> Same 1,999 test records and four-level targets. Precision, recall and F1 are macro averages. Each family uses its development-selected feature configuration; this is a pipeline comparison, not a controlled classifier-only ablation.')
 table([['Family','Accuracy','Precision','Recall','F1']]+[[name]+[f"{r['metrics'][k]*100:.2f}%" for k in ['accuracy','precision_macro','recall_macro','macro_f1']] for name,r in zip(names,selected)],[199,75,75,75,75])
 pic('Current_Three_Classifier_Metrics',210)
 p('Only the selected Logistic Regression pipeline is deployed. The comparison retains HistGradientBoosting and Random Forest as measured alternatives. It does not imply that the GUI combines their predictions.','Small')
 page('Classifier Matrices and Reproducibility')
 pic('Current_Three_Classifier_Confusion_Matrices',160)
 p('<b>Figure 11.</b> Current tuned family representatives. Rows are reference levels, columns are predicted levels, ordered 0 Emergency, 1 Urgent, 2 Standard and 3 Non-urgent. Each matrix contains 1,999 cases. Large diagonal counts indicate correct classifications; off-diagonal counts show errors.')
 p('How to read the complete report','Heading2')
 p('Pages 1-4 contain the current selected model, aggregate metrics and class results. Pages 5-6 retain urgency-coloured text diagnostics as auxiliary evidence. Pages 7-8 provide model verification and input improvements. Pages 9-12 contain the symptom-based scatter plots and the measured progression from the earlier baseline to 90.60%. Pages 13-16 add semantic statistics, publication-style comparison tables and current three-family results.')
 p('What was recomputed in this update','Heading2')
 p('Twenty fixed semantic resamples were measured in four embedding spaces, yielding 80 rows of geometry and permutation statistics. The model tables and confusion matrices were regenerated from saved verified experiment results. No classifier or encoder was retrained, and no label or plotted coordinate was altered to improve presentation.')
 p('Reproduction and sources','Heading2')
 for t in ['Semantic sampling and metric implementation: experiments/triage_study/semantic_statistics.py. Per-run aggregate results and protocol: output/results/semantic_statistics/.','Current family metrics and matrices: output/results/triage_four_level_round7/family_results.json. Selected configuration: lr_pair64_c100.','Figures and aggregate data are included in the current image ZIP. Private sampled records are excluded. The displayed mean +/- SD describes resample variability, not uncertainty on an independent population estimate.','Silhouette definition: <link href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html">scikit-learn silhouette_score documentation</link>. Literature source: Seo et al. (2025), Table 2, DOI 10.1038/s41598-025-99874-0.','The test partition has been examined previously; these scores remain retrospective. Neither positive semantic separation nor 90.60% retrospective accuracy establishes clinical validation or live translation performance.']:p(t)
 while flow and isinstance(flow[-1],Spacer):flow.pop()
 def footer(c,d):c.setFont('Vera',8);c.drawString(48,25,'SapBERT | Measured comparisons and semantic statistics');c.drawRightString(A4[0]-48,25,str(d.page+12))
 supplement=ROOT/'tmp/pdfs/statistics_appendix.pdf'
 SimpleDocTemplate(str(supplement),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=38,bottomMargin=42).build(flow,onFirstPage=footer,onLaterPages=footer)
 assert len(PdfReader(supplement).pages)==4
 source=ROOT/'reports/triage_method_update/SapBERT_Method_Improvement_and_Semantic_Concepts.pdf'
 writer=PdfWriter();writer.append(source,pages=(0,12));writer.append(supplement);writer.add_metadata({'/Title':'SapBERT: Current Results, Method Improvements and Semantic Statistics'})
 tmp=ROOT/'tmp/pdfs/current_complete.pdf'
 with tmp.open('wb') as f:writer.write(f)
 tmp.replace(OUT)
 for name in ['protocol.json','summary.json','per_run_metrics.csv']:(FIG/('Semantic_20_'+name)).write_bytes((stat/name).read_bytes())
 with zipfile.ZipFile(ROOT/'output/pdf/img.zip','w',zipfile.ZIP_DEFLATED) as z:
  for file in sorted(FIG.iterdir()):
   if file.is_file():z.write(file,file.name)
 print(OUT)
if __name__=='__main__':build()
