"""Append measured semantic figures and method-change evidence to the paper report.

Requires saved local experiment caches. Never trains or selects new models.
All fixed semantic samples are retained; no coordinates are edited for appearance.
"""
from pathlib import Path
import json, hashlib, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pypdf import PdfReader, PdfWriter
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from complaint_semantic_audit import mention, RULES, COLORS, SEEDS

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/images/SapBERT_Method_Update'
FINAL=ROOT/'output/pdf/SapBERT_Method_Improvement_and_Semantic_Concepts.pdf'

def build():
 OUT.mkdir(parents=True,exist_ok=True)
 FINAL.parent.mkdir(parents=True,exist_ok=True)
 (ROOT/'tmp/pdfs').mkdir(parents=True,exist_ok=True)
 load=lambda p:json.loads(p.read_text())
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 base=ROOT/'output/results/triage_four_level'
 audit=ROOT/'output/results/complaint_semantic_audit'
 latest=ROOT/'output/results/triage_four_level_round7'
 protocol=load(audit/'protocol.json')
 manifest=load(ROOT/'triage_model_sapbert/model_manifest.json')
 assert manifest['source_config']==load(latest/'selection.json')['config']
 for filename,digest in manifest['artifact_sha256'].items():
  assert sha(ROOT/'triage_model_sapbert'/filename)==digest
 assert sha(base/'dataset_with_splits.csv')==protocol['dataset_sha256']
 assert sha(base/'emb_sapbert_pair.npy')==protocol['embedding_sha256']
 frame=pd.read_csv(base/'dataset_with_splits.csv');dev=frame[frame.partition.eq('development')].copy()
 dev['mention_group']=dev.Clinical_Concept.map(mention)
 pool=dev[dev.mention_group.notna()].drop_duplicates('group')
 emb=np.load(base/'emb_sapbert_pair.npy',mmap_mode='r')
 projection=PCA(n_components=2,svd_solver='full').fit(np.asarray(emb[pool.row_id],dtype=float))
 samples=load(audit/'private_row_ids.json');assert [s['seed'] for s in samples]==SEEDS
 fig,axes=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True)
 def panel(ax,sample,run):
  subset=frame.set_index('row_id').loc[sample['row_ids']].copy()
  subset['mention_group']=subset.Clinical_Concept.map(mention)
  assert len(subset)==100 and subset.group.is_unique and subset.partition.eq('development').all()
  xy=projection.transform(np.asarray(emb[subset.index],dtype=float))
  for cat,color in zip(RULES,COLORS):
   mask=subset.mention_group.eq(cat).to_numpy();assert mask.sum()==20
   ax.scatter(xy[mask,0],xy[mask,1],color=color,s=32,alpha=.8,label=cat)
   mean=xy[mask].mean(axis=0);ax.scatter(*mean,color=color,s=140,marker='X',edgecolor='black',linewidth=1)
  ax.set_title(f'Run {run} | seed {sample["seed"]} | 100 complaints')
  ax.set_xlabel(f'PC1 ({100*projection.explained_variance_ratio_[0]:.1f}% variance)')
  ax.set_ylabel(f'PC2 ({100*projection.explained_variance_ratio_[1]:.1f}% variance)');ax.grid(alpha=.18)
 for i,(ax,sample) in enumerate(zip(axes.flat,samples),1):panel(ax,sample,i)
 handles,labels=axes[0,0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
 fig.suptitle('SapBERT paired-text embeddings by symptom mention\nFour fixed samples; one label-blind diagnostic PCA',fontsize=15)
 fig.tight_layout(rect=[0,.07,1,.93])
 for ext in ['png','pdf']:fig.savefig(OUT/f'Symptom_Concepts_Four_Runs.{ext}',dpi=300)
 plt.close(fig)
 for i,sample in enumerate(samples,1):
  fig,ax=plt.subplots(figsize=(8,6));panel(ax,sample,i);ax.legend(fontsize=8);fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(OUT/f'Symptom_Concepts_Run_{i}_Seed_{sample["seed"]}.{ext}',dpi=300)
  plt.close(fig)
 original=load(base/'results.json');past={r['id']:r for r in original}
 recent=load(latest/'retrospective_results.json');current=next(r for r in recent if r['candidate']=='lr_pair64_c100')
 intermediate=load(latest/'previous_round_result.json')
 ref=next(r for r in recent if r['candidate']=='lr_pca64_c10_balanced1')
 current_predictions=pd.read_csv(latest/'lr_pair64_c100_retrospective_predictions.csv')
 for filename in ['fused_64_logreg_predictions.csv','fused_768_hgb_predictions.csv']:
  baseline_predictions=pd.read_csv(base/filename)
  assert baseline_predictions[['row_id','reference']].equals(current_predictions[['row_id','reference']])
 records=[('Earlier HGB / full 768',past['fused_768_hgb']),('Earlier LR / PCA-64',past['fused_64_logreg']),('Tuned LR / PCA-64',ref),('LR + quadratic + details / PCA-128',intermediate),('Current paired-text LR / PCA-64',current)]
 keys=['accuracy','precision_macro','recall_macro','macro_f1']
 pd.DataFrame([dict(stage=n,**r['metrics']) for n,r in records]).to_csv(OUT/'Measured_Method_Comparison.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,4));names=['Earlier HGB','Earlier LR','Tuned LR','LR + details','Current LR']
 bars=ax.bar(names,[100*r['metrics']['accuracy'] for _,r in records],color=['#9ba9b4']*4+['#eeb827']);ax.bar_label(bars,fmt='%.2f%%');ax.set_ylim(0,104);ax.set_ylabel('Retrospective accuracy (%)');ax.set_title('Measured configurations on the same 1,999 test records');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(OUT/f'Measured_Accuracy_Comparison.{ext}',dpi=300)
 plt.close(fig)
 fonts=Path(reportlab.__file__).parent/'fonts'
 for n,f in [('Vera','Vera.ttf'),('VeraBold','VeraBd.ttf')]:pdfmetrics.registerFont(TTFont(n,str(fonts/f)))
 styles=getSampleStyleSheet()
 for s in styles.byName.values():s.fontName='VeraBold' if s.name in ['Title','Heading1','Heading2'] else 'Vera'
 styles['Title'].fontSize=17;styles['Title'].leading=22
 styles['BodyText'].fontSize=9;styles['BodyText'].leading=13
 styles['Heading2'].fontSize=11;styles['Heading2'].leading=15
 flow=[]
 def p(t,style='BodyText'):flow.extend([Paragraph(t,styles[style]),Spacer(1,8)])
 def page(t):
  if flow:flow.append(PageBreak())
  p(t,'Title')
 def pic(path,maxh):
  from PIL import Image as PIL
  with PIL.open(path) as im:w,h=im.size
  ratio=min(499/w,maxh/h);flow.extend([Image(str(path),width=w*ratio,height=h*ratio),Spacer(1,9)])
 def table(rows,widths):
  rows=[[Paragraph(str(cell),styles['BodyText']) for cell in row] for row in rows]
  t=Table(rows,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f4d45c')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f5f7')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));flow.extend([t,Spacer(1,10)])
 page('Semantic Concepts: Four Symptom-Based Runs')
 p('Method update | Read with the existing results on pages 1-8','Heading2')
 p('The original manuscript grouped complaints by semantic concept, whereas the earlier figure on page 5 colours points by urgency. They answer different questions. Use the symptom-based figure below for the semantic analysis; keep the triage figure only as an auxiliary urgency visualization.')
 pic(OUT/'Symptom_Concepts_Four_Runs.png',390)
 p('<b>Figure 7.</b> Paired-text SapBERT embeddings grouped by arm, back, jaw, palpitations and shoulder mentions. Each panel contains 100 development complaints, with 20 per group. Seeds: 42, 99, 404 and 777. Crosses indicate the arithmetic centre of each coloured group. One PCA is fitted without labels to all 1,915 eligible development groups and reused in every panel. It is a diagnostic projection, not a replacement for the deployed PCA-64 transform.')
 p('Groups are derived from literal mentions in Clinical_Concept; rows mentioning multiple listed groups are excluded. They are not clinician-verified diagnoses, so the labels say “mention” rather than assuming every arm/back/jaw/shoulder mention is a pain diagnosis. All four runs are retained regardless of appearance; samples can overlap between runs.')
 p('The old image used concept embeddings; these figures use the current concept-plus-complaint representation and different sampled records. Identical separation is not expected. Close points indicate similar projected text representations, not correct triage predictions. Overlap is reported rather than removed.')
 page('The Exact Change from About 83% to 90%')
 p('The saved four-level baseline closest to “83%” is full-768 HistGradientBoosting at <b>83.64%</b>. The current selected classifier is Logistic Regression at <b>90.60%</b>: a gain of <b>6.95 percentage points</b> using unrounded scores. A same-family comparison starts with PCA-64 Logistic Regression at <b>82.44%</b>, giving a gain of <b>8.15 percentage points</b>.')
 table([['Measured configuration','Accuracy','Precision*','Recall*','F1*']]+[[n]+[f'{100*r["metrics"][k]:.2f}%' for k in keys] for n,r in records],[199,75,75,75,75])
 pic(OUT/'Measured_Accuracy_Comparison.png',235)
 p('*Precision, recall and F1 are macro averages across levels 0-3. Each row uses the same four-level data and previously examined 1,999-record partition. These are measured configurations, not isolated causal effects. The bars do not mean that each feature change alone caused the difference between neighbouring bars.')
 p('The tuned PCA-64 LR reference reaches 85.19%; the concept-only encoder with quadratic patient features, complaint details, PCA-128 and C=10 reaches 89.29%. The current paired-text/PCA-64/C=100 configuration reaches 90.60%. The final step adds 1.30 percentage points of accuracy and 1.26 points of macro F1, calculated before rounding.')
 p('The earlier and current encoders share the frozen SapBERT checkpoint. Changing the classifier family from the HGB reference must be acknowledged when quoting 83.64% to 90.60%; the same-family LR comparison is provided separately.')
 page('Method Improvements: What Was Actually Changed')
 steps=[('1. Preserve more information in the encoder input','Previously SapBERT encoded the English clinical concept alone. The selected encoder input now combines the concept, [SEP] and original complaint. The token limit increased from 64 to 128 to accommodate the longer input. The checkpoint, CLS pooling and L2 normalization were retained; SapBERT was not fine-tuned.'),('2. Represent patient-feature combinations','The final patient block has 50 features: 35 linear, squared and pairwise numeric terms plus 15 categorical indicators. These terms let a linear classifier represent combinations of patient measurements. The earlier fixed fused baseline used 22 patient features. Patient measurements were already present in that baseline.'),('3. Preserve explicit complaint details','Nineteen features represent severity, onset, exertion, radiation, breathing, fainting, sweating, nausea, history, negation and duration-related clues. They supplement the embedding. Original complaints already supplied these separate features in the 89.29% intermediate model; adding the paired embedding did not introduce them for the first time.'),('4. Tune dimensions and classifier regularization','Candidate PCA sizes and classifier settings were evaluated on grouped development folds. The final choice is 64 text components and balanced Logistic Regression C=100, rather than the intermediate PCA-128/C=10. Larger C means weaker regularization. Final input: 64 text + 50 patient + 19 detail features = 133.'),('5. Select using development evidence','Five-fold grouped validation keeps related complaints in the same fold. Preprocessing is fitted inside each training fold. Selection maximizes mean macro F1 subject to the original emergency-recall constraint, with accuracy used to break ties. The final model is fitted on 8,001 development records.'),('6. Retain weighting and verify deployment','Balanced weighting adjusts each training record’s contribution by class frequency; it does not create new records or change labels. It was already used in the LR baseline, so it cannot be claimed as a newly isolated cause of the gain. The shared GUI/CLI/batch adapter was checked against all 1,999 saved predictions and probabilities.')]
 for title,body in steps:p(title,'Heading2');p(body)
 p('No new training was performed to draw these graphs or assemble this PDF. The deployed bundle is unchanged. Individual percentage gains cannot be assigned to each step without controlled ablation experiments.')
 page('Interpretation and Reproducible Evidence')
 p('Manuscript-ready method summary','Heading2')
 p('We improved the prediction pipeline while retaining the frozen SapBERT encoder. The revised representation combines the English clinical concept with the original complaint, uses a 128-token input limit, and reduces the resulting 768-dimensional vector to 64 PCA components. These components are combined with quadratic patient features and 19 explicit complaint-detail features. Balanced Logistic Regression with C=100 was selected using five-fold grouped development validation. On the same previously examined four-level test partition, accuracy increased from 83.64% for the earlier full-768 HistGradientBoosting baseline to 90.60% for the selected pipeline. Compared with the earlier PCA-64 Logistic Regression baseline, accuracy increased from 82.44% to 90.60%. These comparisons reflect complete pipeline configurations, not the independent effect of any single change.')
 p('What the semantic plots establish','Heading2')
 p('Symptom-concept colours describe similarity of text, whereas triage labels describe urgency. A similar complaint can receive different urgency labels because of patient measurements and clinical context. The diagnostic plot retains approximately 23.5% of embedding variance in two dimensions; its coordinates do not include the full classifier input. Visual separation therefore cannot be used as a substitute for accuracy, recall or F1.')
 p('Current scores and remaining errors','Heading2')
 p('Current grouped CV means are 90.49% accuracy, 90.75% macro precision, 90.97% macro recall and 90.83% macro F1. Retrospective test values are 90.60%, 90.98%, 91.00% and 90.99%, respectively. Some Urgent and Standard class scores remain below 90%. From the 89.29% intermediate pipeline, Emergency recall decreases from 95.10% to 94.36% even though overall undertriage improves from 4.65% to 4.20%.')
 p('Scope of the evidence','Heading2')
 p('Labels and group partitions were unchanged. The test partition had been examined during the development history, and the selected CV scores are also subject to model-selection optimism. The results are retrospective and are not independent clinical validation. Saved English concepts were used for the reported study; live Ollama translation performance remains a separate evaluation. No claim is made that all future scores will exceed 90%.')
 p('Saved evidence used for this update','Heading2')
 for t in ['Baseline metrics/configurations: output/results/triage_four_level/results.json.','Current and intermediate results: triage_four_level_round7/retrospective_results.json and previous_round_result.json under output/results/.','Selection and CV: triage_four_level_round7/selection.json and cross_validation.csv.','Semantic sampling and provenance: reports/complaint_semantic_audit/protocol.json and verification.json.','Individual sample identifiers remain local; the shared package contains figures, aggregate metrics and provenance only.'] :p(t)
 def footer(c,doc):c.setFont('Vera',8);c.drawString(48,25,'SapBERT | Semantic concepts and method improvements');c.drawRightString(A4[0]-48,25,str(doc.page+8))
 while flow and isinstance(flow[-1],Spacer):flow.pop()
 appendix=ROOT/'tmp/pdfs/method_update_appendix.pdf'
 SimpleDocTemplate(str(appendix),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=38,bottomMargin=42).build(flow,onFirstPage=footer,onLaterPages=footer)
 assert len(PdfReader(appendix).pages)==4
 existing=ROOT/'reports/triage_selected_model/SapBERT_Final_Paper_Report.pdf'
 assert len(PdfReader(existing).pages)==8
 writer=PdfWriter();writer.append(existing);writer.append(appendix)
 writer.add_metadata({'/Title':'SapBERT Results, Semantic Concepts and Method Improvements'})
 with FINAL.open('wb') as f:writer.write(f)
 manifest=dict(existing_report_sha256=sha(existing),baseline_results_sha256=sha(base/'results.json'),current_results_sha256=sha(latest/'retrospective_results.json'),semantic_protocol=protocol,diagnostic_variance=projection.explained_variance_ratio_.tolist(),new_training=False,all_four_seeds_retained=True)
 (OUT/'Sources_and_Protocol.json').write_text(json.dumps(manifest,indent=2))
 (OUT/'Captions.txt').write_text('Symptom_Concepts_Four_Runs: Four fixed 100-complaint development samples coloured by rule-derived symptom mentions. Crosses are group centres. One label-blind diagnostic PCA fitted on 1,915 eligible development groups is reused. Current paired-text SapBERT embeddings; not the serving PCA and not triage accuracy. All seeds retained; between-run overlap allowed.\nMeasured_Accuracy_Comparison: Complete measured configurations on the same previously examined 1,999-record four-level partition. Differences are not isolated causal effects of individual changes.\n')
 with zipfile.ZipFile(ROOT/'output/pdf/SapBERT_Method_Update_Figures.zip','w',zipfile.ZIP_DEFLATED) as z:
  for path in sorted(OUT.iterdir()):z.write(path,path.name)
 print(FINAL)
if __name__=='__main__':build()
