"""Compare the uploaded three-level report with the verified current four-level study.

Read-only analysis of saved experiments; no retraining or changes to labels/models.
All confusion counts and comparison deltas are derived from source artifacts.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--previous-pdf',type=Path,required=True,help='Uploaded three-level SapBERT full/PCA comparison PDF')
parser.add_argument('--output',type=Path,default=ROOT/'output/pdf/SapBERT_Previous_vs_Latest_Comparison.pdf')
args=parser.parse_args()
def read(p):return json.loads((ROOT/p).read_text())
OLD=Path('output/results/triage_fixed_full_pca');NEW=Path('output/results/triage_four_level');CUR=Path('output/results/triage_four_level_round5')
old=read(OLD/'results.json');new=read(NEW/'results.json');families=read(CUR/'family_results.json')
previous=read(CUR/'previous_round_result.json');selected=read(CUR/'retrospective_results.json')[-1]
transition=read('reports/label_transition_audit/protocol.json');controlled=read('reports/label_transition_audit/results.json')
verification=read(CUR/'verification.json');refinement=read('reports/triage_error_investigation/refinement_verification.json')
old_pdf=args.previous_pdf
new_pdf=ROOT/'output/pdf/SapBERT_Final_Report.pdf'
old_text=' '.join(p.extract_text() for p in PdfReader(old_pdf).pages)
# Recompute the source matrices and the four headline scores before publishing.
for directory,rows,labels in [(OLD,old,[1,2,3]),(NEW,new,[0,1,2,3])]:
 for r in rows:
  frame=pd.read_csv(ROOT/directory/(r['id']+'_predictions.csv'))
  cm=confusion_matrix(frame.reference,frame.predicted,labels=labels)
  np.testing.assert_array_equal(cm,r['confusion']);assert cm.sum()==1999
  rep=classification_report(frame.reference,frame.predicted,labels=labels,output_dict=True,zero_division=0)
  for metric,expected in [('accuracy',rep['accuracy']),('precision_macro',rep['macro avg']['precision']),('recall_macro',rep['macro avg']['recall']),('macro_f1',rep['macro avg']['f1-score'])]:
   np.testing.assert_allclose(r['metrics'][metric],expected,atol=1e-12)
  if directory==OLD:
   for key in ['accuracy','precision_macro','macro_f1']:
    assert f"{100*r['metrics'][key]:.2f}%" in old_text
for r in families:
 frame=pd.read_csv(ROOT/CUR/(r['candidate']+'_family_predictions.csv'))
 np.testing.assert_array_equal(confusion_matrix(frame.reference,frame.predicted,labels=[0,1,2,3]),r['confusion'])
assert verification['status']=='passed'
a=pd.read_csv(ROOT/'output/results/triage_10000_research/dataset_with_splits.csv').sort_values('source_row')
b=pd.read_csv(ROOT/NEW/'dataset_with_splits.csv').sort_values('source_row')
assert len(a)==len(b)==10000
assert int((a.partition.to_numpy()!=b.partition.to_numpy()).sum())==3122
shared=len(set(a.loc[a.partition.eq('test'),'source_row'])&set(b.loc[b.partition.eq('test'),'source_row']))
assert shared==438
names={'logreg':'Logistic Regression','hgb':'HistGradientBoosting','rf':'Random Forest'}
oldmap={r['id']:r for r in old};newmap={r['id']:r for r in new}
figdir=ROOT/'output/images/Historical_Comparison';figdir.mkdir(parents=True,exist_ok=True)
output=args.output
fonts=Path(reportlab.__file__).parent/'fonts'
for name,file in [('Vera','Vera.ttf'),('VeraBold','VeraBd.ttf')]:pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
styles=getSampleStyleSheet()
for style in styles.byName.values():style.fontName='Vera';style.textColor=colors.black
for key in ['Title','Heading1','Heading2']:styles[key].fontName='VeraBold'
styles['Title'].fontSize=17;styles['Title'].leading=21
styles['Heading2'].fontSize=11;styles['Heading2'].leading=14
styles['BodyText'].fontSize=9;styles['BodyText'].leading=12
story=[]
def p(s,style='BodyText'):story.extend([Paragraph(s,styles[style]),Spacer(1,7)])
def page(title):
 if story:story.append(PageBreak())
 p(title,'Title')
def table(rows,widths,size=8,pad=4):
 st=styles['BodyText'].clone('cell');st.fontSize=size;st.leading=size+2
 data=[[Paragraph(str(c),st) for c in row] for row in rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f4d35e')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),pad),('BOTTOMPADDING',(0,0),(-1,-1),pad),('LINEBELOW',(0,0),(-1,0),.6,colors.black)]))
 story.extend([t,Spacer(1,9)])
def pct(x):return f'{100*x:.2f}'
def delta(a,b):return f'{100*(b-a):+.2f}'
def pic(path,h):story.append(Image(str(path),width=495,height=h,kind='proportional'))
def mat(ax,r,title,labels,normalized=False):
 cm=np.array(r['confusion']);pc=100*cm/cm.sum(1,keepdims=True)
 ax.imshow(pc,vmin=0,vmax=100,cmap='Blues');ax.set_title(title,fontsize=11)
 ax.set_xticks(range(len(labels)),labels);ax.set_yticks(range(len(labels)),labels)
 ax.set_xlabel('Predicted level');ax.set_ylabel('Reference level')
 for i in range(len(labels)):
  for j in range(len(labels)):
   text=f'{pc[i,j]:.1f}%' if normalized else f'{cm[i,j]}\n{pc[i,j]:.1f}%'
   ax.text(j,i,text,ha='center',va='center',fontsize=10,color='white' if pc[i,j]>55 else 'black')

audit=[]
page('Previous versus Latest SapBERT Report')
p('Three-level reference report compared with the current four-level study','Heading2')
p('This document compares your uploaded seven-page SapBERT_Full768_PCA64_Comparison-1.pdf with the eleven-page SapBERT_Final_Report.pdf. "Previous" means the earlier report you called ideal; it is a reference result, not a clinically established gold standard. No model was retrained for this document.')
p('Main answer','Heading2')
p('Both studies test 1,999 records. Smaller confusion-matrix counts do not mean that the new model used fewer test cases. Class sizes, class definitions and which records enter the test set changed. Percentage-based scores really are lower for the new four-level task, but this is not a controlled measure of deterioration on the old task.')
oldbest=oldmap['fused_64_rf']
keys=[('accuracy','Accuracy'),('precision_macro','Macro precision'),('recall_macro','Macro recall'),('macro_f1','Macro F1')]
table([['Metric (%)','Previous RF / PCA-64','Latest selected LR','Change (pp)']]+[[label,pct(oldbest['metrics'][k]),pct(selected['metrics'][k]),delta(oldbest['metrics'][k],selected['metrics'][k])] for k,label in keys],[175,110,110,100])
p('This headline compares different tasks and selected configurations. A negative difference means a lower reported score; it does not isolate the effect of adding one class. Same-family, fixed-configuration comparisons follow.')
table([['Question','Evidence-based answer'],['Are smaller matrix numbers bad?','Not by themselves. Compare correct / total within each row.'],['Are four levels better?','They meet the requested four-level specification and allow finer distinctions. Clinical usefulness is not established by the extra level alone.'],['Are the new scores as high?','No. The latest selected accuracy is 89.29%, versus 99.35% for the old RF task.'],['Has the latest refinement helped?','Versus the immediately previous four-level model: +0.30 pp accuracy and +0.30 pp macro F1. The development gain remains uncertain.']],[170,325])
p('Reading guide: pp means percentage points; all precision, recall and F1 values are macro averages unless a class is named. Count tables and percentage matrices are deliberately distinguished. Differences are calculated before rounding, so subtracting displayed rounded values can differ by 0.01 pp.')

page('Why the Confusion Counts Became Smaller')
table([['Property','Previous report','Latest report'],['Total source records','10,000','10,000'],['Development / test','8,001 / 1,999','8,001 / 1,999'],['Classes','1, 2, 3','0, 1, 2, 3'],['Test support by class','1: 677; 2: 943; 3: 379','0: 408; 1: 542; 2: 600; 3: 449'],['Matrix cells','3 x 3 = 9','4 x 4 = 16'],['Identical records in the two test sets',str(shared),str(shared)]],[175,160,160])
p('The largest old test class had 943 records. The largest new class has 600. A correct-count cell cannot exceed its row total. The old RF cell of 933 means 933 / 943 = 98.94% recall. The latest Standard cell of 522 means 522 / 600 = 87.00% recall. These are different class definitions, so the percentage difference is descriptive, not a paired class comparison.')
p('Not every count decreased: the old RF class-3 diagonal is 379, whereas the latest Non-urgent diagonal is 411. Its row contains 449 records, so recall is 91.54%, below the old 379 / 379 = 100%. A larger count can coexist with a lower percentage.')
p('The complete label transition (all 10,000 records)','Heading2')
table([['Old label','New 0','New 1','New 2','New 3','Old total']]+[[str(i)]+[transition['label_transition'][str(j)][str(i)] for j in range(4)]+[transition['old_counts'][str(i)]] for i in [1,2,3]]+[['New total']+[transition['new_counts'][str(j)] for j in range(4)]+[10000]],[95,80,80,80,80,80])
p('Old class 1 splits into new 0 and 1. Old class 2 spreads across new 1, 2 and 3; old class 3 spreads across new 2 and 3. This is not a one-to-one renaming or simply an extra empty class. Subtracting cells by their numeric labels would be misleading.')
p('The grouped stratified split was rebuilt for the new targets: 3,122 records change partition membership (1,561 in each direction). Both test sets still have 1,999 records, with 438 shared records. Earlier suggestions that a larger old test set explained these particular PDFs were not correct.')

page('What Actually Explains the Score Gap?')
p('A controlled diagnostic already separates target changes from split changes. It fits the old and new labels on the same current rows, split, SapBERT vectors, PCA-64, 22 patient features and fixed classifier settings. It is a retrospective diagnostic, not another model-selection round.')
table([['Classifier','Old-label accuracy','New-label accuracy','Change (pp)','Old / new F1']]+[[names[f],pct(controlled[2*i]['accuracy']),pct(controlled[2*i+1]['accuracy']),delta(controlled[2*i]['accuracy'],controlled[2*i+1]['accuracy']),pct(controlled[2*i]['macro_f1'])+' / '+pct(controlled[2*i+1]['macro_f1'])] for i,f in enumerate(['logreg','hgb','rf'])],[155,85,85,75,95])
p('The large gap remains when the split is held fixed. This supports a change in how predictable the supplied labels are from the available inputs. It does not prove that the new labels are wrong or establish a maximum achievable accuracy.')
table([['Observed change','Why a score can change','Judgment'],['Four target classes','More decision boundaries; distinctions previously inside one class can now count as errors.','Harder observed task; not automatically a worse design.'],['Different class proportions','Macro scores give each class equal weight, now one quarter rather than one third.','Different averaging problem; compare class recall too.'],['Changed test membership','Different cases can be easier or harder despite an identical test-set size.','Confounder in direct PDF comparisons.'],['Structured inputs matter','Old text-only accuracy is around 60%, but fusion is around 99%. Old labels are much more predictable with patient features.','Observed association, not proof of leakage or clinical correctness.'],['Concept compression','The existing audit flags lost or conflicting duration and other complaint details.','Plausible source of missing information; lexical flags need interpretation.'],['More PCA components / tuning','Latest model uses PCA-128, quadratic patient inputs and complaint details.','Measured improvement within the four-level task; no guarantee of 99%.']],[120,255,120],8,4)
p('Renumbering identical classes alone would not change accuracy or macro F1. The transition table demonstrates that records were regrouped here. No investigation can assign an exact causal contribution to each changed prediction without the unavailable label-assignment rules.')

for view,title in [('text','Text Only'),('fused','Text plus Patient Features')]:
 page('Every Baseline Score: '+title)
 p('Each row compares the same fixed classifier and input representation across the two reported tasks. A / P / R / F1 mean accuracy / macro precision / macro recall / macro F1. All entries are percentages; differences are new minus old, in pp. Recall comes from the saved prediction artifacts even where the old summary omitted it.')
 rows=[['Condition','Metric','Previous 3-level','Current 4-level','Change (pp)']]
 for pc in [768,64]:
  for f in names:
   key=f'{view}_{pc}_{f}';o=oldmap[key];n=newmap[key]
   for k,label in keys:
    rows.append([f'{f.upper()} / {pc}' if k=='accuracy' else '',label,pct(o['metrics'][k]),pct(n['metrics'][k]),delta(o['metrics'][k],n['metrics'][k])])
    audit.append(dict(condition=key,metric=k,old=o['metrics'][k],new=n['metrics'][k],change_pp=100*(n['metrics'][k]-o['metrics'][k])))
 table(rows,[115,115,95,95,75],8,3)
 p('Why these values are lower: all these fixed-setting results face the new target definitions and a different grouped test membership. The controlled diagnostic confirms a substantial target-related gap, but cannot attribute a unique cause to each cell or decimal. Lower scores are worse agreement with their own reference labels, not proof that the encoder broke.')
 p('These are baseline conditions, not the latest tuned GUI model. Full 768-D and PCA-64 use the same frozen encoder. PCA reduces representation size; it does not create labels or improve results automatically.')

page('Tuned Families and the Latest Refinement')
p('Old fused PCA-64 versus each latest tuned family. Settings and label task differ; these are descriptive changes, not an isolated classifier experiment.')
rows=[['Family / metric','Old PCA-64','Latest tuned','Change (pp)']]
for r in families:
 f=r['config']['classifier'];o=oldmap['fused_64_'+f]
 for k,label in keys:rows.append([names[f]+' / '+label,pct(o['metrics'][k]),pct(r['metrics'][k]),delta(o['metrics'][k],r['metrics'][k])])
table(rows,[230,90,90,85],8,3)
p('A fairer within-task comparison: immediately prior versus latest selected model','Heading2')
initial=read(CUR/'retrospective_results.json')[0]
rows=[['Metric','Initial ref.','Prior C=100','Latest C=10','Latest - prior']]
for k,label in keys+[('emergency_recall','Emergency recall'),('under_triage_rate','Under-triage'),('over_triage_rate','Over-triage')]:rows.append([label,pct(initial['metrics'][k]),pct(previous['metrics'][k]),pct(selected['metrics'][k]),delta(previous['metrics'][k],selected['metrics'][k])+' pp'])
for k,label in [('qwk','Quadratic weighted kappa'),('mae','Mean absolute level error')]:rows.append([label,f"{initial['metrics'][k]:.4f}",f"{previous['metrics'][k]:.4f}",f"{selected['metrics'][k]:.4f}",f"{selected['metrics'][k]-previous['metrics'][k]:+.4f}"])
table(rows,[195,70,75,75,80],7.5,3)
p('Initial reference means the original tuned PCA-64 LR C=10 model, not the fixed C=1 baseline. Higher accuracy, precision, recall, F1 and kappa are better; lower level error and under-/over-triage are better for these statistical measures. Under-triage is unchanged. The two models use the same rows and 197-feature design, with stronger LR regularization in the latest model. No label editing or SapBERT fine-tuning took place.')
p('Development mean macro F1: 89.38% to 89.65%. The conditional paired gain interval is -0.08 to +0.64 pp and includes zero. The observed improvement is modest and not yet reliably established; repeated selection adds uncertainty beyond that interval.')

page('Latest Four-Level Matrix: Counts and Meaning')
fig,axes=plt.subplots(1,2,figsize=(10,4.3));mat(axes[0],selected,'Count and row percentage',[0,1,2,3]);mat(axes[1],selected,'Row-normalized recall view',[0,1,2,3],True)
fig.tight_layout();path=figdir/'latest_counts_and_percentages.png';fig.savefig(path,dpi=180);plt.close(fig);pic(path,230)
rows=[['Level','Support','Correct','Precision %','Recall %','F1 %']]
for i,label in enumerate(['Emergency','Urgent','Standard','Non-urgent']):
 r=selected['report'][str(i)];rows.append([f'{i} {label}',int(r['support']),selected['confusion'][i][i],pct(r['precision']),pct(r['recall']),pct(r['f1-score'])])
table(rows,[140,65,65,75,75,75],8,4)
p('Read every row','Heading2')
p('Emergency: 388 correct and 20 predicted Urgent. Urgent: 464 correct, 34 predicted Emergency, 42 Standard and 2 Non-urgent. Standard: 522 correct, 49 predicted Urgent and 29 Non-urgent. Non-urgent: 411 correct and 38 predicted Standard.')
cm=np.array(selected['confusion']);errors=int(cm.sum()-np.trace(cm));adj=sum(cm[i,j] for i in range(4) for j in range(4) if abs(i-j)==1)
p(f'Total: {int(np.trace(cm))} correct / 1,999 = 89.29% accuracy; {errors} errors. {adj} errors are between neighbouring levels. Urgent and Standard have lower recall than the two outer classes. This identifies the main observed weakness; it does not justify changing reference labels.')
p('Recall = diagonal / row total. Precision = diagonal / column total. F1 balances precision and recall. Macro averages weight each class equally. Adding a class changes these denominators and weights, so a raw count of 300 cannot be interpreted as "30% accuracy".')
p('The old and new labels are not equivalent; class-by-class subtraction is therefore intentionally not presented. The following six pages include every old and new baseline matrix cell as a count and row percentage.')

for f,name in names.items():
 for view,label in [('text','text only'),('fused','text + patient features')]:
  page(name+': '+label)
  p('Previous three-level matrix on the left; current fixed four-level baseline on the right. Top: full 768-D. Bottom: PCA-64. Every matrix sums to 1,999. Each cell shows count and percentage of its reference row; color uses the same 0-100% scale.')
  fig,axes=plt.subplots(2,2,figsize=(10,9))
  for row,pc in enumerate([768,64]):
   mat(axes[row,0],oldmap[f'{view}_{pc}_{f}'],f'Previous / {pc}-D',[1,2,3])
   mat(axes[row,1],newmap[f'{view}_{pc}_{f}'],f'Current baseline / {pc}-D',[0,1,2,3])
  fig.tight_layout(pad=2);path=figdir/f'{f}_{view}_matrices.png';fig.savefig(path,dpi=180);plt.close(fig);pic(path,510)
  p('Good direction: more of each row on its diagonal and less in off-diagonal cells. Smaller counts caused by smaller class support are neutral; lower diagonal percentages indicate lower recall on that task. Off-diagonal cells below the diagonal are over-triage; above it are under-triage within each ordered label system. Numeric labels must not be aligned across the two systems as if equivalent.')

page('Latest Tuned HGB and Random Forest')
p('These are the latest tuned four-level family results, rather than the fixed baselines on the preceding pages. Both use PCA-64 and original-complaint details. The deployed winner remains Logistic Regression; family comparisons do not override its selection rule.')
fig,axes=plt.subplots(1,2,figsize=(10,4.3))
for ax,f in zip(axes,['hgb','rf']):
 r=next(x for x in families if x['config']['classifier']==f)
 mat(ax,r,names[f],[0,1,2,3])
fig.tight_layout();path=figdir/'latest_tuned_tree_matrices.png';fig.savefig(path,dpi=180);plt.close(fig);pic(path,230)
rows=[['Family / level','Support','Precision %','Recall %','F1 %']]
for f in ['hgb','rf']:
 result=next(x for x in families if x['config']['classifier']==f)
 for i in range(4):
  r=result['report'][str(i)];rows.append([f.upper()+' / '+str(i),int(r['support']),pct(r['precision']),pct(r['recall']),pct(r['f1-score'])])
table(rows,[175,80,80,80,80],8,4)
p('Each cell again shows its count and row percentage. The row totals are identical to those of the latest LR matrix, so comparisons among these three current family results share the same test records and targets. Their feature designs and tuned settings differ.')
p('Latest HGB: 800 iterations, learning rate 0.05, 7 leaves, minimum leaf size 10, L2=5, balanced weights. Latest RF: 500 trees, minimum leaf size 2, feature fraction 0.3, balanced weights. The old fixed PDF used HGB 300 iterations / rate 0.1 and RF 300 trees / leaf size 3 / feature fraction 0.7. These setting changes are an additional reason not to attribute every old-to-new difference solely to the extra level.')

cv=pd.read_csv(ROOT/CUR/'cv_summary.csv')
for idx,part in enumerate([cv.iloc[:35],cv.iloc[35:]]):
 page('Latest Development Search'+(' - Continued' if idx else ''))
 p('All 68 settings and 340 full-size fits are retained. These five-fold development scores have no direct counterpart in the old fixed-comparison PDF. They must not be subtracted from old test scores. Values are percentages; SD is F1 fold standard deviation in pp.')
 table([['Configuration','F1','SD','Accuracy','Emergency recall']]+[[r.candidate,pct(r.macro_f1),pct(r.f1_std),pct(r.accuracy),pct(r.emergency_recall)] for r in part.itertuples()],[215,60,50,70,100],7,2)
 p('Highest mean macro F1 wins subject to the unchanged original emergency-recall threshold; accuracy breaks ties. The selected LR detail128 C=10 achieves 89.65% mean F1 and 94.06% emergency recall. More candidates can overfit development selection; these figures are not new independent validation.')

page('Other Values: Features, Confidence and Audits')
table([['Quantity','Previous report','Latest selected model','Meaning'],['Embedding dimensions','768','768','Same frozen SapBERT encoder.'],['PCA dimensions','64','128','More retained components; not a performance percentage.'],['Patient / detail features','22 / 0','50 / 19','Quadratic numeric terms and original-complaint details.'],['Total fused inputs','790 full / 86 PCA','197','128 + 50 + 19. Larger is not automatically better.'],['PCA retained variance','91.31% at 64','95.51% at 128','Different PCA size/development partition; more variance is not clinical accuracy.']],[125,100,100,170],8,4)
diag=read('reports/triage_error_investigation/diagnostics.json');nd=verification['selected_probability_diagnostics']['development'];nt=verification['selected_probability_diagnostics']['retrospective_test']
table([['Probability diagnostic','Prior 4-level OOF','Latest OOF','Latest test'],['MCC',f"{diag['multiclass_mcc']:.4f}",f"{nd['mcc']:.4f}",f"{nt['mcc']:.4f}"],['Log loss',f"{diag['multiclass_log_loss']:.4f}",f"{nd['log_loss']:.4f}",f"{nt['log_loss']:.4f}"],['Multiclass Brier score',f"{diag['multiclass_brier']:.4f}",f"{nd['multiclass_brier']:.4f}",f"{nt['multiclass_brier']:.4f}"],['10-bin calibration gap',f"{diag['confidence_ece_10_bins']:.4f}",f"{nd['ece_10_bins']:.4f}",f"{nt['ece_10_bins']:.4f}"]],[195,100,100,100],8,4)
p('MCC rises; log loss, Brier score and calibration gap fall: favorable descriptive changes in development predictions. These quantities were not reported in the old ideal PDF, so no old-versus-new probability comparison is invented. OOF means out-of-fold development predictions.')
p('Audit values are not accuracy percentages','Heading2')
p('The pre-refinement model made 881 development errors: 878 adjacent-level errors, 354 Urgent/Standard errors, 220 errors with at least 90% confidence, and 345 errors shared by all three classifier families. These overlapping counts describe the prior model and must not be added together or presented as the latest test errors.')
p('The input audit flags 567 duration disagreements, 142 durations not recovered in concepts, and 48 family-word omissions. Flags are not confirmed labelling errors. The learning audit has 90 fits: 60 nested-size learning fits and 30 detail comparisons. Its 30 full-size detail fits are already included in the combined 340; these totals must not be summed as independent experiments.')
p('The latest PDF retains the learning audit unchanged. Its nine full-size conditions and graph values are reproduced on the next page so their older-stage meaning is explicit.')
z=np.load(ROOT/CUR/'development_oof.npz');oof_cm=confusion_matrix(z['reference'],z['selected'].argmax(1),labels=[0,1,2,3])
table([['Latest development OOF','Pred. 0','Pred. 1','Pred. 2','Pred. 3']]+[[str(i)]+[str(x) for x in row] for i,row in enumerate(oof_cm)],[195,75,75,75,75],8,3)
p('This matrix totals 8,001 development predictions, not 1,999 test predictions. Its larger cells do not imply higher performance; it evaluates a different sample through out-of-fold prediction.')


page('Learning Audit and Literature: Unchanged Context')
learning=pd.read_csv(ROOT/'output/results/triage_learning_detail_audit/summary.csv')
rows=[['Audit condition','Accuracy %','F1 %','Emergency recall %']]
for f,name in names.items():
 for v,lab in [('baseline','Prior features'),('concept_details','Concept details'),('complaint_details','Original details')]:
  r=learning[learning.classifier.eq(f)&learning.variant.eq(v)&learning.fraction.eq(1)].iloc[0];rows.append([name+' / '+lab,pct(r.accuracy),pct(r.macro_f1),pct(r.emergency_recall)])
table(rows,[255,75,75,90],7.5,3)
p('These are earlier development audit conditions, not the C=10 refinement. Nested subsets use 25%, 50%, 75% and 100% of each training fold; validation rows stay fixed. The learning curves suggest potential benefit from more distinct training information but do not predict a score at 20,000 rows.')
pic(ROOT/CUR/'figures/learning_curves.png',145)
lit=read('reports/triage_four_level/literature_sources.json')
table([['Literature model','Accuracy %','Recall %','F1 %']]+[[name,pct(m['accuracy']),pct(m['recall']),pct(m['f1'])] for name,m in lit['models'].items()],[255,80,80,80],8,3)
p('These four literature rows are unchanged between the reports: every delta is zero. They describe a different binary KTAS task and published metric definitions, not our four-level macro metrics. They cannot establish that either project model is clinically superior. Source carried forward from both PDFs: Seo et al. (2025), Scientific Reports 15, 16870, Table 2, DOI 10.1038/s41598-025-99874-0. This report does not newly audit that external study.')

page('Conclusion: Which Changes Are Good or Bad?')
table([['Change','Assessment','Why'],['Lower confusion counts','Neutral on their own','Both tests have 1,999 records; per-class support changes. Use row percentages.'],['Lower old-to-new accuracy / F1','Lower agreement on a different task','The label transition changes what must be predicted. Controlled fits still show a large gap.'],['Four instead of three levels','Meets the requested specification','Finer output categories are possible, but correctness and clinical benefit require their own evidence.'],['Latest versus prior four-level scores','Modest observed improvement','Accuracy +0.30 pp; F1 +0.30 pp; emergency recall +1.23 pp. Development uncertainty includes no gain.'],['Urgent / Standard recall','Remaining weakness','Latest recall is 85.61% / 87.00%; neighbouring-level confusion dominates errors.'],['Separating experiment scopes','Necessary for interpretation','Do not describe old three-level 99% and current four-level results as the same experiment.'],['Claiming a definite cause for every error','Not supported','No assignment rules or independent review establish why each reference label was chosen.']],[120,150,225],8,4)
p('Recommended wording for the article','Heading2')
p('"The previous three-level and current four-level experiments used the same number of test records but different target definitions and partly different test membership. A controlled comparison on a common split retained a substantial performance gap, indicating that the four-level targets were less predictable from the available inputs under the tested configurations. The final four-level model achieved 89.29% accuracy and 89.73% macro F1 in retrospective evaluation. These results are not a direct degradation estimate for the previous task or independent clinical validation."')
p('Source and verification notes','Heading2')
p('Previous source: SapBERT_Full768_PCA64_Comparison-1.pdf, 7 pages. Latest source: SapBERT_Final_Report.pdf, 11 pages. All 24 baseline matrices and their headline metrics were recomputed from saved predictions; old displayed summary values were checked against PDF text. Current scores and the 68-setting table use verified round-five artifacts. The label-transition audit is historical evidence; this document runs no new training and edits no labels.')
p('Internal evidence: output/results/triage_fixed_full_pca; reports/triage_four_level; reports/triage_four_level_round5; reports/label_transition_audit; reports/triage_error_investigation; reports/triage_learning_detail_audit. Unreported values are identified rather than guessed. No direct subtraction of non-equivalent class cells is made.')

def footer(canvas,doc):
 canvas.setFont('Vera',8);canvas.drawString(40,24,'SapBERT | Previous and latest report comparison');canvas.drawRightString(A4[0]-40,24,str(doc.page))
output.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(output),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=35,bottomMargin=40,title='SapBERT Previous versus Latest: Complete Comparison').build(story,onFirstPage=footer,onLaterPages=footer)
pd.DataFrame(audit).to_csv(figdir/'score_differences.csv',index=False)
(figdir/'verification.json').write_text(json.dumps(dict(status='passed',baseline_matrices_verified=24,test_rows_per_matrix=1999,shared_test_records=shared,old_pdf_sha256=hashlib.sha256(old_pdf.read_bytes()).hexdigest(),latest_pdf_sha256=hashlib.sha256(new_pdf.read_bytes()).hexdigest()),indent=2))
print(output)
