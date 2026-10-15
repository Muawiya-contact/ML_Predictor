"""Explain current triage workflow and the four-run semantic diagnostic."""
from pathlib import Path
import json,zipfile,hashlib
import pandas as pd
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'output/results/complaint_semantic_audit';PDF=ROOT/'output/pdf/Roman_Urdu_Triage_Project_Workflow.pdf'
def build():
 verify=json.loads((SRC/'verification.json').read_text());assert verify['status']=='passed'
 plan=json.loads((SRC/'protocol.json').read_text());metrics=pd.read_csv(SRC/'metrics.csv')
 fonts=Path(reportlab.__file__).parent/'fonts'
 for name,file in [('Vera','Vera.ttf'),('VeraBold','VeraBd.ttf')]:pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
 styles=getSampleStyleSheet()
 for s in styles.byName.values():s.fontName='VeraBold' if s.name in ['Title','Heading1','Heading2'] else 'Vera'
 styles['Title'].fontSize=17;styles['Title'].leading=22;styles['BodyText'].fontSize=9;styles['BodyText'].leading=13
 flow=[]
 def p(t,style='BodyText'):flow.extend([Paragraph(t,styles[style]),Spacer(1,9)])
 def page(t):
  if flow:flow.append(PageBreak())
  p(t,'Title')
 def table(rows,widths):
  t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Vera'),('FONTNAME',(0,0),(-1,0),'VeraBold'),('FONTSIZE',(0,0),(-1,-1),8),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#f4d45c')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f5f7')]),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));flow.extend([t,Spacer(1,12)])
 def image(name):
  from PIL import Image as PIL
  path=SRC/'figures'/name
  with PIL.open(path) as im:w,h=im.size
  scale=min(499/w,420/h);flow.extend([Image(str(path),width=w*scale,height=h*scale),Spacer(1,10)])
 page('Roman Urdu Medical Triage: Project Workflow')
 p('Purpose and selected model','Heading2')
 p('This research project combines a patient complaint written in Roman Urdu or English with patient measurements to predict one of four supplied triage levels: 0 Emergency, 1 Urgent, 2 Standard and 3 Non-urgent. The selected pipeline is local Ollama translation, an anatomical check, frozen SapBERT embeddings, PCA-64 and Logistic Regression.')
 p('From input to output','Heading2')
 table([['Stage','Input or operation','Output'],['Complaint','Original wording + checked English','Paired text'],['SapBERT','Up to 128 input tokens','768 numerical features'],['PCA','Compress the embedding','64 text features'],['Combine features','Add patient and complaint details','133 total features'],['Logistic Regression','C=100; balanced training weights','Four class probabilities']],[119,240,140])
 p('The model uses 50 patient features and 19 complaint-detail features alongside the 64 text components. The application presents the predicted level and model probabilities; the embedding plots are analysis tools, not the prediction mechanism.')
 p('Current evaluation','Heading2')
 table([['Metric','Grouped CV mean','Retrospective test'],['Accuracy','90.49%','90.60%'],['Macro precision','90.75%','90.98%'],['Macro recall','90.97%','91.00%'],['Macro F1','90.83%','90.99%']],[199,150,150])
 p('The supplied dataset contains 10,000 records: 8,001 development records and 1,999 test records. The source labels remain unchanged. Class-specific scores do not all exceed 90%.')
 p('The test records were examined previously, so these are retrospective results rather than independent confirmation. Supplied/recovered English concepts were evaluated; live translation accuracy and clinical validity are not established. This is a research system, not a clinically validated triage service.')
 page('The Prediction Workflow in Simple Words')
 for title,text in [('1. Read the complaint','The user enters the original Roman Urdu or English complaint and patient measurements. Local Ollama supplies English text; the anatomical safety check rejects a translation that changes a named body part. In the saved research experiment, supplied/recovered English clinical concepts are used instead of measuring live translation.'),('2. Turn text into numbers','SapBERT receives English text + [SEP] + the original complaint. It converts the text into 768 learned numerical features, called an embedding. Each dimension is one number, not one word, symptom or diagnosis. Meaning is spread across the vector; the dimensions do not have simple human names.'),('3. Keep a compact text representation','PCA creates 64 combinations of the original 768 numbers. The fitted 64 components retain 92.10% of the development embedding variance. This is retained variation, not accuracy. The 2-D plot keeps just two components so it can fit on a page; classification uses 64.'),('4. Add patient information','The model combines 64 text numbers, 50 patient features and 19 complaint-detail features: 133 inputs. The patient features include encoded categories and quadratic numeric combinations. The detail features include severity words, duration and related mentions.'),('5. Predict urgency','The trained Logistic Regression model learns weights connecting these 133 inputs to the four supplied urgency labels. It calculates four class probabilities and uses the highest-probability class as its prediction. It does not find a dot on the plotted chart to make the decision.')]:p(title,'Heading2');p(text)
 p('The 128-token limit is separate: it limits how many text pieces the encoder reads, including both complaint versions and special tokens. Tokens may be words or word fragments. Text exceeding that limit is truncated. Neither 128 tokens nor 768 dimensions is the number of urgency classes.')
 page('CV, Class Balancing and Tuning')
 p('CV means cross-validation, not CSV','Heading2')
 p('CSV is a file format. Cross-validation splits the 8,001 development records into five grouped folds. For each setting, five models are trained: each time four folds train the model and the remaining fold checks it. Related complaint groups stay on the same side. Preprocessing is fitted inside each training fold. The five validation scores are averaged. The final selected model is then fitted on development data.')
 p('Balanced means weighted training errors','Heading2')
 p('The selected classifier uses balanced sample weights. A record in a less frequent class contributes more to the training loss. No records are duplicated and no labels are changed by this setting. The formula is total training records / (4 x records in that class), recalculated inside each fold.')
 table([['Level','Final development records','Weight per record'],['0 Emergency','1,632','1.226'],['1 Urgent','2,172','0.921'],['2 Standard','2,400','0.833'],['3 Non-urgent','1,797','1.113']],[149,200,150])
 p('A separate concept, balanced plotting samples, means taking the same number of records from each mention group. That sampling choice is only for this diagnostic; it is not how training class weights are calculated.')
 p('Tuning means comparing explicit settings','Heading2')
 p('The latest paired-text comparison tested PCA dimensions 64 and 128 with Logistic Regression C values 1, 10 and 100, using the same five folds. The selected setting was PCA-64 and C=100 with balanced weights. C controls the strength of regularization: larger C means a weaker penalty on large coefficients. It is not a confidence setting or the number of classes.')
 p('Selection used development macro F1 and the emergency-recall constraint, across the recorded search. Changing input representation and tuning the pipeline improved the measured scores while keeping the SapBERT checkpoint frozen and the classifier family as Logistic Regression. The individual contribution of each simultaneous change was not isolated.')
 p('The reported 90.60% accuracy and 90.99% macro F1 are retrospective results on previously examined test records. CV selection and repeated investigation can be optimistic; independent confirmation remains necessary.')
 page('Understanding the Embedding Plots')
 p('What a point and colour mean','Heading2')
 p('Each dot represents one complaint embedding after projection into two dimensions. A colour can represent a complaint mention, such as jaw or shoulder, or an urgency level, such as Emergency. Those colour schemes answer different questions. Similar complaint wording can occur at different urgency levels because patient measurements and other details also matter.')
 p('Why dots can overlap','Heading2')
 p('The plotted serving axes retain 17.53% of the embedding variation. The model uses 64 components retaining 92.10%, plus patient/detail features. A two-dimensional view can hide differences present in the remaining dimensions. Mixed colours do not by themselves prove poor prediction accuracy, and separated colours do not establish high accuracy.')
 p('Text representation and projection also affect the picture. The current encoder receives both the English concept and original complaint, which can include multiple symptoms and contextual details. Different samples or PCA fits can change the layout. The available old screenshots are not enough to isolate a single cause of their cleaner separation.')
 p('How the four runs are sampled','Heading2')
 p('Seeds 42, 99, 404 and 777 each select 100 development records: 20 Arm mentions, 20 Back mentions, 20 Jaw mentions, 20 Palpitations mentions and 20 Shoulder mentions. Selection is random within each group, not uniform across the entire dataset. Seeds make the results reproducible; all four runs are retained.')
 p('Mention groups are assigned by explicit word rules in the supplied English concepts. Rows with zero or multiple listed mentions are excluded, and duplicate complaint groups are removed from the pool. These are exploratory text labels, not independent clinical categories or diagnoses.')
 p(f"The eligible pool contains {verify['eligible_group_count']} complaint groups. The four runs contain {verify['distinct_records_across_runs']} distinct records overall. A run contains 100 distinct groups; different runs may share records. Test records are excluded.")
 p('The plots are not required to form separate colour islands. No points are moved, labels rewritten or runs discarded to improve appearance. The classifier is unchanged by this visualization.')
 page('Four Runs: Complaint Mentions in Serving PCA')
 image('01_Complaint_Mentions_Serving_PCA_Four_Runs.png')
 p('Figure 1. Current paired-input SapBERT embeddings, projected onto the first two components of the deployed PCA. Each panel contains 100 development complaints, with 20 from each rule-derived mention group. Four seeds were declared before plotting. The same fitted PCA and shared axes are used throughout.')
 p('Some complaint groups overlap because this view retains only two components. These colours are assigned from explicit text-mention rules; they are not the output of the clustering algorithm and do not represent urgency. A phrase mentioning an arm is not automatically an independently verified arm-pain diagnosis.')
 page('Same Records and Coordinates, Different Colours')
 image('02_Same_Complaints_Urgency_Serving_PCA_Four_Runs.png')
 p('Figure 2. Exactly the same sampled records and serving-PCA coordinates as Figure 1, coloured by supplied urgency label. The plot changes meaning when the colour label changes. Similar text can correspond to different urgency labels.')
 p('This is why visually mixed urgency colours cannot be treated as proof that semantic embeddings failed. The classifier also uses patient measurements and complaint details. Its predictive performance must be measured against held-out reference labels, not estimated by looking at the colours.')
 page('A Separate View for Complaint-Type Geometry')
 image('03_Complaint_Mentions_Diagnostic_PCA_Four_Runs.png')
 p('Figure 3. The same four samples in a separate label-blind two-dimensional PCA, fitted once to all eligible development vectors before viewing the runs. This projection was planned alongside the serving-PCA view. The PCA fit does not receive mention or urgency labels; the eligible pool is explicitly mention-filtered.')
 p('Jaw and palpitations mentions separate more clearly in this view, while arm, back and shoulder mentions still overlap. All four runs are shown. This diagnostic PCA is not the deployed PCA and does not change the classifier or its accuracy. It only gives a different two-dimensional view of the same saved embeddings.')
 page('Measured Geometry and What Can Be Claimed')
 order=['768-D paired SapBERT','64-D serving PCA','2-D serving PCA','2-D diagnostic PCA']
 grouped=metrics.groupby('space')
 def mean_sd(space,key):v=grouped.get_group(space)[key];return f'{v.mean():.3f} ({v.std(ddof=1):.3f})'
 table([['Space','Mention silhouette','Urgency silhouette','KMeans ARI']]+[[space,mean_sd(space,'mention_silhouette'),mean_sd(space,'urgency_silhouette'),mean_sd(space,'kmeans_ARI')] for space in order],[169,115,115,100])
 p('Values are mean (sample standard deviation) over the four runs. Silhouette describes separation by the indicated labels; higher is generally better. KMeans uses five clusters, a fixed seed and 20 initializations on normalized vectors. Adjusted Rand Index compares its unsupervised grouping with the rule-derived mentions; it is not a triage accuracy score.')
 p('The full 768-D nearest-neighbour mention-match mean is 91.75%; it asks whether the closest other complaint in the same sample has the same rule-derived mention. This is a descriptive retrieval statistic, not the reported 90.60% triage accuracy. The two percentages must not be substituted for each other.')
 p('Supplementary metrics','Heading2')
 p('The accompanying CSV reports within/between-group cosine distances, silhouette, nearest-neighbour matching, KMeans ARI and ANOSIM. ANOSIM uses 499 label permutations per run/space, with (exceedances + 1)/500 and Holm correction across all 16 tests. These are exploratory within-sample tests; they do not remove the dependence created by deriving labels from the input text or by overlap between samples.')
 p('Interpretation for research reporting','Heading2')
 p('Semantic similarity and urgency prediction are evaluated separately. Four fixed 100-complaint samples use the current SapBERT embeddings. Complaint-mention groups show structure, while urgency colours overlap. All runs and measured geometry are retained. The classifier uses PCA-64 plus patient/detail features; these visualizations do not alter its 90.60% retrospective accuracy.')
 p('For an independently evaluated complaint-category experiment, the next requirement is reviewed category labels or the original category-labelled program/data. The current rule-derived grouping is useful for exploration but must not be presented as clinician-validated semantic ground truth.')
 p('Method references','Heading2')
 for label,url in [('PCA and explained variance','https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html'),('Balanced sample weights','https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html'),('Grouped cross-validation','https://scikit-learn.org/stable/modules/cross_validation.html')]:p(f'<link href="{url}" color="blue">{label}: scikit-learn documentation</link>')
 def footer(c,doc):c.setFont('Vera',8);c.drawString(48,25,'SapBERT workflow and exploratory complaint geometry');c.drawRightString(A4[0]-48,25,str(doc.page))
 SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=38,bottomMargin=42,title='Roman Urdu Medical Triage: Project Workflow').build(flow,onFirstPage=footer,onLaterPages=footer)
 captions='''Four-run exploratory complaint-mention diagnostic. Model artifacts unchanged.

Figure 1: 100 complaints per seed (42, 99, 404, 777), 20 per rule-derived mention group, projected using the first two fitted serving PCA components. Colours indicate literal mentions, not verified clinical categories.
Figure 2: Same records and coordinates as Figure 1, coloured by supplied urgency levels 0-3. This demonstrates that semantic and urgency labels ask different questions.
Figure 3: Same four samples projected using a separate label-blind PCA fitted once on the eligible development pool. This diagnostic projection is not deployed. All predeclared seeds are retained.

Mention groups are derived from the Clinical_Concept text by the rules in protocol.json. They are not independent category annotations. No test records are used; samples can overlap across runs. Do not label these plots as classification accuracy or as reproductions of the old complaint-category experiment.
Full-run quantitative metrics and the sampling protocol are included. Nearest-mention matching is not triage accuracy. Private sampled records are intentionally excluded.
'''
 (SRC/'Captions_and_Interpretation.txt').write_text(captions)
 with zipfile.ZipFile(ROOT/'output/pdf/SapBERT_Complaint_Group_Figures.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in ['protocol.json','metrics.csv','summary.csv','verification.json','Captions_and_Interpretation.txt']:z.write(SRC/f,'Complaint_Group_Audit/'+f)
  for f in sorted((SRC/'figures').iterdir()):z.write(f,'Complaint_Group_Audit/figures/'+f.name)
 print(PDF)
if __name__=='__main__':build()
