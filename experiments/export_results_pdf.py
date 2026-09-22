"""Export the completed comparison tables and figures to a printable PDF."""
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import csv
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'output/results'
OUTPUT = ROOT / 'output/pdf/SBERT_Classifier_Comparison.pdf'


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fonts = Path(reportlab.__file__).parent / 'fonts'
    pdfmetrics.registerFont(TTFont('Vera', str(fonts/'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('VeraBold', str(fonts/'VeraBd.ttf')))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.textColor = colors.black
        style.fontName = 'VeraBold' if style.name in ['Title', 'Heading2'] else 'Vera'
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 12
    story = []
    def para(text, style='BodyText'):
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 7))
    para('SBERT Classifier Comparison', 'Title')
    para('Sections 3.4 and 3.5 | Research experiment results', 'Heading2')
    para('Encoder: sentence-transformers/all-mpnet-base-v2. Frozen, normalized 768-dimensional embeddings of existing English translations. PCA reduces the text representation to 64 dimensions.')
    para('Dataset: 2,252 synthetic cardiac records. Stratified split: 1,801 training and 451 test rows; seed 42. Every comparison uses the same test rows. PCA and patient-feature preprocessing are fitted on training rows only.')
    rows = list(csv.DictReader((RESULTS / 'all_comparison_metrics.csv').open()))
    names = {'LogisticRegression':'Logistic Regression','HistGradientBoosting':'Hist Gradient Boosting','RandomForest':'Random Forest'}
    for feature, title in [('text_only','Text-only comparison'), ('fusion','Text plus patient features')]:
        para(title, 'Heading2')
        data = [['Text size','Classifier','Accuracy','Precision*','F1*']]
        for r in rows:
            if r['feature_set'] == feature:
                data.append(['768-D' if r['representation']=='full_768' else 'PCA-64',names[r['classifier']],
                             *[f'{float(r[k])*100:.2f}%' for k in ['accuracy','precision_macro','f1_macro']]])
        table = Table(data, colWidths=[60,153,78,78,70], repeatRows=1)
        table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Vera'),('FONTNAME',(0,0),(-1,0),'VeraBold'),('FONTSIZE',(0,0),(-1,-1),9),
                                  ('LINEBELOW',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,-1),(-1,-1),.5,colors.black),
                                  ('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
                                  ('ALIGN',(2,1),(-1,-1),'RIGHT')]))
        story.append(table);story.append(Spacer(1,9))
    para('*Precision and F1 are macro-averaged. Fusion adds patient features to the stated text dimension. PCA retains 89.71% of training embedding variance.')
    para('Interpretation and Section 3.6', 'Heading2')
    para('These are single-split synthetic-data results, not clinical validation or an evaluation of live translation. Repeated complaint phrases may cross the random row split. Section 3.6 remains pending: the article has only a heading and no published comparator list. No literature scores have been invented.')
    for feature, title in [('text_only','Text-only performance'),('fusion','Performance with patient features')]:
        story.append(PageBreak());para(title,'Title')
        for metric in ['accuracy','precision_macro']:
            story.append(Image(str(RESULTS/f'sbert_{feature}/{metric}_comparison.png'),width=490,height=272))
            story.append(Spacer(1,16))
    for model in ['LogisticRegression','HistGradientBoosting','RandomForest']:
        story.append(PageBreak());para(names[model] + ': confusion matrices','Title')
        para('Rows show true triage levels; columns show predictions. Each matrix contains the same 451 test records. Top row: text only. Bottom row: text plus patient features. Left: original 768-D; right: PCA-64.')
        grid=[]
        for feature in ['text_only','fusion']:
            grid.append([Image(str(RESULTS/f'sbert_{feature}/{rep}_{model}_confusion.png'),width=244,height=203.3)
                         for rep in ['full_768','pca_64']])
        story.append(Table(grid,colWidths=[250,250],style=[('VALIGN',(0,0),(-1,-1),'TOP')]))
    def footer(canvas, doc):
        canvas.setFont('Vera',8);canvas.drawString(42,25,'ML_Predictor | SBERT research comparison')
        canvas.drawRightString(A4[0]-42,25,str(doc.page))
    SimpleDocTemplate(str(OUTPUT),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUTPUT)


if __name__ == '__main__':
    main()
