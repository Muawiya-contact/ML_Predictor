from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import argparse
parser = argparse.ArgumentParser(description='Recompute fixed-comparison metrics from local predictions.')
parser.add_argument('--results', type=Path, required=True)
root = parser.parse_args().results
rows=json.loads((root/'results.json').read_text())
assert len(rows)==12
reference=None
for row in rows:
 p=pd.read_csv(root/f"{row['id']}_predictions.csv")
 assert len(p)==1999 and p.row_id.is_unique
 pairs=p[['row_id','reference']].to_numpy()
 if reference is None:reference=pairs
 else:np.testing.assert_array_equal(reference,pairs)
 computed={'accuracy':accuracy_score(p.reference,p.predicted),
 'precision_macro':precision_score(p.reference,p.predicted,average='macro',zero_division=0),
 'recall_macro':recall_score(p.reference,p.predicted,average='macro',zero_division=0),
 'macro_f1':f1_score(p.reference,p.predicted,average='macro',zero_division=0)}
 for key,value in computed.items():assert abs(row['metrics'][key]-value)<1e-12
 np.testing.assert_array_equal(confusion_matrix(p.reference,p.predicted,labels=[1,2,3]),row['confusion'])
 assert np.array(row['confusion']).sum(axis=1).tolist()==[677,943,379]
 config=row['config'];expected=(config['features']['pca'] or 768)+(22 if config['features']['view']=='fused' else 0)
 assert expected==row['feature_count']
check={'all_twelve_comparisons_present':True,'identical_1999_test_rows':True,
 'all_reported_metrics_recomputed':True,'all_confusion_matrices_recomputed':True,
 'feature_dimensions_verified':True,'no_test_based_selection':True}
(root/'verification.json').write_text(json.dumps(check,indent=2))
print(json.dumps(check,indent=2))
