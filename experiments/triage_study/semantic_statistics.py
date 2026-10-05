"""Fixed 20-resample semantic audit. No training or attractive-run selection.

Pair-distance effect sizes are descriptive. Permutations shuffle record labels,
not dependent pairs. Across-resample t/Wilcoxon inference is intentionally absent.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from complaint_semantic_audit import mention,RULES
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/results/semantic_statistics'
def run():
 OUT.mkdir(parents=True,exist_ok=True)
 base=ROOT/'output/results/triage_four_level'
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 old=json.loads((ROOT/'reports/complaint_semantic_audit/protocol.json').read_text())
 assert sha(base/'dataset_with_splits.csv')==old['dataset_sha256']
 assert sha(base/'emb_sapbert_pair.npy')==old['embedding_sha256']
 assert sha(ROOT/'triage_model_sapbert/pca.pkl')==old['serving_pca_sha256']
 f=pd.read_csv(base/'dataset_with_splits.csv');f=f[f.partition.eq('development')].copy()
 f['mention']=f.Clinical_Concept.map(mention);f=f[f.mention.notna()].drop_duplicates('group')
 e=np.load(base/'emb_sapbert_pair.npy',mmap_mode='r');pca=joblib.load(ROOT/'triage_model_sapbert/pca.pkl')
 diag=PCA(n_components=2,svd_solver='full').fit(np.asarray(e[f.row_id],float))
 plan=dict(seeds=list(range(20261005,20261025)),per_group=10,records_per_run=50,runs=20,permutations=4999,spaces=['768-D','64-D PCA','2-D diagnostic','2-D serving'],overlap_between_runs=True,independent_training_runs=False,groups='Literal mentions in Clinical_Concept; excludes zero/multiple matches',source=old,statistical_scope='Conditional association with text-derived groups; not independent semantic ground truth. Holm correction across 80 tests. No pooled t-test or Wilcoxon across overlapping resamples.')
 (OUT/'protocol.json').write_text(json.dumps(plan,indent=2));rows=[];ids=[]
 for run,seed in enumerate(plan['seeds'],1):
  rng=np.random.default_rng(seed);sample=pd.concat([g.iloc[rng.choice(len(g),10,False)] for cat in RULES for g in [f[f.mention.eq(cat)]]])
  assert sample.group.is_unique and len(sample)==50
  ids.append({'seed':seed,'row_ids':sample.row_id.tolist()})
  raw=np.asarray(e[sample.row_id],float);compact=pca.transform(raw);y=sample.mention.map({k:i for i,k in enumerate(RULES)}).to_numpy()
  u=np.triu_indices(50,1);same=y[u[0]]==y[u[1]]
  for space,x in [('768-D',raw),('64-D PCA',compact),('2-D diagnostic',diag.transform(raw)),('2-D serving',compact[:,:2])]:
   unit=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);dist=np.clip(1-unit@unit.T,0,2);np.fill_diagonal(dist,0);pair=dist[u]
   a,b=pair[same],pair[~same];diff=b.mean()-a.mean();pooled=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
   ranks=rankdata(pair);denom=50*49/4;stat=(ranks[~same].mean()-ranks[same].mean())/denom
   rngp=np.random.default_rng(seed+12345);exceed=0
   for start in range(0,4999,250):
    perm=np.array([rngp.permutation(y) for _ in range(min(250,4999-start))]);mask=perm[:,u[0]]==perm[:,u[1]]
    a_rank=(mask*ranks).sum(1)/same.sum();b_rank=((~mask)*ranks).sum(1)/(~same).sum()
    exceed+=np.sum((b_rank-a_rank)/denom>=stat-1e-12)
   rows.append(dict(seed=seed,space=space,intra=a.mean(),inter=b.mean(),difference=diff,cohen_d=diff/pooled,anosim_R=stat,silhouette=silhouette_score(dist,y,metric='precomputed'),p=(exceed+1)/5000))
  print(f'Completed semantic sample {run}/20',flush=True)
 out=pd.DataFrame(rows);order=np.argsort(out.p.to_numpy());adj=np.empty(80);running=0
 for j,i in enumerate(order):running=max(running,(80-j)*out.iloc[i].p);adj[i]=min(1,running)
 out['p_holm']=adj;out.to_csv(OUT/'per_run_metrics.csv',index=False)
 summary=[]
 for space in plan['spaces']:
  group=out[out.space.eq(space)];r={'space':space}
  for k in ['intra','inter','difference','cohen_d','anosim_R','silhouette']:r[k]=float(group[k].mean());r[k+'_sd']=float(group[k].std(ddof=1))
  r.update(positive_difference_runs=int((group.difference>0).sum()),raw_significant=int((group.p<.05).sum()),holm_significant=int((group.p_holm<.05).sum()),min_p=float(group.p.min()),max_p=float(group.p.max()),min_holm=float(group.p_holm.min()),max_holm=float(group.p_holm.max()))
  summary.append(r)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'private_samples.json').write_text(json.dumps(ids))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
