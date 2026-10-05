"""Four predeclared 100-record semantic diagnostics; never select attractive runs.

Groups are explicit text-mention rules, not clinical categories. All embeddings
come from the currently selected paired-input SapBERT cache. No retraining.
"""
from pathlib import Path
import json,re,hashlib
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/results/complaint_semantic_audit'
RULES={'Arm mention':r'\barm\b','Back mention':r'\bback\b','Jaw mention':r'\bjaw\b','Palpitations mention':r'\bpalpitations?\b','Shoulder mention':r'\bshoulders?\b'}
SEEDS=[42,99,404,777]
COLORS=['#d94848','#3688b8','#43a047','#ef8c2f','#9764b5']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mention(text):
 found=[k for k,pattern in RULES.items() if re.search(pattern,str(text),re.I)]
 return found[0] if len(found)==1 else None

def run():
 OUT.mkdir(parents=True,exist_ok=True);figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
 original=ROOT/'output/results/triage_four_level';source=original/'dataset_with_splits.csv';arr=original/'emb_sapbert_pair.npy'
 manifest=json.loads((ROOT/'triage_model_sapbert/model_manifest.json').read_text());protocol=json.loads((ROOT/'output/results/triage_four_level_round7/protocol.json').read_text())
 selected=json.loads((ROOT/'output/results/triage_four_level_round7/selection.json').read_text())
 assert manifest['source_config']==selected['config'], 'Active model differs from the study selection'
 assert sha(source)==protocol['source_data_sha256'] and sha(arr)==protocol['additional_embedding_sha256']
 for f,h in manifest['artifact_sha256'].items():assert sha(ROOT/'triage_model_sapbert'/f)==h
 frame=pd.read_csv(source);dev=frame[frame.partition.eq('development')].copy();dev['mention_group']=dev.Clinical_Concept.map(mention)
 eligible=dev[dev.mention_group.notna()].drop_duplicates('group').copy()
 counts=eligible.mention_group.value_counts();assert all(counts.get(k,0)>=20 for k in RULES)
 plan=dict(seeds=SEEDS,per_run=100,per_mention_group=20,replacement_within_run=False,overlap_between_runs_allowed=True,rule_patterns=RULES,rule_source='Clinical_Concept',category_status='Automatically derived literal text mentions; not independently verified clinical categories or pain diagnoses.',excluded='Rows matching zero or multiple listed mentions; duplicate complaint groups within pool.',candidate_counts={k:int(counts[k]) for k in RULES},plot_selection='All four predeclared seeds retained regardless of geometry',projection_views=['Fitted serving PCA first two components','Label-blind diagnostic PCA fit once to all eligible development vectors'],trained_classifier_changed=False,dataset_sha256=sha(source),embedding_sha256=sha(arr),serving_pca_sha256=sha(ROOT/'triage_model_sapbert/pca.pkl'),permutations=499)
 (OUT/'protocol.json').write_text(json.dumps(plan,indent=2))
 embedding=np.load(arr,mmap_mode='r');pca=joblib.load(ROOT/'triage_model_sapbert/pca.pkl')
 ep=np.asarray(embedding[eligible.row_id],dtype=np.float64)
 diagnostic=PCA(n_components=2,svd_solver='full').fit(ep)
 samples=[];sample_ids=[]
 for seed in SEEDS:
  rng=np.random.default_rng(seed);indices=np.concatenate([rng.choice(eligible[eligible.mention_group.eq(k)].index.to_numpy(),20,replace=False) for k in RULES]);sample=eligible.loc[indices].copy();sample['seed']=seed
  assert len(sample)==100 and sample.group.is_unique and sample.partition.eq('development').all()
  samples.append(sample);sample_ids.append(dict(seed=seed,row_ids=sample.row_id.tolist()))
 # Text and per-record identifiers remain local, outside the shareable package.
 pd.concat(samples).to_csv(OUT/'private_sampled_records.csv',index=False)
 (OUT/'private_row_ids.json').write_text(json.dumps(sample_ids,indent=2))
 def plot_four(filename,projection,colour_by):
  fig,axes=plt.subplots(2,2,figsize=(11,8.3),sharex=True,sharey=True)
  if colour_by=='mention_group':categories=list(RULES);palette=COLORS;legend=categories
  else:categories=[0,1,2,3];palette=['#c83f45','#e9a51d','#2585ab','#4f9d69'];legend=['0 Emergency','1 Urgent','2 Standard','3 Non-urgent']
  for run,(ax,sample) in enumerate(zip(axes.flat,samples),1):
   coords=projection.transform(np.asarray(embedding[sample.row_id],dtype=np.float64))[:,:2]
   for cat,color,label in zip(categories,palette,legend):
    mask=sample[colour_by].eq(cat).to_numpy();ax.scatter(coords[mask,0],coords[mask,1],s=25,alpha=.8,color=color,label=label)
   ax.set_title(f'Run {run}: seed {SEEDS[run-1]} | 100 complaints',fontsize=11);ax.grid(alpha=.15)
   ax.set_xlabel(f'PC1 ({100*projection.explained_variance_ratio_[0]:.1f}% variance)');ax.set_ylabel(f'PC2 ({100*projection.explained_variance_ratio_[1]:.1f}% variance)')
  h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3 if colour_by=='mention_group' else 4,frameon=False)
  view='Serving PCA' if projection is pca else 'Diagnostic PCA fitted to the eligible development pool'
  fig.suptitle(('Rule-derived complaint mentions' if colour_by=='mention_group' else 'Same complaints coloured by supplied urgency')+'\n'+view,fontsize=14);fig.tight_layout(rect=[0,.07,1,.92])
  for ext in ['png','pdf']:fig.savefig(figdir/(filename+'.'+ext),dpi=300)
  plt.close(fig)
 plot_four('01_Complaint_Mentions_Serving_PCA_Four_Runs',pca,'mention_group')
 plot_four('02_Same_Complaints_Urgency_Serving_PCA_Four_Runs',pca,'Labels')
 plot_four('03_Complaint_Mentions_Diagnostic_PCA_Four_Runs',diagnostic,'mention_group')
 metrics=[]
 for seed,sample in zip(SEEDS,samples):
  raw=np.asarray(embedding[sample.row_id],dtype=np.float64);y=sample.mention_group.map({k:i for i,k in enumerate(RULES)}).to_numpy();urgent=sample.Labels.to_numpy()
  for space,x in [('768-D paired SapBERT',raw),('64-D serving PCA',pca.transform(raw)),('2-D serving PCA',pca.transform(raw)[:,:2]),('2-D diagnostic PCA',diagnostic.transform(raw))]:
   unit=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);distance=np.clip(1-unit@unit.T,0,2);np.fill_diagonal(distance,0);upper=np.triu_indices(len(y),1);pairs=distance[upper];same=y[upper[0]]==y[upper[1]]
   intra=float(pairs[same].mean());inter=float(pairs[~same].mean());ranks=rankdata(pairs,method='average');denom=len(y)*(len(y)-1)/4
   statistic=float((ranks[~same].mean()-ranks[same].mean())/denom);rng=np.random.default_rng(seed+10000);ge=0
   for _ in range(499):
    perm=rng.permutation(y);mask=perm[upper[0]]==perm[upper[1]];stat=(ranks[~mask].mean()-ranks[mask].mean())/denom;ge+=stat>=statistic
   near=distance.copy();np.fill_diagonal(near,np.inf);nn=np.argmin(near,axis=1)
   # Unsupervised clustering is scored separately; plotted colours never come from KMeans.
   cluster=KMeans(n_clusters=5,random_state=seed,n_init=20).fit_predict(unit)
   metrics.append(dict(seed=seed,space=space,intra_cosine_distance=intra,inter_cosine_distance=inter,difference=inter-intra,mention_silhouette=float(silhouette_score(distance,y,metric='precomputed')),urgency_silhouette=float(silhouette_score(distance,urgent,metric='precomputed')),anosim_R=statistic,anosim_p_unadjusted=(ge+1)/500,nearest_mention_match=float(np.mean(y[nn]==y)),kmeans_ARI=float(adjusted_rand_score(y,cluster))))
 results=pd.DataFrame(metrics)
 # Holm correction over all 16 reported ANOSIM tests.
 order=np.argsort(results.anosim_p_unadjusted.to_numpy());adjusted=np.empty(len(order));running=0
 for j,i in enumerate(order):running=max(running,(len(order)-j)*results.iloc[i].anosim_p_unadjusted);adjusted[i]=min(1,running)
 results['anosim_p_holm']=adjusted;results.to_csv(OUT/'metrics.csv',index=False)
 summary=results.groupby('space',sort=False).agg({k:['mean','std'] for k in ['mention_silhouette','urgency_silhouette','nearest_mention_match','kmeans_ARI','anosim_R']});summary.to_csv(OUT/'summary.csv')
 final=dict(status='passed',runs=4,records_per_run=100,distinct_records_across_runs=int(pd.concat(samples).row_id.nunique()),eligible_group_count=len(eligible),candidate_counts=plan['candidate_counts'],all_seeds_retained=True,diagnostic_explained_variance_ratio=diagnostic.explained_variance_ratio_.tolist(),serving_first_two_variance=float(pca.explained_variance_ratio_[:2].sum()),summary=results.groupby('space',sort=False).mean(numeric_only=True).drop(columns='seed').to_dict(orient='index'),limitations='Mention labels are derived from the same text encoded, so separation is descriptive lexical/semantic consistency, not an independent category benchmark or triage accuracy. Samples may overlap. Only development records used. Diagnostic PCA is not deployed.')
 (OUT/'verification.json').write_text(json.dumps(final,indent=2));print(json.dumps(final,indent=2))
 assert sha(source)==plan['dataset_sha256'] and sha(arr)==plan['embedding_sha256']
if __name__=='__main__':run()
