"""Reproducible triage-label projections and repeated descriptive geometry."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/images/SapBERT_Paper_Figures'
def run():
 OUT.mkdir(parents=True,exist_ok=True)
 src=ROOT/'output/results/triage_four_level'
 df=pd.read_csv(src/'dataset_with_splits.csv');dev=df[df.partition.eq('development')].drop_duplicates('group')
 embedding=np.load(src/'emb_sapbert_pair.npy',mmap_mode='r');pca=joblib.load(ROOT/'triage_model_sapbert/pca.pkl')
 manifest=json.loads((ROOT/'triage_model_sapbert/model_manifest.json').read_text())
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha(ROOT/'triage_model_sapbert/pca.pkl')==manifest['artifact_sha256']['pca.pkl']
 diagnostic=json.loads((ROOT/'output/results/triage_pair_embedding_diagnostics/diagnostics.json').read_text())
 assert sha(src/'emb_sapbert_pair.npy')==diagnostic['embedding_sha256']
 def sample(seed):
  rng=np.random.default_rng(seed)
  ids=np.concatenate([rng.choice(dev[dev.Labels.eq(k)].row_id.to_numpy(),50,replace=False) for k in range(4)])
  return ids,df.set_index('row_id').loc[ids,'Labels'].to_numpy()
 colors=['#c83f45','#e9a51d','#2585ab','#4f9d69'];names=['0 Emergency','1 Urgent','2 Standard','3 Non-urgent']
 def panel(ax,seed):
  ids,y=sample(seed);xy=pca.transform(embedding[ids])
  for k in range(4):ax.scatter(xy[y==k,0],xy[y==k,1],s=17,alpha=.65,c=colors[k],label=names[k])
  ax.set_title(f'Seed {seed} | 200 development groups',fontsize=11)
  ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)');ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)');ax.grid(alpha=.15)
 seeds=[42,99,404,777];fig,axes=plt.subplots(2,2,figsize=(11,8.4),sharex=True,sharey=True)
 for ax,seed in zip(axes.flat,seeds):panel(ax,seed)
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False)
 fig.suptitle('Current paired-text SapBERT embeddings by triage level\nFour balanced samples; the same fitted PCA in every panel',fontsize=13)
 fig.tight_layout(rect=[0,.05,1,.93])
 for ext in ['png','pdf']:fig.savefig(OUT/f'Figure_05_SapBERT_Triage_Embeddings_Four_Samples.{ext}',dpi=300)
 plt.close(fig)
 for seed in seeds:
  fig,ax=plt.subplots(figsize=(7,5.5));panel(ax,seed);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/f'Figure_05_SapBERT_Triage_Embedding_Seed_{seed}.png',dpi=300);plt.close(fig)
 records=[]
 for seed in range(20261004,20261024):
  ids,y=sample(seed);raw=np.asarray(embedding[ids]);proj=pca.transform(raw)
  upper=np.triu(np.ones((len(y),len(y)),bool),1);same=y[:,None]==y[None,:]
  for label,x in [('768-D SapBERT',raw),('64-D PCA',proj),('2-D PCA display',proj[:,:2])]:
   unit=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);dist=np.clip(1-unit@unit.T,0,2);np.fill_diagonal(dist,0)
   intra=float(dist[upper&same].mean());inter=float(dist[upper&~same].mean())
   records.append(dict(seed=seed,space=label,intra_cosine_distance=intra,inter_cosine_distance=inter,difference=inter-intra,silhouette=float(silhouette_score(dist,y,metric='precomputed'))))
 pd.DataFrame(records).to_csv(OUT/'Embedding_Geometry_20_Samples.csv',index=False)
 summary=[]
 for space,g in pd.DataFrame(records).groupby('space',sort=False):
  row={'space':space}
  for key in ['intra_cosine_distance','inter_cosine_distance','difference','silhouette']:row[key]={'mean':float(g[key].mean()),'sd':float(g[key].std(ddof=1))}
  summary.append(row)
 (OUT/'Embedding_Geometry_Summary.json').write_text(json.dumps(summary,indent=2))
 (OUT/'Embedding_Protocol.json').write_text(json.dumps(dict(figure_seeds=seeds,summary_seeds=list(range(20261004,20261024)),rows_per_sample=200,groups_per_sample=200,rows_per_level=50,pca_refitted=False,test_rows_used=False,labels='supplied triage levels, not complaint categories',samples_may_overlap=True,dataset_sha256=sha(src/'dataset_with_splits.csv'),embedding_sha256=sha(src/'emb_sapbert_pair.npy'),pca_sha256=sha(ROOT/'triage_model_sapbert/pca.pkl'),interpretation='20 reproducible resamples of one development dataset, not independent datasets or encoder training runs. No ANOSIM, permutation p-values or Cohen d claimed.'),indent=2))
 print('Four-panel figure and 20 descriptive resamples generated')
if __name__=='__main__':run()
