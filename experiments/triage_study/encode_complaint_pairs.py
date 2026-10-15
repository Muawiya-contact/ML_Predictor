"""Frozen SapBERT encoding of concepts plus complete original complaints."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from src.sapbert_serving import SapBERTEncoder
from cache_integrity import file_sha256

if __name__=='__main__':
    root=Path('output/results/triage_four_level')
    out=root/'emb_sapbert_pair.npy'
    if out.exists():raise ValueError('Refusing to replace existing paired embeddings')
    source_hash=file_sha256(root/'dataset_with_splits.csv')
    df=pd.read_csv(root/'dataset_with_splits.csv')
    texts=(df.Clinical_Concept.fillna('')+' [SEP] '+df.chief_complaint.fillna('')).tolist()
    manifest=json.loads(Path('triage_model_sapbert/model_manifest.json').read_text())
    manifest['encoder_settings']['max_token_length']=128
    encoder=SapBERTEncoder(manifest)
    values=[];start=time.monotonic()
    for i in range(0,len(texts),32):
        values.append(encoder.encode(texts[i:i+32],batch_size=16))
        if i%320==0:print(f'{min(i+32,len(texts))}/{len(texts)} rows, {time.monotonic()-start:.1f}s',flush=True)
    array=np.vstack(values);assert array.shape==(len(df),768) and np.isfinite(array).all()
    assert file_sha256(root/'dataset_with_splits.csv')==source_hash, 'Source changed during encoding'
    np.save(out,array)
    metadata=dict(shape=list(array.shape),array_sha256=file_sha256(out),source_data_sha256=source_hash,texts_sha256=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest(),input='Clinical_Concept + [SEP] + chief_complaint',encoder_settings=manifest['encoder_settings'],embedding_model=manifest['embedding_model'],encoder_finetuned=False,labels_used=False)
    (root/'emb_sapbert_pair.json').write_text(json.dumps(metadata,indent=2));print('COMPLETE',flush=True)
