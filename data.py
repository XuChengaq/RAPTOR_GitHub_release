from __future__ import annotations
from pathlib import Path
import sys
import json
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
AIRFRANS_VENDOR = ROOT.parent / 'AirfRANS_engineering_validation' / 'vendor'
if AIRFRANS_VENDOR.exists():
    sys.path.insert(0, str(AIRFRANS_VENDOR))
import airfrans as af

def _sample_simulation(arr,n,surface_fraction,seed,token_count,knn_k):
    rng=np.random.default_rng(seed); surface=np.flatnonzero(arr[:,-1]>0.5); volume=np.flatnonzero(arr[:,-1]<=0.5)
    ns=min(len(surface),int(n*surface_fraction)); nv=n-ns
    si=rng.choice(surface,ns,replace=len(surface)<ns) if ns else np.empty(0,int)
    vi=rng.choice(volume,nv,replace=len(volume)<nv); idx=np.concatenate([si,vi]); rng.shuffle(idx); sample=arr[idx].astype('float32')
    xy=sample[:,:2]; tree=cKDTree(xy); knn=tree.query(xy,k=min(knn_k+1,n))[1][:,1:].astype('int64')
    order=np.lexsort((xy[:,1],xy[:,0])); groups=np.empty(n,np.int64); groups[order]=np.minimum(token_count-1,np.arange(n)*token_count//n)
    return sample[:,:7],sample[:,7:11],sample[:,-1].astype('float32'),knn,groups

def build_cache(root,task,train_count,test_count,n,surface_fraction,token_count,knn_k,cache_dir,seed=2026,val_fraction=0.1):
    cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True); path=cache_dir/f'{task}_tr{train_count}_te{test_count}_n{n}.npz'
    if path.exists(): return path
    print(f'[data] loading official AirfRANS task={task}',flush=True)
    tr,_=af.dataset.load(root=str(root),task=task,train=True); te,_=af.dataset.load(root=str(root),task=task,train=False)
    rng=np.random.default_rng(seed)
    tr=[tr[i] for i in rng.permutation(len(tr))[:train_count]]
    te=[te[i] for i in rng.permutation(len(te))[:test_count]]
    val_count=max(1,min(len(tr)-1,int(round(len(tr)*val_fraction))))
    train_items,val_items=tr[:-val_count],tr[-val_count:]
    def process(items,offset):
        rows=[_sample_simulation(a,n,surface_fraction,offset+i,token_count,knn_k) for i,a in enumerate(items)]
        return [np.stack([r[j] for r in rows]) for j in range(5)]
    train=process(train_items,1000); val=process(val_items,5000); test=process(te,9000); xmean=train[0].reshape(-1,7).mean(0); xstd=train[0].reshape(-1,7).std(0)+1e-6; ymean=train[1].reshape(-1,4).mean(0); ystd=train[1].reshape(-1,4).std(0)+1e-6
    payload={}
    for split,arrays in [('train',train),('val',val),('test',test)]:
        for name,array in zip(('x','y','surface','knn','groups'),arrays): payload[f'{split}_{name}']=array
    np.savez_compressed(path,**payload,xmean=xmean,xstd=xstd,ymean=ymean,ystd=ystd)
    path.with_suffix('.json').write_text(json.dumps({'source':'AirfRANS official preprocessed dataset','task':task,'selection_seed':seed,'train_simulations':len(train_items),'validation_simulations':len(val_items),'test_simulations':len(te),'points_per_simulation':n},indent=2),encoding='utf-8')
    print(f'[data] cache saved {path}',flush=True); return path
