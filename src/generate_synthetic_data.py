"""Generate deterministic synthetic data for the educational FL demo."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(20261029)
rows=[]
for client in range(5):
    n=160
    shift=(client-2)*0.22
    X=rng.normal(size=(n,4))
    X[:,0]+=shift
    score=1.1*X[:,0]-0.85*X[:,1]+0.55*X[:,2]+0.25*X[:,3]+shift*0.3
    prob=1/(1+np.exp(-score))
    y=(rng.random(n)<prob).astype(int)
    for xi, label in zip(X,y):
        rows.append([client,*xi.tolist(),int(label)])
df=pd.DataFrame(rows,columns=['client_id','x1','x2','x3','x4','label'])
df.to_csv(ROOT/'data'/'synthetic_clients.csv',index=False)
print(f'Wrote {len(df)} synthetic rows across {df.client_id.nunique()} clients.')
