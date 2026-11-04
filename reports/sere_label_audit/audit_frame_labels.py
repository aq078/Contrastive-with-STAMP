import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from stamp.sere.data import read_table, source_paths, contiguous_runs
root=Path('/zfsauton/scratch/anqix/stamp/dataset/SERE_dataset_SHAREABLE_skeletons')
manifest=json.loads(Path('/zfsauton/scratch/anqix/stamp/experiments/sere_temporal_5fold_v2/without_reduction/manifest.json').read_text())
selected=[2,3,5,6,7,11,15,16,17,18]
records=[]
columns=[f'{j}{a}' for j in range(33) for a in 'xyz']
for ex in manifest['sources']:
 paths=source_paths(root,ex)
 world,labels=read_table(paths['world']).drop_duplicates(),read_table(paths['frames']).drop_duplicates()
 for table in (world,labels):
  if 'affected' not in table:table['affected']=-1
  table['affected']=table['affected'].fillna(-1).astype(int)
  table.set_index(['pid','affected','frame'],inplace=True)
 for t in manifest['trials']:
  if t['exercise']!=ex:continue
  for portion,lo,hi in [('development',t['start'],t['cut']),('test',t['cut'],t['stop'])]:
   frames=np.arange(lo,hi)
   ix=pd.MultiIndex.from_arrays([np.full(len(frames),t['pid']),np.full(len(frames),t['affected']),frames])
   y=pd.to_numeric(labels.reindex(ix)['comp'],errors='coerce').to_numpy(float)
   x=world.reindex(ix)[columns].to_numpy(float)
   valid=np.isfinite(x).all(1)&~np.isin(frames,t['excluded_frames'])
   known=valid&np.isfinite(y)
   runs=[]
   for segment in contiguous_runs(frames[known]):
    values=(y[segment-lo]>=.5).astype(int)
    starts=np.r_[0,np.flatnonzero(np.diff(values)!=0)+1];ends=np.r_[starts[1:],len(values)]
    runs.extend([dict(label=int(values[a]),start=int(segment[a]),stop=int(segment[b-1])+1,length=int(b-a)) for a,b in zip(starts,ends)])
   records.append(dict(trial=t['trial_id'],pid=t['pid'],exercise=ex,portion=portion,start=lo,stop=hi,
                       valid_frames=int(known.sum()),positive_frames=int((y[known]>=.5).sum()),
                       unknown_labels=int((valid&~np.isfinite(y)).sum()),excluded_or_missing_skeleton=int((~valid).sum()),runs=runs))
 print('read',ex,flush=True)
def summarize(rows):
 runs=[r for row in rows for r in row['runs']]
 p=[r['length'] for r in runs if r['label']==1];n=[r['length'] for r in runs if r['label']==0]
 total=sum(r['valid_frames'] for r in rows);positive=sum(r['positive_frames'] for r in rows)
 def stats(v):return dict(count=len(v),median=float(np.median(v)) if v else None,p90=float(np.percentile(v,90)) if v else None,max=max(v) if v else None,total=sum(v))
 return dict(frames=total,positive_frames=positive,positive_pct=100*positive/total,
  positive_runs=stats(p),negative_runs=stats(n),positive_singletons=sum(v==1 for v in p),
  positive_frames_in_runs_ge30_pct=100*sum(v for v in p if v>=30)/positive if positive else None,
  all_positive_tails=sum(r['valid_frames']>0 and r['positive_frames']==r['valid_frames'] for r in rows),
  all_negative_tails=sum(r['valid_frames']>0 and r['positive_frames']==0 for r in rows),
  mixed_tails=sum(0<r['positive_frames']<r['valid_frames'] for r in rows),
  tails=len(rows),unknown_labels=sum(r['unknown_labels'] for r in rows))
report=dict(all_positive_window_patients_L64_to128=selected,records=records,patients={})
for p in manifest['patients']:
 report['patients'][str(p)]={part:summarize([r for r in records if r['pid']==p and r['portion']==part]) for part in ['development','test']}
for key,pids in [('selected10',selected),('other8',[p for p in manifest['patients'] if p not in selected])]:
 report[key]={part:summarize([r for r in records if r['pid'] in pids and r['portion']==part]) for part in ['development','test']}
 print(key,json.dumps(report[key]),flush=True)
for p in selected:print('patient',p,json.dumps(report['patients'][str(p)]['test']),flush=True)
Path('reports/sere_label_audit/frame_run_audit.json').write_text(json.dumps(report,indent=2)+'\n')
