"""Merge per-workload/seed outputs and produce V2.0 follow-up summaries/figures."""
from pathlib import Path
import json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'v2_followup'
FIG=OUT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
files=sorted(glob.glob(str(OUT/'run_level_*_seed*.csv')))
if not files: raise SystemExit('No per-seed run_level files found')
df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
df.to_csv(OUT/'run_level_all.csv',index=False)
metrics=['goodput_req_s','throughput_req_s','throughput_tok_s','ttft_p50_ms','ttft_p99_ms','tpot_reqp99_p50_ms','tpot_reqp99_p99_ms','slo_attainment','reject_rate','avg_kv_frac','avg_util','preemptions','iterations','tilt_ttft_frac','tilt_tpot_frac','tilt_balanced_frac','risk_ttft_mean','risk_tpot_mean','chunk_target_mean','budget_target_mean','effective_budget_mean','effective_chunk_mean','rho_realized','rho_calibration_iters','n_requests']
metrics=[m for m in metrics if m in df.columns]
g=df.groupby(['workload','rho_target','policy'])[metrics].agg(['mean','std']).reset_index()
g.columns=['_'.join([str(x) for x in c if str(x)]) if isinstance(c,tuple) else c for c in g.columns]
g.to_csv(OUT/'summary_by_workload_load.csv',index=False)
macro=df.groupby(['workload','rho_target','policy'],as_index=False)[metrics].mean()
macro2=macro.groupby('policy',as_index=False)[metrics].mean().sort_values('goodput_req_s',ascending=False)
macro2.to_csv(OUT/'summary_policy_macro.csv',index=False)
cell=macro.copy()
cell['goodput_rank']=cell.groupby(['workload','rho_target'])['goodput_req_s'].rank(method='min',ascending=False)
cell.to_csv(OUT/'goodput_rank_by_cell.csv',index=False)
rho_audit=df.groupby(['workload','rho_target'],as_index=False).agg(
    rho_realized_mean=('rho_realized','mean'),rho_realized_sd=('rho_realized','std'),
    requests_mean=('n_requests','mean'),duration_mean=('duration_s','mean'),
    calibration_iters_mean=('rho_calibration_iters','mean'))
rho_audit.to_csv(OUT/'realized_rho_audit.csv',index=False)
base_rows=df[~df.policy.isin(['SLOA-V2','SLOA-V1-emulated'])]
best_idx=base_rows.groupby(['workload','rho_target','seed'])['goodput_req_s'].idxmax()
best_trace=base_rows.loc[best_idx,['workload','rho_target','seed','policy','goodput_req_s']].rename(columns={'policy':'best_non_sloa_policy','goodput_req_s':'best_non_sloa_goodput'})
v2_trace=df[df.policy=='SLOA-V2'][['workload','rho_target','seed','goodput_req_s']].rename(columns={'goodput_req_s':'sloa_v2_goodput'})
oracle=v2_trace.merge(best_trace,on=['workload','rho_target','seed'],how='inner')
oracle['delta_vs_oracle_best']=oracle.sloa_v2_goodput-oracle.best_non_sloa_goodput
oracle.to_csv(OUT/'sloa_vs_oracle_best_baseline.csv',index=False)
idx=['workload','rho_target','seed']
a=df[df.policy=='SLOA-V2'][idx+['goodput_req_s','ttft_p99_ms','tpot_reqp99_p99_ms']].rename(columns={'goodput_req_s':'gp_v2','ttft_p99_ms':'ttft_v2','tpot_reqp99_p99_ms':'tpot_v2'})
for target in ['SLOA-V1-emulated','SLOWEAVE','Fixed-64+SPF']:
    b=df[df.policy==target][idx+['goodput_req_s','ttft_p99_ms','tpot_reqp99_p99_ms']].rename(columns={'goodput_req_s':f'gp_{target}','ttft_p99_ms':f'ttft_{target}','tpot_reqp99_p99_ms':f'tpot_{target}'})
    m=a.merge(b,on=idx,how='inner'); m['delta_goodput']=m.gp_v2-m[f'gp_{target}']
    m.to_csv(OUT/f'paired_v2_vs_{target.replace("+","_").replace("-","_")}.csv',index=False)
rows=[]
for target in ['SLOA-V1-emulated','SLOWEAVE','Fixed-64+SPF']:
    path=OUT/f'paired_v2_vs_{target.replace("+","_").replace("-","_")}.csv'
    m=pd.read_csv(path); x=m.delta_goodput.dropna().to_numpy()
    if len(x):
        rng=np.random.default_rng(20261113)
        boot=np.array([rng.choice(x,size=len(x),replace=True).mean() for _ in range(5000)])
        rows.append({'comparison':f'SLOA-V2 minus {target}','n_paired':len(x),'mean_delta_req_s':x.mean(),'std_delta_req_s':x.std(ddof=1) if len(x)>1 else np.nan,'bootstrap_ci95_low':np.percentile(boot,2.5),'bootstrap_ci95_high':np.percentile(boot,97.5),'fraction_positive':float((x>0).mean())})
pd.DataFrame(rows).to_csv(OUT/'paired_delta_summary.csv',index=False)

def concat_seed_files(pattern):
    paths=sorted(glob.glob(str(OUT/pattern)))
    return pd.concat([pd.read_csv(x) for x in paths],ignore_index=True) if paths else pd.DataFrame()
ab=concat_seed_files('sloa_ablation_seed*.csv')
if not ab.empty:
    ab.groupby('variant').agg({'goodput_req_s':['mean','std'],'ttft_p99_ms':['mean','std'],'tpot_reqp99_p99_ms':['mean','std'],'slo_attainment':['mean','std']}).to_csv(OUT/'sloa_ablation_summary.csv')
ch=concat_seed_files('chunk_sweep_seed*.csv')
if not ch.empty:
    ch.groupby('chunk').agg({'goodput_req_s':['mean','std'],'ttft_p99_ms':['mean','std'],'tpot_reqp99_p99_ms':['mean','std']}).to_csv(OUT/'chunk_sweep_summary.csv')
pa=concat_seed_files('sloa_parameter_sweep_seed*.csv')
if not pa.empty:
    pa.groupby(['risk_threshold','gamma']).agg({'goodput_req_s':['mean','std'],'ttft_p99_ms':['mean','std'],'tpot_reqp99_p99_ms':['mean','std'],'slo_attainment':['mean','std']}).to_csv(OUT/'sloa_parameter_sweep_summary.csv')
    pmean=pa.groupby(['risk_threshold','gamma']).goodput_req_s.mean().unstack('gamma')
    plt.figure(figsize=(6.5,4.5)); im=plt.imshow(pmean.values,aspect='auto',cmap='viridis')
    plt.xticks(range(len(pmean.columns)),[str(x) for x in pmean.columns]); plt.yticks(range(len(pmean.index)),[str(x) for x in pmean.index])
    plt.xlabel('gamma'); plt.ylabel('risk threshold'); plt.title('SLOA parameter sweep: mean Goodput')
    plt.colorbar(im,label='Goodput (req/s)'); plt.tight_layout()
    plt.savefig(FIG/'sloa_parameter_sweep.png',dpi=160); plt.savefig(FIG/'sloa_parameter_sweep.svg'); plt.close()
if not ab.empty:
    abm=ab.groupby('variant',as_index=False).agg(goodput=('goodput_req_s','mean'),ttft_p99=('ttft_p99_ms','mean'),tpot_p99=('tpot_reqp99_p99_ms','mean'))
    plt.figure(figsize=(8,4.5)); plt.bar(abm.variant,abm.goodput); plt.xticks(rotation=25,ha='right')
    plt.ylabel('Goodput (req/s)'); plt.title('SLOA ablation on W3-Mixed, target rho=0.85'); plt.tight_layout()
    plt.savefig(FIG/'sloa_ablation.png',dpi=160); plt.savefig(FIG/'sloa_ablation.svg'); plt.close()
if not ch.empty:
    chm=ch.groupby('chunk',as_index=False).agg(goodput=('goodput_req_s','mean'),ttft_p99=('ttft_p99_ms','mean'),tpot_p99=('tpot_reqp99_p99_ms','mean'))
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(chm.chunk,chm.goodput,marker='o')
    ax.set_xlabel('Fixed prefill chunk'); ax.set_ylabel('Goodput (req/s)'); ax.set_xscale('log',base=2)
    ax.grid(True,alpha=.25); ax2=ax.twinx(); ax2.plot(chm.chunk,chm.tpot_p99,marker='s')
    ax2.set_ylabel('TPOT P99 (ms)'); ax.set_title('Fixed chunk sweep on W3-Mixed, target rho=0.85')
    fig.tight_layout(); fig.savefig(FIG/'chunk_sweep.png',dpi=160); fig.savefig(FIG/'chunk_sweep.svg'); plt.close(fig)

plt.figure(figsize=(9,5.5))
for policy in ['FCFS+Full','Fixed-64+SPF','Fixed-256+SPF','SLOWEAVE','SLAI','RAD','SLOA-V1-emulated','SLOA-V2']:
    sub=df[df.policy==policy].groupby('rho_target').goodput_req_s.mean().sort_index()
    if len(sub): plt.plot(sub.index,sub.values,marker='o',label=policy)
plt.xlabel('Target offered load rho'); plt.ylabel('Goodput (req/s)')
plt.title('V2.0 follow-up: Goodput vs offered load'); plt.grid(True,alpha=.25)
plt.legend(fontsize=7,ncol=2); plt.tight_layout()
plt.savefig(FIG/'goodput_vs_load.png',dpi=160); plt.savefig(FIG/'goodput_vs_load.svg'); plt.close()

base=macro[~macro.policy.isin(['SLOA-V2','SLOA-V1-emulated'])]
best=base.groupby(['workload','rho_target']).goodput_req_s.max().rename('best_baseline')
v2=macro[macro.policy=='SLOA-V2'].set_index(['workload','rho_target'])
rel=(v2.goodput_req_s/best).unstack('rho_target')
plt.figure(figsize=(8,4.5)); im=plt.imshow(rel.values,aspect='auto',vmin=0.5,vmax=1.5,cmap='coolwarm')
plt.xticks(range(len(rel.columns)),[str(x) for x in rel.columns]); plt.yticks(range(len(rel.index)),rel.index)
plt.xlabel('Target rho'); plt.title('SLOA V2 Goodput / best non-SLOA baseline')
plt.colorbar(im,label='Relative Goodput'); plt.tight_layout()
plt.savefig(FIG/'sloa_vs_best_baseline_heatmap.png',dpi=160); plt.savefig(FIG/'sloa_vs_best_baseline_heatmap.svg'); plt.close()

plt.figure(figsize=(8,5.5))
for _,r in macro2.iterrows():
    plt.scatter(r.ttft_p99_ms,r.tpot_reqp99_p99_ms,s=55)
    plt.annotate(r.policy,(r.ttft_p99_ms,r.tpot_reqp99_p99_ms),fontsize=7,xytext=(3,3),textcoords='offset points')
plt.xlabel('TTFT P99 (ms)'); plt.ylabel('Request-level TPOT P99 (ms)')
plt.title('Latency Pareto view'); plt.grid(True,alpha=.25); plt.tight_layout()
plt.savefig(FIG/'latency_pareto.png',dpi=160); plt.savefig(FIG/'latency_pareto.svg'); plt.close()

s=df[df.policy=='SLOA-V2'].groupby(['workload','rho_target'],as_index=False)[['tilt_ttft_frac','tilt_tpot_frac','tilt_balanced_frac','effective_budget_mean','effective_chunk_mean']].mean()
s.to_csv(OUT/'sloa_controller_summary.csv',index=False)
plt.figure(figsize=(8,4.5))
for col in ['tilt_ttft_frac','tilt_tpot_frac','tilt_balanced_frac']:
    plt.plot(sorted(s.rho_target.unique()),[s[s.rho_target==x][col].mean() for x in sorted(s.rho_target.unique())],marker='o',label=col)
plt.xlabel('Target rho'); plt.ylabel('Fraction of scheduler ticks'); plt.title('SLOA V2 tilt-state distribution')
plt.grid(True,alpha=.25); plt.legend(); plt.tight_layout()
plt.savefig(FIG/'sloa_tilt_distribution.png',dpi=160); plt.savefig(FIG/'sloa_tilt_distribution.svg'); plt.close()

meta={'n_runs':int(len(df)),'n_traces':int(df[['workload','rho_target','seed']].drop_duplicates().shape[0]),
      'workloads':sorted(df.workload.unique().tolist()),'policies':sorted(df.policy.unique().tolist()),
      'rho_target':sorted(df.rho_target.unique().tolist()),'seeds':sorted(int(x) for x in df.seed.unique()),
      'source':'V2.0 simulator; no real GPU measurement'}
(OUT/'finalization_meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'experiment_meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
print('runs',len(df),'traces',meta['n_traces'])
print(macro2[['policy','goodput_req_s','ttft_p99_ms','tpot_reqp99_p99_ms','slo_attainment']].round(4).to_string(index=False))
