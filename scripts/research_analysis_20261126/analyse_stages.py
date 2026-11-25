"""Summarize paired pipeline stages; fixes comparison scope before execution."""
import csv
import json
from pathlib import Path
import numpy as np

from research_bootstrap import setup
CODE_ROOT,ROOT,LEGACY=setup()
INPUT=ROOT/'runs/psychoacoustic_spec_refinement_20261126'
OUT=Path(__file__).resolve().parent

def main():
    with (INPUT/'stage_diagnosis/stage_measurements.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    order=('unconstrained_proposal','continuous_projection_10','prequant_projection_8','first_pcm_round','post_repairs_float','final_pcm_repaired')
    groups={}
    for r in rows:groups.setdefault((r['id'],int(r['iteration'])),{})[r['stage']]=r
    assert len(groups)==50 and all(set(g)==set(order) for g in groups.values())
    summary=[]
    for index,stage in enumerate(order):
        allrows=[g[stage] for g in groups.values()]
        first=[g[stage] for (uid,it),g in groups.items() if it==1]
        a=dict(stage=stage,proposals=len(allrows),mean_loss_SPEC=float(np.mean([float(r['loss_SPEC']) for r in allrows])),
            mean_improvement_from_current=float(np.mean([float(r['loss_current'])-float(r['loss_SPEC']) for r in allrows])),
            first_update_mean_loss_SPEC=float(np.mean([float(r['loss_SPEC']) for r in first])),
            first_update_mean_improvement_from_clean=float(np.mean([float(r['loss_clean'])-float(r['loss_SPEC']) for r in first])))
        if index:
            diffs=[float(g[stage]['loss_SPEC'])-float(g[order[index-1]]['loss_SPEC']) for g in groups.values()]
            a['mean_loss_increase_from_previous_stage']=float(np.mean(diffs))
            a['stage_weakens_objective_proposals']=sum(d>0 for d in diffs)
        summary.append(a)
    record=dict(stage_summary=summary,scope='50 runtime-replayed proposals, paired by utterance/iteration; first-update values also supplied.',
        baseline_replay_is_not_historical_pcm_exact=True,
        interpretation='Positive stage loss increase weakens SPEC surrogate, not a measured cloning defense effect.',
        variants_locked=['anchor_backtrack','local_caps_anchor'],max_variants=2,no_followup_after_results=True,
        comparison='Immutable historical3dB WAV and objectives, plus runtime baseline replay sensitivity; same5 IDs,seed52,beta10,10steps,epsilon257codes,unchanged3dB thresholds/radii')
    with (OUT/'stage_summary.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    with (OUT/'stage_summary.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in summary for k in r)));w.writeheader();w.writerows(summary)
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':
    from runner import cli
    cli('analyse_stages')
