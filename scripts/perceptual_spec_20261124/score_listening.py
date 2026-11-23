"""Optional human ABX scoring; non-significance is never equivalence evidence."""
import argparse
import csv
import json
from pathlib import Path
from scipy.stats import binomtest
from experiment_io import OUT,save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--responses',type=Path,default=OUT/'quality/blind_listening/responses.csv')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    key=json.loads((OUT/'quality/blind_listening/answer_key_DO_NOT_OPEN_BEFORE_LISTENING.json').read_text())
    expected={(r['profile'],r['trial']):r['answer'] for r in key}
    with args.responses.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    groups={};seen=set()
    for r in rows:
        pair=(r['profile'],int(r['trial']))
        assert pair in expected and pair not in seen;seen.add(pair)
        answer=r['X_matches_A_or_B'].strip().upper()
        if not answer:continue
        if answer not in ('A','B'):raise ValueError('Use only A or B, leave unlistened trials blank')
        groups.setdefault(r['profile'],[]).append(int(answer==expected[pair]))
    results=[]
    for profile in sorted({r['profile'] for r in key}):
        answers=groups.get(profile,[]);n=len(answers);k=sum(answers)
        results.append(dict(profile=profile,completed_trials=n,correct=k,
            one_sided_chance_pvalue=float(binomtest(k,n,.5,alternative='greater').pvalue) if n else None,
            human_imperceptibility_established=False))
    save(args.output,dict(status='PENDING' if not groups else 'EXPLORATORY_SINGLE_LISTENER_RESULTS',profiles=results,
        interpretation='Above-chance performance may demonstrate a detectable difference in these trials. Failure to reject chance does not establish inaudibility/equivalence; no calibrated SPL, listener population, independent utterance, or seed generalization. No automatic protection recommendation.'))
    print(json.dumps(results))

if __name__=='__main__':main()
