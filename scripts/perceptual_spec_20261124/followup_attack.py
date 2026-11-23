"""One adaptive weaker-penalty follow-up; all models re-evaluated together.
This reuses the same benchmark after looking at phase-one outcomes, so it is
exploratory and not a new independent validation set.
"""
import json
import attack_compare as A
from experiment_io import OUT,save

if __name__=='__main__':
    first=json.loads((OUT/'attacks/training_summary.json').read_text())
    A.TARGET=OUT/'attacks_followup'
    A.CANDIDATES=('mask_eps2_iter10_s05',)
    A.EXTRA_REFERENCES=[r for r in first['training'] if r['condition'] in ('eps1_iter5','mask_eps2_iter10_s2')]
    assert len(A.EXTRA_REFERENCES)==2
    A.SELECTION_NOTE='Adaptive follow-up: reduced-radius and strong-penalty models showed weak/absent defense against Clean. Test the pre-defined weaker strength0.5 only; retain prior model files and re-evaluate all conditions on reused 30 prompts. Not independent validation.'
    try:A.main()
    except BaseException as e:
        if A.TARGET.exists():save(A.TARGET/'failure.json',dict(type=type(e).__name__,message=str(e)))
        raise
