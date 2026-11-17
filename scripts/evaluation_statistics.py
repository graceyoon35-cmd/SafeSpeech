"""Prompt-level descriptive statistics and seeded bootstrap intervals."""
import numpy as np


def statistics(values,seed=52,repeats=5000):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Nonempty finite one-dimensional observations required')
    result={'mean':float(x.mean()),'n':len(x),'sample_variance':float(x.var(ddof=1)) if len(x)>1 else None}
    if len(x)>1:
        draws=np.random.default_rng(seed).choice(x,size=(repeats,len(x)),replace=True).mean(axis=1)
        result.update(ci95_low=float(np.quantile(draws,.025)),ci95_high=float(np.quantile(draws,.975)))
    else:
        result.update(ci95_low=None,ci95_high=None)
    return result
