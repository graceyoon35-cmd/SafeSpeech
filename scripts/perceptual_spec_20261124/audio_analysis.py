"""Aligned, unnormalized WAV metrics and absolute residual band powers."""
import numpy as np
import soundfile as sf
from pystoi import stoi
from experiment_io import sha

BANDS=[(0,200),(200,1000),(1000,4000),(4000,8000),(8000,16000),(16000,22051)]

def db(x):return float(10*np.log10(max(float(x),1e-30)))

def analyze(original,candidate):
    x,sr=sf.read(original,dtype='float64');y,yr=sf.read(candidate,dtype='float64');info=sf.info(candidate)
    assert sr==yr==44100 and x.shape==y.shape and x.ndim==y.ndim==1
    assert np.isfinite(y).all() and info.subtype=='PCM_16'
    delta=y-x;power=float(np.mean(delta**2));N=len(delta)
    bins=np.abs(np.fft.rfft(delta))**2/N**2
    factors=np.full(len(bins),2.);factors[0]=1
    if N%2==0:factors[-1]=1
    bins*=factors;freq=np.fft.rfftfreq(N,1/sr)
    assert abs(bins.sum()-power)<1e-12
    n=round(.02*sr);m=len(x)//n
    frame_clean=np.mean(x[:m*n].reshape(m,n)**2,axis=1)
    frame_noise=np.mean(delta[:m*n].reshape(m,n)**2,axis=1)
    quiet=frame_clean<=np.quantile(frame_clean,.2)
    row=dict(path=str(candidate),sha256=sha(candidate),sample_rate=sr,bit_depth=16,channels=1,samples=N,
        peak=float(np.max(np.abs(y))),rms=float(np.sqrt(np.mean(y*y))),delta_rms=float(np.sqrt(power)),delta_dbfs=db(power),
        snr_db=db(np.mean(x*x)/power) if power else None,stoi=float(stoi(x,y,sr,extended=False)),
        estoi=float(stoi(x,y,sr,extended=True)),linf=float(np.max(np.abs(delta))),
        clipping=int(np.count_nonzero((y<=-1)|(y>=32767/32768))),quiet20_delta_dbfs=db(np.mean(frame_noise[quiet])),
        quiet20_continuity_fraction_above_minus55dbfs=float(np.mean(frame_noise[quiet]>10**(-55/10))))
    for low,high in BANDS:
        p=float(bins[(freq>=low)&(freq<high)].sum())
        row[f'band_{low}_{high}_power']=p
        row[f'band_{low}_{high}_dbfs']=db(p)
        row[f'band_{low}_{high}_fraction']=p/power if power else 0.
    return row
