"""Qin2019-style simultaneous masking model and waveform feasibility tools.

All maskers are treated as tonal (conservative approximation). PSD is normalized
to a 96 dB CLEAN-utterance spectral maximum, not a measured dB SPL. ATH is valid
20 Hz..20 kHz; outside this range an additional -100 dBFS/bin engineering cap
is imposed. No temporal masking, no individual hearing/device calibration.
"""
from dataclasses import dataclass
import numpy as np
import torch

REFERENCE_DB=96.
SAMPLE_RATE=44100
SIZES=(2048,1024)
EPSILON_CODES=257


def bark(frequency):
    f=np.asarray(frequency,dtype=np.float64)
    return 13*np.arctan(.00076*f)+3.5*np.arctan((f/7500.)**2)


def ath(frequency):
    f=np.clip(np.asarray(frequency,dtype=np.float64),20.,20000.)/1000
    return 3.64*f**(-.8)-6.5*np.exp(-.6*(f-3.3)**2)+.001*f**4


def stft_numpy(signal,n_fft):
    """Independent audit transform: periodic Hann, zero-centered padding, hop=N/4."""
    x=np.asarray(signal,dtype=np.float64)
    padded=np.pad(x,(n_fft//2,n_fft//2),mode='constant')
    frames=np.lib.stride_tricks.sliding_window_view(padded,n_fft)[::n_fft//4]
    window=np.hanning(n_fft+1)[:-1]
    return np.fft.rfft(frames*window,axis=-1).T


def retained_maskers(level_db,frequency):
    peaks=np.flatnonzero((level_db[1:-1]>level_db[:-2])&(level_db[1:-1]>=level_db[2:]))+1
    peaks=peaks[(frequency[peaks]>=20)&(frequency[peaks]<=20000)&(level_db[peaks]>=ath(frequency[peaks]))]
    if not len(peaks):return np.array([],dtype=int),np.array([],dtype=np.float64)
    smooth=10*np.log10(np.maximum(10**(level_db[peaks-1]/10)+10**(level_db[peaks]/10)+10**(level_db[peaks+1]/10),1e-30))
    order=np.argsort(-smooth,kind='stable');selected=[];levels=[];b=bark(frequency)
    for i in order:
        k=int(peaks[i])
        if all(abs(b[k]-b[j])>=.5 for j in selected):selected.append(k);levels.append(float(smooth[i]))
    return np.asarray(selected,dtype=int),np.asarray(levels,dtype=np.float64)


@dataclass
class MaskingScale:
    n_fft:int
    length:int
    clean_peak_power:float
    raw_threshold:np.ndarray
    masker_counts:np.ndarray
    window:torch.Tensor

    def limit(self,margin_db,extra_reserve_db=0.):
        raw=self.raw_threshold*10**(-(margin_db+extra_reserve_db)/10)
        f=np.fft.rfftfreq(self.n_fft,1/SAMPLE_RATE)
        raw=raw.copy()
        # Outside the physiological ATH domain: an explicit extra engineering
        # constraint, not an extrapolated claim about human hearing.
        raw[(f<20)|(f>20000)]=np.minimum(raw[(f<20)|(f>20000)],1e-10*10**(-extra_reserve_db/10))
        return raw

    def fft_torch(self,delta):
        return torch.stft(delta,n_fft=self.n_fft,hop_length=self.n_fft//4,
            window=self.window,center=True,pad_mode='constant',return_complex=True)

    def project(self,delta,limit):
        value=torch.as_tensor(np.asarray(delta),dtype=torch.float64)
        spectrum=self.fft_torch(value)
        cap=torch.as_tensor(np.sqrt(limit)*self.n_fft,dtype=torch.float64)
        spectrum=spectrum*torch.clamp(cap/spectrum.abs().clamp_min(1e-30),max=1.)
        restored=torch.istft(spectrum,n_fft=self.n_fft,hop_length=self.n_fft//4,
            window=self.window,center=True,length=self.length)
        return restored.numpy()


def build_scale(clean,n_fft):
    x=np.asarray(clean,dtype=np.float64);s=stft_numpy(x,n_fft);p=np.abs(s/n_fft)**2
    peak=max(float(p.max()),1e-30)
    offset=REFERENCE_DB-10*np.log10(peak)
    levels=10*np.log10(np.maximum(p,1e-30))+offset
    f=np.fft.rfftfreq(n_fft,1/SAMPLE_RATE);b=bark(f)
    thresholds=np.empty_like(p);counts=[]
    for t in range(p.shape[1]):
        indices,masker_levels=retained_maskers(levels[:,t],f);counts.append(len(indices))
        total=10**(ath(f)/10)
        if len(indices):
            distance=b[:,None]-b[indices][None,:]
            upper=-27+.37*np.maximum(masker_levels-40,0)
            spreading=np.where(distance<=0,27*distance,upper[None,:]*distance)
            individual=masker_levels[None,:]-6.025-.275*b[indices][None,:]+spreading
            total+=np.sum(10**(individual/10),axis=1)
        thresholds[:,t]=total*10**(-offset/10)
    return MaskingScale(n_fft,len(x),peak,thresholds,np.asarray(counts),torch.hann_window(n_fft,periodic=True,dtype=torch.float64))


def build_model(clean):return [build_scale(clean,n) for n in SIZES]


def audit(delta,models,margin_db):
    """Always use independent NumPy FFT and realized PCM residual for final audit."""
    result=[]
    for scale in models:
        power=np.abs(stft_numpy(delta,scale.n_fft)/scale.n_fft)**2
        limit=scale.limit(margin_db)
        ratio=power/np.maximum(limit,1e-300)
        excess=10*np.log10(np.maximum(ratio,1e-300))
        violation=ratio>1+1e-6 # 0.0000044 dB numerical tolerance, not a masking relaxation
        positive=excess[violation]
        result.append(dict(n_fft=scale.n_fft,margin_db=margin_db,bins=int(ratio.size),
            violating_bins=int(violation.sum()),violation_fraction=float(violation.mean()),
            max_excess_db=max(float(excess.max()),0.),
            p95_excess_db=max(float(np.quantile(excess,.95)),0.),
            p95_positive_excess_db=float(np.quantile(positive,.95)) if len(positive) else 0.,
            pass_constraint=not bool(violation.any())))
    return result


def project_intersection(delta,models,margin_db,radius_codes,cycles=16,reserve_db=1.):
    """Alternating waveform-box / STFT feasibility projections, not an output filter."""
    d=np.asarray(delta,dtype=np.float64).copy();bound=np.asarray(radius_codes)/32768
    limits=[m.limit(margin_db,reserve_db) for m in models]
    for _ in range(cycles):
        for scale,limit in zip(models,limits):
            d=scale.project(d,limit)
            d=np.clip(d,-bound,bound)
    return d


def pcm_feasible(delta,models,margin_db,radius_codes,repair_cycles=24):
    """Quantize, repair and audit. Residual-gain fallback is explicitly returned.

    Never relax mask/epsilon. A clean fallback is allowed as a failed-protection
    diagnostic, never as success. No scale factor touches the original speech.
    """
    radius=np.asarray(radius_codes,dtype=np.int32)
    d=project_intersection(delta,models,margin_db,radius,cycles=repair_cycles,reserve_db=1.)
    attempts=[]
    # Repair the actual rounding error before using an explicit residual-only
    # attenuation fallback; a feasible saved grid is more important than pretending
    # a float waveform's feasibility survives quantization.
    for i in range(6):
        q=np.clip(np.rint(d*32768),-radius,radius).astype(np.int32)
        checks=audit(q/32768.,models,margin_db)
        attempts.append(dict(repair=i,max_excess_db=max(r['max_excess_db'] for r in checks)))
        if all(r['pass_constraint'] for r in checks):
            return q,dict(repairs=attempts,residual_gain_fallback=1.,clean_fallback=not bool(np.any(q)),audit=checks)
        # Extra interior reserve grows only tighter. It never enlarges the target.
        d=project_intersection(q/32768.,models,margin_db,radius,cycles=8,reserve_db=2.+i)
    for gain in (.9,.8,.7,.6,.5,.4,.3,.2,.1,.05,.02,.01,0.):
        q=np.clip(np.rint(d*gain*32768),-radius,radius).astype(np.int32)
        checks=audit(q/32768.,models,margin_db)
        if all(r['pass_constraint'] for r in checks):
            return q,dict(repairs=attempts,residual_gain_fallback=gain,clean_fallback=not bool(np.any(q)),audit=checks)
    raise AssertionError('Even exact zero residual was not feasible')
