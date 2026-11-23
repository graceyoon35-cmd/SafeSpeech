"""Relative spectral masking surrogate, not an absolute human masking threshold.

Penalty is optimized jointly with SPEC; no output filtering or gain changes.
Original pretrained parameters, sign update, step and epsilon are retained.
Detached gradient balancing makes the penalty coefficient explicit and logged.
"""
import inspect
import torch
from torch.nn import functional as F

class RelativeSpectralPenalty:
    def __init__(self,clean,sr):
        self.n_fft=2048
        self.window=torch.hann_window(self.n_fft,device=clean.device,dtype=clean.dtype)
        self.sr=sr
        with torch.no_grad():
            power=self.spectrum(clean).abs().square()
            local=F.avg_pool2d(power[:,None],(3,3),stride=1,padding=1)[:,0]
            floor=power.amax().clamp_min(1e-12)*1e-6
            self.budget=(.02*local+floor).detach()
            freq=torch.fft.rfftfreq(self.n_fft,1/sr,device=clean.device)
            self.weight=torch.ones_like(freq)
            self.weight[freq<200]=2
            self.weight[freq>=4000]=4
            self.weight[freq>=8000]=8
            self.weight=self.weight[None,:,None]

    def spectrum(self,x):
        return torch.stft(x.reshape(1,-1),n_fft=self.n_fft,hop_length=512,
            win_length=self.n_fft,window=self.window,normalized=True,return_complex=True)

    def __call__(self,delta):
        ratio=self.spectrum(delta).abs().square()/self.budget
        # No calibrated SPL/ATH, tonal masker detection, Bark spreading, or temporal model.
        return (self.weight*torch.relu(ratio-1)).mean()

def balanced_loss(loss,protected,clean,context,strength,raw_grad):
    penalty=context(protected-clean)
    g0=raw_grad(loss,protected,retain_graph=True)[0]
    gp=raw_grad(penalty,protected,retain_graph=True)[0]
    assert torch.isfinite(penalty) and torch.isfinite(g0).all() and torch.isfinite(gp).all()
    scale=(strength*g0.abs().mean()/(gp.abs().mean()+1e-12)).detach() if torch.count_nonzero(gp) else penalty.new_zeros(())
    combined=loss+scale*penalty
    return combined,dict(loss_perceptual=float(penalty.detach()),perceptual_coefficient=float(scale),
                         perceptual_strength=float(strength))

def install(module,folder):
    """Inject only one objective hook into an in-memory copy of pinned perturb()."""
    from experiment_io import save,sha
    original=module.perturb
    source=inspect.getsource(original)
    anchor='        if args.mode == "SafeSpeech":'
    assert source.count(anchor)==1
    injection='''        if getattr(args, "perceptual_strength", 0.) > 0:
            loss, perceptual_items = _perceptual_hook(loss, p_wav, wav, args)
            items.update(perceptual_items)
'''
    adapted=source.replace(anchor,injection+anchor)
    raw_grad=torch.autograd.grad
    def hook(loss,p_wav,wav,args):
        return balanced_loss(loss,p_wav,wav,args._perceptual_context,args.perceptual_strength,raw_grad)
    scope=dict(original.__globals__);scope['_perceptual_hook']=hook
    exec(compile(adapted,str(folder/'perceptual_override.py'),'exec'),scope)
    generated=scope['perturb']
    def wrapper(model,hps,wav,sr,sid,args,stoi_fn):
        if getattr(args,'perceptual_strength',0.)>0:args._perceptual_context=RelativeSpectralPenalty(wav,sr)
        try:return generated(model,hps,wav,sr,sid,args,stoi_fn)
        finally:
            if hasattr(args,'_perceptual_context'):del args._perceptual_context
    module.perturb=wrapper
    with (folder/'perceptual_override.py').open('x',encoding='utf-8') as f:f.write(adapted)
    save(folder/'perceptual_provenance.json',dict(original_source=module.__file__,original_sha256=sha(module.__file__),
        override_sha256=sha(folder/'perceptual_override.py'),implementation_sha256=sha(__file__),
        changes='Only objective hook added before original SafeSpeech branch; model, step, sign update and projection unchanged.',
        calibrated_psychoacoustic_model=False,post_filter=False))
