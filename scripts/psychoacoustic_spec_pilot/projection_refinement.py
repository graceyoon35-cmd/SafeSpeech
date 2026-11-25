"""Two fixed 3 dB repairs, plus exact staged replay of the preserved baseline.

The auditory model, epsilon, step, iterations, SPEC expression and RNG convention
are unchanged. Only conversion of the same continuous proposal to feasible PCM
is varied. No threshold relaxation, parameter training, or independent encoder.
"""
import numpy as np
import torch
from auditory_masking import audit,project_intersection,stft_numpy
from masked_spec import MaskedSpecPerturb

ALPHAS=(1.,.95,.9,.8,.7,.6,.5,.4,.3,.2,.1,.05,.02,.01,0.)


def rounded(delta,radius):
    return np.clip(np.rint(np.asarray(delta)*32768),-radius,radius).astype(np.int32)


def feasible(codes,models):return all(r['pass_constraint'] for r in audit(codes/32768.,models,3))


def anchor_backtrack(delta,anchor_codes,models,radius):
    """Backtrack toward CURRENT feasible PCM, never toward zero/history erasure."""
    assert feasible(anchor_codes,models)
    anchor=anchor_codes/32768.
    for alpha in ALPHAS:
        codes=rounded(anchor+alpha*(delta-anchor),radius)
        if feasible(codes,models):
            return codes,dict(alpha=alpha,local_passes=0,fallback='feasible_anchor',clean_fallback=not bool(np.any(codes)))
    raise AssertionError('Current feasible PCM anchor must be a valid fallback')


def refined_pcm(continuous,anchor_codes,models,radius,mode):
    # Preserve the baseline's initial additional projection, isolating repair.
    d=project_intersection(continuous,models,3,radius,cycles=8,reserve_db=1.)
    if mode=='anchor_backtrack':return anchor_backtrack(d,anchor_codes,models,radius)
    assert mode=='local_caps_anchor'
    caps=[m.limit(3,1.) for m in models]
    for iteration in range(8):
        codes=rounded(d,radius)
        if feasible(codes,models):
            return codes,dict(alpha=1.,local_passes=iteration,fallback='none',clean_fallback=not bool(np.any(codes)))
        for i,m in enumerate(models):
            codes=rounded(d,radius)
            power=np.abs(stft_numpy(codes/32768.,m.n_fft)/m.n_fft)**2
            ratio=power/np.maximum(m.limit(3),1e-300)
            bad=ratio>1+1e-6
            # Tighten only violating time-frequency bins. All target 3dB limits
            # remain immutable; there is no 2..7dB global reserve escalation.
            factors=np.minimum(.98,.95/np.sqrt(np.maximum(ratio[bad],1.)))
            caps[i][bad]*=factors**2
            d=m.project(d,caps[i]);d=np.clip(d,-radius/32768.,radius/32768.)
    codes,info=anchor_backtrack(d,anchor_codes,models,radius)
    info['local_passes']=8;info['fallback']='local_caps_then_anchor'
    return codes,info


def original_pcm_traced(continuous,models,radius,emit):
    """Exact preserved pcm_feasible logic; additional measurements have no writes."""
    d=project_intersection(continuous,models,3,radius,cycles=8,reserve_db=1.)
    emit('prequant_projection_8',d)
    attempts=[]
    for i in range(6):
        q=rounded(d,radius);checks=audit(q/32768.,models,3)
        if i==0:emit('first_pcm_round',q/32768.)
        attempts.append(dict(repair=i,max_excess_db=max(r['max_excess_db'] for r in checks)))
        if all(r['pass_constraint'] for r in checks):
            emit('post_repairs_float',d)
            return q,dict(repairs=attempts,residual_gain_fallback=1.,clean_fallback=not bool(np.any(q)),audit=checks)
        d=project_intersection(q/32768.,models,3,radius,cycles=8,reserve_db=2.+i)
    emit('post_repairs_float',d)
    for gain in (.9,.8,.7,.6,.5,.4,.3,.2,.1,.05,.02,.01,0.):
        q=rounded(d*gain,radius);checks=audit(q/32768.,models,3)
        if all(r['pass_constraint'] for r in checks):
            return q,dict(repairs=attempts,residual_gain_fallback=gain,clean_fallback=not bool(np.any(q)),audit=checks)
    raise AssertionError('Zero residual must be feasible')


class RefinementPerturb(MaskedSpecPerturb):
    def __init__(self,models,radius_codes,references,mode,progress=None):
        super().__init__(models,radius_codes,references,3,progress)
        assert mode in ('baseline_trace','anchor_backtrack','local_caps_anchor')
        self.mode=mode

    def __call__(self,model,hps,wav,sr,sid,args,stoi_fn):
        assert sr==44100 and args.iters==10 and args.epsilon==2 and args.beta==10 and args.seed==52
        assert not any(p.requires_grad for p in model.parameters())
        self.model=model;self.hps=hps;self.sid=sid;self.seed=52;self.device=wav.device
        refs={name:self.measure_file(path) for name,path in self.references.items()}
        clean=wav.detach();initial=refs['clean']['loss_SPEC']
        current=np.zeros(clean.numel(),dtype=np.float64);best_codes=np.zeros(clean.numel(),dtype=np.int32)
        best_loss=initial;history=[];stages=[];best_repair=None;step=(2/255)/10
        for iteration in range(10):
            p=(clean+torch.from_numpy(current).to(self.device,dtype=clean.dtype).view_as(clean)).detach().requires_grad_(True)
            loss,parts=self.objective(p);gradient,=torch.autograd.grad(loss,p)
            direction=gradient.detach().reshape(-1).cpu().numpy()
            assert np.isfinite(direction).all() and np.any(direction)
            def emit(stage,d):
                with torch.no_grad():
                    value=clean+torch.from_numpy(d).to(self.device,dtype=clean.dtype).view_as(clean)
                    _,record=self.objective(value)
                checks=audit(d,self.models,3)
                stages.append(dict(iteration=iteration+1,stage=stage,**record,loss_current=parts['loss_SPEC'],loss_clean=initial,
                    max_excess_db=max(r['max_excess_db'] for r in checks),
                    violating_bins=sum(r['violating_bins'] for r in checks),bins=sum(r['bins'] for r in checks),
                    max_delta_codes=float(np.max(np.abs(d)))*32768))
            proposed=current-step*np.sign(direction)
            if self.mode=='baseline_trace':emit('unconstrained_proposal',proposed)
            continuous=project_intersection(proposed,self.models,3,self.radius_codes,cycles=10,reserve_db=1.)
            if self.mode=='baseline_trace':
                emit('continuous_projection_10',continuous)
                codes,repair=original_pcm_traced(continuous,self.models,self.radius_codes,emit)
            else:codes,repair=refined_pcm(continuous,best_codes,self.models,self.radius_codes,self.mode)
            with torch.no_grad():
                qwave=clean+torch.from_numpy(codes/32768.).to(self.device,dtype=clean.dtype).view_as(clean)
                _,evaluated=self.objective(qwave)
            if self.mode=='baseline_trace':emit('final_pcm_repaired',codes/32768.)
            assert feasible(codes,self.models) and int(np.abs(codes).max())<=257
            accepted=evaluated['loss_SPEC']<best_loss-1e-7
            if accepted:
                best_codes=codes.copy();best_loss=evaluated['loss_SPEC'];best_repair=repair;current=best_codes/32768.
            else:step*=.5
            history.append(dict(iter=iteration,**parts,accepted=accepted,step_after_acceptance=step,
                projected_pcm_loss=evaluated['loss_SPEC'],best_pcm_loss=best_loss,repair=repair))
            if self.progress:self.progress(iteration=iteration+1,best_pcm_loss=best_loss,accepted=accepted,mode=self.mode,
                alpha=repair.get('alpha'),local_passes=repair.get('local_passes'))
        result=clean+torch.from_numpy(best_codes/32768.).to(self.device,dtype=clean.dtype).view_as(clean)
        with torch.no_grad():_,final=self.objective(result)
        ratios={}
        for name in ('eps2_iter10','silence_only_smooth','old_mask3'):
            denominator=initial-refs[name]['loss_SPEC']
            ratios[name]=(initial-final['loss_SPEC'])/denominator if denominator>0 else None
        return result,dict(history=history,stage_measurements=stages,objective_references=refs,final_pcm_objective=final,
            reference_improvement_retained=ratios,accepted_steps=sum(r['accepted'] for r in history),selected_repair=best_repair,
            clean_fallback=not bool(np.any(best_codes)),masking_checks=audit(best_codes/32768.,self.models,3),
            recon_mel_l1_clean=refs['clean']['loss_mel'],recon_mel_l1_protected=final['loss_mel'],
            no_attack_training=True,no_sim_cer=True,projection_mode=self.mode)
