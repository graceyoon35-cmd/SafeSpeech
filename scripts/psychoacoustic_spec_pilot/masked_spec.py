"""SPEC-centered projected updates under fixed clean-derived masking limits."""
import numpy as np
import soundfile as sf
import torch
from auditory_masking import pcm_feasible,project_intersection,audit


class MaskedSpecPerturb:
    def __init__(self,models,radius_codes,references,margin_db,progress=None):
        self.models=models;self.radius_codes=radius_codes;self.references=references
        self.margin_db=margin_db;self.progress=progress

    def objective(self,wave):
        """Original SPEC expression; fixed forward and target RNG for fair comparisons.

        Unlike historical iter9 logs, every reference/final PCM is measured under
        the SAME forward seed as every proposed update. Original train-mode frozen
        posterior sampling is retained; no encoder/attacker performance is measured.
        """
        import common as C
        model=self.model;hps=self.hps
        torch.manual_seed(self.seed)
        g=model.emb_g(self.sid).unsqueeze(-1)
        spec=C.linear_spec(hps,wave.squeeze(1))
        lengths=torch.tensor([spec.shape[-1]],device=wave.device)
        z,_,_,_=model.enc_q(spec,lengths,g=g)
        reconstructed=model.dec(z,g=g)
        torch.manual_seed(self.seed)
        # Match the legacy CPU Gaussian target, then transfer to the GPU.
        noise=torch.randn(reconstructed.shape,device='cpu',dtype=reconstructed.dtype).to(reconstructed.device)
        mel=C.reconstruction_loss(hps,wave,reconstructed)
        nr=C.reconstruction_loss(hps,reconstructed,noise)
        kl=C.kl_divergence(hps,reconstructed,noise)
        loss=mel+10*(nr+kl)
        assert torch.isfinite(loss)
        return loss,dict(loss_mel=float(mel.detach()),loss_nr=float(nr.detach()),loss_kl=float(kl.detach()),loss_SPEC=float(loss.detach()))

    def measure_file(self,path):
        x,sr=sf.read(path,dtype='float32');assert sr==44100
        wave=torch.from_numpy(x.copy()).view(1,1,-1).to(self.device)
        with torch.no_grad():_,record=self.objective(wave)
        return record

    def __call__(self,model,hps,wav,sr,sid,args,stoi_fn):
        assert sr==44100 and args.mode=='SPEC' and args.epsilon==2 and args.iters==10 and args.beta==10
        self.model=model;self.hps=hps;self.sid=sid;self.seed=args.seed;self.device=wav.device
        refs={}
        for name,path in self.references.items():refs[name]=self.measure_file(path)
        clean=wav.detach();initial=refs['clean']['loss_SPEC']
        current=np.zeros(clean.numel(),dtype=np.float64);best_codes=np.zeros(clean.numel(),dtype=np.int32)
        best_loss=initial;best_repair=None;history=[]
        step=(args.epsilon/255)/10
        for iteration in range(args.iters):
            p=(clean+torch.from_numpy(current).to(self.device,dtype=clean.dtype).view_as(clean)).detach().requires_grad_(True)
            loss,parts=self.objective(p)
            gradient,=torch.autograd.grad(loss,p)
            direction=gradient.detach().reshape(-1).cpu().numpy()
            assert np.isfinite(direction).all() and np.any(direction)
            proposed=current-step*np.sign(direction)
            continuous=project_intersection(proposed,self.models,self.margin_db,self.radius_codes,cycles=10,reserve_db=1.)
            codes,repair=pcm_feasible(continuous,self.models,self.margin_db,self.radius_codes,repair_cycles=8)
            qwave=clean+torch.from_numpy(codes/32768.).to(self.device,dtype=clean.dtype).view_as(clean)
            with torch.no_grad():_,evaluated=self.objective(qwave)
            accepted=evaluated['loss_SPEC']<best_loss-1e-7
            if accepted:
                best_codes=codes.copy();best_loss=evaluated['loss_SPEC'];best_repair=repair
                current=best_codes/32768.
            else:step*=.5
            history.append(dict(iter=iteration,**parts,step_after_acceptance=step,accepted=accepted,
                projected_pcm_loss=evaluated['loss_SPEC'],best_pcm_loss=best_loss,repair=repair))
            if self.progress:self.progress(iteration=iteration+1,accepted=accepted,best_pcm_loss=best_loss,clean_pcm_loss=initial,
                residual_codes_nonzero=int(np.count_nonzero(best_codes)),repair_gain=repair['residual_gain_fallback'])
        output=clean+torch.from_numpy(best_codes/32768.).to(self.device,dtype=clean.dtype).view_as(clean)
        with torch.no_grad():_,final=self.objective(output)
        checks=audit(best_codes/32768.,self.models,self.margin_db)
        assert all(r['pass_constraint'] for r in checks) and abs(best_codes).max()<=257
        ratios={}
        for ref in ('eps2_iter10','silence_only_smooth'):
            improvement=initial-refs[ref]['loss_SPEC']
            ratios[ref]=(initial-final['loss_SPEC'])/improvement if improvement>0 else None
        stats=dict(history=history,objective_references=refs,final_pcm_objective=final,
            reference_improvement_retained=ratios,
            guard_95pct_baseline=ratios['eps2_iter10'] is not None and ratios['eps2_iter10']>=.95,
            guard_95pct_silence=ratios['silence_only_smooth'] is not None and ratios['silence_only_smooth']>=.95,
            masking_checks=checks,selected_repair=best_repair,clean_fallback=not bool(np.any(best_codes)),
            accepted_steps=sum(r['accepted'] for r in history),
            recon_mel_l1_clean=refs['clean']['loss_mel'],recon_mel_l1_protected=final['loss_mel'],
            max_abs_delta=float(np.max(np.abs(best_codes)))/32768,
            objective_seed_convention='Seed52 before posterior forward AND before Gaussian target; all final PCM references remeasured identically.',
            no_attack_training=True,defense_effect_unmeasured=True)
        return output,stats
