"""Numerical/physical sanity checks; no model or saved waveform generation."""
import unittest
import numpy as np
import torch
from auditory_masking import ath,bark,stft_numpy,build_scale,build_model,audit,pcm_feasible


class MaskingTests(unittest.TestCase):
    def test_ath_and_bark_reference(self):
        self.assertAlmostEqual(float(ath(1000)),3.369066525895342,places=5)
        self.assertTrue(np.all(np.diff(bark(np.arange(1,22050)))>0))

    def test_independent_transform_and_roundtrip(self):
        x=np.random.default_rng(1).normal(0,.01,5000)
        model=build_scale(x,2048)
        ts=model.fft_torch(torch.tensor(x,dtype=torch.float64)).numpy()
        np.testing.assert_allclose(ts,stft_numpy(x,2048),atol=1e-14,rtol=1e-12)
        recovered=torch.istft(torch.from_numpy(ts),n_fft=2048,hop_length=512,window=model.window,length=len(x)).numpy()
        np.testing.assert_allclose(x,recovered,atol=1e-15)

    def test_maskers_and_margin_monotonicity(self):
        t=np.arange(8192)/44100;x=.1*np.sin(2*np.pi*1000*t)
        model=build_scale(x,2048)
        self.assertTrue(np.any(model.masker_counts>0))
        self.assertTrue(np.all(model.limit(6)<=model.limit(3)))
        mid=model.raw_threshold[:,5];f=np.fft.rfftfreq(2048,1/44100)
        # A tonal masker must raise nearby threshold relative to the same model ATH.
        offset=96-10*np.log10(model.clean_peak_power)
        ath_only=10**((ath(f)-offset)/10)
        near=np.argmin(abs(f-1000))
        self.assertGreater(mid[near]/ath_only[near],100)

    def test_pcm_grid_and_feasibility(self):
        t=np.arange(4096)/44100;x=.1*np.sin(2*np.pi*1000*t)
        models=build_model(x);radius=np.full(len(x),257,dtype=np.int32)
        q,info=pcm_feasible(np.random.default_rng(2).normal(0,.001,len(x)),models,6,radius)
        self.assertLessEqual(int(np.abs(q).max()),257)
        self.assertTrue(all(r['pass_constraint'] for r in audit(q/32768.,models,6)))
        self.assertTrue(all(r['pass_constraint'] for r in audit(np.zeros_like(x),models,6)))

    def test_nonzero_tonal_perturbation_is_not_erased(self):
        t=np.arange(8192)/44100;x=.1*np.sin(2*np.pi*1000*t)
        models=build_model(x);radius=np.full(len(x),257,dtype=np.int32)
        q,_=pcm_feasible(.001*np.sin(2*np.pi*1000*t),models,3,radius)
        self.assertGreater(int(np.count_nonzero(q)),100)
        self.assertTrue(all(r['pass_constraint'] for r in audit(q/32768.,models,3)))


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
