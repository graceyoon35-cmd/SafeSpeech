"""Projection contracts and exact traced-baseline equivalence on synthetic data."""
import unittest
import numpy as np
import torch
from auditory_masking import build_model,pcm_feasible,project_intersection,audit
from projection_refinement import original_pcm_traced,refined_pcm,anchor_backtrack


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t=np.arange(4096)/44100
        cls.x=.1*np.sin(2*np.pi*1000*t);cls.models=build_model(cls.x);cls.radius=np.full(len(t),257,dtype=np.int32)

    def test_trace_does_not_change_original_repair(self):
        d=np.random.default_rng(2).normal(0,.001,len(self.x))
        a,ai=pcm_feasible(d,self.models,3,self.radius,repair_cycles=8)
        stages=[];b,bi=original_pcm_traced(d,self.models,self.radius,lambda name,value:stages.append(name))
        np.testing.assert_array_equal(a,b)
        self.assertEqual(ai['residual_gain_fallback'],bi['residual_gain_fallback'])
        self.assertIn('first_pcm_round',stages)

    def test_both_variants_preserve_hard_constraints(self):
        anchor=np.zeros(len(self.x),dtype=np.int32)
        d=np.random.default_rng(3).normal(0,.001,len(self.x))
        for mode in ('anchor_backtrack','local_caps_anchor'):
            q,info=refined_pcm(d,anchor,self.models,self.radius,mode)
            self.assertLessEqual(int(abs(q).max()),257)
            self.assertTrue(all(r['pass_constraint'] for r in audit(q/32768.,self.models,3)))

    def test_anchor_identity_fallback_keeps_existing_perturbation(self):
        t=np.arange(len(self.x))/44100
        anchor,_=pcm_feasible(.0005*np.sin(2*np.pi*1000*t),self.models,3,self.radius)
        self.assertTrue(np.any(anchor))
        q,info=anchor_backtrack(anchor/32768.,anchor,self.models,self.radius)
        np.testing.assert_array_equal(q,anchor);self.assertEqual(info['alpha'],1.)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
