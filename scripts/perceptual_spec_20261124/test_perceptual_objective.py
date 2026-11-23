"""CPU checks of the new objective only; no speech/defense success fixtures."""
import unittest
import torch
from perceptual_objective import RelativeSpectralPenalty,balanced_loss

class ObjectiveTests(unittest.TestCase):
    def test_spectral_penalty_and_gradient(self):
        torch.set_num_threads(2)
        t=torch.arange(44100,dtype=torch.float32)/44100
        clean=(.15*torch.sin(2*torch.pi*500*t))[None,None]
        context=RelativeSpectralPenalty(clean,44100)
        self.assertEqual(context(torch.zeros_like(clean)).item(),0)
        masked=.0005*torch.sin(2*torch.pi*500*t).reshape_as(clean)
        hiss=.0005*torch.sin(2*torch.pi*10000*t).reshape_as(clean)
        self.assertGreater(context(hiss).item(),context(masked).item())
        protected=(clean+hiss).requires_grad_(True)
        base=protected.square().mean()
        loss,stats=balanced_loss(base,protected,clean,context,.5,torch.autograd.grad)
        grad=torch.autograd.grad(loss,protected)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(torch.count_nonzero(grad).item(),0)
        self.assertGreater(stats['perceptual_coefficient'],0)

if __name__=='__main__':unittest.main()
