"""Checkpoint integrity, optimizer state and exact RNG restoration contracts."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from melo_checkpoint import save_checkpoint,verify_checkpoint,restore_rng


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        dataset=self.root/'dataset.json';dataset.write_text('{}')
        self.cfg={'train':{'learning_rate':.001},'experiment':{'scope':'test','dataset_manifest':str(dataset)}}
        self.nets=[torch.nn.Linear(2,1) for _ in range(3)]
        self.opts=[torch.optim.AdamW(n.parameters(),lr=.001) for n in self.nets]
        for net,opt in zip(self.nets,self.opts):
            net(torch.ones(1,2)).sum().backward();opt.step()
        self.directory=self.root/'step_1'
        save_checkpoint(self.directory,self.cfg,self.nets,self.opts,1,[{'step':1}],torch.ones(1))

    def test_committed_checkpoint_and_optimizer(self):
        manifest=verify_checkpoint(self.directory,self.cfg)
        state=torch.load(self.directory/manifest['files']['state']['path'],weights_only=True)
        self.assertEqual(state['step'],1)
        self.assertEqual(int(next(iter(state['optimizers'][0]['state'].values()))['step']),1)

    def test_rejects_config_change(self):
        changed=copy.deepcopy(self.cfg);changed['train']['learning_rate']=.01
        with self.assertRaisesRegex(ValueError,'config mismatch'):
            verify_checkpoint(self.directory,changed)

    def test_rejects_corruption(self):
        (self.directory/'G_1.pth').write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'integrity'):
            verify_checkpoint(self.directory,self.cfg)

    def test_rejects_changed_dataset(self):
        Path(self.cfg['experiment']['dataset_manifest']).write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'dataset'):
            verify_checkpoint(self.directory,self.cfg)

    def test_incomplete_save_rejected(self):
        (self.directory/'checkpoint.json').unlink()
        with self.assertRaises(FileNotFoundError):
            verify_checkpoint(self.directory,self.cfg)

    def test_rng_replays_exactly(self):
        state=torch.load(self.directory/'training_state.pth',weights_only=True)['rng']
        restore_rng(state);a=(torch.rand(3),np.random.rand(3),random.random())
        restore_rng(state);b=(torch.rand(3),np.random.rand(3),random.random())
        self.assertTrue(torch.equal(a[0],b[0]));np.testing.assert_array_equal(a[1],b[1]);self.assertEqual(a[2],b[2])


if __name__=='__main__':
    unittest.main()
