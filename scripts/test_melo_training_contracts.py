"""Training must reject missing permission, unreviewed humans and changed inputs."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from run_korean_melo_spec import sha256
from train_melo_bounded import check_dataset


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.source=self.root/'source';self.source.mkdir()
        wav=self.root/'original.wav'
        sf.write(wav,.1*np.sin(np.arange(44100)*.1),44100,subtype='PCM_16')
        features=self.root/'features.pt';torch.save({'text':'가상 문장입니다.'},features)
        self.row=dict(utterance_id='fixture',input_kind='mock',language='ko',speaker_id='fixture',
            model_speaker='KR',split='train',text='가상 문장입니다.',transcript_verified=True,
            transcript_reference='fixture prompt',source='mathematical signal',
            permission_reference='generated fixture',research_use_permitted=True,
            permission_scope='technical_test_only',original_path=str(wav),
            original_sha256=sha256(wav),sample_rate=44100)
        self.dataset={'utterances':[{'utterance_id':'fixture','input_kind':'mock','split':'train',
            'text':self.row['text'],'wav':str(wav),'wav_sha256':sha256(wav),
            'features':str(features),'features_sha256':sha256(features)}]}
        self.cfg={'experiment':{'permission_declarations_validated':True,
            'source_spec_run':str(self.source),'dataset_manifest':str(self.root/'dataset.json')}}
        self.write()

    def write(self):
        (self.source/'input_manifest.json').write_text(json.dumps({'schema_version':1,'utterances':[self.row]}))
        (self.root/'dataset.json').write_text(json.dumps(self.dataset))

    def test_accepts_valid_technical_fixture(self):
        self.assertEqual(len(check_dataset(self.cfg)),1)

    def test_rejects_missing_execution_permission_before_files(self):
        self.cfg['experiment']['permission_declarations_validated']=False
        self.cfg['experiment']['dataset_manifest']='missing.json'
        with self.assertRaisesRegex(ValueError,'permission'):
            check_dataset(self.cfg)

    def test_revalidates_human_review_even_with_true_config_gate(self):
        self.row.update(input_kind='human_recording',transcript_verified=False,
                        permission_scope='spec_and_finetuning')
        self.dataset['utterances'][0]['input_kind']='human_recording';self.write()
        with self.assertRaisesRegex(ValueError,'reviewed transcript'):
            check_dataset(self.cfg)

    def test_rejects_changed_wav_and_features(self):
        for role in ('wav','features'):
            with self.subTest(role=role):
                path=Path(self.dataset['utterances'][0][role]);before=path.read_bytes()
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'integrity'):
                    check_dataset(self.cfg)
                path.write_bytes(before)

    def test_rejects_split_leakage_or_changed_transcript(self):
        for key,value in [('split','test'),('text','다른 대본입니다.')]:
            before=self.dataset['utterances'][0][key]
            self.dataset['utterances'][0][key]=value;self.write()
            with self.assertRaisesRegex(ValueError,'differs'):
                check_dataset(self.cfg)
            self.dataset['utterances'][0][key]=before

    def test_rejects_duplicate_ids(self):
        self.dataset['utterances']*=2;self.write()
        with self.assertRaisesRegex(ValueError,'duplicate'):
            check_dataset(self.cfg)


if __name__=='__main__':
    unittest.main()
