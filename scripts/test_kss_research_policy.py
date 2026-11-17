"""Explicit research scope must not manufacture listening, consent or publication."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import kss_research_policy as P
from run_korean_melo_spec import check_manifest,sha256


class ResearchPolicyTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        audio=self.root/'fixture.wav';sf.write(audio,.1*np.sin(np.arange(44100)*.1),44100,subtype='PCM_16')
        row=dict(utterance_id='fixture_only_not_human',speaker_id='kss_f1',input_kind='human_recording',
            model_speaker='KR',language='ko',split='train',text='수학 신호 계약 시험입니다.',
            transcript_verified=False,research_use_permitted=False,transcript_reference='fixture declaration',
            permission_reference='legacy pending',permission_scope='spec_and_finetuning',source='KSS fixture contract only',
            original_path=str(audio),original_sha256=sha256(audio),sample_rate=44100)
        original=self.root/'original.json';original.write_text(json.dumps({'utterances':[row]}))
        evidence=self.root/'fixture_license.txt';evidence.write_text('test fixture, no real rights granted')
        self.patch=patch.object(P,'CARD_SHA',sha256(evidence));self.patch.start();self.addCleanup(self.patch.stop)
        self.data={'schema_version':1,'research_policy':{'name':P.POLICY,'basis':'public_dataset_license',
            'license_url':P.LICENSE,'original_source_url':P.SOURCE,'noncommercial_research':True,
            'operations':['internal_spec','internal_tts_finetuning','internal_evaluation_synthesis'],
            'separate_speaker_consent':'not_verified','external_publication':'not_approved_by_this_record',
            'license_evidence':{'path':str(evidence),'sha256':sha256(evidence)},
            'original_manifest':str(original),'original_manifest_sha256':sha256(original)},'utterances':[row]}
        self.manifest=self.root/'research.json'

    def write(self):self.manifest.write_text(json.dumps(self.data))

    def test_optin_accepts_fixture_without_falsifying_legacy_flags(self):
        self.write();rows=check_manifest(self.manifest,allow_licensed_research=True)
        self.assertFalse(rows[0]['transcript_verified']);self.assertFalse(rows[0]['research_use_permitted'])

    def test_default_strict_behavior_preserved(self):
        self.write()
        with self.assertRaisesRegex(ValueError,'reviewed transcript'):check_manifest(self.manifest)

    def test_false_listening_or_consent_claim_rejected(self):
        for key in ('transcript_verified','research_use_permitted'):
            self.data['utterances'][0][key]=True
            with self.assertRaisesRegex(ValueError,'declaration changed'):P.validate_policy(self.data)
            self.data['utterances'][0][key]=False
        self.data['research_policy']['separate_speaker_consent']='obtained'
        with self.assertRaisesRegex(ValueError,'fabricate consent'):P.validate_policy(self.data)

    def test_changed_split_or_text_rejected(self):
        for key,value in [('split','test'),('text','다른 문장입니다.')]:
            before=self.data['utterances'][0][key];self.data['utterances'][0][key]=value
            with self.assertRaisesRegex(ValueError,'identity'):P.validate_policy(self.data)
            self.data['utterances'][0][key]=before

    def test_out_of_scope_or_wrong_license_rejected(self):
        for key,value in [('external_publication','approved'),('noncommercial_research',False),('license_url','wrong')]:
            old=self.data['research_policy'][key];self.data['research_policy'][key]=value
            with self.assertRaises(ValueError):P.validate_policy(self.data)
            self.data['research_policy'][key]=old

    def test_evidence_or_original_change_rejected(self):
        Path(self.data['research_policy']['license_evidence']['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'evidence mismatch'):P.validate_policy(self.data)


if __name__=='__main__':unittest.main()
