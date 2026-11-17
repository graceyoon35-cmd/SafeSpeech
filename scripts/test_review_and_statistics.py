"""Listening actions, unchanged sources, and prompt-level uncertainty contracts."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from kss_review_tool import ReviewStore,digest
from evaluation_statistics import statistics
from build_authorized_kss_manifest import build


class ReviewTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.bundle=self.root/'bundle';self.bundle.mkdir()
        wav=self.bundle/'fixture.wav'
        with wave.open(str(wav),'wb') as stream:
            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000);stream.writeframes(b'\x01\x00'*16000)
        uid='math_fixture'
        self.original={'utterance_id':uid,'original_path':'fixture.wav','original_sha256':digest(wav),
            'text':'수학 신호 대본 자리입니다.','transcript_verified':False,'research_use_permitted':False}
        (self.bundle/'spec_input.json').write_text(json.dumps({'utterances':[self.original]}),encoding='utf-8')
        self.priority=self.root/'priority.csv'
        with self.priority.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['utterance_id','sha256','reference','asr_hypothesis','normalized_cer','split'])
            writer.writeheader();writer.writerow(dict(utterance_id=uid,sha256=digest(wav),reference=self.original['text'],
                asr_hypothesis='수학 신호',normalized_cer=.5,split='test'))
        self.store=ReviewStore(self.bundle,self.priority,self.root/'reviews')

    def test_initial_state_is_unreviewed(self):
        self.assertFalse(self.store.decisions)
        self.assertFalse(self.store.original['math_fixture']['research_use_permitted'])

    def test_cannot_confirm_before_playback_or_before_end(self):
        with self.assertRaises(ValueError):self.store.decide('math_fixture','예','fixture_reviewer','')
        with patch('kss_review_tool.time.monotonic',return_value=0):self.store.begin_playback('math_fixture')
        with patch('kss_review_tool.time.monotonic',return_value=.1),self.assertRaises(ValueError):
            self.store.decide('math_fixture','예','fixture_reviewer','')

    def test_explicit_fixture_decision_preserves_source_and_reloads(self):
        before=digest(self.bundle/'spec_input.json')
        with patch('kss_review_tool.time.monotonic',return_value=0):self.store.begin_playback('math_fixture')
        with patch('kss_review_tool.time.monotonic',return_value=2):self.store.decide('math_fixture','예','fixture_reviewer','test only')
        self.assertEqual(before,digest(self.bundle/'spec_input.json'))
        reloaded=ReviewStore(self.bundle,self.priority,self.root/'reviews')
        self.assertEqual(reloaded.decisions['math_fixture']['review_status'],'MATCH_CONFIRMED_BY_REVIEWER')
        self.assertFalse(reloaded.original['math_fixture']['transcript_verified'])

    def test_pending_does_not_approve(self):
        self.store.decide('math_fixture','보류','fixture_reviewer','')
        self.assertEqual(self.store.decisions['math_fixture']['heard_and_matches(예/아니오)'],'')

    def test_changed_audio_rejected(self):
        (self.bundle/'fixture.wav').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.store.begin_playback('math_fixture')

    def test_rights_template_cannot_authorize(self):
        rights=self.root/'rights.json';rights.write_text(json.dumps({'schema_version':1,'dataset':'KSS'}))
        with self.assertRaisesRegex(ValueError,'use-scope confirmation'):
            build(self.bundle/'spec_input.json',self.priority,rights)


class StatisticsTests(unittest.TestCase):
    def test_single_observation_has_no_interval(self):
        result=statistics([.2]);self.assertIsNone(result['ci95_low']);self.assertIsNone(result['sample_variance'])

    def test_bootstrap_reproducible(self):
        a=statistics([.1,.2,.4,.5]);self.assertEqual(a,statistics([.1,.2,.4,.5]))
        self.assertLessEqual(a['ci95_low'],a['mean']);self.assertGreaterEqual(a['ci95_high'],a['mean'])

    def test_sample_variance(self):
        self.assertAlmostEqual(statistics([1.,2.,3.])['sample_variance'],1.)

    def test_empty_or_nonfinite_rejected(self):
        for values in ([],[float('nan')],[float('inf')]):
            with self.assertRaises(ValueError):statistics(values)


if __name__=='__main__':unittest.main()
