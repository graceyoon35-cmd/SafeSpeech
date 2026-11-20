import tempfile
from pathlib import Path
import unittest

import numpy as np
import soundfile as sf

from audit_kss_final import distance, normalize, quality
from protect_wav import protect, project_pcm16


class FinalValidationTests(unittest.TestCase):
    def test_cer_insertions_can_exceed_100_percent(self):
        self.assertEqual(distance('가나다', '가나다라라라라'), 4)
        self.assertGreater(distance('가나다', '가나다라라라라')/3, 1)

    def test_edit_distance_substitution_deletion_and_empty(self):
        self.assertEqual(distance('가나다', '가마'), 2)
        self.assertEqual(distance('', '가나'), 2)
        self.assertEqual(distance('가나', ''), 2)

    def test_existing_korean_normalization_is_explicit(self):
        self.assertEqual(normalize('세일은 1월 15일까지입니다.'), '세일은1월15일까지입니다')

    def test_pcm16_one_lsb_tolerance_is_distinct_from_strict_bound(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d)/'a.wav'; b = Path(d)/'b.wav'
            t = np.arange(44100)/44100
            x = .2*np.sin(2*np.pi*220*t)
            sf.write(a,x,44100,subtype='PCM_16')
            x,_ = sf.read(a)
            sf.write(b,x+1/32768,44100,subtype='PCM_16')
            result = quality(a,b,.5/32768)
            self.assertFalse(result['strict_epsilon_pass'])
            self.assertTrue(result['epsilon_pass'])

    def test_reject_unsupported_rate_before_creating_output(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a.wav'; output=Path(d)/'result'
            sf.write(a,np.ones(16000)*.1,16000,subtype='PCM_16')
            with self.assertRaisesRegex(ValueError,'44100'):
                protect(a,output,Path(d)/'absent-model')
            self.assertFalse(output.exists())

    def test_reject_silent_input(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a.wav'
            sf.write(a,np.zeros(44100),44100,subtype='PCM_16')
            with self.assertRaisesRegex(ValueError,'non-silent'):
                protect(a,Path(d)/'out',Path(d)/'absent-model')

    def test_reject_nonfinite_epsilon(self):
        with self.assertRaisesRegex(ValueError,'epsilon'):
            protect(Path('unused.wav'),Path('unused'),Path('unused'),epsilon=float('nan'))

    def test_reject_invalid_iteration(self):
        with self.assertRaisesRegex(ValueError,'iterations'):
            protect(Path('unused.wav'),Path('unused'),Path('unused'),iterations=0)

    def test_pcm16_projection_enforces_strict_epsilon_without_wraparound(self):
        clean = np.array([-32768,-32000,0,32000,32767],dtype=np.float64)/32768
        raw = np.array([-32768,-32768,1029,32767,32767],dtype=np.float64)/32768
        codes, adjusted = project_pcm16(clean,raw,8.)
        self.assertEqual(codes.dtype,np.int16)
        self.assertLessEqual(float(np.max(np.abs(codes.astype(np.float64)/32768-clean))),8/255)
        self.assertEqual(int(codes[2]),1028)
        self.assertEqual(adjusted,1)


if __name__ == '__main__':
    unittest.main()
