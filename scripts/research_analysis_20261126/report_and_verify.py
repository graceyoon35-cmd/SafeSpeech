"""Independent saved-audio checks and Korean pilot report; never loads a model."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import wave
import zipfile
import numpy as np
import soundfile as sf

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
OUT=BASE/'pilot'
CODE=BASE/'code_worktree/scripts/psychoacoustic_spec_pilot'
from research_bootstrap import setup
CODE_ROOT,ROOT,LEGACY=setup()
BASE=ROOT/'runs/psychoacoustic_spec_pilot_20261125'
OUT=BASE/'pilot'
CODE=CODE_ROOT/'scripts/psychoacoustic_spec_pilot'
from auditory_masking import build_model,audit


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,o):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)


def git(p,*args):
    win=subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
    return subprocess.check_output(['/mnt/c/Program Files/Git/cmd/git.exe','-C',win,*args],text=True).strip()


def main():
    complete=read(OUT/'complete.json');assert complete['status']=='FIVE_UTTERANCE_PILOT_COMPLETE'
    quality=rows(OUT/'quality_per_file.csv');assert len(quality)==25
    sums=read(OUT/'summary.json');by={r['condition']:r for r in sums}
    inputs={r['id']:r['path'] for r in quality if r['condition']=='clean'}
    masking=[];extrema=[];forwards=[]
    for uid,original in inputs.items():
        x,sr=sf.read(original,dtype='int16');models=build_model(x.astype(np.float64)/32768)
        stored=np.load(OUT/'thresholds'/(uid+'.npz'))
        for m in models:np.testing.assert_array_equal(m.raw_threshold,stored[f'raw_threshold_{m.n_fft}'])
        for margin in (3,6):
            label=f'mask{margin}db_iter10';r=next(r for r in quality if r['id']==uid and r['condition']==label)
            path=Path(r['path']);assert sha(path)==r['sha256']
            y,rate=sf.read(path,dtype='int16');assert rate==sr==44100 and x.shape==y.shape
            delta=y.astype(np.int32)-x.astype(np.int32)
            assert np.abs(delta).max()<=257 and not np.any((y==-32768)|(y==32767))
            checks=audit(delta/32768.,models,margin);assert all(c['pass_constraint'] for c in checks)
            for c in checks:masking.append(dict(id=uid,condition=label,**c))
            gen=read(path.parent/'generation.json')['stats']
            assert gen['clean_fallback'] is False and not gen['guard_95pct_silence']
            assert all(not p['accepted'] or p['projected_pcm_loss']<=p['loss_SPEC']+.01 for p in gen['history'])
            assert abs(gen['reread_saved_pcm_objective']['loss_SPEC']-gen['final_pcm_objective']['loss_SPEC'])<.01
            forwards.append(dict(id=uid,margin=margin,references=gen['objective_references']))
            extrema.append(dict(id=uid,condition=label,maximum_delta_codes=int(np.abs(delta).max()),
                residual_gain_fallback=gen['selected_repair']['residual_gain_fallback'],accepted_steps=gen['accepted_steps']))
        a,b=[r['references'] for r in forwards if r['id']==uid]
        assert all(abs(a[c]['loss_SPEC']-b[c]['loss_SPEC'])<.01 for c in a)
    assert len(masking)==20
    blind=read(OUT/'blind_verification.json');zip_path=Path(blind['zip']);assert sha(zip_path)==blind['zip_sha256']
    mapping=rows(OUT/'organizer_private/answer_mapping.csv');assert len(mapping)==25
    for r in mapping:
        p=OUT/'blind_listening'/r['filename'];assert sha(p)==r['copied_sha256']
        with wave.open(str(p),'rb') as f:data=f.readframes(f.getnframes())
        assert hashlib.sha256(data).hexdigest()==r['pcm_sha256']
        decoded=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(p),'-f','null','-'],capture_output=True)
        assert decoded.returncode==0
    with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None and len(z.namelist())==26
    own_stats={}
    allmask=rows(OUT/'masking_per_file.csv')
    for label in by:
        for margin in (3,6):
            a=[r for r in allmask if r['condition']==label and int(r['margin_db'])==margin]
            own_stats[label,margin]=dict(violation_fraction=sum(int(r['violating_bins']) for r in a)/sum(int(r['bins']) for r in a),
                max_excess_db=max(float(r['max_excess_db']) for r in a))
    sil=by['silence_only_smooth'];changes={}
    for label in ('mask3db_iter10','mask6db_iter10'):
        changes[label]={r:20*float(np.log10(by[label][r+'_noise_rms']/sil[r+'_noise_rms'])) for r in ('silence','weak','voiced')}
    processes=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True)
    active=[r for r in processes.splitlines() if 'run_pilot.py' in r or 'train_melo_bounded.py' in r or 'evaluate_paired_melo_smoke.py' in r]
    assert not active
    assert git(ROOT,'branch','--show-current')==complete['original_branch'] and git(ROOT,'rev-parse','HEAD')==complete['original_head']
    assert not git(ROOT,'status','--porcelain','--untracked-files=all')
    assert git(BASE/'code_worktree','branch','--show-current')=='work/psychoacoustic-spec-pilot-20261125'
    assert git(BASE/'code_worktree','rev-parse','HEAD')=='954a06cde13065074328bc9476eaf8238845d1ea'
    assert not git(BASE/'code_worktree','diff','--cached','--name-only')
    assert not read(OUT/'preservation.json')['changed']
    save(OUT/'independent_final_verification.json',dict(status='PASS',source_utterances=5,new_protected_wavs=10,blind_wavs=25,
        strict_epsilon_pcm16=True,max_code_delta=max(r['maximum_delta_codes'] for r in extrema),masking_checks=masking,
        saved_thresholds_recomputed_equal=True,fixed_rng_references_agree=True,final_pcm_objectives_reread=True,
        sample_diagnostics=extrema,noise_rms_changes_vs_silence_db=changes,zip_crc_pass=True,zip_sha256=sha(zip_path),
        objective_guard_failures=10,human_listening_verified=False,defense_unmeasured=True,recommend_expand_140=False,
        active_generation_training_evaluation_processes=active,no_commit_push=True))
    lines=['청각 마스킹 역치 SPEC: 5발화 pilot 결과 — 2026-11-25',
        '', '판정: 품질 수치와 모델 기반 마스킹 제약은 개선/통과했지만 SPEC 목적 보존에 실패해140개 확대를 권하지 않는다.',
        '실제 사람 청취와 Melo1000step/SIM/CER는 이번에 수행하지 않았다. 방어 성능이 약해졌다고 확정할 수 없다.',
        '', '구현: 원본 전역PSD피크96dB 기준,tonal masker선택/0.5Bark중복제거/두기울기확산/ATH의 선형전력합.',
        'periodic Hann2048/512와1024/256. ATH범위밖은 -100dBFS/bin공학상한. 실제SPL보정/시간적마스킹은 없다.',
        '기존 국소 STFT평균 고정벌점과 달리 원본에서 계산한 프레임·주파수별 실제모델역치를 제약으로 쓴다.',
        'epsilon2/255,seed52,beta10,10회SPEC gradient,무음시간별epsilonbox/두STFT제약 교대투영.',
        '원음 볼륨은 바꾸지 않았다. PCM양자화 뒤 재검사하며 필요했던 잔차감쇠fallback은 숨기지 않고 모두 기록했다.',
        '전체clean으로 대체한 결과는0개지만, 보정의 감쇠 비용 때문에 방어목적 손실이 클 수 있다.',
        '부호gradient10회만의 pilot이며 제약하 최적해 또는 모든세팅의 불가능성을 증명한 것은 아니다.',
        '', '동일5개: '+', '.join(complete['ids']),
        '조건 | SNRdB | STOI | 무음RMS | 약한발화RMS | 강한발화RMS | 평균L_SPEC']
    for r in sums:
        snr='해당없음' if r['snr_db'] is None else f"{r['snr_db']:.5f}"
        lines.append(f"{r['condition']} | {snr} | {r['stoi']:.8f} | {r['silence_noise_rms']:.9f} | {r['weak_noise_rms']:.9f} | {r['voiced_noise_rms']:.9f} | {r['mean_loss_SPEC']:.6f}")
    lines+=['','5개평균은 이전140개평균26.22dB/STOI0.98117과 표본이 다르다. 여기서는같은5개끼리 비교한다.',
        '구간은 원본20ms RMS/10ms 간격 기반 대리VAD이며 사람검수VAD가 아니다.',
        '', '마스킹 검사: 조건 | 검사여유dB | 위반비율 | 최대초과dB (두해상도 pooled)']
    for (label,margin),r in own_stats.items():lines.append(f"{label} | {margin} | {r['violation_fraction']:.9f} | {r['max_excess_db']:.6f}")
    lines+=['','새후보10개모두 clipping0, 원본/길이/SR/채널대응일치, PCM16엄격epsilon통과.',
        '자기여유3/6dB의2048/1024검사모두 위반0bin, 최대/P95초과0dB.',
        '수치허용은 전력비1e-6(0.0000044dB)뿐. 생리적불가청/어떤재생조건에서도불가청을 입증한 것이 아니다.',
        '', 'SPEC 목적 변화: final PCM을동일forward/Gaussianseed52로 재측정. 작을수록생성목적 최적화가강함.',
        '개선량은L_clean-L_protected. 보존율은발화별(새개선량/기존개선량)의평균이며 실제방어보존율이아니다.']
    for label in ('mask3db_iter10','mask6db_iter10'):
        r=by[label]
        lines.append(f"{label}: eps2대비개선량 {r['mean_retained_vs_eps2']*100:.3f}%, silence대비 {r['mean_retained_vs_silence']*100:.3f}%, 95%guard {r['guard95_silence_pass_count']}/5통과.")
        lines.append('silence대비구간RMS변화dB: '+json.dumps(changes[label],ensure_ascii=False))
    lines+=['','무음억제형은SPEC loss상으로eps2보다약해도기존공격SIM은더낮았다. 따라서이proxy와실제방어가단조관계라고가정하지않는다.',
        '새후보는 loss보존미달과실제방어미측정으로현재성공설정으로추천하지않는다.',
        '3dB는6dB보다목적보존이낫지만, 둘다당장140개신규공격학습으로확대할근거가부족하다.',
        '다음은5개범위에서양자화보정감쇠/제약투영과목적방향을개선하고사람청취를먼저확인하는것이다.',
        '이번요청에서는추가반복/대규모학습을시작하지않았다.',
        '', '블라인드청취: '+str(OUT/'blind_listening'),
        '공유ZIP: '+str(zip_path),
        '정답표(평가자에게공유금지): '+str(OUT/'organizer_private/answer_mapping.csv'),
        '5발화×5조건=25WAV,안내문포함ZIP26파일,CRC/SHA통과.복사본메타데이터만통일하고PCM샘플은같다.',
        '', '품질CSV: quality_per_file.csv / summary.csv',
        '역치초과CSV: masking_per_file.csv',
        '목적함수CSV: objective_per_file.csv',
        '세부업데이트/감쇠/최종저장재측정: candidates/<조건>/<발화>/generation.json',
        '원본고정역치배열: thresholds/*.npz / threshold_manifest.json',
        '최종독립검증: independent_final_verification.json',
        '구현/근사설명: '+str(CODE/'METHOD_KO.txt'),
        '작업브랜치 work/psychoacoustic-spec-pilot-20261125.코드는그worktree에미커밋보존.기존연구/앱팀브랜치수정없음.',
        '기존입력/가중치/실험/백업삭제덮어쓰기없음.패키지설치/Commit/Push없음.실행중생성/학습/평가없음.']
    with (OUT/'REPORT_KO.txt').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS',guard95_passes=[by[c]['guard95_silence_pass_count'] for c in ('mask3db_iter10','mask6db_iter10')],
        noise_changes_db=changes,max_delta_codes=max(r['maximum_delta_codes'] for r in extrema),masking=own_stats_to_json(own_stats)),ensure_ascii=False,indent=2))


def own_stats_to_json(stats):return [dict(condition=c,margin_db=m,**r) for (c,m),r in stats.items()]


if __name__=='__main__':
    from runner import cli
    cli('report_and_verify')
