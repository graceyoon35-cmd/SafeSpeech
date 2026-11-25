"""Read-only final verification and negative-result report. No extra experiments."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import soundfile as sf

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/'runs/psychoacoustic_spec_pilot_20261125'
WT=OLD/'code_worktree'
CODE=WT/'scripts/psychoacoustic_spec_pilot'
from research_bootstrap import setup
CODE_ROOT,ROOT,LEGACY=setup()
OUT=ROOT/'runs/psychoacoustic_spec_refinement_20261126'
OLD=ROOT/'runs/psychoacoustic_spec_pilot_20261125'
WT=OLD/'code_worktree'
CODE=CODE_ROOT/'scripts/psychoacoustic_spec_pilot'
from auditory_masking import build_model,audit


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def save(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def git(p,*args):
    win=subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
    return subprocess.check_output(['/mnt/c/Program Files/Git/cmd/git.exe','-C',win,*args],text=True).strip()


def main():
    before=read(OUT/'preservation_before.json');archive=Path(before['archive'])
    assert sha(archive)==before['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for e in before['entries']:
            assert sha(e['source'])==e['sha256']==hashlib.sha256(z.read(e['archive_path'])).hexdigest()
    for phase in ('stage_diagnosis','comparison'):assert read(OUT/phase/'complete.json')['status']=='COMPLETE'
    assert read(OUT/'comparison/complete.json')['improvement_variants']==2
    source_versions=[]
    for phase in ('stage_diagnosis','comparison'):
        preserved=read(OUT/phase/'preservation.json');assert not preserved['changed']
        for p,h in preserved['source_sha256'].items():
            if sha(p)!=h:
                assert phase=='stage_diagnosis' and Path(p).name=='run_refinement.py'
                snapshot=OUT/'run_refinement_diagnosis_version.py';assert sha(snapshot)==h
                source_versions.append(dict(phase=phase,source=p,archived=str(snapshot),sha256=h,
                    reason='Post-diagnosis summary now tolerates missing SNR for zero residual; projection/SPEC logic unchanged'))
    summary=read(OUT/'comparison/summary.json');by={r['condition']:r for r in summary}
    q=rows(OUT/'comparison/quality_per_file.csv');assert len(q)==15
    checked=[]
    for uid in read(OLD/'pilot/complete.json')['ids']:
        saved=np.load(OLD/'pilot/thresholds'/(uid+'.npz'))
        gen=read(OUT/'comparison/anchor_backtrack'/uid/'generation.json')
        x,sr=sf.read(gen['source'],dtype='int16');assert sha(gen['source'])==gen['source_sha256']
        models=build_model(x.astype(np.float64)/32768)
        for m in models:np.testing.assert_array_equal(m.raw_threshold,saved[f'raw_threshold_{m.n_fft}'])
        for r in [r for r in q if r['id']==uid]:
            p=Path(r['path']);assert sha(p)==r['sha256']
            y,ysr=sf.read(p,dtype='int16');assert sr==ysr==44100 and y.shape==x.shape
            assert sf.info(p).subtype=='PCM_16'
            delta=y.astype(np.int32)-x.astype(np.int32)
            assert int(abs(delta).max())<=257 and np.all(abs(delta)<=saved['radius_codes'])
            assert not np.any((y==-32768)|(y==32767))
            checks=audit(delta/32768.,models,3);assert all(a['pass_constraint'] for a in checks)
            decoded=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(p),'-f','null','-'],capture_output=True)
            assert decoded.returncode==0
            checked.append(dict(id=uid,condition=r['condition'],sha256=sha(p),max_delta_codes=int(abs(delta).max()),masking=checks))
    replay=read(OUT/'stage_diagnosis/baseline_reproduction.json')
    replay_obj=rows(OUT/'stage_diagnosis/objective_per_file.csv')
    replay_retained=float(np.mean([float(r['retained_vs_eps2']) for r in replay_obj]))
    stage=rows(OUT/'stage_diagnosis/stage_measurements.csv')
    stage_summary=read(OUT/'stage_summary.json')['stage_summary']
    mask_diagnostics=[]
    for name in dict.fromkeys(r['stage'] for r in stage):
        a=[r for r in stage if r['stage']==name]
        mask_diagnostics.append(dict(stage=name,proposals=len(a),violating_proposals=sum(int(r['violating_bins'])>0 for r in a),
            mean_max_excess_db=float(np.mean([float(r['max_excess_db']) for r in a])),max_excess_db=max(float(r['max_excess_db']) for r in a)))
    old=by['old_mask3'];improved=[]
    for name in ('anchor_backtrack','local_caps_anchor'):
        r=by[name]
        if r['mean_retained_vs_eps2']>old['mean_retained_vs_eps2'] and not r['masking_violating_bins'] and not r['epsilon_failures']:
            improved.append(name)
    assert not improved,'Report interpretation must be revisited if actual improvement exists'
    assert git(WT,'branch','--show-current')==before['branch'] and git(WT,'rev-parse','HEAD')==before['head']
    assert git(ROOT,'for-each-ref','--format=%(refname) %(objectname)','refs/heads','refs/remotes')==before['refs_before']
    assert not git(ROOT,'status','--porcelain','--untracked-files=all')
    assert not git(WT,'diff','--name-only') and not git(WT,'diff','--cached','--name-only')
    old_names={line[3:] for line in before['worktree_status_before'].splitlines()}
    new_status=git(WT,'status','--porcelain','--untracked-files=all')
    new_names={line[3:] for line in new_status.splitlines()}
    added=sorted(new_names-old_names);assert not old_names-new_names
    processes=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True)
    active=[r for r in processes.splitlines() if any(s in r for s in ('run_refinement.py','train_melo_bounded.py','evaluate_paired_melo_smoke.py'))]
    assert not active
    result=dict(status='COMPLETE_NO_IMPROVED_CONFIGURATION',summary=summary,stage_summary=stage_summary,
        projection_mask_diagnostics=mask_diagnostics,replay_bit_identical_count=sum(r['pcm_equal'] for r in replay),
        replay_mean_retained_vs_eps2=replay_retained,replay_sensitivity=replay,
        actual_improvement_configs_executed=2,extra_followup_experiments=0,improved_configurations=improved,
        original_comparison_is_immutable_historical_wav=True,spec_metric_is_not_defense_efficacy=True,
        human_listening_verified=False,no_140_expansion=True,no_melo_training=True,no_sim_cer=True,
        before_archive_sha256=sha(archive),before_archive_crc_pass=True,prior86_files_unchanged=True,
        checked_audio=checked,intended_source_version_archives=source_versions,
        branch=before['branch'],head=before['head'],new_uncommitted_source_files=added,
        prior_source_files_modified=[],source_status=new_status,no_commit_push=True,active_processes=active)
    save(OUT/'results.json',result)
    lines=['SafeSpeech 3dB 투영 개선 최종 소규모 결과 — 2026-11-26',
        '', '결론: 개선 설정 없음. 두 개선안의 SPEC 목적 보존은 기존보다 모두 낮았다. 추가 실험을 반복하지 않고 종료했다.',
        '기존3dB파형은 그대로 비교기준으로 유지했다. 실제복제방어를측정하지않았으므로방어성능개선/저하로단정하지않는다.',
        '', '1. 보존',
        '원래미커밋소스5개와기존실험81파일을 새ZIP에보존. CRC/SHA/원본파일해시전후일치.',
        str(archive), 'ZIP SHA256: '+sha(archive),
        '기존데이터/WAV/체크포인트/백업/앱팀브랜치의수정·삭제·이동없음.',
        '', '2. 비교 조건과 두 개선안',
        '동일KSS train5개,epsilon2/255,3dB역치,2048/1024,동일시간별epsilon반경,seed52,beta10,10회.',
        'A anchor_backtrack: 누적섭동전체를줄이는대신현재feasible PCM쪽으로제안step을되돌림.',
        'B local_caps_anchor: 위반시간·주파수bin만최대8pass국소보정후feasible anchor로되돌림.',
        '고정한두설정각각1회씩만5개에실행했다.추가하이퍼파라미터탐색/반복/대규모학습없음.',
        '', '3. 실제 품질과목적보존: 같은5발화평균',
        '조건 | SNR dB | STOI | 무음RMS | 약한발화RMS | 강한발화RMS | SPEC 개선량보존(기존eps2대비) | 평균L_SPEC']
    for r in summary:
        snr='해당없음' if r['snr_db'] is None else f"{r['snr_db']:.5f}"
        lines.append(f"{r['condition']} | {snr} | {r['stoi']:.8f} | {r['silence_noise_rms']:.9f} | {r['weak_noise_rms']:.9f} | {r['voiced_noise_rms']:.9f} | {100*r['mean_retained_vs_eps2']:.5f}% | {r['mean_loss_SPEC']:.6f}")
    lines+=['SPEC은작을수록생성목적최적화가강하다. 개선량보존율은(Lclean-Lcandidate)/(Lclean-Leps2)의발화별평균이다.',
        '두개선안은각각5/5발화에서기존3dB보다손실이커졌다. 순수Clean으로대체된출력은0개다.',
        '음질수치가높아진것은섭동이더작아진결과이며보호성공이아니다.실제청취검수도하지않았다.',
        '최종15개(기존5+신규10) 모두clipping0,엄격PCM16차<=257codes,시간별box통과.',
        '3dB2048/1024마스킹위반0bin/0%,최대초과0dB.원본대응/길이/SR/mono/재생검사통과.',
        '', '4. 단계별 원인:50개갱신제안을짝지은진단 재실행',
        '양의손실증가는SPEC생성목적을약화한다.중간단계는보호후보가아닌진단용값이다.',
        '단계 | 평균SPEC손실 | 직전단계대비증가 | 위반제안수/50 | 평균최대역치초과dB']
    md={r['stage']:r for r in mask_diagnostics}
    for r in stage_summary:
        name=r['stage'];cost=r.get('mean_loss_increase_from_previous_stage')
        lines.append(f"{name} | {r['mean_loss_SPEC']:.6f} | {cost if cost is not None else '해당없음'} | {md[name]['violating_proposals']}/50 | {md[name]['mean_max_excess_db']:.6f}")
    lines+=['가장큰비용은첫연속투영의+875.18이다.시간box/두STFT역치/내부1dB여유가함께포함돼각각의기여분은분리하지않았다.',
        '첫투영뒤에도47/50제안이마스킹을위반했고,추가8cycle후45/50이위반했다.즉양자화전부터잔여위반이있었다.',
        '단순rounding의평균목적영향은-0.119로작다.그러나위반제안수는50/50으로늘어정수grid보정은필요했다.',
        '추가reserve보정+13.21,최종gain/PCM+29.15도목적을약화하지만첫투영비용보다훨씬작았다.',
        '고정회수STFTclip/역변환/시간box는동시제약의수렴을보장하지않는다.현재최종PCM심사는필수다.',
        '두개선안은기본제안의잔여위반을해결하지못해작은step만수락하고정체되는경향이었다.',
        '', '5. 재현성 제한과 실패 이력',
        '첫진단은과거PCM완전일치검사에서중단됐다. failure_*.json과초기코드를보존했다.',
        '첫발화완료파일을재생성하지않고나머지4개진단만이어갔다.기존파일을대체하지않았다.',
        f"과거PCM과완전일치한진단은{sum(r['pcm_equal'] for r in replay)}/5.재실행기준보존율평균은{replay_retained*100:.5f}%다(기존10.09835%).",
        '수치변동이부호갱신/양자화에증폭됐을가능성은있으나근본원인을확정하지않았다.',
        '비교표는실제고정된과거WAV를직접다시측정한기준이다.진단수치는과거파형의exact replay라고주장하지않는다.',
        '새CPU단위검사3개는통과했다.이는GPU/과거실행의bitwise재현성을보장하지않는다.',
        '', '6. 종료 판단',
        '이번두개선안은채택하지않는다.140개확대/1000step학습/앱연동을진행하지않았다.',
        '근거는실제방어평가가아니라품질·제약·생성목적결과다.현재3dB도목적보존문제가해결된설정이아니다.',
        '개선실패와투영/재현성한계를연구결과로보존한다.추가실험을반복하지않는다.',
        '', '7. 변경 파일 — 모두미커밋',*added,
        '기존미커밋5파일의내용은그대로다.기존tracked수정/삭제와staging은없다.',
        '작업브랜치: '+before['branch'],'HEAD: '+before['head'],
        'Commit/Push없음.기존연구/master/앱팀/원격참조모두변경없음.',
        '', '8. 결과 경로',
        '최종측정: '+str(OUT/'comparison/summary.csv'),
        '발화별측정: '+str(OUT/'comparison/quality_per_file.csv'),
        '발화별목적: '+str(OUT/'comparison/objective_per_file.csv'),
        '역치검사: '+str(OUT/'comparison/masking_per_file.csv'),
        '단계별원시값: '+str(OUT/'stage_diagnosis/stage_measurements.csv'),
        '단계요약: '+str(OUT/'stage_summary.csv'),
        '재현성기록: '+str(OUT/'stage_diagnosis/baseline_reproduction.json'),
        '후보WAV: comparison/anchor_backtrack/<발화ID>/protected.wav 및 comparison/local_caps_anchor/<발화ID>/protected.wav',
        '전체정리: results.json / changed_files.json / preservation_before.json',
        '현재실행중인생성·학습·평가프로세스없음.']
    with (OUT/'REPORT_KO.txt').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    created=[dict(relative_path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file()]
    save(OUT/'changed_files.json',dict(new_source_files=added,prior_source_files_modified=[],
        source_worktree=str(WT),new_result_files=created,self_not_included=True,no_commit_push=True))
    print(json.dumps(dict(status=result['status'],improved_configurations=improved,
        old_files_unchanged=before['source_file_count'],new_source_files=added,
        replay_mean_retained_vs_eps2=replay_retained,max_new_codes=max(r['max_delta_codes'] for r in checked if r['condition']!='old_mask3'),
        source_versions=source_versions,active_processes=active),ensure_ascii=False,indent=2))


if __name__=='__main__':
    from runner import cli
    cli('report_results')
