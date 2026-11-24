"""Copy 15 existing WAVs into a blind listening packet; no audio/model generation."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import time
import traceback
import wave
import zipfile

import numpy as np
import soundfile as sf

from workspace_paths import ROOT, LEGACY, STUDY, LISTENING as OUT
PUBLIC=OUT/'evaluator'
PRIVATE=OUT/'organizer_private'
FORMS=OUT/'forms_drafts'
CONDITIONS=('clean','eps2_iter10','silence_only_smooth')
WATCH={}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def watch(p,expected=None):
    p=Path(p);h=sha(p)
    if expected is not None:assert h==expected,(str(p),'hash mismatch')
    if str(p) in WATCH:assert WATCH[str(p)]==h
    WATCH[str(p)]=h
    return p


def read(p):return json.loads(watch(p).read_text(encoding='utf-8'))


def save(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def write(p,text):
    with Path(p).open('x',encoding='utf-8') as f:f.write(text)


def table(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def csvread(p):
    with watch(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def windows(p):
    value=str(Path(p))
    if value.startswith('/mnt/c/'):return 'C:\\'+value[len('/mnt/c/'):].replace('/','\\')
    return value


def db(a):return 20*math.log10(max(float(a),1e-15))


def riff_chunks(p):
    chunks=[]
    with Path(p).open('rb') as f:
        header=f.read(12);assert header[:4]==b'RIFF' and header[8:]==b'WAVE'
        while True:
            h=f.read(8)
            if not h:break
            assert len(h)==8
            label=h[:4].decode('ascii');size=struct.unpack('<I',h[4:])[0]
            chunks.append(dict(chunk=label,bytes=size))
            # Inventory original metadata privately; copies use a common PCM header.
            f.seek(size+(size%2),1)
    return chunks


def main():
    for directory in (PUBLIC,PRIVATE,FORMS):
        if directory.exists():assert not any(directory.iterdir()),'Refuse to overwrite an existing package'
        else:directory.mkdir()
    measured=csvread(STUDY/'analysis/per_file.csv')
    by={}
    for r in measured:
        if r['condition'] in CONDITIONS:by.setdefault(r['utterance_id'],{})[r['condition']]=r
    clean_dataset=read(LEGACY/'runs/paired_melo_experiment_ong85l9i/clean/dataset.json')
    clean_by={r['utterance_id']:r for r in clean_dataset['utterances']}
    candidate_dataset=read(STUDY/'plans/silence_only_smooth/dataset.json')
    candidate_by={r['utterance_id']:r for r in candidate_dataset['utterances']}
    settings=read(STUDY/'method.json')
    assert 'silence_only_smooth' in settings['variants']
    baseline=read(ROOT/'runs/demo_noise_fix_20261124/defense_attack_140/SPEC_eps2_iter10/quality.json')
    baseline_by={r['utterance_id']:r for r in baseline['per_file']}
    rechecked=csvread(STUDY/'defense_validation_20261124/quality_recheck.csv')
    rechecked_by={r['utterance_id']:r for r in rechecked if r['condition']=='silence_only_smooth'}
    assert read(STUDY/'defense_validation_20261124/final_verification.json')['status']=='PASS'
    eligible=sorted(uid for uid,variants in by.items() if set(variants)==set(CONDITIONS)
        and clean_by[uid]['split']=='train' and all(Path(r['path']).is_file() for r in variants.values()))
    assert len(eligible)==140
    # Select without using SNR/STOI/SIM/CER, improvement size, or listening feedback.
    selection_rng=random.Random(20261125)
    selected=selection_rng.sample(eligible,5)
    order_rng=random.Random(2026112501);order_rng.shuffle(selected)
    name_rng=random.Random(2026112502);used=set()
    records=[];playback=[];volumes=[];decode_results=[]
    source_time=1795532400 # identical presentation metadata on copies only
    for group,uid in enumerate(selected,1):
        original=clean_by[uid];variants=by[uid]
        assert sha(watch(variants['clean']['path'],variants['clean']['sha256']))==original['wav_sha256']
        assert variants['eps2_iter10']['sha256']==baseline_by[uid]['sha256']
        assert variants['silence_only_smooth']['sha256']==candidate_by[uid]['wav_sha256']==rechecked_by[uid]['sha256']
        meta_path=Path(variants['eps2_iter10']['path']).parent/'generation.json'
        gen=read(meta_path)
        assert gen['settings']['epsilon']==2 and gen['settings']['iters']==10 and gen['settings']['seed']==52
        assert sha(watch(gen['stats']['input']))==original['wav_sha256']
        assert gen['quality']['sha256']==variants['eps2_iter10']['sha256']
        clean_codes,sr=sf.read(original['wav'],dtype='int16')
        condition_order=list(CONDITIONS);order_rng.shuffle(condition_order)
        group_frames=[];group_hashes=[]
        for position,condition in enumerate(condition_order,1):
            source=watch(variants[condition]['path'],variants[condition]['sha256'])
            chunks=riff_chunks(source)
            code=f'clip_{name_rng.getrandbits(32):08X}'
            assert code not in used;used.add(code)
            target=PUBLIC/(code+'.wav')
            # Copy PCM bytes without resampling/requantization/gain changes. BWF and
            # other metadata can identify the Clean condition, so omit on every copy.
            with wave.open(str(source),'rb') as src:
                assert src.getsampwidth()==2 and src.getcomptype()=='NONE'
                pcm=src.readframes(src.getnframes())
                params=src.getparams()
            with target.open('xb') as stream:
                with wave.open(stream,'wb') as dst:
                    dst.setnchannels(params.nchannels);dst.setsampwidth(params.sampwidth)
                    dst.setframerate(params.framerate);dst.writeframes(pcm)
            os.utime(target,(source_time,source_time))
            with wave.open(str(target),'rb') as copied:
                pcm_copy=copied.readframes(copied.getnframes())
            assert pcm_copy==pcm
            assert [r['chunk'] for r in riff_chunks(target)]==['fmt ','data']
            info=sf.info(target);y,rate=sf.read(target,dtype='int16')
            assert info.subtype=='PCM_16' and info.channels==1 and rate==sr==44100 and len(y)==len(clean_codes)
            with wave.open(str(target),'rb') as wav:
                assert wav.getsampwidth()==2 and wav.getnchannels()==1 and wav.getnframes()==len(y)
            assert np.array_equal(y,sf.read(source,dtype='int16')[0])
            delta=y.astype(np.int32)-clean_codes.astype(np.int32)
            limit=0 if condition=='clean' else 257
            assert int(np.max(np.abs(delta)))<=limit
            a=y.astype(np.float64)/32768
            peak=float(np.max(np.abs(a)));rms=float(np.sqrt(np.mean(a*a)))
            clipping=int(np.sum((y==-32768)|(y==32767)));assert clipping==0
            decoded=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(target),'-f','null','-'],capture_output=True,text=True)
            assert decoded.returncode==0 and not decoded.stderr
            decode_results.append(dict(file=target.name,exit_code=decoded.returncode))
            play_order=len(playback)+1
            public=dict(play_order=play_order,group=group,position_in_group=position,file_code=code,
                filename=target.name,local_path=windows(target),audio_share_url='')
            playback.append(public)
            record=dict(**public,utterance_id=uid,text=original['text'],split='train',condition=condition,
                epsilon=0 if condition=='clean' else 2/255,iterations=0 if condition=='clean' else 10,
                postprocess='silence_only_smooth' if condition=='silence_only_smooth' else 'none',
                source_path=windows(source),source_sha256=sha(source),copied_sha256=sha(target),
                source_pcm_sha256=hashlib.sha256(pcm).hexdigest(),copied_pcm_sha256=hashlib.sha256(pcm_copy).hexdigest(),
                pcm_bytes_equal=True,metadata_removed_in_copy=True,whole_wav_bytes_equal=sha(source)==sha(target),
                original_path=windows(Path(original['wav'])),original_sha256=original['wav_sha256'],
                sample_rate=rate,bit_depth=16,channels=1,samples=len(y),seconds=len(y)/rate,
                peak=peak,peak_dbfs=db(peak),rms=rms,rms_dbfs=db(rms),clipping=clipping,
                max_delta_codes=int(np.max(np.abs(delta))),strict_epsilon_pass=True,riff_chunks=chunks,
                original_volume_preserved=True,ffmpeg_decode_pass=True)
            records.append(record);group_frames.append(len(y));group_hashes.append(sha(target))
        assert len(set(group_frames))==1 and len(set(group_hashes))==3
        cr=next(r for r in records if r['utterance_id']==uid and r['condition']=='clean')
        for r in records:
            if r['utterance_id']==uid:
                volumes.append(dict(group=group,utterance_id=uid,filename=r['filename'],condition=r['condition'],
                    seconds=r['seconds'],peak_dbfs=r['peak_dbfs'],rms_dbfs=r['rms_dbfs'],
                    rms_db_difference_vs_original=r['rms_dbfs']-cr['rms_dbfs'],
                    gain_applied_db=0,normalization=False,clipping=0))
    assert len(records)==15
    table(PRIVATE/'answer_mapping.csv',[{k:v for k,v in r.items() if k!='riff_chunks'} for r in records])
    save(PRIVATE/'answer_mapping.json',records)
    table(PRIVATE/'volume_length_audit.csv',volumes)
    table(FORMS/'audio_file_list.csv',playback)
    guide='''한국어 음성 청취 품질 평가 안내

이 패키지는 같은 문장을 읽은 음성의 듣기 품질을 비교하기 위한 것입니다.
파일에는 조건을 알 수 없는 코드만 붙어 있습니다. 파일 이름을 보고 조건을 추측하지 마세요.
평가용 음성 15개와 이 안내문만 들어 있습니다. 약 10~15분을 예상합니다.

1. 가능하면 조용한 장소에서 같은 이어폰 또는 헤드폰으로 들어 주세요.
2. 처음에 편안한 음량을 정한 뒤 모든 파일에서 기기 음량을 유지해 주세요.
3. 플레이어의 자동 음량 조절, 음장 효과, 잡음 제거 기능은 가능하면 꺼 주세요.
4. 아래 순서대로 듣고 설문에서 해당 파일 코드의 문항에 답해 주세요.
5. 한 묶음은 같은 발화의 파일 3개입니다. 필요하면 자유롭게 다시 들어도 됩니다.
6. 음성의 시작·끝과 말 사이의 쉼, 작은 목소리 구간도 함께 들어 주세요.
7. 내용의 이해 정도, 음성의 자연스러움, 거슬리는 잡음 정도를 각각 평가해 주세요.
8. 차이를 느끼지 못하면 그대로 답하고, 다른 평가자와 답을 상의하지 마세요.

평가 척도
잡음 거슬림: 1 전혀 거슬리지 않음 / 2 약간 / 3 보통 / 4 많이 / 5 매우 거슬림
자연스러움: 1 매우 부자연스러움 / 2 부자연스러움 / 3 보통 / 4 자연스러움 / 5 매우 자연스러움
내용 이해: 1 거의 이해 못함 / 2 일부만 이해 / 3 절반 정도 / 4 대부분 이해 / 5 모두 이해

기술 문제로 재생하지 못한 경우 낮은 점수로 대신 평가하지 말고 '재생 불가'로 기록해 주세요.
편안하지 않으면 청취를 중단해도 됩니다. 이름이나 연락처를 제출할 필요는 없습니다.
설문 주소는 평가 진행자가 별도로 안내합니다.

청취 순서
'''
    for group in range(1,6):
        guide+=f'\n묶음 {group}\n'
        for p in playback:
            if p['group']==group:guide+=f"{p['play_order']:02d}. {p['filename']}\n"
    write(PUBLIC/'README_KO.txt',guide)
    questions=[]
    def question(qid,group,code,text,kind,choices='',required=True):
        questions.append(dict(question_id=qid,group=group,file_code=code,question=text,response_type=kind,
            choices_or_scale=choices,required=required))
    question('consent','','','안내를 읽었으며 자발적으로 청취 평가에 참여하시겠습니까?','single_choice','동의함|동의하지 않음')
    question('participant_code','','','진행자가 안내한 익명 참가 코드를 입력해 주세요.','short_text')
    question('device','','','청취에 사용한 기기는 무엇입니까?','single_choice','이어폰|헤드폰|스피커')
    question('environment','','','청취 장소의 주변 소음은 어느 정도입니까?','single_choice','조용함|약간 있음|많음')
    question('constant_volume','','','평가 중 기기 음량을 유지할 수 있었습니까?','single_choice','유지함|일부 변경함|잘 모르겠음')
    for group in range(1,6):
        pp=[p for p in playback if p['group']==group]
        for p in pp:
            code=p['file_code']
            question(code+'_play',group,code,f'[{code}] 정상적으로 재생하여 들었습니까?','single_choice','정상 재생|재생 불가|청취하지 못함')
            question(code+'_noise',group,code,f'[{code}] 잡음이 얼마나 거슬렸습니까?','scale_1_to_5','1 전혀 거슬리지 않음|5 매우 거슬림',False)
            question(code+'_natural',group,code,f'[{code}] 목소리가 얼마나 자연스럽게 들렸습니까?','scale_1_to_5','1 매우 부자연스러움|5 매우 자연스러움',False)
            question(code+'_understand',group,code,f'[{code}] 말의 내용을 얼마나 이해할 수 있었습니까?','scale_1_to_5','1 거의 이해 못함|5 모두 이해',False)
        codes='|'.join(p['file_code'] for p in pp)
        question(f'group{group}_preference',group,'',f'[묶음 {group}] 전체적으로 가장 편안하게 들린 파일은 무엇입니까?','single_choice',codes+'|차이를 느끼지 못함|판단하기 어려움')
        question(f'group{group}_difference',group,'',f'[묶음 {group}] 세 파일 사이의 듣기 품질 차이를 느꼈습니까?','single_choice','분명한 차이|약한 차이|차이를 느끼지 못함|판단하기 어려움')
        question(f'group{group}_comment',group,'',f'[묶음 {group}] 거슬린 소리나 구간이 있다면 파일 코드와 함께 적어 주세요.','paragraph','',False)
    question('overall_comment','','','그 밖의 청취 의견이나 기술 문제를 적어 주세요.','paragraph','',False)
    table(FORMS/'question_draft.csv',questions)
    form_text='''Google Forms 구성용 초안 — 온라인 폼은 생성하지 않았습니다.

제목: 한국어 음성 청취 품질 평가
설명: evaluator/README_KO.txt의 안내와 척도를 사용하세요. 처리 조건, 실험 성능,
추천 후보 또는 원본이 어느 파일인지 설명하지 마세요.
audio_file_list.csv는 블라인드 코드/순서/로컬 경로 목록입니다.
audio_share_url 열은 비워 두었습니다. 실제 배포 시 평가자가 접근할 수 있는
음성 파일 공유 주소를 채우고, 각 파일 문항 설명에 코드·파일명·주소를 넣으세요.
CSV는 구성 참고자료이며 폼 자동 생성 또는 직접 가져오기를 보장하는 파일이 아닙니다.
기존 정답표와 원본 경로는 설문이나 공유 폴더에 올리지 마세요.

시작: 안내 및 자발적 참여 확인 -> 익명 코드 -> 재생 기기/환경/음량 유지 여부.
동의하지 않은 경우 평가를 진행하지 않도록 안내하세요.
묶음1~5: audio_file_list.csv 순서대로 각 파일을 제시하고 아래 문항을 반복하세요.
1) 정상 재생 여부. 재생 불가/미청취면 그 파일의 점수는 비워 두도록 안내.
2) 잡음 거슬림 1~5 (낮을수록 좋음).
3) 자연스러움 1~5 (높을수록 좋음).
4) 내용 이해 1~5 (높을수록 좋음).
각 묶음 후: 가장 편안한 코드/차이 없음/판단 어려움, 차이 체감 정도, 선택적 구간 의견.
종료: 선택적 전체 의견. 전체 문항과 코드가 question_draft.csv에 있습니다.
이름·이메일·전화번호는 요청하지 않으며 익명 코드로 중복 응답을 점검하세요.
공유 전 진행자가 실제 평가자 권한으로 음성 접근과 문항 코드 일치를 확인하세요.
평가 참여자가 직접 제출한 실제 응답은 아직 없습니다.
'''
    write(FORMS/'FORM_DRAFT_KO.txt',form_text)
    zip_path=OUT/'SafeSpeech_listening_20261125.zip'
    expected={p.name:p for p in PUBLIC.iterdir() if p.is_file()}
    assert len(expected)==16 and sum(n.endswith('.wav') for n in expected)==15
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        # Equal timestamps hide generation chronology; bytes remain unchanged.
        for name in ['README_KO.txt']+[p['filename'] for p in playback]:
            zi=zipfile.ZipInfo(name,date_time=(2026,11,25,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED
            zi.external_attr=0o100644<<16;z.writestr(zi,expected[name].read_bytes())
    members=[]
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None and set(z.namelist())==set(expected)
        assert all('/' not in n and '\\' not in n for n in z.namelist())
        for info in z.infolist():
            data=z.read(info.filename);h=hashlib.sha256(data).hexdigest();assert h==sha(expected[info.filename])
            members.append(dict(name=info.filename,bytes=info.file_size,crc32=f'{info.CRC:08x}',sha256=h))
    changed=[p for p,h in WATCH.items() if sha(p)!=h];assert not changed
    save(PRIVATE/'source_hashes.json',WATCH)
    save(PRIVATE/'collection_audit.json',dict(selection_seed=20261125,order_seed=2026112501,filename_seed=2026112502,
        selection='Uniform random sample of five from sorted 140 complete train utterances before quality/result inspection; no regeneration or substitution.',
        common_utterance_count=len(eligible),selected_utterance_ids=selected,conditions=CONDITIONS,
        missing=[],source_files_changed=changed,formal_human_listening_completed=False,
        sample_scope='Existing KSS train recordings only; not new held-out recordings or a population audibility proof.',
        gain_normalization=False,all_sample_equal_to_sources=True,all_ffmpeg_decode_pass=True,
        all_matching_lengths_per_utterance=True,all_strict_pcm16_epsilon_pass=True,all_clipping_zero=True,
        metadata_side_channel='Original BWF metadata inventoried privately; copies canonical fmt/data only, exact source PCM bytes retained. Copy and ZIP timestamps unified.',
        distribution='One fixed randomly ordered packet, shared order across raters; no per-rater counterbalancing.'))
    write(PRIVATE/'ORGANIZER_NOTES_KO.txt','''진행자 전용 — 정답표를 평가자에게 배포하지 마세요.
세 조건은 Clean 원본, 기존 SPEC eps2/255 10회, 같은 기준 섭동의 silence_only_smooth입니다.
5개 발화는 기존 train140 공통 파일 중 seed20261125로 균등 무작위 선정했습니다.
결과가 좋아 보이는 파일만 고르지 않았습니다. 새 음성 생성이나 학습은 없습니다.
PCM 음성 바이트와 재생 레벨은 원본 그대로이며 볼륨 차이는 volume_length_audit.csv에 기록했습니다.
원본 일부의 BWF 메타데이터는 조건 단서를 숨기기 위해 평가용 사본에서만 제거했습니다.
따라서 전체 WAV 해시는 달라질 수 있으나 PCM 바이트와 PCM SHA-256은 원본과 같습니다.
평가자에게는 ZIP만 배포하고 정답표·실험 수치·추천 조건을 공개하지 마세요.
Forms 초안/목록에는 정답이 없으나 URL은 진행자가 배포 전에 채워야 합니다.
question_draft.csv는 참고용 초안입니다. 온라인 Google Form 생성/업로드는 하지 않았습니다.
재생 불가를 저품질 점수로 처리하지 말고 결측으로 구분하세요. 실제 미응답을 생성하지 마세요.
현재 평가 응답 0명입니다. 블라인드 준비 완료를 청취 검증 완료로 보고하지 마세요.
현재 한 고정 순서의 패킷이므로 순서 효과와 이전 파일 기억의 영향을 기록하세요.
사전 평가 기준·모집 인원·분석 방식을 실제 응답 확인 전에 정하고,
평가자와 같은 발화에 반복 측정한 응답의 짝을 유지해 분석하세요.
차이의 통계적 유의성 부재만으로 불가청/동등성을 주장하지 마세요.
기존 KSS 단일 화자/학습 발화5개의 제한된 청취 결과임을 명시하세요.
''')
    winroot=windows(ROOT)
    git=lambda *a:subprocess.check_output(['/mnt/c/Program Files/Git/cmd/git.exe','-C',winroot,*a],text=True).strip()
    branch=git('branch','--show-current');head=git('rev-parse','HEAD');status=git('status','--porcelain')
    assert branch=='research/perceptual-spec-20261124' and head=='954a06cde13065074328bc9476eaf8238845d1ea' and not status
    audit=dict(status='BLIND_PACKET_READY_HUMAN_RESPONSES_PENDING',common_utterances=140,selected_utterances=5,wav_count=15,
        zip=windows(zip_path),zip_bytes=zip_path.stat().st_size,zip_sha256=sha(zip_path),zip_crc_all_pass=True,
        zip_members=members,zip_file_count=len(members),mapping_csv=windows(PRIVATE/'answer_mapping.csv'),
        mapping_json=windows(PRIVATE/'answer_mapping.json'),forms_file_list=windows(FORMS/'audio_file_list.csv'),
        forms_question_draft=windows(FORMS/'question_draft.csv'),
        existing_source_hash_changes=changed,missing=[],decode_checks=decode_results,
        branch=branch,head=head,git_status=status,no_commit_push=True,no_training_or_spec_generation=True,
        google_form_created=False,human_listening_completed=False)
    save(OUT/'package_verification.json',audit)
    print(json.dumps(dict(status=audit['status'],selected=selected,zip=audit['zip'],zip_bytes=audit['zip_bytes'],
        zip_sha256=audit['zip_sha256'],wav_count=15,zip_file_count=16,missing=[],source_files_changed=changed),ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (OUT/('failure_'+str(time.time_ns())+'.json')).open('x',encoding='utf-8') as f:
            json.dump(dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()),f,ensure_ascii=False,indent=2)
        raise
