"""Collect existing WAVs only; never import SPEC or training/model code."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import shutil
import subprocess
import wave
import zipfile
import numpy as np
import soundfile as sf

OUT=Path(__file__).resolve().parent
from research_bootstrap import setup
CODE_ROOT,ROOT,OLD=setup()
PRIVATE=OUT/'organizer_private'
PUBLIC=OUT/'evaluator'
SPECS=[('clean',0.,0,0.),('spec8_10_strict',8.,10,0.),('spec2_10',2.,10,0.),
       ('spec1_5',1.,5,0.),('spec2_10_perceptual05',2.,10,.5)]
WATCH={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def watch(p):
    p=Path(p)
    WATCH.setdefault(str(p),sha(p));return p

def read(p):return json.loads(watch(p).read_text(encoding='utf-8'))

def save(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def table(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)

def pcm(p):
    with wave.open(str(p),'rb') as f:
        params=f.getparams();raw=f.readframes(f.getnframes())
    assert params.nchannels==1 and params.sampwidth==2 and params.framerate==44100 and params.comptype=='NONE'
    assert len(raw)==params.nframes*2
    a,sr=sf.read(p,dtype='int16')
    assert sr==44100 and np.array_equal(a,np.frombuffer(raw,dtype='<i2'))
    return params,raw,a

def main():
    PRIVATE.mkdir(exist_ok=False);PUBLIC.mkdir(exist_ok=False)
    (PRIVATE/'source_copies').mkdir()
    dataset=read(OLD/'runs/paired_melo_experiment_ong85l9i/clean/dataset.json')['utterances']
    handoff=read(OLD/'runs/korean_melo_spec_mqxs1q1s/handoff_manifest.json')['utterances']
    old10={r['utterance_id']:r for r in handoff if r['iterations']==10}
    corrected_table=watch(OLD/'runs/final_validation_20261119/strict_corrected_inputs/quality.csv')
    with corrected_table.open(encoding='utf-8-sig') as f:corrected={r['utterance_id']:r for r in csv.DictReader(f)}
    roots={
        'spec2_10':ROOT/'runs/demo_noise_fix_20261124/defense_attack_140/SPEC_eps2_iter10/protected_train',
        'spec1_5':ROOT/'runs/perceptual_spec_20261124/attacks/eps1_iter5/protected_train',
        'spec2_10_perceptual05':ROOT/'runs/perceptual_spec_20261124/attacks_followup/mask_eps2_iter10_s05/protected_train'}
    eligible=[];inventory=[];missing=[]
    for row in sorted(dataset,key=lambda r:r['utterance_id']):
        uid=row['utterance_id']
        paths={'clean':Path(row['wav'])}
        paths.update({c:d/uid/'protected.wav' for c,d in roots.items()})
        if uid in corrected:paths['spec8_10_strict']=Path(corrected[uid]['output'])
        absent=[c for c,e,n,s in SPECS if c not in paths or not paths[c].is_file()]
        absent.extend(c+'_generation_metadata' for c,d in roots.items() if not (d/uid/'generation.json').is_file())
        if uid not in old10:absent.append('spec8_original_generation_link')
        inventory.extend(dict(utterance_id=uid,condition=c,source=str(paths.get(c,'')),exists=c in paths and paths[c].is_file()) for c,e,n,s in SPECS)
        if not absent:eligible.append((row,paths))
        else:missing.append(dict(utterance_id=uid,missing=absent))
    # Fixed filename ordering, before listening/quality inspection; no quality cherry-picking.
    chosen=eligible[:3]
    if len(chosen)!=3:
        save(PRIVATE/'inventory_incomplete.json',dict(common_count=len(eligible),missing=missing))
        raise RuntimeError('Fewer than three complete utterances; no replacements or generation permitted')
    table(PRIVATE/'existing_inventory.csv',inventory)
    rows=[]
    for row,paths in chosen:
        uid=row['utterance_id'];clean=watch(paths['clean']);clean_sha=sha(clean)
        assert clean_sha==row['wav_sha256']==old10[uid]['clean_sha256']==old10[uid]['original_sha256']
        assert sha(watch(old10[uid]['clean_path']))==clean_sha
        params,raw,clean_codes=pcm(clean)
        for condition,epsilon,iters,strength in SPECS:
            source=watch(paths[condition]);metadata=[]
            if condition in roots:
                meta_path=roots[condition]/uid/'generation.json';meta=read(meta_path);metadata.append(str(meta_path))
                args=meta['settings'];q=meta['quality']
                assert (args['epsilon'],args['iters'],args['mode'],args.get('perceptual_strength',0.))==(epsilon,iters,'SPEC',strength)
                assert args['random_init'] is False and args['seed']==52
                input_path=watch(meta['stats']['input'])
                assert sha(input_path)==clean_sha and input_path.resolve()==clean.resolve()
                assert Path(q['path']).resolve()==source.resolve() and q['sha256']==sha(source)
            elif condition=='spec8_10_strict':
                r=corrected[uid];original=old10[uid]
                assert original['epsilon']==8/255 and original['iterations']==10
                validation=read(Path(original['validation_path']));metadata.append(original['validation_path'])
                assert validation['iterations']==10 and validation['epsilon_units_255']==8
                assert validation['expected_original_sha256']==clean_sha
                assert sha(watch(original['protected_path']))==original['protected_sha256']==r['source_sha256']
                assert sha(source)==r['sha256'] and r['strict_epsilon_pass']=='True'
                _,_,old_codes=pcm(original['protected_path'])
                expected=np.clip(old_codes.astype(np.int32),np.maximum(clean_codes.astype(np.int32)-1028,-32768),
                    np.minimum(clean_codes.astype(np.int32)+1028,32767)).astype(np.int16)
                assert np.array_equal(pcm(source)[2],expected)
                metadata.append(str(corrected_table))
            par,data,codes=pcm(source)
            assert par==params and codes.shape==clean_codes.shape
            delta=np.max(np.abs(codes.astype(np.int32)-clean_codes.astype(np.int32)))/32768
            assert delta<=epsilon/255
            stored=PRIVATE/'source_copies'/f'{uid}_{condition}.wav'
            shutil.copyfile(source,stored);assert sha(stored)==sha(source)
            rows.append(dict(utterance_id=uid,text=row['text'],condition=condition,epsilon_units_255=epsilon,
                iterations=iters,perceptual_strength=strength,seed=52 if epsilon else None,source_path=str(source),
                original_source_sha256=sha(source),clean_sha256=clean_sha,metadata_paths=metadata,
                private_exact_copy=str(stored),sample_rate=44100,bit_depth=16,channels=1,samples=par.nframes,
                duration_seconds=par.nframes/44100,strict_linf=float(delta),strict_epsilon_pass=True,
                pcm_sha256=hashlib.sha256(data).hexdigest(),clipping_samples=int(np.count_nonzero((codes==-32768)|(codes==32767)))))
    order_seed=secrets.randbits(128);random.Random(order_seed).shuffle(rows)
    for i,r in enumerate(rows,1):
        name=f'{i:02d}_{secrets.token_hex(6)}.wav';dest=PUBLIC/name
        p,data,codes=pcm(r['source_path'])
        # Canonical PCM container only. Never normalize/filter/resample audio.
        with dest.open('xb') as stream:
            with wave.open(stream,'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(44100);w.writeframes(data)
        assert pcm(dest)[1]==data
        b=dest.read_bytes();assert b[:4]==b'RIFF' and b[8:12]==b'WAVE' and b[12:16]==b'fmt ' and b[36:40]==b'data'
        assert len(b)==44+len(data)
        dec=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(dest),'-f','null','-'],capture_output=True,text=True)
        assert dec.returncode==0,dec.stderr
        os.utime(dest,(1795478400,1795478400))
        r.update(evaluation_order=i,blind_filename=name,public_sha256=sha(dest),
            pcm_identical_to_source=True,ffmpeg_decode_pass=True,container_metadata_removed=True)
    names=[r['blind_filename'] for r in rows]
    guide='''블라인드 음성 청취 평가 안내

이 폴더에는 한국어 음성 파일 15개가 있습니다. 파일명 순서대로 듣고, 가능하면 같은 장치와 같은 재생 음량을 유지해 주세요. 반복 청취는 가능합니다. 각 파일에 맞춰 볼륨을 개별 조절하지 마세요.

파일별로 다음 내용을 메모하거나 별도 설문에 기록해 주세요.
- 잡음 정도: 1 거의 없음 ~ 5 매우 심함
- 자연스러움: 1 매우 부자연스러움 ~ 5 매우 자연스러움
- 말소리 명료도: 1 매우 불명확 ~ 5 매우 명확
- 들린 잡음의 특징과 거슬리는 구간: 자유 서술
설정이나 처리 방법을 추측하기보다 실제로 느낀 음질을 평가해 주세요. 음질 평가는 음성 복제 방어 성능 평가가 아닙니다. 상세 처리 정보와 매핑은 평가 완료 후 별도로 공개합니다.

청취 순서/기록용 파일명:
'''+ '\n'.join(names)+'''

출처: KSS Dataset, Kyubyong Park. CC BY-NC-SA 4.0.
https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
비상업적 내부 연구용 청취 자료입니다. 연구 목적으로 처리된 음성이 포함되며, 평가 패키지에서는 파일명과 WAV 컨테이너 메타데이터를 정리했습니다. 원래 샘플레이트·음량·PCM 샘플은 해당 수집 음성과 동일합니다. 청취 평가자의 이름 등 개인정보를 파일명에 덧붙이지 마세요.
'''
    (PUBLIC/'LISTENING_GUIDE_KO.txt').write_text(guide,encoding='utf-8-sig')
    os.utime(PUBLIC/'LISTENING_GUIDE_KO.txt',(1795478400,1795478400))
    save(PRIVATE/'answer_mapping.json',dict(randomization_seed=str(order_seed),records=rows))
    table(PRIVATE/'answer_mapping.csv',[{**r,'metadata_paths':json.dumps(r['metadata_paths'],ensure_ascii=False)} for r in rows])
    save(PRIVATE/'collection_audit.json',dict(selected_utterance_ids=[r['utterance_id'] for r,p in chosen],
        selection='First three utterance IDs with all five existing conditions, sorted before quality/listening inspection.',
        common_utterance_count=len(eligible),selected_missing=[],inventory_missing=missing,
        condition_counts={c:sum(r['condition']==c for r in rows) for c,e,n,s in SPECS},
        playback_scope='WAV parsing, independent SoundFile PCM equality and full FFmpeg decode; no speaker-device/human listening claim',
        no_spec_generation_or_model_training=True,normalization=False,filters=False,existing_audio_modified=False))
    archive=OUT/'Listening_Evaluation_15_WAV.zip'
    public_files=sorted(PUBLIC.iterdir());assert len(public_files)==16 and len(list(PUBLIC.glob('*.wav')))==15
    with archive.open('xb') as raw:
        with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for file in public_files:
                entry=zipfile.ZipInfo(file.name,date_time=(2026,11,24,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
                entry.external_attr=0o100644<<16;z.writestr(entry,file.read_bytes())
    archive_sha=sha(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())=={p.name for p in public_files}
        assert len(z.infolist())==16 and not z.comment
        for file in public_files:assert hashlib.sha256(z.read(file.name)).hexdigest()==sha(file)
        assert not any(any(t in name.lower() for t in ('mapping','answer','spec','clean','epsilon','kss','json','csv')) for name in z.namelist())
    assert all(sha(p)==h for p,h in WATCH.items())
    (OUT/'Listening_Evaluation_15_WAV.zip.sha256').write_text(archive_sha+'  '+archive.name+'\n',encoding='ascii')
    save(PRIVATE/'source_hashes.json',WATCH)
    save(OUT/'verification.json',dict(status='PASS',archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=archive_sha,
        wav_count=15,guide_count=1,zip_entries=16,all_crc_pass=True,all_zip_member_sha256_match=True,
        decoded_wavs=15,all_pcm_samples_identical_to_sources=True,all_conditions_complete=True,
        common_utterance_count=len(eligible),selected_utterance_count=3,missing_selected=[],
        mapping_outside_evaluator_folder=True,mapping_excluded_from_zip=True,metadata_and_zip_timestamps_uniform=True,
        watched_sources_unchanged=len(WATCH),new_spec_or_training=False,git_commit_or_push=False,
        organizer_mapping=str(PRIVATE/'answer_mapping.csv')))
    print(json.dumps(dict(status='PASS',common=len(eligible),selected=[r['utterance_id'] for r,p in chosen],wav_count=15,
        zip_entries=16,zip_bytes=archive.stat().st_size,zip_sha256=archive_sha,archive=str(archive)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    from runner import cli
    cli('package_existing')
