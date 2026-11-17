"""Held-out synthesis, ECAPA cosine SIM and Korean Whisper CER for smoke only."""
import argparse
import csv
import gc
import json
from pathlib import Path
import re
import tempfile

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio

from melo_training_core import infer, model_from_config, setup
from run_korean_melo_spec import ROOT, save, sha256
from evaluation_statistics import statistics


def normalize(text):
    return re.sub(r'[^0-9A-Za-z가-힣]', '', text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-summary',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--evaluation-models',type=Path,default=Path('/home/safespeech/models/melo-evaluation-20261118'))
    p.add_argument('--evaluation-manifest',type=Path)
    args=p.parse_args()
    trained=json.loads(args.training_summary.read_text())
    if trained['status'] not in ('PAIRED_SYNTHETIC_TRAINING_SMOKE_PASS','PAIRED_BOUNDED_TRAINING_PASS'):
        raise ValueError('Complete paired training required')
    experiment=Path(trained['experiment'])
    cfg=json.loads((experiment/'clean/config.json').read_text())
    data=json.loads((experiment/'clean/dataset.json').read_text())['utterances']
    tests=[r for r in data if r['split']=='test']
    base_seed=2026
    if args.evaluation_manifest:
        prompts=json.loads(args.evaluation_manifest.read_text())
        if prompts.get('schema_version')!=1 or Path(prompts['training_experiment']).resolve()!=experiment.resolve():
            raise ValueError('Evaluation prompts belong to a different experiment')
        tests=prompts['prompts'];base_seed=prompts['base_seed']
        if len({r['utterance_id'] for r in tests})!=len(tests) or len({normalize(r['text']) for r in tests})!=len(tests):
            raise ValueError('Duplicate evaluation ID/text')
        if {normalize(r['text']) for r in tests} & {normalize(r['text']) for r in data if r['split']!='test'}:
            raise ValueError('Evaluation/training text overlap')
    refs=[r for r in data if r['split']=='train'][:5]
    if not tests or not refs:
        raise ValueError('Held-out test and training references required')
    setup(args.assets)
    torch.set_num_threads(4)
    folder=Path(tempfile.mkdtemp(prefix='melo_smoke_evaluation_',dir=ROOT/'runs'))
    save(folder/'plan.json',{'training_summary':str(args.training_summary.resolve()),'scope':trained['scope'],
        'test_utterances':len(tests),'conditions':['pretrained']+[r['condition'] for r in trained['training']]})
    status={'status':'FAIL','scope':trained['scope'],'test_utterances':len(tests),
        'train_reference_utterances':len(refs),'training_summary':str(args.training_summary.resolve()),
        'limitations':'Synthetic benchmark, one training seed. Not KSS or human defense-efficacy evidence.',
        'steps_per_condition':trained['steps_per_condition'],'base_evaluation_seed':base_seed,
        'evaluation_manifest_sha256':sha256(args.evaluation_manifest) if args.evaluation_manifest else None,
        'interval_scope':'95% bootstrap over prompts only; not variability across speakers or training seeds.',
        'sim_method':'Mean ECAPA cosine against fixed clean training references; no success threshold.',
        'cer_method':'Whisper small multilingual, language ko; remove whitespace and punctuation.'}
    if trained['scope']=='kss_noncommercial_research':
        status['limitations']='KSS single-speaker noncommercial research; listening unreviewed, one training seed. No population generalization.'
        status['manual_listening_verified']=False
        status['separate_speaker_consent']='not_verified'
    try:
        checkpoints=[('pretrained',args.assets/'model/checkpoint.pth')]+[
            (r['condition'],Path(r['checkpoint'])) for r in trained['training']]
        outputs=[]
        for condition,checkpoint in checkpoints:
            checkpoint_hash=sha256(checkpoint)
            model=model_from_config(cfg,checkpoint).eval()
            target=folder/condition; target.mkdir()
            metadata=[]
            for i,row in enumerate(tests):
                if sha256(row['features'])!=row['features_sha256']:
                    raise ValueError('Test feature hash changed')
                feature=torch.load(row['features'],map_location='cpu',weights_only=True)
                audio=infer(model,feature,seed=base_seed+i)
                output=target/(row['utterance_id']+'.wav')
                sf.write(output,audio,44100,subtype='PCM_16')
                record={'condition':condition,'reference':str(checkpoint),'sentence_id':row['utterance_id'],
                    'text':row['text'],'output':str(output),'output_sha256':sha256(output),
                    'checkpoint_sha256':checkpoint_hash,'evaluation_seed':base_seed+i,
                    'output_duration_seconds':len(audio)/44100,
                    'output_rms':float(np.sqrt(np.mean(audio.astype(np.float64)**2))),
                    'output_peak':float(np.max(np.abs(audio)))}
                if trained['scope']=='kss_noncommercial_research':
                    qc=row.get('source_qc') or {}
                    record.update(manual_listening_verified=row.get('transcript_verified') is True,
                        original_recording_asr_cer=qc.get('asr_cer'),
                        original_recording_asr_exact_match=qc.get('asr_exact_match'),
                        quality_review_recommended=qc.get('review_recommended'))
                metadata.append(record); outputs.append(record)
            with (target/'metadata.csv').open('x',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(metadata[0])); writer.writeheader(); writer.writerows(metadata)
            if sha256(checkpoint)!=checkpoint_hash:raise ValueError('Checkpoint changed during evaluation synthesis')
            del model; gc.collect(); torch.cuda.empty_cache()
        from speechbrain.inference.speaker import SpeakerRecognition
        provenance=json.loads((args.evaluation_models/'model_provenance.json').read_text())
        speaker=SpeakerRecognition.from_hparams(source=provenance['speechbrain_directory'],
            savedir=str(args.evaluation_models/'speaker_runtime'),run_opts={'device':'cuda:0'})

        def audio16(path):
            wave,sr=sf.read(path,dtype='float32')
            if wave.ndim!=1 or not np.isfinite(wave).all():
                raise ValueError('Invalid evaluation audio')
            return torchaudio.functional.resample(torch.from_numpy(wave),sr,16000)[None].cuda()

        reference_audio=[audio16(r['wav']) for r in refs]
        with torch.no_grad():
            reference_embeddings=[speaker.encode_batch(r,normalize=False) for r in reference_audio]
            for index,row in enumerate(outputs):
                x=audio16(row['output'])
                embedding=speaker.encode_batch(x,normalize=False)
                scores=[float(speaker.similarity(r,embedding).reshape(-1)[0]) for r in reference_embeddings]
                if index==0:
                    standard=float(speaker.verify_batch(reference_audio[0],x)[0].reshape(-1)[0])
                    difference=abs(standard-scores[0])
                    if difference>1e-6:raise ValueError('Cached ECAPA differs from verify_batch')
                    status['cached_sim_equivalence_abs_difference']=difference
                row['sim']=sum(scores)/len(scores)
        del speaker,reference_audio,reference_embeddings,embedding; gc.collect(); torch.cuda.empty_cache()
        import whisper
        asr=whisper.load_model('small',device='cuda',download_root=provenance['whisper_directory'])
        for row in outputs:
            hypothesis=asr.transcribe(row['output'],language='ko',fp16=True,temperature=0)['text'].strip()
            reference=normalize(row['text']); hyp=normalize(hypothesis)
            if not reference:
                raise ValueError('Empty normalized CER reference')
            edits=jiwer.process_characters(reference,hyp)
            row.update(asr_transcript=hypothesis,cer=edits.cer,reference_characters=len(reference),
                       substitutions=edits.substitutions,deletions=edits.deletions,insertions=edits.insertions)
            print(json.dumps({k:row[k] for k in ('condition','sim','cer')},ensure_ascii=False),flush=True)
        with (folder/'per_file.csv').open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(outputs[0]));writer.writeheader();writer.writerows(outputs)
        summary=[]
        for condition,_ in checkpoints:
            chosen=[r for r in outputs if r['condition']==condition]
            summary.append({'condition':condition,'test_count':len(chosen),
                'mean_sim':float(np.mean([r['sim'] for r in chosen])),
                'mean_cer':float(np.mean([r['cer'] for r in chosen])),
                'scope':trained['scope']})
            for metric in ('sim','cer'):
                stats=statistics([r[metric] for r in chosen])
                summary[-1].update({f'{metric}_{key}':value for key,value in stats.items() if key not in ('mean','n')})
            summary[-1]['corpus_cer']=sum(r['substitutions']+r['deletions']+r['insertions'] for r in chosen)/sum(r['reference_characters'] for r in chosen)
        with (folder/'summary.csv').open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        if trained['scope']=='kss_noncommercial_research':
            breakdown=[]
            for condition,_ in checkpoints:
                for label,match in [('source_ASR_exact',True),('source_ASR_difference',False)]:
                    subset=[r for r in outputs if r['condition']==condition and r.get('original_recording_asr_exact_match') is match]
                    if subset:
                        breakdown.append({'condition':condition,'source_quality_stratum':label,'n':len(subset),
                            'mean_sim':float(np.mean([r['sim'] for r in subset])),
                            'mean_cer':float(np.mean([r['cer'] for r in subset])),
                            'note':'ASR screening is not human transcript verification; no test items excluded.'})
            if breakdown:
                with (folder/'source_quality_breakdown.csv').open('x',encoding='utf-8-sig',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(breakdown[0]));writer.writeheader();writer.writerows(breakdown)
        status.update(status='SYNTHETIC_SMOKE_EVALUATION_PASS' if trained['scope']=='synthetic_training_smoke' else 'EVALUATION_PASS',
            measurements=summary,evaluation_models=provenance)
    except Exception as error:
        status.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        save(folder/'evaluation.json',status)
        print(json.dumps({'directory':str(folder),'status':status['status']}),flush=True)


if __name__=='__main__':
    main()
