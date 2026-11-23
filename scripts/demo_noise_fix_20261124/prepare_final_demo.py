"""Freeze first candidate separately so old noisy WAV cannot be selected by mistake.
This is an audition/operational candidate, pending the separate defense result.
"""
import json
from pathlib import Path
import shutil
from prepare_demo import HERE,ROOT,OLD,sha,save,quality

target=HERE/'final_demo_eps2';target.mkdir(exist_ok=False)
source=HERE/'demo_wavs/00_original.wav'
protected=HERE/'demo_wavs/02_candidate_eps2_iter10.wav'
shutil.copyfile(source,target/'original.wav')
shutil.copyfile(protected,target/'protected_eps2_iter10.wav')
shutil.copyfile(HERE/'demo_wavs/AB_original_then_02_candidate_eps2_iter10.wav',target/'AB_original_then_protected.wav')
save(target/'demo_settings.json',dict(status='FIRST_CANDIDATE_AUDIO_READY_DEFENSE_PENDING',
    original=str(target/'original.wav'),protected=str(target/'protected_eps2_iter10.wav'),
    original_sha256=sha(source),protected_sha256=sha(protected),
    epsilon_units_255=2,epsilon_normalized=2/255,iterations=10,step_normalized=(2/255)/10,seed=52,
    format=dict(sample_rate=44100,bit_depth=16,channels=1),
    source_text='우리는 서로 답을 비교해 보았다.',
    quality=quality(source,protected,2),human_listening_verified=False,defense_evaluation='PENDING: see separate defense_attack_140 results',
    app_launcher=str(HERE/'Start-Demo.ps1'),backend_repository=str(OLD),
    do_not_use_as_final_demo=['01_existing_profile_eps8_iter10.wav','optimized.wav','any previous job or glob-selected latest file'],
    no_gain_or_filter=True))
save(HERE/'app_team_handoff.json',dict(
    app_source_revision='d678c2051e336b2be2a595cc37562030c7df1954',target_branch='feature/ui (not modified or merged)',
    prepared_on_branch='feature/spec-perturbation',patch=str(ROOT/'patches/spec_demo_ui_20261124.patch'),
    modules=[str(ROOT/'scripts'/p) for p in ('spec_demo_profile.py','spec_demo_app.py','launch_spec_demo.py')],
    changes=['Replace DummyProtectionProcessor with real SpecProtectionProcessor factory',
        'Explicit epsilon=2 (/255), iterations=10, seed=52; existing SpecClient generates a fresh UUID for each call',
        'Only 44.1kHz mono PCM16 WAV of 1..10 seconds for this demo; reject unsupported media rather than silently convert',
        'Clear old current_result and playback sources on start (existing behavior verified)',
        'Use only result.output_path whose hash, input hash and epsilon/iterations are verified against result.json',
        'Confidence is unmeasured; no fixed 92% or None% value',
        'Block closing while CUDA job is running; stop playback on close'],
    configure_environment=['SAFESPEECH_SPEC_REPOSITORY = validated backend repository path',
        'SAFESPEECH_DEMO_OUTPUT_ROOT = new live_app_jobs folder; never point at historical output folders'],
    run_now=str(HERE/'Start-Demo.ps1'),snapshot=str(HERE/'app_snapshot/ui'),
    scope='Prepared app snapshot tested under WSL Qt offscreen; branch merge and native Windows GUI/device validation not performed.',
    defense_status='PENDING; generation success does not mean cloning defense success'))
print(str(target/'protected_eps2_iter10.wav'))
