"""Freeze measured recommendation; human approval remains explicitly pending."""
import json
from prepare_demo import HERE,ROOT,save,sha

comparison=json.loads((HERE/'comparison.json').read_text())
rows={r['condition']:r for r in comparison['rows']}
pairs={r['condition']:r for r in comparison['paired_differences']}
assert pairs['SPEC_eps2_iter10']['both_prompt_intervals_exclude_zero_in_defense_direction']
demo=json.loads((HERE/'demo_result.json').read_text())
profiles={r['condition']:r for r in demo['profiles']}
before=profiles['01_existing_profile_eps8_iter10'];after=profiles['02_candidate_eps2_iter10']
selection=dict(status='RECOMMENDED_DEMO_PROFILE_LIMITED_ATTACK_EVALUATED_HUMAN_LISTENING_PENDING',
    recommended=dict(epsilon_units_255=2,epsilon_normalized=2/255,iterations=10,seed=52,step='default epsilon_normalized/10'),
    reason='Best measured input quality of the two priority candidates, with lower clone SIM and higher CER observed against Clean in the fresh same-protocol attack. eps4 had stronger attack degradation but worse input quality.',
    actual_demo_quality=dict(snr_db=after['snr_db'],stoi=after['stoi'],clipping=after['clipping'],strict_epsilon_pass=after['strict_epsilon_pass'],
        max_abs_delta=after['linf'],snr_improvement_db=after['snr_db']-before['snr_db'],
        delta_rms_reduction_percent=100*(1-after['delta_rms']/before['delta_rms']),
        original_rms=profiles['original']['rms'],protected_rms=after['rms']),
    defense_measurements=dict(clean=rows['clean'],eps2=rows['SPEC_eps2_iter10'],eps4=rows['SPEC_eps4_iter10'],archived_eps8_re_evaluated=rows['SPEC_10']),
    eps2_vs_clean_paired_prompt_differences=pairs['SPEC_eps2_iter10'],
    defense_scope='New eps2/eps4 training inputs and fresh 1000-step attacks. KSS one speaker, training seed52, 140 train utterances, 30 test prompts, same Melo architecture. Archived Clean/eps8 checkpoints re-evaluated as their own references.',
    limits=['Human listening not performed (user answered not yet listened)',
        'One attack architecture and one training seed; no general protection probability or success rate',
        'Prompt bootstrap does not measure speaker/training-seed variability',
        'Old eps8 checkpoint used historical unprojected inputs; today strict eps8 demo WAV is a separate quality control',
        'Live GPU re-generation is not guaranteed byte-identical to the frozen demo WAV',
        'Native Windows Qt and presentation speakers not tested; WSLg startup and offscreen actual workflow tested'],
    human_naturalness_approved=False,universal_cloning_defense_verified=False,
    volume_or_filter_change=False,additional_training_planned=False,
    app_settings_source=str(ROOT/'scripts/spec_demo_app.py'),ui_patch=str(ROOT/'patches/spec_demo_ui_20261124.patch'),
    app_launcher=str(HERE/'Start-Demo.ps1'),app_team_branch_modified=False,
    original=str(HERE/'final_demo_eps2/original.wav'),protected=str(HERE/'final_demo_eps2/protected_eps2_iter10.wav'),
    listening_AB=str(HERE/'final_demo_eps2/AB_original_then_protected.wav'),
    comparison=str(HERE/'comparison.json'),verification=str(HERE/'final_verification.json'))
save(HERE/'final_selection.json',selection)
# Only newly created demo metadata is finalized. No WAV or old artifact is rewritten.
manifest_path=HERE/'final_demo_eps2/demo_settings.json'
manifest=json.loads(manifest_path.read_text())
manifest.update(status=selection['status'],defense_evaluation_completed=True,
    defense_effect_observed_in_limited_attack=True,defense_evaluation=str(HERE/'comparison.json'),
    human_listening_verified=False,universal_cloning_defense_verified=False,final_selection=str(HERE/'final_selection.json'))
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
handoff_path=HERE/'app_team_handoff.json';handoff=json.loads(handoff_path.read_text())
handoff.update(defense_status='LIMITED_SAME_ARCHITECTURE_ATTACK_EFFECT_OBSERVED; human listening pending; no calibrated probability',
    final_selection=str(HERE/'final_selection.json'),comparison=str(HERE/'comparison.json'))
handoff_path.write_text(json.dumps(handoff,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert sha(HERE/'final_demo_eps2/protected_eps2_iter10.wav')==after['sha256']
print(json.dumps(dict(status=selection['status'],actual_demo_quality=selection['actual_demo_quality']),ensure_ascii=False,indent=2))
