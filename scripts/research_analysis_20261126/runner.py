"""Safe CLI for preserved, existing-data-only analysis tools."""
import argparse
import importlib
import json
import os
from pathlib import Path

TASKS = ('analyze_existing_only', 'package_existing', 'analyse_stages',
         'report_and_verify', 'report_results')
ARCHIVAL = ('report_and_verify', 'report_results')


def required_inputs(task, root, backend):
    study = root / 'runs/segment_noise_balance_20261124'
    pilot = root / 'runs/psychoacoustic_spec_pilot_20261125/pilot'
    refine = root / 'runs/psychoacoustic_spec_refinement_20261126'
    return {
        'analyze_existing_only': [study / 'analysis/per_file.csv',
            root / 'scripts/perceptual_spec_20261124/perceptual_objective.py',
            root / 'runs/perceptual_spec_20261124/attacks_followup/mask_eps2_iter10_s05/backend/perceptual_override.py',
            backend / 'scripts/train_melo_bounded.py'],
        'package_existing': [backend / 'runs/paired_melo_experiment_ong85l9i/clean/dataset.json',
            backend / 'runs/korean_melo_spec_mqxs1q1s/handoff_manifest.json',
            backend / 'runs/final_validation_20261119/strict_corrected_inputs/quality.csv'],
        'analyse_stages': [refine / 'stage_diagnosis/stage_measurements.csv'],
        'report_and_verify': [pilot / n for n in ('complete.json', 'quality_per_file.csv', 'summary.json', 'threshold_manifest.json')],
        'report_results': [refine / n for n in ('preservation_before.json', 'comparison/quality_per_file.csv',
            'comparison/summary.json', 'stage_diagnosis/complete.json', 'comparison/complete.json')]
    }[task]


def reserve_output(path, root):
    """Create a new result directory; reject old experiment roots and existing paths."""
    path = Path(path).resolve()
    protected = ('segment_noise_balance_20261124', 'algorithm_design_20261125',
                 'blind_listening_distribution_20261124', 'psychoacoustic_spec_pilot_20261125',
                 'psychoacoustic_spec_refinement_20261126', 'demo_noise_fix_20261124',
                 'perceptual_spec_20261124', 'human_listening_20261125')
    if path.exists():
        raise FileExistsError('Output must be a new directory')
    if any(path.is_relative_to(root / 'runs' / name) for name in protected):
        raise ValueError('Output must be outside historical experiment directories')
    path.mkdir(exist_ok=False)
    return path


def cli(task=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if task is None:
        parser.add_argument('task', choices=TASKS)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--backend-workspace', type=Path)
    parser.add_argument('--output-dir', type=Path, help='New directory, outside historical experiments')
    parser.add_argument('--check-inputs', action='store_true', help='Read-only required-file check; does not run the historical writer')
    args = parser.parse_args()
    task = task or args.task
    root = args.workspace.resolve()
    backend = Path(args.backend_workspace or os.environ.get('SAFESPEECH_LEGACY_WORKSPACE',
                   str(root.parent / 'SafeSpeech-desktop-integration-20261118'))).resolve()
    inputs = required_inputs(task, root, backend)
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        parser.error('Missing required files: ' + '; '.join(missing))
    if args.check_inputs:
        print(json.dumps(dict(status='REQUIRED_FILES_PRESENT', task=task, inputs=[str(p) for p in inputs],
            scope='Required-file presence only; nested WAV/config/hash checks occur in the actual analysis or saved-PCM verifier',
            historical_writer=task in ARCHIVAL), ensure_ascii=False))
        return
    if task in ARCHIVAL:
        parser.error('Historical report writer requires archived HEAD/refs/source snapshots. Use --check-inputs and verify_saved_pcm.py; original guards remain unchanged.')
    if args.output_dir is None:
        parser.error('--output-dir is required for execution')
    os.environ['SAFESPEECH_WORKSPACE'] = str(root)
    os.environ['SAFESPEECH_LEGACY_WORKSPACE'] = str(backend)
    module = importlib.import_module(task)
    out = reserve_output(args.output_dir, root)
    module.ROOT = root
    module.OUT = out
    if task == 'analyze_existing_only':
        module.STUDY = root / 'runs/segment_noise_balance_20261124'
        module.LEGACY = backend
    elif task == 'package_existing':
        module.OLD = backend
        module.PRIVATE = out / 'organizer_private'
        module.PUBLIC = out / 'evaluator'
    elif task == 'analyse_stages':
        module.INPUT = root / 'runs/psychoacoustic_spec_refinement_20261126'
    module.main()


if __name__ == '__main__':
    cli()
