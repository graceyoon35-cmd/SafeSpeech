"""Connect validated SPEC handoffs to the pinned team's paired robustness sweep."""
import argparse
import json
from pathlib import Path
import tempfile

from run_korean_melo_spec import ROOT, save, validate_handoff
from run_robustness_sweep_b import default_conditions


def export_inputs(run, output_root):
    run = Path(run).resolve()
    report = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    handoff = json.loads((run / 'handoff_manifest.json').read_text(encoding='utf-8'))
    expected = {'PASS': 'READY_FOR_MODEL_REVIEW',
                'NONHUMAN_EXECUTION_PASS': 'NONHUMAN_NOT_FOR_FINETUNING'}
    if report.get('status') not in expected or handoff.get('status') != expected[report['status']]:
        raise ValueError('Only completed, consistent SPEC runs may be exported')
    if report.get('input_files_unchanged') is not True or report.get('assets_unchanged') is not True:
        raise ValueError('Run must confirm preserved source and model files')
    rows = handoff.get('utterances', [])
    if not rows or len(rows) != report.get('handoff_count'):
        raise ValueError('Incomplete handoff')
    human = all(row.get('input_kind') == 'human_recording' for row in rows)
    if human != report.get('human_recording_experiment') or human != (report['status'] == 'PASS'):
        raise ValueError('Human/nonhuman classification mismatch')
    validate_handoff(rows)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='spec_robustness_inputs_', dir=output_root))
    paths = []
    for row in rows:
        name = f"pair_{len(paths)+1:04d}"
        path = folder / f'{name}.json'
        save(path, {'schema_version': 1,
                    'experiment_id': f"{run.name}_{row['utterance_id']}_{row['iterations']}",
                    'audio': {'original': row['clean_path'], 'protected': row['protected_path']},
                    'conditions': default_conditions()})
        paths.append(path)
    save(folder / 'provenance.json', {'spec_run': str(run), 'source_status': report['status'],
         'human_recording_experiment': human, 'input_kinds': sorted({r['input_kind'] for r in rows}),
         'manifests': [str(path) for path in paths],
         'limitations': 'Transforms and STOI cannot establish cloning-defense efficacy. Mock input is not speech.'})
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-run', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    parser.add_argument('--run-batch', action='store_true')
    args = parser.parse_args()
    paths = export_inputs(args.spec_run, args.output_root)
    print(json.dumps({'manifests': [str(p) for p in paths]}, ensure_ascii=False), flush=True)
    if args.run_batch:
        from run_robustness_batch_b import run_batch
        summary, report = run_batch(paths, args.output_root, repeats=2)
        print(json.dumps({'summary': str(summary), 'status': report['status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
