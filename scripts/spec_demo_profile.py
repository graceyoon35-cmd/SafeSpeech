"""Explicit low-noise SPEC profile; never reuse a previously generated WAV.

Import create_client from the app's processor factory, or run this CLI. PASS
means the WAV contract passed, never a claim of measured cloning defense.
The backend repository is read-only and must contain the validated SPEC bridge.
"""
import argparse
import os
from pathlib import Path
import sys

def create_client(repository, output_root, *, execution='auto', epsilon=2., iterations=10):
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    repository = Path(repository).resolve(strict=True)
    sys.path.insert(0, str(repository/'scripts'))
    from spec_desktop_bridge import SpecClient, SpecConfig
    return SpecClient(SpecConfig(repository=repository, output_root=Path(output_root),
        execution=execution, epsilon=epsilon, iterations=iterations, seed=52))

def main():
    import json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--epsilon', type=float, choices=(2.,4.,8.), default=2.)
    p.add_argument('--iterations', type=int, choices=(1,10), default=10)
    p.add_argument('--execution', choices=('auto','local','wsl'),default='auto')
    a=p.parse_args()
    result=create_client(a.repository,a.output_root,execution=a.execution,
                         epsilon=a.epsilon,iterations=a.iterations).protect(a.input)
    print(json.dumps(result.as_dict(),ensure_ascii=False),flush=True)

if __name__=='__main__': main()
