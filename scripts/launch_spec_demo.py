"""Launch the prepared UI copy; never switch or change the app team's branch."""
import argparse
import os
from pathlib import Path
import runpy
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ui-snapshot',type=Path,required=True)
    p.add_argument('--backend-repository',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args()
    sys.dont_write_bytecode=True
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['SAFESPEECH_SPEC_REPOSITORY']=str(a.backend_repository.resolve(strict=True))
    os.environ['SAFESPEECH_DEMO_OUTPUT_ROOT']=str(a.output_root.resolve())
    sys.path[:0]=[str(a.ui_snapshot.resolve(strict=True)),str(a.backend_repository.resolve()/'scripts')]
    sys.argv=[str(a.ui_snapshot/'main.py')]
    runpy.run_path(str(a.ui_snapshot/'main.py'),run_name='__main__')

if __name__=='__main__':main()
