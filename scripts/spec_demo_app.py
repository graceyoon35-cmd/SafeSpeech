"""App processor factory with an explicit demo profile and isolated outputs."""
import os
from pathlib import Path
from spec_demo_profile import create_client

def build_processor(parent=None):
    # Required configuration prevents falling back to old jobs or stale files.
    repository=Path(os.environ['SAFESPEECH_SPEC_REPOSITORY'])
    output=Path(os.environ['SAFESPEECH_DEMO_OUTPUT_ROOT'])
    client=create_client(repository,output,epsilon=2.,iterations=10)
    from spec_desktop_processor import SpecProtectionProcessor
    return SpecProtectionProcessor(client,parent)
