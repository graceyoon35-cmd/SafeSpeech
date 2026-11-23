"""Brief real WSLg window launch through the exact demo entry point."""
import os
import runpy
import sys
from unittest.mock import patch
from prepare_demo import HERE,ROOT,OLD,save
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

assert os.environ.get('QT_QPA_PLATFORM')!='offscreen'
actual_exec=QApplication.exec
observed={}
def bounded_exec(self):
    def inspect_and_close():
        windows=[w for w in self.topLevelWidgets() if w.isVisible()]
        observed.update(platform=self.platformName(),visible_windows=len(windows),titles=[w.windowTitle() for w in windows])
        for w in windows:w.close()
        self.quit()
    QTimer.singleShot(1800,inspect_and_close)
    return actual_exec()

sys.argv=['launch_spec_demo.py','--ui-snapshot',str(HERE/'app_snapshot/ui'),
          '--backend-repository',str(OLD),'--output-root',str(HERE/'live_app_jobs')]
with patch.object(QApplication,'exec',bounded_exec):
    try:runpy.run_path(str(ROOT/'scripts/launch_spec_demo.py'),run_name='__main__')
    except SystemExit as e:
        if e.code not in (None,0):raise
assert observed['visible_windows']>=1 and observed['platform']!='offscreen'
save(HERE/'wslg_startup_test.json',dict(status='WSLG_GRAPHICAL_STARTUP_PASS',**observed,
    scope='Visible WSLg window initialized through the exact launcher, then automatically closed. Interactive workflow separately tested offscreen. Native Windows Qt and physical speakers not verified.'))
print(observed)
