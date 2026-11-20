from pathlib import Path
import threading
import time
import unittest

try:
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
    from spec_desktop_processor import SpecProtectionProcessor
except ImportError:
    QCoreApplication = None


@unittest.skipIf(QCoreApplication is None,'PySide6 not installed; run in the separate Qt test venv')
class DesktopProcessorTests(unittest.TestCase):
    def test_batch_signals_continue_after_failure_and_keep_event_loop_alive(self):
        application=QCoreApplication.instance() or QCoreApplication([])
        main_thread=threading.get_ident()
        worker_threads=[]
        class Client:
            def protect(self,item):
                worker_threads.append(threading.get_ident())
                time.sleep(.03)
                if item=='bad':raise ValueError('expected invalid input')
                return Path(item)
        processor=SpecProtectionProcessor(Client())
        self.assertFalse(processor.start([]))
        self.assertFalse(processor.start(None))
        events=[]; ticks=[]; done=[]
        processor.file_started.connect(lambda index,total:events.append(('start',index,total)))
        processor.file_finished.connect(lambda index:events.append(('success',index)))
        processor.file_failed.connect(lambda index,message:events.append(('failure',index)))
        processor.progress_changed.connect(lambda value:events.append(('progress',value)))
        processor.finished.connect(lambda:done.append(True))
        timer=QTimer();timer.setInterval(2);timer.timeout.connect(lambda:ticks.append(True));timer.start()
        self.assertTrue(processor.start(['first','bad','last']))
        self.assertFalse(processor.start(['overlap']))
        limit=time.monotonic()+10
        while not done and time.monotonic()<limit:
            application.processEvents(QEventLoop.ProcessEventsFlag.AllEvents,10)
            time.sleep(.001)
        timer.stop()
        self.assertTrue(done)
        self.assertFalse(processor.running)
        self.assertEqual(processor.results,(Path('first'),Path('last')))
        self.assertEqual(set(processor.failures),{1})
        self.assertIn(('start',2,3),events)
        self.assertIn(('progress',100),events)
        self.assertGreater(len(ticks),5)
        self.assertTrue(all(t!=main_thread for t in worker_threads))
        done.clear()
        self.assertTrue(processor.start(['again']))
        limit=time.monotonic()+10
        while not done and time.monotonic()<limit:
            application.processEvents(QEventLoop.ProcessEventsFlag.AllEvents,10)
            time.sleep(.001)
        self.assertTrue(done)
        self.assertEqual(processor.results,(Path('again'),))
        self.assertEqual(processor.failures,{})


if __name__=='__main__':
    unittest.main()
