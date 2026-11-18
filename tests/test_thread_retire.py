"""Retiring a worker thread must not block the GUI thread.

The old pattern, run from a worker's ``finished`` slot on the GUI thread::

    thread.quit(); thread.wait(); thread.deleteLater()

has no timeout on ``wait()``, so a worker that is slow to unwind freezes the
window until Windows reports "Python not responding". In the Spectrum tab it ran
once per ~10 s identification cycle.

``ThreadRetirer`` asks the thread to quit and releases it only when
``QThread.finished`` arrives, so neither thread nor worker is destroyed while
still running (which aborts Qt), and the GUI thread never waits — except in
``wait_all()``, which exists for shutdown.
"""

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEventLoop, QObject, QThread, Slot
from PySide6.QtWidgets import QApplication


class _SlowWorker(QObject):
    """Holds its thread busy, so a retire during the hold proves it is async."""

    def __init__(self, hold_s: float = 0.4) -> None:
        super().__init__()
        self._hold_s = float(hold_s)
        self.ran = False

    @Slot()
    def run(self) -> None:
        # Sleeping here is on the WORKER thread, which is the point.
        time.sleep(self._hold_s)
        self.ran = True


class TestThreadRetirer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = QCoreApplication.instance()
        if inst is not None and not isinstance(inst, QApplication):
            raise unittest.SkipTest("a QCoreApplication exists; widgets cannot be created")
        cls.app = inst or QApplication([])

    def _pump(self, seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.app.processEvents(QEventLoop.AllEvents, 10)

    def _started_pair(self, hold_s: float = 0.4):
        thread = QThread()
        worker = _SlowWorker(hold_s)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        thread.start()
        # Let the worker actually enter run() before we retire it.
        self._pump(0.05)
        return thread, worker

    def test_retire_returns_at_once_while_the_worker_is_still_busy(self):
        from sensepi.gui.thread_retire import ThreadRetirer

        retirer = ThreadRetirer()
        thread, worker = self._started_pair(0.4)
        self.addCleanup(retirer.wait_all, 3000)

        t0 = time.perf_counter()
        retirer.retire(thread, worker)
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 0.1,
                        f"retire() blocked the GUI thread for {elapsed * 1000:.0f} ms")
        self.assertTrue(thread.isRunning(),
                        "the worker should still be running: retire must not wait for it")

    def test_the_thread_is_kept_referenced_until_it_finishes(self):
        from sensepi.gui.thread_retire import ThreadRetirer

        retirer = ThreadRetirer()
        thread, worker = self._started_pair(0.3)
        self.addCleanup(retirer.wait_all, 3000)

        destroyed = []
        thread.destroyed.connect(lambda *_: destroyed.append(True))

        retirer.retire(thread, worker)
        # Destroying a running QThread aborts the process, so while it runs it
        # must stay in _pending and must NOT have been deleted yet.
        self.assertEqual(len(retirer._pending), 1)
        self.assertEqual(destroyed, [], "a running QThread was deleted")

        self._pump(1.0)

        self.assertEqual(retirer._pending, [],
                         "a finished thread was never released")
        self.assertEqual(destroyed, [True],
                         "the finished thread was released but never deleted")
        self.assertTrue(worker.ran, "the worker was cut short instead of unwinding")

    def test_wait_all_reports_success_once_every_thread_has_ended(self):
        from sensepi.gui.thread_retire import ThreadRetirer

        retirer = ThreadRetirer()
        for _ in range(3):
            thread, worker = self._started_pair(0.15)
            retirer.retire(thread, worker)

        self.assertTrue(retirer.wait_all(3000),
                        "wait_all timed out on threads that do finish")
        self.assertEqual(retirer._pending, [])

    def test_wait_all_reports_failure_when_a_thread_outlives_the_timeout(self):
        from sensepi.gui.thread_retire import ThreadRetirer

        retirer = ThreadRetirer()
        thread, worker = self._started_pair(0.6)
        self.addCleanup(retirer.wait_all, 3000)
        retirer.retire(thread, worker)

        # A short deadline must be reported honestly, not waited out.
        t0 = time.perf_counter()
        ok = retirer.wait_all(50)
        elapsed = time.perf_counter() - t0

        self.assertFalse(ok, "wait_all claimed success while a thread still ran")
        self.assertLess(elapsed, 0.4, "wait_all ignored its own timeout")

    def test_a_worker_with_no_thread_is_still_disposed(self):
        from sensepi.gui.thread_retire import ThreadRetirer

        retirer = ThreadRetirer()
        worker = _SlowWorker(0.0)

        retirer.retire(None, worker)      # must not raise

        self.assertEqual(retirer._pending, [])

    def test_retiring_an_already_finished_thread_releases_it(self):
        from sensepi.gui.thread_retire import ThreadRetirer

        retirer = ThreadRetirer()
        thread = QThread()
        worker = _SlowWorker(0.0)
        worker.moveToThread(thread)
        thread.start()
        thread.quit()
        thread.wait(2000)                 # finished BEFORE retire() is called

        retirer.retire(thread, worker)

        # finished has already been emitted, so the connect alone would never
        # fire; retire() has to notice and release it itself.
        self.assertEqual(retirer._pending, [],
                         "a thread that finished before retire() leaked")


if __name__ == "__main__":
    unittest.main()
