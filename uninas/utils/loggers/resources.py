"""
to log file system / RAM / device usage to tensorboard
due to stupid python parallel processing, may need to call .wakeup() to ensure logging
"""


import threading
import subprocess
from uninas.training.devices.abstract import AbstractDeviceMover
from uninas.utils.loggers.exp import LightningLoggerBase


class AbortableSleep:
    """
    A class that enables sleeping with interrupts
    see https://stackoverflow.com/questions/28478291/abortable-sleep-in-python
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._aborted = False

    def __call__(self, secs):
        with self._condition:
            self._aborted = False
            self._condition.wait(timeout=secs)
            return not self._aborted

    def abort(self):
        with self._condition:
            self._condition.notify()
            self._aborted = True


class ResourceLogThread(threading.Thread):
    """
    simple thread to log file system / ram / device util to tensorboard
    did not invest much into thread safety
    """

    def __init__(self, exp_logger: LightningLoggerBase, seconds=300, daemon=True, mover: AbstractDeviceMover = None,
                 log_fs=True, log_ram=True, log_devices=True):
        """
        :param exp_logger: experiment logger to save to
        :param seconds: log each 'seconds' to the writer
        :param daemon: launch as a daemon thread, will be ended automatically when the main thread stops
        :param mover: logs the devices associated with this mover
        :param log_fs: whether to log file system stats
        :param log_ram: whether to log RAM stats
        :param log_devices: whether to log devices stats
        """
        super().__init__()
        self.daemon = daemon

        self.mover = mover
        self.log_fs = log_fs
        self.log_ram = log_ram
        self.log_devices = log_devices and self.mover is not None

        self.exp_logger = exp_logger
        self.seconds = seconds
        self.step = 0
        self.keep_running = True
        self._abortable_sleep = AbortableSleep()
        self._log_condition = threading.Condition()

    @staticmethod
    def _bash_output(command: str) -> str:
        sp = subprocess.Popen("exec %s" % command, shell=True, stdout=subprocess.PIPE)
        out = sp.stdout.read()
        sp.kill()
        sp.communicate()
        return str(out)

    def log_resources(self, log_fs=True, log_ram=True, log_devices=True, mover: AbstractDeviceMover = None):
        with self._log_condition:
            metrics = {}
            # file system
            if log_fs:
                fs_lines = self._bash_output("df --block-size=1M").replace("'", "").split(r'\n')
                fs_lines = [line.split() for line in fs_lines]
                for line in fs_lines[1:-1]:
                    name, used, available = line[0], int(line[2]), int(line[3])
                    if ':' in name:
                        continue
                    total = used + available
                    metrics['res_fs/%s/%s' % (name, 'total')] = total
                    metrics['res_fs/%s/%s' % (name, 'used')] = used / total

            # RAM
            if log_ram:
                ram_lines = self._bash_output("free -tmw").replace("'", "").split(r'\n')
                ram_lines = [line.split() for line in ram_lines]
                ram_names = ram_lines[0][1:]
                for line in ram_lines[1:-1]:
                    name = line[0][:-1]
                    for i, v in enumerate(line[1:]):
                        metrics['res_ram/%s/%s' % (name, ram_names[i])] = int(v)

            # devices
            if log_devices:
                for k, v in mover.get_usage_dict(log_all=self.step == 0).items():
                    metrics['res_%s' % k] = v

            self.exp_logger.log_metrics(metrics, self.step)
            self.exp_logger.save()
            self.step += 1
            self._log_condition.notify()

    def wakeup(self):
        self._abortable_sleep.abort()

    def run(self):
        self.log_resources(log_ram=True, log_fs=True, log_devices=self.mover is not None, mover=self.mover)
        while self.keep_running:
            self._abortable_sleep(self.seconds)
            self.log_resources(log_ram=self.log_ram, log_fs=self.log_fs, log_devices=self.log_devices, mover=self.mover)

    def stop(self):
        self.keep_running = False
        self._abortable_sleep.abort()
