"""
shared loggers with a custom logging format
"""


import logging
import subprocess
from logging import Logger, Filter
from uninas.utils.meta import Singleton


class CustomLoggerFilter(Filter):
    def filter(self, record):
        pathname = record.pathname
        s = pathname.split('/')
        if len(s) == 1:
            record.rel_path = s[-1]
            return True
        record.rel_path = '%s/%s' % (s[-2], s[-1])
        return True


def log_in_columns(logger: Logger, lines: [[str]], start_space=0, between_space=4, add_bullets=False):
    """ log all lines, making sure each column has the same width """
    if len(lines) <= 0:
        return
    lines = [[str(v) for v in line] for line in lines]
    max_lens = [0 for _ in range(len(lines[0]))]
    for line in lines:
        for i, c in enumerate(line):
            max_lens[i] = max(max_lens[i], len(c))
    start = " " * start_space
    if add_bullets:
        start = " > %s" % start
    fmt = " " * between_space
    fmt = fmt.join(["{:<%d}" % n for n in max_lens])
    for line in lines:
        logger.info("%s%s" % (start, fmt.format(*line)))


def log_args(logger: Logger, writer, args, add_git_hash=True, descriptions: dict = None):
    """ log argparse args to a given logging.logger and a tensorboard writer (may be None) """
    descriptions = descriptions if isinstance(descriptions, dict) else {}
    log_headline(logger, 'Args')
    others = [('current git hash', get_git_hash())] if add_git_hash else []
    entries = list(vars(args).items()) + others
    log_in_columns(logger, [(k, v, descriptions.get(k, "")) for k, v in entries], add_bullets=True)
    for k, v in entries:
        k, v = str(k), str(v)
        if writer is not None:
            writer.add_text('args', '%s: %s' % (k, v))


def log_headline(logger: Logger, text: str, target_len=100, level=20):
    l0 = max((target_len - len(text) - 4) // 2, 0)
    l1 = max((target_len - len(text) - 4) - l0, 0)
    logger.log(level, '>' + '-'*l0 + ' ' + text + ' ' + '-'*l1 + '<')


def get_git_hash() -> str:
    try:
        sp = subprocess.Popen("exec git log --pretty=format:'%h' -n 1", shell=True, stdout=subprocess.PIPE)
        out = sp.stdout.read()
        sp.kill()
        sp.communicate()
        if len(out) > 0:
            return str(out).split("'")[1]
        return 'N/A'
    except:
        return 'N/A'


class LoggerManager(metaclass=Singleton):

    def __init__(self):
        """
        A class to create and keep track of loggers
        """
        self.log_format = '%(asctime)s %(levelname)8s %(rel_path)30s %(lineno)4d  |  %(message)s'
        self.log_level = logging.INFO
        self.set_logging(self.log_format, self.log_level)
        self.default_save_file = None
        self.loggers = {}

        for h in logging.root.handlers:
            h.addFilter(CustomLoggerFilter())

    def set_logging(self, fmt: str = None, level: int = None, default_save_file: str = None):
        """
        set logging defaults

        :param fmt: format string for logging
        :param level: log level
        :param default_save_file: save all logs to this file, unless specified otherwise
        """
        if fmt:
            self.log_format = fmt
        if level:
            self.log_level = level
        logging.basicConfig(level=self.log_level,
                            format=self.log_format,
                            datefmt='%d.%m.%y %H:%M:%S')
        if default_save_file:
            self.default_save_file = default_save_file

    def get_logger(self, name=None, default_level=logging.INFO, save_file=None) -> Logger:
        """
        get an (existing) logger
        """
        logger, has_save = self.loggers.get(name, (None, False))
        save_file = save_file if (save_file is not None) else self.default_save_file
        if isinstance(logger, Logger) and (has_save or save_file is None):
            return logger

        formatter = logging.Formatter(self.log_format)
        logging.basicConfig(format=self.log_format, datefmt='%d.%m.%y %H:%M:%S')
        logger = logging.getLogger(name if name is not None else __name__)
        logger.setLevel(default_level)
        # ch = logging.StreamHandler()
        # ch.setFormatter(formatter)
        # logger.addHandler(ch)
        if save_file is not None:
            fh = logging.FileHandler(save_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        for h in logger.handlers:
            h.addFilter(CustomLoggerFilter())

        self.loggers[name] = (logger, save_file is not None)
        return logger

    def __deepcopy__(self, memo):
        # avoid copy issues
        return self

    def cleanup(self):
        """
        delete all loggers
        """
        for (logger, _) in self.loggers.values():
            del logger
        del self.loggers
        self.loggers = {}
