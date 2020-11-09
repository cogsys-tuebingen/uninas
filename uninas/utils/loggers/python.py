"""
shared loggers with a custom logging format
"""


import logging
import subprocess
from logging import Logger, Filter


_existing_loggers = {}


class CustomLoggerFilter(Filter):
    def filter(self, record):
        pathname = record.pathname
        s = pathname.split('/')
        if len(s) == 1:
            record.rel_path = s[-1]
            return True
        record.rel_path = '%s/%s' % (s[-2], s[-1])
        return True


log_format = '%(asctime)s %(rel_path)30s    %(message)s'
logging.basicConfig(level=logging.INFO,
                    format=log_format,
                    datefmt='%d.%m.%y %H:%M:%S')

for h in logging.root.handlers:
    h.addFilter(CustomLoggerFilter())


def get_logger(name=None, default_level=logging.INFO, save_file=None) -> Logger:
    global _existing_loggers
    logger, has_save = _existing_loggers.get(name, (None, False))
    if isinstance(logger, Logger) and (has_save or save_file is None):
        return logger

    formatter = logging.Formatter(log_format)
    logging.basicConfig(format=log_format, datefmt='%d.%m.%y %H:%M:%S')
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

    _existing_loggers[name] = (logger, save_file is not None)
    return logger


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


def log_args(logger: Logger, writer, args, add_git_hash=True):
    """ log argparse args to a given logging.logger and a tensorboard writer (may be None) """
    log_headline(logger, 'Args')
    others = [('current git hash', get_git_hash())] if add_git_hash else []
    entries = list(vars(args).items()) + others
    log_in_columns(logger, [(k, v) for k, v in entries], add_bullets=True)
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
