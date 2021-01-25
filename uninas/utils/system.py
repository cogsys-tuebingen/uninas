import os
import platform
import subprocess
import GPUtil
from pip._internal.operations.freeze import freeze
import torch.utils.collect_env as collect_env
import torch.backends.cudnn as cudnn


def headline(text: str) -> str:
    return '\n\n\n' + '-'*100 + '\n' + text + '\n' + '-'*100 + '\n'


def get_command_result(command: str) -> str:
    try:
        sp = subprocess.Popen("exec %s" % command, shell=True, stdout=subprocess.PIPE)
        out = sp.stdout.read()
        sp.kill()
        sp.communicate()
        if len(out) > 0:
            return str(out).split("'")[1]
        return '<failed: no output>'
    except Exception as e:
        return '<failed: %s>' % str(e)


def dump_system_info(file_path: str):
    if os.path.isfile(file_path):
        os.remove(file_path)
    with open(file_path, 'w+') as o:
        o.write(headline('torch collect_env'))
        o.write(collect_env.get_pretty_env_info())

        o.write(headline('system info'))
        o.write('platform: %s\n' % platform.platform())
        o.write('python: %s\n' % platform.python_version())

        o.write(headline('gpus'))
        try:
            for i, gpu in enumerate(GPUtil.getGPUs()):
                o.write('gpu %d\n' % i)
                for k in ['id', 'driver', 'name', 'memoryTotal']:
                    o.write('\t%s=%s\n' % (k, gpu.__dict__[k]))
        except ValueError as e:
            o.write("%s" % repr(e))

        o.write(headline('cuda / cudnn'))
        o.write('cuda via cat: %s\n' % get_command_result('cat /usr/local/cuda/version.txt'))
        o.write('cuda via dpkg: %s\n' % get_command_result('dpkg -l | grep cuda-toolkit'))
        o.write('cuda via nvcc: %s\n' % get_command_result('nvcc --version'))
        o.write('cudnn version: %s\n' % cudnn.version())
        # o.write('\nnvidia-smi:\n%s\n' % get_command_result('nvidia-smi'))

        o.write(headline('pip freeze'))
        for r in freeze(local_only=True):
            o.write('%s\n' % r)


if __name__ == '__main__':
    dump_system_info('/tmp/uninas/sysinfo.txt')
