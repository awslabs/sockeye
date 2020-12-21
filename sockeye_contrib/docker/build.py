#!/usr/bin/env python3

import os
import subprocess
import sys


SOCKEYE_DIR = os.path.dirname(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))
DOCKERFILE_CPU = os.path.join(SOCKEYE_DIR, 'sockeye_contrib', 'docker', 'Dockerfile.cpu')
DOCKERFILE_GPU = os.path.join(SOCKEYE_DIR, 'sockeye_contrib', 'docker', 'Dockerfile.gpu')
REQS_BASE = os.path.join(SOCKEYE_DIR, 'requirements', 'requirements.txt')
REQS_HOROVOD = os.path.join(SOCKEYE_DIR, 'requirements', 'requirements.horovod.txt')

GIT = 'git'
DOCKER = 'docker'

REPOSITORY = 'sockeye'


def check_command(cmd):
    try:
        retcode = subprocess.call([cmd, '--version'])
    except FileNotFoundError:
        retcode = None
    if retcode != 0:
        msg = 'Please install {}'.format(cmd)
        raise subprocess.SubprocessError(msg)


def run_command(cmd_args, get_output=False):
    print('Running: {}'.format(' '.join(cmd_args)), file=sys.stderr)
    if get_output:
        return subprocess.check_output(cmd_args, cwd=SOCKEYE_DIR).decode('utf-8').strip()
    return subprocess.call(cmd_args, cwd=SOCKEYE_DIR)


def read_requirements(fname):
    with open(fname, 'rt') as reqs_in:
        # MXNet is installed separately in the Dockerfile
        return ' '.join(line.strip() for line in reqs_in if not line.startswith('mxnet'))


def main():
    for fname in (DOCKERFILE_CPU, DOCKERFILE_GPU, REQS_BASE, REQS_HOROVOD):
        if not os.path.exists(fname):
            msg = 'Cannot find {}. Please make sure {} is a properly cloned repository.'.format(fname, SOCKEYE_DIR)
            raise FileNotFoundError(msg)

    if len(sys.argv[1:]) != 1:
        print('Usage: {} (cpu|gpu)'.format(SOCKEYE_DIR), file=sys.stderr)
        sys.exit(2)

    if sys.argv[1] == 'cpu':
        dockerfile = DOCKERFILE_CPU
        repository = REPOSITORY + '-cpu'
    else:
        dockerfile = DOCKERFILE_GPU
        repository = REPOSITORY + '-gpu'

    check_command(GIT)
    check_command(DOCKER)

    print('Running commands in {}'.format(SOCKEYE_DIR), file=sys.stderr)

    sockeye_commit = run_command([GIT, 'rev-parse', 'HEAD'], get_output=True)
    tag = run_command([GIT, 'rev-parse', '--short', 'HEAD'], get_output=True)

    run_command([DOCKER, 'build',
                 '-t', '{}:{}'.format(repository, tag),
                 '-f', dockerfile,
                 '.',
                 '--build-arg', 'SOCKEYE_COMMIT={}'.format(sockeye_commit),
                 '--build-arg', 'REQS_BASE={}'.format(read_requirements(REQS_BASE)),
                 '--build-arg', 'REQS_HOROVOD={}'.format(read_requirements(REQS_HOROVOD))])

    run_command([DOCKER, 'tag', '{}:{}'.format(repository, tag), '{}:latest'.format(repository)])


if __name__ == '__main__':
    main()
