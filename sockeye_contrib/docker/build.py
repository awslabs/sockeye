#!/usr/bin/env python3

import os
import subprocess
import sys


SOCKEYE_DIR = os.path.dirname(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))
DOCKERFILE = os.path.join(SOCKEYE_DIR, 'sockeye_contrib', 'docker', 'Dockerfile')

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


def main():
    if not os.path.exists(DOCKERFILE):
        msg = 'Cannot find {}. Please make sure {} is a properly cloned repository.'.format(DOCKERFILE, SOCKEYE_DIR)
        raise FileNotFoundError(msg)

    check_command(GIT)
    check_command(DOCKER)

    print('Running commands in {}'.format(SOCKEYE_DIR), file=sys.stderr)

    sockeye_commit = run_command([GIT, 'rev-parse', 'HEAD'], get_output=True)
    tag = run_command([GIT, 'rev-parse', '--short', 'HEAD'], get_output=True)

    run_command([DOCKER, 'build', '-t', '{}:{}'.format(REPOSITORY, tag), '-f', DOCKERFILE, '.', '--build-arg',
                 'SOCKEYE_COMMIT={}'.format(sockeye_commit)])


if __name__ == '__main__':
    main()
