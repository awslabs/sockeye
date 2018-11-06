# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# \Author: {zhiyuan.huang, rong.a.zhang}@intel.com


# Describtion: This script is used to process per core translation, which can greatly speedup translate perefomance.

import argparse
import subprocess
import threading
import tempfile
import logging
import string
import os
import time
import mxnet as mx

def add_args(parser):
    parser.add_argument("--module" , '-m', help='the pretrained module', default="wmt_model")
    parser.add_argument("--input_file", '-i', help='the file input to translate', default='newstest2016.tc.BPE.de')
    parser.add_argument("--output_file", '-o', help='the file saved the result of translation', default='newstest2016.tc.BPE.en')
    parser.add_argument("--batch_size", '-bs', help='the batch size of process', default=32)
    parser.add_argument("--data_type", '-t', help='if uses run benchmark or not', default="benchmark")

def task(args):
    os.system(args)

def benchmark(cores, args):
    """
    benchmark is used for Processing per core translation. Each core translates the whole input file.
    Return after all translations done.

    :param cores: the number of cores used for translation, each core will launch a thread to translate
    :param args: input parameters
    """
    model = args.module
    fileInput = args.input_file
    fileOutput = args.output_file
    batchsize = args.batch_size
   
    thread = []
    for i in range(cores): 
        command = "taskset -c %d-%d python3 -m sockeye.translate -m %s -i %s -o %s --batch-size %d --output-type benchmark --use-cpu > /dev/null 2>&1 " % (i, i, model, fileInput, fileOutput, batchsize)
        t = threading.Thread(target = task, args=(command,))
        thread.append(t)
        t.start()
    
    for t in thread:
        t.join()

def split_file(splitNum, fileInput, lines):
    """
    split_file is used to split fileInput into splitNum small pieces file.
    For example, when splitNum is 56, a 112 lines file will be split into 56 files and each file has 2 lines.

    :param splitNum: split into splitNum files
    :param fileInput: file to be split
    :param lines: lines of fileInput
    """
    quot = lines // splitNum
    rema = lines % splitNum
    files = []
    current_line = 0
    for i in range(splitNum):
        if i < rema:
            read_line = quot + 1
        else:
            read_line = quot
        temp = tempfile.NamedTemporaryFile()
        os.system("head -n%d %s| tail -n%d > %s" % (current_line + read_line, fileInput, read_line, temp.name))
        current_line += read_line
        files.append(temp)

    return files

def translate(cores, files, args):
    """
    translate is used for Processing per core translation. cores[i] will translate files[i]

    :param cores: the number of cores used for translation, each core will launch a thread to translate
    :param files: file list to be translated
    :param args: input parameters
    :return: list of translated file
    """
    model = args.module
    batchsize = args.batch_size

    # split inputfile to a series of small files which number is equal cores
    file = []
    thread = []
    for i in range(cores):
        files[i].seek(0)
        temp = tempfile.NamedTemporaryFile()
        command = "taskset -c %d-%d python3 -m sockeye.translate -m %s -i %s -o %s --batch-size %d --output-type benchmark --use-cpu > /dev/null 2>&1 " % (i, i, model, files[i].name, temp.name, batchsize)
        file.append(temp)

        t = threading.Thread(target = task, args=(command,))
        thread.append(t) 
        t.start()
    #wait for all translation done
    for t in thread:
        t.join()
    
    return file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MXnet Sockeye  cpu_process_per_core_translation.py -m model-name -i file_to_translate -o result_to_save --batch-size 32')
    add_args(parser)
    args = parser.parse_args()
    fileInput = args.input_file
    fileOutput = args.output_file

    socket = int(os.popen("grep -w 'physical id' /proc/cpuinfo | sort -u | wc -l").read().strip())
    cores = int(os.popen("grep -w 'core id' /proc/cpuinfo | sort -u | wc -l").read().strip())
    total_cores = socket * cores
    print(total_cores)

    os.system("export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0")
    ompStr = "export OMP_NUM_THREADS=" + str(total_cores)
    os.system(ompStr)

    # the total lines of input file will be translated
    lines = 0
    for line in open(args.input_file): lines += 1
    print('Translating...')
    # clear file for output
    os.system("rm %s -f"%fileOutput)
    if args.data_type == "benchmark":
        lines = lines * total_cores
        start = time.time()
        order = benchmark(total_cores, args)
        end = time.time()
    else:
        splited_files = split_file(total_cores, fileInput, lines)
        start = time.time()
        translated_files = translate(total_cores, splited_files, args)
        end = time.time()
        for i in range(total_cores):
            os.system("cat %s >> %s" % (translated_files[i].name, fileOutput))
            splited_files[i].close()
            translated_files[i].close()

    total_time = end - start

    print("Instance nums: %d" %  total_cores)
    print("Total Processed lines: %d" % lines)
    print("Total time(s): %.3f s" % total_time)
    print("Speed: %.3f sent/sec" % (lines / total_time))
    
