# Describtion: This script is used for CPU Multi-instance Translate, which process per core translation.
# It can greatly speedup translate perefomance.
# FileName: cpu_process_per_core_translation.py
# usage: cpu_process_per_core_translation.py -m model-name -i file_to_translate -o result_to_save --batch-size 32
# Version: 2.0

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

# benchmark: multi-instances translating, each instance trans the same input file separately
def benchmark(cores, args):
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

# split file to small files
def split_file(cores, fileInput, lines):
    quot = lines // cores
    rema = lines % cores
    files = []
    current_line = 0
    for i in range(cores):
        if i < rema:
            read_line = quot + 1
        else:
            read_line = quot
        temp = tempfile.NamedTemporaryFile()
        os.system("head -n%d %s| tail -n%d > %s" % (current_line + read_line, fileInput, read_line, temp.name))
        current_line += read_line
        files.append(temp)

    return files

# translate: multi-instances translation, each instance trans whole_lines/instance_num lines of file, and merge into one complete output file
def translate(cores, files, args):
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
    
    for t in thread:
        t.join()
    
    return file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MXnet Sockeye Multi-instances Benchmark')
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
            os.system("wc -l %s"%translated_files[i].name)
            os.system("cat %s >> %s" % (translated_files[i].name, fileOutput))
            splited_files[i].close()
            translated_files[i].close()

    total_time = end - start

    print("Instance nums: %d" %  total_cores)
    print("Total Processed lines: %d" % lines)
    print("Total time(s): %.3f s" % total_time)
    print("Speed: %.3f sent/sec" % (lines / total_time))
    
