# Describtion: This script is used for CPU Multi-instance Translate,
# which can greatly translate perefomance.
# FileName: mlt-trans.py
# usage: python mlt-trans.py -m model-name -i file_to_translate -o result_to_save --batch-size 32
# Version: 2.0

import argparse
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


# benchmark: multi-instances translate using same input file
def benchmark(cores, args):
    model = args.module
    fileInput = args.input_file
    fileOutput = args.output_file
    batchsize = args.batch_size

    for i in range(cores): 
        command = ("taskset -c %s-%s python3 -m sockeye.translate -m %s -i %s -o %s --batch-size %s --output-type benchmark --use-cpu > /dev/null 2>&1 " % (str(i), str(i), model, fileInput, fileOutput, str(batchsize)))
        if i != cores - 1:
            command += "&"
        os.system(command)

# split file to small files
def split_file(cores, fileInput):
    lines = len(open(fileInput).readlines())
    quot = lines // cores
    rema = lines % cores
    if rema != 0:
        quot += 1
        os.system("cp %s mlt-nmt-temp.log && head -n%s %s >> mlt-nmt-temp.log " % (fileInput, str(quot * cores - lines), fileInput))

    os.system("split -l %s mlt-nmt-temp.log -d -a 2 mlt-nmt.log." % str(quot))

# merge_translate: multi-instances translation, each instance trans  whole_lines/instance_num lines of file, and merge into one complete output file
def merge_translate(cores, args):
    model = args.module
    batchsize = args.batch_size

    # split inputfile to a series of small files which number is equal cores
    split_file(cores, fileInput)
    for i in range(cores):
        ifile = ''
        if i < 10:
            ifile = "mlt-nmt.log.0" + str(i) 
        else:
            ifile = "mlt-nmt.log." + str(i)

        command = ("taskset -c %s-%s python3 -m sockeye.translate -m %s -i %s -o %s --batch-size %s --output-type benchmark --use-cpu > /dev/null 2>&1 " % (str(i), str(i), model, ifile, ifile+".result.en", str(batchsize)))
        if i != cores - 1:
            command += "&"
        os.system(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MXnet Sockeye Multi-instances Benchmark')
    add_args(parser)
    args = parser.parse_args()
    fileInput = args.input_file
    fileOutput = args.output_file

    socket = int(os.popen("grep -w 'physical id' /proc/cpuinfo | sort -u | wc -l").read().split('\n')[0])
    cores = int(os.popen("grep -w 'core id' /proc/cpuinfo | sort -u | wc -l").read().split('\n')[0])
    total_cores = socket * cores

    os.system("export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0")
    ompStr = "export OMP_NUM_THREADS=" + str(total_cores)
    os.system(ompStr)

    # the total lines of input file will be translated
    lines = len(open(args.input_file).readlines())
    print('Translating...')
    start = .0
    end = .0
    if args.data_type == "benchmark":
        lines = lines * total_cores
        start = time.time()
        benchmark(total_cores, args)
        end = time.time()
    else:
        start = time.time()
        merge_translate(total_cores, args)
        #wait all process complete
        process_cnt = cores
        stop_cnt = 0   
        #force stop 100s after the fisrt instance complete to jump out of the loop
        while (process_cnt > 0 and stop_cnt < 1000):
            process_cnt = int(os.popen("ps -ef | grep 'sockeye.translate' | grep -v grep | wc -l").read().split('\n')[0])
            time.sleep(0.1)
            stop_cnt += 1
        end = time.time()
        #merge each part into one complete output file
        os.system("find . -name '*.result.en' | sort | xargs cat | head -n %s > %s" % (str(lines), fileOutput))
        #rm temp file
        os.system("rm mlt-nmt* -rf")

    total_time = end - start

    print("Instance nums: %d" %  total_cores)
    print("Total Processed lines: %d" % lines)
    print("Total time(s): %.3f s" % total_time)
    print("Speed: %.3f sent/sec" % (lines / total_time))
    
