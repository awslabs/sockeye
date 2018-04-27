#!/bin/sh
# Describtion: This scipt is used for CPU  Multi-instance Translate, 
#              which can  greatly speedup translate performance.
# FileName: mlt-trans.sh
# Usage: ./mlt-trans.sh model_name file_to_translate batch_size [benchmark]
# Version: 2.0


# split the real testset file to multi small files for multi-instances
# param 1: file to translate
# param 2: numbers of instances
function split_file(){
    file=$1
	workers=$2
	lines=`cat $1 | wc -l`

	quot=$(($lines/$workers))
	rema=$(($lines%workers))
	if [ $rema -ne 0 ];then
		quot=$(($quot+1))
		cat $file|head -n $(($quot*$workers-$lines)) >> $file
	fi
	split -l $quot $file -d -a 2 mlt-gnmt.log.
	head -n $lines $file > temp.log
	mv temp.log $file
}

# benchmark: inference use dummy data
# param 1: total_cores
# param 2: model
# param 3: file to translate
# param 4: batch_size
function benchmark(){
    for ((i=0; i<$1; i++))
	do
		if [ $i == $[$1-1] ]
		then
 			taskset -c $i-$i python3 -m sockeye.translate -m $2 -i $3 -o mlt-en.result.en --batch-size $4 --output-type benchmark --use-cpu > /dev/null 2>&1 
		else
			taskset -c $i-$i python3 -m sockeye.translate -m $2 -i $3 -o mlt-en.result.en --batch-size $4 --output-type benchmark --use-cpu  > /dev/null 2>&1 &
		fi
	done
}

# readdata: inference use input file
# param 1: total_cores
# param 2: model
# param 3: batch_size
function realdata(){
	for ((i=0; i<$1; i++))
	do
	    file=''
		if [ $i -lt 10 ];then
			file="mlt-gnmt.log.0"$i
		else
			file="mlt-gnmt.log."$i
		fi

		if [ $i == $[$1-1] ]
		then
 			taskset -c $i-$i python3 -m sockeye.translate -m $2 -i $file -o $file'.result.en' --batch-size $3 --output-type benchmark --use-cpu > /dev/null 2>&1 
		else
			taskset -c $i-$i python3 -m sockeye.translate -m $2 -i $file -o $file'.result.en' --batch-size $3 --output-type benchmark --use-cpu  > /dev/null 2>&1 &
		fi
	done
}

if [ "$#" -eq "3" ];then
	model=$1
	filetotrans=$2
	batch_size=$3
	run_benchmark="true"
elif [ "$#" -eq "4" ];then
	model=$1
	filetotrans=$2
	batch_size=$3
	run_benchmark=$4
	if [ ! -d $1 ];then
		echo "$1 model does not exist!"
		exit
	fi
	if [ ! -f $2 ];then
		echo "$2 translate file does not exist"
		exit
	fi
else
	echo 'Run error!'
	echo 'Usage: ./mlt-trans.sh model_name file_to_translate batch_size [-benchmark true/false]'
	echo 'Example: ./mlt-trans.sh wmt_model newstest2016.tc.BPE.de 32 true'
	exit
fi

if [ ! -d $1 ];then
	echo "$1 model does not exist!"
	exit
fi
if [ ! -f $2 ];then
	echo "$2 translate file does not exist"
	exit
fi

export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
SOCKETS=$(grep -w "physical id" /proc/cpuinfo | sort -u | wc -l)
CORES=$(grep -w "core id" /proc/cpuinfo | sort -u | wc -l)
total_cores=$[$SOCKETS*$CORES]
loop=$total_cores

sentences=`cat $filetotrans | wc -l`
line=$sentences
split_file $filetotrans $total_cores
export OMP_NUM_THREADS=$total_cores

echo 'Translating...'
start=$(($(date +%s%N)/1000000000))
if [ $run_benchmark == "true" ];then
	benchmark $total_cores $model $filetotrans $batch_size
	sentences=$(($sentences*$loop))
else
	realdata $total_cores $model $batch_size
fi
end=$(($(date +%s%N)/1000000000))

output=$2".en"
find . -name "*.result.en" | sort | xargs cat | head -n $line > $output
rm mlt-gnmt.log* -f
rm mlt-en.result.en* -f 
echo "output to file ", $output

diff=$(($end-$start))
speed=`echo "scale=3;$sentences/$diff"|bc`

echo "Instance nums : " $total_cores
echo "Total Processed lines : " $sentences
echo "Total time(s) : " `echo "scale=3;$diff"|bc`
echo "speed : " $speed "sent/sec"
