#!/bin/sh
# Describtion: This scipt is used for CPU  Multi-instance Translate, 
#              which can  greatly speedup translate performance.
# FileName: mlt-trans.sh
# Usage: ./mlt-trans.sh model_name file_to_translate batch_size
# Version: 1.0


if [ "$#" -eq "3" ];then
	model=$1
	filetotrans=$2
	bs=$3
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
	echo 'Usage: ./mlt-trans.sh model_name file_to_translate batch_size'
	echo 'Example: ./mlt-trans.sh wmt_model newstest2016.tc.BPE.de 32'
	exit
fi

export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

SOCKETS=$(grep -w "physical id" /proc/cpuinfo | sort -u | wc -l)
CORES=$(grep -w "core id" /proc/cpuinfo | sort -u | wc -l)

total_cores=$[$SOCKETS*$CORES]

loop=$total_cores
stride=$(($total_cores/$loop))
export OMP_NUM_THREADS=$stride

echo 'Translating...'
start=$(($(date +%s%N)/1000000000))	
for ((i=0; i<$loop; i++))
	do
		startCore=$[${i}*${stride}]
		endCore=$[${startCore}+${stride}-1]
    
		if [ $i == $[$loop-1] ]
		then
 			taskset -c ${startCore}-${endCore} python3 -m sockeye.translate -m $model -i $filetotrans -o newstest2016.tc.BPE.en --batch-size $bs --output-type benchmark --use-cpu > /dev/null 2>&1 
		else
			taskset -c ${startCore}-${endCore} python3 -m sockeye.translate -m $model -i $filetotrans -o newstest2016.tc.BPE.en --batch-size $bs --output-type benchmark --use-cpu  > /dev/null 2>&1 & 
		fi
	done
end=$(($(date +%s%N)/1000000000))

diff=$(($end-$start))
sentences=`cat newstest2016.tc.BPE.de | wc -l`
speed=`echo "scale=3;$sentences*$loop/$diff"|bc`

echo "Instance nums : " $total_cores
echo "Total Processed lines : " $[$sentences*$total_cores]
echo "Total time(s) : " `echo "scale=3;$diff"|bc`
echo "speed : " $speed "sent/sec"

