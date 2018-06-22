rm -rf model

export PYTHONPATH=~/workspace/src/Sockeye
rm -rf model.regular
python3 -m sockeye.train \
	-s data/train.source -t data/train.target \
	-vs data/dev.source -vt data/dev.target \
        --encoder rnn --decoder rnn \
        --num-embed 32 --rnn-num-hidden 64 --rnn-attention-type dot --num-layers 1:1 \
        --use-cpu \
        --metrics perplexity accuracy \
        --max-num-checkpoint-not-improved 3 \
        -o model.regular \
        --shared-vocab \
	"$@"
