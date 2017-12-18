. params.txt

SOURCE=$(echo $PAIR | cut -d- -f1)
TARGET=$(echo $PAIR | cut -d- -f2)

[[ ! -d model ]] && mkdir model
[[ ! -d validate ]] && mkdir validate

if [[ ! -s model/vocab.$TARGET.yml ]]; then
    echo "Creating shared vocabularies..."
    cat $DATADIR/$PAIR/train.tok.bpe.{$SOURCE,$TARGET} | $MARIAN/build/marian-vocab | tee model/vocab.$SOURCE.yml > model/vocab.$TARGET.yml
fi

$MARIAN/build/marian \
    --model $(pwd)/model/model.npz --type transformer \
    --train-sets $DATADIR/$PAIR/train.tok.bpe.{$SOURCE,$TARGET} \
    --vocabs $(pwd)/model/vocab.$SOURCE.yml $(pwd)/model/vocab.$TARGET.yml \
    --max-length 100 \
    --mini-batch-fit -w 13000 --maxi-batch 1000 \
    --early-stopping 10 \
    --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
    --valid-metrics cross-entropy perplexity translation \
    --valid-sets $DATADIR/$PAIR/dev.bpe.{$SOURCE,$TARGET} \
    --valid-script-path ../validate-transformer.sh \
    --valid-log model/valid.log --log model/train.log \
    --valid-translation-output validate/dev.output.bpe --quiet-translation \
    --valid-mini-batch 64 \
    --beam-size 6 --normalize 0.6 \
    --enc-depth 6 --dec-depth 6 \
    --transformer-heads 8 \
    --transformer-postprocess-emb d \
    --transformer-postprocess dan \
    --transformer-dropout 0.1 --label-smoothing 0.1 \
    --learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
    --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
    --exponential-smoothing \
    --tied-embeddings-all \
    --devices $DEVICES --sync-sgd --seed 1111 -T .
