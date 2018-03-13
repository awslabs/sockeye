#!/bin/bash

. env.sh

USRDIR=t2t_usr
DATADIR=data_$PAIR
TMPDIR=tmp_$PAIR
PROBLEM=wmt17_${SOURCE}_${TARGET}_bpe32k
HPARAMS=transformer_wmt17_base
MODEL=transformer.$PROBLEM.$HPARAMS

BEAM=5
ALPHA=1.0
INPUTS=$DATDIR/$PAIR/test.bpe.$SOURCE
OUTPUTS=$(basename $INPUTS).transformer.$HPARAMS.$PROBLEM.beam$BEAM.alpha$ALPHA.decodes

t2t-decoder --t2t_usr_dir=$USRDIR --data_dir=$DATADIR --tmp_dir=$TMPDIR --problems=$PROBLEM \
    --model=transformer \
    --hparams_set=$HPARAMS \
    --output_dir=$OUTPUT \
    --decode_hparams="beam_size=$BEAM,alpha=$ALPHA,batch_size=16,use_last_position_only=True" \
    --decode_from_file=$INPUTS \
    --decode_to_file=$OUTPUTS

cat $OUTPUTS | \
    sed 's/\@\@ //g' | \
    $MOSES/scripts/tokenizer/detokenizer.perl -q -l $TARGET | \
    tee $OUTPUTS.detok | \
    sacrebleu -t wmt17 -l $PAIR > $OUTPUT.bleu
