# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
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

"""
Scoring CLI.
"""
import argparse
import sys
import os
import time
from math import ceil
from contextlib import ExitStack
from typing import Callable, Optional, Iterable, List, NamedTuple, Dict, Tuple, cast
from collections import defaultdict

import mxnet as mx
import numpy as np

import sockeye
from . import arguments
from . import constants as C
from . import data_io
from . import vocab
from . import model
from . import utils
from . import loss
from . import translate

from sockeye.log import setup_main_logger
from sockeye.utils import acquire_gpus, get_num_gpus, log_basic_info
from sockeye.utils import check_condition



class ScoringModel(model.SockeyeModel):
    """
    Defines a model to score input/output data.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU)
    :param data_iter: The iterator over the data set.
    :param config: Configuration object holding details about the model.
    :param checkpoint: Checkpoint to load. If None, finds best parameters in model_folder.
    :param bucketing: If True bucketing will be used, if False the computation graph will always be
            unrolled to the full length.
    """

    def __init__(self,
                 model_folder: str,
                 context: List[mx.context.Context],
                 data_iter: data_io.BaseParallelSampleIter,
                 config: model.ModelConfig,
                 checkpoint: Optional[int] = None,
                 bucketing: bool = None) -> None:
        super().__init__(config)
        self.context = context
        self.bucketing = bucketing
        self._build_model_components()
        self.module = self._build_module(data_iter)
        
        self.config=config
        self.batch_size=data_iter.batch_size
        self.fname_params = os.path.join(model_folder, C.PARAMS_NAME % checkpoint if checkpoint else C.PARAMS_BEST_NAME)
        
        self.module.bind(data_shapes=data_iter.provide_data, label_shapes=data_iter.provide_label, for_training=False, force_rebind=True, grad_req='null')
       
        self.load_params_from_file(self.fname_params)
        self.module.init_params(arg_params=self.params, allow_missing=False)

    
    def _build_module(self, data_iter: data_io.BaseParallelSampleIter):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """

        source = mx.sym.Variable(C.SOURCE_NAME)
        source_words = source.split(num_outputs=self.config.config_embed_source.num_factors,
                                    axis=2, squeeze_axis=True)[0]
        source_length = utils.compute_lengths(source_words)
        target = mx.sym.Variable(C.TARGET_NAME)
        target_length = utils.compute_lengths(target)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        model_loss = loss.get_loss(self.config.config_loss)

        data_names = [x[0] for x in data_iter.provide_data]
        label_names = [x[0] for x in data_iter.provide_label]
        #scores = [[None] for x in data_iter.provide_label]
        

        def sym_gen(seq_lens):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # target embedding
            (target_embed,
             target_embed_length,
             target_embed_seq_len) = self.embedding_target.encode(target, target_length, target_seq_len)

            # encoder
            # source_encoded: (batch_size, source_encoded_length, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source_embed,
                                                           source_embed_length,
                                                           source_embed_seq_len)

            # decoder
            # target_decoded: (batch-size, target_len, decoder_depth)
            target_decoded = self.decoder.decode_sequence(source_encoded, source_encoded_length, source_encoded_seq_len,
                                                          target_embed, target_embed_length, target_embed_seq_len)

            # target_decoded: (batch_size * target_seq_len, rnn_num_hidden)
            target_decoded = mx.sym.reshape(data=target_decoded, shape=(-3, 0))

            # output layer
            logits = self.output_layer(target_decoded)
            
            #probs = mx.symbol.softmax(logits)
            # TODO: check if we can use mx.symbol.softmax_cross_entropy
            #probs= mx.symbol.softmax_cross_entropy(logits, labels)
            #token_probs = mx.symbol.pick(probs, labels)
            #log_probs = - mx.symbol.log(token_probs)
            #scores = mx.symbol.sum(log_probs, axis=0)
            #return scores, data_names, label_names
            
            probs = model_loss.get_loss(logits, labels)
            return mx.symbol.Group(probs), data_names, label_names
        

        if self.bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", data_iter.default_bucket_key)
            return mx.mod.BucketingModule(sym_gen=sym_gen,
                                          logger=logger,
                                          default_bucket_key=data_iter.default_bucket_key,
                                          context=self.context)
        else:
            logger.info("No bucketing. Unrolled to (%d,%d)",
                        self.config.config_data.max_seq_len_source, self.config.config_data.max_seq_len_target)
            symbol, _, __ = sym_gen(data_iter.buckets[0])
            return mx.mod.Module(symbol=symbol,
                                 data_names=data_names,
                                 label_names=label_names,
                                 logger=logger,
                                 context=self.context)


    ## TODO: print alignments?
    def score(self, 
              data_iter: data_io.BaseParallelSampleIter,
              mapid: defaultdict(lambda: defaultdict(int)),
              batch_size: int,
              normalize: Optional[bool] = True) -> Dict[int,int]:
        """
        Reads batches of input/output pairs and scores them.
        :param data_iter: The iterator over the data set.
        :param mapid: dictionary that maps input order to samples in buckets.
        :param batch_size: Batch size.
        :param normalize: If true, normalizes score by length of target.
        :return: Dictionary where key=input_id, value=score.
        """
        results = dict()
        
        for nbatch, batch in enumerate(data_iter):
            self.module.forward(batch, is_train=False)
            outputs = self.module.get_outputs()
            #print("outputs {} {}".format(len(outputs[0]),outputs))
            
            ## split output array into probs per batch
            sample_length = int(len(outputs[0])/batch_size)
           
            probs = mx.nd.array(outputs[0]) ## shape is (t_len*batch_size, t_vocab)
            
            probs_per_batch = [probs[i*sample_length:(i+1)*sample_length] for i in range(batch_size)]
            
            for sample_number, sample_probs in enumerate(probs_per_batch):
                if sample_number in mapid[nbatch]:
                    labels = batch.label[0][sample_number]
                    #print("sample nr, {}, probs {}, labels {}".format(sample_number,sample_probs.shape, labels.shape))
                    scores = mx.nd.pick(sample_probs, labels)
                    scores = scores.asnumpy()
                    log_probs = - np.log(scores)
                    score = np.nansum(log_probs)
                    # TODO: inf?
                    if normalize:
                        score = np.mean(log_probs)
                    sentence_id = mapid[nbatch][sample_number]  
                    results[sentence_id] = score
                #else:
                    #logger.info("sample was filler")
        return results        


def load_models(context: mx.context.Context,
                batch_size: int,
                model_folders: List[str],
                data_iters: List[data_io.BaseParallelSampleIter],
                configs: List[data_io.DataConfig],
                checkpoints: Optional[List[int]] = None,
                bucketing: bool = None) -> List[ScoringModel]: 
    
    """
    Loads a list of models for scoring.

    :param context: MXNet context to bind modules to.
    :param batch_size: Batch size.
    :param model_folders: List of model folders to load models from.
    :param data_iter: List of iterators over the data set (one per model, can use different vocabularies).
    :param configs: List of configuration objects holding details about the models (one per model).
    :param checkpoints: List of checkpoints to use for each model in model_folders. Use None to load best checkpoint.
    :return: List of models.
    """

    models = []
    if checkpoints is None:
        checkpoints = [None] * len(model_folders)
        
    for model_folder, checkpoint, config, data_iter in zip(model_folders, checkpoints, configs, data_iters):
        
        model = ScoringModel(model_folder=model_folder,
                               context=context,
                               data_iter=data_iter,
                               config=config,
                               checkpoint=checkpoint,
                               bucketing=bucketing)
        models.append(model)

    return models


def create_data_iter_and_vocab(args: argparse.Namespace, 
                                max_seq_len_source: int,
                                max_seq_len_target: int,
                                model_dir: str)-> Tuple['data_io.BaseParallelSampleIter', 'data_io.DataConfig']:
    """
    Create the data iterator.

    :param args: Arguments as returned by argparse.
    :param max_seq_len_source: Max length input.
    :param max_seq_len_target: Max length output.
    :model_dir: model folder to load vocabularies and config from.
    :return: The data iterator.
    """
    ### 

    batch_num_devices = 1 if args.use_cpu else sum(-di if di < 0 else 1 for di in args.device_ids)
    batch_by_words = args.batch_type == C.BATCH_TYPE_WORD
        
    source_vocabs = vocab.load_source_vocabs(model_dir)
    target_vocab = vocab.vocab_from_json(os.path.join(model_dir, C.VOCAB_TRG_NAME))
       
    ## Recover the vocabulary path from the data info file:
    data_info = cast(data_io.DataInfo, sockeye.config.Config.load(os.path.join(model_dir, C.DATA_INFO)))
    source_vocab_paths = data_info.source_vocabs
    target_vocab_path = data_info.target_vocab
    
    # get max_seq_len_source and max_seq_len_target from config, warn if smaller than given values
    config = model.SockeyeModel.load_config(os.path.join(model_dir, C.CONFIG_NAME))
    if max_seq_len_source > config.config_data.max_seq_len_source:
            logger.warning("Source sentence of length %d in test set exceeds maximum source sentence length in config of %d",max_seq_len_source, config.config_data.max_seq_len_source)
    if max_seq_len_target > config.config_data.max_seq_len_target:
            logger.warning("Target sentence of length %d in test set exceeds maximum target sentence length in config of %d",max_seq_len_target, config.config_data.max_seq_len_target)       
            
    
    
    check_condition(len(args.source_factors) == len(args.source_factors_num_embed),
                        "Number of source factor data (%d) differs from provided source factor dimensions (%d)" % (len(args.source_factors), len(args.source_factors_num_embed)))
    
    sources = [args.source] + args.source_factors
    sources = [str(os.path.abspath(source)) for source in sources]
    
    #print("sources {} source vocabs {}".format(sources, source_vocabs))
    sources_sentences = [data_io.SequenceReader(source, vocab, add_bos=False) for source, vocab in zip(sources, source_vocabs)]
    target_sentences = data_io.SequenceReader(args.target, target_vocab, add_bos=True, limit=None)
    
    ## Pass 1: get target/source length ratios.
    length_statistics = data_io.analyze_sequence_lengths(sources, args.target, source_vocabs, target_vocab, max_seq_len_source, max_seq_len_target)
   
   ## define buckets
    bucketing=not args.no_bucketing
    buckets = data_io.define_parallel_buckets(max_seq_len_source, max_seq_len_target, args.bucket_width, length_statistics.length_ratio_mean) if bucketing else [(max_seq_len_source, max_seq_len_target)]

    
    ### get iter
    ## 2. pass: Get data statistics
    data_statistics = data_io.get_data_statistics(sources_sentences, 
                                                          target_sentences, 
                                                          buckets,
                                                          length_statistics.length_ratio_mean, length_statistics.length_ratio_std,
                                                          source_vocabs, 
                                                          target_vocab)

    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets,
                                                   args.batch_size,
                                                   batch_by_words, 
                                                   batch_num_devices,
                                                   data_statistics.average_len_target_per_bucket)

    data_statistics.log(bucket_batch_sizes)

    data_loader = data_io.RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=target_vocab[C.EOS_SYMBOL],
                                           pad_id=C.PAD_ID)

    parallel_data = data_loader.load(sources_sentences, target_sentences,
                                     data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes, args.fill_up) 
    map_buckets2sentence_ids = data_loader.map_buckets2sentence_ids
    
    data_iter = data_io.ParallelSampleIter(parallel_data,
                                    buckets,
                                    args.batch_size,
                                    bucket_batch_sizes,
                                    no_shuffle=True)
    
    return data_iter, config, map_buckets2sentence_ids


def get_max_source_and_target(args: argparse.Namespace) -> Tuple[int, int]:
    source_lines = utils.smart_open(args.source).readlines()
    target_lines = utils.smart_open(args.target).readlines()
    max_len_source = max([len(line.rstrip().split()) for line in source_lines])
    max_len_target = max([len(line.rstrip().split()) for line in target_lines])
    # +1 for EOS
    return max_len_source+1, max_len_target+1


def main():
    ### TODO adapt output_handler?
    ### TODO: test with factors
    params = argparse.ArgumentParser(description='Scoring CLI')
    
    arguments.add_scoring_args(params)
    args = params.parse_args()

    
    global logger
    if args.output is not None:
        logger = setup_main_logger(__name__,
                                   console=not args.quiet,
                                   file_logging=True,
                                   path="%s.%s" % (args.output, C.LOG_NAME))
    else:
        logger = setup_main_logger(__name__,
                                   console=not args.quiet,
                                   file_logging=False)
    utils.log_basic_info(args)
    
    if args.checkpoints is not None:
        check_condition(len(args.checkpoints) == len(args.models), "must provide checkpoints for each model")  
    
    with ExitStack() as exit_stack:
        context = translate._setup_context(args, exit_stack)
        
       
        ## if --max-seq-len given, use this, else get maximum sentence length from test data
        if(args.max_seq_len is not None):
            max_len_source, max_len_target = args.max_seq_len
        else:
            max_len_source, max_len_target = get_max_source_and_target(args)
        logger.info("Using max length source %d, max length target %d", max_len_source, max_len_target)
       
        ## create iterator for each model (vocabularies can be different)
        data_iters, configs, mapids = [], [], []
        for model in args.models:
            data_iter, config, mapid = create_data_iter_and_vocab(args=args,
                                                           max_seq_len_source=max_len_source, max_seq_len_target=max_len_target, model_dir=model)
            data_iters.append(data_iter)
            configs.append(config)
            mapids.append(mapid)
            
        models = load_models(context,
             args.batch_size,
             args.models,
             data_iters,
             configs,
             args.checkpoints,
             not args.no_bucketing)
        
        results = []
        for model, data_iter, mapid in zip(models, data_iters, mapids):
           result = model.score(data_iter, mapid, args.batch_size)
           results.append(result)
             
        for sentence_id in range(len(results[0])):
            for model in range(len(results)):
                print("model {} : score {}".format(model, results[model][sentence_id]), end=" ")
            print()
     
           
        
if __name__ == '__main__':
    main()        
        
        
        
