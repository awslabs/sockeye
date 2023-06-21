import subprocess
import os
import sys
import argparse
import logging
from typing import Any

TEST_SETS = ["EMEA", "EUbookshop", "Europarl", "JRC-Acquis"]

# Create a logger with basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_and_run(cmd):
    logger.info(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


class BPE:
    def __init__(self, bpe_model_file, vocab_file, min_vocab_freq):
        self.bpe_model_file = bpe_model_file
        self.vocab_file = vocab_file
        self.min_vocab_freq = min_vocab_freq

    def __call__(self, input_file, output_file):
        # log_and_run(f"fast applybpe {data_folder}/preproc/train.{lang}.bpe {data_folder}/train.{lang} {bpe_model_file} {bpe_vocab[lang]}")
        log_and_run(f"subword-nmt apply-bpe -c {self.bpe_model_file} --vocabulary {self.vocab_file} --vocabulary-threshold {self.min_vocab_freq} < {input_file} > {output_file}")


def preprocess_data(data_folder, test_set_folder, src_lang, tgt_lang):
    preproc_folder = "preproc"
    vocab_size = 32000
    min_vocab_freq = 100

    data_folder = data_folder.rstrip("/")
    os.makedirs(f"{data_folder}/{preproc_folder}/", exist_ok=True)

    raw_src_data = f"{data_folder}/{src_lang}_sentences/aligned_sentences.{src_lang}.txt"
    raw_tgt_data = f"{data_folder}/{tgt_lang}_sentences/aligned_sentences.{tgt_lang}.txt"
    assert os.path.exists(raw_src_data), f"Source data does not exist {raw_src_data}."
    assert os.path.exists(raw_tgt_data), f"Target data does not exist {raw_tgt_data}."
    raw_data = {
        src_lang: raw_src_data,
        tgt_lang: raw_tgt_data,
    }

    bpe_train_data = f"{data_folder}/{preproc_folder}/train.bpe_train"
    if not os.path.isfile(bpe_train_data):
        logger.info("Creating {bpe_train_data}.")
        log_and_run(f"cat {raw_src_data} {raw_tgt_data} | shuf > {bpe_train_data}")

    # Create BPE model
    bpe_model_file = f"{data_folder}/{preproc_folder}/bpe_model"
    if not os.path.isfile(bpe_model_file):
        # log_and_run(f"fast learnbpe {vocab_size} {bpe_train_data} > {data_folder}/{preproc_folder}/bpe_model")
        log_and_run(f"subword-nmt learn-bpe -s {vocab_size} < {bpe_train_data} > {bpe_model_file}")

    # Create BPE vocab
    bpe_model = {}
    for lang in [src_lang, tgt_lang]:
        bpe_vocab = f"{bpe_model_file}.{lang}.vocab"
        if not os.path.isfile(bpe_vocab):
            # log_and_run(f"fast applybpe_stream {bpe_model_file} < {data_folder}/train.{lang} | fast getvocab - > {bpe_vocab}")
            log_and_run(f"subword-nmt apply-bpe -c {bpe_model_file} < {raw_data[lang]} | subword-nmt get-vocab > {bpe_vocab}")
        bpe_model[lang] = BPE(bpe_model_file, bpe_vocab, min_vocab_freq)

    # Apply BPE
    for lang in [src_lang, tgt_lang]:
        bped_data = f"{data_folder}/preproc/train.{lang}.bpe"
        if not os.path.isfile(bped_data):
            bpe_model[lang](raw_data[lang], bped_data)

    for data_set in ["dev", "test"]:
        src_file = f"{data_folder}/preproc/{data_set}.{src_lang}"
        tgt_file = f"{data_folder}/preproc/{data_set}.{tgt_lang}"
        if not os.path.isfile(f"{tgt_file}.bpe"):
            if os.path.isfile(f"{tgt_file}"):
                os.remove(f"{tgt_file}")
            if os.path.isfile(f"{src_file}.{src_lang}"):
                os.remove(f"{src_file}.{src_lang}")

            for test_set in TEST_SETS:
                logger.info(f"Adding {data_set} set: {test_set_folder}/{test_set}.{data_set}.{src_lang}-{tgt_lang}.*")
                assert os.path.exists(f"{test_set_folder}/{test_set}.{data_set}.{src_lang}-{tgt_lang}.{tgt_lang}")
                assert os.path.exists(f"{test_set_folder}/{test_set}.{data_set}.{src_lang}-{tgt_lang}.{src_lang}")
                log_and_run(f"cat {test_set_folder}/{test_set}.{data_set}.{src_lang}-{tgt_lang}.{tgt_lang} >> {tgt_file}")
                log_and_run(f"cat {test_set_folder}/{test_set}.{data_set}.{src_lang}-{tgt_lang}.{src_lang} >> {src_file}")

            for lang in [src_lang, tgt_lang]:
                bped_data = f"{data_folder}/preproc/{data_set}.{lang}.bpe"
                if not os.path.isfile(bped_data):
                    bpe_model[lang](f"{data_folder}/preproc/{data_set}.{lang}", bped_data)


def train_model(data_folder, src_lang, tgt_lang, batch_size, update_interval, max_preproc_processes):
    if not os.path.isdir(f"{data_folder}/preproc/train.prepared"):
        log_and_run(f"python3 -m sockeye.prepare_data -s {data_folder}/preproc/train.{src_lang}.bpe -t {data_folder}/preproc/train.{tgt_lang}.bpe -o {data_folder}/preproc/train.prepared --shared-vocab --max-processes {max_preproc_processes}")

    model_size = 512
    num_heads = 8
    num_layers = 6
    ff_size = 2048
    amp = True
    checkpoint_interval = 500
    max_checkpoint_not_improved = 12
    optimizer_betas = '0.9:0.98'
    initial_learning_rate = 0.06325
    learning_rate_scheduler = 'inv-sqrt-decay'
    learning_rate_warmup = 4000
    seed = 3
    quiet_secondary_workers = True
    decode_and_evaluate = 500
    model_dir = f"{data_folder}/model"
    if not os.path.isdir(model_dir):
        log_and_run(f"torchrun --no_python --nproc_per_node 8 sockeye-train --prepared-data {data_folder}/preproc/train.prepared --validation-source {data_folder}/preproc/dev.{src_lang}.bpe --validation-target {data_folder}/preproc/dev.{tgt_lang}.bpe --output {model_dir} --num-layers {num_layers} --transformer-model-size {model_size} --transformer-attention-heads {num_heads} --transformer-feed-forward-num-hidden {ff_size} {'--amp' if amp else ''} --batch-type max-word --batch-size {batch_size} --update-interval {update_interval} --checkpoint-interval {checkpoint_interval} --max-num-checkpoint-not-improved {max_checkpoint_not_improved} --optimizer-betas {optimizer_betas} --dist --initial-learning-rate {initial_learning_rate} --learning-rate-scheduler-type {learning_rate_scheduler} --learning-rate-warmup {learning_rate_warmup} --seed {seed} {'--quiet-secondary-workers' if quiet_secondary_workers else ''} --decode-and-evaluate {decode_and_evaluate}")
    else:
        logger.info(f"Model directory {model_dir} already exists. Skipping training.")


def evaluate_model(data_folder, src_lang, tgt_lang):
    for test_set in TEST_SETS:
        test_input = f"{data_folder}/preproc/test.{test_set}.{src_lang}.bpe"
        test_output = f"{data_folder}/preproc/test.{test_set}.{tgt_lang}.bpe.modelout"
        test_output_debpe = f"{data_folder}/preproc/test.{test_set}.{tgt_lang}.modelout"

        if not os.path.isfile(test_output):
            subprocess.run(f"sockeye-translate -m model -i {test_input} -o {test_output}", shell=True, check=True)

        # TODO: replace with Python code...
        subprocess.run(f"cat {test_output} | sed 's,@@ ,,g' > {test_output_debpe}", shell=True, check=True)

        print(f"Evaluating {test_set}")
        # TODO: replace with Python code... (to just get the BLEU scores)
        subprocess.run(f"cat {test_output_debpe} | sacrebleu -l {tgt_lang}-{src_lang} {data_folder}/testsets_v1/{test_set}.dev.{src_lang}-{tgt_lang}.{tgt_lang}", shell=True, check=True)


def main():
    parser = argparse.ArgumentParser(description="Train a Sockeye model on the WMT23 data shared task data and run evaluation to produce BLEU scores.")
    parser.add_argument("--input-dir", help="Path to the folder containing the training data.", required=True)
    parser.add_argument("--test-set-dir", help="Path to the directory containing the test sets.", required=True)
    parser.add_argument("--batch-size-per-gpu", type=int, default=4096, help="Batch size for training (in tokens).", required=True)
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs available.", required=True)
    parser.add_argument("--max-preproc-processes", type=int, default=8, help="Number of processes working on data preparation.")

    # Check if CLI tools exist in PATH
    required_tools = ["subword-nmt", "sockeye-train", "sockeye-translate", "sacrebleu", "torchrun"]
    for tool in required_tools:
        if subprocess.run(f"command -v {tool}", shell=True, check=False, stdout=subprocess.DEVNULL).returncode != 0:
            print(f"Error: {tool} is not found in PATH. Please make sure the tool is installed and available.")
            sys.exit(1)

    # TODO: set the batch size and update interval correctly
    args = parser.parse_args()
    src_lang = "et"
    tgt_lang = "lt"

    num_gpus = args.num_gpus
    if num_gpus is None:
        import torch
        logger.info(f"Using {torch.cuda.device_count()} GPUs.")
        num_gpus = torch.cuda.device_count()

    effective_batch_size = 400_000 / num_gpus
    update_interval = int(effective_batch_size / args.batch_size_per_gpu)

    preprocess_data(args.input_dir, args.test_set_dir, src_lang=src_lang, tgt_lang=tgt_lang)
    train_model(args.input_dir, src_lang=src_lang, tgt_lang=tgt_lang, batch_size=args.batch_size_per_gpu, update_interval=update_interval, max_preproc_processes=args.max_preproc_processes)
    # evaluate_model(args.input_dir, src_lang=src_lang, tgt_lang=tgt_lang)

if __name__ == "__main__":
    main()
