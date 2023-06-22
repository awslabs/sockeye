import subprocess
import os
import sys
import argparse
import logging
from typing import Any
from collections import defaultdict
import gzip
import os
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
from contextlib import ExitStack
import math
import sacrebleu

TEST_SETS = ["EMEA", "EUbookshop", "Europarl", "JRC-Acquis"]

# Create a logger with basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_out_aligned_sentences(alignments_file, output_dir, et_sentences, lt_sentences, num_tmp_files, excluded_sentences_file):
    assert et_sentences.exists(), f"{et_sentences} not found."
    assert lt_sentences.exists(), f"{lt_sentences} not found."
    with open(excluded_sentences_file) as fin:
        excluded_sentences = set(line.rstrip("\n") for line in fin)

    et_sent_ids = set()
    et_sent_id_to_alignment_id = defaultdict(list)
    lt_sent_ids = set()
    lt_sent_id_to_alignment_id = defaultdict(list)
    expected_alignments = set()

    num_excluded = 0
    with gzip.open(alignments_file, "rt") as indata:
        sent_alignment_id = 0
        for line in indata:
            et_sent_id, lt_sent_id = line.rstrip("\n").split("\t")
            if et_sent_id in excluded_sentences or lt_sent_id in excluded_sentences:
                num_excluded += 1
                continue
            et_sent_ids.add(et_sent_id)
            et_sent_id_to_alignment_id[et_sent_id].append(sent_alignment_id)
            lt_sent_ids.add(lt_sent_id)
            lt_sent_id_to_alignment_id[lt_sent_id].append(sent_alignment_id)
            expected_alignments.add(sent_alignment_id)
            sent_alignment_id += 1
    logger.info(f"Read {len(expected_alignments)} alignments from {alignments_file} ({num_excluded} were exlcuded).")

    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory {output_dir}")
        os.makedirs(output_dir)

    write_out_aligned_sentences_for_lang(et_sentences, output_dir / "et_sentences", "aligned_sentences.et.txt", et_sent_ids, et_sent_id_to_alignment_id, 53_279_844, expected_alignments, num_tmp_files)
    write_out_aligned_sentences_for_lang(lt_sentences, output_dir / "lt_sentences", "aligned_sentences.lt.txt", lt_sent_ids, lt_sent_id_to_alignment_id, 63_556_320, expected_alignments, num_tmp_files)


def write_out_aligned_sentences_for_lang(sentence_tsv_file: Path, folder: Path, output_fname: str, sent_ids, sent_id_to_alignment_id, num_sents, expected_alignments, num_tmp_files = 256):
    if not os.path.exists(folder):
        logger.info(f"Creating {folder}")
        os.makedirs(folder)
    output_file = folder / output_fname
    if output_file.exists():
        logger.info(f"Skipping {output_file}, reusing it.")
        return
    num_lines_per_file = math.ceil(len(expected_alignments) / num_tmp_files)

    alignments_found = set()
    with ExitStack() as stack:
        tmp_files = []
        for i in range(0, num_tmp_files):
            tmp_files.append(stack.enter_context(open(folder / f"{i}_aligned_sentences.tsv", "wt")))
        logger.info(f"Reading {sentence_tsv_file} and writing aligned sentences to {output_file}.")
        covered_sent_ids = set()
        with tqdm(total=num_sents) as pbar:
            with gzip.open(sentence_tsv_file, "rt") as tsv_file:
                header = next(tsv_file).rstrip("\n").split("\t")
                row_idx = 0
                for row_idx, line in enumerate(tsv_file):
                    if row_idx % 100_000 == 0:
                        pbar.update(n=row_idx - pbar.n)
                    cols = line.rstrip("\n").split("\t")
                    assert len(cols) == len(header), f"Row {row_idx} has {len(cols)} columns, but header has {len(header)} columns."
                    row = dict(zip(header, cols))
                    sentence_id = row['SentenceId']
                    if sentence_id in sent_ids:
                        if sentence_id in covered_sent_ids:
                            # Note: this can happen when strings appear with multiple language ids. Their text content will be identical though
                            continue
                        covered_sent_ids.add(sentence_id)
                        # Match
                        alignment_ids = sent_id_to_alignment_id[sentence_id]
                        for alignment_id in alignment_ids:
                            sentence_text = row['Sentence']
                            assert "\n" not in sentence_text, f"Sentence text contains newline: '{sentence_text}'"
                            assert "\r" not in sentence_text, f"Sentence text contains newline: '{sentence_text}'"
                            assert "\t" not in sentence_text, f"Sentence text contains tab: '{sentence_text}'"
                            assert alignment_id not in alignments_found, f"alignment_id={alignment_id} appears multiple times."
                            tmp_file_for_alignment = alignment_id // num_lines_per_file
                            tmp_files[tmp_file_for_alignment].write(f"{alignment_id}\t{sentence_text}\n")
                            alignments_found.add(alignment_id)
        if alignments_found != expected_alignments:
            # Missing alignments:
            missing_alignments = expected_alignments.difference(alignments_found)
            raise RuntimeError(f"Not all alignments were found in {sentence_tsv_file}. Missing ET: {missing_alignments}")

    # Sort and merge:
    with open(output_file, "wt") as output_file:
        prev_align_id = -1
        for i in range(0, num_tmp_files):
            fname = folder / f"{i}_aligned_sentences.tsv"
            with open(fname, "rt") as tmp_file:
                lines = [line.split("\t") for line in tmp_file]
                lines = [(int(alignment_id), sentence_text) for alignment_id, sentence_text in lines]
                for alignment_id, sentence_text in sorted(lines):
                    # Quick sanity check that we cover all alignments and they are in order
                    assert alignment_id == prev_align_id + 1
                    output_file.write(sentence_text)
                    prev_align_id = alignment_id
            # delete temporary file
            os.remove(fname)


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

    data_set = "dev"
    # Files for a combined dev set
    src_file = f"{data_folder}/preproc/{data_set}.{src_lang}"
    tgt_file = f"{data_folder}/preproc/{data_set}.{tgt_lang}"
    if not os.path.isfile(f"{tgt_file}.bpe"):
        if os.path.isfile(f"{tgt_file}"):
            os.remove(f"{tgt_file}")
        if os.path.isfile(f"{src_file}.{src_lang}"):
            os.remove(f"{src_file}.{src_lang}")

        for test_set in TEST_SETS:
            logger.info(f"Preprocessing {data_set} set: {test_set_folder}/{test_set}.{data_set}.{src_lang}-{tgt_lang}.*")
            assert os.path.exists(f"{test_set_folder}/{test_set}.{data_set}.{src_lang}-{tgt_lang}.{tgt_lang}")
            assert os.path.exists(f"{test_set_folder}/{test_set}.{data_set}.{src_lang}-{tgt_lang}.{src_lang}")
            log_and_run(f"cat {test_set_folder}/{test_set}.{data_set}.{src_lang}-{tgt_lang}.{tgt_lang} >> {tgt_file}")
            log_and_run(f"cat {test_set_folder}/{test_set}.{data_set}.{src_lang}-{tgt_lang}.{src_lang} >> {src_file}")

        for lang in [src_lang, tgt_lang]:
            bped_data = f"{data_folder}/preproc/{data_set}.{lang}.bpe"
            if not os.path.isfile(bped_data):
                bpe_model[lang](f"{data_folder}/preproc/{data_set}.{lang}", bped_data)

    data_set = "test"
    for test_set in TEST_SETS:
        logger.info(f"Preprocessing {data_set} set: {test_set_folder}/{test_set}.{data_set}.{src_lang}-{tgt_lang}.*")
        test_data = {
            src_lang: f"{test_set_folder}/{test_set}.{data_set}.{src_lang}-{tgt_lang}.{src_lang}",
            tgt_lang: f"{test_set_folder}/{test_set}.{data_set}.{src_lang}-{tgt_lang}.{tgt_lang}"
        }
        test_data_bpe = {
            src_lang: f"{data_folder}/preproc/{data_set}.{test_set}.{src_lang}.bpe",
            tgt_lang: f"{data_folder}/preproc/{data_set}.{test_set}.{tgt_lang}.bpe"
        }

        for lang in [src_lang, tgt_lang]:
            assert os.path.exists(test_data[lang]), f"Could not find test file {test_data[lang]}."
            bped_data = test_data_bpe[lang]
            print(bped_data)
            if not os.path.isfile(bped_data):
                bpe_model[lang](test_data[lang], bped_data)


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
    return model_dir


def evaluate_model(model_dir, data_folder, test_folder, src_lang, tgt_lang, batch_size):
    test_set_scores = {}
    for test_set in TEST_SETS:
        test_input = f"{data_folder}/preproc/test.{test_set}.{src_lang}.bpe"
        test_output = f"{data_folder}/preproc/test.{test_set}.{tgt_lang}.bpe.modelout"
        test_output_debpe = f"{data_folder}/preproc/test.{test_set}.{tgt_lang}.modelout"

        if not os.path.isfile(test_output) or os.stat(test_output).st_size == 0:
            log_and_run(f"sockeye-translate -m {model_dir} -i {test_input} -o {test_output} --batch-size {batch_size}")
        else:
            logger.info(f"Test output {test_output} already exists.")

        if not os.path.isfile(test_output_debpe) or os.stat(test_output_debpe).st_size == 0:
            log_and_run(f"cat {test_output} | sed 's,@@ ,,g' > {test_output_debpe}")

        print(f"Evaluating {test_set}")
        # at wmt23_data_task_data/top1_cosine_eval/preproc/test.JRC-Acquis.lt.modelout | sacrebleu -l lt-et ./wmt23_data_task_data/testsets_v2//JRC-Acquis.test.et-lt.lt
        # log_and_run(f"cat {test_output_debpe} | sacrebleu -l {tgt_lang}-{src_lang} {reference_file}")
        reference_file = f"{test_folder}/{test_set}.test.{src_lang}-{tgt_lang}.{tgt_lang}"
        hypothesis_lines = [line.rstrip("\n") for line in open(test_output_debpe).readlines()]
        reference_lines = [line.rstrip("\n") for line in open(reference_file).readlines()]
        score = sacrebleu.corpus_bleu(hypothesis_lines, [reference_lines])
        test_set_scores[test_set] = score

    for test_set, score  in test_set_scores.items():
        score_rounded = round(score.score, 1)
        print(f"{test_set}: {score_rounded:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Extract sentence text, then train a Sockeye model and run evaluation to produce BLEU scores for the WMT23 data shared task.")
    parser.add_argument('-a', '--alignments-file', type=str, help='Input file with et-lt alignments (tsv) format.', required=True)
    parser.add_argument('-o', '--working-dir', type=str, help='Output directory (we 1. extract sentences, 2. train a model and 3. store BLEU scores in this folder.) Expected to be empty.', required=True)
    parser.add_argument('--et-sentences', type=str, help="path to sentences.et.tsv.gz", default="wmt23_data_task_data/sentences/sentences.et.tsv.gz", required=False)
    parser.add_argument('--lt-sentences', type=str, help="path to sentences.lt.tsv.gz", default="wmt23_data_task_data/sentences/sentences.lt.tsv.gz", required=False)
    parser.add_argument('--excluded-sentences', type=str, help="path to exclude_sent_ids_et-lt.txt", default="wmt23_data_task_data/exclude_sent_ids_et-lt.txt", required=False)
    
    parser.add_argument('--num-tmp-files', type=int, help="Number of temporary files to write (num. alignments // num_tmp_files lines of text need to fit in memory) ", default=256, required=False)

    parser.add_argument("--test-set-dir", help="Path to the directory containing the test sets.", required=True)
    parser.add_argument("--batch-size-per-gpu", type=int, default=4096, help="Batch size for training (in tokens).", required=True)
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs available.", required=True)
    parser.add_argument("--max-preproc-processes", type=int, default=8, help="Number of processes working on data preparation.")
    parser.add_argument("--inference-batch-size", type=int, default=1, help="Batch size for decoding.")
    args = parser.parse_args()
    src_lang = "et"
    tgt_lang = "lt"

    # Check if CLI tools exist in PATH
    required_tools = ["subword-nmt", "sockeye-train", "sockeye-translate", "sacrebleu", "torchrun"]
    for tool in required_tools:
        if subprocess.run(f"command -v {tool}", shell=True, check=False, stdout=subprocess.DEVNULL).returncode != 0:
            print(f"Error: {tool} is not found in PATH. Please make sure the tool is installed and available.")
            sys.exit(1)

    # First let's retrieve sentence text given sentence ids
    alignments_file = args.alignments_file
    working_dir = Path(args.working_dir)
    et_sentences = Path(args.et_sentences)
    lt_sentences = Path(args.lt_sentences)
    num_tmp_files = args.num_tmp_files
    excluded_sentences_file = args.excluded_sentences
    write_out_aligned_sentences(alignments_file, working_dir, et_sentences, lt_sentences, num_tmp_files, excluded_sentences_file)

    # Now let's run training
    num_gpus = args.num_gpus
    if num_gpus is None:
        import torch
        logger.info(f"Using {torch.cuda.device_count()} GPUs.")
        num_gpus = torch.cuda.device_count()

    effective_batch_size = 400_000 / num_gpus
    update_interval = int(effective_batch_size / args.batch_size_per_gpu)

    preprocess_data(args.working_dir, args.test_set_dir, src_lang=src_lang, tgt_lang=tgt_lang)
    model_dir = train_model(args.working_dir, src_lang=src_lang, tgt_lang=tgt_lang, batch_size=args.batch_size_per_gpu, update_interval=update_interval, max_preproc_processes=args.max_preproc_processes)
    inference_batch_size = args.inference_batch_size
    evaluate_model(model_dir, args.working_dir, args.test_set_dir, src_lang=src_lang, tgt_lang=tgt_lang, batch_size=inference_batch_size)

if __name__ == "__main__":
    main()
