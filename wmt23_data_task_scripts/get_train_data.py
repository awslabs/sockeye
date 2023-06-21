# Go through `expected_output_format.et-lt.tsv.gz` and retrieve the sentences from the sentence data (sentences/sentences.et.tsv.gz and sentences/sentences.lt.tsv.gz)
from collections import defaultdict
import gzip
import os
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
from contextlib import ExitStack
import math


# Set up a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_out_aligned_sentences(sentence_tsv_file: Path, folder: Path, output_fname: str, sent_ids, sent_id_to_alignment_id, num_sents, expected_alignments, num_tmp_files = 256):
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


if __name__ == "__main__":
    # Parse CLI arguments (receive an input tsv file with et-lt alignments and an output directory)
    # Example invocation:
    # python3 get_train_data.py --alignments-file ./expected_output_format.et-lt.tsv --output-dir eval_dir
    parser = argparse.ArgumentParser(description='Evaluate sentence alignments (provide sentence alginments and get a BLEU score).')
    parser.add_argument('-a', '--alignments-file', type=str, help='Input file with et-lt alignments (tsv) format.', required=True)
    parser.add_argument('-o', '--output-dir', type=str, help='Output directory (we 1. extract sentences, 2. train a model and 3. store BLEU scores in this folder.) Expected to be empty.', required=True)
    parser.add_argument('--et-asentences', type=str, help="path to sentences.et.tsv.gz", default="sentences/sentences.et.tsv.gz", required=False)
    parser.add_argument('--lt-asentences', type=str, help="path to sentences.lt.tsv.gz", default="sentences/sentences.lt.tsv.gz", required=False)
    parser.add_argument('--num-tmp-files', type=int, help="Number of temporary files to write (num. alignments // num_tmp_files lines of text need to fit in memory) ", default=256, required=False)
    args = parser.parse_args()
    alignments_file = args.alignments_file
    output_dir = Path(args.output_dir)
    et_sentences = Path(args.et_asentences)
    lt_sentences = Path(args.lt_asentences)

    assert et_sentences.exists(), f"{et_sentences} not found."
    assert lt_sentences.exists(), f"{lt_sentences} not found."

    et_sent_ids = set()
    et_sent_id_to_alignment_id = defaultdict(list)
    lt_sent_ids = set()
    lt_sent_id_to_alignment_id = defaultdict(list)
    expected_alignments = set()

    with gzip.open(alignments_file, "rt") as indata:
        for sent_alignment_id, line in enumerate(indata):
            et_sent_id, lt_sent_id = line.rstrip("\n").split("\t")
            et_sent_ids.add(et_sent_id)
            et_sent_id_to_alignment_id[et_sent_id].append(sent_alignment_id)
            lt_sent_ids.add(lt_sent_id)
            lt_sent_id_to_alignment_id[lt_sent_id].append(sent_alignment_id)
            expected_alignments.add(sent_alignment_id)
    logger.info(f"Read {len(expected_alignments)} alignments from {alignments_file}.")

    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory {output_dir}")
        os.makedirs(output_dir)

    write_out_aligned_sentences(et_sentences, output_dir / "et_sentences", "aligned_sentences.et.txt", et_sent_ids, et_sent_id_to_alignment_id, 53_279_844, expected_alignments, args.num_tmp_files)
    write_out_aligned_sentences(lt_sentences, output_dir / "lt_sentences", "aligned_sentences.lt.txt", lt_sent_ids, lt_sent_id_to_alignment_id, 63_556_320, expected_alignments, args.num_tmp_files)

    print("done")