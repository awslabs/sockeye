import argparse
import gzip
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take the cosine_similarity files and produce a baseline alignmentin the expected format.')
    parser.add_argument('--cosine-similarity-folder', type=str, help='The folder that contains the TSV files.', default="cosine_similarity")
    parser.add_argument('--min-cosine-similarity', type=float, default=0.9, help='The minimum cosine similarity to consider.')
    parser.add_argument('-o', '--output-file', type=str, default="top1_cosine.et-lt.tsv.gz", help='The file that the alignments will be written to.', required=False)

    args = parser.parse_args()
    folder = Path(args.cosine_similarity_folder)
    min_cosine_similarity = args.min_cosine_similarity
    
    num_skipped = 0
    num_total = 0
    with gzip.open(args.output_file, 'wt') as fout:

        for lang_pair in ["et-lt"]: # , "lt-et"
            for suffix in list(map(str, range(0, 9))) + ['a', 'b', 'c', 'd', 'e', 'f']:
                fname = folder / f'cosine_similarity.{lang_pair}.part_{suffix}.tsv.gz'
                print(f"reading {fname}")
                with gzip.open(fname, 'rt') as finput:
                    header = next(finput)
                    header = header.rstrip("\n").split("\t")
                    for row_idx, line in enumerate(finput):
                        num_total += 1
                        cols = line.rstrip("\n").split("\t")
                        assert len(cols) == len(header), f"Row {row_idx} has {len(cols)} columns, but header has {len(header)} columns."
                        row = dict(zip(header, cols))
                        # QueryId ResultId1       Score1 
                        query_sentence_id = row["QueryId"]
                        result_sentence_id = row["ResultId1"]
                        score = float(row["Score1"])
                        if score < min_cosine_similarity:
                            num_skipped += 1
                            continue
                        if lang_pair == "et-lt":
                            fout.write(f"{query_sentence_id}\t{result_sentence_id}\n")
                        elif lang_pair == "et-lt":
                            fout.write(f"{result_sentence_id}\t{query_sentence_id}\n")
                        else:
                            raise ValueError(f"Unknown language pair: {lang_pair}")
    print(f"Skipped {num_skipped} out of {num_total} lines due to low cosine similarity.")

