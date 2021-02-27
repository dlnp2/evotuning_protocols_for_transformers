import argparse
import functools
import json
from pathlib import Path
from typing import List, Optional

from Bio import SeqIO
from sklearn.model_selection import train_test_split

SEED = 42


def create_target_name_generator(
    result_file: Path, evalue_thr: float = float(1e-3), exclude_query_seq: bool = False
):
    with open(result_file) as fin:
        for line in fin:
            if line.startswith("#"):  # skip description lines
                continue
            else:
                spec = line.split()
                target_name = spec[0]
                evalue = float(spec[4])
                if evalue > evalue_thr:  # take confident records
                    continue
                if exclude_query_seq:
                    query_name = spec[2].split("|")[1]
                    if query_name in target_name:
                        continue

                yield target_name


def main(
    input_file: Path,
    output_dir: Path,
    seqdb_file: str,
    other_input_files: Optional[List[Path]] = None,
    evalue_thr: float = float(1e-3),
    max_len: int = 1024,
    valid_size: float = 0.1,
    exclude_query_seq: bool = False,
):
    """Parse jackhmmer output file and retrieve sequences with evalue larget than
    a theshold in fasta format.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Reading {input_file.name}...")

    name_gen = functools.partial(
        create_target_name_generator,
        evalue_thr=evalue_thr,
        exclude_query_seq=exclude_query_seq,
    )

    target_names = list(name_gen(input_file))
    # To speed up name search, create a dictionary with its keys composed of 'target_names'
    target_names = {name: 0 for name in target_names}

    # Add more search results and merge if necessary
    if other_input_files is not None:
        for other_file in other_input_files:
            _target_names = list(name_gen(other_file))
            _target_names = {name: 0 for name in _target_names}
            target_names.update(_target_names)  # merge results

    seq_records = [
        rec
        for rec in SeqIO.parse(seqdb_file, "fasta")
        if rec.name in target_names and len(rec.seq) < max_len
    ]
    print(
        f"Total number of sequences with E-value less than the threshold ({evalue_thr}): "
        f"{len(seq_records)}."
    )
    splits = train_test_split(seq_records, test_size=valid_size, random_state=SEED)
    for split, split_name in zip(splits, ["train", "valid"]):
        outpath = (
            output_dir
            / f"{input_file.stem}_EV{evalue_thr}_ML{max_len}_{split_name}.fasta"
        )
        SeqIO.write(split, outpath, "fasta")

    # Save command line options
    options = {
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "seqdb_file": str(seqdb_file),
        "other_input_files": [str(f) for f in other_input_files]
        if other_input_files is not None
        else None,
        "evalue_thr": evalue_thr,
        "max_len": max_len,
        "valid_size": valid_size,
        "exclude_query_sequence": exclude_query_seq,
    }
    with open(output_dir / f"{input_file.stem}_args.json", "w") as fout:
        json.dump(options, fout, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", type=Path, help="Jackhmer output file specified by tblout"
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "seqdb_file", type=str, help="Sequence database fasta file e.g. uniref50.fasta"
    )
    parser.add_argument(
        "--other_input_files",
        type=Path,
        nargs="*",
        help="In addition to `input_file`, these files are also parsed and "
        "the results are merged together. Any duplicated records are removed.",
    )
    parser.add_argument(
        "--evalue_thr",
        type=float,
        default=float(1e-3),
        help="Jackhmmer results with e-value <= this value will be included. Default: 0.001",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1024,
        help="Sequences with lengths < this value will include in the result. Default: 1024",
    )
    parser.add_argument(
        "--valid_size",
        type=float,
        default=0.1,
        help="Fraction of validation split. Default: 0.1",
    )
    parser.add_argument(
        "--exclude_query_sequence",
        action="store_true",
        help="if set, exclude the query sequence itself from the search results. "
        "This should be set for a query fasta file with the canonical description "
        "line format such that starts with sp|P00552|KKA2_KLEPN",
    )
    args = parser.parse_args()
    main(
        args.input_file,
        args.output_dir,
        args.seqdb_file,
        other_input_files=args.other_input_files,
        evalue_thr=args.evalue_thr,
        max_len=args.max_len,
        valid_size=args.valid_size,
        exclude_query_seq=args.exclude_query_sequence,
    )
