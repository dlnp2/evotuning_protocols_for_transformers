import argparse
import pickle
from pathlib import Path
from typing import Iterator

import lmdb
import numpy as np
import tqdm
from Bio import SeqIO
from Bio.Alphabet import SingleLetterAlphabet
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def gen_seqs_from_lmdb(lmdb_path: str) -> Iterator[SeqRecord]:
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin(write=False) as txn:
        n_seqs = pickle.loads(txn.get(b"num_examples"))
        for index in tqdm.tqdm(range(n_seqs)):
            item = pickle.loads(txn.get(str(index).encode()))
            id_ = f"{index} clan={item['clan']} family={item['family']}"
            seq_record = SeqRecord(
                seq=Seq(item["primary"], alphabet=SingleLetterAlphabet),
                id=id_,
                description="",
            )
            yield seq_record


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(exist_ok=True, parents=True)

    gen = gen_seqs_from_lmdb(str(args.lmdb))
    SeqIO.write(gen, str(args.output_dir / f"{args.lmdb.stem}.fasta"), "fasta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lmdb", type=Path)
    parser.add_argument("output_dir", type=Path)
    main(parser.parse_args())
