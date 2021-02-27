import argparse
import pickle
from pathlib import Path

import lmdb
from Bio import SeqIO


def main(fasta_path: Path, lmdb_outdir: Path):
    lmdb_outdir.mkdir(exist_ok=True, parents=True)

    sequences = SeqIO.parse(fasta_path, "fasta")
    outpath = lmdb_outdir / f"{fasta_path.stem}.lmdb"
    env = lmdb.open(str(outpath), map_size=50e9)
    with env.begin(write=True) as txn:
        num_examples = 0
        for record in sequences:
            item = {}
            item["primary"] = str(record.seq)
            item["protein_length"] = len(record.seq)
            item["clan"] = 0  # dummy
            item["family"] = 0  # dummy
            txn.put(str(num_examples).encode(), pickle.dumps(item))
            num_examples += 1
        txn.put(b"num_examples", pickle.dumps(num_examples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_path", type=Path)
    parser.add_argument("lmdb_outdir", type=Path)
    args = parser.parse_args()
    main(args.fasta_path, args.lmdb_outdir)
