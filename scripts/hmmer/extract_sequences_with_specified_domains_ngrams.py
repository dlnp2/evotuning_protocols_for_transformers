import argparse
import logging
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s | %(levelname)s - %(name)s - %(message)s")
logger.setLevel(logging.DEBUG)


def gen_target_sequences(
    sequences: Iterator[SeqRecord], target_domain_ids: List[str], remove: bool = False
) -> Iterator[SeqRecord]:
    """Exctract sequences meeting the specified condition."""
    for sequence in sequences:
        clan_ids = []
        pfam_ids = []
        for d in sequence.description.split():
            if "clan_id" in d:
                id_ = d.split("=")[1]
                if id_ != "nan":
                    clan_ids.append(id_)
            elif "pfam_id" in d:
                id_ = d.split("=")[1]
                if id_ != "nan":
                    pfam_ids.append(id_)
        ids = clan_ids + pfam_ids

        # Exact match:
        # domains in `ids` must be included in `target_domain_ids`.
        if remove:
            included = set(ids).issubset(set(target_domain_ids))
        # Partial match:
        # all the domains in `target_domain_ids` must be in the domains.
        else:
            included = np.in1d(target_domain_ids, ids).all()

        if included:
            yield sequence


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(exist_ok=True, parents=True)

    sequences = SeqIO.parse(str(args.seq_fasta), "fasta")
    logger.info(f"#sequences: {len(list(sequences))}.")
    sequences = SeqIO.parse(str(args.seq_fasta), "fasta")

    target_domain_ids = args.target_domain_ids.split(",")
    logger.info(f"Target domain ids: {target_domain_ids}.")
    if args.remove:
        logger.info("Removing sequences with not-specified domains.")
    else:
        logger.info("Finding sequences containing the specified domains.")
    target_sequences = gen_target_sequences(
        sequences, target_domain_ids, remove=args.remove
    )
    n_saved_seqs = SeqIO.write(
        target_sequences, args.output_dir / args.seq_fasta.name, "fasta"
    )
    logger.info(f"Saved {n_saved_seqs:,} sequences.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target_domain_ids", type=str, help="comma separated clan/pfam ids of domains"
    )
    parser.add_argument("seq_fasta", type=Path, help="parsed jackhmmer results fasta")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--remove",
        action="store_true",
        help="if True, remove sequences with domains not in the specified ones. "
        "Otherwise, keep sequences containing the specified domains (default).",
    )
    main(parser.parse_args())
