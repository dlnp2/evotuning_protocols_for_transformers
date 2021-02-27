import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s | %(levelname)s - %(name)s - %(message)s")
logger.setLevel(logging.DEBUG)


def _get_target_query_names(
    target_domain_ids: List[str], grouped: pd.DataFrame, remove: bool = False
) -> pd.DataFrame:
    """Extract query names meeting the specified condition."""

    no_clan_ids = grouped["clan_id"].isnull()
    clan_ids = grouped.loc[~no_clan_ids, "clan_id"].values.tolist()
    pfam_ids = grouped.loc[no_clan_ids, "pfam_id"].values.tolist()
    ids = clan_ids + pfam_ids

    # Exact match:
    # domains in `ids` must be included in `target_domain_ids`.
    # Not all the domains in `target_domain_ids` must be in `ids`. 
    if remove:
        included = set(ids).issubset(set(target_domain_ids))
    # Partial match:
    # all the domains in `target_domain_ids` must be in the domains,
    # but other domains can be contained.
    else:
        included = np.in1d(target_domain_ids, ids).all()
    ret = grouped if included else pd.DataFrame(columns=["query_name"])
    return ret


def get_target_query_names(
    domain_for_hits: pd.DataFrame, target_domain_ids: List[str], remove: bool = False
) -> List[str]:
    searcher = partial(_get_target_query_names, target_domain_ids, remove=remove)
    searched = domain_for_hits.groupby("query_name", as_index=False).apply(searcher)
    return searched["query_name"].unique().tolist()


def gen_target_sequences(
    sequences: Iterator[SeqRecord], target_query_names: List[str]
) -> Iterator[SeqRecord]:
    for sequence in sequences:
        if sequence.id in target_query_names:
            yield sequence


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Reading {args.domain_csv.name}.")
    domains_for_hits = pd.read_csv(args.domain_csv)
    sequences = SeqIO.parse(str(args.seq_fasta), "fasta")
    logger.info(f"#sequences: {len(list(sequences))}.")
    target_domain_ids = args.target_domain_ids.split(",")
    logger.info(f"Target domain ids: {target_domain_ids}.")
    if args.remove:
        logger.info(f"Removing sequences with not-specified domains.")
    target_query_names = get_target_query_names(
        domains_for_hits, target_domain_ids, args.remove
    )
    sequences = SeqIO.parse(str(args.seq_fasta), "fasta")
    target_sequences = gen_target_sequences(sequences, target_query_names)
    n_saved_seqs = SeqIO.write(
        target_sequences, args.output_dir / args.seq_fasta.name, "fasta"
    )
    logger.info(f"Saved {n_saved_seqs:,} sequences.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target_domain_ids", type=str, help="comma separated clan/pfam ids of domains"
    )
    parser.add_argument("domain_csv", type=Path, help="hmmscan annotated domains csv")
    parser.add_argument("seq_fasta", type=Path, help="parsed jackhmmer results fasta")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--remove",
        action="store_true",
        help="if True, remove sequences with non-specified domains"
    )
    main(parser.parse_args())
