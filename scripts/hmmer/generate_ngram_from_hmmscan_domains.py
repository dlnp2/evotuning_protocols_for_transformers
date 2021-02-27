import argparse
from pathlib import Path
from typing import Dict, Iterator, List
from tqdm.auto import tqdm

import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import SingleLetterAlphabet
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def parse_ngrams(ngrams_str: str) -> List[int]:
    ngrams = list(map(int, ngrams_str.split(",")))
    assert all([n > 0 for n in ngrams]), "ngrams must be positive integers."
    return ngrams


def read_sequences(fasta: Iterator[SeqRecord]) -> Dict[str, Dict[str, str]]:
    sequences = {}
    for seq_record in fasta:
        desc = seq_record.description
        query_name = desc.split()[0]
        if not query_name in sequences:
            sequences[query_name] = {
                "sequence": str(seq_record.seq),
                "description": desc,
            }
    return sequences


def _generate_ngram_likes(domains: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return sequences composed of domains and amino acids in between (n-gram like)."""
    data = []
    n_domains = domains.shape[0]
    if n_domains >= n:
        # We assume non-overlapping domains.
        domains = domains.sort_values("domain_from")
        full_sequence = domains["sequence"].iloc[0]
        org_desc = domains["description"].iloc[0]
        endpoints = domains[["domain_from", "domain_to"]].values
        for i in range(0, n_domains - n + 1):
            desc = f"{org_desc}_{i} "
            from_ = None  # 1-indexed
            to = None
            for domain_index, j in enumerate(range(i, i + n)):  # [i, i + n - 1]
                _from, _to = endpoints[j]
                pfam_id = domains.iloc[j]["pfam_id"]
                clan_id = domains.iloc[j]["clan_id"]
                desc += f"pfam_id_{domain_index + 1}={pfam_id} "
                desc += f"clan_id_{domain_index + 1}={clan_id} "
                desc += f"domain_from_{domain_index + 1}={_from} "
                desc += f"domain_to_{domain_index + 1}={_to} "
                if from_ is None:
                    from_ = _from
                to = _to
            seq = full_sequence[from_ - 1 : to]
            data.append([n, seq, desc])

    return pd.DataFrame(data, columns=["n", "sequence", "description"])


def generate_ngram_likes(
    group: pd.DataFrame, ngrams: List[int] = [1, 2, 3]
) -> pd.DataFrame:
    ngram_dfs = [_generate_ngram_likes(group, n) for n in ngrams]
    return pd.concat(ngram_dfs, ignore_index=True)  # n, sequence, description


def gen_seq_records(ngram_likes: pd.DataFrame) -> Iterator[SeqRecord]:
    for data in ngram_likes.itertuples():
        seq_record = SeqRecord(
            seq=Seq(data.sequence, alphabet=SingleLetterAlphabet),
            id=data.description,
            description="",
        )
        yield seq_record


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(exist_ok=True, parents=True)

    ngrams = parse_ngrams(args.ngrams)

    domains = pd.read_csv(
        args.domains_csv_path,
        usecols=["pfam_id", "clan_id", "query_name", "domain_from", "domain_to"],
    )
    print(f"Reading sequences from {args.parsed_jackhmmer_results_path}")
    sequences = read_sequences(
        SeqIO.parse(str(args.parsed_jackhmmer_results_path), "fasta")
    )
    sequences = (
        pd.DataFrame.from_dict(sequences, orient="index")
        .reset_index()
        .rename(columns={"index": "query_name"})
    )
    org_len = domains.shape[0]
    domains = pd.merge(domains, sequences, on="query_name", how="left")
    assert domains.shape[0] == org_len
    assert (~domains.sequence.isnull()).all(), "sequence missing"

    print("Generating n-grams")
    tqdm.pandas()  # adds `progress_apply` method to DataFrame/Series for monitoring progress
    results = domains.groupby("query_name", as_index=False).progress_apply(
        generate_ngram_likes, ngrams=ngrams
    )
    for n in ngrams:
        seq_records = gen_seq_records(results.loc[results.n == n])
        fname = str(
            args.output_dir
            / f"{args.parsed_jackhmmer_results_path.stem}_{n}-gram.fasta"
        )
        SeqIO.write(seq_records, fname, "fasta")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "domains_csv_path",
        type=Path,
        help="parsed hmmscan results with non-overlapping domains in each line",
    )
    parser.add_argument(
        "parsed_jackhmmer_results_path",
        type=Path,
        help="parsed jackhmmer results in fasta format",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--ngrams",
        type=str,
        default="1,2,3",
        help="comma concatenated n of n-grams to be generated",
    )
    main(parser.parse_args())
