import argparse
from pathlib import Path
from typing import Iterator

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def extract_from_fasta(
    fasta_path: Path, domain_from: int, domain_to: int
) -> Iterator[SeqRecord]:
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        org_seq = record.seq
        seq = Seq(str(org_seq)[domain_from - 1 : domain_to], org_seq.alphabet)
        record.seq = seq
        yield record


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(exist_ok=True, parents=True)

    extracted_fasta = extract_from_fasta(args.fasta, args.domain_from, args.domain_to)
    SeqIO.write(
        extracted_fasta,
        str(
            args.output_dir
            / f"{args.fasta.stem}_{args.domain_from}_{args.domain_to}.fasta"
        ),
        "fasta",
    )

    extracted_df = pd.read_csv(args.csv).rename(columns={"sequence": "org_seq"})
    extracted_df.loc[:, "sequence"] = extracted_df["org_seq"].apply(
        lambda s: s[args.domain_from - 1 : args.domain_to]
    )
    del extracted_df["org_seq"]
    extracted_df.to_csv(
        args.output_dir / f"{args.csv.stem}_{args.domain_from}_{args.domain_to}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta", type=Path, help="path to variant_seq_XXX.fasta")
    parser.add_argument("csv", type=Path, help="path to variant_seq_XXX.csv")
    parser.add_argument("domain_from", type=int, help="1-indexed domain start position")
    parser.add_argument(
        "domain_to", type=int, help="1-indexed domain end position (inclusive)"
    )
    parser.add_argument("output_dir", type=Path)
    main(parser.parse_args())
