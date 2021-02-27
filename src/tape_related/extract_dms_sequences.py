# Retrieve DMS sequences in fasta format.

import argparse
from os import pardir
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def mut_seq(seq: str, variant: str) -> str:
    position = int(variant[1:-1])
    wt_aa = variant[0]
    mut_aa = variant[-1]

    if wt_aa == mut_aa:
        return seq

    seq_list = list(seq)
    seq_list[position - 1] = mut_aa
    return "".join(seq_list)


def retrieve_fasta(uniprot_id: str, output_dir: Path) -> None:
    outpath = output_dir / f"{uniprot_id}.fasta"
    if outpath.exists():
        print(f"{outpath.name} already exists.")
    else:
        print(f"Retrieving {outpath.name}.")
        ret = requests.get(f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta")
        with open(outpath, "w") as fout:
            fout.write(ret.text)


def retrieve_fasta_files(uniprot_ids: List[str], output_dir: Path) -> None:
    for uid in uniprot_ids:
        retrieve_fasta(uid, output_dir)


def add_original(missense: pd.DataFrame) -> pd.DataFrame:
    # restore the original sequences and score(=1)
    returned_cols = [
        "uniprot_id",
        "protein",
        "Variant",
        "mut_type",
        "scaled_effect1",
        "sequence",
    ]

    def _get_original_sequence(variant: str, seq: str) -> Tuple[str, str]:
        position = int(variant[1:-1])
        wt_aa = variant[0]
        wt = wt_aa + str(position) + wt_aa
        seq_list = list(seq)
        seq_list[position - 1] = wt_aa
        return "".join(seq_list), wt

    originals = []
    for protein in missense.drop_duplicates("uniprot_id")[returned_cols].itertuples():
        original_seq, wt = _get_original_sequence(protein.Variant, protein.sequence)
        originals.append([protein.uniprot_id, protein.protein, wt, "synonymous", 1, original_seq])
    originals = pd.DataFrame(originals, columns=returned_cols)

    return pd.concat([missense[returned_cols], originals], ignore_index=True)


def save_in_fasta_format(missense: pd.DataFrame, output_dir: Path, output_name: str) -> None:
    record_gen = (
        SeqRecord(
            Seq(protein.sequence, IUPAC.protein),
            id=f"{protein.protein}_{protein.Variant}_{protein.Index}",
        )
        for protein in missense.itertuples()
    )
    with open(output_dir / output_name, "w") as fout:
        SeqIO.write(record_gen, fout, "fasta")


def main(dms_training_file_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    fasta_dir = output_dir / "fasta"
    fasta_dir.mkdir(exist_ok=True)

    dms = pd.read_csv(dms_training_file_path)
    missense = dms.query("mut_type == 'missense'")

    retrieve_fasta_files(missense.drop_duplicates("uniprot_id").uniprot_id.tolist(), fasta_dir)

    org_seq = dict()
    for protein in missense.drop_duplicates("uniprot_id")[
        ["uniprot_id", "protein_size"]
    ].itertuples():
        uid = protein.uniprot_id
        print("Reading {}...".format(uid))
        seq_len = protein.protein_size
        with open(fasta_dir / f"{uid}.fasta") as infile:
            infile.readline()
            lines = infile.readlines()
            seq = "".join([s.strip() for s in lines])
        assert seq_len == len(seq), "length mismatch for {} (DataFrame: {}, FASTA: {})".format(
            uid, seq_len, len(seq)
        )
        org_seq[uid] = seq

    missense.loc[:, "sequence"] = missense[["uniprot_id", "Variant"]].apply(
        lambda s: mut_seq(org_seq[s[0]], s[1]), axis=1
    )
    missense = add_original(missense)
    missense.reset_index(drop=True)

    output_dir = output_dir / "variant_seq"
    output_dir.mkdir(exist_ok=True)

    missense.to_csv(output_dir / "variant_seq.csv", index=False)
    save_in_fasta_format(missense, output_dir, "variant_seq.fasta")

    for uid in missense["uniprot_id"].unique():
        _missense = missense.loc[missense["uniprot_id"]==uid].reset_index(drop=True)
        _missense.to_csv(output_dir / f"variant_seq_{uid}.csv", index=False)
        save_in_fasta_format(_missense, output_dir, f"variant_seq_{uid}.fasta")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dms_training_file_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    main(args.dms_training_file_path, args.output_dir)
