import argparse
import multiprocessing as mp
from pathlib import Path
from typing import NamedTuple, Tuple

import pandas as pd


def read_hmmscan_result(filepath: Path) -> pd.DataFrame:
    data = []
    with filepath.open(encoding="utf-8") as fin:
        for line in fin:
            if line.startswith("#"):
                continue  # comment lines
            else:
                record = line.split()
                domain_name = record[0]
                accession = record[1]
                pfam_id = accession.split(".")[0]  # not raw data but needed later
                query_name = record[3]
                evalue = float(record[6])
                ivalue = float(record[12])
                domain_from = int(record[19])  # TODO: check if ali coord is suitable
                domain_to = int(record[20])
                description = " ".join(record[22:])
                data.append(
                    [
                        domain_name,
                        accession,
                        pfam_id,
                        query_name,
                        evalue,
                        ivalue,
                        domain_from,
                        domain_to,
                        description,
                    ]
                )
    df = pd.DataFrame(
        data,
        columns=[
            "domain_name",
            "accession",
            "pfam_id",
            "query_name",
            "evalue",
            "ivalue",
            "domain_from",
            "domain_to",
            "description",
        ],
    )
    print(f"Successfully read hmmscan file: found {df.shape[0]:,} records.")
    return df


def check_overlap(interval1: Tuple[int, int], interval2: Tuple[int, int]) -> bool:
    return (interval2[0] <= interval1[1]) and (interval2[1] >= interval1[0])


def check_overlap_pd(s1: NamedTuple, s2: NamedTuple) -> bool:
    return check_overlap((s1.domain_from, s1.domain_to), (s2.domain_from, s2.domain_to))


def filter_domains(query_name_group: pd.DataFrame) -> pd.DataFrame:
    """remove domain overlaps. It is assumed that `query_name_group` is sorted 
    according to E-values in ascending manner."""

    non_overlaps = []
    # TODO: speeding up the below needed...
    # TODO: leave only family and domain
    for row in query_name_group.itertuples():
        if len(non_overlaps) == 0:
            non_overlaps.append(row)
        else:
            for interval1 in non_overlaps:
                is_overlapped = check_overlap_pd(interval1, row)
                if is_overlapped:
                    break
            if not is_overlapped:
                non_overlaps.append(row)

    return pd.DataFrame(non_overlaps).drop(columns=["Index"])


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(exist_ok=True)
    hmmscan = read_hmmscan_result(args.hmmscan_file)
    hmmscan = hmmscan.loc[hmmscan.ivalue <= args.ivalue_thr]

    hmmscan_groups = hmmscan.groupby("query_name", group_keys=False)
    print("Filtering overlapped domains...")
    if args.n_jobs < 0:  # single process
        domains = hmmscan_groups.apply(filter_domains)
    else:
        with mp.Pool(args.n_jobs) as pool:
            domains = pool.map(filter_domains, [group for group in hmmscan_groups])
            domains = pd.concat(domains, ignore_index=True)
    print(f"Filtered out overlapped domains: left with {domains.shape[0]:,} domains.")

    # Add clan ids
    clans = pd.read_csv(
        args.clans_file,
        sep="\t",
        header=None,
        names=["pfam_id", "clan_id", "col2", "col3", "description"],
    )
    clans.drop(columns=["col2", "col3", "description"], inplace=True)
    org_shape = domains.shape
    domains = pd.merge(domains, clans, on="pfam_id", how="left")
    assert domains.shape[0] == org_shape[0]

    domains.to_csv(args.output_dir / f"{args.hmmscan_file.stem}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hmmscan_file",
        type=Path,
        help="path to hmmscan output specified by domtblout option",
    )
    parser.add_argument("clans_file", type=Path, help="path to Pfam-A.clans.tst")
    parser.add_argument(
        "ivalue_thr",
        type=float,
        help="a threshold value for inclusion independent E-value (i-value)",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "-j",
        "--n_jobs",
        type=int,
        default=-1,
        help="number of workers to process. default: -1 (no parallel job)",
    )
    main(parser.parse_args())
