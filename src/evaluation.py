import warnings
from collections import defaultdict
from functools import reduce
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import altair as alt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats

from .tape_related.variant_effect_prediction_one_protein import METRICS


def tidy_name(name: str) -> str:
    if "bert" in name:
        name = "bert"
    elif "masked" in name:
        if "gram" in name:
            ngram = "_" + name.split("_")[-1]
        else:
            ngram = ""
        if "domain" in name:
            name = f"domain{ngram}"
        elif "assay" in name:
            name = f"assay{ngram}"
        else:
            name = f"full{ngram}"
    else:
        raise ValueError(f"not considered case: {name}")
    return name


def get_testsize(fpath: Path) -> float:
    return float(
        str(list(fpath.relative_to(DATADIR).parents)[-2]).replace("testsize", "")
    )


def get_avg_hidden_layers(fpath: Path) -> str:
    return list(fpath.relative_to(DATADIR).parents)[-3].name.split("_")[-1]


def get_model_name(fpath: Path, prefix: Optional[str] = None) -> str:
    # ./testsizeXX/avg_hidden_layers_XX/jh/u/
    _prefix = list(fpath.relative_to(DATADIR).parents)[-5]
    model_name = str(list(fpath.relative_to(DATADIR / _prefix).parents)[2])
    model_name = tidy_name(model_name.split("/")[-1])

    if "extracted_sequences_with_mutagenized_domains" in str(fpath):
        if "extracted_sequences_with_mutagenized_domains_20200809" in str(
            fpath
        ) or "extracted_sequences_with_mutagenized_domains_20200824" in str(fpath):
            if "gram" in str(fpath):
                prefix2 = "mt_strict_"
            else:
                prefix2 = "mt_"
        else:
            prefix2 = "mt_"
    elif "extracted_sequences_with_specified_domains" in str(fpath):
        prefix2 = "wt_"
    elif "extracted_sequences_with_specified_mutagenized_domains" in str(fpath):
        if "extracted_sequences_with_specified_mutagenized_domains_20200809" in str(
            fpath
        ) or "extracted_sequences_with_specified_mutagenized_domains_20200824" in str(
            fpath
        ):
            if "gram" in str(fpath):
                prefix2 = "mt_wt_strict_"
            else:
                prefix2 = "mt_wt_"
        else:
            prefix2 = "mt_wt_"
    else:
        prefix2 = ""

    if prefix is not None:
        model_name = f"{prefix}{prefix2}{model_name}"
    else:
        model_name = f"{prefix2}{model_name}"

    return model_name


def offdiagonal_mask(shape):
    mask = np.ones(shape)
    for i in range(len(mask)):
        mask[i, i] = 0
    mask = mask.astype(bool)
    return mask


def collect_file_paths(
    input_dir: Path,
    fname: str,
    uniprot_id: str,
    regressor_type: str,
    domain: str,
    dom_embed_npz: str,
    excluded_dirs: Optional[Union[str, List[str]]] = None,
    assay_embed_npz: Optional[Union[str, List[str]]] = None,
    test_size: Optional[float] = None,
    avg_hidden_layers: Optional[str] = None,
) -> List[Path]:
    print(f"Collecting files from {input_dir}...")

    files = [
        f
        for f in input_dir.rglob(fname)
        if uniprot_id in str(f) and regressor_type in str(f)
    ]
    if test_size is not None:
        files = [f for f in files if get_testsize(f) == test_size]
    if avg_hidden_layers is not None:
        files = [f for f in files if get_avg_hidden_layers(f) == avg_hidden_layers]

    if excluded_dirs is not None:
        if isinstance(excluded_dirs, str):
            excluded_dirs = [excluded_dirs]
        files = [
            f for f in files if all(exc_dir not in str(f) for exc_dir in excluded_dirs)
        ]

    ret = []
    for csv in files:
        is_old_model = (
            str(csv).find(f"{uniprot_id}_domain") != -1 and str(csv).find(domain) == -1
        )
        npzs = (
            [dom_embed_npz, assay_embed_npz]
            if assay_embed_npz is not None
            else [dom_embed_npz]
        )
        is_old_embedding = all(
            [
                str(csv).find(f"variant_seq_{uniprot_id}_") != -1
                and str(csv).find(npz) == -1
                for npz in npzs
            ]
        )
        if not (is_old_model or is_old_embedding):
            ret.append(csv)

    print(f"Found {len(ret)} files.")
    return ret


def read_evaluation_results(
    uniprot_id: str,
    domain: str,
    dom_embed_npz: str,
    assay_embed_npz: Optional[str] = None,
    excluded_dirs: Optional[Union[str, List[str]]] = None,
    regressor_type="lasso_lars",
) -> pd.DataFrame:
    def _get_embedding_type(s: str) -> str:
        if assay_embed_npz is not None and assay_embed_npz in s:
            return "assay"
        else:
            return "domain" if dom_embed_npz in s else "full"

    def read_from_directory(d: Path) -> pd.DataFrame:
        protein_csvs = collect_file_paths(
            d,
            "evaluation.csv",
            uniprot_id,
            regressor_type,
            domain,
            dom_embed_npz,
            excluded_dirs=excluded_dirs,
            assay_embed_npz=assay_embed_npz,
        )

        protein_df = []
        for csv in protein_csvs:
            df = pd.read_csv(csv)
            _csv = csv.relative_to(DATADIR)
            _csv_path = _csv.relative_to(list(_csv.parents)[-3])
            df.loc[:, "model_name"] = get_model_name(csv)
            df.loc[:, "test_size"] = get_testsize(csv)
            df.loc[:, "hidden_layers"] = get_avg_hidden_layers(csv)
            df.loc[:, "fpath"] = str(csv)
            protein_df.append(df)

        if len(protein_df) == 0:
            protein_df = pd.DataFrame()
        else:
            protein_df = pd.concat(protein_df, ignore_index=True)
            nans = protein_df.loc[protein_df["name"].isnull()]
            if nans.shape[0] != 0:
                print(f"Dropping nan records... {d}:", nans)
                protein_df = protein_df.loc[~no_names]
            protein_df.loc[:, "embedding_type"] = protein_df["name"].apply(
                lambda s: _get_embedding_type(s)
            )

        return protein_df

    protein_df = read_from_directory(DATADIR)
    protein_df = protein_df.loc[protein_df["test_size"].isin(ALL_TEST_SIZES)]

    org_len = len(protein_df)
    protein_df.sort_values("fpath", ascending=False).drop_duplicates(
        ["model_name", "test_size", "hidden_layers", "embedding_type"], inplace=True
    )
    new_len = len(protein_df)
    if org_len != new_len:
        print(f"Dropped {org_len - new_len} records because of model duplication.")

    protein_df.sort_values(
        ["test_size", "hidden_layers", "model_name"],
        ascending=[True, False, True],
        inplace=True,
    )

    org_len = len(protein_df)
    not_ngram = protein_df.loc[~protein_df["model_name"].str.contains("gram")]
    ngram = protein_df.loc[protein_df["model_name"].str.contains("gram")]
    wt_ngram = ngram.loc[~ngram["model_name"].str.contains("mt")]
    mt_ngram = ngram.loc[ngram["model_name"].str.contains("mt")].loc[
        ngram["model_name"].str.contains("strict")
    ]
    protein_df = pd.concat([not_ngram, wt_ngram, mt_ngram], ignore_index=True)
    new_len = len(protein_df)
    if org_len != new_len:
        print(f"Droppped {org_len - new_len} records because of old cleansing.")

    return protein_df


def gen_test_results(
    protein_df: pd.DataFrame,
    dom_embed_npz: str,
    test_size: float,
    hidden_layers: str,
    assay_embed_npz: Optional[str] = None,
    debug: bool = False,
    test_fn: Callable = stats.wilcoxon,
) -> pd.DataFrame:
    def _get_prefix(s: str) -> str:
        if assay_embed_npz is not None and assay_embed_npz in s:
            return "assay_emb | "
        else:
            return "dom_emb | " if dom_embed_npz in s else "full_emb | "

    def read_metrics_raw(protein_df: pd.DataFrame):
        _df = protein_df.loc[
            (protein_df["test_size"] == test_size)
            & (protein_df["hidden_layers"] == hidden_layers)
        ]
        protein_raws = [
            Path(fpath).parent / "metrics_raw.csv" for fpath in _df["fpath"]
        ]

        metrics_raw = {}
        for f in protein_raws:
            prefix = _get_prefix(str(f))
            model_name = get_model_name(f, prefix=prefix)  # must be unique
            data = pd.read_csv(f)

            if debug:
                print(model_name, str(f))

            for metric_name, values in data.to_dict(orient="list").items():
                if metric_name == "seed":
                    continue
                if not metric_name in metrics_raw:
                    metrics_raw[metric_name] = {}

                msg = (
                    f"model_name: {model_name}, "
                    f"model_names: {list(metrics_raw[metric_name].keys())}"
                )
                assert model_name not in metrics_raw[metric_name], msg
                metrics_raw[metric_name][model_name] = np.array(values)

        return metrics_raw

    metrics_raw = read_metrics_raw(protein_df)

    test_results_raw = {}
    test_results = {}
    for metric_name, data in metrics_raw.items():
        test_results_raw[metric_name] = []
        for model1, model2 in product(data.keys(), data.keys()):
            if debug:
                print(model1, model2)

            if model1 == model2:
                pvalue = 1
            else:
                val1 = data[model1]
                val2 = data[model2]
                if (val1 == val2).all():
                    # In this case Wilcoxon test doesn't work
                    pvalue = 1
                else:
                    _, pvalue = test_fn(val1, val2)
            test_results_raw[metric_name].append([model1, model2, pvalue])
        test_results[metric_name] = pd.DataFrame(
            test_results_raw[metric_name], columns=["model1", "model2", "pvalue"]
        ).pivot(index="model1", columns="model2", values="pvalue")

    return test_results


def get_better_models(
    metric_name: str,
    data: pd.DataFrame,
    significances: pd.DataFrame,
    base_model_name: str = "full_emb | full",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_metric_val = data.loc[
        data["model_name2"] == base_model_name, metric_name
    ].values[0]
    if metric_name == "spearman":
        better_models = data.loc[data[metric_name] > base_metric_val]
        better_models.loc[:, "delta"] = (
            better_models.loc[:, metric_name] / base_metric_val - 1
        ) * 100
        best_model_names = better_models.loc[
            better_models[metric_name] == better_models[metric_name].max(),
            "model_name2",
        ].values
    elif metric_name == "rmse":
        better_models = data.loc[data[metric_name] < base_metric_val]
        better_models.loc[:, "delta"] = (
            1 - better_models.loc[:, metric_name] / base_metric_val
        ) * 100
        best_model_names = better_models.loc[
            better_models[metric_name] == better_models[metric_name].min(),
            "model_name2",
        ].values
    else:
        raise NotImplementedError

    if better_models.shape[0] == 0:
        return better_models, better_models

    base_sig = significances[base_model_name]
    better_models.loc[:, "base_significance"] = better_models["model_name2"].apply(
        lambda n: base_sig[n]
    )
    # significantly better compared with the base model
    better_models = better_models.loc[better_models["base_significance"] < PVALUE_THR]

    for i, best_model_name in enumerate(best_model_names):
        best_sig = significances[best_model_name]
        better_models.loc[:, f"best_significance_{i}"] = better_models[
            "model_name2"
        ].apply(lambda n: best_sig[n])

    # NOT significant compared with the best model
    condition = reduce(
        lambda a, b: a | b,
        [
            better_models[f"best_significance_{i}"] >= PVALUE_THR
            for i in range(len(best_model_names))
        ],
    )
    best_models = (
        better_models.loc[condition]
        .loc[:, ["model_name2", "delta", metric_name]]
        .sort_values("delta", ascending=False)
    )
    return better_models.sort_values("delta", ascending=False), best_models


def set_inplace_embedding_type_short(
    data: pd.DataFrame, best_models: pd.DataFrame, better_models: pd.DataFrame
) -> None:
    data.loc[:, "embedding_type_short"] = data[["model_name2", "embedding_type"]].apply(
        lambda s: list(s[1])[0] + "**"
        if s[0] in best_models["model_name2"].values
        else (
            (
                list(s[1])[0] + "*"
                if s[0] in better_models["model_name2"].values
                else list(s[1])[0]
            )
        ),
        axis=1,
    )


def generate_chart(
    data: pd.DataFrame,
    metric_name: str,
    test_size: float,
    hidden_layers: str,
    base_model_name: str = "full_emb | full",
    ymin: float = 0.0,
    ymax: float = 1.0,
) -> alt.vegalite.v4.api.Chart:
    model_name2 = data["model_name2"].values.tolist()
    colors = [
        "orange"
        if name == base_model_name
        else ("lightsteelblue" if name == "full_emb | bert" else "steelblue")
        for name in model_name2
    ]

    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "model_name2",
                sort="-y" if metric_name == "spearman" else "y",
                axis=alt.Axis(
                    labelExpr="replace(replace(split(datum.value, ' | ')[1], 'cleansed', 'c'), 'gram', 'g')"
                ),
            ),
            y=alt.Y(metric_name, scale=alt.Scale(domain=[ymin, ymax])),
            # https://github.com/altair-viz/altair/issues/2101#issuecomment-616243831
            color=alt.Color(
                "model_name2",
                scale=alt.Scale(domain=model_name2, range=colors),
                legend=None,
            ),
        )
    )
    text = bars.mark_text(
        align="center",
        baseline="bottom",
    ).encode(text="embedding_type_short")

    return (bars + text).properties(
        title=f"test_size={test_size}, hidden_layers={hidden_layers}, "
        f"benchmark_model={base_model_name}"
    )


def shorten_name(s: str) -> str:
    if "full" in s:
        name1 = "full"
    elif "domain" in s:
        name1 = "domain"
    elif "assay" in s:
        name1 = "assay"
    else:
        name1 = "bert"

    if "-g" in s.split("_")[-1]:
        name2 = f"_" + s.split("_")[-1]
    else:
        name2 = ""

    return name1 + name2


def run_evaluation(
    protein: str,
    domain: str,
    dom_embed_npz: str,
    assay_embed_npz: Optional[str] = None,
    excluded_dirs: Optional[List[str]] = None,
    regressor_type: str = "lasso_lars",
    chart_width: int = 1920,
    debug: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict[float, Dict[str, pd.DataFrame]]]]:
    protein_df = read_evaluation_results(
        protein,
        domain,
        dom_embed_npz,
        assay_embed_npz=assay_embed_npz,
        excluded_dirs=excluded_dirs,
        regressor_type=regressor_type,
    )
    print(f"protein_df.shape: {protein_df.shape}")
    display(protein_df.head())

    print("Computing significances...")
    n_tests = len(ALL_TEST_SIZES)

    significances = {}
    for test_size in ALL_TEST_SIZES:
        for hidden_layers in ALL_AVG_HIDDEN_LAYERS:
            print(f"test_size: {test_size}, hidden_layers: {hidden_layers}")
            test_results = gen_test_results(
                protein_df,
                dom_embed_npz,
                test_size,
                hidden_layers,
                assay_embed_npz=assay_embed_npz,
                debug=debug,
                test_fn=TEST_FN,
            )
            for metric_name in METRICS:
                df = test_results.get(metric_name)
                if df is None:
                    print(
                        f"No evaluation result found for {protein}, {test_size}, and {hidden_layers}."
                    )
                    continue
                if significances.get(metric_name) is None:
                    significances[metric_name] = {}
                if significances[metric_name].get(test_size) is None:
                    significances[metric_name][test_size] = {}
                significances[metric_name][test_size][hidden_layers] = df
    print("Done")

    all_better_models = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_best_models = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for metric_name in METRICS.keys():
        charts = defaultdict(list)
        for i, test_size in enumerate(ALL_TEST_SIZES):
            for j, hidden_layers in enumerate(ALL_AVG_HIDDEN_LAYERS):
                data = protein_df.loc[
                    (protein_df["test_size"] == test_size)
                    & (protein_df["hidden_layers"] == hidden_layers),
                    [
                        metric_name,
                        "model_name",
                        "embedding_type",
                    ],
                ]
                if data.empty:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data.loc[:, "model_name2"] = data[
                        ["model_name", "embedding_type"]
                    ].apply(
                        lambda s: f"{s[1].replace('domain', 'dom')}_emb | {s[0]}",
                        axis=1,
                    )
                for base_model_name in ["full_emb | bert", "full_emb | full"]:
                    try:
                        _significance = significances[metric_name][test_size][
                            hidden_layers
                        ]
                    except KeyError as e:
                        print(f"No evaluation result found, {e}")
                        continue

                    better_models, best_models = get_better_models(
                        metric_name,
                        data,
                        _significance,
                        base_model_name=base_model_name,
                    )
                    print(f"\nbase_model_name={base_model_name}")
                    if not best_models.empty:
                        display(best_models.style.hide_index())
                    else:
                        display(best_models)

                    set_inplace_embedding_type_short(data, best_models, better_models)
                    if not better_models.empty:
                        set_inplace_embedding_type_short(
                            better_models, best_models, better_models
                        )

                    if base_model_name == "full_emb | full":
                        chart = generate_chart(
                            data.copy(),
                            metric_name,
                            test_size,
                            hidden_layers,
                            base_model_name,
                        ).properties(width=chart_width)
                        charts[base_model_name].append(chart)

                        all_better_models[metric_name][test_size][
                            hidden_layers
                        ] = better_models
                        all_best_models[metric_name][test_size][
                            hidden_layers
                        ] = best_models

        for base_model_name, _charts in charts.items():
            for _chart in _charts:
                display(_chart)

    print(
        f"Displaying common better models (based on metric: {METRIC_TO_REPORT} and "
        f"test sizes: {CONSENSUS_TESTSIZES})"
    )
    print("=====================================================================")
    names = {}
    for ts, hl_df in all_better_models[METRIC_TO_REPORT].items():
        if ts not in CONSENSUS_TESTSIZES:
            continue
        for hidden_layers, better_models in hl_df.items():
            if hidden_layers not in names:
                names[hidden_layers] = []
            if not better_models.empty:
                names[hidden_layers].append(
                    set(better_models["model_name2"].values.tolist())
                )
            else:
                names[hidden_layers].append(set())
    common_model_names = {}
    for hidden_layers, _names in names.items():
        common_names = list(reduce(lambda x, y: x & y, _names)) if _names else []
        common_model_names[hidden_layers] = common_names

    for ts, hl_df in all_better_models[METRIC_TO_REPORT].items():
        for hidden_layers, better_models in hl_df.items():
            print(f"test_size: {ts}, hidden_layers: {hidden_layers}")
            common_better_models = better_models.loc[
                better_models["model_name2"].isin(common_model_names[hidden_layers])
            ]
            common_better_models.loc[:, "model_name2"] = common_better_models[
                "model_name2"
            ].apply(
                lambda s: s.split(" | ")[1]
                .replace("gram", "g")
                .replace("cleansed", "c")
            )
            common_better_models.rename(
                columns={"embedding_type_short": "emb"}, inplace=True
            )
            common_better_models.loc[:, "protocol"] = common_better_models[
                "model_name2"
            ].apply(lambda s: shorten_name(s))
            if not common_better_models.empty:
                display(
                    common_better_models[
                        ["model_name2", "protocol", "emb", METRIC_TO_REPORT]
                    ].style.hide_index()
                )
            else:
                display(
                    common_better_models[
                        ["model_name2", "protocol", "emb", METRIC_TO_REPORT]
                    ]
                )

    return protein_df, significances
