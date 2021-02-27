import argparse
import json
from collections import OrderedDict, defaultdict
from functools import partial
from pathlib import Path, PosixPath
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tqdm
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from typing_extensions import Protocol

from .utils import spearmanr

SEED = 42
N_LAYERS = 12  # number of Transformer layers
METRICS = OrderedDict(
    {
        "mae": mean_absolute_error,
        "rmse": partial(mean_squared_error, squared=False),
        "spearman": spearmanr,
    }
)


class Predictable(Protocol):
    def predict(self, data: np.ndarray) -> np.ndarray:
        ...


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("protein_seq_csv", type=Path)
    parser.add_argument("features_npz", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--regressor_type",
        type=str,
        default="lasso_lars",
        choices=["lasso", "lasso_lars"],
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="avg",
        choices=["avg", "cls"],
    )
    parser.add_argument("--include_aux_tokens", action="store_true")
    parser.add_argument(
        "--hidden_layers",
        type=str,
        help="comma-separated integers indicating which hidden layers to use for "
        "BERT embedding calculation. The token embedding layer corresponds to 0.",
    )
    parser.add_argument(
        "--avg_hidden_layers",
        type=str,
        help="comma-separated integers indicating averaged hidden layers embeddings. "
        "When this option is set, `features_npz` is assumed to have a "
        "'avg_hidden_<avg_hidden_layers>' key in each entry.",
    )
    parser.add_argument("--eps", type=int, default=int(1e6))
    parser.add_argument("--n_alphas", type=int, default=20)
    parser.add_argument("--alphas", type=str)
    parser.add_argument("--n_cv", type=int, default=10)
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--max_iter", type=int, default=int(1e5))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--n_test_resampling", type=int, default=1)
    parser.add_argument("--test_size_resampling", type=float, default=1.0)
    parser.add_argument("--n_seeds", type=int, default=32)
    parser.add_argument("--seeds_and_clfs_pkl", type=Path)
    parser.add_argument("--args_json", type=Path)

    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if args.n_seeds > 0 and (
        args.n_test_resampling != 1 or args.test_size_resampling != 1.0
    ):
        raise ValueError(
            "n_test_resampling must be 1 and test_size_resampling must be 1.0."
            f" Given values: {args.n_test_resampling}, {args.test_size_resampling}"
        )

    if args.feature_type == "cls" and not args.include_aux_tokens:
        raise ValueError("feature_type cls must be with --include_aux_tokens.")

    if args.hidden_layers is not None and args.avg_hidden_layers is not None:
        raise ValueError(f"Either hidden_layers or avg_hidden_layers can be specified.")

    if args.hidden_layers is not None:
        if args.feature_type != "avg":
            raise ValueError(
                "hidden_layers must be specified with feature_type == 'avg'."
            )
        data = np.load(args.features_npz, allow_pickle=True)
        hidden = data[data.files[0]].item()["hidden"]
        _n_layers = hidden.shape[0]
        if _n_layers != N_LAYERS + 1:
            raise ValueError(
                f"the number of embedded layers must be {N_LAYERS + 1}, "
                f"but given {_n_layers}."
            )
        hidden_layers = get_hidden_layer_indices(args.hidden_layers)
        if not all([l in range(N_LAYERS + 1) for l in hidden_layers]):
            raise ValueError(
                f"hidden_layers must be in [0, {N_LAYERS}], but given {hidden_layers}."
            )

    if args.avg_hidden_layers is not None:
        npz = np.load(args.features_npz, allow_pickle=True)
        if (
            f"avg_hidden_{args.avg_hidden_layers}"
            not in npz[npz.files[0]].item().keys()
        ):
            raise RuntimeError(
                f"The key avg_hidden_{args.avg_hidden_layers} "
                f"not found in {args.features_npz}."
            )
        if args.feature_type != "avg":
            raise ValueError(
                f"avg_hidden_layer must set with feature_type == 'avg', given {args.feature_type}"
            )

    if (args.seeds_and_clfs_pkl is None and args.args_json is not None) or (
        args.seeds_and_clfs_pkl is not None and args.args_json is None
    ):
        raise ValueError("seeds_and_clfs_pkl and args_json must be specified")


def load_args(
    args: argparse.Namespace, args_json: Path, skip_args: List[str] = ["output_dir"]
) -> argparse.Namespace:
    with args_json.open() as fin:
        params = json.load(fin)
    for param_name, value in params.items():
        if param_name in skip_args:
            continue
        setattr(args, param_name, value)
    return args


def get_hidden_layer_indices(hidden_layers: str) -> List[int]:
    return [int(s) for s in hidden_layers.split(",")]


def set_alphas(alphas_str: Optional[str]) -> Optional[np.ndarray]:
    if alphas_str is None:
        print(f"No alphas given. Setting automatically.")
        return None
    else:
        alphas = np.array(list(map(float, alphas_str.split(","))))
        print(f"alphas set to {alphas}")
        return alphas


def load_features(
    features_npz: Path,
    protein_seq_df: pd.DataFrame,
    include_aux_tokens: bool,
    feature_type: str = "avg",
    hidden_layers: Optional[str] = None,
    avg_hidden_layers: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(features_npz, allow_pickle=True)
    n_features = len(list(npz.keys()))
    len_df = protein_seq_df.shape[0]
    assert (
        len_df == n_features
    ), f"data length mismatch: protein_seq_df={len_df}, npz={n_features}"

    print(f"Loading features from {features_npz}")
    features = []
    targets = []
    variants = []

    for protein in tqdm.tqdm(protein_seq_df.itertuples(), total=len_df):
        key = f"{protein.protein}_{protein.Variant}_{protein.Index}"
        if hidden_layers is None:
            if avg_hidden_layers is not None:
                emb_key = f"avg_hidden_{avg_hidden_layers}"
            else:
                emb_key = "seq"
            _features = npz[key].item()[emb_key]

            if not include_aux_tokens:
                _features = _features[1:-1, ...]  # remove [CLS], [SEP] tokens
            if feature_type == "avg":
                feature = _features.mean(0)
            elif feature_type == "cls":
                feature = _features[0]
            else:
                raise NotImplementedError(
                    f"feature_type: {feature_type} is not implemented."
                )
        else:
            _features = npz[key].item()["hidden"]
            _layer_indices = get_hidden_layer_indices(hidden_layers)
            _features = _features[_layer_indices]
            if not include_aux_tokens:
                _features = _features[:, 1:-1, ...]
            feature = _features.mean(axis=(0, 1))

        features.append(feature)
        targets.append(protein.scaled_effect1)
        variants.append(protein.Variant)
    features = np.array(features)

    targets = np.array(targets)
    print(
        f"Done loading features and targets of shape: {features.shape}, {targets.shape}"
    )

    return features, targets, np.array(variants)


def train(
    args: argparse.Namespace, train_data: Tuple[np.ndarray, np.ndarray], seed: int
) -> Predictable:
    train_x, train_y = train_data

    if args.regressor_type == "lasso":
        alphas = set_alphas(args.alphas)
        clf = LassoCV(
            eps=args.eps,
            n_alphas=args.n_alphas,
            alphas=alphas,
            max_iter=args.max_iter,
            cv=args.n_cv,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
            random_state=seed,
        )
        clf.fit(train_x, train_y)
        best_alpha = clf.alpha_
        mse = clf.mse_path_[np.argmax((clf.alphas_ == best_alpha).astype(int))]
        r2 = clf.score(train_x, train_y)
        print(
            f"Cross validation results | MSE: {mse.mean():.5}, "
            f"Best alpha: {best_alpha:.5}, R^2: {r2:.5}"
        )
    elif args.regressor_type == "lasso_lars":
        # Hyperparameter search via cross validation as done in UniRep paper
        clf = LassoLarsCV(
            max_iter=args.max_iter,
            cv=args.n_cv,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
        )
        clf.fit(train_x, train_y)
    else:
        raise NotImplementedError()

    return clf


def evaluate(
    args: argparse.Namespace,
    clf: Predictable,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    print("Begin evaluation")

    n_test = len(test_x)
    sample_size = int(n_test * args.test_size_resampling)
    result = {
        "name": [args.protein_seq_csv.stem],
        "n_train": [train_x.shape[0]],
        "n_test": [n_test],
        "n_test_resampling": [sample_size],
    }

    metrics_raw = defaultdict(lambda: [])
    predictions = clf.predict(test_x)
    for metric_name, metric in METRICS.items():
        metrics_raw[metric_name].append(metric(test_y, predictions))
    metrics_raw = dict(metrics_raw)

    metrics = {
        metric_name: np.mean(values) for metric_name, values in metrics_raw.items()
    }
    result.update(metrics)
    # Add baseline metrics, which are metrics for constant value predictions.
    const_preds = np.array([train_y.mean()] * len(train_y))
    baselines = {
        f"baseline_{metric_name}": [metric(train_y, const_preds)]
        for metric_name, metric in METRICS.items()
    }
    aux_info = {
        "regressor_type": [args.regressor_type],
        "feature_type": [args.feature_type],
    }
    result.update(metrics)
    result.update(baselines)
    result.update(aux_info)
    result = pd.DataFrame.from_dict(result, orient="columns")
    print(f"Evaluation results:")
    print(f"{result}")

    return result, pd.DataFrame.from_dict(metrics_raw), predictions


def main(args: argparse.Namespace) -> None:
    check_args(args)

    if args.seeds_and_clfs_pkl is not None:
        # Loading the trained models and the args for evaluation
        load_args(args, args.args_json)
        seeds_and_clfs = joblib.load(args.seeds_and_clfs_pkl)
        seeds = [item[0] for item in seeds_and_clfs]
        clfs = [item[1] for item in seeds_and_clfs]
        run_training = False
    else:
        seeds = list(range(args.n_seeds)) if args.n_seeds > 0 else [SEED]
        clfs = []
        run_training = True

    args.output_dir.mkdir(exist_ok=True, parents=True)

    protein_seq_df = pd.read_csv(args.protein_seq_csv)
    features, targets, variants = load_features(
        args.features_npz,
        protein_seq_df,
        args.include_aux_tokens,
        feature_type=args.feature_type,
        hidden_layers=args.hidden_layers,
        avg_hidden_layers=args.avg_hidden_layers,
    )
    indices = np.arange(len(features))

    results = []
    raw = []
    predictions = []
    print("Begin seed loop")
    for idx in range(len(seeds)):
        seed = seeds[idx]
        print(f"seed: {seed}")
        train_indices, test_indices = train_test_split(
            indices, test_size=args.test_size, random_state=seed
        )
        train_x = features[train_indices]
        test_x = features[test_indices]
        train_y = targets[train_indices]
        test_y = targets[test_indices]
        test_variants = variants[test_indices]
        print(
            "Split dataset into: "
            f"{train_x.shape}, {train_y.shape} / {test_x.shape}, {test_y.shape}"
        )

        if run_training:
            clf = train(args, (train_x, train_y), seed=seed)
        else:
            clf = clfs[idx]
        evaluation_results, metrics_raw, _preds = evaluate(
            args, clf, train_x, test_x, train_y, test_y
        )
        metrics_raw.loc[:, "seed"] = seed
        clfs.append([seed, clf])
        results.append(evaluation_results)
        raw.append(metrics_raw)
        predictions.append(
            pd.DataFrame(
                {
                    "target": test_y,
                    "pred": _preds,
                    "variant": test_variants,
                    "seed": seed,
                }
            )
        )
    evaluation_results = pd.concat(results, ignore_index=True)
    metrics_raw = pd.concat(raw, ignore_index=True)
    predictions = pd.concat(predictions, ignore_index=True)

    if args.n_seeds > 0:
        for metric_name in METRICS:
            mean = evaluation_results[metric_name].mean()
            evaluation_results.loc[:, metric_name] = mean
            baseline_name = f"baseline_{metric_name}"
            baseline_mean = evaluation_results[baseline_name].mean()
            evaluation_results.loc[:, baseline_name] = baseline_mean
        evaluation_results = evaluation_results.drop_duplicates("name")
        print("Mean evaluation result:")
        print(f"{evaluation_results}")

    print(f"Results to dumped to {args.output_dir}")
    evaluation_results.to_csv(args.output_dir / "evaluation.csv", index=False)
    metrics_raw.to_csv(args.output_dir / "metrics_raw.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    joblib.dump(clfs, args.output_dir / "seeds_and_clfs.pkl")
    with open(args.output_dir / "args.json", "w") as fout:
        args = {k: str(v) if type(v) == PosixPath else v for k, v in vars(args).items()}
        json.dump(args, fout, indent=2)


if __name__ == "__main__":
    args = get_args()
    main(args)
