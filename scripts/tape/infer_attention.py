import argparse
import json
import sys
import typing
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO

sys.path.insert(
    0, str(Path(__file__).absolute().parents[2] / "src" / "third_party" / "tape")
)
from tape import ProteinBertModel, TAPETokenizer


def get_attention_layers(layers_str: typing.Optional[str]) -> typing.Optional[str]:
    return None if layers_str is None else [int(i) for i in layers_str.split(",")]


def check_args(args: argparse.Namespace) -> None:
    if args.attention_layers is not None:
        attention_layers = get_attention_layers(args.attention_layers)
        assert all(
            [i in list(range(args.n_layers)) for i in attention_layers]
        ), f"Attention layers must be in [0, {args.n_layers - 1}], but given {attention_layers}."


def create_model(
    from_pretrained: str, no_hidden: bool, no_attention: bool
) -> ProteinBertModel:
    config = ProteinBertModel.config_class.from_pretrained(from_pretrained)
    config.output_attentions = not no_attention
    config.output_hidden_states = not no_hidden
    model = ProteinBertModel.from_pretrained(from_pretrained, config=config)
    return model


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda:0")
    model = create_model(str(args.from_pretrained), args.no_hidden, args.no_attention)
    model.to(device)
    tokenizer = TAPETokenizer(vocab="iupac")

    attention_layers = get_attention_layers(args.attention_layers)

    outputs = {}
    for record in SeqIO.parse(str(args.fasta), "fasta"):
        name = record.id
        sequence = str(record.seq)
        token_ids = torch.tensor([tokenizer.encode(sequence)], device=device)
        output = model(token_ids)  # seq, pool, hidden, attention
        if args.save_seq:
            outputs[name + "_seq"] = output[0].detach().cpu().numpy()
        if not args.no_hidden:
            outputs[name + "_hidden"] = np.array(
                [item.detach().cpu().numpy() for item in output[2]]
            )
            if not args.no_attention:
                attention = np.array(
                    [item.detach().cpu().numpy() for item in output[3]]
                )
                if attention_layers is not None:
                    attention = attention[attention_layers]
                outputs[name + "_attn"] = attention
        else:
            if not args.no_attention:
                attention = np.array(
                    [item.detach().cpu().numpy() for item in output[2]]
                )
                if attention_layers is not None:
                    attention = attention[attention_layers]
                outputs[name + "_attn"] = attention

    with open(args.output_dir / f"{args.fasta.stem}.npz", "wb") as fout:
        np.savez(fout, **outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("from_pretrained", type=Path)
    parser.add_argument("fasta", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--save_seq", action="store_true")
    parser.add_argument("--no_hidden", action="store_true")
    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument(
        "--attention_layers",
        type=str,
        help="comma-separated 0-indexed integers to specify attention layers",
    )
    args = parser.parse_args()
    check_args(args)
    main(args)
