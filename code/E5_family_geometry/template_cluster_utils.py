"""Template-cluster helpers for E5 family geometry."""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass


_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PairTemplate:
    template_signature: str
    orig_value: str
    pert_value: str
    diff_kind: str


def normalise_text(text: str) -> str:
    return _WS_RE.sub(" ", text.strip())


def tokenise(text: str) -> list[str]:
    return normalise_text(text).split(" ")


def _span_text(tokens: list[str], start: int, end: int) -> str:
    return " ".join(tokens[start:end]).strip()


def infer_pair_template(orig: str, pert: str) -> PairTemplate:
    """Infer a template by replacing orig/pert changed spans with <SLOT>.

    This is intentionally conservative and transparent. It clusters pairs that share
    the same unchanged sentence context and differ only in slot values.
    """
    o_toks = tokenise(orig)
    p_toks = tokenise(pert)
    matcher = difflib.SequenceMatcher(a=o_toks, b=p_toks, autojunk=False)
    template: list[str] = []
    orig_values: list[str] = []
    pert_values: list[str] = []
    kinds: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            template.extend(o_toks[i1:i2])
            continue
        template.append("<SLOT>")
        orig_values.append(_span_text(o_toks, i1, i2))
        pert_values.append(_span_text(p_toks, j1, j2))
        kinds.append(tag)

    if not orig_values and not pert_values:
        return PairTemplate(
            template_signature=normalise_text(orig),
            orig_value="",
            pert_value="",
            diff_kind="equal",
        )

    signature = normalise_text(" ".join(template))
    return PairTemplate(
        template_signature=signature,
        orig_value=" | ".join(orig_values),
        pert_value=" | ".join(pert_values),
        diff_kind="+".join(kinds),
    )


def stable_template_id(family: str, signature: str) -> str:
    digest = hashlib.sha1(f"{family}\n{signature}".encode("utf-8")).hexdigest()[:10]
    return f"{family}::tpl_{digest}"


def lexical_stats(values: list[str]) -> dict[str, float]:
    tokens: list[str] = []
    nonempty = [v for v in values if v]
    for value in nonempty:
        tokens.extend(tokenise(value))
    unique_values = len(set(nonempty))
    unique_tokens = len(set(tokens))
    return {
        "n_values": float(len(nonempty)),
        "unique_values": float(unique_values),
        "unique_value_fraction": float(unique_values / max(1, len(nonempty))),
        "unique_tokens": float(unique_tokens),
        "mean_value_tokens": float(sum(len(tokenise(v)) for v in nonempty) / max(1, len(nonempty))),
    }
