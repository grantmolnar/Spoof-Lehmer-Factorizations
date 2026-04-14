"""LaTeX rendering of the organized census.

This module follows the rendering-as-strings approach: it operates on
:class:`CensusOrganization` objects and emits LaTeX strings.  It does
no I/O of its own; callers are responsible for writing to files.

Two top-level renderers are provided:

  * :func:`render_chain_inventory` --- compact summary table (one row
    per chain).
  * :func:`render_full_census` --- hierarchical longtable listing every
    Lehmer factorization grouped under its plus-seed parent.

Both produce self-contained LaTeX fragments suitable for ``\\input{}``
into a paper preamble or direct copy-paste into a section.
"""
from __future__ import annotations

from collections.abc import Iterator
from math import prod

from spoof_lehmer.analysis.census_organization import CensusOrganization
from spoof_lehmer.analysis.chain import chain_parent


def _fmt_factors(F: tuple[int, ...]) -> str:
    """Format a factor tuple as LaTeX math content.

    Empty tuple becomes ``\\emptyset``; otherwise comma-separated with
    LaTeX thin-spaces.
    """
    if not F:
        return r"\emptyset"
    return "(" + ",\\,".join(str(x) for x in F) + ")"


def _depth_in_chain(P: tuple[int, ...]) -> int:
    """Depth of $P$ within its chain (0 = root)."""
    depth = 0
    cur = P
    while True:
        par = chain_parent(cur)
        if par is None:
            return depth
        cur = par
        depth += 1


def render_chain_inventory(org: CensusOrganization) -> str:
    """Render the chain inventory table (Table A).

    Compact summary: one row per chain, columns chain name, root,
    member lengths observed in the census, and total Lehmer
    descendants from any chain member.
    """
    rows: list[str] = []
    # Sort: Fermat first (root length 0 or 1), then sporadic by root length.
    sorted_chains = sorted(
        org.chains.values(),
        key=lambda c: (0 if c.is_fermat else 1, len(c.root), c.root),
    )
    for chain in sorted_chains:
        # Member lengths observed: walk chain up to the longest member appearing
        # in our census's parent set.
        parent_seeds = (
            set(org.descended_to_children) | set(org.companion_to_child)
        )
        max_len = max((len(p) for p in parent_seeds), default=0)
        members_in_chain = chain.members(up_to_length=max_len)
        member_lengths = sorted({len(m) for m in members_in_chain
                                 if m in parent_seeds})
        if not member_lengths:
            # Only happens if root itself isn't a parent (e.g. empty Fermat root
            # if we don't have any extensions yet); skip such chains.
            continue
        n_descendants = org.descendants_of_chain(chain)
        root_disp = "$\\emptyset$" if not chain.root else f"${_fmt_factors(chain.root)}$"
        lengths_str = ", ".join(f"${n}$" for n in member_lengths)
        rows.append(
            f"{chain.label} & {root_disp} & {lengths_str} & ${n_descendants}$ \\\\"
        )

    body = "\n".join(rows)
    return f"""\\begin{{tabular}}{{@{{}}lllr@{{}}}}
\\toprule
Chain & Root & Member lengths & Lehmer descendants \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}"""


def _iter_chain_organized_parents(org: CensusOrganization) -> Iterator[tuple[int, ...]]:
    """Yield parent plus-seeds in chain-organized order: Fermat chain
    first by depth, then sporadic chains by root length and root.
    """
    parent_seeds = set(org.descended_to_children) | set(org.companion_to_child)
    # Group by chain root.
    by_chain: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for P in parent_seeds:
        from spoof_lehmer.analysis.chain import chain_root
        root = chain_root(P)
        by_chain.setdefault(root, []).append(P)
    # Order chains: Fermat first.
    sorted_roots = sorted(
        by_chain.keys(),
        key=lambda r: (0 if r in ((), (3,)) else 1, len(r), r),
    )
    for root in sorted_roots:
        # Within chain: sort by depth (i.e. by length).
        chain_members = sorted(by_chain[root], key=lambda P: (len(P), P))
        yield from chain_members


def render_full_census(org: CensusOrganization) -> str:
    """Render the full hierarchical census as a ``longtable``.

    Requires ``\\usepackage{longtable}`` and ``\\usepackage{booktabs}``
    in the host document.  The longtable page-breaks gracefully across
    pages.
    """
    lines: list[str] = []
    lines.append(r"\begin{longtable}{@{}p{0.95\linewidth}@{}}")
    lines.append(r"\caption{The $r \le 7$ $k = 2$-Lehmer factorizations grouped by")
    lines.append(r"parent plus-seed.  Each plus-seed appears as a section header,")
    lines.append(r"followed by its descended-pair and Lehmer-companion children.")
    lines.append(r"Truly primitive factorizations (no plus-seed parent of either")
    lines.append(r"kind) appear at the end.}\label{tab:census-organized}\\")
    lines.append(r"\toprule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{1}{@{}l}{\textit{(continued from previous page)}}\\")
    lines.append(r"\toprule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{1}{r@{}}{\textit{(continued on next page)}}\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for P in _iter_chain_organized_parents(org):
        chain = org.chain_for(P)
        depth = _depth_in_chain(P)
        anno = f"{chain.label} root" if depth == 0 else f"{chain.label}, depth {depth}"
        E = prod(P) if P else 1
        P_disp = "$\\emptyset$" if not P else f"${_fmt_factors(P)}$"
        lines.append(
            rf"\textbf{{{P_disp}}} \quad ({anno}, $\eps = {E}$): \\"
        )
        if P in org.companion_to_child:
            F = org.companion_to_child[P]
            lines.append(
                rf"\quad ${_fmt_factors(F)}$ \hfill {{\small companion}} \\"
            )
        for F in org.descended_to_children.get(P, []):
            lines.append(
                rf"\quad ${_fmt_factors(F)}$ \hfill {{\small descended}} \\"
            )
        lines.append(r"\addlinespace")

    if org.primitives_by_length:
        lines.append(r"\midrule")
        lines.append(r"\textbf{Truly primitive factorizations (no plus-seed parent)}: \\")
        for r in sorted(org.primitives_by_length):
            for F in sorted(org.primitives_by_length[r]):
                lines.append(
                    rf"\quad ${_fmt_factors(F)}$ \hfill {{\small $r = {r}$}} \\"
                )

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def render_chain_inventory_table(org: CensusOrganization) -> str:
    """Wrap :func:`render_chain_inventory` in a ``table`` float with
    a caption, suitable for direct ``\\input{}``.
    """
    inner = render_chain_inventory(org)
    return f"""\\begin{{table}}[ht]
\\centering
\\caption{{Plus-seed chains observed in the census ($k = 2$, odd).
Each chain is rooted at a \\emph{{fresh}} plus-seed and continues
by $x = \\eps + 2$.  The \\emph{{member lengths}} column lists the
lengths at which the chain has been observed; the \\emph{{Lehmer
descendants}} column gives the total Lehmer factorizations
descending from any chain member, counting both
Hasanalizade-descended and Lehmer-companion children.}}\\label{{tab:chains-auto}}
\\smallskip
{inner}
\\end{{table}}"""
