# Papers

Drafts accompanying the spoof-lehmer computational work.

## Current draft

- **`spoof_lehmer_forests_and_finishing.tex`** — G. Molnar, *Spoof
  Lehmer Factorizations: Forest Structure, Plus-Seed Chains, and a
  Closed-Form Finishing Algorithm* (17 pages, draft). The merged
  paper, combining the forest-structure result with the
  closed-form finishing algorithm and the plus-seed chain theory.

  Main results:
  1. Only $\alpha = +1$ seeds (plus-seeds) produce positive extensions.
  2. The descent formula is $k$-independent.
  3. The extension graph $\mathcal{G}_k$ is a forest for every $k \ge 2$.
  4. Three closed-form finishing-feasibility identities at $s = r-1, r-2, r-3$.
  5. Complete $r \le 7$ census across all valid $k$ (107 factorizations).
  6. Plus-seed chain extension theorem: every plus-seed initiates an infinite chain.
  7. Plus-seed closedness theorem: $\tau(M)/2$ length-$(s+2)$ Lehmer extensions
     from any plus-seed of length $s$, where $M = E^2 + E - 1$.
  8. Closed-form $r = 8$ partial enumeration via plus-seed closedness.
  9. Sharp $u_{\max} \le 2E + 1$ bound on the $s = r - 3$ outer loop.

  The full hierarchical census appendix lists all 107 factorizations
  organized by chain root → plus-seed → Lehmer descendants, with
  truly primitive factorizations grouped at the end.

## Archive

The two precursor drafts that were merged into the current paper are
preserved in `archive/`:

- **`archive/extension_graph_forest.tex`** — original forest paper
  (theorems 1-3 above).
- **`archive/finishing_feasibility.tex`** — original finishing paper
  (theorems 4, 5, 8 above).

These should not be revised further. Please update only the merged
paper.
