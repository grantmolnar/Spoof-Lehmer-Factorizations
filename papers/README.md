# Papers

Drafts accompanying the spoof-lehmer computational work.

- **`extension_graph_forest.tex`** — G. Molnar, *The Extension Graph of
  Spoof Lehmer Factorizations Is a Forest* (draft). Proves that for
  every $k \ge 2$ the extension graph $\mathcal{G}_k$ is a forest, and
  gives the census for $k = 2$ with $r \le 6$ (17 plus-seeds,
  39 Lehmer factorizations, 9 primitives).

- **`finishing_feasibility.tex`** — G. Molnar and G. Singh, *Finishing-
  Feasibility at the Last Two Levels for Spoof Lehmer Enumeration*
  (draft). Derives the closed-form solves at $s = r-1$ and $s = r-2$
  implemented in `src/spoof_lehmer/search/bounds_propagation.py`.

Build with `pdflatex <file>.tex` (both use standard AMS packages;
the forest paper additionally uses TikZ for Figure 1).
