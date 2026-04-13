# spoof-lehmer

Computational investigation of the Hasanalizade extension graph for spoof
Lehmer factorizations. Companion code to the paper "The Extension Graph of
Spoof Lehmer Factorizations Is a Forest".

## What this code does

Given the parameter $k \ge 2$, a *spoof Lehmer factorization* is a multiset
$F = \{x_1, \ldots, x_r\}$ of integers $\ge 2$ satisfying
$k \cdot \prod (x_i - 1) = \prod x_i - 1$. The Hasanalizade extension lemma
shows that two-factor extensions of certain "plus-seed" factorizations are
governed by a single integer factorization. This code:

1. Enumerates plus-seeds and Lehmer factorizations via two complementary
   strategies (bottom-up recurrence and top-down cascade).
2. Stores them in a SQLite database for resumable, accumulating searches.
3. Verifies the forest property of the descent graph (no Lehmer
   factorization admits more than one descended pair).
4. Classifies plus-seeds (Fermat / Fermat-derived / sporadic) and computes
   the inclusion graph among them.
5. Investigates primitive (non-derived) Lehmer factorizations and computes
   structural invariants relevant to the conjecture that the set of
   primitives is finite.

## Search strategies

Three strategies live alongside each other under the `SearchStrategy`
protocol. They write to the same repository and their results are
deduplicated via the `(k, factors)` UNIQUE constraint.

### `RecurrenceStrategy` (bottom-up, box-bounded)

Walks the multiset tree from `g(empty) = k - 1` outward, terminating
each branch when the deficiency `g` reaches `-1` (Lehmer) or `+1`
(plus-seed). Bounded by `(max_r, max_N)`. Exhaustive *within its box*:
every $k$-Lehmer factorization with $|F| \le \mathtt{max\_r}$ and
$\epsilon(F) \le \mathtt{max\_N}$ is found. Discovers both Lehmer
factorizations and plus-seeds.

Use for: small-bound exhaustive coverage that feeds the cascade and
the analysis modules. Fast for $r \le 6$ at $\epsilon \le 10^{12}$.

### `CascadeStrategy` (top-down, factoring-driven)

Starts from known plus-seeds and generates extensions by factoring
$\Delta_L(N) = N^2 + N - 1$ (Lehmer extensions) and
$\Delta_S(N) = N^2 + N + 1$ (seed-to-seed extensions). Iterates to
fixpoint. Discovers nothing on its own — its coverage depends entirely
on which seeds it has been given. Cannot find primitive factorizations.

Use for: extending the bottom-up census to higher $r$ as a *lower
bound*. The cascade is what gets you interesting data at $r = 7, 8, 9$
without claiming exhaustiveness.

### `BoundsPropagationStrategy` (Molnar–Singh algorithm)

The original algorithm from
[grantmolnar/Spoof-Lehmer-Factorizations](https://github.com/grantmolnar/Spoof-Lehmer-Factorizations).
Walks prefixes maintaining a lower bound $L$ and upper bound $U$ on
$k(F)$ over all completions. The next-factor loop terminates because
$U$ decreases monotonically as $a$ grows — *no a priori bound on
$\epsilon$ is needed*. Uses Lehmer's congruence pruning ($a \not\equiv 1 \pmod{x}$
for any existing factor $x$).

Use for: provably exhaustive completeness claims at fixed $r$, without
committing to a `max_N` box. This is the strategy that lets you say
"there are no $k$-Lehmer factorizations with $|F| \le R$ other than
these," for whatever $R$ you can run to completion. Currently a pure
Python port; reaching $r = 7$ exhaustively will require porting the
inner loop to a faster language.

## Notation and glossary

The same letters appear throughout the code, the CLI flags, and the paper.

| Symbol | Name | Meaning |
| --- | --- | --- |
| $k$ | Lehmer parameter | The integer multiplier in the defining equation. $k = 2$ recovers the classical case; the code supports any $k \ge 2$. |
| $F$ | factorization | A spoof factorization: a sorted multiset of integers $\ge 2$. Stored as a `Factorization` value object. |
| $r = \lvert F \rvert$ | length | Number of factors in $F$, counted with multiplicity. The CLI flag `--max-r` bounds the search depth. |
| $N = \epsilon(F)$ | evaluation | The product $\prod_i x_i$. The CLI flag `--max-N` bounds the search by cutting off factorizations whose product exceeds this. |
| $\varphi^*(F)$ | spoof totient | The product $\prod_i (x_i - 1)$. |
| $g(F)$ | deficiency | $k \cdot \varphi^*(F) - \epsilon(F)$. A factorization is *Lehmer* iff $g = -1$ and a *plus-seed* iff $g = +1$. |
| $\Delta_L(N)$ | Lehmer delta | $N^2 + N - 1$. Factoring this gives Lehmer extensions of a plus-seed with evaluation $N$. |
| $\Delta_S(N)$ | seed delta | $N^2 + N + 1$. Factoring this gives plus-seed-to-plus-seed extensions. |

The descent formula $N(a + b - 1) = ab \cdot ((a-1)(b-1) + 1)$ involves
only $N$, $a$, and $b$ — it does not depend on $k$. This independence is
the structural fact powering the forest theorem, and it is enforced in the
code by giving `descent_holds(N, a, b)` no `k` parameter.

## Architecture

The codebase is organized around three independent seams: search
strategies, factoring backends, and persistence. Each is hidden behind a
protocol so a new implementation can be slotted in without changes
elsewhere.

```
src/spoof_lehmer/
├── domain/               # Pure mathematics. No I/O, no external deps.
│   ├── factorization.py    Factorization (frozen dataclass)
│   ├── descent.py          descent_holds(N, a, b)  -- no k argument
│   └── extension.py        lehmer_delta, seed_delta, extensions_from_seed
├── search/               # SearchStrategy protocol
│   ├── __init__.py         RecurrenceStrategy, CascadeStrategy, BoundsPropagationStrategy
│   └── bounds_propagation.py  Molnar-Singh original algorithm
├── factoring/            # FactoringBackend protocol
│   ├── __init__.py         TrialDivisionBackend, SympyBackend, AutoBackend
│   └── yafu_backend.py     YafuBackend (subprocess wrapper)
├── storage/              # FactorizationRepository protocol
│   └── __init__.py         SQLiteRepository, InMemoryRepository, Provenance
├── analysis/             # Question-driven modules. Read-only on the repo.
│   ├── __init__.py         run_census, format_report
│   ├── primitives.py       analyze_primitives, format_primitives_report
│   └── sporadic.py         analyze_sporadic_seeds, format_sporadic_report
└── cli/                  # Thin CLI wrappers
    ├── census.py           spoof-census    (populate the database)
    ├── extend.py           spoof-extend    (cascade-only top-down extension)
    ├── enumerate.py        spoof-enumerate (all-k sweep at fixed r, parity)
    └── analyze.py          spoof-analyze   (run reports against the database)
```

### The k-independence theorem, encoded in code

The descent formula
$N(a + b - 1) = ab \cdot ((a - 1)(b - 1) + 1)$
is independent of $k$. This is encoded structurally in
`domain/descent.py`: the function `descent_holds(N, a, b)` takes no `k`
argument, and the test `test_descent_formula_no_k_argument` uses
`inspect.signature` to enforce this invariant.

If a future refactor adds a `k` parameter, that test will fail
immediately, signaling a regression in the architectural intent.

## Installation

```bash
poetry install
```

For factoring integers larger than $10^{30}$, install
[yafu](https://github.com/bbuhrow/yafu) and ensure it is on your `PATH`.
The `AutoBackend` will detect it and use it automatically for large
inputs.

## CLI

### `spoof-census`: populate the database

```bash
# Reproduce the small-bound census
poetry run spoof-census --k 2 --max-r 5 --max-N 1e8 --cascade-rounds 2

# Push further
poetry run spoof-census --k 2 --max-r 7 --max-N 1e13 --cascade-rounds 5

# Other values of k
poetry run spoof-census --k 3 --max-r 6
poetry run spoof-census --k 5 --max-r 6
```

The database at `data/census.db` accumulates results across runs. Re-running
with larger bounds extends the existing census without recomputing.

Options:
- `--k`: Lehmer parameter $k$ in the equation $k \cdot \varphi^*(F) = \epsilon(F) - 1$ (default 2).
- `--max-r`: maximum factorization length $r$ for the bottom-up search.
- `--max-N`: maximum evaluation $N = \epsilon(F)$ for the bottom-up search (factorizations with larger product are skipped).
- `--cascade-rounds`: rounds of top-down seed extension. **Default 0 means run until natural termination** (the loop halts as soon as a round produces no new seeds). A positive integer caps the number of rounds for predictable wall time. Each round factors $\Delta_L(N) = N^2 + N - 1$ and $\Delta_S(N) = N^2 + N + 1$ for every known plus-seed.
- `--no-recurrence`, `--no-cascade`: skip one of the two strategies.
- `--db`: path to the SQLite database (default `data/census.db`).
- `--strategy`: bottom-up strategy, either `recurrence` (default) or `bounds` (the finishing-feasibility-enabled `BoundsPropagationStrategy`; ignores `--max-N`).

### `spoof-extend`: cascade-only top-down extension

A thin wrapper that runs only the `CascadeStrategy` against an existing
database. Useful when you've already populated seeds with `spoof-census`
and just want to push one or more additional rounds of extension.

```bash
poetry run spoof-extend --k 2 --rounds 0  # until fixpoint (default)
poetry run spoof-extend --k 2 --rounds 3  # cap at 3 rounds
```

### `spoof-enumerate`: every k-Lehmer at a given length and parity

Reproduces the behavior of the original Molnar-Singh repository: for a
given $r$ and parity, sweep every $k$ in the finite valid range
(odd: $k \le \lfloor (3/2)^r \rfloor$; even: $k \le 2^r - 1$) and
collect every $k$-Lehmer factorization that exists. Results go to a
SQLite database and, optionally, a JSON dump suitable for paper census
tables.

```bash
poetry run spoof-enumerate --max-r 6 --parity odd  --dump-json data/census_odd_r6.json
poetry run spoof-enumerate --max-r 5 --parity even --dump-json data/census_even_r5.json
```

Or use the script wrapper:

```bash
poetry run python scripts/enumerate_all.py --max-r 6 --parity odd
```

### `spoof-analyze`: run reports

```bash
# Census summary
poetry run spoof-analyze --k 2 --report census

# Primitives investigation (length distribution, residues, near-misses)
poetry run spoof-analyze --k 2 --report primitives

# Sporadic seed classification (Fermat / sporadic / inclusion families)
poetry run spoof-analyze --k 2 --report sporadic

# All three
poetry run spoof-analyze --k 2 --report all
```

The analyze command is read-only — it never mutates the database.

## Adding new investigations

The architecture is designed so each new research question is a new file
in `analysis/`, not a modification to the search or storage layers.

### Adding a new factoring backend

Implement the `FactoringBackend` protocol (a `name` field, a `max_n`
field, and a `factor(n) -> dict[int, int]` method). For example, a
CADO-NFS wrapper would look like the existing `yafu_backend.py`:

```python
class CadoBackend:
    name = "cado"
    max_n = 10**100

    def factor(self, n: int) -> dict[int, int]:
        # subprocess invocation, output parsing, verification
        ...
```

Then either pass it explicitly to `CascadeStrategy(backend=...)` or add it
to the dispatch chain in `AutoBackend`.

### Adding a new search strategy

Implement the `SearchStrategy` protocol:

```python
class TargetedPrimitiveSearch:
    name = "targeted-primitives"

    def __init__(self, k: int, ...): ...

    def discover(self, repo: FactorizationRepository) -> int:
        # find candidates, check if Lehmer, write to repo via repo.add()
        ...
```

### Adding a new analysis report

Read-only modules that consume the repository and produce a report. Add a
file like `analysis/residues.py`:

```python
def analyze_residue_classes(repo: FactorizationRepository, k: int) -> ResidueReport:
    ...

def format_residue_report(report: ResidueReport) -> str:
    ...
```

Then expose them from `analysis/__init__.py` and add a `--report residues`
option to `cli/analyze.py`. No changes to domain, search, factoring, or
storage are needed.

## Tests

```bash
poetry run pytest
```

The test suite has three tiers:

- **Mathematical invariants** (`test_descent_formula.py`): the descent
  formula has no $k$ parameter, known seeds and Lehmer factorizations are
  classified correctly, the `Factorization` value object validates input.

- **The forest property** (`test_forest_property.py`): after every search
  run, no factorization in the repository has more than one descended
  pair. If this fails, you have either found a counterexample to the
  forest theorem or a bug. Either way you want to know immediately.

- **Analysis modules** (`test_analysis.py`): the primitives report finds
  the known primitives, the sporadic classifier identifies the Fermat
  chain, the inclusion graph has the expected structure.

## License

(Your choice. Recommended: MIT or BSD for a research artifact.)
