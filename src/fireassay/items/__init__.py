"""`fireassay items` -- item analysis for evaluation sets: "is this eval
question worth keeping?"

`items.core` is pure (no I/O, no `fireassay.store` import -- see its module
docstring for why that is load-bearing) and usable standalone by someone
who has never heard of fireassay. `items.adapters.tabular` builds its
input from a plain CSV/JSONL; `items.adapters.store` (the only module in
this package allowed to import `fireassay.store`) builds it from
fireassay's own DB. `items.review` and `items.seed` add blind review and
deterministic seeded corruptions for measuring detector precision/recall
without depending on any of the above.

Deliberately no package-level re-export of `adapters.store`: importing
`fireassay.items` itself must never require touching the store, so a
standalone caller only ever imports `fireassay.items.core` (plus
`fireassay.items.adapters.tabular`/`review`/`seed` as needed) and never
pulls in `fireassay.store` unless they explicitly reach for
`fireassay.items.adapters.store`.
"""
