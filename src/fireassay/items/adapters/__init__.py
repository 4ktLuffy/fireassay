"""Adapters that build `items.core.ItemResponses`/`ItemMeta` from a
concrete source. `tabular.py` reads anyone's CSV/JSONL (no fireassay
dependency at all); `store.py` reads fireassay's own DB and is the only
module in `fireassay.items` permitted to import `fireassay.store`.
"""
