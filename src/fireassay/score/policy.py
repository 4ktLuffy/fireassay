"""`PolicyScorer` (`policy@1.0.0`) — deterministic rule-pack matching.

Rules are loaded from YAML (see configs/policy.example.yaml) into
`ScoringContext.policy_rules` by the caller (the CLI's `run` command);
this module also provides `load_policy_rules` to do that loading.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from fireassay.models import Question, Score, SystemOutput
from fireassay.score.base import PolicyRule, ScoringContext

_NAME = "policy"
_VERSION = "1.0.0"


def load_policy_rules(path: Path | str) -> tuple[PolicyRule, ...]:
    """Load a policy rule pack from YAML.

    Expected shape::

        rules:
          - id: no_email_egress
            pattern: '[\\w.+-]+@[\\w-]+\\.[\\w.]+'
            applies_to: answer
            severity: high
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return tuple(PolicyRule(**rule) for rule in raw.get("rules", []))


class PolicyScorer:
    name = _NAME
    version = _VERSION
    requires_llm = False

    def score(self, question: Question, output: SystemOutput, ctx: ScoringContext) -> list[Score]:
        """Count how many rules in `ctx.policy_rules` fire against this
        output.

        `policy.violations` is the **count of rules that fired, not the
        count of regex matches** — a rule that matches an answer three
        times over still counts once, because the rule is what represents
        a distinct policy concern; counting matches would make one verbose
        answer with a single leaked email look three times as bad as three
        separate answers that each leak one.

        Emits `0.0` when `output.answer is None` (there is nothing to check
        rules with `applies_to: answer` against, and nothing rules with
        `applies_to: retrieved` in M1 ever match — see below).

        `rationale` lists the ids of the rules that fired (comma-separated),
        or is `None` when no rule fired.

        Known M1 limitation, not a bug: rules with `applies_to: retrieved`
        are accepted and validated but **never fire** in M1. `RetrievedChunk`
        (models.py) carries `doc_id`/`chunk_id`/`score`/`rank`/
        `char_start`/`char_end` — a *location*, not chunk text — so there
        is no retrieved *content* available at scoring time to check a
        pattern against. Checking retrieved content would require plumbing
        the corpus/chunk store into `Scorer.score`, which is out of scope
        for M1's per-question, corpus-agnostic scorer signature. This is
        called out explicitly rather than silently matching against, say,
        chunk ids (which would produce misleading "policy compliant"
        results).
        """
        scorer_tag = f"{self.name}@{self.version}"
        if output.answer is None:
            return [Score(metric="policy.violations", value=0.0, scorer=scorer_tag, rationale=None)]

        fired: list[str] = []
        for rule in ctx.policy_rules:
            if rule.applies_to != "answer":
                continue
            if re.search(rule.pattern, output.answer):
                fired.append(rule.id)

        rationale = ", ".join(fired) if fired else None
        return [
            Score(metric="policy.violations", value=float(len(fired)), scorer=scorer_tag, rationale=rationale)
        ]
