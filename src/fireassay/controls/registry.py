"""The full list of controls `fireassay controls run` executes, in a fixed
order: the four controls with a real PASSED/FAILED implementation first,
then `null_questions` and `judge_calibration` — both always `NOT_RUN`
until M4 (see their modules for why) — last, so their guaranteed `NOT_RUN`
is easy to spot at the end of a report."""

from __future__ import annotations

from fireassay.controls.base import Control
from fireassay.controls.corpus_ablation import CorpusAblationControl
from fireassay.controls.identical_config import IdenticalConfigControl
from fireassay.controls.judge_calibration import JudgeCalibrationControl
from fireassay.controls.no_retrieval import NoRetrievalControl
from fireassay.controls.null_questions import NullQuestionsControl
from fireassay.controls.shuffled_gold import ShuffledGoldControl

ALL_CONTROLS: tuple[Control, ...] = (
    NoRetrievalControl(),
    ShuffledGoldControl(),
    CorpusAblationControl(),
    IdenticalConfigControl(),
    NullQuestionsControl(),
    JudgeCalibrationControl(),
)
