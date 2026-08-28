from __future__ import annotations

from pathlib import Path

from fireassay.models import Question
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import chunk_corpus, load_corpus

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _build_system(top_k: int = 5) -> BM25System:
    docs = load_corpus(FIXTURES_DIR / "corpus.jsonl")
    chunks = list(chunk_corpus(docs, size=800, overlap=100))
    return BM25System(chunks, top_k=top_k)


def _question(text: str) -> Question:
    return Question(text=text, qtype="factual", difficulty="easy", provenance="synthetic")


def test_ranks_obviously_relevant_doc_first() -> None:
    system = _build_system()
    output = system.answer(_question("How do I reset my password?"))
    assert output.retrieved
    assert output.retrieved[0].doc_id == "doc01"
    assert output.retrieved[0].rank == 1


def test_ranks_obviously_relevant_doc_first_different_topic() -> None:
    system = _build_system()
    output = system.answer(_question("What is the API rate limit per minute?"))
    assert output.retrieved[0].doc_id == "doc07"


def test_deterministic_across_two_runs() -> None:
    system = _build_system()
    question = _question("How do I cancel my subscription?")
    out1 = system.answer(question)
    out2 = system.answer(question)
    assert [(rc.doc_id, rc.chunk_id, rc.rank, rc.score) for rc in out1.retrieved] == [
        (rc.doc_id, rc.chunk_id, rc.rank, rc.score) for rc in out2.retrieved
    ]


def test_deterministic_across_separately_constructed_systems() -> None:
    question = _question("How do I cancel my subscription?")
    out1 = _build_system().answer(question)
    out2 = _build_system().answer(question)
    assert [(rc.doc_id, rc.rank) for rc in out1.retrieved] == [(rc.doc_id, rc.rank) for rc in out2.retrieved]


def test_answer_is_none_and_abstained_true_with_no_tokens() -> None:
    system = _build_system()
    output = system.answer(_question("anything at all"))
    assert output.answer is None
    assert output.abstained is True
    assert output.tokens_in == 0
    assert output.tokens_out == 0
    assert "total" in output.latency_ms


def test_generates_answers_is_false() -> None:
    """BM25System is retrieval-only; runner.run_matrix reads this flag to
    suppress AbstentionScorer entirely (see test_abstention_scorer.py)."""
    system = _build_system()
    assert system.generates_answers is False


def test_top_k_limits_number_of_results() -> None:
    system = _build_system(top_k=3)
    output = system.answer(_question("How do I reset my password?"))
    assert len(output.retrieved) <= 3


def test_ranks_are_1_based_and_contiguous() -> None:
    system = _build_system(top_k=4)
    output = system.answer(_question("How do I change my email address?"))
    assert [rc.rank for rc in output.retrieved] == list(range(1, len(output.retrieved) + 1))


def test_retrieved_chunks_carry_char_start_and_end_from_the_source_chunk() -> None:
    docs = load_corpus(FIXTURES_DIR / "corpus.jsonl")
    chunks = list(chunk_corpus(docs, size=800, overlap=100))
    chunk_by_id = {c.chunk_id: c for c in chunks}
    system = BM25System(chunks, top_k=5)

    output = system.answer(_question("How do I reset my password?"))
    for rc in output.retrieved:
        source_chunk = chunk_by_id[rc.chunk_id]
        assert rc.char_start == source_chunk.char_start
        assert rc.char_end == source_chunk.char_end


def test_tie_break_deterministic_regardless_of_corpus_load_order() -> None:
    """Real bug, not hypothetical: sorting candidates by score alone
    leaves ties in corpus load order, which is not reproducible across a
    corpus loaded/generated differently on another machine. A query that
    shares no terms with any document forces every chunk to score exactly
    0.0 -- a full tie across the whole corpus -- which must still resolve
    to the identical ranking whether the corpus was loaded forward or
    reversed.
    """
    docs = load_corpus(FIXTURES_DIR / "corpus.jsonl")
    chunks_forward = list(chunk_corpus(docs, size=800, overlap=100))
    chunks_reversed = list(reversed(chunks_forward))

    system_forward = BM25System(chunks_forward, top_k=12)
    system_reversed = BM25System(chunks_reversed, top_k=12)

    question = _question("zzzznonexistentqueryword thiswordisnotinanydocument")
    out_forward = system_forward.answer(question)
    out_reversed = system_reversed.answer(question)

    assert len(out_forward.retrieved) == 12
    assert all(rc.score == 0.0 for rc in out_forward.retrieved)  # confirms this is a full tie
    assert [rc.chunk_id for rc in out_forward.retrieved] == [rc.chunk_id for rc in out_reversed.retrieved]
