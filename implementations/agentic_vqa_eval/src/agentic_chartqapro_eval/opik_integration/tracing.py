"""Lightweight wrappers around opik Trace/Span for the MEP pipeline.

All helpers accept ``None`` as the client/trace and become no-ops, so the
rest of the codebase can call them unconditionally.
"""

import contextlib
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def sample_trace(
    client,
    sample_id: str,
    question: str,
    expected_output: str,
    question_type: str,
    config_name: str,
    run_id: str,
    project_name: str = "chartqapro-eval",
):
    """Open an Opik Trace for one sample; yield the trace (or None)."""
    if client is None:
        yield None
        return

    trace = client.trace(
        name=f"chartqapro/{sample_id}",
        input={"question": question, "expected_output": expected_output},
        tags=[config_name, question_type, "chartqapro"],
        metadata={
            "run_id": run_id,
            "config": config_name,
            "question_type": question_type,
        },
        project_name=project_name,
    )
    try:
        yield trace
    finally:
        trace.end()


def open_llm_span(
    trace,
    name: str,
    input_data: dict,
    model: str,
    metadata: Optional[dict] = None,
    parent_span_id: Optional[str] = None,
):
    """Create an Opik LLM span on *trace* (or return None)."""
    if trace is None:
        return None
    return trace.span(
        name=name,
        type="llm",
        input=input_data,
        model=model,
        metadata=metadata or {},
        parent_span_id=parent_span_id,
    )


def close_span(
    span,
    output: Optional[dict] = None,
    usage: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    """End an Opik span (no-op if span is None)."""
    if span is None:
        return
    kwargs: dict = {}
    if output is not None:
        kwargs["output"] = output
    if usage:
        kwargs["usage"] = usage
    if error:
        from opik.types import ErrorInfoDict

        kwargs["error_info"] = ErrorInfoDict(message=error)
    span.end(**kwargs)


def log_trace_scores(trace, scores: dict) -> None:
    """Log a dict of {metric_name: float} as feedback scores on *trace*."""
    if trace is None:
        return
    for name, value in scores.items():
        if isinstance(value, (int, float)):
            with contextlib.suppress(Exception):
                trace.log_feedback_score(name=name, value=float(value))
