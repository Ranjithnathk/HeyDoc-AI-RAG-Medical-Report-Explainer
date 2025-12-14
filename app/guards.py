from __future__ import annotations

from typing import List, Dict, Any


DISCLAIMER_TEXT = (
    "Important: I am not a doctor and this is not medical advice. "
    "This explanation is for informational purposes only."
)


def validate_report_input(report_text: str) -> tuple[bool, str]:
    """
    Validate report input before processing.
    Returns (is_valid, error_message).
    """
    if not report_text or not report_text.strip():
        return False, "Please provide a radiology report to analyze."

    if len(report_text.strip()) < 30:
        return False, "The provided text is too short to be a radiology report."

    return True, ""


def validate_question_input(question: str) -> tuple[bool, str]:
    if not question or not question.strip():
        return False, "Please enter a question."

    return True, ""


def validate_retrieval_results(results: List[Dict[str, Any]], min_results: int = 2) -> tuple[bool, str]:
    """
    Ensure we have enough evidence to answer reliably.
    """
    if not results or len(results) < min_results:
        return False, (
            "I don't have enough reliable evidence from the knowledge base "
            "to answer this question."
        )

    return True, ""


def enforce_disclaimer(answer: str) -> str:
    """
    Ensure disclaimer is always present.
    """
    if DISCLAIMER_TEXT not in answer:
        return answer.strip() + "\n\n" + DISCLAIMER_TEXT
    return answer
