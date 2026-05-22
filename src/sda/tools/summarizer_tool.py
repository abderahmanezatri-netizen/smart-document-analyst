from __future__ import annotations
import re
try:
    from crewai.tools import BaseTool
except Exception:
    try:
        from crewai_tools import BaseTool
    except Exception:
        class BaseTool:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
from sda.tools.schemas import SummaryInput, SummaryOutput
from sda.utils.logging import log_action


class SimpleSummarizerTool(BaseTool):
    name: str = "local_summarizer_tool"
    description: str = "Free local extractive summarizer fallback. No paid API required."
    args_schema: type[SummaryInput] = SummaryInput

    def _run(self, text: str, max_sentences: int = 5) -> dict:
        try:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 20]
            if not sentences:
                sentences = [text[:300]] if text else ["No extractable text found."]
            scored = []
            keywords = ["total", "experience", "objective", "invoice", "dear", "report", "conclusion", "skills", "amount"]
            for s in sentences:
                score = len(s.split()) + sum(10 for k in keywords if k.lower() in s.lower())
                scored.append((score, s))
            top = [s for _, s in sorted(scored, reverse=True)[:max_sentences]]
            bullets = [f"- {s[:180]}" for s in top]
            output = SummaryOutput(summary=" ".join(top), bullets=bullets)
            log_action("Extraction & Summarization Agent", "summarize_content", "success", {"sentences": len(top)})
            return output.model_dump()
        except Exception as exc:
            log_action("Extraction & Summarization Agent", "summarize_content", "error", {"error": str(exc)})
            raise
