from __future__ import annotations

from pathlib import Path
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

from sda.tools.schemas import ExtractionInput, ExtractionOutput
from sda.utils.document_io import extract_text
from sda.utils.logging import log_action


class DocumentExtractionTool(BaseTool):
    name: str = "document_extraction_tool"
    description: str = "Extracts text and common key fields from a scanned document image."
    args_schema: type[ExtractionInput] = ExtractionInput

    def _looks_like_phone(self, value: str) -> bool:
        digits = re.sub(r"\D", "", value)

        # Real phone numbers usually have at least 9 digits.
        if len(digits) < 9:
            return False

        # Reject simple year ranges like 1963-1964 or 1964 -1965.
        if re.fullmatch(r"\s*\d{4}\s*[-–]\s*\d{4}\s*", value):
            return False

        # Reject values that are mostly dates/years.
        year_like_groups = re.findall(r"\b(?:19|20)\d{2}\b", value)
        if len(year_like_groups) >= 2:
            return False

        return True

    def _extract_first_valid_phone(self, text: str) -> str | None:
        phone_candidates = re.findall(
            r"(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}",
            text,
        )

        for candidate in phone_candidates:
            candidate = candidate.strip()
            if self._looks_like_phone(candidate):
                return candidate

        return None

    def _run(self, document_path: str) -> dict:
        try:
            text = extract_text(Path(document_path))
            fields = {}

            patterns = {
                "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                "date": r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4})\b",
                "amount": r"(?:USD|MAD|EUR|\$|€)\s?\d+(?:[,.]\d{2})?",
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    fields[key] = match.group(0)

            phone = self._extract_first_valid_phone(text)
            if phone:
                fields["phone"] = phone

            output = ExtractionOutput(
                text=text,
                word_count=len(text.split()),
                key_fields=fields,
            )

            log_action(
                "Extraction and Summarization Agent",
                "extract_content",
                "success",
                {
                    "word_count": output.word_count,
                    "key_fields": fields,
                },
            )

            return output.model_dump()

        except Exception as exc:
            log_action(
                "Extraction and Summarization Agent",
                "extract_content",
                "error",
                {"error": str(exc)},
            )
            raise