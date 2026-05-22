from __future__ import annotations
from pathlib import Path
from datetime import datetime
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
from sda.tools.schemas import ReportInput, ReportOutput, ClassificationOutput, ExtractionOutput, SummaryOutput, ApprovalOutput
from sda.utils.logging import log_action


class ReportWriterTool(BaseTool):
    name: str = "structured_report_writer"
    description: str = "Writes the final structured markdown report after human approval."
    args_schema: type[ReportInput] = ReportInput
    output_dir: Path

    def __init__(self, output_dir: Path):
        super().__init__(output_dir=Path(output_dir))

    def _run(self, document_path: str, classification: dict, extraction: dict, summary: dict, approval: dict) -> dict:
        try:
            c = ClassificationOutput(**classification)
            e = ExtractionOutput(**extraction)
            s = SummaryOutput(**summary)
            a = ApprovalOutput(**approval)
            if not a.approved:
                raise PermissionError("Human reviewer did not approve final report generation.")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out = self.output_dir / f"analysis_{Path(document_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            probs = "\n".join([f"- {k}: {v:.4f}" for k, v in c.class_probabilities.items()])
            fields = "\n".join([f"- **{k}**: {v}" for k, v in e.key_fields.items()]) or "- No standard fields detected."
            bullets = "\n".join(s.bullets)
            content = f"""# Smart Document Analysis Report

**Document:** `{document_path}`  
**Generated:** {datetime.now().isoformat(timespec='seconds')}  
**Human approval:** {a.reviewer_note}

## 1. Classification

- **Predicted class:** `{c.predicted_class}`
- **Confidence:** `{c.confidence:.4f}`
- **Low confidence:** `{c.low_confidence}`
- **Human review required:** `{c.human_review_required}`

### Class probabilities

{probs}

## 2. Extracted Content

- **Word count:** {e.word_count}

### Key fields

{fields}

## 3. Summary

{s.summary}

### Key bullets

{bullets}

## 4. Robustness Notes

- If confidence is below the configured threshold, the prediction must be treated as uncertain.
- The final report is generated only after human approval.
- Full agent/tool trace is available in `outputs/logs/agent_actions.jsonl`.

## 5. Extracted Text Preview

```text
{e.text[:2500]}
```
"""
            out.write_text(content, encoding="utf-8")
            output = ReportOutput(report_path=str(out))
            log_action("Report Generation Agent", "write_final_report", "success", output.model_dump())
            return output.model_dump()
        except Exception as exc:
            log_action("Report Generation Agent", "write_final_report", "error", {"error": str(exc)})
            raise
