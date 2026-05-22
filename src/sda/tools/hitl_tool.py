from __future__ import annotations
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
from sda.tools.schemas import ApprovalInput, ApprovalOutput
from sda.utils.logging import log_action


class HumanApprovalTool(BaseTool):
    name: str = "human_approval_checkpoint"
    description: str = "Requires human approval before final report generation."
    args_schema: type[ApprovalInput] = ApprovalInput
    auto_approve: bool = False

    def __init__(self, auto_approve: bool = False):
        super().__init__(auto_approve=auto_approve)

    def _run(self, classification: str, confidence: float, summary: str, low_confidence: bool = False) -> dict:
        try:
            if self.auto_approve:
                note = "Auto-approved for demo/test run."
                approved = True
            else:
                print("\nHUMAN-IN-THE-LOOP CHECKPOINT")
                print(f"Predicted class: {classification} (confidence={confidence:.3f})")
                if low_confidence:
                    print("WARNING: Low confidence. Please review carefully.")
                print("Summary preview:")
                print(summary[:800])
                answer = input("Approve final report generation? [y/N]: ").strip().lower()
                approved = answer in {"y", "yes"}
                note = "Approved by human reviewer." if approved else "Rejected by human reviewer."
            output = ApprovalOutput(approved=approved, reviewer_note=note)
            log_action("Orchestrator Agent", "human_approval_checkpoint", "success", output.model_dump())
            return output.model_dump()
        except Exception as exc:
            log_action("Orchestrator Agent", "human_approval_checkpoint", "error", {"error": str(exc)})
            raise
