from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from crewai import Agent, Task, Crew, Process, LLM
except Exception:
    Agent = Task = Crew = Process = LLM = None

from sda.tools.classifier_tool import CNNDocumentClassifierTool
from sda.tools.extraction_tool import DocumentExtractionTool
from sda.tools.summarizer_tool import SimpleSummarizerTool
from sda.tools.hitl_tool import HumanApprovalTool
from sda.tools.report_tool import ReportWriterTool
from sda.utils.logging import log_action


class SmartDocumentCrew:
    """CrewAI orchestration with an audited tool-grounded execution workflow."""

    def __init__(
        self,
        model_path: Path,
        classes_path: Path,
        confidence_threshold: float,
        output_dir: Path,
        use_llm: bool = True,
        auto_approve: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.use_llm = use_llm

        self.classifier_tool = CNNDocumentClassifierTool(
            model_path,
            classes_path,
            confidence_threshold,
        )
        self.extraction_tool = DocumentExtractionTool()
        self.summary_tool = SimpleSummarizerTool()
        self.approval_tool = HumanApprovalTool(auto_approve=auto_approve)
        self.report_tool = ReportWriterTool(self.output_dir)

        self.agents = self._build_agents()

    def _llm(self):
        if not self.use_llm:
            return None

        if LLM is None:
            raise ImportError(
                "CrewAI LLM class is not available. Check your CrewAI installation."
            )

        model = os.getenv("OLLAMA_MODEL", "ollama/llama3.1")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        return LLM(
            model=model,
            base_url=base_url,
        )

    def _build_agents(self):
        if Agent is None:
            raise ImportError("CrewAI is not installed. Run: pip install -r requirements.txt")

        llm = self._llm()
        common = {"verbose": True, "allow_delegation": False}

        if llm is not None:
            common["llm"] = llm

        orchestrator = Agent(
            role="Orchestrator Manager Agent",
            goal=(
                "Coordinate the specialist agents, enforce the project requirements, "
                "and ensure the final analysis is completed through the audited workflow."
            ),
            backstory=(
                "A careful project manager that validates the workflow and ensures that "
                "classification, extraction, summarization, human approval, and report generation "
                "happen in the correct order."
            ),
            tools=[],
            **common,
        )

        classifier = Agent(
            role="Document Classification Agent",
            goal=(
                "Plan and validate the document classification step. "
                "The actual CNN execution is handled by the audited Python workflow."
            ),
            backstory=(
                "A specialist in visual document layout classification. It understands that "
                "the trained PyTorch CNN must be used, but does not call tools directly in CrewAI "
                "to avoid invalid tool arguments or repeated calls."
            ),
            tools=[],
            **common,
        )

        extractor = Agent(
            role="Extraction and Summarization Agent",
            goal=(
                "Plan and validate OCR extraction, key-field detection, and summarization. "
                "The actual extraction and summarization tools are executed by the audited Python workflow."
            ),
            backstory=(
                "A document analyst that focuses on factual extraction and avoids hallucinating "
                "missing information. It does not call tools directly in CrewAI."
            ),
            tools=[],
            **common,
        )

        reporter = Agent(
            role="Report Generation Agent",
            goal=(
                "Review the workflow status and confirm that the final report must be generated "
                "only by the audited Python workflow after human approval."
            ),
            backstory=(
                "A reporting supervisor. It does not call tools directly because the final "
                "human approval and report writing are handled by the deterministic audited workflow."
            ),
            tools=[],
            **common,
        )

        log_action(
            "System",
            "build_crewai_agents",
            "success",
            {
                "agents": [
                    orchestrator.role,
                    classifier.role,
                    extractor.role,
                    reporter.role,
                ]
            },
        )

        return {
            "orchestrator": orchestrator,
            "classifier": classifier,
            "extractor": extractor,
            "reporter": reporter,
        }

    def _build_crewai_tasks(self, document_path: Path):
        if Task is None:
            raise ImportError("CrewAI is not installed.")

        return [
            Task(
                description=(
                    f"Plan the document classification step for {document_path}. "
                    "Do not call tools directly. "
                    "Explain that the audited Python workflow will call the trained CNN classifier tool "
                    "with the document path and will return predicted class, confidence, class probabilities, "
                    "and the human-review flag."
                ),
                expected_output=(
                    "A short classification plan confirming that the trained CNN classifier tool "
                    "will be executed by the audited workflow."
                ),
                agent=self.agents["classifier"],
            ),
            Task(
                description=(
                    f"Plan the extraction and summarization step for {document_path}. "
                    "Do not call tools directly. "
                    "Explain that the audited Python workflow will extract OCR text, detect key fields, "
                    "and summarize the extracted content."
                ),
                expected_output=(
                    "A short extraction and summarization plan confirming that OCR extraction "
                    "and summarization will be executed by the audited workflow."
                ),
                agent=self.agents["extractor"],
            ),
            Task(
                description=(
                    "Review the previous planning outputs. "
                    "Do not invent file paths, confidence scores, class names, extracted fields, or tool outputs. "
                    "Do not generate the final report yourself. "
                    "Do not write fake JSON. "
                    "Only state that the final audited report will be generated by the structured report workflow "
                    "after the external human-in-the-loop checkpoint."
                ),
                expected_output=(
                    "A short validation note confirming that classification, extraction, summarization, "
                    "and human approval are required before final report generation."
                ),
                agent=self.agents["reporter"],
            ),
        ]

    def analyze(self, document_path: Path) -> Path:
        log_action(
            "Orchestrator Agent",
            "start_analysis",
            "started",
            {
                "document_path": str(document_path),
                "use_llm": self.use_llm,
            },
        )

        if not self.use_llm:
            log_action(
                "Orchestrator Agent",
                "no_llm_mode",
                "started",
                {
                    "reason": "LLM disabled by user",
                    "workflow": "tool_grounded_multi_agent_workflow",
                },
            )
            return self._run_tool_grounded_workflow(document_path)

        try:
            tasks = self._build_crewai_tasks(document_path)
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
            )

            log_action(
                "Orchestrator Agent",
                "crewai_kickoff",
                "started",
                {"document_path": str(document_path)},
            )

            crew_result = crew.kickoff(inputs={"document_path": str(document_path)})

            log_action(
                "Orchestrator Agent",
                "crewai_kickoff",
                "success",
                {
                    "document_path": str(document_path),
                    "crew_result_preview": str(crew_result)[:500],
                },
            )

            return self._run_tool_grounded_workflow(document_path)

        except Exception as exc:
            log_action(
                "Orchestrator Agent",
                "crewai_kickoff",
                "error",
                {
                    "error": str(exc),
                    "fallback": "tool_grounded_multi_agent_workflow",
                },
            )
            return self._run_tool_grounded_workflow(document_path)

    def _safe_json_loads(self, raw, fallback_key: str = "text"):
        if isinstance(raw, dict):
            return raw

        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return {fallback_key: raw}

        return {fallback_key: str(raw)}

    def _run_tool_grounded_workflow(self, document_path: Path) -> Path:
        try:
            log_action(
                "Document Classification Agent",
                "classify_document",
                "started",
                {"document_path": str(document_path)},
            )

            classification_raw = self.classifier_tool._run(str(document_path))
            classification = self._safe_json_loads(classification_raw, fallback_key="error")

            confidence = float(classification.get("confidence", 0.0))
            needs_review = bool(classification.get("needs_human_review", confidence < 0.60))

            classification["low_confidence"] = confidence < 0.60
            classification["human_review_required"] = needs_review

            log_action(
                "Document Classification Agent",
                "classify_document",
                "success",
                {
                    "predicted_class": classification.get("predicted_class"),
                    "confidence": confidence,
                    "needs_human_review": needs_review,
                },
            )

            log_action(
                "Extraction and Summarization Agent",
                "extract_document_text",
                "started",
                {"document_path": str(document_path)},
            )

            extraction_raw = self.extraction_tool._run(str(document_path))
            extraction = self._safe_json_loads(extraction_raw, fallback_key="text")

            extracted_text = (
                extraction.get("text")
                or extraction.get("extracted_text")
                or extraction.get("content")
                or str(extraction_raw)
            )

            log_action(
                "Extraction and Summarization Agent",
                "extract_document_text",
                "success",
                {
                    "text_preview": extracted_text[:300],
                    "text_length": len(extracted_text),
                },
            )

            log_action(
                "Extraction and Summarization Agent",
                "summarize_document",
                "started",
                {"text_length": len(extracted_text)},
            )

            summary_raw = self.summary_tool._run(extracted_text)
            summary = self._safe_json_loads(summary_raw, fallback_key="summary")

            log_action(
                "Extraction and Summarization Agent",
                "summarize_document",
                "success",
                {
                    "summary_preview": str(summary.get("summary", summary_raw))[:300],
                },
            )

            log_action(
                "Report Generation Agent",
                "request_human_approval",
                "started",
                {
                    "classification": classification.get("predicted_class", "unknown"),
                    "confidence": confidence,
                },
            )

            approval_raw = self.approval_tool._run(
                classification=classification.get("predicted_class", "unknown"),
                confidence=confidence,
                summary=summary.get("summary", str(summary_raw)),
            )
            approval = self._safe_json_loads(approval_raw, fallback_key="approval")

            log_action(
                "Report Generation Agent",
                "request_human_approval",
                "success",
                {"approval": approval},
            )

            log_action(
                "Report Generation Agent",
                "write_final_report",
                "started",
                {"document_path": str(document_path)},
            )

            report = self.report_tool._run(
                str(document_path),
                classification,
                extraction,
                summary,
                approval,
            )

            report = self._safe_json_loads(report, fallback_key="report_path")

            log_action(
                "Report Generation Agent",
                "write_final_report",
                "success",
                {"report_path": report.get("report_path")},
            )

            log_action(
                "Orchestrator Agent",
                "finish_analysis",
                "success",
                {"report_path": report.get("report_path")},
            )

            return Path(report.get("report_path", "outputs/reports/final_report.md"))

        except Exception as exc:
            log_action(
                "Orchestrator Agent",
                "finish_analysis",
                "error",
                {"error": str(exc)},
            )
            raise