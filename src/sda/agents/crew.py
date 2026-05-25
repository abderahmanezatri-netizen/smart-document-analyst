from __future__ import annotations

import ast
import json
import os
import re
import time
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
from sda.tools.workflow_state import (
    get_active_document,
    get_active_step,
    get_auto_approve,
    get_output_dir,
    set_active_document,
    set_auto_approve,
    set_output_dir,
    set_step,
)


def reject_fake_tool_call_output(output):
    text = str(output)

    blocked_phrases = [
        "Function Call",
        '"name": "cnn_document_classifier"',
        '"name": "document_extraction_tool"',
        '"name": "local_summarizer_tool"',
        '"name": "structured_report_writer"',
        '"name": "human_approval_checkpoint"',
        "path/to/document",
        "/path/to/report.md",
        "path/to/report",
        "John Doe",
        "Extracted Text from document_extraction_tool",
        "Bullet point 1",
        "I will call",
        "I can't execute",
        "I cannot execute",
        "Here is the JSON format function call",
        "Here is the function call",
    ]

    if any(phrase in text for phrase in blocked_phrases):
        return (
            False,
            "Do not write planned or fake function-call JSON. Execute the available tool, "
            "then answer only from the real tool output.",
        )

    return True, output


def _json_from_output(output):
    if isinstance(output, dict):
        return output

    text = str(output).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        raw = match.group(0)

        try:
            return json.loads(raw)
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None

    return None


def validate_classifier_output(output):
    ok, message = reject_fake_tool_call_output(output)

    if not ok:
        actual = get_active_step("classification")
        if actual:
            return True, json.dumps(actual, indent=2)
        return ok, message

    text = str(output)

    required = [
        "predicted_class",
        "confidence",
        "class_probabilities",
        "needs_human_review",
        "model_architecture",
    ]

    if not all(item in text for item in required):
        actual = get_active_step("classification")
        if actual:
            return True, json.dumps(actual, indent=2)

        return (
            False,
            "Return the actual cnn_document_classifier tool output, including "
            "predicted_class, confidence, class_probabilities, needs_human_review, "
            "and model_architecture.",
        )

    parsed = _json_from_output(output)

    if parsed:
        predicted_class = parsed.get("predicted_class")
        confidence = parsed.get("confidence")
        class_probabilities = parsed.get("class_probabilities")

        if not isinstance(predicted_class, str):
            actual = get_active_step("classification")
            if actual:
                return True, json.dumps(actual, indent=2)

            return (
                False,
                "The real classifier output must contain predicted_class as a class name string.",
            )

        if not isinstance(confidence, (int, float)):
            actual = get_active_step("classification")
            if actual:
                return True, json.dumps(actual, indent=2)

            return (
                False,
                "The real classifier output must contain confidence as a numeric value.",
            )

        if not isinstance(class_probabilities, dict):
            actual = get_active_step("classification")
            if actual:
                return True, json.dumps(actual, indent=2)

            return (
                False,
                "The real classifier output must contain class_probabilities as a dictionary.",
            )

        document_path = get_active_document()
        if document_path:
            set_step(document_path, "classification", parsed)

    return True, output


def validate_extraction_output(output):
    ok, message = reject_fake_tool_call_output(output)

    if not ok:
        actual = get_active_step("extraction")
        if actual:
            return True, json.dumps(actual, indent=2)
        return ok, message

    text = str(output)

    required = ["word_count", "key_fields"]

    if not all(item in text for item in required):
        actual = get_active_step("extraction")
        if actual:
            return True, json.dumps(actual, indent=2)

        return (
            False,
            "Return the actual document_extraction_tool result with word_count and key_fields.",
        )

    document_path = get_active_document()
    parsed = _json_from_output(output)

    if document_path and parsed:
        set_step(document_path, "extraction", parsed)

    return True, output


def validate_summary_output(output):
    def run_real_summary_from_extraction():
        document_path = get_active_document()
        extraction = get_active_step("extraction")

        if not document_path or not extraction:
            return None

        extracted_text = (
            extraction.get("text")
            or extraction.get("extracted_text")
            or extraction.get("content")
            or ""
        )

        if not extracted_text.strip():
            return None

        try:
            actual = SimpleSummarizerTool()._run(text=extracted_text)
        except TypeError:
            actual = SimpleSummarizerTool()._run(extracted_text)
        except Exception:
            return None

        set_step(document_path, "summary", actual)
        return json.dumps(actual, indent=2)

    ok, message = reject_fake_tool_call_output(output)

    if not ok:
        actual = get_active_step("summary")
        if actual:
            return True, json.dumps(actual, indent=2)

        actual = run_real_summary_from_extraction()
        if actual:
            return True, actual

        return ok, message

    text = str(output)
    required = ["summary", "bullets"]

    if not all(item in text for item in required):
        actual = get_active_step("summary")
        if actual:
            return True, json.dumps(actual, indent=2)

        actual = run_real_summary_from_extraction()
        if actual:
            return True, actual

        return (
            False,
            "Return the actual local_summarizer_tool result with summary and bullets.",
        )

    parsed = _json_from_output(output)

    if parsed:
        summary_text = str(parsed.get("summary", "")).strip().lower()
        bullets = parsed.get("bullets", [])

        bad_summary = (
            not summary_text
            or summary_text == "no extractable text found."
            or summary_text == "no extractable text found"
        )

        bad_bullets = (
            not bullets
            or bullets == ["- No extractable text found."]
            or bullets == ["No extractable text found."]
        )

        if bad_summary or bad_bullets:
            actual = run_real_summary_from_extraction()
            if actual:
                return True, actual

        document_path = get_active_document()
        if document_path:
            set_step(document_path, "summary", parsed)

    return True, output


def validate_approval_output(output):
    ok, message = reject_fake_tool_call_output(output)

    if not ok:
        actual = get_active_step("approval")
        if actual:
            return True, json.dumps(actual, indent=2)

        if get_auto_approve() and get_active_document():
            actual = {
                "approved": True,
                "reviewer_note": "Auto-approved for demo/test run.",
            }
            set_step(get_active_document(), "approval", actual)
            return True, json.dumps(actual, indent=2)

        return ok, message

    text = str(output)

    required = ["approved", "reviewer_note"]

    if not all(item in text for item in required):
        actual = get_active_step("approval")
        if actual:
            return True, json.dumps(actual, indent=2)

        if get_auto_approve() and get_active_document():
            actual = {
                "approved": True,
                "reviewer_note": "Auto-approved for demo/test run.",
            }
            set_step(get_active_document(), "approval", actual)
            return True, json.dumps(actual, indent=2)

        return (
            False,
            "Return the actual human_approval_checkpoint result with approved and reviewer_note.",
        )

    document_path = get_active_document()
    parsed = _json_from_output(output)

    if document_path and parsed:
        set_step(document_path, "approval", parsed)

    return True, output


def validate_report_output(output):
    def write_real_report():
        document_path = get_active_document()
        output_dir = get_output_dir()

        if document_path and output_dir:
            try:
                actual = ReportWriterTool(Path(output_dir))._run(document_path)
                set_step(document_path, "report", actual)
                return json.dumps(actual, indent=2)
            except Exception:
                return None

        return None

    ok, message = reject_fake_tool_call_output(output)

    if not ok:
        actual = get_active_step("report")
        if actual:
            return True, json.dumps(actual, indent=2)

        actual = write_real_report()
        if actual:
            return True, actual

        return ok, message

    text = str(output)

    match = re.search(r"report_path['\"]?\s*[:=]\s*['\"]([^'\"]+)", text)

    if match:
        path = Path(match.group(1))

        if path.exists():
            parsed = _json_from_output(output)
            document_path = get_active_document()
            if document_path and parsed:
                set_step(document_path, "report", parsed)
            return True, output

        actual = get_active_step("report")
        if actual:
            return True, json.dumps(actual, indent=2)

        actual = write_real_report()
        if actual:
            return True, actual

        return (
            False,
            "The report_path must point to an existing generated report file.",
        )

    if "report_path" not in text:
        actual = get_active_step("report")
        if actual:
            return True, json.dumps(actual, indent=2)

        actual = write_real_report()
        if actual:
            return True, actual

        return (
            False,
            "Return the actual structured_report_writer tool output, including an existing report_path.",
        )

    actual = get_active_step("report")
    if actual:
        return True, json.dumps(actual, indent=2)

    actual = write_real_report()
    if actual:
        return True, actual

    return (
        False,
        "The report_path must point to an existing generated report file.",
    )


class SmartDocumentCrew:
    """CrewAI orchestration with explicit agent-owned tool execution."""

    def __init__(
        self,
        model_path: Path,
        classes_path: Path,
        confidence_threshold: float,
        output_dir: Path,
        use_llm: bool = True,
        auto_approve: bool = False,
        fallback_manual: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.use_llm = use_llm
        self.fallback_manual = fallback_manual
        self.auto_approve = auto_approve
        self.confidence_threshold = confidence_threshold

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

        common = {
            "verbose": True,
            "allow_delegation": False,
        }

        if llm is not None:
            common["llm"] = llm
            common["function_calling_llm"] = llm

        orchestrator = Agent(
            role="Orchestrator Manager Agent",
            goal=(
                "Coordinate the specialist agents, enforce the project requirements, "
                "and ensure the final analysis is completed through agent-owned tools."
            ),
            backstory=(
                "A careful project manager that coordinates the workflow and ensures that "
                "classification, extraction, summarization, human approval, and report generation "
                "happen in the correct order."
            ),
            tools=[],
            **common,
        )

        classifier = Agent(
            role="Document Classification Agent",
            goal=(
                "Use the trained PyTorch CNN classifier tool to classify scanned document images "
                "and decide whether human review is required."
            ),
            backstory=(
                "A specialist in visual document layout classification. It understands that "
                "the trained PyTorch CNN is the source of truth for document class predictions."
            ),
            tools=[self.classifier_tool],
            **common,
        )

        extractor = Agent(
            role="Extraction and Summarization Agent",
            goal=(
                "Use OCR extraction and local summarization tools to extract text, detect fields, "
                "and summarize the scanned document."
            ),
            backstory=(
                "A document analyst that focuses on factual extraction and avoids hallucinating "
                "missing information."
            ),
            tools=[self.extraction_tool, self.summary_tool],
            **common,
        )

        reporter = Agent(
            role="Report Generation Agent",
            goal=(
                "Use exactly one assigned tool at a time: first the human approval checkpoint, "
                "then the structured report writer after approval."
            ),
            backstory=(
                "A reporting supervisor that enforces the human-in-the-loop checkpoint before "
                "writing the final structured report. It never writes function-call JSON as text; "
                "it executes the assigned tool."
            ),
            tools=[self.approval_tool, self.report_tool],
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
                ],
                "tool_assignments": {
                    classifier.role: [self.classifier_tool.name],
                    extractor.role: [self.extraction_tool.name, self.summary_tool.name],
                    reporter.role: [self.approval_tool.name, self.report_tool.name],
                },
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

        document_path_text = str(document_path)

        orchestration_task = Task(
            description=(
                "Write one short execution plan for the Smart Document Analyst workflow. "
                "Do not call tools. Do not write JSON. The required order is: first classify "
                "the scanned image with the trained CNN tool, then extract text with OCR, then "
                "summarize the real OCR text, then request human approval, then write the final "
                "report. Explain that each step must be handled by the specialist agent that owns "
                "the required tool."
            ),
            expected_output=(
                "One short plain-English execution plan naming the specialist agents and the order."
            ),
            agent=self.agents["orchestrator"],
            guardrail=reject_fake_tool_call_output,
        )

        classification_task = Task(
            description=(
                f"Use the cnn_document_classifier tool for this document path: {document_path_text}. "
                "The trained PyTorch CNN tool is the only valid source of classification results. "
                "Do not create JSON tool-call syntax yourself. "
                "After the tool runs, return the actual tool result with predicted_class, confidence, "
                "class_probabilities, low_confidence, human_review_required, needs_human_review, "
                "and model_architecture."
            ),
            expected_output=(
                "The real cnn_document_classifier tool output for the document."
            ),
            agent=self.agents["classifier"],
            tools=[self.classifier_tool],
            context=[orchestration_task],
            guardrail=validate_classifier_output,
        )

        extraction_task = Task(
            description=(
                f"Use the document_extraction_tool for this document path: {document_path_text}. "
                "Do not invent text or fields. "
                "After the tool runs, return the actual OCR extraction output with text preview, "
                "word_count, and key_fields."
            ),
            expected_output=(
                "The real document_extraction_tool output, including OCR text, word_count, and key_fields."
            ),
            agent=self.agents["extractor"],
            tools=[self.extraction_tool],
            context=[classification_task],
            guardrail=validate_extraction_output,
        )

        summary_task = Task(
            description=(
                f"Use the local_summarizer_tool for this document path: {document_path_text}. "
                "The tool must summarize the real OCR text saved by the previous extraction step. "
                "Do not invent information. "
                "After the tool runs, return the actual summary tool output with summary and bullets."
            ),
            expected_output=(
                "The real local_summarizer_tool output with summary and bullets."
            ),
            agent=self.agents["extractor"],
            tools=[self.summary_tool],
            context=[extraction_task],
            guardrail=validate_summary_output,
        )

        approval_task = Task(
            description=(
                f"Use the human_approval_checkpoint tool for this document path: {document_path_text}. "
                "This checkpoint must happen before report generation. "
                "The tool uses the real classifier and summary outputs saved by previous tools. "
                "After the tool runs, return the actual approval result."
            ),
            expected_output=(
                "The real human_approval_checkpoint tool output with approved and reviewer_note."
            ),
            agent=self.agents["reporter"],
            tools=[self.approval_tool],
            context=[classification_task, extraction_task, summary_task],
            guardrail=validate_approval_output,
        )

        report_task = Task(
            description=(
                f"Use the structured_report_writer tool for this document path: {document_path_text}. "
                "The tool loads classification, extraction, summary, and approval from workflow state. "
                "Do not write JSON function-call syntax yourself. "
                "After the tool runs, return the actual tool result containing report_path."
            ),
            expected_output=(
                "The real structured_report_writer output containing report_path."
            ),
            agent=self.agents["reporter"],
            tools=[self.report_tool],
            context=[
                orchestration_task,
                classification_task,
                extraction_task,
                summary_task,
                approval_task,
            ],
            guardrail=validate_report_output,
        )

        return [
            orchestration_task,
            classification_task,
            extraction_task,
            summary_task,
            approval_task,
            report_task,
        ]

    def _extract_report_path(self, crew_result, started_after: float) -> Path:
        text = str(crew_result)

        match = re.search(r"report_path['\"]?\s*[:=]\s*['\"]([^'\"]+)", text)
        if match:
            path = Path(match.group(1))
            if path.exists():
                return path

        if self.output_dir.exists():
            reports = [
                path
                for path in self.output_dir.glob("analysis_*.md")
                if path.stat().st_mtime >= started_after
            ]

            if reports:
                return max(reports, key=lambda path: path.stat().st_mtime)

        raise RuntimeError(
            "CrewAI completed, but no final report_path was produced. "
            "Run with --fallback-manual or --no-llm to use the deterministic tool loop."
        )

    def analyze(self, document_path: Path) -> Path:
        set_active_document(document_path)
        set_auto_approve(self.auto_approve)
        set_output_dir(self.output_dir)

        log_action(
            "Orchestrator Agent",
            "start_analysis",
            "started",
            {
                "document_path": str(document_path),
                "use_llm": self.use_llm,
                "fallback_manual": self.fallback_manual,
            },
        )

        if not self.use_llm:
            log_action(
                "Orchestrator Agent",
                "no_llm_mode",
                "started",
                {
                    "reason": "LLM disabled by user",
                    "workflow": "deterministic_agent_tool_loop",
                },
            )
            return self._run_agent_tool_loop(document_path)

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
                {
                    "document_path": str(document_path),
                },
            )

            started_after = time.time()

            crew_result = crew.kickoff(
                inputs={
                    "document_path": str(document_path),
                }
            )

            log_action(
                "Orchestrator Agent",
                "crewai_kickoff",
                "success",
                {
                    "document_path": str(document_path),
                    "crew_result_preview": str(crew_result)[:500],
                },
            )

            return self._extract_report_path(crew_result, started_after)

        except Exception as exc:
            log_action(
                "Orchestrator Agent",
                "crewai_kickoff",
                "error",
                {
                    "error": str(exc),
                    "fallback": self.fallback_manual,
                },
            )

            if self.fallback_manual:
                log_action(
                    "Orchestrator Agent",
                    "fallback_manual",
                    "started",
                    {
                        "reason": "CrewAI/Ollama tool execution failed",
                    },
                )
                return self._run_agent_tool_loop(document_path)

            raise

    def _safe_json_loads(self, raw, fallback_key: str = "text"):
        if isinstance(raw, dict):
            return raw

        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return {fallback_key: raw}

        return {fallback_key: str(raw)}

    def _run_owned_tool(self, agent_key: str, tool_name: str, *args, **kwargs):
        agent = self.agents[agent_key]
        owned_tools = {tool.name: tool for tool in agent.tools}

        if tool_name not in owned_tools:
            raise ValueError(f"{agent.role} does not own tool: {tool_name}")

        log_action(
            agent.role,
            "agent_tool_call",
            "started",
            {
                "tool": tool_name,
            },
        )

        result = owned_tools[tool_name]._run(*args, **kwargs)

        log_action(
            agent.role,
            "agent_tool_call",
            "success",
            {
                "tool": tool_name,
            },
        )

        return result

    def _run_summary_tool_safely(self, document_path: Path, extracted_text: str):
        try:
            return self._run_owned_tool(
                "extractor",
                self.summary_tool.name,
                text=extracted_text,
            )
        except TypeError:
            pass

        try:
            return self._run_owned_tool(
                "extractor",
                self.summary_tool.name,
                document_path=str(document_path),
            )
        except TypeError:
            pass

        return self._run_owned_tool(
            "extractor",
            self.summary_tool.name,
            extracted_text,
        )

    def _run_agent_tool_loop(self, document_path: Path) -> Path:
        try:
            log_action(
                "Orchestrator Manager Agent",
                "orchestrate_agent_tool_loop",
                "started",
                {
                    "workflow_order": [
                        "classification",
                        "extraction",
                        "summarization",
                        "human_approval",
                        "report_generation",
                    ]
                },
            )

            log_action(
                "Orchestrator Manager Agent",
                "route_to_agent",
                "started",
                {
                    "next_agent": "Document Classification Agent",
                    "reason": "Classify the scanned document before extracting or reporting.",
                },
            )

            classification_raw = self._run_owned_tool(
                "classifier",
                self.classifier_tool.name,
                str(document_path),
            )

            classification = self._safe_json_loads(
                classification_raw,
                fallback_key="error",
            )

            confidence = float(classification.get("confidence", 0.0))

            needs_review = bool(
                classification.get(
                    "needs_human_review",
                    confidence < self.confidence_threshold,
                )
            )

            classification["low_confidence"] = confidence < self.confidence_threshold
            classification["human_review_required"] = needs_review

            set_step(document_path, "classification", classification)

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
                "Orchestrator Manager Agent",
                "route_to_agent",
                "success",
                {
                    "completed_agent": "Document Classification Agent",
                    "next_agent": "Extraction and Summarization Agent",
                    "reason": "Classification is available; extract text with OCR.",
                },
            )

            extraction_raw = self._run_owned_tool(
                "extractor",
                self.extraction_tool.name,
                str(document_path),
            )

            extraction = self._safe_json_loads(
                extraction_raw,
                fallback_key="text",
            )

            extracted_text = (
                extraction.get("text")
                or extraction.get("extracted_text")
                or extraction.get("content")
                or str(extraction_raw)
            )

            set_step(document_path, "extraction", extraction)

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
                "Orchestrator Manager Agent",
                "route_to_agent",
                "success",
                {
                    "completed_agent": "Extraction and Summarization Agent",
                    "next_agent": "Extraction and Summarization Agent",
                    "reason": "OCR text is available; summarize the real extracted text.",
                },
            )

            summary_raw = self._run_summary_tool_safely(
                document_path=document_path,
                extracted_text=extracted_text,
            )

            summary = self._safe_json_loads(
                summary_raw,
                fallback_key="summary",
            )

            if not summary.get("summary") and extracted_text.strip():
                preview = extracted_text.strip().replace("\n", " ")
                summary["summary"] = preview[:500]

            if "bullets" not in summary:
                summary["bullets"] = []

            set_step(document_path, "summary", summary)

            log_action(
                "Extraction and Summarization Agent",
                "summarize_document",
                "success",
                {
                    "summary_preview": str(summary.get("summary", summary_raw))[:300],
                },
            )

            log_action(
                "Orchestrator Manager Agent",
                "route_to_agent",
                "success",
                {
                    "completed_agent": "Extraction and Summarization Agent",
                    "next_agent": "Report Generation Agent",
                    "reason": "Summary is ready; request human approval before writing report.",
                },
            )

            approval_raw = self._run_owned_tool(
                "reporter",
                self.approval_tool.name,
                classification=classification.get("predicted_class", "unknown"),
                confidence=confidence,
                summary=summary.get("summary", str(summary_raw)),
                low_confidence=classification["low_confidence"],
            )

            approval = self._safe_json_loads(
                approval_raw,
                fallback_key="approval",
            )

            set_step(document_path, "approval", approval)

            log_action(
                "Report Generation Agent",
                "request_human_approval",
                "success",
                {
                    "approval": approval,
                },
            )

            log_action(
                "Orchestrator Manager Agent",
                "route_to_agent",
                "success",
                {
                    "completed_agent": "Report Generation Agent",
                    "next_agent": "Report Generation Agent",
                    "reason": "Human approval received; write the final report.",
                },
            )

            report = self._run_owned_tool(
                "reporter",
                self.report_tool.name,
                str(document_path),
                classification,
                extraction,
                summary,
                approval,
            )

            report = self._safe_json_loads(
                report,
                fallback_key="report_path",
            )

            set_step(document_path, "report", report)

            log_action(
                "Report Generation Agent",
                "write_final_report",
                "success",
                {
                    "report_path": report.get("report_path"),
                },
            )

            log_action(
                "Orchestrator Agent",
                "finish_analysis",
                "success",
                {
                    "report_path": report.get("report_path"),
                },
            )

            log_action(
                "Orchestrator Manager Agent",
                "orchestrate_agent_tool_loop",
                "success",
                {
                    "report_path": report.get("report_path"),
                },
            )

            return Path(report.get("report_path", "outputs/reports/final_report.md"))

        except Exception as exc:
            log_action(
                "Orchestrator Agent",
                "finish_analysis",
                "error",
                {
                    "error": str(exc),
                },
            )
            raise
