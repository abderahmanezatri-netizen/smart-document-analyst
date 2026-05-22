from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, List

class ClassificationInput(BaseModel):
    document_path: str = Field(..., description="Path to document file")

class ClassificationOutput(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    low_confidence: bool
    human_review_required: bool

class ExtractionInput(BaseModel):
    document_path: str

class ExtractionOutput(BaseModel):
    text: str
    word_count: int
    key_fields: Dict[str, str]

class SummaryInput(BaseModel):
    text: str
    max_sentences: int = 5

class SummaryOutput(BaseModel):
    summary: str
    bullets: List[str]

class ApprovalInput(BaseModel):
    classification: str
    confidence: float
    summary: str
    low_confidence: bool = False

class ApprovalOutput(BaseModel):
    approved: bool
    reviewer_note: str

class ReportInput(BaseModel):
    document_path: str
    classification: ClassificationOutput
    extraction: ExtractionOutput
    summary: SummaryOutput
    approval: ApprovalOutput

class ReportOutput(BaseModel):
    report_path: str
