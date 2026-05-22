# Smart Document Analyst

Smart Document Analyst is a multi-agent AI system for scanned document analysis. It takes a scanned document image as input, classifies the document with a trained PyTorch CNN, extracts text using OCR, summarizes the content, asks for human approval, and generates a structured Markdown report.

The project was built for the Integrated Project: Building Multi-Agent AI Systems.

## Submission Links

### Dataset

The dataset used in this project is based on scanned real-world document images from Kaggle.

Dataset link: https://www.kaggle.com/datasets/shaz13/real-world-documents-collections

In the local project, the dataset should be placed under:

```text
data/real_raw/docs-sm/
```

### Trained Model

The trained PyTorch CNN model is too large for GitHub web upload, so it is provided through Google Drive.

Model link: https://drive.google.com/file/d/1_Os89OvO8noVGviHDVc9XM4I0g0bOlMG/view?usp=sharing

After downloading, place the file under:

```text
models/document_cnn.pt
```

### Demo Video

The demo video shows the project structure, model evaluation files, end-to-end document analysis, human-in-the-loop approval, generated report, JSON logs, and a second test on another document class.

Demo video link: https://drive.google.com/file/d/10BhiPLxFm0TzibWhdLN4XZ4v8HtPiTMN/view?usp=sharing

### Report

The final PDF report is included in the repository:

```text
Smart_Document_Analyst_Report.pdf
```

### Slides

Slides should be added before final submission.

## Supported Input

This version supports scanned document images only:

```text
.png
.jpg
.jpeg
.tif
.tiff
.bmp
```

PDF, TXT, DOCX, and other non-image formats are not supported in the final workflow. This is intentional because the classifier is a CNN model that learns from the visual layout and appearance of scanned documents.

## What the System Does

For each scanned image, the system performs the following steps:

```text
Scanned image
  -> CNN document classification
  -> OCR text extraction
  -> Key-field detection
  -> Local summarization
  -> Human approval checkpoint
  -> Markdown report generation
  -> JSON action logging
```

The output report includes the predicted document class, confidence score, class probabilities, extracted text statistics, detected fields, summary, approval status, and robustness notes.

## Architecture

The project uses CrewAI for the multi-agent layer and Ollama/Llama 3.1 as the local LLM backend.

The agents are:

- **Orchestrator Manager Agent**: coordinates the workflow and starts the analysis.
- **Document Classification Agent**: validates the classification step and ensures the trained CNN is used.
- **Extraction and Summarization Agent**: handles OCR extraction, simple field detection, and summarization.
- **Report Generation Agent**: ensures the final report is generated only after human approval.

The final execution is handled by a controlled Python workflow. CrewAI and the LLM are used for planning and validation, while the Python workflow executes the tools in a fixed order. This avoids unreliable tool calls and keeps the demo reproducible.

## Tools

The workflow uses these main tools:

- `cnn_document_classifier`: loads the trained PyTorch model and returns the predicted class, confidence score, class probabilities, and human-review flag.
- `document_extraction_tool`: extracts OCR text from the scanned image and detects simple fields such as dates, emails, amounts, and phone numbers.
- `local_summarizer_tool`: creates a short extractive summary from the OCR text.
- `human_approval_checkpoint`: asks the user to approve report generation.
- `structured_report_writer`: writes the final Markdown report.
- JSON logger: records each important action in `outputs/logs/agent_actions.jsonl`.

## Model

The document classifier is a PyTorch CNN based on ResNet18 transfer learning.

Supported classes:

```text
invoice
letter
memo
resume
scientific_report
```

Model files:

```text
models/document_cnn.pt
models/classes.json
```

The model receives RGB scanned document images resized to 224 x 224.

## Evaluation

Evaluation outputs are stored in:

```text
outputs/evaluation/
```

Included files:

```text
metrics.json
classification_report.txt
confusion_matrix.png
```

Current test results:

```text
test_accuracy: 0.8101
num_test_samples: 237
architecture: resnet18_transfer_learning
```

The model performs best on resumes and invoices. The most difficult class is `scientific_report`, mainly because some reports are visually similar to other text-heavy documents.

## Requirements

Python 3.10+ is recommended.

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

External tools:

- Tesseract OCR
- Ollama with Llama 3.1

Set Ollama environment variables before running with LLM mode:

```powershell
$env:OLLAMA_MODEL="ollama/llama3.1"
$env:OLLAMA_BASE_URL="http://localhost:11434"
```

If needed, add Tesseract to PATH:

```powershell
$env:Path += ";C:\Program Files\Tesseract-OCR"
```

## Run the Project

Normal mode with CrewAI and Ollama:

```powershell
python main.py analyze --document "data\real_raw\docs-sm\resume\00071736_00071737.jpg"
```

When prompted:

```text
Approve final report generation? [y/N]:
```

Type:

```text
y
```

Fast reproducible mode without LLM:

```powershell
python main.py analyze --document "data\real_raw\docs-sm\resume\00071736_00071737.jpg" --no-llm --auto-approve
```

View recent logs:

```powershell
python main.py logs --tail 30
```

Generated reports are written to:

```text
outputs/reports/
```

## Important Files

```text
main.py
src/sda/agents/crew.py
src/sda/tools/classifier_tool.py
src/sda/tools/extraction_tool.py
src/sda/tools/summarizer_tool.py
src/sda/tools/hitl_tool.py
src/sda/tools/report_tool.py
src/sda/utils/logging.py
scripts/train_cnn.py
scripts/evaluate_cnn.py
models/classes.json
outputs/evaluation/
outputs/reports/
outputs/logs/
```

## Limitations

The system currently supports scanned images only. OCR quality depends on the quality of the scan. Key-field extraction is rule-based and limited to simple patterns. The model is suitable for the project demo but is not intended as a production document understanding system.

