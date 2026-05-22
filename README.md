# Smart Document Analyst

## 1. Project Overview

Smart Document Analyst is a multi-agent AI system for analyzing scanned document images.

The system takes a scanned document image as input, then:

1. classifies the document using a trained PyTorch CNN model;
2. extracts text from the image using OCR;
3. detects basic key fields;
4. summarizes the extracted text;
5. asks for human approval before generating the final report;
6. writes a structured Markdown report;
7. logs every important agent/tool action in JSON format.

This project is designed for the **Integrated Project: Building Multi-Agent AI Systems**.

## 2. Supported Input Format

This project supports scanned document images only.

Supported formats:

```text
.png
.jpg
.jpeg
.tif
.tiff
.bmp
```

PDF, TXT, DOCX, and other document formats are not supported in this version.

This choice is intentional because the trained CNN model classifies documents visually based on their scanned layout and appearance.

## 3. Main Features

- Multi-agent architecture with CrewAI
- Ollama LLM backend
- PyTorch CNN document classifier
- OCR text extraction with Tesseract
- Local summarization tool
- Human-in-the-loop checkpoint
- Structured Markdown report generation
- JSON action logging with timestamps
- Reproducible fallback mode without LLM

## 4. Project Architecture

The system uses these main agents:

### Orchestrator Manager Agent

Coordinates the full workflow and ensures the process runs in the correct order.

### Document Classification Agent

Responsible for the classification step. The actual classification is executed by the audited Python workflow using the trained CNN tool.

### Extraction and Summarization Agent

Responsible for the OCR extraction and summarization step. The actual OCR and summarization tools are executed by the audited Python workflow.

### Report Generation Agent

Checks that classification, extraction, summarization, and human approval are required before final report generation.

The final audited workflow executes the tools in this order:

```text
Scanned image
    ↓
CNN classifier tool
    ↓
OCR extraction tool
    ↓
Summarizer tool
    ↓
Human approval checkpoint
    ↓
Report writer tool
    ↓
Final Markdown report
```

## 5. Model Information

The document classifier is a PyTorch CNN model based on ResNet18 transfer learning.

Model file:

```text
models/document_cnn.pt
```

Classes file:

```text
models/classes.json
```

Supported classes:

```text
invoice
letter
memo
resume
scientific_report
```

The current trained model uses:

```text
architecture: resnet18_transfer_learning
```

Evaluation files are available in:

```text
outputs/evaluation/
```

Important evaluation files:

```text
outputs/evaluation/metrics.json
outputs/evaluation/classification_report.txt
outputs/evaluation/confusion_matrix.png
```

Current test performance:

```text
test_accuracy: 0.8101
num_test_samples: 237
```

## 6. Requirements

Recommended Python version:

```text
Python 3.10+
```

Python packages are listed in `requirements.txt`.

The project also requires two external tools:

1. **Ollama** for the LLM backend
2. **Tesseract OCR** for image text extraction

## 7. Install Python Dependencies

From the project root folder, run:

```powershell
pip install -r requirements.txt
```

The `requirements.txt` should contain at least:

```text
crewai>=0.80.0
crewai-tools>=0.14.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24
pillow>=10.0
pytesseract>=0.3.10
scikit-learn>=1.3
matplotlib>=3.7
pydantic>=2.0
python-dotenv>=1.0
rich>=13.0
pytest>=7.0
```

## 8. Install and Configure Tesseract OCR

Tesseract is required for OCR extraction from scanned document images.

### Windows Installation

Install Tesseract OCR.

Recommended installation path:

```text
C:\Program Files\Tesseract-OCR
```

Make sure this file exists:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Add Tesseract to PATH

In PowerShell, you can temporarily add it like this:

```powershell
$env:Path += ";C:\Program Files\Tesseract-OCR"
```

Test it:

```powershell
tesseract --version
```

Test Python access:

```powershell
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

If both commands show a version, Tesseract is correctly configured.

## 9. Install and Configure Ollama

Ollama is used as the local LLM backend for CrewAI.

After installing Ollama, open PowerShell and test:

```powershell
ollama --version
```

Download the Llama 3.1 model:

```powershell
ollama pull llama3.1
```

Check installed models:

```powershell
ollama list
```

Before running the project with LLM mode, set these environment variables:

```powershell
$env:OLLAMA_MODEL="ollama/llama3.1"
$env:OLLAMA_BASE_URL="http://localhost:11434"
```

## 10. How to Run the Project

Use this command from the project root folder:

```powershell
python main.py analyze --document "data\real_raw\docs-sm\resume\00071736_00071737.jpg"
```

The system will:

1. start CrewAI with Ollama;
2. run the planning/validation agents;
3. execute the audited Python workflow;
4. classify the image using the CNN;
5. extract OCR text;
6. summarize the extracted text;
7. ask for human approval;
8. generate the final Markdown report.

When prompted:

```text
Approve final report generation? [y/N]:
```

Type:

```text
y
```

The report will be written to:

```text
outputs/reports/
```

Example:

```text
outputs/reports/analysis_00071736_00071737_20260517_051408.md
```

## 11. Fast Demo Mode

For quick testing without manual approval:

```powershell
python main.py analyze --document "data\real_raw\docs-sm\resume\00071736_00071737.jpg" --auto-approve
```

This still uses the LLM unless `--no-llm` is added.

## 12. Reproducible No-LLM Mode

For debugging or testing without Ollama:

```powershell
python main.py analyze --document "data\real_raw\docs-sm\resume\00071736_00071737.jpg" --no-llm --auto-approve
```

This mode skips the LLM planning layer and directly executes the audited tool-grounded workflow.

Use this mode only for testing/debugging. For the final demo, use the normal mode with Ollama.

## 13. View Logs

Every important action is logged in JSON format.

Log file:

```text
outputs/logs/agent_actions.jsonl
```

Show the last 30 logs:

```powershell
python main.py logs --tail 30
```

Expected log actions include:

```text
build_crewai_agents
start_analysis
crewai_kickoff
classify_document
cnn_classification_tool
extract_document_text
extract_content
summarize_document
human_approval_checkpoint
write_final_report
finish_analysis
```

## 14. Expected Successful Output

A successful run should end with something like:

```text
HUMAN-IN-THE-LOOP CHECKPOINT
Predicted class: resume (confidence=1.000)
Summary preview:
...
Approve final report generation? [y/N]: y

Final report written to: outputs\reports\analysis_xxx.md
```

The generated report should contain:

- document path;
- generation date;
- human approval status;
- predicted class;
- confidence;
- class probabilities;
- extracted content statistics;
- detected key fields;
- summary;
- key bullets;
- robustness notes;
- extracted text preview.

## 15. Troubleshooting

### Problem: `tesseract is not installed or it's not in your PATH`

Fix:

```powershell
$env:Path += ";C:\Program Files\Tesseract-OCR"
tesseract --version
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

Then rerun the project.

### Problem: `ollama is not recognized`

Ollama is not in PATH or not installed correctly. Try closing and reopening PowerShell.

Then run:

```powershell
ollama --version
```

If it still does not work, find `ollama.exe` and add its folder to PATH.

Example:

```powershell
$env:Path += ";C:\Users\YOUR_USERNAME\AppData\Local\Programs\Ollama"
```

Then test again:

```powershell
ollama --version
```

### Problem: CrewAI tries to use OpenAI instead of Ollama

Make sure these variables are set:

```powershell
$env:OLLAMA_MODEL="ollama/llama3.1"
$env:OLLAMA_BASE_URL="http://localhost:11434"
```

Then rerun:

```powershell
python main.py analyze --document "data\real_raw\docs-sm\resume\00071736_00071737.jpg"
```

### Problem: Unsupported document format

The system only supports scanned document images.

Use one of:

```text
.png
.jpg
.jpeg
.tif
.tiff
.bmp
```

Do not use PDF, TXT, DOCX, or MD files.

### Problem: `Document not found`

Check that the path exists.

Example command to find images:

```powershell
Get-ChildItem -Recurse -Include *.png,*.jpg,*.jpeg,*.tif,*.tiff,*.bmp
```

Then copy the correct path into the command.

## 16. Important Project Files

```text
main.py
```

Command-line entry point.

```text
src/sda/agents/crew.py
```

CrewAI agents and audited workflow orchestration.

```text
src/sda/tools/classifier_tool.py
```

CNN document classifier tool.

```text
src/sda/tools/extraction_tool.py
```

OCR text extraction and key-field detection tool.

```text
src/sda/tools/summarizer_tool.py
```

Local summarization tool.

```text
src/sda/tools/hitl_tool.py
```

Human-in-the-loop approval checkpoint.

```text
src/sda/tools/report_tool.py
```

Markdown report writer.

```text
src/sda/utils/logging.py
```

JSON action logging.

```text
models/document_cnn.pt
```

Trained PyTorch model.

```text
models/classes.json
```

Document class labels.

```text
outputs/evaluation/
```

Model evaluation outputs.

```text
outputs/reports/
```

Generated analysis reports.

```text
outputs/logs/
```

JSON logs.

## 17. Final Demo Command


```powershell
$env:OLLAMA_MODEL="ollama/llama3.1"
$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:Path += ";C:\Program Files\Tesseract-OCR"
```


```powershell
python main.py analyze --document "data\real_raw\docs-sm\resume\00071736_00071737.jpg"
```

Approve the report when asked:

```text
y
```

logs:

```powershell
python main.py logs --tail 30
```

And open the generated report in:

```text
outputs/reports/
```
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
