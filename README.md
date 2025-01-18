# Financial Statement Analysis Using LLM

This project automated financial statement analysis by extracting key metrics from PDFs and generating comprehensive summaries using LLM. It provides an end-to-end solution for data extraction, processing, evaluation, and reporting.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Directory Structure](#directory-structure)
4. [Installation](#installation)
5. [Usage Instructions](#usage-instructions)
6. [Configuration](#configuration)
7. [Evaluation](#evaluation)
8. [Contributing](#contributing)
9. [License](#license)

---

## Overview

The project aims to automate the extraction of financial metrics from PDF financial statements and generate summaries highlighting key metrics such as:
- Revenue
- Net Income
- Operating Expenses
- Cash Flow

It employs **LangChain**, **OpenAI**, and other NLP techniques for accurate and efficient processing.

---

## Features

- **PDF Text Extraction**: Converts PDF documents into structured JSON data.
- **Data Cleaning**: Fixes syntax issues and prepares data for further analysis.
- **Metric Extraction**: Identifies and extracts financial metrics using LangChain.
- **Summary Generation**: Produces narrative financial summaries.
- **Evaluation**: Measures extraction accuracy and summary quality using BLEU, ROUGE, BERTScore, and LLM-based evaluations.

---

## Directory Structure

```
project/
│
├── src/
│   ├── data_ingestion.py       # Handles PDF text extraction.
│   ├── data_processing.py      # Processes extracted text into structured JSON.
│   ├── data_evaluation.py      # Evaluates data extraction and summaries.
│   ├── reports.py              # Generates Excel reports and summaries.
│   ├── config.py               # Configuration settings.
│
├── raw_texts/                  # Folder for raw extracted text from PDFs.
├── pdfs_data/                  # Folder for processed JSON data.
├── reports/                    # Folder for reports (e.g., Excel, summaries).
├── pdfs/                       # Input PDF files.
│
├── requirements.txt            # Dependencies list.
├── README.md                   # Project instructions and setup guide.
└── .env                        # Environment variables (optional).
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- A valid OpenAI API key

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the `.env` file (optional):
   Create a `.env` file in the project root and add your configuration:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

---

## Usage Instructions

### 1. Prepare Input PDFs
Place your financial statement PDFs in the `pdfs/` folder.

### 2. Extract Data from PDFs
Run the `data_ingestion.py` script to extract raw text:
```bash
python src/data_ingestion.py
```

### 3. Process Extracted Data
Convert the raw text into structured JSON using `data_processing.py`:
```bash
python src/data_processing.py
```

### 4. Generate Reports
Use the `reports.py` script to generate Excel reports and summaries:
```bash
python src/reports.py
```

### 5. Evaluate the Solution
Run `data_evaluation.py` to evaluate the data extraction and summary generation:
```bash
python src/data_evaluation.py
```

---

## Configuration

All configurable parameters are stored in `config.py` or can be set via environment variables. Key settings include:
- `OPENAI_API_KEY`: Your OpenAI API key.
- `PDF_FOLDER`: Path to the folder containing input PDFs.
- `PDF_RAW_TEXT_FOLDER`: Path to store raw extracted text.
- `MODEL_NAME_*`: Default OpenAI models for different tasks.
- `SYSTEM_PROMPT_PDF`: Prompt for extracting structured data from PDFs.

---

## Evaluation

The solution includes comprehensive evaluation methods:
1. **Data Extraction**:
   - Precision, Recall, and Extraction Score.
2. **Key Metric Accuracy**:
   - Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Exact Match.
3. **Summary Quality**:
   - BLEU, ROUGE, and BERTScore.
   - LLM-based evaluation for fluency, coherence, relevance, and conciseness.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a detailed explanation.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
