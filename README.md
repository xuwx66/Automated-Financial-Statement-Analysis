# Financial Statement Analysis Using LLM

This project automated financial statement analysis by extracting key metrics from PDFs and generating comprehensive summaries using LLM. It provides an end-to-end solution for data extraction, processing, evaluation, and reporting.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Directory Structure](#directory-structure)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Evaluation](#evaluation)
7. [Usage Instructions](#usage-instructions)
8. [Contributing](#contributing)
9. [License](#license)

---

## Overview

This project aims to build an **end-to-end pipeline** for **financial data extraction**, **key metric retrieval**, **summary generation**, and **evaluation**. Specifically, it aims to:
- **Extract** structured information from raw or partially structured financial statements (PDFs).
- **Identify and convert** core financial metrics (e.g., revenue, net income) into numeric values, ensuring accuracy.
- **Handle large or lengthy financial statements** by using a **retriever mechanism** to extract the most relevant information for downstream tasks.
- **Generate** a concise **narrative summary** describing the organization’s financial health, supported by relevant data.
- **Evaluate** the effectiveness of each step (data extraction, metric accuracy, and summary quality) using a set of performance metrics (e.g., partial precision/recall, MAE, RMSE, exact-match F1).

It employs **LangChain**, **OpenAI**, and other NLP techniques for accurate and efficient processing.

---

## Features

- **PDF Text Extraction**: Converts PDF documents into structured data.
- **Data Cleaning**: Fixes syntax issues and prepares data for further analysis.
- **Metric Extraction**: Identifies and extracts financial metrics using LLM.
- **Summary Generation**: Produces narrative financial summaries.
- **Evaluation**: Measures extraction accuracy and summary quality using BLEU, ROUGE, BERTScore, and LLM-based evaluations.

---

## Directory Structure

```
project/
│
├── src/
│   ├── data_ingestion.py       # Handles PDF text extraction.
│   ├── data_processing.py      # Processes extracted text into structured data.
│   ├── data_evaluation.py      # Evaluates data extraction and summaries.
│   ├── reports.py              # Extract financial metrics and Generates reports and summaries.
│   ├── config.py               # Configuration settings.
│
├── raw_texts/                  # Folder for raw extracted text from PDFs.
├── pdfs_data/                  # Folder for processed data.
├── reports/                    # Folder for reports (e.g., Excel, summaries).
├── pdfs/                       # Input PDF files.
│
├── requirements.txt            # Dependencies list.
├── README.md                   # Project instructions and setup guide.
├── Documentation.md            # A concise report describing the approach, challenges faced, and solutions implemented
├── allinone.ipynb              # Jupyter Notebook including code, usecase and examples
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
   git clone https://github.com/xuwx66/Financial-Statement-Analysis.git
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

## Usage Instructions

### 1. Prepare Input PDFs
Place your financial statement PDFs in the `pdfs/` folder.

### 2. Extract Data from PDFs
Run `DataIngestion` from `data_ingestion.py` script to extract raw text:

#### Extract Text from a Single PDF File
```python
from data_ingestion import DataIngestion
DI = DataIngestion()
pdf_path = "./pdfs/your_file.pdf"
DI.extract_text_from_pdf(pdf_path)
```

#### Extract Text from PDFs in a List
```python
DI = DataIngestion()
pdf_path_list = ["./pdfs/your_file1.pdf", "./pdfs/your_file2.pdf"]
DI.extract_text_from_pdf_list(pdf_path_list)
```

#### Extract Text from All PDFs in a Folder
```python
DI = DataIngestion()
DI.extract_text_from_pdf_folder(pdf_folder_path="./pdfs")
DI.extract_text_from_pdf_folder()
```

### 3. Process Extracted Data
Convert the raw text into structured output using `DataProcessing` from `data_processing.py`:

#### Process Extracted Text from a Single Folder
```python
from data_processing import DataProcessing
DP = DataProcessing()
input_text_folder = "./raw_texts/your_file"
DP.data_processing(input_text_folder)
```

#### Process Extracted Text from a List of Folders
```python
DP = DataProcessing()
input_text_folder_list = ["./raw_texts/your_file1", "./raw_texts/your_file2"]
DP.data_processing_list(input_text_folder_list)
```

#### Process All Extracted Text
```python
DP = DataProcessing()
DP.data_processing_all()
```

### 4. Generate Reports
Use the `Reports` from `reports.py` script to extract metrics, generate reports and summaries:

#### Extracted all financial metrics in a tabular format (excel)
```python
from reports import Reports
input_file = "./pdfs_data/your_file.json"
RP = Reports()
RP.save_to_excel_all(input_file_path=input_file, report_path="./reports")
```

#### Extract financial metrics into CSV
```python
from langchain.output_parsers import ResponseSchema
from reports import Reports
input_file = "./pdfs_data/your_file.json"
metrics_schemas = [
    ResponseSchema(name="Revenue Last Year", description="Total revenue recognized in Last Year"),
    ResponseSchema(name="Revenue Previous Year", description="Total revenue recognized in Previous Year"),
    ResponseSchema(name="Net Income Last Year", description="Net income after taxes in Last Year"),
    ResponseSchema(name="Net Income Previous Year", description="Net income after taxes in Previous Year")
]
output_file = "./reports/key_metrics.csv"
RP = Reports()
key_metrics = RP.extract_financial_metrics(
   metrics_schemas=metrics_schemas,
   data_input=input_file,
   csv_file=output_file
)
```

#### Retrieve relevant Info/Data from `Large statements` using RAG
```python
input_file="./pdfs_data/large_statements.json"
specific_request="add if you have..."
RP=Reports()
financial_statements=RP.extract_data_from_json(input_file)
RP.build_vector_store(input_data=financial_statements)
relevant_data=RP.financial_data_retriever(user_query=specific_request)
```

#### Generate summary report for Statement of comprehensive income
```python
input_file="./pdfs_data/your_file.json"
output_summary_file="./reports/financial_summary_statement_comprehensive_income.txt"
RP=Reports()
financial_statements=RP.extract_data_from_json(input_file)
financial_summary=RP.generate_financial_summary(
    key_metrics=None,
    financial_data=financial_statements,
    add_request="Generate summary report based on Statement of comprehensive income",
    output_file=output_summary_file
)
```

#### Generate summary report based on the provided metric
```python
input_file="./pdfs_data/your_file.json"
output_summary_file="./reports/financial_summary.txt"
RP=Reports()
financial_statements=RP.extract_data_from_json(input_file)
financial_summary=RP.generate_financial_summary(
    key_metrics=key_metrics, 
    financial_data=None,
    add_request=None,
    output_file=output_summary_file
)
```

### 5. Evaluate the Solution
Run `DataEvaluation` from `data_evaluation.py` to evaluate the data extraction and summary generation:

#### Data extraction Evaluation
```python
extracted_data_file="./pdfs_data/your_file.json"
ground_truth_data={
    "Revenue": {"Last Year": 2222, "Previous Year": 1111}, 
    "Interest": {"Last Year": 3333, "Previous Year": 2222}, 
    "Rental income": {"Last Year": 4444, "Previous Year": 3333}, 
...
}
DE=DataEvaluation()
eval_data_extraction=DE.evaluate_data_extraction(
   extracted_data=extracted_data_file,
   ground_truth_data=ground_truth_data
)
```

#### Extracted metric Evaluation
```python
extracted_metrics={
    'Revenue Last Year': 4444,
    'Revenue Previous Year': 3333,
......
}

ground_truth_metrics={
    'Revenue Last Year': 4444,
    'Revenue Previous Year': 3332,
......
}
DE=DataEvaluation()
metric_accuracy=DE.evaluate_key_metric_accuracy(
   extracted_metrics=extracted_metrics,
   ground_truth_metrics=ground_truth_metrics
)
```

#### Summary result Evaluation
```python
generated_summary = """
The company has shown a slight increase in revenue but a notable decrease in net income due to rising 
operating expenses. .....                                                                                               
"""
ground_truth = """
The company experienced a modest increase in revenue, but a notable decline in net income, primarily driven by 
rising operating expenses. ....                                                                                             
"""
DE=DataEvaluation()
summary_accuracy=DE.evaluate_summary(
   generated_summary=generated_summary,
   ground_truth=ground_truth
)
```

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
