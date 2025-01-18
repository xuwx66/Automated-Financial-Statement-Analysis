"""
config.py
=========
Holds configuration variables and constants for the application.

This script centralizes configuration for:
1. General settings (folder paths, API keys).
2. Data extraction settings (models, tokens, prompts).
3. Summary and report generation settings.
4. Evaluation settings.

Environment variables are used for flexibility, with defaults provided for local development.
"""

import os

# ------------------------------------------------------------------------------
# General Settings
# ------------------------------------------------------------------------------
# Folder containing PDF files to process.
PDF_FOLDER = os.environ.get("PDF_FOLDER", "./pdfs")

# Folder for storing initially extracted text from PDFs.
PDF_RAW_TEXT_FOLDER = os.environ.get("PDF_RAW_TEXT_FOLDER", "./raw_texts")

# Folder for saving processed text data.
PDF_DATA_FOLDER = os.environ.get("PDF_TEXT_FOLDER", "./pdfs_data")

# Folder for saving reports (e.g., tables, summaries).
REPORT_FOLDER = os.environ.get("PDF_DATA_FOLDER", "./reports")

# API key for accessing OpenAI services.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ------------------------------------------------------------------------------
# Data Extraction Settings 
# ------------------------------------------------------------------------------
# Model for extracting text from PDF images.
MODEL_NAME_PDF = os.environ.get("MODEL_NAME_PDF", "gpt-4o")

# Maximum number of tokens for a single API call.
MODEL_TOKEN_PDF = os.environ.get("MODEL_TOKEN_PDF", 4096)

# Model for preprocessing extracted text from PDFs.
MODEL_TEXT_PREPRO = os.environ.get("MODEL_TEXT_PREPRO", "gpt-4o")

# System prompt for the model during data extraction.
SYSTEM_PROMPT_PDF = """
You are a highly accurate data extraction assistant tasked with parsing images of PDF-based financial statements. 
Your goal is to produce a well-structured JSON output containing all relevant information from each page, while following these guidelines:
Ensure that:
    1. Preserve Key Statements: Focus on these statements in particular:
       • Statement of Comprehensive Income
       • Statement of Financial Position
       • Statement of Changes in Equity
       • Statement of Cash Flows
    2. Remove Irrelevant Content: Exclude any headers, footers, page numbers, or boilerplate text that does not contain meaningful financial data.
    3. Combine Multi-Page Tables or Paragraphs: If a table or paragraph is split across pages, merge them seamlessly into a single section in your JSON output.
    4. Data Format:
      • Retain the original table structures and field/column names.
      • For numeric fields, convert the extracted strings into numerical values rather than leaving them as text.
      • If a “Notes” column has no entry, include it but assign a null (or None) value.
    5. Completeness: Do not omit any lines or data points from the source documents. Capture everything carefully, line by line.
    6. Output Requirements:
      • Return the final output as well-formed JSON.
      • Only include fields that appear in the original statements; do not add extra commentary or irrelevant keys.
Process the document meticulously, page by page and line by line, ensuring no data is missed.
"""

# Prompt for preprocessing text extraction from pdf
PROMPT_PREPRO = """
You are a JSON data cleaning assistant. You have been given a JSON-like string that might contain:
    • Missing or extra braces ({}) or brackets ([]),
    • Incorrect commas or other syntax issues,
    • Numbers stored as strings (e.g., "123" instead of 123).

Insert the JSON-like string here:
[

TEXT_INPUT

]

Your tasks are as follows:
    1. Preserve All Existing Data: Do not rename or remove any keys, values, or array elements.
    2. Check and Fix JSON Structure: Identify and correct issues such as missing braces, misplaced commas, or unbalanced brackets so that the final result is valid JSON.
    3. Convert Numeric Strings: If a value is clearly numeric (e.g., "123", "45.67"), convert it to the appropriate numerical type in JSON (e.g., 123, 45.67).
    4. Do Not Change Actual Data: Apart from the syntax repairs and type conversions for numeric fields, do not alter the content of any fields.
    5. Final Output:
      • Return only the corrected JSON.
      • Enclose it between triple quotes (``` at the start and ``` at the end).
      • Do not include any additional commentary or text beyond the JSON itself.  
Proceed carefully, ensuring all structural issues are fixed and numbers are converted from string types where appropriate, then provide the final JSON wrapped in triple backticks.
"""


# ------------------------------------------------------------------------------
# Summary and Reports setting
# ------------------------------------------------------------------------------
# Model for generating summaries and reports.
MODEL_NAME_REPORT = os.environ.get("MODEL_NAME_REPORT", "gpt-4o")

# Temperature setting for controlling creativity & randomness in LLM outputs.
TEMPERATURE_REPORT = float(os.environ.get("TEMPERATURE_REPORT", 0))

# Chunk size for splitting text during embedding creation.
CHUNK_SIZE_REPORT = int(os.environ.get("CHUNK_SIZE_REPORT", 3000))

# Overlap size between consecutive chunks to maintain context.
CHUNK_OVERLAP_REPORT = int(os.environ.get("CHUNK_OVERLAP_REPORT", 100))

# Model for creating text embeddings.
MODEL_TEXT_EMBED = os.environ.get("MODEL_TEXT_EMBED", "text-embedding-ada-002")


# ------------------------------------------------------------------------------
# Evaluation Setting
# ------------------------------------------------------------------------------

# Model for evaluating summaries and extractions.
MODEL_NAME_EVAL = os.environ.get("MODEL_NAME_EVAL", "gpt-4o")

# Temperature setting for controlling creativity & randomness during evaluation.
TEMPERATURE_EVAL = float(os.environ.get("TEMPERATURE_EVAL", 0))
