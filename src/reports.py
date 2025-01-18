"""
Reports Module
==============
This module provides utilities to generate financial reports from (extracted) data, including:
- Generating Excel reports with multiple sheets.
- Extracting financial metrics using LLMs.
- Generating financial summaries using LangChain with LLM.
"""
import os
import json
import csv
import pandas as pd
import warnings
import pathlib
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from typing import Any, List, Optional, Union

# Config loading
from config import (
    OPENAI_API_KEY,
    MODEL_NAME_REPORT,
    TEMPERATURE_REPORT,
    CHUNK_SIZE_REPORT,
    CHUNK_OVERLAP_REPORT,
    MODEL_TEXT_EMBED,
    REPORT_FOLDER,
)


class Reports:
    """
    Handles the generation of financial reports, including key metric extraction,
    tabular format generation, and financial data summarization.
    """

    def __init__(self):
        self.openai_api_key = OPENAI_API_KEY
        self.model_name = MODEL_NAME_REPORT
        self.model_temperature = TEMPERATURE_REPORT
        self.chunk_size = CHUNK_SIZE_REPORT
        self.chunk_overlap = CHUNK_OVERLAP_REPORT
        self.model_text_embed = MODEL_TEXT_EMBED
        self.report_folder = REPORT_FOLDER
        self.vector_store = None

    def extract_data_from_json(self, json_file_path: str) -> dict:
        """
        Reads and parses a JSON file.

        Parameters
        ----------
        json_file_path : str
            Path to the JSON file.

        Returns
        -------
        dict
            Parsed JSON data as a dictionary.
        """

        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
        return json_data

    def flatten_json_to_rows(self, json_obj: dict, parent_keys: Optional[List[str]] = None) -> List[dict]:
        """
        Recursively flattens a nested JSON object into a list of dictionaries, each representing a row.

        Parameters
        ----------
        json_obj : dict
            The nested JSON object to flatten.
        parent_keys : Optional[List[str]]
            A list of parent keys representing the hierarchy (default: None).

        Returns
        -------
        List[dict]
            A list of dictionaries, each representing a flattened row.
        """

        if parent_keys is None:
            parent_keys = []

        rows = []
        for key, value in json_obj.items():
            current_keys = parent_keys + [key]
            if isinstance(value, dict):
                rows.extend(self.flatten_json_to_rows(value, current_keys))
            else:
                row = {f"Level {i+1}": parent_keys[i]
                       for i in range(len(parent_keys))}
                row[f"Level {len(current_keys)}"] = key
                row["Value"] = value
                rows.append(row)
        return rows

    def save_to_excel_all(self, input_file_path: str, report_path: Optional[str] = None) -> None:
        """
        Saves the flattened JSON data into an Excel file with multiple sheets.

        Parameters
        ----------
        input_file_path : str
            Path to the JSON file to process.
        report_path : Optional[str], optional
            Directory to save the generated Excel report (default: self.report_folder).

        Returns
        -------
        None
        """

        if report_path is None:
            report_path = self.report_folder

        # Open and read the JSON file
        with open(input_file_path, 'r') as file:
            input_json_data = json.load(file)

        file_name = os.path.splitext(os.path.basename(input_file_path))[0]

        file_path = os.path.join(report_path, f"{file_name}.xlsx")

        # Flatten the JSON data
        all_rows = self.flatten_json_to_rows(json_obj=input_json_data)
        df_all = pd.DataFrame(all_rows)  # Create a DataFrame for all data

        # Create an Excel writer object
        print(f"Start to translate the data [{input_file_path}] to excel ...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Save all data to "Allinone" sheet
            df_all.to_excel(writer, index=False, sheet_name="Allinone")

            # Extract first-level keys and create separate sheets
            for first_level_key in input_json_data.keys():
                filtered_rows = [
                    {k: v for k, v in row.items() if row.get("Level 1") !=
                     first_level_key or k != "Level 1"}
                    for row in all_rows
                    if row.get("Level 1") == first_level_key
                ]
                df_filtered = pd.DataFrame(filtered_rows)
                df_filtered.to_excel(writer, index=False,
                                     sheet_name=first_level_key)

        print(f"Data has been written to {file_path}")

        return None

    def value_convert(self, value: Any) -> Union[int, float, None, str]:
        """
        Converts string values to numerical values where possible.

        Parameters
        ----------
        value : Any
            The value to convert.

        Returns
        -------
        Union[int, float, None, str]
            The converted value.
        """
        # Keep numerical types as is
        if isinstance(value, (int, float)):
            return value
        # Convert null-like values to None
        if value is None or value == "" or str(value).lower() == "null":
            return None
        try:
            # Try to convert to integer
            return int(value)
        except ValueError:
            try:
                # Try to convert to float
                return float(value)
            except ValueError:
                # Return the original value if it's not a number
                return value

    def convert_str_to_num(self, data: Any) -> Any:
        """
        Recursively converts string values in a dictionary or JSON object to numerical values.

        Parameters
        ----------
        data : Any
            The dictionary, list, or value to process.

        Returns
        -------
        Any
            The processed data with numerical values converted.
        """

        # Process the dictionary recursively
        if isinstance(data, dict):
            return {key: self.convert_str_to_num(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [convert_str_to_num(item) for item in data]
        else:
            return self.value_convert(data)

    def build_vector_store(
        self,
        input_data: Union[str, dict, list],
        separator: str = "\n",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Any:
        """
        Builds a vector store for input data.

        Parameters
        ----------
        input_data : Union[str, dict, list]
            The data to process (JSON, list, or plain text).
        separator : str, optional
            Text chunk separator (default: "\n").
        chunk_size : Optional[int], optional
            Maximum chunk size (default: self.chunk_size).
        chunk_overlap : Optional[int], optional
            Overlap size between chunks (default: self.chunk_overlap).

        Returns
        -------
        Any
            A Chroma vector store object.
        """

        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap

        print("Building vector store...")
        # Convert input_data to a single string
        if isinstance(input_data, dict) or isinstance(input_data, list):
            raw_text = json.dumps(input_data, ensure_ascii=False)
        else:
            # For simplicity: treat everything else as plain text
            raw_text = str(input_data)

        # Split text into chunks
        splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(raw_text)
        print(f"Created {len(chunks)} chunks from input data.")

        # Build vector store
        embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key, model=self.model_text_embed)
        self.vector_store = Chroma.from_texts(chunks, embeddings)
        print("Vector store built and stored.")

        return self.vector_store

    def financial_data_retriever(self, vector_store=None, user_query: Optional[str] = None, k: int = 5) -> str:
        """
        Retrieves relevant text about the company's financial health from a vector store.

        Parameters
        ----------
        user_query : str, optional (default None)
            Additional user request specifying what to focus on (e.g. 'cash flow changes').
        vector_store : Any, optional (default None)
            A vector store object (e.g. Chroma or FAISS). If None, uses self.vector_store.
        k : int, default 5
            Number of top relevant chunks to retrieve.

        Returns
        -------
        str
            A single string concatenating all retrieved text chunks.
            (If no docs are found or the vector store is missing, returns an empty string.)
        """

        # 1. Use self.vector_store if no store is passed
        store = vector_store if vector_store is not None else self.vector_store
        if not store:
            warnings.warn(
                "Warning: No vector store available. Returning empty string.")
            return ""

        # 2. Construct the retrieval prompt
        combined_prompt = (
            "Provide relevant text about the company's financial statements, including:\n"
            "- Key metrics such as revenue, net income, operating expenses, cash flow\n"
            "- Any trends or observations relevant to the company's financial health\n"
            "- Any details relevant for constructing a high-level financial summary\n"
        )

        # If user_query is provided, incorporate it
        if user_query:
            combined_prompt += f"\nSpecific user request: {user_query}"

        # 3. Retrieve top-k docs
        retriever = store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(combined_prompt)

        # 4. Concatenate doc text
        if not docs:
            print("No documents retrieved from the vector store.")
            return ""

        retrieved_text = "\n\n".join(doc.page_content for doc in docs)
        return retrieved_text

    def extract_financial_metrics(
        self,
        metrics_schemas: List[ResponseSchema],
        data_input: Any,
        csv_file: Optional[str] = None
    ) -> dict:
        """
        Extracts key financial metrics from input data and saves them to a CSV file if specified.

        Parameters
        ----------
        metrics_schemas : List[ResponseSchema]
            A list of `ResponseSchema` objects defining the metrics to extract.
            Example:
                [ResponseSchema(name="revenue", description="Total revenue recognized"),
                 ResponseSchema(name="net_income", description="Net income after taxes")]
        data_input : Any
            Input data containing financial information. Can be a file path, dictionary, list, or plain text.
        csv_file : Optional[str], optional
            Path to save the extracted metrics in a CSV file (default: None).

        Returns
        -------
        dict
            Extracted financial metrics as a dictionary.
        """

        # 1. Convert input to string
        data_str = None

        # Case A: data_input is a file path
        if isinstance(data_input, str) and os.path.isfile(data_input):
            file_path = pathlib.Path(data_input)
            try:
                if file_path.suffix.lower() == ".json":
                    # It's a JSON file, parse it into a dict or list
                    with open(file_path, "r", encoding="utf-8") as f:
                        parsed_json = json.load(f)
                    # Convert it back to a string for the LLM
                    data_str = json.dumps(parsed_json, ensure_ascii=False)
                else:
                    # It's a text file (or unknown extension)
                    with open(file_path, "r", encoding="utf-8") as f:
                        data_str = f.read()
            except Exception as e:
                warnings.warn(
                    f"Failed to read or parse file '{data_input}': {e}")
                return {}

        # Case B: data_input is a Python object (dict, list, etc.)
        elif isinstance(data_input, (dict, list)):
            data_str = json.dumps(data_input, ensure_ascii=False)

        # Case C: data_input is a string (not an existing file path)
        elif isinstance(data_input, str):
            data_str = data_input

        # Optional: If there are other cases (int, float, custom objects), convert them to string
        else:
            data_str = str(data_input)

        # If data_str is empty or whitespace, return empty
        if not data_str or not data_str.strip():
            warnings.warn(
                "Resulting string is empty or whitespace only. Returning empty metrics.")
            return {}

        # 2. Build structured parser from provided metrics_schemas
        output_parser = StructuredOutputParser.from_response_schemas(
            metrics_schemas)
        format_instructions = output_parser.get_format_instructions()

        # 3. Craft system & user messages
        system_message = SystemMessage(
            content=(
                "You are a financial data extraction assistant. "
                "You will receive arbitrary data (now converted to a string). "
                "Your task is to extract the requested metrics in valid JSON format, "
                "with no extra commentary."
            )
        )

        user_message_content = f"""
        Below is the input data:
    
        {data_str}
    
        Please parse it and return a JSON object with these fields:
        {format_instructions}
        """.strip()
        user_message = HumanMessage(content=user_message_content)

        # 4. Call LLM OPENAI_API_KEY,
        llm = ChatOpenAI(openai_api_key=self.openai_api_key,
                         model_name=self.model_name, temperature=self.model_temperature)
        response_content = llm.invoke([system_message, user_message]).content

        # 5. Parse the LLM response
        try:
            parsed_metrics = output_parser.parse(response_content)
        except Exception as e:
            warnings.warn(f"Failed to parse structured LLM output: {e}", )
            return {}
        parsed_metrics = self.convert_str_to_num(parsed_metrics)

        # 6. Save to CSV if csv_file is provided
        if csv_file is not None and parsed_metrics:
            try:
                with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
                    fieldnames = ["Financial Metrics", "Value"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    # Each key-value pair in parsed_metrics becomes one row
                    for metric_key, metric_value in parsed_metrics.items():
                        writer.writerow({
                            "Financial Metrics": metric_key,
                            "Value": metric_value
                        })
                print(
                    f"Metrics saved to {csv_file} with columns 'Financial Metrics' and 'Value'")
            except Exception as e:
                warnings.warn(f"Error writing to CSV file '{csv_file}': {e}")

        # 7. Return the dictionary
        return parsed_metrics

    def generate_financial_summary(
        self,
        key_metrics: Optional[dict] = None,
        financial_data: Optional[Union[dict, str]] = None,
        add_request: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generates a textual summary report highlighting the company's financial health.

        Parameters
        ----------
        key_metrics : Optional[dict]
            A dictionary of key financial metrics (e.g., revenue, net income).
        financial_data : Optional[Union[dict, str]]
            Additional financial information, either as a dictionary or plain text.
        add_request : Optional[str], optional
            Additional instructions or custom requests for the summary (default: None).
        output_file : Optional[str], optional
            Path to save the generated summary report (default: None).

        Returns
        -------
        str
            A summary report of the company's financial health.
        """

        # 1. Validate that there's at least some data to summarize
        metrics_is_valid = bool(key_metrics)
        financial_data_is_valid = bool(financial_data)

        if not metrics_is_valid and not financial_data_is_valid:
            raise ValueError(
                "Insufficient data: Both 'key_metrics' and 'financial_data' are missing or empty."
            )

        # 2. Convert 'key_metrics' and 'financial_data' to strings
        if key_metrics:
            metrics_str = str(key_metrics)
        else:
            metrics_str=""

        if financial_data:
            data_str = str(financial_data)
        else:
            data_str=""

        # 3. Build the system and user messages for the LLM
        system_message_content = (
            "You are a financial analysis assistant.\n"
            "Your goal is to generate a concise summary of the company's financial health. "
            "Only consider the provided metrics/data, and do not invent facts."
        )
        system_message = SystemMessage(content=system_message_content)

        # Base instructions
        user_instructions = """
        Based on the provided data, generate a summary report that highlights the financial health of the company. The report should include:
          - Key financial metrics
          - Any notable trends or observations
          - A short narrative summary in natural language
        """

        # If 'add_request' exists, append it
        if add_request:
            user_instructions += f"\nAdditional user request:\n{add_request}\n"

        user_message_content = f"""
            --- Financial Statement Metrics ---
            {metrics_str}
    
            --- Additional Financial Data ---
            {data_str}
    
            {user_instructions}
            """.strip()

        user_message = HumanMessage(content=user_message_content)

        # 4. Call LLM
        llm = ChatOpenAI(openai_api_key=self.openai_api_key,
                         model_name=self.model_name, temperature=self.model_temperature)

        # 5. Generate summary
        print("Requesting summary from LLM...")
        response = llm.invoke([system_message, user_message])
        summary_text = response.content.strip()

        # 6. Save summary to 'output_file' if provided
        if output_file:
            try:
                with open(output_file, mode="w", encoding="utf-8") as f:
                    f.write(summary_text)
                print(f"Summary is saved to {output_file} \n")
            except Exception as e:
                warnings.warn(
                    f"Error writing summary to file '{output_file}': {e}")

        # display(Markdown(summary_text))
        return summary_text
