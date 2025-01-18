"""
Data Processing Module
======================
This module handles text preprocessing, data extraction and transformation for further analysis.
Features include:
- Preprocessing text with LLMs using LangChain and OpenAI.
- Extracting structured-formatted data from raw text content.
- Processing text files in single or batch mode to further clean/process the extracted data.
"""

import os
import re
import json
import warnings
from langchain import OpenAI
from langchain.chains import LLMChain
from typing import Dict, Optional, List

# Config loading
from config import (
    OPENAI_API_KEY,
    MODEL_TEXT_PREPRO,
    PROMPT_PREPRO,
    PDF_RAW_TEXT_FOLDER,
    PDF_DATA_FOLDER,
)


class DataProcessing:
    """
    Handles text preprocessing, JSON extraction, and processing extracted text data.
    """

    def __init__(self):
        self.openai_api_key = OPENAI_API_KEY
        self.model_name = MODEL_TEXT_PREPRO
        self.pdf_data_folder = PDF_DATA_FOLDER
        self.raw_text_folder = PDF_RAW_TEXT_FOLDER

    def data_preprocess(self, str_data: str, prompt: str = PROMPT_PREPRO) -> str:
        """
        Preprocess text using an LLM through LangChain.

        Parameters
        ----------
        str_data : str
            The input text data to preprocess.
        prompt : str, optional
            Prompt template guiding the LLM behavior (default: PROMPT_PREPRO).

        Returns
        -------
        str
            Processed text response from the LLM.
        """
        llm = OpenAI(openai_api_key=self.openai_api_key,
                     model_name=self.model_name)
        prompt = prompt.replace("TEXT_INPUT", str_data)
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run()
        # print(response)
        return response

    def extract_json(self, str_content: str, llm_flag: int = 0) -> Dict:
        """
        Extract JSON data from a string. Optionally preprocess with an LLM.

        Parameters
        ----------
        str_content : str
            The string containing JSON content.
        llm_flag : int, optional
            Whether to preprocess the extracted JSON with an LLM (default: 0).

        Returns
        -------
        Dict
            Extracted JSON content as a dictionary.
        """
        # Locate JSON block in the string
        # .find() will return the BEGINNING of the match
        start_pos = str_content.find("```json\n") + len("```json\n")
        end_pos = str_content.find("\n```")
        json_chunk = str_content[start_pos:end_pos]

        if llm_flag == 1:
            # Preprocess extracted json text using Langchain+OpenAI
            json_chunk = self.data_preprocess(
                str_data=json_chunk, prompt=PROMPT_PREPRO)
            start_pos = json_chunk.find("```") + len("```")
            end_pos = json_chunk.find("```")
            json_chunk = json_chunk[start_pos:end_pos]

        # Parse the JSON string
        data_dict = json.loads(json_chunk)

        return data_dict

    def data_processing(self, input_text_folder: str, output_folder: Optional[str]) -> None:
        """
        Process raw text files in a folder to generate consolidated JSON data.

        Parameters
        ----------
        input_text_folder : str
            Folder containing raw text files.
        output_folder : Optional[str]
            Folder to store the processed JSON file (default: self.pdf_data_folder).

        Returns
        -------
        None
        """
        if output_folder is None:
            output_folder = self.pdf_data_folder
        text_folder_name = os.path.basename(
            os.path.normpath(input_text_folder))
        # List all text files in the folder
        text_files = [f for f in os.listdir(
            input_text_folder) if f.endswith(".txt")]
        text_files.sort(key=lambda x: int(re.findall(r'\d+', x)
                        [0]) if re.findall(r'\d+', x) else 0)

        pdf_json_data = {}
        print(f"Start to process data from {input_text_folder}")
        # Process each text file
        for file_name in text_files:
            file_path = os.path.join(input_text_folder, file_name)
            with open(file_path, "r") as file:
                text_data = file.read()

            # Convert extracted text to JSON
            content_json = self.extract_json(text_data)
            pdf_json_data.update(content_json)

        # Save consolidated JSON data
        output_file_path = os.path.join(
            output_folder, f"{text_folder_name}.json")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as f:
            json.dump(pdf_json_data, f, ensure_ascii=False)
        print(f"Complete processing data from {input_text_folder}")

        return None

    def data_processing_list(self, input_text_folder_list: List[str], output_folder: Optional[str]) -> None:
        """
        Process text files from multiple folders.

        Parameters
        ----------
        input_text_folder_list : List[str]
            List of folder paths containing text files.
        output_folder : Optional[str]
            Folder to store processed JSON files (default: self.pdf_data_folder).

        Returns
        -------
        None
        """
        if output_folder is None:
            output_folder = self.pdf_data_folder
        if len(input_text_folder_list) < 1:
            warnings.warn("There is not file in the provided list!")
            return None
        else:
            for folder in input_text_folder_list:
                self.data_processing(
                    input_text_folder=folder, output_folder=output_folder)
            return None

    def data_processing_all(self, input_text_parent_folder: Optional[str], output_folder: Optional[str]) -> None:
        """
        Process all subfolders within a parent folder.

        Parameters
        ----------
        input_text_parent_folder : Optional[str]
            Parent folder containing subfolders with text files (default: self.raw_text_folder).
        output_folder : Optional[str]
            Folder to store processed JSON files (default: self.pdf_data_folder).

        Returns
        -------
        None
        """
        if input_text_parent_folder is None:
            input_text_parent_folder = self.raw_text_folder
        if output_folder is None:
            output_folder = self.pdf_data_folder
        # List all subfolders
        subfolders = [os.path.join(input_text_parent_folder, name)
                      for name in os.listdir(input_text_parent_folder)
                      if os.path.isdir(os.path.join(input_text_parent_folder, name))]
        if len(subfolders) < 1:
            warnings.warn("There is not file in the provided Folder!")
            return None
        else:
            for folder in subfolders:
                self.data_processing(
                    input_text_folder=folder, output_folder=output_folder)
            return None
