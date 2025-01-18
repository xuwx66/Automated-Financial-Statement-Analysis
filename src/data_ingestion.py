"""
Data Ingestion Module
=====================
This module provides utilities to process PDF files for text extraction using LLM-based methods.
Features include:
- Conversion of PDF pages for extraction.
- Extracting text from PDFs using LLM.
- Batch processing of PDFs, supporting individual files or directories.
"""

import io
import os
import openai
import shutil
import base64
import warnings
import pandas as pd
from openai import OpenAI
from PIL.Image import Image
from typing import Dict, Optional, List
from pdf2image import convert_from_path

# Config loading
from config import (
    OPENAI_API_KEY,
    MODEL_NAME_PDF,
    MODEL_TOKEN_PDF,
    SYSTEM_PROMPT_PDF,
    PDF_FOLDER,
    PDF_RAW_TEXT_FOLDER,
)


class DataIngestion:
    """
    Handles PDF processing and text extraction using LLMs.
    """

    def __init__(self):
        self.openai_api_key = OPENAI_API_KEY
        self.model_name = MODEL_NAME_PDF
        self.model_token = MODEL_TOKEN_PDF
        self.pdf_folder = PDF_FOLDER
        self.raw_text_folder = PDF_RAW_TEXT_FOLDER

    # Convert pdf pages to images
    def convert_doc_to_images(self, path: str) -> List[Image]:
        """
        Convert a PDF document into a list of image objects.

        Parameters
        ----------
        path : str
            Path to the PDF file.

        Returns
        -------
        list
            A list of PIL Image objects, each representing a page of the PDF.
        """

        return convert_from_path(path)

    def get_img_uri(self, img: Image) -> str:
        """
        Encode a PIL Image object as a Base64 data URI.

        Parameters
        ----------
        img : PIL.Image.Image
            The image to encode.

        Returns
        -------
        str
            The Base64 encoded string in data URI format.
        """

        png_buffer = io.BytesIO()
        img.save(png_buffer, format="PNG")
        png_buffer.seek(0)

        base64_png = base64.b64encode(png_buffer.read()).decode('utf-8')

        data_uri = f"data:image/png;base64,{base64_png}"
        return data_uri

    def analyze_image(self, imag_info: List[dict], system_prompt: str = SYSTEM_PROMPT_PDF) -> str:
        """
        Use LLM to extract text from image information.

        Parameters
        ----------
        imag_info : List[dict]
            List of image data encoded as Base64 URIs.
        system_prompt : str, optional
            System prompt to guide the LLM's behavior.

        Returns
        -------
        str
            Extracted text from the LLM's response.
        """
        # Initializing OpenAI client
        openai.api_key = self.openai_api_key
        client = OpenAI()

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": imag_info
                },
            ],
            max_tokens=self.model_token,
            temperature=0,
            top_p=0.1
        )
        return response.choices[0].message.content

    def extract_text_from_pdf(
        self,
        pdf_path: str,
        raw_text_folder: Optional[str] = None,
        k: int = 4,
        system_prompt: str = SYSTEM_PROMPT_PDF,
    ) -> None:
        """
        Extract text from a single PDF file.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.
        raw_text_folder : Optional[str], optional
            Folder to store extracted raw text files (default: self.raw_text_folder).
        k : int, optional
            Number of pages to process per batch (default: 4).
        system_prompt : str, optional
            System prompt for the LLM (default: SYSTEM_PROMPT_PDF).

        Returns
        -------
        None
        """

        if raw_text_folder is None:
            raw_text_folder = self.raw_text_folder

        file_name = os.path.splitext(os.path.basename(pdf_path))[0]
        images = self.convert_doc_to_images(pdf_path)
        image_cnt = len(images)

        # Create a folder for storing output
        folder_path = os.path.join(raw_text_folder, file_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)

        # Process images in batches
        for i in range(0, image_cnt, k):
            imag_info = []
            batch = images[i: i + k]
            for img in batch:
                img_url = self.get_img_uri(img)
                imag_info.append({
                    "type": "image_url",
                    "image_url": {"url": img_url}
                })

            print(
                f"Processing [{pdf_path}]; Pages: {i+1} to {min(i+k,image_cnt)}")

            # Extract data using LLM
            extract_content = self.analyze_image(
                imag_info, system_prompt=SYSTEM_PROMPT_PDF)

            # Save the extracted data to a text file
            text_path = os.path.join(
                folder_path, f"pages_{i+1}_to_{min(i+k,image_cnt)}.txt")
            with open(text_path, "w") as file:
                file.write(extract_content)
        return None

    def extract_text_from_pdf_list(
        self,
        pdf_paths: List[str],
        raw_text_folder: Optional[str] = None,
        k: int = 4,
        system_prompt: str = SYSTEM_PROMPT_PDF,
    ) -> None:
        """
        Extract text from a list of PDF files.

        Parameters
        ----------
        pdf_paths : List[str]
            List of paths to PDF files.
        raw_text_folder : Optional[str], optional
            Folder to store extracted raw text files (default: self.raw_text_folder).
        k : int, optional
            Number of pages to process per batch (default: 4).
        system_prompt : str, optional
            System prompt for the LLM (default: SYSTEM_PROMPT_PDF).

        Returns
        -------
        None
        """

        if raw_text_folder is None:
            raw_text_folder = self.raw_text_folder

        # List all PDF files in provided list
        pdf_paths = [file for file in file_paths if file.endswith(".pdf")]
        if len(pdf_paths) < 1:
            warnings.warn("There is not pdf in the provided list!")
            return None
        else:
            for pdf_path in pdf_paths:
                print(f"Start to extract data from {pdf_path}")
                self.extract_text_from_pdf(
                    pdf_path=pdf_path, raw_text_folder=raw_text_folder, k=k, system_prompt=system_prompt)
            return None

    # extract text from all pdfs in a folder
    def extract_text_from_pdf_all(
        self,
        pdf_folder_path: Optional[str] = None,
        raw_text_folder: Optional[str] = None,
        k: int = 4,
        system_prompt: str = SYSTEM_PROMPT_PDF,
    ) -> None:
        """
        Extract text from all PDFs in a specified folder.

        Parameters
        ----------
        pdf_folder_path : Optional[str], optional
            Path to the folder containing PDF files (default: self.pdf_folder).
        raw_text_folder : Optional[str], optional
            Folder to store extracted raw text files (default: self.raw_text_folder).
        k : int, optional
            Number of pages to process per batch (default: 4).
        system_prompt : str, optional
            System prompt for the LLM (default: SYSTEM_PROMPT_PDF).

        Returns
        -------
        None
        """
        if pdf_folder_path is None:
            pdf_folder_path = self.pdf_folder
        if raw_text_folder is None:
            raw_text_folder = self.raw_text_folder

        # Get all PDF file paths in the folder
        pdf_paths = [
            os.path.join(pdf_folder_path, file)
            for file in os.listdir(pdf_folder_path)
            if file.endswith(".pdf")
        ]

        self.extract_text_from_pdf_list(
            pdf_paths=pdf_paths, raw_text_folder=raw_text_folder, k=k, system_prompt=system_prompt)

        return None