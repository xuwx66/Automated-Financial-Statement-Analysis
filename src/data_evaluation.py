"""
Data Evaluation Module
======================
This module provides methods to evaluate:
1) Data Extraction: Measures the precision, recall, and overall extraction score.
2) Key Metric Accuracy: Evaluates numeric errors (MAE, RMSE) and exact match metrics.
3) Summary Quality: Uses traditional NLP metrics (BLEU, ROUGE, BERTScore) and LLM-based evaluation.
"""

import re
import os
import json
import math
import warnings
from typing import Dict, List, Optional, Union
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate

from config import (
    OPENAI_API_KEY,
    MODEL_NAME_EVAL,
    TEMPERATURE_EVAL,
)


class DataEvaluation:
    """
    Provides methods to evaluate:
    1) Data Extraction: Precision, Recall, and overall extraction score.
    2) Key Metric Accuracy: Using MAE, RMSE, and exact match metrics.
    3) Summary Quality: NLP-based metrics (BLEU, ROUGE, BERTScore) and LLM-based evaluation.
    """

    def evaluate_data_extraction(self, extracted_data: str, ground_truth_data: str) -> Dict[str, float]:
        """
        Evaluate the quality of data extraction based on:
        - Partial Precision: The proportion of extracted tokens that match the ground truth.
        - Partial Recall: The proportion of ground truth tokens found in the extracted data.
        - Extraction Score: An average of partial precision and partial recall.

        Parameters
        ----------
        extracted_data : str
            Extracted text or path to a file containing extracted data.
        ground_truth_data : str
            Reference text confirmed to be correct.

        Returns
        -------
        dict
            A dictionary containing partial precision, partial recall, and extraction score.
        """

        if ground_truth_data:
            ground_truth_str = str(ground_truth_data)

        # Read extracted data if it is a file path
        if isinstance(extracted_data, str) and os.path.isfile(extracted_data):
            with open(extracted_data, 'r') as file:
                extracted_str = str(json.load(file))
        else:
            extracted_str = extracted_data

        # Tokenize ground truth and extracted data
        gt_tokens = set(re.findall(r"\w+", ground_truth_str.lower()))
        ex_tokens = set(re.findall(r"\w+", extracted_str.lower()))

        # Ensure ground truth tokens exist
        if not gt_tokens:
            warnings.warn("Ground truth data is empty or no tokens found. "
                          "Cannot compute meaningful partial metrics.")
            return {
                "partial_precision": 0.0,
                "partial_recall": 0.0,
                "extraction_score": 0.0
            }

        # Calculate true positives
        true_positives = gt_tokens.intersection(ex_tokens)
        partial_recall = len(true_positives) / \
            len(gt_tokens) if len(gt_tokens) else 0.0

        # Precision considers only the tokens in extracted data that match ground truth
        extracted_in_gt = ex_tokens.intersection(
            gt_tokens)  # same as true_positives
        partial_precision_denom = len(extracted_in_gt)
        partial_precision = (
            len(true_positives) / partial_precision_denom
            if partial_precision_denom > 0
            else (1.0 if len(gt_tokens) == 0 else 0.0)
        )

        extraction_score = 0.5 * partial_precision + 0.5 * partial_recall

        # print(
        #     f"Data Extraction (Partial) -> Precision: {partial_precision:.2f}, "
        #     f"Recall: {partial_recall:.2f}, Score: {extraction_score:.2f}"
        # )

        return {
            "partial_precision": partial_precision,
            "partial_recall": partial_recall,
            "extraction_score": extraction_score
        }

    def evaluate_key_metric_accuracy(self,
                                     extracted_metrics: Dict[str, Union[int, float]],
                                     ground_truth_metrics: Dict[str,
                                                                Union[int, float]]
                                     ) -> Dict[str, float]:
        """
        Evaluate the accuracy of key financial metrics extracted by the pipeline.

        Metrics:
        - Numeric Error Metrics: MAE, RMSE, naive accuracy_score
        - Exact Match Metrics: Precision, Recall, F1

        Parameters
        ----------
        extracted_metrics : Dict[str, Union[int, float]]
            Metrics extracted by the pipeline.
        ground_truth_metrics : Dict[str, Union[int, float]]
            Correct reference metrics.

        Returns
        -------
        dict
            A dictionary with numeric error metrics (MAE, RMSE) and exact match metrics (Precision, Recall, F1).
        """

        # ---------------------------
        #  Numeric Error Computation
        # ---------------------------
        absolute_errors = []
        squared_errors = []
        total_items = 0

        # Compare only keys that appear in ground_truth
        for key, gt_val in ground_truth_metrics.items():
            ex_val = extracted_metrics.get(key, None)
            if ex_val is None:
                # Missing in extraction, skip numeric error
                continue
            # Compare numeric difference
            try:
                diff = float(gt_val) - float(ex_val)
                absolute_errors.append(abs(diff))
                squared_errors.append(diff ** 2)
                total_items += 1
            except (ValueError, TypeError):
                warnings.warn(
                    f"Could not convert extracted metric {ex_val} to float for key '{key}'. Skipping.")

        # If we have no comparable numeric items
        if total_items == 0:
            mae = 0.0
            rmse = 0.0
            naive_accuracy = 0.0
        else:
            mae = sum(absolute_errors) / total_items
            mse = sum(squared_errors) / total_items
            rmse = math.sqrt(mse)

            # A naive approach to define an "accuracy_score": 1 - (rmse / average_of_ground_truth_values)
            gt_sum = 0.0
            gt_count = 0
            for val in ground_truth_metrics.values():
                try:
                    gt_sum += float(val)
                    gt_count += 1
                except:
                    pass
            if gt_count == 0:
                naive_accuracy = 0.0
            else:
                avg_gt = gt_sum / gt_count
                if avg_gt > 0:
                    naive_accuracy = max(0.0, 1.0 - (rmse / avg_gt))
                else:
                    naive_accuracy = 0.0

        # print(f"Numeric Error => MAE: {mae:.2f}, RMSE: {rmse:.2f}, naive_accuracy: {naive_accuracy:.2f}")

        # --------------------------------
        # Exact Match (Precision/Recall)
        # --------------------------------
        # A "true positive" means the key is in both dicts with EXACT same numeric value.
        # A "false negative" means the key is in ground_truth but is either missing or
        # has a different numeric value in extracted.
        tp = 0
        fn = 0
        fp = 0

        for key, gt_val in ground_truth_metrics.items():
            ex_val = extracted_metrics.get(key, None)
            if ex_val is None:
                fn += 1
                continue
            # Check if numeric value is exactly the same
            try:
                if abs(float(gt_val) - float(ex_val)) < 1e-12:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            except (ValueError, TypeError):
                fp += 1
                fn += 1

        exact_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        exact_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (exact_precision + exact_recall) > 0:
            exact_f1 = 2.0 * (exact_precision * exact_recall) / \
                (exact_precision + exact_recall)
        else:
            exact_f1 = 0.0

        # print(f"Exact Match => Precision: {exact_precision:.2f}, Recall: {exact_recall:.2f}, F1: {exact_f1:.2f}")

        return {
            "mae": mae,
            "rmse": rmse,
            "accuracy_score": naive_accuracy,
            "exact_match_precision": exact_precision,
            "exact_match_recall": exact_recall,
            "exact_match_f1": exact_f1
        }

    def evaluate_summary(self, generated_summary, ground_truth) -> Dict[str, float]:
        """
        Evaluate the quality of a generated summary using traditional NLP metrics (BLEU, ROUGE, BERTScore)
        and LLM-based evaluation for human-like scoring.

        Parameters
        ----------
        generated_summary : str
            Summary generated by the pipeline.
        ground_truth : str
            Reference summary to compare against.

        Returns
        -------
        Dict[str, Union[float, str]]
            A dictionary containing scores for BLEU, ROUGE, BERTScore, and an LLM-based assessment.
        """

        evaluation_results = {}

        # Traditional NLP Metrics

        # BLEU Score
        print("Start caculating BLEU Score...")
        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu(
            [ground_truth.split()],
            generated_summary.split(),
            smoothing_function=smoothing_function
        )
        evaluation_results["BLEU"] = bleu_score

        # ROUGE Score
        print("Start caculating ROUGE Score...")
        rouge = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = rouge.score(generated_summary, ground_truth)
        evaluation_results["ROUGE-1"] = rouge_scores['rouge1'].fmeasure
        evaluation_results["ROUGE-2"] = rouge_scores['rouge2'].fmeasure
        evaluation_results["ROUGE-L"] = rouge_scores['rougeL'].fmeasure

        # BERTScore
        print("Start caculating BERTScore...")
        P, R, F1 = bert_score([generated_summary], [ground_truth], lang="en")
        evaluation_results["BERTScore-Precision"] = P.mean().item()
        evaluation_results["BERTScore-Recall"] = R.mean().item()
        evaluation_results["BERTScore-F1"] = F1.mean().item()

        # LLM-Based Evaluation
        print("Start LLM-Based Evaluation...")
        system_message_content = (
            "You are an expert in financial analysis and language assessment. Evaluate the following generated summary of a financial statement against the reference summary. \n"
            "Assess the quality of the summary based on these criteria: \n\n"
            "1. Fluency: Is the language clear, grammatically correct, and professional?\n"
            "2. Coherence: Does the summary logically flow and connect relevant points effectively?\n"
            "3. Relevance: Does the summary accurately reflect the key financial information and metrics?\n"
            "4. Conciseness: Is the summary brief yet comprehensive, avoiding unnecessary details?\n\n"
            "Provide a score (out of 10) for each criterion and include a brief explanation for your ratings. Conclude with an overall evaluation of the summary's quality."
        )
        system_message = SystemMessage(content=system_message_content)

        prompt_template = PromptTemplate(
            template="""
            Reference Summary:
            {GROUND_TRUTH}
    
            Generated Summary:
            {GENERATED_SUMMARY}
            """,
            input_variables=["ground_truth", "generated_summary"]
        )

        evaluation_prompt = prompt_template.format(
            GROUND_TRUTH=ground_truth,
            GENERATED_SUMMARY=generated_summary
        )
        user_message = HumanMessage(content=evaluation_prompt)

        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                         model_name=MODEL_NAME_EVAL, temperature=TEMPERATURE_EVAL)
        response = llm.invoke([system_message, user_message])
        llm_evaluation = response.content.strip()

        evaluation_results["LLM_Evaluation"] = llm_evaluation

        return evaluation_results
