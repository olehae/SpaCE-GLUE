# Adapted from https://github.com/jiayuww/SpatialEval/blob/main/evals/evaluation.py
import json
import re
from typing import Optional, Tuple, List, Dict


def extract_answer_from_text_spatialmap(text: str) -> Optional[str]:
    """Extracts answers from spatial map text."""

    dirs = ["southeast", "northeast", "northwest", "southwest"]
    dir_pattern = rf"\b({'|'.join(dirs)})\b"

    if text is None:
        return None

    direction_match = re.search(
        r"\b[A-D]\.\s*(" + "|".join(dirs) + r")\b", text, re.IGNORECASE
    )
    if direction_match:
        return direction_match.group(1).lower()

    match = re.search(dir_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def extract_answer_from_text_mazenav(text: str) -> Optional[str]:
    """Extracts answers from maze navigation text."""
    number_mapping = {
        "zero": 0,
        "no": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }

    # Check for textual number patterns first
    for text_num, num in number_mapping.items():
        pattern = rf"\b{text_num}\b"
        if re.search(pattern, text, re.IGNORECASE):
            return num

    patterns = [  # For right turns
        r"\bThere are\s*(\d+)\s*right turns\b",  # for proprietary
        r"\bThere is\s*(\d+)\s*right turn\b",
        r"\b(\d+)\s+right turn(s)?",
        r"answer is\s+(\d+)",
        r"answer is:\s*\n*\s*(\d+)",
        r"from S to E is\s+(\d+)",
        r"Answer:\*\*\s*(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))  # Return the first matching group as integer

    # If no specific pattern matches, try to extract the first number in the text
    fallback_match = re.search(r"\d+", text)
    if fallback_match:
        return int(fallback_match.group(0))

    return None  # Return None if no number or textual number is found at all


def extract_answer_from_text_spatialgrid(text: str) -> Optional[int]:
    """Extracts answers from spatial grid text."""
    number_mapping = {
        "zero": 0,
        "no": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }

    found_numbers = []

    # Check for textual numbers and their positions
    for text_num, num in number_mapping.items():
        for match in re.finditer(rf"\b{text_num}\b", text, re.IGNORECASE):
            found_numbers.append((match.start(), num))

    # Check for digit sequences and their positions, specifically ignoring list markers at the start
    # Exclude numbers following "\n\n" and directly followed by ". "
    text = re.sub(
        r"^\n\n\d+\.\s", "", text
    )  # Remove the leading list marker if it exists

    for match in re.finditer(r"\d+", text):
        found_numbers.append((match.start(), int(match.group(0))))

    # Sort found numbers by their positions (smallest position first)
    if found_numbers:
        found_numbers.sort(key=lambda x: x[0])
        # Return the number associated with the earliest position
        return found_numbers[0][1]

    return None


def evaluate_model_accuracy(
    model_output_path: str, eval_summary_path: str, model_name: Optional[str] = None
) -> Tuple[float, int]:
    """Evaluates the accuracy of the model based on the output and summary paths."""
    eval_summary: List[Dict[str, str]] = []
    correct_answers = 0
    line_count = 0

    with open(model_output_path, "r") as f:
        for line in f:
            data = json.loads(line)
            question_id = int(data["id"].split(".")[-1])
            task = data["id"].split(".")[0]
            line_count += 1
            try:
                if task == "spatialmap":
                    model_answer = extract_answer_from_text_spatialmap(
                        data["answer"], question_id, model_name
                    )
                elif task == "mazenav":
                    model_answer = extract_answer_from_text_mazenav(
                        data["answer"], question_id, model_name
                    )
                elif task == "spatialgrid":
                    model_answer = extract_answer_from_text_spatialgrid(
                        data["answer"], question_id, model_name
                    )

                ref_ans = str(data["oracle_answer"]).lower()
                model_answer = str(model_answer).lower()
                eval_result = int(ref_ans.lower() in model_answer.lower())
                correct_answers += eval_result
                eval_summary.append(
                    {
                        "ref": ref_ans,
                        "model_output": model_answer,
                        "eval_result": eval_result,
                    }
                )
            except ValueError:
                continue
