"""
Predefined evaluation checklist for Medical Reports.
"""

from typing import List, Dict

# The 10 predefined evaluation criteria
# 0 = Incorrect/Missing, 1 = Partially Correct, 2 = Fully Correct
EVALUATION_CRITERIA: List[Dict[str, str]] = [
    {
        "id": "q1",
        "question": "Does the report correctly reflect the predicted classification?",
        "description": "Ensure the clinical classification matches the model's prediction precisely."
    },
    {
        "id": "q2",
        "question": "Is the confidence score correctly presented?",
        "description": "Check if the numeric confidence score is included and accurate."
    },
    {
        "id": "q3",
        "question": "Does the Grad-CAM observation align with the highlighted regions?",
        "description": "Crucial: Does the report's explanation of the Grad-CAM heatmap accurately match the image's highlighting and the predicted condition?"
    },
    {
        "id": "q4",
        "question": "Is the clinical interpretation consistent with the classifier output?",
        "description": "The interpretation should not contradict the AI prediction."
    },
    {
        "id": "q5",
        "question": "Does the report avoid unsupported medical claims?",
        "description": "The report must not make definitive diagnoses beyond the scope of a screening AI."
    },
    {
        "id": "q6",
        "question": "Does the report avoid hallucinating patient information?",
        "description": "Ensure no fake patient names, IDs, or history are invented."
    },
    {
        "id": "q7",
        "question": "Is the recommendation appropriate?",
        "description": "Recommendations should advise clinical correlation and follow-up, not extreme measures."
    },
    {
        "id": "q8",
        "question": "Is the disclaimer included?",
        "description": "The AI limitation/medical disclaimer must be explicitly stated."
    },
    {
        "id": "q9",
        "question": "Is the report internally consistent?",
        "description": "Ensure there are no contradictory statements across paragraphs."
    },
    {
        "id": "q10",
        "question": "Is the language professional and medically appropriate?",
        "description": "Tone must be objective, clinical, and reassuring."
    }
]
