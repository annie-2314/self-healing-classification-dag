# inference_pipeline.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import datetime
from langgraph.graph import StateGraph, END
from typing import TypedDict

# === Setup ===
MODEL_PATH = "./fine_tuned_model"
CONFIDENCE_THRESHOLD = 0.75
LOG_FILE = "logs.txt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# 3-class label mapping
LABELS = ["negative", "neutral", "positive"]

# === State Type ===
class State(TypedDict):
    input_text: str
    prediction: str
    confidence: float
    user_feedback: str
    final_label: str

# === Logging ===
def log_event(entry: str):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()}: {entry}\n")

# === Node: Inference ===
def inference_node(state: State):
    inputs = tokenizer(state["input_text"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        confidence, predicted = torch.max(probs, dim=1)
        label = LABELS[predicted.item()]
        state["prediction"] = label
        state["confidence"] = confidence.item()
        log_event(f"[InferenceNode] Prediction: {label} | Confidence: {confidence.item():.2f}")
    return state

# === Node: Confidence Check ===
def confidence_check_node(state: State):
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        log_event(f"[ConfidenceCheckNode] Low confidence ({state['confidence']:.2f}). Triggering fallback.")
        return {"next": "fallback_node"}
    log_event(f"[ConfidenceCheckNode] Confidence OK ({state['confidence']:.2f}).")
    state["final_label"] = state["prediction"]
    return {"next": END}

# === Node: Fallback (Ask user) ===
def fallback_node(state: State):
    question = f"Could you clarify your intent? Was this a negative, neutral, or positive review?\nUser: "
    user_input = input(question).strip().lower()
    if user_input in LABELS:
        state["final_label"] = user_input
    else:
        state["final_label"] = "neutral"  # default fallback
    state["user_feedback"] = user_input
    log_event(f"[FallbackNode] User clarified as: {state['final_label']}")
    return state

# === LangGraph DAG ===
builder = StateGraph(State)
builder.add_node("inference_node", inference_node)
builder.add_node("confidence_check_node", confidence_check_node)
builder.add_node("fallback_node", fallback_node)

builder.set_entry_point("inference_node")
builder.add_edge("inference_node", "confidence_check_node")
builder.add_conditional_edges("confidence_check_node", confidence_check_node)
builder.add_edge("fallback_node", END)

graph = builder.compile()

# === CLI Loop ===
def run_cli():
    print("\nLangGraph Classification CLI (type 'exit' to quit)")
    while True:
        user_input = input("\nEnter a review: ").strip()
        if user_input.lower() == "exit":
            break
        initial_state: State = {
            "input_text": user_input,
            "prediction": "",
            "confidence": 0.0,
            "user_feedback": "",
            "final_label": ""
        }
        final_state = graph.invoke(initial_state)
        print(f"🧠 Final Label: {final_state['final_label']} (Confidence: {final_state['confidence']:.2f})")

if __name__ == "__main__":
    run_cli()
