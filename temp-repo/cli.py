# === cli.py ===

from model_utils import classify_text, backup_classify
from nodes import ConfidenceCheckNode, FallbackNode
import logging
import os
import warnings
import matplotlib.pyplot as plt

# === Suppress warnings ===
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# === Logging setup ===
logging.basicConfig(
    filename="logs.txt",
    filemode="a",
    format="%(asctime)s - %(message)s",
    level=logging.INFO
)

# === Instantiate nodes ===
confidence_checker = ConfidenceCheckNode(threshold=80.0)
fallback_handler = FallbackNode()

# === Tracking variables ===
confidence_log = []
fallback_count = 0
no_fallback_count = 0
input_count = 0

print("LangGraph Classification CLI (type 'exit' to quit)\n")

while True:
    user_input = input("Input: ").strip()
    if user_input.lower() == "exit":
        break

    input_count += 1

    # === Inference ===
    predicted_label, confidence = classify_text(user_input)
    confidence_log.append(confidence)

    print(f"[InferenceNode] Predicted label: {predicted_label} | Confidence: {confidence:.2f}%")
    logging.info(f"[InferenceNode] Predicted label: {predicted_label} | Confidence: {confidence:.2f}%")

    # === Prepare state ===
    state = {
        "text": user_input,
        "predicted_label": predicted_label,
        "confidence": confidence
    }

    # === Confidence check ===
    state = confidence_checker(state)

    # === Fallback logic ===
    if state["fallback"]:
        fallback_count += 1
        print("[BackupModel] Trying zero-shot fallback model...")
        backup_label, backup_confidence = backup_classify(user_input)
        print(f"[BackupModel] Predicted label: {backup_label} | Confidence: {backup_confidence:.2f}%")

        if backup_confidence >= 80.0:
            state["final_label"] = backup_label
            print(f"Final Label: {state['final_label']} (From backup model)\n")
            logging.info(f"[BackupModel] Final label used: {backup_label}")
        else:
            state = fallback_handler(state)
            print(f"Final Label: {state['final_label']} (Corrected via user clarification)\n")
            logging.info(f"[FallbackNode] Final label (user clarification): {state['final_label']}")
    else:
        no_fallback_count += 1
        print(f"Final Label: {state['final_label']} (Confidence: {confidence:.2f}%)\n")
        logging.info(f"[Final Output] Final label (accepted): {state['final_label']}")

    logging.info("-" * 50)

# === Plot Confidence Curve ===
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(confidence_log) + 1), confidence_log, marker='o', linestyle='-')
plt.title("Confidence Curve Across Inputs")
plt.xlabel("Input Instance")
plt.ylabel("Confidence (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.savefig("confidence_curve.png")
print("📈 Saved confidence curve as confidence_curve.png")

# === Fallback Summary ===
print("\n📊 Fallback Summary")
print(f"Total Inputs: {input_count}")
print(f"Fallback Triggered: {fallback_count} times")
print(f"No Fallback: {no_fallback_count} times")
