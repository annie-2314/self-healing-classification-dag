# === nodes.py ===

class ConfidenceCheckNode:
    def __init__(self, threshold: float = 80.0):  # percent threshold
        self.threshold = threshold

    def __call__(self, state: dict) -> dict:
        confidence = state.get("confidence", 0.0)
        # print(f"[ConfidenceCheckNode] Confidence: {confidence:.2f}%")

        if confidence < self.threshold:
            print("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
            state["fallback"] = True
        else:
            print("[ConfidenceCheckNode] Confidence sufficient. No fallback needed.")
            state["final_label"] = state["predicted_label"]
            state["fallback"] = False

        return state


class FallbackNode:
    def __call__(self, state: dict) -> dict:
        # Dynamically suggest likely fallback based on predicted label
        suggested_label = state.get("predicted_label", "neutral").lower()
        print(f"[FallbackNode] Could you clarify your intent? Was this a {suggested_label.lower()} review?")
        clarification = input("User: ")  # ← Do NOT print it again

        # Match common variations
        if "positive" in clarification.lower():
            state["final_label"] = "Positive"
        elif "negative" in clarification.lower():
            state["final_label"] = "Negative"
        elif "neutral" in clarification.lower():
            state["final_label"] = "Neutral"
        else:
            print("[FallbackNode] Couldn't understand. Keeping original prediction.")
            state["final_label"] = state.get("predicted_label", "Neutral")

        return state

