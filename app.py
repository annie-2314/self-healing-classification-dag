import streamlit as st
from model_utils import classify_text, backup_classify
from nodes import ConfidenceCheckNode, FallbackNode
import matplotlib.pyplot as plt

# === Globals ===
confidence_checker = ConfidenceCheckNode(threshold=80.0)
fallback_handler = FallbackNode()

# === Session State ===
if "history" not in st.session_state:
    st.session_state.history = []
if "fallback_count" not in st.session_state:
    st.session_state.fallback_count = 0
if "total_inputs" not in st.session_state:
    st.session_state.total_inputs = 0
if "clarifications" not in st.session_state:
    st.session_state.clarifications = {}
if "multi_inputs" not in st.session_state:
    st.session_state.multi_inputs = []
if "ready_for_graph" not in st.session_state:
    st.session_state.ready_for_graph = False
if "single_input_state" not in st.session_state:
    st.session_state.single_input_state = None  # To store the current single input state

# **CSS Styling removed: .final-box { background-color: #eaf9ea; border-left: 5px solid #33cc33; padding: 10px; margin: 10px 0; font-weight: bold; } .user-box { background-color: #e6f0ff; border-left: 5px solid #3399ff; padding: 10px; margin: 0px 0; }**

# === UI Layout ===
st.set_page_config(page_title="Self-Healing Classifier", layout="centered")
st.title("🤖 Self-Healing Sentiment Classifier")

mode = st.radio("Choose Input Mode:", ["Single", "Multiple Inputs"])

# === SINGLE INPUT ===
if mode == "Single":
    user_input = st.text_input("Enter your review:", key="single_input")
    if st.button("Classify") and user_input.strip() and st.session_state.single_input_state is None:
        st.session_state.total_inputs += 1
        st.write(f"Input: {user_input}")
        predicted_label, confidence = classify_text(user_input)
        st.write(f"[InferenceNode] Predicted label: {predicted_label} | Confidence: {confidence:.2f}%")

        state = {
            "text": user_input,
            "predicted_label": predicted_label,
            "confidence": confidence
        }

        state = confidence_checker(state)
        final_label = predicted_label  # default

        if state["fallback"]:
            st.write(f"[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
            st.session_state.fallback_count += 1

            backup_label, backup_conf = backup_classify(user_input)
            st.write("[BackupModel] Trying zero-shot fallback model...")
            st.write(f"[BackupModel] Predicted label: {backup_label} | Confidence: {backup_conf:.2f}%")

            st.session_state.single_input_state = {
                "text": user_input,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "fallback": state["fallback"],
                "backup_label": backup_label,
                "backup_conf": backup_conf
            }
        else:
            st.markdown(f'<div class="final-box">Final Label: {final_label} (Confidence: {confidence:.2f}%)</div>', unsafe_allow_html=True)
            st.session_state.history.append((user_input, final_label, confidence))

    # Handle clarification for single input
    if st.session_state.single_input_state:
        clarification = st.text_input("[FallbackNode] Could you clarify your intent? (positive/negative/neutral)", key="clarify_single")
        state = st.session_state.single_input_state

        if clarification:
            cleaned = clarification.strip().lower()
            st.markdown(f'<div class="user-box">User: {clarification.strip()}</div>', unsafe_allow_html=True)
            if cleaned in ["positive", "negative", "neutral"]:
                final_label = cleaned
                st.markdown(f'<div class="final-box">Final Label: {final_label} (Corrected via user clarification)</div>', unsafe_allow_html=True)
            else:
                final_label = state["backup_label"]
                st.warning("⚠️ Invalid input. Defaulting to backup model.")
                st.markdown(f'<div class="final-box">Final Label: {final_label} (From backup model)</div>', unsafe_allow_html=True)
            st.session_state.history.append((state["text"], final_label, state["confidence"]))
            st.session_state.single_input_state = None  # Reset after processing
        else:
            final_label = state["backup_label"]
            st.markdown(f'<div class="final-box">Final Label: {final_label} (From backup model)</div>', unsafe_allow_html=True)

    if st.button("Generate Graph") and st.session_state.history:
        st.session_state.ready_for_graph = True

# === MULTIPLE INPUTS ===
elif mode == "Multiple Inputs":
    batch_text = st.text_area("Enter multiple reviews (one per line):", height=200)

    if st.button("Run Multiple"):
        inputs = [line.strip() for line in batch_text.split("\n") if line.strip()]
        st.session_state.multi_inputs = []
        st.session_state.total_inputs += len(inputs)

        for i, user_input in enumerate(inputs):
            predicted_label, confidence = classify_text(user_input)
            state = {
                "text": user_input,
                "predicted_label": predicted_label,
                "confidence": confidence
            }
            state = confidence_checker(state)

            backup_label, backup_conf = None, None
            if state["fallback"]:
                backup_label, backup_conf = backup_classify(user_input)
                st.session_state.fallback_count += 1

            st.session_state.multi_inputs.append({
                "index": i,
                "text": user_input,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "fallback": state["fallback"],
                "backup_label": backup_label,
                "backup_conf": backup_conf
            })

    for entry in st.session_state.multi_inputs:
        i = entry["index"]
        st.write(f"Input {i+1}: {entry['text']}")
        st.write(f"[InferenceNode] Predicted label: {entry['predicted_label']} | Confidence: {entry['confidence']:.2f}%")

        if entry["fallback"]:
            st.write("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
            st.write("[BackupModel] Trying zero-shot fallback model...")
            st.write(f"[BackupModel] Predicted label: {entry['backup_label']} | Confidence: {entry['backup_conf']:.2f}%")

            clarification_key = f"clarify_{i}"
            clarification = st.text_input("[FallbackNode] Could you clarify your intent? (positive/negative/neutral)", key=clarification_key)

            if clarification:
                cleaned = clarification.strip().lower()
                st.markdown(f'<div class="user-box">User: {clarification.strip()}</div>', unsafe_allow_html=True)
                if cleaned in ["positive", "negative", "neutral"]:
                    final_label = cleaned
                    st.markdown(f'<div class="final-box">Final Label: {final_label} (Corrected via user clarification)</div>', unsafe_allow_html=True)
                else:
                    final_label = entry["backup_label"]
                    st.warning("⚠️ Invalid input. Defaulting to backup model.")
                    st.markdown(f'<div class="final-box">Final Label: {final_label} (From backup model)</div>', unsafe_allow_html=True)
            else:
                final_label = entry["backup_label"]
                st.markdown(f'<div class="final-box">Final Label: {final_label} (From backup model)</div>', unsafe_allow_html=True)
        else:
            final_label = entry["predicted_label"]
            st.markdown(f'<div class="final-box">Final Label: {final_label} (Confidence: {entry['confidence']:.2f}%)</div>', unsafe_allow_html=True)

        # Update history only once with the final label after all steps
        st.session_state.history.append((entry["text"], final_label, entry["confidence"]))

    if st.button("Generate Graph") and st.session_state.history:
        st.session_state.ready_for_graph = True

# === Confidence Curve and Stats ===
if st.session_state.ready_for_graph and st.session_state.history:
    st.subheader("📈 Confidence Curve")
    labels = [entry[1] for entry in st.session_state.history]
    confs = [entry[2] for entry in st.session_state.history]
    texts = [entry[0] for entry in st.session_state.history]

    fig, ax = plt.subplots()
    ax.plot(range(1, len(confs)+1), confs, marker='o')
    ax.set_title("Confidence Scores Over Inputs")
    ax.set_xlabel("Input Index")
    ax.set_ylabel("Confidence %")
    ax.set_xticks(range(1, len(texts)+1))
    ax.set_xticklabels(range(1, len(texts)+1))
    ax.grid(True)
    st.pyplot(fig)

    fallback_ratio = (st.session_state.fallback_count / st.session_state.total_inputs * 100) if st.session_state.total_inputs else 0
    st.markdown(f"""
    ---
    ### 📊 Fallback Statistics
    - Total Inputs: `{st.session_state.total_inputs}`
    - Fallbacks Triggered: `{st.session_state.fallback_count}`
    - Fallback Rate: **{fallback_ratio:.2f}%**
    """)
