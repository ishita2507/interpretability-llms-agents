from dotenv import load_dotenv
load_dotenv()

# get the API Key
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(GEMINI_API_KEY[:5])

# Now you need to enter the password/API key to access the models/data

# Enter this in terminal to download all the google packages

## >> uv pip install google-genai   ---- run in terminal
from google import genai
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))  
# pass your api key to google to provide the models they house


# =========================
# 1. Setup
# =========================
import torch
import json
import re
import html
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# :white_check_mark: FIX 1: define device
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# =========================
# 2. Load Model
# =========================
# gemma-4-31B-it requires access via HuggingFace Hub.
# Run: huggingface-cli login
# And accept license at: https://huggingface.co/google/gemma-4-31b-it

model_name = "google/gemma-4-31b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # :white_check_mark: FIX 2
model.eval()

# =========================
# 3. Preprocessing
# =========================
def preprocess_text(text: str) -> str | None:
    """
    Clean and validate a civil_comments text entry.
    Returns None if the text should be skipped.
    """
    if not isinstance(text, str):
        return None

    # Decode HTML entities (e.g. &amp; -> &, &lt; -> <)
    text = html.unescape(text)

    # Strip HTML tags (e.g. <br>, <b>, <i>)
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize unicode punctuation to ASCII equivalents
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================
# 4. Load + Filter Dataset
# =========================
raw_data = load_dataset("google/civil_comments")["test"].select(range(100))

processed_data = []
for item in raw_data:
    clean_text = preprocess_text(item["text"])
    if clean_text is None:
        continue

    processed_data.append({
        "text": clean_text,
        "raw_toxicity": item["toxicity"]   # Keep raw float score as-is
    })

# Deduplicate by exact text match
seen = set()
deduplicated_data = []
for item in processed_data:
    if item["text"] not in seen:
        seen.add(item["text"])
        deduplicated_data.append(item)

print(f"Samples after preprocessing + dedup: {len(deduplicated_data)}")
data = deduplicated_data[:20]

# =========================
# 5. Prompt (Instruction Format) - CLAUDE
# =========================

# PROMPT TYPE 1
def create_chat_prompt(text):
    """
    Gemma instruction-tuned models use a structured chat format.
    Using apply_chat_template is strongly preferred over raw strings.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a content moderation assistant. "
                "Classify whether the following comment is toxic or non-toxic. "
            )
        },
        {
            "role": "user",
            "content": f"Comment: {text}"
        }
    ]

# ******************** ISHITA's CODE ******************** 

# =========================
# 5. Zero-shot scoring
# =========================
def score_text(text):
    prompt = create_chat_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    last_token_logits = logits[0, -1]

    # :white_check_mark: FIX 3: safer tokens
    toxic_token = tokenizer.encode(" toxic", add_special_tokens=False)[0]
    safe_token = tokenizer.encode(" safe", add_special_tokens=False)[0]

    score = last_token_logits[toxic_token] - last_token_logits[safe_token]
    pred = "toxic" if score > 0 else "non-toxic"

    return score.item(), pred

# =========================
# 6. Integrated Gradients
# =========================
def integrated_gradients(text, steps=20):
    prompt = create_chat_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # :white_check_mark: FIX 1: detach embeddings (no graph reuse)
    embeddings = model.transformer.wte(input_ids).detach()

    baseline = torch.zeros_like(embeddings)

    grads = []

    for i in range(steps + 1):
        alpha = float(i) / steps

        scaled = baseline + alpha * (embeddings - baseline)
        scaled.requires_grad_(True)

        outputs = model(inputs_embeds=scaled)
        logits = outputs.logits[:, -1, :]

        toxic_token = tokenizer.encode(" toxic", add_special_tokens=False)[0]
        score = logits[:, toxic_token].sum()

        # :white_check_mark: FIX 2: use autograd.grad instead of backward
        grad = torch.autograd.grad(score, scaled)[0]
        grads.append(grad.detach())

    avg_grads = torch.mean(torch.stack(grads), dim=0)

    attributions = (embeddings - baseline) * avg_grads
    attributions = attributions.sum(dim=-1).squeeze()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return tokens, attributions.cpu().numpy()

# =========================
# 7. Run
# =========================
results = []

for item in data:
    text = item["text"]

    score, pred = score_text(text)
    tokens, attributions = integrated_gradients(text)

    results.append({
        "text": text,
        "prediction": pred,
        "score": score,
        "tokens": tokens,
        "importance": attributions.tolist()
    })

# =========================
# 8. Save
# =========================
with open("output_with_ig.json", "w") as f:
    json.dump(results, f, indent=4)

print("Done! Saved to output_with_ig.json")







# # ******************** CLAUDE CODE ******************** 

# # =========================
# # 6. Zero-shot Scoring
# # =========================
# def score_text(text: str) -> tuple[float, str]:
#     """
#     Score a text for toxicity using next-token log-probability comparison.
#     Compares P('toxic') vs P('non') at the first response token position.
#     """
#     messages = create_chat_prompt(text)

#     # apply_chat_template formats with <start_of_turn> / <end_of_turn> tags
#     prompt = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True  # Adds the model's turn opener
#     )

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits

#     last_token_logits = logits[0, -1]

#     # Compare first token of "toxic" vs "non" (as in "non-toxic")
#     # Using the first subword token is more robust for multi-token words
#     toxic_token = tokenizer.encode("toxic", add_special_tokens=False)[0]
#     safe_token = tokenizer.encode("non", add_special_tokens=False)[0]

#     score = last_token_logits[toxic_token] - last_token_logits[safe_token]
#     pred = "toxic" if score > 0 else "non-toxic"

#     return score.item(), pred


# # =========================
# # 7. Integrated Gradients
# # =========================
# def get_embedding_layer(model):
#     """
#     Retrieve the token embedding layer from Gemma.
#     Gemma 4 uses model.model.embed_tokens (not model.transformer.wte).
#     """
#     return model.model.embed_tokens


# def integrated_gradients(text: str, steps: int = 20) -> tuple[list[str], list[float]]:
#     """
#     Compute Integrated Gradients attributions for the toxicity score.
#     Attributes importance of each input token toward the 'toxic' prediction.
#     """
#     messages = create_chat_prompt(text)
#     prompt = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
#     input_ids = inputs["input_ids"]

#     embed_layer = get_embedding_layer(model)

#     # Detach to avoid retaining graph across IG steps
#     embeddings = embed_layer(input_ids).detach()
#     baseline = torch.zeros_like(embeddings)

#     toxic_token_id = tokenizer.encode("toxic", add_special_tokens=False)[0]

#     grads = []
#     for i in range(steps + 1):
#         alpha = float(i) / steps
#         scaled = baseline + alpha * (embeddings - baseline)
#         scaled.requires_grad_(True)

#         outputs = model(inputs_embeds=scaled)
#         logits = outputs.logits[:, -1, :]
#         score = logits[:, toxic_token_id].sum()

#         grad = torch.autograd.grad(score, scaled)[0]
#         grads.append(grad.detach())

#     avg_grads = torch.mean(torch.stack(grads), dim=0)
#     attributions = (embeddings - baseline) * avg_grads

#     # Sum across embedding dimension, squeeze batch dim
#     attributions = attributions.sum(dim=-1).squeeze()

#     tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
#     return tokens, attributions.cpu().numpy()


# # =========================
# # 8. Evaluation Helpers
# # =========================
# def compute_metrics(results: list[dict]) -> dict:
#     """Compute accuracy, precision, recall, F1 against binarized ground truth."""
#     tp = fp = tn = fn = 0
#     for r in results:
#         pred_toxic = r["prediction"] == "toxic"
#         true_toxic = r["label"] == 1
#         if pred_toxic and true_toxic:   tp += 1
#         elif pred_toxic and not true_toxic: fp += 1
#         elif not pred_toxic and not true_toxic: tn += 1
#         else:                           fn += 1

#     accuracy  = (tp + tn) / len(results) if results else 0
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
#     f1        = (2 * precision * recall / (precision + recall)
#                  if (precision + recall) > 0 else 0)

#     return {"accuracy": accuracy, "precision": precision,
#             "recall": recall, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# # =========================
# # 9. Run
# # =========================
# results = []

# for i, item in enumerate(data):
#     text  = item["text"]
#     label = item["label"]

#     print(f"[{i+1}/{len(data)}] Processing...")

#     score, pred = score_text(text)
#     tokens, attributions = integrated_gradients(text)

#     results.append({
#         "text": text,
#         "label": label,              # Ground truth (0 = non-toxic, 1 = toxic)
#         "raw_toxicity": item["raw_toxicity"],
#         "prediction": pred,
#         "score": score,
#         "tokens": tokens,
#         "importance": attributions.tolist()
#     })

# metrics = compute_metrics(results)
# print(f"\nMetrics: {json.dumps(metrics, indent=2)}")

# # =========================
# # 10. Save
# # =========================
# output = {"metrics": metrics, "results": results}
# with open("output_gemma4_ig.json", "w") as f:
#     json.dump(output, f, indent=4)

# print("Done! Saved to output_gemma4_ig.json")