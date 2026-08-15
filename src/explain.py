import argparse
import string
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

def custom_forward(inputs, model):
    return model(inputs).logits

def highlight_distortion(text: str, model_path: str, threshold_ratio: float = 0.6) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids).logits
        pred_class = torch.argmax(torch.sigmoid(logits), dim=-1).item()

    lig = LayerIntegratedGradients(
        lambda x: custom_forward(x, model), 
        model.roberta.embeddings
    )
    
    attributions, _ = lig.attribute(
        inputs=input_ids, 
        target=pred_class, 
        return_convergence_delta=True
    )

    scores = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

    # 5. Calculate threshold, entirely ignoring punctuation tokens
    valid_scores = [
        s for t, s in zip(tokens, scores) 
        if t not in ["<s>", "</s>", "<pad>"] and not all(c in string.punctuation for c in t.replace("Ġ", ""))
    ]
    max_score = max(valid_scores) if valid_scores else 0
    cutoff = max_score * threshold_ratio

    # 6. Group sub-words into full words and isolate punctuation
    grouped_words = []
    
    for token, score in zip(tokens, scores):
        if token in ["<s>", "</s>", "<pad>"]:
            continue

        clean_token = token.replace("Ġ", "")
        is_new_word = token.startswith("Ġ")
        is_punct = all(char in string.punctuation for char in clean_token)

        if is_new_word or not grouped_words:
            # Start a new word group
            grouped_words.append({
                "text": clean_token,
                "score": 0.0 if is_punct else score,
                "prefix": " " if is_new_word else ""
            })
        else:
            if is_punct:
                # Keep punctuation separate so it never inherits a word's highlight
                grouped_words.append({
                    "text": clean_token,
                    "score": 0.0,
                    "prefix": ""
                })
            else:
                # Combine sub-words (e.g., "shouldn" + "'t" = "shouldn't")
                grouped_words[-1]["text"] += clean_token
                # Give the combined word the highest score of its sub-parts
                grouped_words[-1]["score"] = max(grouped_words[-1]["score"], score)

    # 7. Apply HTML tags to the cleanly reconstructed words
    highlighted_words = []
    for gw in grouped_words:
        if gw["score"] >= cutoff and cutoff > 0:
            formatted_word = f"{gw['prefix']}<mark style='background-color: #ffcccc; padding: 0.1em; border-radius: 3px; font-weight: bold;'>{gw['text']}</mark>"
        else:
            formatted_word = f"{gw['prefix']}{gw['text']}"
            
        highlighted_words.append(formatted_word)

    return "".join(highlighted_words).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Captum Explainability with MentalRoBERTa")
    parser.add_argument("--text", type=str, required=True, help="The self-reflection text to analyze")
    parser.add_argument(
        "--model", 
        type=str, 
        default="checkpoints/multilabel_mental-roberta-base_42",
        help="Path to the saved model checkpoint directory"
    )
    args = parser.parse_args()

    html_output = highlight_distortion(args.text, args.model)
    print("\n--- Original Text ---")
    print(args.text)
    print("\n--- Highlighted HTML Output ---")
    print(html_output)