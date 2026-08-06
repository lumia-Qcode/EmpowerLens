import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

def custom_forward(inputs, model):
    """
    Captum requires a function that explicitly returns the raw logits 
    from the model to calculate the backward gradients.
    """
    return model(inputs).logits

def highlight_distortion(text: str, model_path: str, threshold_ratio: float = 0.6) -> str:
    """
    Analyzes text and returns an HTML string with distorted phrases highlighted.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the tokenizer and your locally fine-tuned MentalRoBERTa model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # 2. Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)

    # 3. Predict the dominant cognitive distortion
    with torch.no_grad():
        logits = model(input_ids).logits
        pred_class = torch.argmax(torch.sigmoid(logits), dim=-1).item()

    # 4. Initialize Captum on RoBERTa's embedding layer
    # UPDATE: Changed from model.bert.embeddings to model.roberta.embeddings
    lig = LayerIntegratedGradients(
        lambda x: custom_forward(x, model), 
        model.roberta.embeddings
    )
    
    attributions, _ = lig.attribute(
        inputs=input_ids, 
        target=pred_class, 
        return_convergence_delta=True
    )

    # 5. Process the attribution scores for each token
    scores = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

    # Calculate the dynamic cutoff score based on the highest-scoring word
    max_score = max(scores) if len(scores) > 0 else 0
    cutoff = max_score * threshold_ratio

    # 6. Reconstruct the sentence and apply HTML <mark> tags
    highlighted_words = []
    
    for token, score in zip(tokens, scores):
        # UPDATE: RoBERTa uses <s>, </s>, and <pad> as special tokens instead of [CLS], [SEP]
        if token in ["<s>", "</s>", "<pad>"]:
            continue

        # UPDATE: RoBERTa uses 'Ġ' to indicate a space BEFORE the word
        if token.startswith("Ġ"):
            clean_token = token.replace("Ġ", "")
            prefix = " " # Add a space
        else:
            clean_token = token
            prefix = "" # No space; attach directly to the previous sub-word

        # Apply the HTML highlight if the word passes the threshold
        if score >= cutoff and cutoff > 0:
            formatted_word = f"{prefix}<mark style='background-color: #ffcccc; padding: 0.1em; border-radius: 3px; font-weight: bold;'>{clean_token}</mark>"
        else:
            formatted_word = f"{prefix}{clean_token}"

        highlighted_words.append(formatted_word)

    # Clean up any leading whitespace and return the final HTML string
    return "".join(highlighted_words).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Captum Explainability with MentalRoBERTa")
    parser.add_argument("--text", type=str, required=True, help="The self-reflection text to analyze")
    parser.add_argument(
        "--model", 
        type=str, 
        # UPDATE: Point to your MentalRoBERTa checkpoint
        default="checkpoints/multilabel_mental-roberta-base_42",
        help="Path to the saved model checkpoint directory"
    )
    args = parser.parse_args()

    html_output = highlight_distortion(args.text, args.model)
    print("\n--- Original Text ---")
    print(args.text)
    print("\n--- Highlighted HTML Output ---")
    print(html_output)