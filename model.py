import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(model_name=MODEL_NAME, device=None):
    """Load tokenizer and causal language model from Hugging Face."""
    if device is None:
        device = get_device()

    print(f"Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model for '{model_name}' on device '{device}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        attn_implementation="eager",
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt,
    max_new_tokens=256,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    device=None,
):
    """Generate a response for a given text prompt."""
    if device is None:
        device = next(model.parameters()).device

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Determine sampling mode based on temperature
    do_sample = temperature is not None and float(temperature) > 0.0

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }

    if do_sample:
        generate_kwargs["temperature"] = float(temperature)
        if top_k is not None and top_k > 0:
            generate_kwargs["top_k"] = int(top_k)
        if top_p is not None and 0.0 < float(top_p) <= 1.0:
            generate_kwargs["top_p"] = float(top_p)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            **generate_kwargs,
        )

    # Extract only newly generated tokens
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ]
    return response


def main():
    device = get_device()
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device=device)

    test_input = "Explain what a Large Language Model is in one concise sentence."
    print("\n--- Testing Model Generation ---")
    print(f"Input Prompt: {test_input}")

    response = generate_response(
        model,
        tokenizer,
        test_input,
        max_new_tokens=256,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        device=device,
    )
    print(f"Generated Response: {response}")
    print("--------------------------------\n")
    print("Local model test completed successfully!")


if __name__ == "__main__":
    main()
