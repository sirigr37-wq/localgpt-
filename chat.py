"""
Chat logic module for LocalGPT (Phase 2 - Step 2: Streaming Response).
Handles prompt building with system prompt, multi-turn history, and Qwen streaming generation.
"""

from typing import List, Dict, Optional, Iterator
from threading import Thread
import torch
from transformers import TextIteratorStreamer


def stream_chat_response(
    model,
    tokenizer,
    chat_messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device=None,
) -> Iterator[str]:
    """
    Stream response chunks from Qwen model given a multi-turn list of chat messages.
    Yields text chunks as they are generated.
    """
    if device is None:
        device = next(model.parameters()).device

    formatted_text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    safe_max_tokens = max(1, min(2048, int(max_new_tokens) if max_new_tokens is not None else 512))
    safe_temp = max(0.0, min(2.0, float(temperature))) if temperature is not None else 0.7
    do_sample = safe_temp > 0.0

    generate_kwargs = {
        **model_inputs,
        "max_new_tokens": safe_max_tokens,
        "do_sample": do_sample,
        "streamer": streamer,
    }

    if do_sample:
        generate_kwargs["temperature"] = safe_temp
        if top_k is not None:
            safe_top_k = max(1, min(100, int(top_k)))
            generate_kwargs["top_k"] = safe_top_k
        if top_p is not None:
            safe_top_p = max(0.01, min(1.0, float(top_p)))
            generate_kwargs["top_p"] = safe_top_p

    def generate_worker():
        with torch.inference_mode():
            model.generate(**generate_kwargs)

    thread = Thread(target=generate_worker)
    thread.start()

    for text_chunk in streamer:
        if text_chunk:
            yield text_chunk

    thread.join()


def generate_chat_response(
    model,
    tokenizer,
    chat_messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device=None,
) -> str:
    """
    Generate complete response from Qwen model given a multi-turn list of chat messages.
    (Non-streaming fallback).
    """
    if device is None:
        device = next(model.parameters()).device

    formatted_text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    safe_max_tokens = max(1, min(2048, int(max_new_tokens) if max_new_tokens is not None else 512))
    safe_temp = max(0.0, min(2.0, float(temperature))) if temperature is not None else 0.7
    do_sample = safe_temp > 0.0

    generate_kwargs = {
        "max_new_tokens": safe_max_tokens,
        "do_sample": do_sample,
    }

    if do_sample:
        generate_kwargs["temperature"] = safe_temp
        if top_k is not None:
            safe_top_k = max(1, min(100, int(top_k)))
            generate_kwargs["top_k"] = safe_top_k
        if top_p is not None:
            safe_top_p = max(0.01, min(1.0, float(top_p)))
            generate_kwargs["top_p"] = safe_top_p

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            **generate_kwargs,
        )

    # Extract only newly generated tokens
    new_token_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ][0]

    response = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return response.strip()
