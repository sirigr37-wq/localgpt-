"""
Tokenizer helper module for LLM X-Ray.
Provides functions for tokenizing text, extracting token IDs, and formatting token statistics.
"""

from typing import Dict, List
import pandas as pd


def tokenize_input(tokenizer, text: str, add_special_tokens: bool = False) -> Dict:
    """
    Tokenize input text and return tokens, IDs, and decoded representations.
    """
    if not text:
        return {
            "tokens": [],
            "token_ids": [],
            "display_tokens": [],
            "count": 0,
            "df": pd.DataFrame(),
        }

    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    token_ids = encoded["input_ids"]
    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)

    display_tokens = []
    table_rows = []

    for idx, (tok, tid) in enumerate(zip(raw_tokens, token_ids)):
        decoded = tokenizer.decode([tid])
        cleaned = str(tok).replace("\n", "\\n").replace("\t", "\\t")
        if not cleaned.strip():
            cleaned = f"␣ ({repr(tok)})"
        display_tokens.append(cleaned)

        table_rows.append({
            "Position": idx + 1,
            "Raw Token": str(tok),
            "Decoded Text": decoded,
            "Token ID": tid,
            "Length (chars)": len(decoded),
            "Is Special": tid in tokenizer.all_special_ids,
        })

    df = pd.DataFrame(table_rows)

    return {
        "tokens": raw_tokens,
        "token_ids": token_ids,
        "display_tokens": display_tokens,
        "count": len(token_ids),
        "df": df,
    }


def format_token_chips(display_tokens: List[str], token_ids: List[int]) -> str:
    """
    Generate an HTML string representing interactive visual token pills/chips.
    """
    chips_html = []
    colors = [
        "#e0e7ff", "#dbeafe", "#fce7f3", "#ecfdf5",
        "#fef3c7", "#ede9fe", "#ffedd5", "#cffafe"
    ]
    border_colors = [
        "#818cf8", "#60a5fa", "#f472b6", "#34d399",
        "#fbbf24", "#a78bfa", "#fb923c", "#22d3ee"
    ]

    for idx, (tok, tid) in enumerate(zip(display_tokens, token_ids)):
        bg = colors[idx % len(colors)]
        border = border_colors[idx % len(border_colors)]
        chip = (
            f'<span style="display:inline-block; margin:3px 4px; padding:3px 8px; '
            f'background-color:{bg}; border:1px solid {border}; border-radius:6px; '
            f'font-family:monospace; font-size:0.85rem; color:#1e293b;" '
            f'title="Token: {tok} | ID: {tid} | Pos: {idx+1}">'
            f'<strong>{tok}</strong> <sub style="color:#64748b; font-size:0.7rem;">#{tid}</sub>'
            f'</span>'
        )
        chips_html.append(chip)

    return "".join(chips_html)
