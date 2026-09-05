"""Manifest definitions, schema columns, and dataset utilities."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from singing_identity.utils.research_utils import (
    UTTERANCE_COLUMNS,
    PAIR_COLUMNS,
    PHONE_EXAMPLE_COLUMNS,
    TECHNIQUE_PAIR_COLUMNS,
    read_table,
    write_table,
    write_json,
    load_json,
)


def normalize_lexical_text(value: Any) -> str:
    """Normalize text by removing boundary markers and whitespace."""
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"<(?:ap|sp)>", "", text, flags=re.IGNORECASE)
    return "".join(char for char in text if not char.isspace())


def select_clean_control_pairs(
    pairs: list[dict[str, Any]], utterances: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select verified same-text control pairs matching both speaker and text."""
    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    seen_utt_pairs: set[tuple[str, str]] = set()

    for pair in pairs:
        pair_id = str(pair.get("pair_id", ""))
        speech_id = str(pair.get("speech_utt_id", ""))
        singing_id = str(pair.get("singing_utt_id", ""))
        reason = "accepted"
        speech = utterances.get(speech_id)
        singing = utterances.get(singing_id)

        if str(pair.get("technique", "")).strip().lower() != "control":
            reason = "not_control"
        elif not speech or not singing:
            reason = "missing_utterance"
        elif str(speech.get("speaker_id", "")) != str(pair.get("speaker_id", "")) or str(
            singing.get("speaker_id", "")
        ) != str(pair.get("speaker_id", "")):
            reason = "speaker_mismatch"
        elif normalize_lexical_text(speech.get("text")) != normalize_lexical_text(singing.get("text")):
            reason = "lexical_mismatch"
        elif not normalize_lexical_text(speech.get("text")):
            reason = "empty_text"
        elif pair_id in seen_pair_ids or (speech_id, singing_id) in seen_utt_pairs:
            reason = "duplicate_pair"

        if reason == "accepted":
            seen_pair_ids.add(pair_id)
            seen_utt_pairs.add((speech_id, singing_id))
            accepted.append(pair)

        audit.append(
            {
                "pair_id": pair_id,
                "speaker_id": pair.get("speaker_id", ""),
                "language": pair.get("language", ""),
                "song_id": pair.get("song_id", ""),
                "technique": pair.get("technique", ""),
                "speech_utt_id": speech_id,
                "singing_utt_id": singing_id,
                "manifest_same_text_flag": pair.get("same_text_flag", ""),
                "selection_status": reason,
            }
        )

    if not accepted:
        raise ValueError("no clean same-text control pairs found in selection")

    return accepted, audit


__all__ = [
    "UTTERANCE_COLUMNS",
    "PAIR_COLUMNS",
    "PHONE_EXAMPLE_COLUMNS",
    "TECHNIQUE_PAIR_COLUMNS",
    "read_table",
    "write_table",
    "write_json",
    "load_json",
    "normalize_lexical_text",
    "select_clean_control_pairs",
]
