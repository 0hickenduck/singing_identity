#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import ExperimentError, read_table  # noqa: E402


def rel(path: str, base: Path) -> str:
    target = Path(path)
    try:
        return target.relative_to(base.parent).as_posix()
    except ValueError:
        return target.as_uri()


def audio(src: str, base: Path) -> str:
    return f'<audio controls preload="none" src="{html.escape(rel(src, base), quote=True)}"></audio>'


def read_scores(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    rows = read_table(path)
    return {
        f"{row['target_pair_id']}::{row['condition']}": row
        for row in rows
    }


def by_condition(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["condition"]): row for row in rows}


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local HTML page for paired audio review.")
    parser.add_argument("--listening-manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--out-html", type=Path, required=True)
    parser.add_argument("--title", default="Listening Review")
    args = parser.parse_args()

    try:
        rows = read_table(args.listening_manifest)
        if not rows:
            raise ExperimentError("listening manifest is empty")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["target_pair_id"])].append(row)
        scores = read_scores(args.scores)
        summary = json.loads(args.summary.read_text()) if args.summary and args.summary.exists() else {}
    except (ExperimentError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cards = []
    for idx, (pair_id, group) in enumerate(sorted(grouped.items()), 1):
        conds = by_condition(group)
        speech = conds.get("target_speech_prompt", {})
        singing = conds.get("target_singing_prompt", {})
        first = group[0]
        speech_score = scores.get(f"{pair_id}::target_speech_prompt", {})
        singing_score = scores.get(f"{pair_id}::target_singing_prompt", {})
        delta_target_singing = ""
        delta_target_speech = ""
        if speech_score and singing_score:
            delta_target_singing = fmt(
                float(singing_score["sim_to_target_singing_prompt"])
                - float(speech_score["sim_to_target_singing_prompt"])
            )
            delta_target_speech = fmt(
                float(singing_score["sim_to_target_speech_prompt"])
                - float(speech_score["sim_to_target_speech_prompt"])
            )
        cards.append(
            f"""
      <section class="pair">
        <header>
          <h2>{idx}. {html.escape(str(first.get("target_speaker_id", "")))}</h2>
          <div class="meta">
            Source: {html.escape(str(first.get("source_speaker_id", "")))} ·
            Pair: <code>{html.escape(pair_id)}</code>
          </div>
        </header>
        <div class="refs">
          <div>
            <h3>Source Singing</h3>
            {audio(str(first.get("source_wav", "")), args.out_html)}
          </div>
          <div>
            <h3>Target Speech Ref</h3>
            {audio(str(speech.get("target_wav", "")), args.out_html) if speech else ""}
          </div>
          <div>
            <h3>Target Singing Ref</h3>
            {audio(str(singing.get("target_wav", "")), args.out_html) if singing else ""}
          </div>
        </div>
        <div class="compare">
          <div class="condition">
            <h3>Generated: Speech Prompt</h3>
            {audio(str(speech.get("audio_wav", "")), args.out_html) if speech else ""}
            <dl>
              <dt>sim target singing</dt><dd>{fmt(speech_score.get("sim_to_target_singing_prompt"))}</dd>
              <dt>sim target speech</dt><dd>{fmt(speech_score.get("sim_to_target_speech_prompt"))}</dd>
              <dt>RMS dB</dt><dd>{fmt(speech.get("rms_db"))}</dd>
            </dl>
          </div>
          <div class="condition">
            <h3>Generated: Singing Prompt</h3>
            {audio(str(singing.get("audio_wav", "")), args.out_html) if singing else ""}
            <dl>
              <dt>delta target singing</dt><dd class="delta">{delta_target_singing}</dd>
              <dt>delta target speech</dt><dd class="delta">{delta_target_speech}</dd>
              <dt>RMS dB</dt><dd>{fmt(singing.get("rms_db"))}</dd>
            </dl>
          </div>
        </div>
        <div class="decision">
          <label><input type="checkbox"> Singing prompt sounds closer to target singer</label>
          <label><input type="checkbox"> Singing prompt keeps source melody/content</label>
          <label><input type="checkbox"> Artifact or loudness issue</label>
        </div>
      </section>
"""
        )

    summary_bits = []
    for key in (
        "conditions",
        "pairs",
        "mean_singing_prompt_delta_to_target_singing",
        "mean_singing_prompt_delta_to_target_speech",
        "mean_singing_prompt_delta_to_source",
    ):
        if key in summary:
            summary_bits.append(f"<dt>{html.escape(key)}</dt><dd>{fmt(summary[key], 4)}</dd>")

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(args.title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --ink: #202124;
      --muted: #626866;
      --line: #d8d9d2;
      --panel: #ffffff;
      --accent: #116466;
      --warn: #8a5a00;
    }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0;
      font-size: 20px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 14px;
      color: var(--muted);
      letter-spacing: 0;
      text-transform: uppercase;
    }}
    audio {{
      width: 100%;
      height: 42px;
    }}
    code {{
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .top {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(280px, 420px);
      gap: 18px;
      align-items: start;
      margin-bottom: 20px;
    }}
    .note, .summary {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .note p {{
      margin: 0;
      color: var(--muted);
    }}
    .summary dl, .condition dl {{
      display: grid;
      grid-template-columns: minmax(120px, 1fr) auto;
      gap: 4px 10px;
      margin: 0;
    }}
    dt {{
      color: var(--muted);
    }}
    dd {{
      margin: 0;
      font-variant-numeric: tabular-nums;
    }}
    .pair {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-top: 14px;
    }}
    .pair header {{
      margin-bottom: 14px;
    }}
    .meta {{
      color: var(--muted);
      margin-top: 4px;
      font-size: 13px;
    }}
    .refs, .compare {{
      display: grid;
      gap: 12px;
    }}
    .refs {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-bottom: 14px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }}
    .compare {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .condition {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .condition dl {{
      margin-top: 10px;
      font-size: 13px;
    }}
    .delta {{
      color: var(--accent);
      font-weight: 650;
    }}
    .decision {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 14px;
    }}
    .decision input {{
      transform: translateY(1px);
    }}
    @media (max-width: 860px) {{
      .top, .refs, .compare {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="top">
      <div>
        <h1>{html.escape(args.title)}</h1>
        <div class="note">
          <p>Review each pair left-to-right: source singing, target references, generated speech-prompt output, generated singing-prompt output. The checkboxes are for local listening only; record final decisions in the research log or a separate note.</p>
        </div>
      </div>
      <div class="summary">
        <h3>Automatic Triage</h3>
        <dl>{''.join(summary_bits)}</dl>
      </div>
    </div>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(document, encoding="utf-8")
    print(f"wrote {args.out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
