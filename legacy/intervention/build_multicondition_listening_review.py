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


CONDITION_ORDER = [
    "baseline_all_speech",
    "oracle_all_singing",
    "singing_prompt_seq_only",
    "singing_mel_only",
    "singing_style_only",
    "singing_prompt_seq_mel",
    "singing_prompt_seq_style",
    "singing_mel_style",
]


def rel(path: str, base: Path) -> str:
    target = Path(path)
    try:
        return target.relative_to(base.parent).as_posix()
    except ValueError:
        return target.as_uri()


def audio(path: str, base: Path) -> str:
    return f'<audio controls preload="none" src="{html.escape(rel(path, base), quote=True)}"></audio>'


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def read_summary(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local HTML page for multi-condition Seed-VC audio review.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--out-html", type=Path, required=True)
    parser.add_argument("--title", default="Multi-condition Listening Review")
    args = parser.parse_args()

    try:
        rows = read_table(args.manifest)
        if not rows:
            raise ExperimentError("manifest is empty")
        summary = read_summary(args.summary)
    except (ExperimentError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_pair_id"])].append(row)

    delta_lookup = {
        f"{row['target_pair_id']}::{row['condition']}": row
        for row in summary.get("pair_deltas", [])
    }
    condition_summary = summary.get("condition_summary", {})

    summary_rows = []
    for condition in CONDITION_ORDER:
        item = condition_summary.get(condition)
        if not item:
            continue
        summary_rows.append(
            f"<tr><td><code>{html.escape(condition)}</code></td>"
            f"<td>{fmt(item.get('mean_delta_to_target_singing'))}</td>"
            f"<td>{fmt(item.get('mean_delta_to_target_speech'))}</td>"
            f"<td>{fmt(item.get('mean_delta_to_source'))}</td>"
            f"<td>{fmt(item.get('mean_delta_rms_db'))}</td>"
            f"<td>{html.escape(str(item.get('positive_delta_to_target_singing', '')))}"
            f"/{html.escape(str(item.get('pairs', '')))}</td></tr>"
        )

    cards = []
    for idx, (pair_id, group) in enumerate(sorted(grouped.items()), 1):
        by_condition = {str(row["condition"]): row for row in group}
        first = group[0]
        condition_cards = []
        for condition in CONDITION_ORDER:
            row = by_condition.get(condition)
            if not row:
                continue
            delta = delta_lookup.get(f"{pair_id}::{condition}", {})
            condition_cards.append(
                f"""
          <div class="condition">
            <h3>{html.escape(condition)}</h3>
            {audio(str(row.get("audio_wav", "")), args.out_html)}
            <dl>
              <dt>d target singing</dt><dd>{fmt(delta.get("delta_to_target_singing"))}</dd>
              <dt>d target speech</dt><dd>{fmt(delta.get("delta_to_target_speech"))}</dd>
              <dt>d source</dt><dd>{fmt(delta.get("delta_to_source"))}</dd>
              <dt>d RMS dB</dt><dd>{fmt(delta.get("delta_rms_db"))}</dd>
            </dl>
          </div>
"""
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
          <div><h3>Source Singing</h3>{audio(str(first.get("source_wav", "")), args.out_html)}</div>
          <div><h3>Target Speech Ref</h3>{audio(str(first.get("target_speech_wav", "")), args.out_html)}</div>
          <div><h3>Target Singing Ref</h3>{audio(str(first.get("target_singing_wav", "")), args.out_html)}</div>
        </div>
        <div class="conditions">
          {''.join(condition_cards)}
        </div>
        <div class="decision">
          <label><input type="checkbox"> mel/style changes identity</label>
          <label><input type="checkbox"> mel/style mostly changes technique/domain</label>
          <label><input type="checkbox"> prompt_seq audible</label>
          <label><input type="checkbox"> artifact/loudness issue</label>
        </div>
      </section>
"""
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(args.title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f6f3;
      --ink: #202124;
      --muted: #626866;
      --line: #d8d9d2;
      --panel: #ffffff;
    }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    h1 {{ margin: 0 0 12px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0; font-size: 20px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); letter-spacing: 0; }}
    audio {{ width: 100%; height: 42px; }}
    code {{ font-size: 12px; overflow-wrap: anywhere; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 18px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      font-variant-numeric: tabular-nums;
      font-size: 13px;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .pair {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-top: 14px;
    }}
    .meta {{ color: var(--muted); margin-top: 4px; font-size: 13px; }}
    .refs, .conditions {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .condition {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }}
    dl {{
      display: grid;
      grid-template-columns: minmax(110px, 1fr) auto;
      gap: 3px 8px;
      margin: 8px 0 0;
      font-size: 12px;
    }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; font-variant-numeric: tabular-nums; }}
    .decision {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 820px) {{
      .refs, .conditions {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(args.title)}</h1>
    <table>
      <thead>
        <tr>
          <th>condition</th><th>d target singing</th><th>d target speech</th>
          <th>d source</th><th>d RMS dB</th><th>positive</th>
        </tr>
      </thead>
      <tbody>
        {''.join(summary_rows)}
      </tbody>
    </table>
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
