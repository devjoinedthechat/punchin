"""A self-contained page that plays two calls side by side, with the fork marked.

A table says a call got worse. This lets somebody hear it. The page is a single HTML file with the
audio embedded, so it can be opened from disk or attached to a bug report without a server.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

from punchin.call import Call

MAX_EMBED_BYTES = 40 * 1024 * 1024


def _audio(turn_audio: str | None) -> str | None:
    """The turn's wav as a data URI, or None when it was never spoken or has gone missing."""
    if not turn_audio:
        return None
    path = Path(turn_audio)
    if not path.exists():
        return None
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode()


def _turns(call: Call, forked_at: int | None) -> list[dict[str, Any]]:
    return [
        {
            "index": turn.index,
            "who": turn.speaker,
            "said": turn.spoken,
            "heard": turn.heard if turn.heard and turn.heard.strip() != turn.spoken.strip() else None,
            "audio": _audio(turn.audio),
            "ms": turn.audio_ms or turn.model_ms,
            "tools": [
                {"tool": c.tool, "arguments": c.arguments, "failed": bool(c.error)} for c in turn.tool_calls
            ],
            "replayed": forked_at is not None and turn.index < forked_at,
        }
        for turn in call.turns
    ]


def _side(call: Call, label: str) -> dict[str, Any]:
    forked_at = call.notes.get("forked_at")
    return {
        "label": label,
        "id": call.id,
        "scenario": call.scenario,
        "agent": call.agent,
        "forkedAt": forked_at,
        "booked": call.bookings,
        "endedBy": call.notes.get("ended_by"),
        "turns": _turns(call, forked_at if isinstance(forked_at, int) else None),
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg:#fbfaf8; --ink:#1a1a1a; --dim:#6b6b6b; --line:#e2ded8;
    --agent:#1f4e5f; --kunde:#7a3e12; --bad:#b3261e; --ok:#2e7d32; --fork:#c2410c; --card:#fff;
  }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --bg:#16151a; --ink:#ece9e4; --dim:#9b968f; --line:#312e38;
    --agent:#7fb3c4; --kunde:#d79a6a; --bad:#f2837c; --ok:#84c98a; --fork:#fb923c; --card:#1e1d24;
  } }
  :root[data-theme="dark"] {
    --bg:#16151a; --ink:#ece9e4; --dim:#9b968f; --line:#312e38;
    --agent:#7fb3c4; --kunde:#d79a6a; --bad:#f2837c; --ok:#84c98a; --fork:#fb923c; --card:#1e1d24;
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--ink); padding:24px 16px 64px;
         font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif }
  header { max-width:1180px; margin:0 auto 20px }
  h1 { font-size:20px; margin:0 0 4px; font-weight:650 }
  .sub { color:var(--dim); font-size:13px }
  .pair { max-width:1180px; margin:0 auto; display:grid; gap:16px;
          grid-template-columns:repeat(2,minmax(0,1fr)) }
  @media (max-width:760px) { .pair { grid-template-columns:1fr } }
  .col { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px }
  .col h2 { font-size:14px; margin:0 0 2px; font-weight:650 }
  .meta { color:var(--dim); font-size:12px; margin-bottom:10px; font-variant-numeric:tabular-nums }
  .turn { padding:7px 0 7px 10px; border-left:3px solid transparent; margin-bottom:2px }
  .turn.agent { border-left-color:var(--agent) }
  .turn.kunde { border-left-color:var(--kunde) }
  .turn.replayed { opacity:.55 }
  .turn.forkpoint { border-left-color:var(--fork); border-radius:0 6px 6px 0;
                    background:color-mix(in srgb,var(--fork) 9%,transparent) }
  .who { font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--dim);
         font-variant-numeric:tabular-nums }
  .said { margin:2px 0 0; white-space:pre-wrap }
  .heard { margin:3px 0 0; font-size:13px; color:var(--bad) }
  .heard b { font-weight:600; font-style:normal }
  .tool { font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--dim); margin-top:3px }
  .tool.failed { color:var(--bad) }
  button.play { font:inherit; font-size:12px; margin-top:5px; cursor:pointer; color:var(--ink);
                background:transparent; border:1px solid var(--line); border-radius:999px; padding:2px 10px }
  button.play:hover { border-color:var(--ink) }
  .foot { margin-top:12px; padding-top:10px; border-top:1px solid var(--line); font-size:13px }
  .ok { color:var(--ok) } .bad { color:var(--bad) }
  .banner { font-size:12px; color:var(--fork); margin:8px 0 2px; letter-spacing:.02em }
</style></head>
<body>
<header><h1>__TITLE__</h1><div class="sub">__SUB__</div></header>
<div class="pair" id="pair"></div>
<script>
const DATA = __DATA__;
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function turnEl(t, forkedAt) {
  const d = document.createElement('div');
  const marks = (t.replayed ? ' replayed' : '') + (t.index === forkedAt ? ' forkpoint' : '');
  d.className = 'turn ' + (t.who === 'agent' ? 'agent' : 'kunde') + marks;
  const ms = t.ms ? ` · ${t.ms} ms` : '';
  const who = t.who === 'agent' ? 'Agent' : 'Kunde';
  let h = `<div class="who">${String(t.index).padStart(2,'0')} ${who}${ms}</div>`;
  h += `<p class="said">${esc(t.said)}</p>`;
  if (t.heard) h += `<p class="heard"><b>heard:</b> ${esc(t.heard)}</p>`;
  for (const c of t.tools) {
    h += `<div class="tool${c.failed ? ' failed' : ''}">→ ${esc(c.tool)}(${esc(JSON.stringify(c.arguments))})`
       + `${c.failed ? ' ERROR' : ''}</div>`;
  }
  d.innerHTML = h;
  if (t.audio) {
    const b = document.createElement('button');
    b.className = 'play'; b.textContent = '▶ play';
    const a = new Audio(t.audio);
    b.onclick = () => {
      document.querySelectorAll('audio').forEach(x => x.pause());
      a.currentTime = 0; a.play();
    };
    d.appendChild(b); d.appendChild(a);
  }
  return d;
}

for (const side of DATA.sides) {
  const col = document.createElement('div');
  col.className = 'col';
  const booked = side.booked.length
    ? side.booked.map(b => `${b.reg} ${b.date} ${b.time}${b.note ? ' · ' + b.note : ''}`).join(', ')
    : 'nothing booked';
  col.innerHTML = `<h2>${esc(side.label)}</h2>`
    + `<div class="meta">${esc(side.agent)} · ended by ${esc(side.endedBy ?? '?')}</div>`;
  if (side.forkedAt !== null && side.forkedAt !== undefined) {
    const n = document.createElement('div');
    n.className = 'banner';
    n.textContent = `turns 0 to ${side.forkedAt - 1} served from the recording`
      + ` \u00b7 forked at ${side.forkedAt}`;
    col.appendChild(n);
  }
  for (const t of side.turns) col.appendChild(turnEl(t, side.forkedAt));
  const f = document.createElement('div');
  f.className = 'foot';
  f.innerHTML = `<span class="${side.booked.length ? 'ok' : 'bad'}">${esc(booked)}</span>`;
  col.appendChild(f);
  document.getElementById('pair').appendChild(col);
}
</script>
</body></html>
"""


def build(before: Call, after: Call, *, title: str | None = None) -> str:
    """The page, as one string. Audio is embedded, so the file stands alone."""
    data = {"sides": [_side(before, "Before"), _side(after, "After")]}
    # UTF-8 rather than \u escapes: the page declares the charset, Danish stays readable in the source,
    # and the file is smaller. `</` is escaped because nothing inside a <script> may close it early.
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    if len(blob) > MAX_EMBED_BYTES:
        raise ValueError(
            f"the two calls carry {len(blob) / 1e6:.0f} MB of audio, over the {MAX_EMBED_BYTES / 1e6:.0f} MB "
            f"a single page should hold; record them with fewer turns or without --audio"
        )
    heading = title or f"{before.scenario}: before and after"
    forked = after.notes.get("forked_at")
    sub = f"{before.id} → {after.id}"
    if forked is not None:
        sub += f" · forked at turn {forked}"
    return (
        PAGE.replace("__DATA__", blob)
        .replace("__TITLE__", html.escape(heading))
        .replace("__SUB__", html.escape(sub))
    )


def write(before: Call, after: Call, dest: Path, *, title: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(build(before, after, title=title))
    return dest
