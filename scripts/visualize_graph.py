"""
Mermaid-схемы агента на LangGraph (блок 6.3): custom_graph и prebuilt_graph из app/services/agent_graph.py.

    python scripts/visualize_graph.py            # docs/agent-graph-custom.mmd, docs/agent-graph-prebuilt.mmd
    python scripts/visualize_graph.py --png      # и docs/agent-graph.png (рендер через mermaid.ink — нужна сеть)

Схема строится по скомпилированному графу (graph.get_graph().draw_mermaid()), модель при этом не
вызывается. Файлы .mmd открываются в https://mermaid.live. PNG — по желанию: draw_mermaid_png() по
умолчанию отправляет схему (только имена узлов и рёбра) в сервис mermaid.ink; без сети остаются .mmd.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs"


def save(out_dir: Path = OUT, *, png: bool = False) -> list[Path]:
    from app.services.agent_graph import custom_graph, prebuilt_graph

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, graph in (("custom", custom_graph), ("prebuilt", prebuilt_graph)):
        path = out_dir / f"agent-graph-{name}.mmd"
        path.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
        saved.append(path)
    if png:
        path = out_dir / "agent-graph.png"
        try:
            path.write_bytes(custom_graph.get_graph().draw_mermaid_png())
            saved.append(path)
        except Exception as exc:  # noqa: BLE001 — нет сети или рендерера: остаются .mmd
            print(f"PNG не сохранён ({type(exc).__name__}: {exc}) — остаются .mmd", file=sys.stderr)
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mermaid-схемы custom_graph и prebuilt_graph → docs/")
    parser.add_argument("--png", action="store_true", help="ещё docs/agent-graph.png через mermaid.ink (нужна сеть)")
    parser.add_argument("--out", type=Path, default=OUT, help="папка для файлов")
    args = parser.parse_args(argv)
    for path in save(args.out, png=args.png):
        print(f"saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
