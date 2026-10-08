"""
Какую долю промптов garak останавливает проверка входа — без модели и без сервиса (блок 3.8).

Запуск — в окружении с garak (оно отдельное, см. README, «Блок 3.8»), из корня проекта:

    .venv-garak\\Scripts\\python eval/security/validator_coverage.py          # Windows
    .venv-garak/bin/python eval/security/validator_coverage.py              # Linux, macOS

Промпты берутся из самих проб garak (seed 42, как в прогонах baseline и after) и
проверяются validate_input из app/services/security/input_validator.py. Пробы в двух
группах:
- «подбор» — три пробы прогонов baseline/after: по их промптам подбирались шаблоны,
  поэтому высокая доля здесь ожидаема и мало говорит о новых атаках;
- «контроль» — пробы, которых при подборе не было. По ним видно, как валидатор
  работает на незнакомых атаках. HijackKillHumans и HijackLongPrompt используют те же
  шаблоны атаки, что HijackHateHumans, — они не независимы; DanInTheWild — 256
  джейлбрейков «из жизни», честная оценка обобщения.

Ниже — ложные срабатывания: вопросы golden dataset (ожидается только faq_023), статьи
руководства и эталонные ответы.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.security.input_validator import validate_input  # noqa: E402

TUNED = ["promptinject.HijackHateHumans", "encoding.InjectBase64", "dan.Ablation_Dan_11_0"]
CONTROL = ["promptinject.HijackKillHumans", "promptinject.HijackLongPrompt", "dan.DanInTheWild",
           "goodside.Tag", "encoding.InjectHex", "encoding.InjectBase32", "encoding.InjectROT13"]
GENERATOR_NAME = "multapi-support"     # name из rest_config.json — его DAN-пробы подставляют в текст


def probe_prompts(name: str, seed: int) -> list[str]:
    from garak import _config, _plugins

    if not getattr(_config, "loaded", False):
        _config.load_base_config()
    _config.run.seed = seed
    random.seed(seed)
    probe = _plugins.load_plugin(f"probes.{name}")
    texts = []
    for prompt in probe.prompts:
        text = prompt if isinstance(prompt, str) else getattr(prompt, "text", None) or str(prompt)
        texts.append(text.replace("{generator.name}", GENERATOR_NAME))
    return texts


def coverage(texts: list[str]) -> tuple[int, Counter]:
    rules = Counter(validate_input(t).rule or "прошло" for t in texts)
    return len(texts) - rules["прошло"], rules


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Доля промптов garak, которые останавливает validate_input")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    print("| Группа | Проба | Промптов | Остановлено | Доля | По правилам |")
    print("|---|---|---:|---:|---:|---|")
    for group, names in (("подбор", TUNED), ("контроль", CONTROL)):
        for name in names:
            texts = probe_prompts(name, args.seed)
            blocked, rules = coverage(texts)
            by_rule = ", ".join(f"{rule} {n}" for rule, n in rules.most_common() if rule != "прошло") or "—"
            print(f"| {group} | `{name}` | {len(texts)} | {blocked} | {blocked / len(texts):.1%} | {by_rule} |")

    golden = json.loads((ROOT / "eval" / "golden_dataset.json").read_text(encoding="utf-8"))["items"]
    kb = json.loads((ROOT / "data" / "knowledge_base.json").read_text(encoding="utf-8"))["articles"]
    print("\nЛожные срабатывания:")
    flagged = [i["id"] for i in golden if not validate_input(i["question"]).ok]
    print(f"  вопросы golden dataset: {len(flagged)} из {len(golden)} — {', '.join(flagged) or 'нет'} (ожидается faq_023)")
    for label, texts in (("статьи руководства", [a["text"] for a in kb]),
                         ("эталонные ответы", [i["expected_answer"] for i in golden])):
        print(f"  {label}: {coverage(texts)[0]} из {len(texts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
