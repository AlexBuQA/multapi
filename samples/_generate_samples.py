"""
Генератор образцовых входных изображений для демо Vision API.
Создаёт три файла разного типа: фото (имитация), скриншот UI, график.

Запускается один раз при сборке проекта. Картинки коммитятся в репозиторий,
поэтому повторный запуск обычно не требуется.

    python samples/_generate_samples.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Пытаемся взять системный TTF, иначе — встроенный bitmap-шрифт."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_photo(path: str) -> None:
    """Имитация фотографии: закат над морем (градиент + солнце + блики)."""
    w, h = 960, 640
    img = Image.new("RGB", (w, h))
    px = img.load()
    # Небо: вертикальный градиент от оранжевого к фиолетовому
    top = np.array([255, 150, 60])
    bottom = np.array([60, 30, 90])
    for y in range(h):
        t = y / h
        color = (top * (1 - t) + bottom * t).astype(int)
        for x in range(w):
            px[x, y] = tuple(color)
    draw = ImageDraw.Draw(img)
    # Солнце
    cx, cy, r = w // 2, int(h * 0.42), 70
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 230, 150))
    # Море снизу с горизонтальными бликами
    sea_top = int(h * 0.62)
    for y in range(sea_top, h):
        t = (y - sea_top) / (h - sea_top)
        base = np.array([40, 25, 70]) * (1 - t) + np.array([20, 12, 40]) * t
        for x in range(w):
            shimmer = 35 * np.sin(x / 18.0 + y / 4.0) * (1 - t)
            color = np.clip(base + shimmer, 0, 255).astype(int)
            px[x, y] = tuple(color)
    # Дорожка от солнца
    for y in range(sea_top, h):
        spread = int(8 + (y - sea_top) * 0.5)
        for x in range(cx - spread, cx + spread):
            if 0 <= x < w:
                cur = np.array(px[x, y])
                px[x, y] = tuple(np.clip(cur + 60, 0, 255).astype(int))
    img.save(path, quality=88)


def make_screenshot(path: str) -> None:
    """Имитация скриншота интерфейса: окно с тулбаром, формой и кнопкой."""
    w, h = 900, 600
    img = Image.new("RGB", (w, h), (243, 244, 248))
    draw = ImageDraw.Draw(img)
    f_title = _font(20)
    f_label = _font(16)
    f_small = _font(14)

    # Шапка окна
    draw.rectangle([0, 0, w, 56], fill=(46, 26, 120))
    draw.ellipse([18, 22, 30, 34], fill=(255, 95, 86))
    draw.ellipse([40, 22, 52, 34], fill=(255, 189, 46))
    draw.ellipse([62, 22, 74, 34], fill=(39, 201, 63))
    draw.text((110, 18), "Личный кабинет — Настройки профиля", font=f_title, fill=(255, 255, 255))

    # Боковое меню
    draw.rectangle([0, 56, 210, h], fill=(255, 255, 255))
    for i, item in enumerate(["Профиль", "Безопасность", "Уведомления", "Биллинг", "API-ключи"]):
        y = 90 + i * 46
        if i == 0:
            draw.rectangle([0, y - 10, 210, y + 26], fill=(237, 233, 255))
        draw.text((28, y), item, font=f_label, fill=(46, 26, 120) if i == 0 else (90, 90, 110))

    # Контент: форма
    draw.text((250, 90), "Имя пользователя", font=f_small, fill=(110, 110, 130))
    draw.rounded_rectangle([250, 112, 760, 150], 8, outline=(210, 210, 225), width=2, fill=(255, 255, 255))
    draw.text((266, 122), "ivan_petrov", font=f_label, fill=(40, 40, 60))

    draw.text((250, 172), "E-mail", font=f_small, fill=(110, 110, 130))
    draw.rounded_rectangle([250, 194, 760, 232], 8, outline=(210, 210, 225), width=2, fill=(255, 255, 255))
    draw.text((266, 204), "i.petrov@example.com", font=f_label, fill=(40, 40, 60))

    # Тоггл
    draw.text((250, 262), "Двухфакторная аутентификация", font=f_small, fill=(110, 110, 130))
    draw.rounded_rectangle([250, 286, 300, 314], 14, fill=(39, 201, 63))
    draw.ellipse([278, 288, 298, 312], fill=(255, 255, 255))
    draw.text((312, 290), "Включена", font=f_small, fill=(39, 150, 63))

    # Кнопка
    draw.rounded_rectangle([250, 360, 420, 404], 10, fill=(46, 26, 120))
    draw.text((292, 372), "Сохранить", font=f_label, fill=(255, 255, 255))

    # Уведомление-плашка об ошибке
    draw.rounded_rectangle([250, 430, 760, 478], 8, fill=(255, 235, 235), outline=(229, 115, 115), width=2)
    draw.text((266, 446), "Ошибка: пароль должен содержать не менее 8 символов", font=f_small, fill=(190, 40, 40))

    img.save(path)


def make_chart(path: str) -> None:
    """График: выручка и прибыль по кварталам (бизнес-визуализация)."""
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    revenue = [4.2, 5.1, 4.8, 6.7]   # млн руб.
    profit = [0.9, 1.3, 1.1, 2.0]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    x = np.arange(len(quarters))
    width = 0.38
    ax.bar(x - width / 2, revenue, width, label="Выручка", color="#2e1a78")
    ax.bar(x + width / 2, profit, width, label="Прибыль", color="#8b7bd8")

    ax.set_title("Финансовые показатели по кварталам, 2025 (млн ₽)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(quarters)
    ax.set_ylabel("млн ₽")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for i, (r, p) in enumerate(zip(revenue, profit)):
        ax.text(i - width / 2, r + 0.1, f"{r}", ha="center", fontsize=9)
        ax.text(i + width / 2, p + 0.1, f"{p}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    photo = os.path.join(SAMPLES_DIR, "photo.jpg")
    screenshot = os.path.join(SAMPLES_DIR, "screenshot.png")
    chart = os.path.join(SAMPLES_DIR, "chart.png")
    make_photo(photo)
    make_screenshot(screenshot)
    make_chart(chart)
    print("Создано:")
    for p in (photo, screenshot, chart):
        print(f"  {p}  ({os.path.getsize(p)} байт)")


if __name__ == "__main__":
    main()
