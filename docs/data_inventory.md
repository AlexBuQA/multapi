# Опись корпуса RAG (блок 5.5)

Сгенерировано `python scripts/data_inventory.py --out docs/data_inventory.md` — числа сверяет тест `tests/unit/test_ingestion.py`. Корпус собран из базы знаний проекта (`data/help_center.jsonl`, `data/faq.md`, `data/release_notes.md`) скриптом `scripts/build_corpus.py`; что в `data/` не входит в корпус — `data/.ragignore`. Размеры — файлов с переводами строк LF, как в архиве и в git; после checkout в Windows с core.autocrlf HTML и Markdown на диске на несколько байт больше.

## Итог

- Файлов: **66** в 11 категориях (`data/<категория>/…`).
- Форматы: HTML — 26, Markdown — 17, DOCX — 14, PDF — 9.
- Общий размер: **916,4 КБ** (938436 байт).
- Документов LlamaIndex: **102** (страница PDF и раздел Markdown — отдельный документ), фрагментов при нарезке `recursive:512/64`: **102**.
- С автором в самом документе: 66, с редакцией в имени файла: 4.

## По форматам

| Формат | Ридер | Файлов | Документов | Фрагментов | Размер, КБ |
|---|---|---:|---:|---:|---:|
| HTML | HTMLTagReader: `<article>` | 26 | 26 | 26 | 53,2 |
| Markdown | MarkdownReader: раздел — документ | 17 | 30 | 30 | 16,8 |
| DOCX | DocxReader (docx2txt) | 14 | 14 | 14 | 507,4 |
| PDF | PyMuPDFReader: страница — документ | 9 | 32 | 32 | 339,0 |

## По категориям

Категория — первая папка пути (`data/billing/…` → `billing`).

| Категория | Файлов | Форматы | Авторы | Размер, КБ |
|---|---:|---|---|---:|
| `account` | 12 | HTML 11, Markdown 1 | Редакция справочного центра; Служба поддержки | 24,8 |
| `api` | 9 | Markdown 9 | Команда API; Служба поддержки | 6,4 |
| `billing` | 9 | DOCX 8, Markdown 1 | Служба поддержки; Финансовый отдел | 292,3 |
| `data-management` | 5 | Markdown 1, PDF 4 | Служба информационной безопасности; Служба поддержки | 92,1 |
| `integrations` | 3 | DOCX 2, Markdown 1 | Отдел внедрения; Служба поддержки | 73,1 |
| `mobile` | 5 | HTML 4, Markdown 1 | Редакция справочного центра; Служба поддержки | 9,2 |
| `notifications` | 7 | HTML 6, Markdown 1 | Редакция справочного центра; Служба поддержки | 13,0 |
| `releases` | 4 | PDF 4 | Продуктовая команда | 219,7 |
| `start` | 5 | HTML 5 | Редакция справочного центра | 10,3 |
| `support` | 2 | Markdown 1, PDF 1 | Служба поддержки | 29,7 |
| `team` | 5 | DOCX 4, Markdown 1 | Отдел внедрения; Служба поддержки | 145,9 |

## Файлы

| Файл | Формат | КБ | Документов | Фрагментов | Автор | Создан | Редакция |
|---|---|---:|---:|---:|---|---|---|
| `account/account-lock.html` | HTML | 2,1 | 1 | 1 | Редакция справочного центра | 2025-04-08 | — |
| `account/active-sessions.html` | HTML | 2,2 | 1 | 1 | Редакция справочного центра | 2025-04-08 | — |
| `account/change-email.html` | HTML | 1,9 | 1 | 1 | Редакция справочного центра | 2025-03-12 | — |
| `account/change-password.html` | HTML | 2,1 | 1 | 1 | Редакция справочного центра | 2025-03-12 | — |
| `account/change-phone.html` | HTML | 2,1 | 1 | 1 | Редакция справочного центра | 2025-05-06 | — |
| `account/delete-account.html` | HTML | 1,9 | 1 | 1 | Редакция справочного центра | 2025-03-12 | — |
| `account/faq-account.md` | Markdown | 1,7 | 3 | 3 | Служба поддержки | 2026-06-15 | — |
| `account/login-history.html` | HTML | 2,1 | 1 | 1 | Редакция справочного центра | 2025-04-08 | — |
| `account/lost-authenticator.html` | HTML | 2,3 | 1 | 1 | Редакция справочного центра | 2025-04-08 | — |
| `account/password-requirements_v2.html` | HTML | 2,1 | 1 | 1 | Редакция справочного центра | 2026-02-10 | 2 |
| `account/password-reset.html` | HTML | 2,2 | 1 | 1 | Редакция справочного центра | 2025-03-12 | — |
| `account/two-factor-auth.html` | HTML | 2,1 | 1 | 1 | Редакция справочного центра | 2025-04-08 | — |
| `api/api-error-401.md` | Markdown | 0,6 | 1 | 1 | Команда API | 2025-07-22 | — |
| `api/api-ip-allowlist.md` | Markdown | 0,6 | 1 | 1 | Команда API | 2026-06-30 | — |
| `api/api-key-permissions.md` | Markdown | 0,7 | 1 | 1 | Команда API | 2025-07-22 | — |
| `api/api-keys.md` | Markdown | 0,6 | 1 | 1 | Команда API | 2025-07-22 | — |
| `api/api-limits.md` | Markdown | 0,6 | 1 | 1 | Команда API | 2025-12-16 | — |
| `api/api-sandbox.md` | Markdown | 0,6 | 1 | 1 | Команда API | 2025-07-22 | — |
| `api/api-versions.md` | Markdown | 0,6 | 1 | 1 | Команда API | 2025-07-22 | — |
| `api/faq-api.md` | Markdown | 1,5 | 3 | 3 | Служба поддержки | 2026-06-15 | — |
| `api/webhooks.md` | Markdown | 0,6 | 1 | 1 | Команда API | 2026-02-24 | — |
| `billing/auto-renewal.docx` | DOCX | 36,3 | 1 | 1 | Финансовый отдел | 2025-03-12 | — |
| `billing/card-payment-fails.docx` | DOCX | 36,3 | 1 | 1 | Финансовый отдел | 2025-05-20 | — |
| `billing/change-tariff.docx` | DOCX | 36,2 | 1 | 1 | Финансовый отдел | 2025-12-16 | — |
| `billing/company-details.docx` | DOCX | 36,3 | 1 | 1 | Финансовый отдел | 2025-03-12 | — |
| `billing/faq-billing.md` | Markdown | 2,2 | 4 | 4 | Служба поддержки | 2026-06-15 | — |
| `billing/payment-documents.docx` | DOCX | 36,3 | 1 | 1 | Финансовый отдел | 2025-03-12 | — |
| `billing/payment-methods.docx` | DOCX | 36,2 | 1 | 1 | Финансовый отдел | 2025-05-20 | — |
| `billing/promo-codes.docx` | DOCX | 36,2 | 1 | 1 | Финансовый отдел | 2025-10-14 | — |
| `billing/refunds_v2.docx` | DOCX | 36,2 | 1 | 1 | Финансовый отдел | 2026-05-19 | 2 |
| `data-management/data-export.pdf` | PDF | 22,5 | 1 | 1 | Служба информационной безопасности | 2025-10-28 | — |
| `data-management/data-import.pdf` | PDF | 22,9 | 1 | 1 | Служба информационной безопасности | 2025-03-12 | — |
| `data-management/faq-data-management.md` | Markdown | 1,4 | 3 | 3 | Служба поддержки | 2026-06-15 | — |
| `data-management/personal-data-protection.pdf` | PDF | 22,7 | 1 | 1 | Служба информационной безопасности | 2025-03-12 | — |
| `data-management/restore-deleted-records.pdf` | PDF | 22,6 | 1 | 1 | Служба информационной безопасности | 2026-06-09 | — |
| `integrations/1c-integration_v2.docx` | DOCX | 36,3 | 1 | 1 | Отдел внедрения | 2025-11-25 | 2 |
| `integrations/deadline-calendar.docx` | DOCX | 36,2 | 1 | 1 | Отдел внедрения | 2026-08-18 | — |
| `integrations/faq-integrations.md` | Markdown | 0,6 | 1 | 1 | Служба поддержки | 2026-06-15 | — |
| `mobile/biometric-login.html` | HTML | 2,0 | 1 | 1 | Редакция справочного центра | 2025-08-26 | — |
| `mobile/faq-mobile.md` | Markdown | 1,2 | 2 | 2 | Служба поддержки | 2026-06-15 | — |
| `mobile/install-mobile-app.html` | HTML | 1,9 | 1 | 1 | Редакция справочного центра | 2025-08-26 | — |
| `mobile/mobile-login-problems.html` | HTML | 2,1 | 1 | 1 | Редакция справочного центра | 2025-08-26 | — |
| `mobile/offline-mode.html` | HTML | 1,9 | 1 | 1 | Редакция справочного центра | 2026-04-28 | — |
| `notifications/emails-not-arriving.html` | HTML | 2,1 | 1 | 1 | Редакция справочного центра | 2025-03-12 | — |
| `notifications/faq-notifications.md` | Markdown | 1,2 | 2 | 2 | Служба поддержки | 2026-06-15 | — |
| `notifications/notification-types.html` | HTML | 1,9 | 1 | 1 | Редакция справочного центра | 2025-03-12 | — |
| `notifications/push-notifications.html` | HTML | 2,0 | 1 | 1 | Редакция справочного центра | 2025-08-26 | — |
| `notifications/quiet-hours.html` | HTML | 1,9 | 1 | 1 | Редакция справочного центра | 2026-04-07 | — |
| `notifications/telegram-notifications.html` | HTML | 1,9 | 1 | 1 | Редакция справочного центра | 2025-06-17 | — |
| `notifications/unsubscribe.html` | HTML | 2,0 | 1 | 1 | Редакция справочного центра | 2025-03-12 | — |
| `releases/release-notes_2025-h1.pdf` | PDF | 29,4 | 4 | 4 | Продуктовая команда | 2025-06-17 | — |
| `releases/release-notes_2025-h2.pdf` | PDF | 78,4 | 6 | 6 | Продуктовая команда | 2025-12-16 | — |
| `releases/release-notes_2026-h1.pdf` | PDF | 35,2 | 9 | 9 | Продуктовая команда | 2026-06-30 | — |
| `releases/release-notes_2026-h2.pdf` | PDF | 76,7 | 5 | 5 | Продуктовая команда | 2026-10-06 | — |
| `start/home-page.html` | HTML | 2,1 | 1 | 1 | Редакция справочного центра | 2025-03-12 | — |
| `start/language-and-timezone.html` | HTML | 2,0 | 1 | 1 | Редакция справочного центра | 2025-03-12 | — |
| `start/login-yandex-gosuslugi.html` | HTML | 2,1 | 1 | 1 | Редакция справочного центра | 2026-03-17 | — |
| `start/login.html` | HTML | 2,1 | 1 | 1 | Редакция справочного центра | 2025-03-12 | — |
| `start/registration.html` | HTML | 2,0 | 1 | 1 | Редакция справочного центра | 2025-03-12 | — |
| `support/faq-support.md` | Markdown | 1,2 | 2 | 2 | Служба поддержки | 2026-06-15 | — |
| `support/support-regulation_v2.3.pdf` | PDF | 28,6 | 4 | 4 | Служба поддержки | 2026-01-20 | 2.3 |
| `team/faq-team.md` | Markdown | 1,1 | 2 | 2 | Служба поддержки | 2026-06-15 | — |
| `team/invite-members.docx` | DOCX | 36,2 | 1 | 1 | Отдел внедрения | 2025-03-12 | — |
| `team/member-roles.docx` | DOCX | 36,2 | 1 | 1 | Отдел внедрения | 2025-09-30 | — |
| `team/remove-member.docx` | DOCX | 36,2 | 1 | 1 | Отдел внедрения | 2025-09-30 | — |
| `team/transfer-ownership.docx` | DOCX | 36,2 | 1 | 1 | Отдел внедрения | 2026-07-21 | — |
