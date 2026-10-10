"""
Admin API (блок 4.4): /chats/admin/* под заголовком X-Admin-Token.

    app/admin/deps.py      require_admin — проверка X-Admin-Token == ADMIN_TOKEN из .env
    app/admin/schemas.py   StatsOut, UserOut, BroadcastIn/Out, BroadcastClaimOut, BroadcastResultIn
    app/admin/routes.py    GET /stats, GET /users, POST /broadcast (+ claim и result для бота)

Полноценная аутентификация (JWT, роли) — за рамками задания: один общий токен в заголовке.
"""
