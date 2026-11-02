"""Server-rendered HTML for the admin page. Every dynamic value goes through :func:`e` (HTML escaping).

No inline scripts, styles or event handlers (the CSP forbids them) and no external resources: behaviour lives in
``admin_static/admin.js``, which calls the JSON API with the CSRF token from the ``csrf-token`` meta tag.
Viewing works without JavaScript; creating, renaming and revoking keys and managing members need it.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from html import escape
from typing import Any
from urllib.parse import urlencode

from keyservice import Viewer

APP_NAME = "Decision API admin"


def e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _t(ts: str | None) -> str:
    """A UTC timestamp for a table cell (``—`` when unset)."""
    if not ts:
        return '<span class="muted">—</span>'
    return f'<time datetime="{e(ts)}">{e(ts.replace("T", " ").replace("Z", " UTC"))}</time>'


def layout(title: str, body: str, viewer: Viewer | None = None, csrf: str | None = None, current: str = "") -> str:
    meta = f'<meta name="csrf-token" content="{e(csrf)}">\n' if csrf else ""
    nav = ""
    if viewer is not None:
        links = [("/admin", "My keys")]
        if viewer.is_owner:
            links += [("/admin/keys", "All keys"), ("/admin/usage", "Usage"), ("/admin/members", "Members"),
                      ("/admin/audit", "Audit log")]  # fmt: skip
        here = ' aria-current="page"'
        items = "".join(
            f'<li><a href="{href}"{here if href == current else ""}>{e(label)}</a></li>' for href, label in links
        )
        nav = (
            f'<header><nav aria-label="Admin"><ul>{items}</ul>'
            f'<p class="who">{e(viewer.email)} ({e(viewer.role)}) '
            f'<button type="button" data-action="logout">Sign out</button></p></nav></header>\n'
        )
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{e(title)} · {APP_NAME}</title>\n"
        '<link rel="stylesheet" href="/admin/static/admin.css">\n'
        f"{meta}"
        '<script src="/admin/static/admin.js" defer></script>\n'
        "</head>\n<body>\n"
        '<a class="skip" href="#main">Skip to content</a>\n'
        f"{nav}"
        f'<main id="main">\n<h1>{e(title)}</h1>\n'
        '<p id="status" role="status" aria-live="polite"></p>\n'
        f"{body}\n</main>\n</body>\n</html>\n"
    )


def message(title: str, text: str, link: tuple[str, str] | None = None) -> str:
    extra = f' <a href="{e(link[0])}">{e(link[1])}</a>.' if link else ""
    return layout(title, f"<p>{e(text)}{extra}</p>")


def signed_out() -> str:
    body = (
        "<p>Manage your API keys for the decision API and see their usage.</p>\n"
        '<p><a class="button" href="/admin/auth/login">Sign in with Google</a></p>\n'
        '<p class="muted">Only the owner and invited members can sign in.</p>'
    )
    return layout(APP_NAME, body)


# ---------------------------------------------------------------- keys


def keys_table(keys: list[dict], *, show_owner: bool, caption: str) -> str:
    if not keys:
        return f"<p>{e(caption)}: none yet.</p>"
    head = '<th scope="col">Name</th><th scope="col">Prefix</th>'
    if show_owner:
        head += '<th scope="col">Owner</th>'
    head += ('<th scope="col">Created</th><th scope="col">Expires</th><th scope="col">Last used</th>'
             '<th scope="col">Status</th><th scope="col">Actions</th>')  # fmt: skip
    rows = []
    for k in keys:
        kid = e(k["id"])
        cells = f"<td>{e(k['name'])}</td><td><code>{e(k['prefix'])}</code></td>"
        if show_owner:
            cells += f"<td>{e(k['owner_email'])}</td>"
        cells += (
            f"<td>{_t(k['created_at'])}</td><td>{_t(k['expires_at'])}</td><td>{_t(k['last_used_at'])}</td>"
            f'<td><span class="status status-{e(k["status"])}">{e(k["status"])}</span></td>'
        )
        if k["status"] == "revoked":
            actions = '<span class="muted">—</span>'
        else:
            actions = (
                f'<form class="rename" data-key="{kid}">'
                f'<label for="rename-{kid}" class="visually-hidden">New name for {e(k["name"])}</label>'
                f'<input id="rename-{kid}" name="name" maxlength="64" required value="{e(k["name"])}">'
                '<button type="submit">Rename</button></form> '
                f'<button type="button" class="danger" data-action="revoke" data-key="{kid}" '
                f'data-name="{e(k["name"])}">Revoke</button>'
            )
        cells += f"<td>{actions}</td>"
        rows.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="table-wrap"><table><caption>{e(caption)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


CREATE_FORM = """<section aria-labelledby="create-h">
<h2 id="create-h">Create a key</h2>
<form id="create-key">
<p><label for="key-name">Name</label>
<input id="key-name" name="name" required maxlength="64" autocomplete="off"></p>
<p><label for="key-expires">Expires (optional; the key stops working at the end of that day, UTC)</label>
<input id="key-expires" name="expires" type="date"></p>
<p><button type="submit">Create key</button></p>
</form>
<noscript><p>Creating, renaming and revoking keys needs JavaScript.</p></noscript>
<div id="new-key" class="new-key" hidden>
<h3>Your new key</h3>
<p><strong>Copy it now. You won't see it again:</strong> only a hash is stored.</p>
<p><code id="new-key-value"></code></p>
<p><button type="button" data-action="copy">Copy key</button>
<button type="button" data-action="reload">I saved it</button></p>
</div>
</section>"""


# ---------------------------------------------------------------- usage


def usage_chart(report: dict, label: str) -> str:
    """An inline SVG bar chart of requests per day over the report's range, plus a table of the days with data."""
    start, end = date.fromisoformat(report["from"]), date.fromisoformat(report["to"])
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    by_day = {r["date"]: r for r in report["rows"]}
    peak = max([r["requests"] for r in report["rows"]] or [0])
    bar_w, gap, height = 10, 2, 100
    bars = []
    for i, d in enumerate(days):
        n = by_day.get(d.isoformat(), {}).get("requests", 0)
        h = round(height * n / peak) if peak else 0
        x = i * (bar_w + gap)
        bars.append(
            f'<rect class="bar" x="{x}" y="{height - h}" width="{bar_w}" height="{max(h, 0)}">'
            f"<title>{d.isoformat()}: {n} requests</title></rect>"
        )
    width = len(days) * (bar_w + gap)
    svg = (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
        f'aria-label="{e(label)}: requests per day from {e(report["from"])} to {e(report["to"])}, '
        f'peak {peak}">{"".join(bars)}</svg>'
    )
    total = report["total"]
    summary = (
        f"<p>{e(total['requests'])} requests, {e(total['errors'])} errors "
        f"({e(total['client_errors'])} client, {e(total['server_errors'])} server)"
        + (f", average latency {e(round(total['avg_latency_ms']))} ms" if total["avg_latency_ms"] is not None else "")
        + f", {e(report['from'])} to {e(report['to'])} (UTC).</p>"
    )
    if not report["rows"]:
        return summary + f"<p>No requests from {e(report['from'])} to {e(report['to'])}.</p>"
    rows = "".join(
        f"<tr><td>{e(r['date'])}</td><td>{e(r['requests'])}</td><td>{e(r['errors'])}</td>"
        f"<td>{e(round(r['avg_latency_ms'])) if r['avg_latency_ms'] is not None else '—'}</td></tr>"
        for r in report["rows"]
    )
    table = (
        f'<div class="table-wrap"><table><caption>{e(label)} by day</caption><thead><tr><th scope="col">Date</th>'
        '<th scope="col">Requests</th><th scope="col">Errors</th><th scope="col">Avg latency (ms)</th></tr>'
        f"</thead><tbody>{rows}</tbody></table></div>"
    )
    return summary + svg + table


def home(viewer: Viewer, csrf: str, keys: list[dict], usage: dict) -> str:
    body = (
        '<section aria-labelledby="keys-h"><h2 id="keys-h">Your keys</h2>'
        "<p>Send a key as <code>Authorization: Bearer &lt;key&gt;</code> to the <code>/v1</code> API.</p>"
        f"{keys_table(keys, show_owner=False, caption='Your API keys')}</section>\n"
        f"{CREATE_FORM}\n"
        '<section aria-labelledby="usage-h"><h2 id="usage-h">Usage of your keys (last 30 days)</h2>'
        f"{usage_chart(usage, 'Usage of your keys')}</section>"
    )
    return layout("My keys", body, viewer, csrf, current="/admin")


def _user_select(members: list[dict], selected: str | None, field_id: str) -> str:
    options = ['<option value="">Everyone</option>'] + [
        f'<option value="{e(m["email"])}"{" selected" if m["email"] == selected else ""}>{e(m["email"])}</option>'
        for m in members
    ]
    return f'<label for="{field_id}">User</label> <select id="{field_id}" name="user">{"".join(options)}</select>'


def all_keys(viewer: Viewer, csrf: str, keys: list[dict], members: list[dict], selected: str | None) -> str:
    body = (
        f'<form method="get" action="/admin/keys" class="filter">{_user_select(members, selected, "user-filter")} '
        '<button type="submit">Filter</button></form>\n'
        f"{keys_table(keys, show_owner=True, caption='Keys of ' + (selected or 'everyone'))}"
    )
    return layout("All keys", body, viewer, csrf, current="/admin/keys")


def usage_page(viewer: Viewer, csrf: str, by_day: dict, by_key: dict, members: list[dict], selected: str | None):
    rows = "".join(
        f"<tr><td><code>{e(r['key_id'])}</code></td><td>{e(r['key_name'])}</td><td>{e(r['owner_email'])}</td>"
        f"<td>{e(r['requests'])}</td><td>{e(r['errors'])}</td>"
        f"<td>{e(round(r['avg_latency_ms'])) if r['avg_latency_ms'] is not None else '—'}</td></tr>"
        for r in by_key["rows"]
    )
    table = (
        '<div class="table-wrap"><table><caption>Requests by key</caption><thead><tr><th scope="col">Key</th>'
        '<th scope="col">Name</th><th scope="col">Owner</th><th scope="col">Requests</th><th scope="col">Errors</th>'
        f'<th scope="col">Avg latency (ms)</th></tr></thead><tbody>{rows}</tbody></table></div>'
        if rows
        else ""
    )
    who = selected or "everyone"
    body = (
        '<form method="get" action="/admin/usage" class="filter">'
        f"{_user_select(members, selected, 'usage-user')} "
        f'<label for="usage-from">From</label> <input id="usage-from" name="from" type="date" value="{e(by_day["from"])}"> '
        f'<label for="usage-to">To</label> <input id="usage-to" name="to" type="date" value="{e(by_day["to"])}"> '
        '<button type="submit">Show</button></form>\n'
        f"{usage_chart(by_day, 'Usage of ' + who)}\n{table}"
    )
    return layout("Usage", body, viewer, csrf, current="/admin/usage")


# ---------------------------------------------------------------- members and audit


def members_page(viewer: Viewer, csrf: str, members: list[dict]) -> str:
    rows = []
    for m in members:
        action = (
            '<span class="muted">owner</span>'
            if m["role"] == "owner"
            else f'<button type="button" class="danger" data-action="remove-member" data-email="{e(m["email"])}">'
            "Remove</button>"
        )
        rows.append(
            f"<tr><td>{e(m['email'])}</td><td>{e(m['role'])}</td><td>{_t(m['invited_at'])}</td>"
            f"<td>{e(m['invited_by']) or '—'}</td><td>{_t(m['last_sign_in_at'])}</td>"
            f"<td>{e(m['active_keys'])}</td><td>{action}</td></tr>"
        )
    body = (
        '<div class="table-wrap"><table><caption>Owner and members</caption><thead><tr><th scope="col">Email</th>'
        '<th scope="col">Role</th><th scope="col">Invited</th><th scope="col">Invited by</th>'
        '<th scope="col">Last sign-in</th><th scope="col">Active keys</th><th scope="col">Actions</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>\n"
        "<p>Removing a member revokes all of their active keys and signs them out at once. Their keys and usage stay "
        "visible here.</p>\n"
        '<section aria-labelledby="invite-h"><h2 id="invite-h">Invite a member</h2>\n'
        '<form id="invite-member"><p><label for="invite-email">Email</label> '
        '<input id="invite-email" name="email" type="email" required maxlength="254" autocomplete="off" '
        'aria-describedby="invite-hint"> <button type="submit">Invite</button></p></form>\n'
        '<p id="invite-hint" class="hint">No email is sent: tell them to sign in at this page. The Google consent '
        "screen is in Testing mode, so also add the address as a <strong>Google test user</strong> in the GCP "
        "project <code>huynle-infra</code> (APIs &amp; Services, OAuth consent screen, Test users; at most 100). "
        "Without that, Google refuses their sign-in. After removing a member, remove the test user there too.</p>\n"
        "<noscript><p>Inviting and removing members needs JavaScript.</p></noscript></section>"
    )
    return layout("Members", body, viewer, csrf, current="/admin/members")


def audit_page(viewer: Viewer, csrf: str, page: dict, params: dict) -> str:
    rows = "".join(
        f"<tr><td>{_t(a['at'])}</td><td><code>{e(a['event'])}</code></td><td>{e(a['actor_email']) or '—'}</td>"
        f"<td>{e(a['target_email']) or '—'}</td><td>{e(a['key_id']) or '—'}</td>"
        f"<td><code>{e(json.dumps(a['detail'], sort_keys=True))}</code></td></tr>"
        for a in page["entries"]
    )
    more = ""
    if page["next_before"] is not None:
        query = urlencode({**params, "before": page["next_before"]})
        more = f'<p><a href="/admin/audit?{e(query)}">Older entries</a></p>'
    body = (
        '<div class="table-wrap"><table><caption>Audit log, newest first (last 30 days)</caption><thead><tr>'
        '<th scope="col">Time</th><th scope="col">Event</th><th scope="col">Actor</th><th scope="col">Target</th>'
        f'<th scope="col">Key</th><th scope="col">Detail</th></tr></thead><tbody>{rows}</tbody></table></div>{more}'
    )
    return layout("Audit log", body, viewer, csrf, current="/admin/audit")
