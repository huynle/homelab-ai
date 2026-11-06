// Decision API admin page. Calls the JSON API (/admin/api) with the session's CSRF token from the
// <meta name="csrf-token"> tag; same-origin fetches send the Origin header the server checks.
// No inline scripts: the page's Content-Security-Policy allows only this file.
"use strict";

(function () {
  const meta = document.querySelector('meta[name="csrf-token"]');
  const csrf = meta ? meta.getAttribute("content") : null;
  const status = document.getElementById("status");

  function say(text, isError) {
    if (!status) return;
    status.textContent = text;
    status.classList.toggle("error", Boolean(isError));
  }

  async function api(method, url, body) {
    const headers = { "X-CSRF-Token": csrf || "" };
    const init = { method, headers, credentials: "same-origin", cache: "no-store" };
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    const resp = await fetch(url, init);
    let data = null;
    if (resp.status !== 204) {
      try {
        data = await resp.json();
      } catch (_) {
        data = null;
      }
    }
    if (!resp.ok) {
      const message = data && data.error ? data.error.message : "HTTP " + resp.status;
      if (resp.status === 401) {
        window.location.assign("/admin");
      }
      throw new Error(message);
    }
    return data;
  }

  function run(promise, done) {
    promise.then(done, (err) => say("Failed: " + err.message, true));
  }

  // ---- create a key: the full key is shown once
  const create = document.getElementById("create-key");
  if (create) {
    create.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const name = create.elements.name.value.trim();
      const day = create.elements.expires.value;
      const body = { name };
      if (day) body.expires_at = day + "T23:59:59Z";
      run(api("POST", "/admin/api/keys", body), (key) => {
        const box = document.getElementById("new-key");
        document.getElementById("new-key-value").textContent = key.key;
        box.hidden = false;
        create.reset();
        say("Key \u201c" + key.name + "\u201d created. Copy it now.");
        box.querySelector('[data-action="copy"]').focus();
      });
    });
  }

  // ---- rename
  document.querySelectorAll("form.rename").forEach((form) => {
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const id = form.getAttribute("data-key");
      run(api("PATCH", "/admin/api/keys/" + encodeURIComponent(id), { name: form.elements.name.value.trim() }), () =>
        window.location.reload()
      );
    });
  });

  // ---- invite
  const invite = document.getElementById("invite-member");
  if (invite) {
    invite.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const email = invite.elements.email.value.trim();
      run(api("POST", "/admin/api/members", { email }), () => {
        say("Invited " + email + ". Also add them as a Google test user in huynle-infra.");
        window.setTimeout(() => window.location.reload(), 1500);
      });
    });
  }

  // ---- buttons
  document.addEventListener("click", (ev) => {
    const button = ev.target.closest("button[data-action]");
    if (!button) return;
    const action = button.getAttribute("data-action");
    if (action === "copy") {
      const value = document.getElementById("new-key-value").textContent;
      navigator.clipboard.writeText(value).then(
        () => say("Key copied to the clipboard."),
        () => say("Copy failed: select the key and copy it by hand.", true)
      );
    } else if (action === "reload") {
      window.location.reload();
    } else if (action === "revoke") {
      const name = button.getAttribute("data-name");
      if (!window.confirm("Revoke the key \u201c" + name + "\u201d? It stops working at once. This cannot be undone.")) {
        return;
      }
      const id = button.getAttribute("data-key");
      run(api("POST", "/admin/api/keys/" + encodeURIComponent(id) + "/revoke"), () => window.location.reload());
    } else if (action === "toggle-paid") {
      // owner only (the server refuses members): paid OpenRouter models for this key, from its next request
      const id = button.getAttribute("data-paid-key");
      const allow = button.getAttribute("data-allow") === "true";
      const name = button.getAttribute("data-name");
      const question = allow
        ? "Allow the key \u201c" + name + "\u201d to use paid models? Its requests then cost OpenRouter credits."
        : "Stop the key \u201c" + name + "\u201d from using paid models?";
      if (!window.confirm(question)) return;
      run(api("PATCH", "/admin/api/keys/" + encodeURIComponent(id), { allow_paid: allow }), () =>
        window.location.reload()
      );
    } else if (action === "remove-member") {
      const email = button.getAttribute("data-email");
      const ok = window.confirm(
        "Remove " + email + "? All of their active keys are revoked now and they are signed out. " +
          "Also remove them from the Google test users in huynle-infra."
      );
      if (!ok) return;
      run(api("DELETE", "/admin/api/members/" + encodeURIComponent(email)), (res) => {
        say("Removed " + res.email + "; revoked " + res.revoked_keys.length + " key(s).");
        window.setTimeout(() => window.location.reload(), 1500);
      });
    } else if (action === "logout") {
      run(api("POST", "/admin/auth/logout"), () => window.location.assign("/admin"));
    }
  });
})();
