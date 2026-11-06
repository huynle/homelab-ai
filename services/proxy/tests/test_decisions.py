"""POST /v1/decisions: the OpenAI Decisions API wire format, translated onto Laya (text) and SigLIP2 (image)."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from conftest import IMAGE_URL, TEXT_URL

DECISIONS = "/v1/decisions"
PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()
PNG_URL = f"data:image/png;base64,{PNG_B64}"

DEPARTMENT = {
    "type": "choice",
    "name": "department",
    "instructions": "Which department should handle this request?",
    "choices": [
        {"value": "billing", "description": "invoices, payments, refunds"},
        {"value": "technical", "description": "bugs, outages, system errors"},
        {"value": "sales", "description": "pricing, new contracts"},
        {"value": "other", "description": "everything else"},
    ],
}
URGENCY = {
    "type": "score",
    "name": "urgency",
    "instructions": "How urgent is this request?",
    "levels": [
        {"label": "not urgent"},
        {"label": "soon"},
        {"label": "critical", "description": "deadline or blocking issue"},
    ],
}
CHURN = {"type": "predicate", "name": "churn_risk", "instructions": "Does the user threaten to cancel or leave?"}
TICKET = "We were billed twice for March. Please refund it today or we will cancel."


def laya_echo(request: httpx.Request) -> httpx.Response:
    """A fake Laya that answers every question it is sent, deterministically from the question itself."""
    body = json.loads(request.content)
    answers = {}
    for qid, q in body["questions"].items():
        if q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 0.8, "confidence": 0.8, "answer_confidence": 0.8}
        elif q["type"] == "choice":
            labels = list(q["criteria"])
            probs = {label: (0.7 if i == 0 else round(0.3 / (len(labels) - 1), 4)) for i, label in enumerate(labels)}
            answers[qid] = {"type": "choice", "choice": labels[0], "probabilities": probs, "confidence": 0.6}
        else:
            n = len(q["criteria"])
            probs = {str(i): (1.0 if i == n - 1 else 0.0) for i in range(n)}
            legend = {str(i): c for i, c in enumerate(q["criteria"])}
            answers[qid] = {
                "type": "score",
                "score": float(n - 1),
                "legend": legend,
                "probabilities": probs,
                "confidence": 0.9,
            }
    return httpx.Response(
        200,
        json={
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "routing": {"model": "english", "repo": "/home/huy/laya-test/models/laya"},
        },
    )


@pytest.fixture
def echo(backends):
    backends.text = laya_echo
    return backends


def predict_calls(backends) -> list[dict]:
    return [json.loads(r.content) for r in backends.requests if r.url.path == "/predict"]


def classify_calls(backends) -> list[dict]:
    return [json.loads(r.content) for r in backends.requests if r.url.path == "/classify"]


def post(client, auth, body):
    return client.post(DECISIONS, json=body, headers=auth)


def assert_error(resp, status: int, code: str, type_: str = "invalid_request_error"):
    assert resp.status_code == status, resp.text
    err = resp.json()["error"]
    assert err["code"] == code
    assert err["type"] == type_
    assert "param" in err
    assert isinstance(err["message"], str) and err["message"]
    return err


# ---------------------------------------------------------------- text: request translation


def test_translates_openai_questions_to_laya(client, auth, echo):
    resp = post(client, auth, {"model": "gpt-6-luna", "input": TICKET, "questions": [DEPARTMENT, URGENCY, CHURN]})
    assert resp.status_code == 200, resp.text
    (sent,) = predict_calls(echo)
    assert sent == {
        "state": TICKET,
        "questions": {
            "department": {
                "type": "choice",
                "instructions": DEPARTMENT["instructions"],
                "criteria": {c["value"]: c["description"] for c in DEPARTMENT["choices"]},
            },
            "urgency": {
                "type": "score",
                "instructions": URGENCY["instructions"],
                "criteria": ["not urgent", "soon", "critical: deadline or blocking issue"],
            },
            "churn_risk": {"type": "noul", "instructions": CHURN["instructions"]},
        },
    }


def test_gpt_6_luna_keeps_laya_auto_routing_and_registry_ids_pin(client, auth, echo):
    post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [CHURN]})
    post(client, auth, {"model": "laya-multilingual", "input": "x", "questions": [CHURN]})
    post(client, auth, {"input": "x", "questions": [CHURN]})
    first, pinned, omitted = predict_calls(echo)
    assert "model" not in first and "model" not in omitted
    assert pinned["model"] == "multilingual"


def test_unknown_or_image_model_is_rejected_for_text(client, auth, echo):
    err = assert_error(
        post(client, auth, {"model": "gpt-4o", "input": "x", "questions": [CHURN]}), 400, "invalid_request"
    )
    assert err["param"] == "model"
    assert_error(
        post(client, auth, {"model": "siglip2-base-patch16-224", "input": "x", "questions": [CHURN]}),
        400,
        "invalid_request",
    )


def test_unnamed_and_odd_names_get_safe_laya_ids(client, auth, echo):
    q1 = dict(CHURN)
    q1.pop("name")
    q2 = dict(CHURN, name="has spaces / slashes")
    q3 = dict(CHURN, name="dup")
    q4 = dict(CHURN, name="dup")
    resp = post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [q1, q2, q3, q4]})
    assert resp.status_code == 200, resp.text
    (sent,) = predict_calls(echo)
    assert list(sent["questions"]) == ["q0", "q1", "dup", "q3"]
    assert [a["name"] for a in resp.json()["answers"]] == [None, "has spaces / slashes", "dup", "dup"]


def test_instruction_parts_are_joined_as_text(client, auth, echo):
    q = dict(
        CHURN,
        instructions=[
            {"type": "input_text", "text": "Does this order"},
            {"type": "input_json", "data": {"charges": [49, 49]}},
            {"type": "input_text", "text": "show a duplicate charge?"},
        ],
    )
    assert post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [q]}).status_code == 200
    (sent,) = predict_calls(echo)
    assert sent["questions"]["churn_risk"]["instructions"] == (
        'Does this order\n{"charges":[49,49]}\nshow a duplicate charge?'
    )


# ---------------------------------------------------------------- text: answers


def test_answers_follow_openai_shapes_in_question_order(client, auth, echo):
    body = post(client, auth, {"model": "gpt-6-luna", "input": TICKET, "questions": [URGENCY, CHURN, DEPARTMENT]})
    body = body.json()
    assert body["model"] == "laya-english"
    urgency, churn, department = body["answers"]
    assert churn == {"type": "predicate", "name": "churn_risk", "probability": 0.8}
    assert department["type"] == "choice" and department["name"] == "department"
    assert department["choice"] == "billing"
    assert department["confidence"] == 0.6
    assert [p["value"] for p in department["probabilities"]] == ["billing", "technical", "sales", "other"]
    assert department["probabilities"][0]["probability"] == 0.7
    assert urgency == {
        "type": "score",
        "name": "urgency",
        "score": 2.0,
        "probabilities": [
            {"value": 0, "label": "not urgent", "probability": 0.0},
            {"value": 1, "label": "soon", "probability": 0.0},
            {"value": 2, "label": "critical", "probability": 1.0},
        ],
        "confidence": 0.9,
    }
    assert body["usage"] == {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 10,
    }
    assert "/home/" not in json.dumps(body)


def test_real_laya_capture_maps_to_openai_answers(client, auth, laya_predict):
    body = post(client, auth, {"model": "gpt-6-luna", "input": TICKET, "questions": [DEPARTMENT, URGENCY, CHURN]})
    assert body.status_code == 200, body.text
    department, urgency, churn = body.json()["answers"]
    assert department["choice"] == "billing" and department["confidence"] == 0.9141
    assert {p["value"]: p["probability"] for p in department["probabilities"]} == (
        laya_predict["answers"]["department"]["probabilities"]
    )
    assert urgency["score"] == 1.4386
    assert [p["probability"] for p in urgency["probabilities"]] == [0.104, 0.3535, 0.5426]
    assert churn["probability"] == 0.8184
    assert body.json()["usage"]["input_tokens"] == 249


def test_boolean_choice_values_round_trip(client, auth, echo):
    q = {
        "type": "choice",
        "name": "is_bug",
        "instructions": "Is this a defect?",
        "choices": [{"value": True, "description": "broken behaviour"}, {"value": False}],
    }
    body = post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [q]}).json()
    (sent,) = predict_calls(echo)
    assert sent["questions"]["is_bug"]["criteria"] == {"true": "broken behaviour", "false": "false"}
    (answer,) = body["answers"]
    assert answer["choice"] is True
    assert [p["value"] for p in answer["probabilities"]] == [True, False]


def test_colliding_choice_values_are_rejected(client, auth, echo):
    q = {"type": "choice", "name": "x", "instructions": "?", "choices": [{"value": True}, {"value": "true"}]}
    err = assert_error(
        post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [q]}), 400, "invalid_request"
    )
    assert err["param"] == "questions[0].choices"
    dup = {"type": "choice", "name": "x", "instructions": "?", "choices": [{"value": "a"}, {"value": "a"}]}
    assert_error(post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [dup]}), 400, "invalid_request")


def test_missing_laya_answer_becomes_refusal(client, auth, backends):
    def partial(request):
        resp = laya_echo(request)
        data = resp.json()
        data["answers"].pop("churn_risk")
        return httpx.Response(200, json=data)

    backends.text = partial
    body = post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [DEPARTMENT, CHURN]}).json()
    assert body["answers"][1] == {"type": "refusal", "name": "churn_risk"}
    assert body["answers"][0]["type"] == "choice"


def test_more_than_32_questions_are_chunked(client, auth, echo):
    questions = [dict(CHURN, name=f"p{i}") for i in range(70)]
    body = post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": questions}).json()
    calls = predict_calls(echo)
    assert [len(c["questions"]) for c in calls] == [32, 32, 6]
    assert [a["name"] for a in body["answers"]] == [f"p{i}" for i in range(70)]
    assert body["usage"]["input_tokens"] == 30


def test_question_count_limits(client, auth, echo):
    assert_error(post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": []}), 400, "invalid_request")
    too_many = [dict(CHURN, name=f"p{i}") for i in range(401)]
    err = assert_error(
        post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": too_many}), 400, "invalid_request"
    )
    assert err["param"] == "questions"
    assert predict_calls(echo) == []


def test_choice_and_level_limits(client, auth, echo):
    many = {"type": "choice", "name": "x", "instructions": "?", "choices": [{"value": f"c{i}"} for i in range(65)]}
    assert_error(post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [many]}), 400, "invalid_request")
    levels = {"type": "score", "name": "x", "instructions": "?", "levels": [{"label": f"l{i}"} for i in range(11)]}
    assert_error(
        post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [levels]}), 400, "invalid_request"
    )
    one = {"type": "choice", "name": "x", "instructions": "?", "choices": [{"value": "a"}]}
    assert_error(post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [one]}), 400, "invalid_request")
    assert predict_calls(echo) == []


def test_unknown_question_type_and_fields_are_rejected(client, auth, echo):
    assert_error(
        post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [{"type": "box", "instructions": "?"}]}),
        400,
        "invalid_request",
    )
    assert_error(
        post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [CHURN], "stream": True}),
        400,
        "invalid_request",
    )


def test_prompt_cache_and_safety_fields_are_accepted(client, auth, echo):
    resp = post(
        client,
        auth,
        {
            "model": "gpt-6-luna",
            "input": "x",
            "questions": [CHURN],
            "prompt_cache_key": "k",
            "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
            "safety_identifier": "user-1",
        },
    )
    assert resp.status_code == 200, resp.text
    (sent,) = predict_calls(echo)
    assert set(sent) == {"state", "questions"}


# ---------------------------------------------------------------- text: input forms


def test_single_input_json_part_becomes_a_json_state(client, auth, echo):
    data = {"ticket": {"body": "Please refund the duplicate charge."}, "order": {"charges": [49, 49]}}
    body = {
        "model": "gpt-6-luna",
        "questions": [CHURN],
        "input": [{"role": "user", "content": [{"type": "input_json", "data": data}]}],
    }
    assert post(client, auth, body).status_code == 200
    assert predict_calls(echo)[0]["state"] == data


def test_single_user_text_message_becomes_plain_text(client, auth, echo):
    body = {
        "model": "gpt-6-luna",
        "questions": [CHURN],
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": "line one"}, {"type": "input_text", "text": "line two"}],
            }
        ],
    }
    assert post(client, auth, body).status_code == 200
    assert predict_calls(echo)[0]["state"] == "line one\nline two"


def test_conversation_is_rendered_as_a_transcript(client, auth, echo):
    body = {
        "model": "gpt-6-luna",
        "questions": [CHURN],
        "input": [
            {"role": "system", "content": "You route tickets."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Refund order 7"},
                    {"type": "input_json", "data": {"order": 7}},
                ],
            },
            {"type": "function_call", "call_id": "c1", "name": "lookup_order", "arguments": '{"id":7}'},
            {"type": "function_call_output", "call_id": "c1", "output": "charged twice"},
            {"type": "custom_tool_call", "call_id": "c2", "name": "grep", "input": "refund"},
            {"type": "custom_tool_call_output", "call_id": "c2", "output": [{"type": "input_text", "text": "1 match"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Looking into it."},
                    {"type": "refusal", "refusal": "Cannot refund."},
                ],
            },
        ],
    }
    resp = post(client, auth, body)
    assert resp.status_code == 200, resp.text
    assert predict_calls(echo)[0]["state"] == "\n".join(  # noqa: FLY002 (one line per transcript entry)
        [
            "developer: You route tickets.",
            'user: Refund order 7\n{"order":7}',
            'function_call c1 lookup_order: {"id":7}',
            "function_call_output c1: charged twice",
            "custom_tool_call c2 grep: refund",
            "custom_tool_call_output c2: 1 match",
            "assistant: Looking into it.\nrefusal: Cannot refund.",
        ]
    )


@pytest.mark.parametrize(
    "items",
    [
        [{"type": "function_call_output", "call_id": "nope", "output": "x"}],
        [
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "custom_tool_call_output", "call_id": "c1", "output": "x"},
        ],
        [{"role": "developer", "content": [{"type": "input_json", "data": {"a": 1}}]}],
        [{"role": "assistant", "content": [{"type": "input_text", "text": "x"}]}],
        [{"role": "tool", "content": "x"}],
        [{"type": "item_reference", "id": "x"}],
        [{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}],
    ],
)
def test_invalid_input_items_are_rejected(client, auth, echo, items):
    err = assert_error(
        post(client, auth, {"model": "gpt-6-luna", "input": items, "questions": [CHURN]}), 400, "invalid_request"
    )
    assert err["param"].startswith("input")
    assert predict_calls(echo) == []


def test_empty_input_is_rejected(client, auth, echo):
    assert_error(post(client, auth, {"model": "gpt-6-luna", "input": "", "questions": [CHURN]}), 400, "invalid_request")
    assert_error(post(client, auth, {"model": "gpt-6-luna", "input": [], "questions": [CHURN]}), 400, "invalid_request")


def test_oversized_text_state_is_413(client, auth, echo):
    assert_error(
        post(client, auth, {"model": "gpt-6-luna", "input": "a" * (32 * 1024 + 1), "questions": [CHURN]}),
        413,
        "payload_too_large",
    )


# ---------------------------------------------------------------- images (SigLIP2)


def image_input(*urls, text=None):
    """One user message holding the images; with ``text`` the text comes first (then the image is ignored)."""
    parts = [{"type": "input_text", "text": text}] if text is not None else []
    parts += [{"type": "input_image", "image_url": u} for u in urls]
    return [{"role": "user", "content": parts}]


CATEGORY = {
    "type": "choice",
    "name": "category",
    "instructions": "What is in the photo?",
    "choices": [{"value": "german shepherd"}, {"value": "dog"}, {"value": "wolf"}],
}


def test_image_choice_routes_to_siglip(client, auth, backends):
    resp = post(client, auth, {"model": "gpt-6-luna", "input": image_input(PNG_URL), "questions": [CATEGORY]})
    assert resp.status_code == 200, resp.text
    (sent,) = classify_calls(backends)
    assert sent == {
        "image": {"base64": PNG_B64, "media_type": "image/png"},
        "candidate_labels": ["german shepherd", "dog", "wolf"],
    }
    assert predict_calls(backends) == []
    body = resp.json()
    assert body["model"] == "siglip2-base-patch16-224"
    (answer,) = body["answers"]
    assert answer["type"] == "choice" and answer["name"] == "category"
    assert answer["choice"] == "german shepherd"
    assert [p["value"] for p in answer["probabilities"]] == ["german shepherd", "dog", "wolf"]
    assert answer["confidence"] == answer["probabilities"][0]["probability"]
    assert body["usage"]["total_tokens"] == 0


def test_each_image_choice_question_is_one_siglip_call(client, auth, backends):
    second = dict(CATEGORY, name="again")
    body = post(
        client, auth, {"model": "gpt-6-luna", "input": image_input(PNG_URL), "questions": [CATEGORY, second]}
    ).json()
    assert len(classify_calls(backends)) == 2
    assert [a["name"] for a in body["answers"]] == ["category", "again"]


def test_image_model_can_be_pinned_but_text_model_cannot_read_images(client, auth, backends):
    ok = post(
        client, auth, {"model": "siglip2-base-patch16-224", "input": image_input(PNG_URL), "questions": [CATEGORY]}
    )
    assert ok.status_code == 200, ok.text
    assert_error(
        post(client, auth, {"model": "laya-english", "input": image_input(PNG_URL), "questions": [CATEGORY]}),
        400,
        "invalid_request",
    )


def test_jpg_data_url_is_sent_as_jpeg(client, auth, backends):
    url = "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/"
    post(client, auth, {"model": "gpt-6-luna", "input": image_input(url), "questions": [CATEGORY]})
    assert classify_calls(backends)[0]["image"]["media_type"] == "image/jpeg"


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"input": image_input(PNG_URL), "questions": [CHURN]}, "questions[0]"),
        ({"input": image_input(PNG_URL), "questions": [CATEGORY, URGENCY]}, "questions[1]"),
        ({"input": image_input(PNG_URL, PNG_URL), "questions": [CATEGORY]}, "input"),
        ({"input": image_input(PNG_URL, PNG_URL, text="two shots"), "questions": [CATEGORY]}, "input"),
        (
            {"input": image_input("https://example.com/cat.jpg", text="x"), "questions": [CATEGORY]},
            "input[0].content[1].image_url",
        ),
        (
            {"input": image_input("data:image/gif;base64,R0lGODlh", text="x"), "questions": [CATEGORY]},
            "input[0].content[1].image_url",
        ),
        (
            {"input": image_input("data:image/png;base64,not base64!", text="x"), "questions": [CATEGORY]},
            "input[0].content[1].image_url",
        ),
        (
            {"input": image_input("https://example.com/cat.jpg"), "questions": [CATEGORY]},
            "input[0].content[0].image_url",
        ),
        (
            {"input": "x", "questions": [dict(CATEGORY, instructions=[{"type": "input_image", "image_url": PNG_URL}])]},
            "questions[0].instructions",
        ),
    ],
)
def test_unsupported_image_requests_are_rejected(client, auth, backends, body, param):
    err = assert_error(post(client, auth, dict(body, model="gpt-6-luna")), 400, "invalid_request")
    assert err["param"] == param
    assert classify_calls(backends) == [] and predict_calls(backends) == []


def test_too_many_image_labels_are_rejected(client, auth, backends):
    q = dict(CATEGORY, choices=[{"value": f"c{i}"} for i in range(65)])
    assert_error(
        post(client, auth, {"model": "gpt-6-luna", "input": image_input(PNG_URL), "questions": [q]}),
        400,
        "invalid_request",
    )


def test_tool_call_with_image_output_goes_to_laya(client, auth, backends):
    items = [
        {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_image", "image_url": PNG_URL}]}
    ]
    items.insert(0, {"type": "function_call", "call_id": "c1", "name": "snap", "arguments": ""})
    resp = post(client, auth, {"model": "gpt-6-luna", "input": items, "questions": [CATEGORY]})
    assert resp.status_code == 200, resp.text
    # the call itself carries a name, which is evidence: text wins
    assert classify_calls(backends) == []
    assert resp.headers["x-decision-image"] == "ignored"


# ---------------------------------------------------------------- text wins: image + text/JSON goes to Laya

SCREENSHOT_STATE = {"url": "https://shop.example/cart", "elements": ["[3] button Checkout", "[4] link Help"]}
OPERATION = {
    "type": "choice",
    "name": "operation",
    "instructions": "Next browser action?",
    "choices": [{"value": "CLICK"}, {"value": "TYPE"}, {"value": "DONE"}],
}
TARGET = {
    "type": "choice",
    "name": "target",
    "instructions": "Which element?",
    "choices": [{"value": "3"}, {"value": "4"}],
}


def tabcrew_input(*extra_parts):
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_json", "data": SCREENSHOT_STATE},
                {"type": "input_image", "image_url": PNG_URL},
                *extra_parts,
            ],
        }
    ]


def test_screenshot_with_page_state_goes_to_laya_and_is_flagged(client, auth, echo):
    resp = post(client, auth, {"model": "gpt-6-luna", "input": tabcrew_input(), "questions": [OPERATION, TARGET]})
    assert resp.status_code == 200, resp.text
    assert classify_calls(echo) == []
    (sent,) = predict_calls(echo)
    assert sent["state"] == SCREENSHOT_STATE  # the image is dropped; the JSON stays a JSON state
    assert set(sent["questions"]) == {"operation", "target"}
    assert resp.headers["x-decision-image"] == "ignored"
    body = resp.json()
    assert body["model"] == "laya-english"
    assert [a["choice"] for a in body["answers"]] == ["CLICK", "3"]


def test_image_with_text_answers_any_question_type(client, auth, echo):
    resp = post(
        client,
        auth,
        {"model": "gpt-6-luna", "input": image_input(PNG_URL, text=TICKET), "questions": [CHURN, URGENCY, DEPARTMENT]},
    )
    assert resp.status_code == 200, resp.text
    assert predict_calls(echo)[0]["state"] == TICKET
    assert [a["type"] for a in resp.json()["answers"]] == ["predicate", "score", "choice"]
    assert resp.headers["x-decision-image"] == "ignored"


def test_text_in_another_message_also_wins(client, auth, echo):
    items = [{"role": "developer", "content": "Route the ticket."}, *image_input(PNG_URL)]
    resp = post(client, auth, {"model": "gpt-6-luna", "input": items, "questions": [DEPARTMENT]})
    assert resp.status_code == 200, resp.text
    assert classify_calls(echo) == [] and len(predict_calls(echo)) == 1
    assert "image" not in predict_calls(echo)[0]["state"]


def test_blank_text_does_not_count_as_evidence(client, auth, backends):
    resp = post(
        client, auth, {"model": "gpt-6-luna", "input": image_input(PNG_URL, text="  \n"), "questions": [CATEGORY]}
    )
    assert resp.status_code == 200, resp.text
    assert len(classify_calls(backends)) == 1
    assert "x-decision-image" not in resp.headers


def test_image_only_is_not_flagged(client, auth, backends):
    resp = post(client, auth, {"model": "gpt-6-luna", "input": image_input(PNG_URL), "questions": [CATEGORY]})
    assert resp.status_code == 200 and "x-decision-image" not in resp.headers


def test_text_requests_are_not_flagged(client, auth, echo):
    resp = post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [CHURN]})
    assert resp.status_code == 200 and "x-decision-image" not in resp.headers


def test_ignored_images_are_counted(client, auth, echo):
    def ignored():
        return (
            client.app.state.metrics.registry.get_sample_value(
                "decision_api_decision_images_ignored_total", {"model": "laya-english"}
            )
            or 0.0
        )

    post(client, auth, {"model": "gpt-6-luna", "input": tabcrew_input(), "questions": [OPERATION]})
    post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [CHURN]})
    assert ignored() == 1.0


def test_pinned_image_model_with_text_is_rejected(client, auth, echo):
    err = assert_error(
        post(client, auth, {"model": "siglip2-base-patch16-224", "input": tabcrew_input(), "questions": [OPERATION]}),
        400,
        "invalid_request",
    )
    assert err["param"] == "model"


def test_oversized_image_is_413(client, auth, backends):
    big = "data:image/png;base64," + "A" * (11184812 + 4)
    assert (
        client.post(
            DECISIONS, json={"model": "gpt-6-luna", "input": image_input(big), "questions": [CATEGORY]}, headers=auth
        ).status_code
        == 413
    )


# ---------------------------------------------------------------- errors, auth, backend failures


def test_requires_api_key(client, echo):
    resp = client.post(DECISIONS, json={"model": "gpt-6-luna", "input": "x", "questions": [CHURN]})
    assert_error(resp, 401, "unauthorized", "authentication_error")


def test_body_must_be_json_object(client, auth):
    assert_error(client.post(DECISIONS, content=b"[]", headers=auth), 400, "invalid_request")
    assert_error(client.post(DECISIONS, content=b"{nope", headers=auth), 400, "invalid_request")


def test_backend_failures_use_server_error_type(client, auth, backends):
    backends.text = lambda req: httpx.Response(500, json={"detail": "boom"})
    assert_error(
        post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [CHURN]}),
        502,
        "backend_error",
        "server_error",
    )
    backends.text = lambda req: (_ for _ in ()).throw(httpx.ConnectError("down"))
    assert_error(
        post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [CHURN]}),
        503,
        "backend_unavailable",
        "server_error",
    )


def test_backend_urls(client, auth, echo):
    post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [CHURN]})
    post(client, auth, {"model": "gpt-6-luna", "input": image_input(PNG_URL), "questions": [CATEGORY]})
    urls = {str(r.url) for r in echo.requests}
    assert urls == {f"{TEXT_URL}/predict", f"{IMAGE_URL}/classify"}


# ---------------------------------------------------------------- hardening (found by adversarial verification)

P = {"type": "predicate", "instructions": "x"}


@pytest.mark.parametrize(
    "body",
    [
        {"input": [{"role": ["user"], "content": "x"}], "questions": [P]},
        {"input": [{"type": ["message"], "role": "user", "content": "x"}], "questions": [P]},
        {"input": [{"role": "user", "content": [{"type": ["input_text"], "text": "x"}]}], "questions": [P]},
        {"input": "x", "questions": [{"type": {}, "instructions": "x"}]},
        {"input": "x", "questions": [{"type": ["predicate"], "instructions": "x"}]},
        {"input": [{"role": "user", "content": []}], "questions": [P]},
        {"input": image_input("data:image/png;base64,"), "questions": [CATEGORY]},
    ],
)
def test_malformed_client_values_are_400_with_a_param(client, auth, echo, body):
    err = assert_error(post(client, auth, body), 400, "invalid_request")
    assert err["param"]
    assert predict_calls(echo) == [] and classify_calls(echo) == []


def test_non_finite_json_numbers_are_rejected(client, auth, echo):
    raw = b'{"input": [{"role": "user", "content": [{"type": "input_json", "data": {"x": NaN}}]}], "questions": [{"type": "predicate", "instructions": "x"}]}'
    assert_error(client.post(DECISIONS, content=raw, headers=auth), 400, "invalid_request")
    raw = b'{"state": {"x": Infinity}, "questions": {"q": {"type": "noul", "instructions": "x"}}}'
    assert_error(client.post("/v1/decide/text", content=raw, headers=auth), 400, "invalid_request")
    raw = b'{"input": [{"role": "user", "content": [{"type": "input_json", "data": {"x": 1e999}}]}], "questions": [{"type": "predicate", "instructions": "x"}]}'
    assert_error(client.post(DECISIONS, content=raw, headers=auth), 400, "invalid_request")
    raw = b'{"state": {"x": -1e999}, "questions": {"q": {"type": "noul", "instructions": "x"}}}'
    assert_error(client.post("/v1/decide/text", content=raw, headers=auth), 400, "invalid_request")
    assert predict_calls(echo) == []


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "choice", "choice": ["billing"], "probabilities": {}, "confidence": 0.5},
        {"type": "choice", "choice": "billing", "probabilities": {"billing": float("nan")}, "confidence": 0.5},
        {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.5}, "confidence": float("inf")},
    ],
)
def test_malformed_laya_answers_are_502(client, auth, backends, answer):
    def reply(request):
        body = json.dumps({"answers": {"department": answer}}, allow_nan=True)
        return httpx.Response(200, content=body.encode(), headers={"content-type": "application/json"})

    backends.text = reply
    assert_error(
        post(client, auth, {"model": "gpt-6-luna", "input": "x", "questions": [DEPARTMENT]}),
        502,
        "backend_error",
        "server_error",
    )


@pytest.mark.parametrize("result", [{"label": ["dog"], "score": 0.9}, {"label": "dog", "score": float("nan")}])
def test_malformed_siglip_results_are_502(client, auth, backends, result):
    def reply(request):
        body = json.dumps({"results": [result]}, allow_nan=True)
        return httpx.Response(200, content=body.encode(), headers={"content-type": "application/json"})

    backends.image = reply
    assert_error(
        post(client, auth, {"model": "gpt-6-luna", "input": image_input(PNG_URL), "questions": [CATEGORY]}),
        502,
        "backend_error",
        "server_error",
    )


# ---------------------------------------------------------------- backend hardening on the native routes


def _nan_json(payload):
    return httpx.Response(
        200, content=json.dumps(payload, allow_nan=True).encode(), headers={"content-type": "application/json"}
    )


def test_native_text_route_rejects_non_finite_backend_numbers(client, auth, backends, laya_request, laya_predict):
    bad = json.loads(json.dumps(laya_predict))
    bad["answers"]["department"]["confidence"] = float("nan")
    backends.text = lambda req: _nan_json(bad)
    assert_error(client.post("/v1/decide/text", json=laya_request, headers=auth), 502, "backend_error", "server_error")


def test_native_image_route_rejects_non_finite_backend_numbers(client, auth, backends, image_request):
    backends.image = lambda req: _nan_json({"results": [{"label": "dog", "score": float("inf")}]})
    assert_error(
        client.post("/v1/classify/image", json=image_request, headers=auth), 502, "backend_error", "server_error"
    )


@pytest.mark.parametrize(
    "detail", ["bad file /home/huy/laya-test/models/laya/x.bin", "C:\\Users\\huy\\x", "/Users/huy/a b/c"]
)
def test_backend_4xx_detail_has_paths_redacted(client, auth, backends, laya_request, detail):
    backends.text = lambda req: httpx.Response(422, json={"detail": detail})
    err = assert_error(client.post("/v1/decide/text", json=laya_request, headers=auth), 400, "invalid_request")
    assert "huy" not in err["message"] and "[path]" in err["message"]


def test_backend_4xx_validation_text_is_kept(client, auth, backends, laya_request):
    backends.text = lambda req: httpx.Response(
        422, json={"detail": [{"loc": ["body", "questions"], "msg": "too many"}]}
    )
    err = assert_error(client.post("/v1/decide/text", json=laya_request, headers=auth), 400, "invalid_request")
    assert "body.questions: too many" in err["message"]
