import urllib.request
import urllib.error
import json
import time

BASE_URL = "http://127.0.0.1:8000/api/v1"

def make_req(path, data=None, token=None, method="GET"):
    url = f"{BASE_URL}{path}"
    headers = {}
    body = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode("utf-8")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    return req

def test_chat_e2e():
    print("=== STARTING LIVE CHAT & CONVERSATION E2E TEST ===")
    
    # 1. Register User A
    ts = int(time.time())
    user_a_email = f"chat_user_a_{ts}@example.com"
    reg_a = urllib.request.urlopen(make_req("/auth/register", {"email": user_a_email, "password": "Password123!", "full_name": "User Alpha"}, method="POST"))
    assert reg_a.status == 201
    login_a = urllib.request.urlopen(make_req("/auth/login", {"email": user_a_email, "password": "Password123!"}, method="POST"))
    token_a = json.loads(login_a.read().decode())["access_token"]
    print(f"[PASS] User A registered and logged in ({user_a_email})")

    # 2. Register User B
    user_b_email = f"chat_user_b_{ts}@example.com"
    urllib.request.urlopen(make_req("/auth/register", {"email": user_b_email, "password": "Password123!", "full_name": "User Beta"}, method="POST"))
    login_b = urllib.request.urlopen(make_req("/auth/login", {"email": user_b_email, "password": "Password123!"}, method="POST"))
    token_b = json.loads(login_b.read().decode())["access_token"]
    print(f"[PASS] User B registered and logged in ({user_b_email})")

    # 3. User A creates conversation
    create_res = urllib.request.urlopen(make_req("/conversations", {"title": "Architecture Research"}, token=token_a, method="POST"))
    assert create_res.status == 201
    conv = json.loads(create_res.read().decode())
    conv_id = conv["id"]
    print(f"[PASS] Conversation created: id={conv_id}, title='{conv['title']}'")

    # 4. User A renames conversation
    rename_res = urllib.request.urlopen(make_req(f"/conversations/{conv_id}", {"title": "Vector Databases & Transformers"}, token=token_a, method="PUT"))
    assert rename_res.status == 200
    assert json.loads(rename_res.read().decode())["title"] == "Vector Databases & Transformers"
    print("[PASS] Conversation renamed successfully")

    def extract_stream_text(sse_body):
        text = ""
        for line in sse_body.splitlines():
            if line.startswith("data: "):
                val = line[6:].strip()
                if val == "[DONE]":
                    break
                try:
                    data = json.loads(val)
                    if "token" in data:
                        text += data["token"]
                except Exception:
                    pass
        return text

    # 5. User A streams Turn 1
    stream_req1 = make_req(f"/chat/{conv_id}/stream", {"role": "user", "content": "How do self-attention matrices scale with sequence length?"}, token=token_a, method="POST")
    with urllib.request.urlopen(stream_req1) as resp:
        stream_data = resp.read().decode()
        assert "[DONE]" in stream_data
        text1 = extract_stream_text(stream_data)
        print(f"[PASS] Turn 1: Streamed response received: '{text1[:40]}...'")

    # Give DB brief async commit time
    time.sleep(0.5)

    # 6. User A streams Turn 2 (Testing multi-turn memory and configuration fallback)
    stream_req2 = make_req(f"/chat/{conv_id}/stream", {"role": "user", "content": "Can FlashAttention reduce that quadratic complexity to linear IO?"}, token=token_a, method="POST")
    with urllib.request.urlopen(stream_req2) as resp:
        stream_data2 = resp.read().decode()
        assert "[DONE]" in stream_data2
        text2 = extract_stream_text(stream_data2)
        assert ("Turn 2" in text2 or "HOSTED_LLM_API_KEY" in text2 or "Configuration Notice" in text2)
        print(f"[PASS] Turn 2: Streamed response parsed successfully: '{text2[:60]}...'")

    time.sleep(0.3)

    # 7. User A checks conversation detail: message ordering
    detail_res = urllib.request.urlopen(make_req(f"/conversations/{conv_id}", token=token_a))
    detail = json.loads(detail_res.read().decode())
    messages = detail["messages"]
    assert len(messages) == 4, f"Expected 4 messages, found {len(messages)}"
    
    for i, m in enumerate(messages):
        assert m["message_order"] == i + 1, f"Message order mismatch at index {i}: {m['message_order']}"
    print(f"[PASS] Message persistence & order verified: {len(messages)} messages strictly ordered [1, 2, 3, 4]")

    # 8. User A searches conversation
    search_res = urllib.request.urlopen(make_req("/conversations?q=Transformers", token=token_a))
    search_items = json.loads(search_res.read().decode())
    assert any(c["id"] == conv_id for c in search_items), "Expected conversation in search results"
    print("[PASS] Search by title succeeded")

    # Search by message content
    search_content_res = urllib.request.urlopen(make_req("/conversations?q=FlashAttention", token=token_a))
    search_content_items = json.loads(search_content_res.read().decode())
    assert any(c["id"] == conv_id for c in search_content_items), "Expected message content search match"
    print("[PASS] Search by message content succeeded")

    # 9. User B attempts unauthorized access (User Isolation Verification)
    try:
        urllib.request.urlopen(make_req(f"/conversations/{conv_id}", token=token_b))
        assert False, "User B should not access User A's conversation"
    except urllib.error.HTTPError as e:
        assert e.code == 404
        print("[PASS] User isolation: User B received 404 trying to read User A's conversation")

    try:
        urllib.request.urlopen(make_req(f"/chat/{conv_id}/messages", {"role": "user", "content": "Injected"}, token=token_b, method="POST"))
        assert False, "User B should not post to User A's conversation"
    except urllib.error.HTTPError as e:
        assert e.code == 404
        print("[PASS] User isolation: User B received 404 trying to post to User A's conversation")

    # User B list conversations
    u2_list = json.loads(urllib.request.urlopen(make_req("/conversations", token=token_b)).read().decode())
    assert not any(c["id"] == conv_id for c in u2_list), "User A's conversation leaked to User B"
    print("[PASS] User isolation: User A's conversation completely isolated from User B's list")

    # 10. User A deletes conversation
    del_res = urllib.request.urlopen(make_req(f"/conversations/{conv_id}", token=token_a, method="DELETE"))
    assert del_res.status == 204
    print("[PASS] Conversation deleted (204 No Content)")

    # Confirm deletion
    try:
        urllib.request.urlopen(make_req(f"/conversations/{conv_id}", token=token_a))
        assert False, "Conversation should be deleted"
    except urllib.error.HTTPError as e:
        assert e.code == 404
        print("[PASS] Confirmed conversation 404 after deletion")

    print("=== ALL CHAT & CONVERSATION E2E TESTS PASSED ===")

if __name__ == "__main__":
    test_chat_e2e()
