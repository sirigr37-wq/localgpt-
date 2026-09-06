import urllib.request
import urllib.error
import json
import time

def test_endpoints():
    base = 'http://127.0.0.1:8000/api/v1'
    
    # 1. Protected route without token
    req = urllib.request.Request(f'{base}/conversations')
    try:
        urllib.request.urlopen(req)
        assert False, 'Expected 401 Unauthorized'
    except urllib.error.HTTPError as e:
        assert e.code == 401, f'Expected 401, got {e.code}'
        print('[PASS] Protected route rejected unauthorized access (401)')

    # 2. Google OAuth URL
    with urllib.request.urlopen(f'{base}/auth/google/url') as resp:
        data = json.loads(resp.read().decode('utf-8'))
        assert 'configured' in data
        print(f'[PASS] Google OAuth URL endpoint: configured={data.get("configured")}')

    # 3. Register a test user
    email = f'live_verify_{int(time.time())}@example.com'
    reg_payload = json.dumps({'email': email, 'password': 'SecurePassword123!', 'full_name': 'Live Verifier'}).encode('utf-8')
    reg_req = urllib.request.Request(f'{base}/auth/register', data=reg_payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(reg_req) as resp:
        user_data = json.loads(resp.read().decode('utf-8'))
        assert user_data['email'] == email
        print(f'[PASS] Registration succeeded: {user_data["id"]}')

    # 4. Login
    login_payload = json.dumps({'email': email, 'password': 'SecurePassword123!'}).encode('utf-8')
    login_req = urllib.request.Request(f'{base}/auth/login', data=login_payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(login_req) as resp:
        token_data = json.loads(resp.read().decode('utf-8'))
        token = token_data['access_token']
        print('[PASS] Login succeeded, access token obtained')

    # 5. Protected route with token
    auth_req = urllib.request.Request(f'{base}/auth/me', headers={'Authorization': f'Bearer {token}'})
    with urllib.request.urlopen(auth_req) as resp:
        me_data = json.loads(resp.read().decode('utf-8'))
        assert me_data['email'] == email
        print(f'[PASS] Protected /auth/me verified for {me_data["email"]}')

    # 6. Password reset request
    reset_req_payload = json.dumps({'email': email}).encode('utf-8')
    reset_req = urllib.request.Request(f'{base}/auth/password-reset/request', data=reset_req_payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(reset_req) as resp:
        reset_res = json.loads(resp.read().decode('utf-8'))
        reset_token = reset_res['reset_token']
        print('[PASS] Password reset token generated')

    # 7. Password reset confirm
    confirm_payload = json.dumps({'token': reset_token, 'new_password': 'BrandNewPassword456!'}).encode('utf-8')
    confirm_req = urllib.request.Request(f'{base}/auth/password-reset/confirm', data=confirm_payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(confirm_req) as resp:
        confirm_res = json.loads(resp.read().decode('utf-8'))
        assert confirm_res['status'] == 'success'
        print('[PASS] Password reset confirmed')

    # 8. Login with new password
    login2_payload = json.dumps({'email': email, 'password': 'BrandNewPassword456!'}).encode('utf-8')
    login2_req = urllib.request.Request(f'{base}/auth/login', data=login2_payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(login2_req) as resp:
        token2_data = json.loads(resp.read().decode('utf-8'))
        assert 'access_token' in token2_data
        print('[PASS] Authenticated with new password successfully')

if __name__ == '__main__':
    test_endpoints()
