"""
Antigravity IPv6 Bridge
Routes antigravity.google and IPv4-only services over IPv6 NAT64 gateway
when operating on IPv6-only mobile hotspots (such as Jio/cellular).
"""

import socket
import select
import threading
import winreg
import ctypes
import time
import sys

PORT = 8899
NAT64_IP = '2a00:1098:2b::1:d8ef:203d'

PAC_CONTENT = f"""function FindProxyForURL(url, host) {{
    if (shExpMatch(host, "*.antigravity.google") || host === "antigravity.google") {{
        return "PROXY 127.0.0.1:{PORT}; DIRECT";
    }}
    return "DIRECT";
}}
"""

def notify_wininet():
    try:
        INTERNET_OPTION_SETTINGS_CHANGED = 39
        INTERNET_OPTION_REFRESH = 37
        wininet = ctypes.windll.wininet
        wininet.InternetSetOptionW(0, INTERNET_OPTION_SETTINGS_CHANGED, 0, 0)
        wininet.InternetSetOptionW(0, INTERNET_OPTION_REFRESH, 0, 0)
    except Exception as e:
        print(f"Failed to notify WinINet: {e}", flush=True)

def enable_pac():
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings", 0, winreg.KEY_SET_VALUE)
        pac_url = f"http://127.0.0.1:{PORT}/proxy.pac"
        winreg.SetValueEx(key, "AutoConfigURL", 0, winreg.REG_SZ, pac_url)
        winreg.CloseKey(key)
        print(f"System AutoConfigURL set to: {pac_url}", flush=True)
        notify_wininet()
    except Exception as e:
        print(f"Failed to set Windows proxy setting: {e}", flush=True)

def disable_pac():
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings", 0, winreg.KEY_SET_VALUE)
        try:
            winreg.DeleteValue(key, "AutoConfigURL")
            print("AutoConfigURL removed from Windows settings.", flush=True)
        except FileNotFoundError:
            pass
        winreg.CloseKey(key)
        notify_wininet()
    except Exception as e:
        print(f"Failed to disable proxy setting: {e}", flush=True)

def handle_client(client_sock):
    try:
        client_sock.settimeout(15.0)
        raw_req = client_sock.recv(4096)
        if not raw_req:
            return
        req_str = raw_req.decode('latin1', errors='ignore')
        first_line = req_str.split('\r\n')[0]
        parts = first_line.split(' ')
        if len(parts) < 2:
            return
        method, target = parts[0], parts[1]

        # Serve PAC file to Windows / Edge / Chrome
        if method.upper() == 'GET' and ('proxy.pac' in target or target == '/'):
            body = PAC_CONTENT.encode('utf-8')
            resp = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/x-ns-proxy-autoconfig\r\n"
                b"Content-Length: " + str(len(body)).encode('ascii') + b"\r\n"
                b"Connection: close\r\n\r\n" + body
            )
            client_sock.sendall(resp)
            return

        # Handle HTTPS CONNECT
        if method.upper() == 'CONNECT':
            host, port_str = target.split(':')
            port = int(port_str)
            
            # Route antigravity.google to NAT64 IPv6 gateway
            if 'antigravity.google' in host:
                remote_ip = NAT64_IP
            else:
                remote_ip = host

            remote_sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            remote_sock.settimeout(15.0)
            remote_sock.connect((remote_ip, port))
            
            client_sock.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            
            # Bidirectional streaming
            client_sock.setblocking(False)
            remote_sock.setblocking(False)
            sockets = [client_sock, remote_sock]
            
            while True:
                r, _, _ = select.select(sockets, [], [], 60)
                if not r:
                    break
                for s in r:
                    other = remote_sock if s is client_sock else client_sock
                    try:
                        data = s.recv(16384)
                        if not data:
                            return
                        other.sendall(data)
                    except Exception:
                        return
    except Exception:
        pass
    finally:
        try:
            client_sock.close()
        except Exception:
            pass

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--disable':
        disable_pac()
        return

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', PORT))
    server.listen(50)
    print(f"Antigravity Bridge active on 127.0.0.1:{PORT}", flush=True)

    enable_pac()

    try:
        while True:
            client, _ = server.accept()
            t = threading.Thread(target=handle_client, args=(client,), daemon=True)
            t.start()
    except KeyboardInterrupt:
        pass
    finally:
        disable_pac()

if __name__ == '__main__':
    main()
