@echo off
echo ===================================================
echo Starting LLM - XRay Servers and Cloudflare Tunnel
echo ===================================================

start "Backend API (:8000)" cmd /k "cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
start "Frontend UI (:3000)" cmd /k "cd frontend && npm.cmd run dev -- -H 0.0.0.0"

echo Waiting 5 seconds for frontend to boot...
timeout /t 5 /nobreak >nul

start "Cloudflare Tunnel" cmd /k "npx.cmd --yes cloudflared tunnel --url http://localhost:3000"

echo.
echo ========================================================
echo  SERVICES LAUNCHED!
echo ========================================================
echo.
echo [OPTION 1] SAME WI-FI ACCESS (PHONE, TABLET, OTHER PC):
echo    http://172.21.3.157:3000
echo    -> This link NEVER expires and has NO DNS ERRORS!
echo.
echo [OPTION 2] EXTERNAL / CELLULAR DATA ACCESS:
echo    Look at the "Cloudflare Tunnel" window to copy your
echo    NEW random https://....trycloudflare.com link.
echo    (Remember: The link changes EVERY time you restart!)
echo ========================================================
echo.
pause
