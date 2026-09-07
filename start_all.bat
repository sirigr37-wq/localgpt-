@echo off
echo ===================================================
echo Starting LLM - XRay Servers and Cloudflare Tunnel
echo ===================================================

start "Backend API (:8000)" cmd /k "cd backend && python -m uvicorn app.main:app --reload --port 8000"
start "Frontend UI (:3000)" cmd /k "cd frontend && npm.cmd run dev"

echo Waiting 5 seconds for frontend to boot...
timeout /t 5 /nobreak >nul

start "Cloudflare Tunnel" cmd /k "npx.cmd --yes cloudflared tunnel --url http://localhost:3000"

echo.
echo All services launched!
echo Check the Cloudflare Tunnel window for your new public .trycloudflare.com link.
echo.
pause
