# reap.ps1 — stop the resident llama-server and verify the GPU is clear. PID-scoped.
$ErrorActionPreference = "Continue"
$ls = Get-Process llama-server -ErrorAction SilentlyContinue
if ($ls) { $ls | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue } }
for ($i=0; $i -lt 30; $i++) { if (-not (Get-Process llama-server -ErrorAction SilentlyContinue)) { Write-Host "reaped (0 resident)"; exit 0 }; Start-Sleep -Seconds 1 }
Write-Host "WARNING: residual llama-server"; exit 1
