# serve_model.ps1 — launch ONE OCR VLM on llama-server (single-launch) and leave it resident.
# Idempotent: if a healthy server is already up on the port, returns immediately. Orphan-GPU-python abort.
# Model registry + paths come from a shared serve config (location via the SERVE_CONFIG env var,
# default ./serve_config.json next to this script). status!=ok models are refused (no wasted health wait).
#   .\serve_model.ps1 -Model GLM-OCR
#   .\serve_model.ps1 -Model GLM-OCR -Port 8090 -DryRun
#
# serve_config.json schema (JSON):
#   {
#     "server":     "C:\\path\\to\\llama-server.exe",   // llama.cpp server exe (env LLAMA_SERVER overrides)
#     "models_dir": "C:\\path\\to\\models",              // dir holding the gguf + mmproj files (env MODELS_DIR overrides)
#     "port":       8080,                                  // default port (-Port overrides)
#     "common":     ["--n-gpu-layers","999"],            // args passed to every model launch
#     "models": {
#       "GLM-OCR": {
#         "status": "ok",                                  // only status=="ok" models will launch
#         "model":  "glm-ocr.gguf",                        // gguf filename under models_dir
#         "mmproj": "glm-ocr-mmproj.gguf",                 // multimodal projector filename under models_dir
#         "extra":  ["--ctx-size","8192"],               // optional per-model args
#         "env":    { "VAR": "value" },                   // optional per-model environment
#         "note":   "free-text note shown when refused"
#       }
#     }
#   }
param(
  [Parameter(Mandatory=$true)][string]$Model,
  [int]$Port = 0,
  [switch]$DryRun
)
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
$RH = $PSScriptRoot

# --- load shared serve config (env SERVE_CONFIG overrides location) ---
$CfgPath = if ($env:SERVE_CONFIG) { $env:SERVE_CONFIG } else { Join-Path $PSScriptRoot "serve_config.json" }
if (-not (Test-Path $CfgPath)) { Write-Host "FATAL: serve config not found at $CfgPath (set SERVE_CONFIG)"; exit 8 }
try { $cfg = Get-Content -Raw -Path $CfgPath | ConvertFrom-Json } catch { Write-Host "FATAL: cannot parse $CfgPath : $($_.Exception.Message)"; exit 8 }

# server + models dir: env override > config
$SERVER = if ($env:LLAMA_SERVER) { $env:LLAMA_SERVER } else { $cfg.server }
$M      = if ($env:MODELS_DIR)   { $env:MODELS_DIR }   else { $cfg.models_dir }
if ($Port -le 0) { $Port = [int]$cfg.port }
$ENDPOINT = "http://127.0.0.1:$Port"
$COMMON   = @($cfg.common) + @("--host","127.0.0.1","--port","$Port")

# --- resolve + gate the requested model ---
if (-not ($cfg.models.PSObject.Properties.Name -contains $Model)) {
  Write-Host "UNKNOWN model '$Model'. Known: $($cfg.models.PSObject.Properties.Name -join ', ')"; exit 3
}
$entry = $cfg.models.$Model
if ($entry.status -ne "ok") {
  Write-Host "REFUSING to launch '$Model' — status='$($entry.status)'. $($entry.note)"
  Write-Host "  (Only status='ok' models serve. Edit $CfgPath to change a verdict.)"
  exit 7
}
$mpath  = Join-Path $M $entry.model
$mmproj = Join-Path $M $entry.mmproj
$ARGS   = @("-m",$mpath,"--mmproj",$mmproj) + @($entry.extra) + $COMMON

if ($DryRun) {
  Write-Host "[$Model] DRYRUN status=ok port=$Port"
  Write-Host "  server : $SERVER"
  Write-Host "  cmd    : $SERVER $($ARGS -join ' ')"
  if (-not (Test-Path $SERVER)) { Write-Host "  WARN: server exe not found" }
  if (-not (Test-Path $mpath))  { Write-Host "  WARN: model gguf not found: $mpath" }
  if (-not (Test-Path $mmproj)) { Write-Host "  WARN: mmproj not found: $mmproj" }
  exit 0
}

# per-model environment (e.g. suppress Qianfan's layout-as-thinking)
if ($entry.env) { foreach ($p in $entry.env.PSObject.Properties) { Set-Item -Path "Env:\$($p.Name)" -Value $p.Value } }

# orphan GPU python guard (never blanket-kill; abort instead)
$orphans = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'hopperline|run_rscd|gpu_compose|gpu_namer|engine.server' }
if ($orphans) { Write-Host "ABORT: orphan GPU python(s) present"; exit 2 }

# already healthy?
try { if ((Invoke-WebRequest "$ENDPOINT/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) {
  $cur = Get-Process llama-server -ErrorAction SilentlyContinue
  Write-Host "ALREADY HEALTHY (PID=$($cur.Id)) — reusing for $Model on :$Port"; exit 0 } } catch {}

# clear any stale server + free the port, then launch (robust against the intermittent
# Start-Process "cannot find the file specified" redirect flake: unique log per attempt + retry).
$ls = Get-Process llama-server -ErrorAction SilentlyContinue
if ($ls) { $ls | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }; Start-Sleep 2 }
for ($p=0; $p -lt 20; $p++) {
  try { if (-not (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) { break } } catch { break }
  Start-Sleep 1
}
if (-not (Test-Path $RH)) { New-Item -ItemType Directory -Force -Path $RH | Out-Null }
$proc = $null; $slog = $null
foreach ($try in 1..3) {
  $stamp = (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
  $slog = Join-Path $RH "_server_${Model}_$stamp.log"
  try {
    # llama.cpp logs to STDERR; redirecting BOTH stdout+stderr intermittently throws
    # "cannot find the file specified" on Start-Process, so redirect stderr only.
    $proc = Start-Process -FilePath $SERVER -ArgumentList $ARGS -PassThru -RedirectStandardError "$slog.err" -WindowStyle Hidden
    if ($proc) { break }
  } catch { Write-Host "[$Model] launch attempt $try failed: $($_.Exception.Message)"; Start-Sleep 3 }
}
if (-not $proc) { Write-Host "[$Model] LAUNCH FAILED after 3 attempts"; exit 6 }
Write-Host "[$Model] launched PID=$($proc.Id) on :$Port; waiting /health ..."
for ($i=0; $i -lt 150; $i++) {
  Start-Sleep -Seconds 2
  if ($proc.HasExited) {
    Write-Host "[$Model] EXITED early code=$($proc.ExitCode) — last lines of $slog.err:"
    try { Get-Content "$slog.err" -Tail 15 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    | $_" } } catch {}
    exit 4
  }
  try { if ((Invoke-WebRequest "$ENDPOINT/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { Write-Host "[$Model] HEALTHY ~$($i*2)s PID=$($proc.Id) on :$Port"; exit 0 } } catch {}
}
Write-Host "[$Model] never healthy after 300s — last lines of $slog.err:"
try { Get-Content "$slog.err" -Tail 15 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    | $_" } } catch {}
exit 5
