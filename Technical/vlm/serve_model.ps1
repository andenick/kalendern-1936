# serve_model.ps1 — launch ONE OCR VLM on llama-server (single-launch) and leave it resident.
# Idempotent: if a healthy server is already up on :8090, returns immediately. Orphan-GPU-python abort.
#   .\serve_model.ps1 -Model dots.ocr
param([Parameter(Mandatory=$true)][string]$Model)
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
# Path to a llama.cpp llama-server.exe build (set LLAMA_SERVER) and the directory
# holding the GGUF OCR models (set MODELS_DIR). See README.
$SERVER   = if ($env:LLAMA_SERVER) { $env:LLAMA_SERVER } else { "llama-server.exe" }
$M        = if ($env:MODELS_DIR)   { $env:MODELS_DIR }   else { "models" }
$ENDPOINT = "http://127.0.0.1:8090"
$RH       = $PSScriptRoot
$COMMON   = @("-ngl","99","-fa","on","--cache-type-k","q8_0","--cache-type-v","q8_0","--host","127.0.0.1","--port","8090")

$MODELS = @{
  "dots.ocr"        = @("-m","$M\dots.ocr\dots.ocr-Q8_0.gguf","--mmproj","$M\dots.ocr\mmproj-dots.ocr-f16.gguf","-c","16384","-b","4096","--image-min-tokens","1024")
  "olmOCR-2-7B"     = @("-m","$M\olmOCR-2-7B\olmOCR-2-7B-1025-Q8_0.gguf","--mmproj","$M\olmOCR-2-7B\mmproj-olmOCR-2-7B-1025-vision.gguf","-c","8192","--image-min-tokens","1024")
  "GLM-OCR"         = @("-m","$M\GLM-OCR\GLM-OCR-Q8_0.gguf","--mmproj","$M\GLM-OCR\mmproj-GLM-OCR-Q8_0.gguf","-c","8192","--image-min-tokens","1024")
  "Qianfan-OCR"     = @("-m","$M\Qianfan-OCR\Qianfan-OCR-q8_0.gguf","--mmproj","$M\Qianfan-OCR\Qianfan-OCR-mmproj-f16.gguf","-c","8192")
  "Chandra-OCR-2"   = @("-m","$M\Chandra-OCR-2\chandra-ocr-2.Q8_0.gguf","--mmproj","$M\Chandra-OCR-2\chandra-ocr-2.mmproj-f16.gguf","-c","8192","--image-min-tokens","1024")
  "PaddleOCR-VL-1.5"= @("-m","$M\PaddleOCR-VL-1.5\PaddleOCR-VL-1.5.gguf","--mmproj","$M\PaddleOCR-VL-1.5\PaddleOCR-VL-1.5-mmproj.gguf","-c","8192","--image-min-tokens","1024")
  "PaddleOCR-VL-1.6"= @("-m","$M\PaddleOCR-VL-1.6\PaddleOCR-VL-1.6-GGUF.gguf","--mmproj","$M\PaddleOCR-VL-1.6\PaddleOCR-VL-1.6-GGUF-mmproj.gguf","-c","8192","--image-min-tokens","1024")
  "Qwen3-VL-8B"     = @("-m","$M\Qwen3-VL-8B-Instruct\Qwen3VL-8B-Instruct-Q8_0.gguf","--mmproj","$M\Qwen3-VL-8B-Instruct\mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf","-c","8192","--image-min-tokens","1024")
}
# per-model environment (e.g. suppress Qianfan's layout-as-thinking)
$MODEL_ENV = @{ "Qianfan-OCR" = @{ LLAMA_CHAT_TEMPLATE_KWARGS = '{"enable_thinking":false}' } }
if (-not $MODELS.ContainsKey($Model)) { Write-Host "UNKNOWN model $Model"; exit 3 }
if ($MODEL_ENV.ContainsKey($Model)) { foreach ($k in $MODEL_ENV[$Model].Keys) { Set-Item -Path "Env:\$k" -Value $MODEL_ENV[$Model][$k] } }

# orphan GPU python guard (never blanket-kill; abort instead)
$orphans = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'hopperline|run_rscd|gpu_compose|gpu_namer|engine.server' }
if ($orphans) { Write-Host "ABORT: orphan GPU python(s) present"; exit 2 }

# already healthy?
try { if ((Invoke-WebRequest "$ENDPOINT/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) {
  $cur = Get-Process llama-server -ErrorAction SilentlyContinue
  Write-Host "ALREADY HEALTHY (PID=$($cur.Id)) — reusing for $Model"; exit 0 } } catch {}

# clear any stale server, then launch
$ls = Get-Process llama-server -ErrorAction SilentlyContinue
if ($ls) { $ls | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }; Start-Sleep 2 }
$slog = Join-Path $RH "_server_$Model.log"
$proc = Start-Process -FilePath $SERVER -ArgumentList ($MODELS[$Model] + $COMMON) -PassThru -RedirectStandardOutput $slog -RedirectStandardError "$slog.err" -WindowStyle Hidden
Write-Host "[$Model] launched PID=$($proc.Id); waiting /health ..."
for ($i=0; $i -lt 150; $i++) {
  Start-Sleep -Seconds 2
  if ($proc.HasExited) { Write-Host "[$Model] EXITED early code=$($proc.ExitCode) — see $slog.err"; exit 4 }
  try { if ((Invoke-WebRequest "$ENDPOINT/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { Write-Host "[$Model] HEALTHY ~$($i*2)s PID=$($proc.Id)"; exit 0 } } catch {}
}
Write-Host "[$Model] never healthy"; exit 5
