# Synthesize 60 E2E pilot WAVs using Ella voice (KO 30 + EN 30)
# Ella voice_id: tc_62fb679683a541c351dc7c3a
# KO: language param omitted (Ella+kor=500 error, omitted=OK)
# EN: language=eng

param(
    [string]$PilotJson = "Dev_Plan\e2e_voice_pilot_scripts.json",
    [string]$OutDir = "assets\e2e_voice_inputs",
    [string]$ApiKeyFile = "C:\Users\danie\OneDrive\바탕 화면\00.Porduct_Dev\260227_Project_ mung-i\typecast.md"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$VOICE_ID = "tc_62fb679683a541c351dc7c3a"
$VOICE_NAME = "Ella"
$MODEL = "ssfm-v21"
$API_URL = "https://api.typecast.ai/v1/text-to-speech"
$MAX_RETRIES = 5
$RETRY_DELAYS = @(2, 4, 8, 16, 32)

# Load API key
if ($env:TYPECAST_API_KEY) {
    $apiKey = $env:TYPECAST_API_KEY
} else {
    try {
        $apiKey = ((Get-Content $ApiKeyFile -Encoding UTF8) -join "").Trim()
    } catch {
        Write-Host "[ERROR] Cannot read API key file. Set TYPECAST_API_KEY env var." -ForegroundColor Red
        exit 1
    }
}
Write-Host "[INFO] API key loaded" -ForegroundColor Green

# Load pilot JSON
$pilotRaw = (Get-Content $PilotJson -Encoding UTF8) -join "`n"
$pilot = $pilotRaw | ConvertFrom-Json

# Create output dirs
$koDir = Join-Path $OutDir "ko"
$enDir = Join-Path $OutDir "en"
New-Item -ItemType Directory -Path $koDir -Force | Out-Null
New-Item -ItemType Directory -Path $enDir -Force | Out-Null

function Invoke-TypecastTTS {
    param(
        [string]$Text,
        [string]$Lang,
        [string]$OutputPath,
        [string]$RoundId,
        [int]$MsgIdx
    )

    $body = @{
        voice_id = $VOICE_ID
        text = $Text
        model = $MODEL
        prompt = @{
            emotion_type = "preset"
            emotion_preset = "normal"
            emotion_intensity = 1.0
        }
        output = @{
            audio_format = "wav"
            audio_tempo = 1.0
            audio_pitch = 0
            volume = 100
        }
    }

    # Only add language for English; omit for Korean (Ella+kor=500)
    if ($Lang -eq "en") {
        $body["language"] = "eng"
    }

    $jsonBody = $body | ConvertTo-Json -Depth 3
    $bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($jsonBody)
    $headers = @{
        "X-API-KEY" = $apiKey
        "Content-Type" = "application/json"
    }

    for ($attempt = 1; $attempt -le $MAX_RETRIES; $attempt++) {
        try {
            Invoke-WebRequest -Uri $API_URL -Method Post -Headers $headers -Body $bodyBytes -OutFile $OutputPath | Out-Null
            $sz = (Get-Item $OutputPath).Length
            return $sz
        } catch {
            if ($attempt -eq $MAX_RETRIES) {
                Write-Host "[ERROR] ${RoundId} msg=${MsgIdx}: FAILED after $MAX_RETRIES attempts - $($_.Exception.Message)" -ForegroundColor Red
                return -1
            }
            $delay = $RETRY_DELAYS[$attempt - 1]
            Write-Host "[WARN] ${RoundId} msg=${MsgIdx}: attempt $attempt failed, retrying in ${delay}s..." -ForegroundColor Yellow
            Start-Sleep -Seconds $delay
        }
    }
    return -1
}

# Synthesize
$totalSuccess = 0
$totalFail = 0
$startTime = Get-Date

foreach ($lang in @("ko", "en")) {
    $rounds = $pilot.rounds.$lang
    Write-Host "`n[INFO] === Synthesizing $($lang.ToUpper()) ($($rounds.Count) rounds) ===" -ForegroundColor Cyan

    $msgCount = 0
    foreach ($round in $rounds) {
        $roundId = $round.round_id
        for ($i = 0; $i -lt $round.messages.Count; $i++) {
            $msgIdx = $i + 1
            $text = $round.messages[$i]
            $outPath = Join-Path (Join-Path $OutDir $lang) "${roundId}_m${msgIdx}.wav"

            # Skip if already exists and > 1KB
            if ((Test-Path $outPath) -and ((Get-Item $outPath).Length -gt 1024)) {
                $msgCount++
                $totalSuccess++
                Write-Host "[SKIP] ${roundId} m${msgIdx} (exists, $((Get-Item $outPath).Length) bytes)" -ForegroundColor DarkGray
                continue
            }

            $sz = Invoke-TypecastTTS -Text $text -Lang $lang -OutputPath $outPath -RoundId $roundId -MsgIdx $msgIdx
            $msgCount++
            if ($sz -gt 0) {
                $totalSuccess++
                Write-Host "[OK] ${roundId} m${msgIdx}: ${sz} bytes - $($text.Substring(0, [Math]::Min(40, $text.Length)))..." -ForegroundColor Green
            } else {
                $totalFail++
            }

            # Rate limit: 0.5s between calls
            Start-Sleep -Milliseconds 500
        }
    }
    Write-Host "[INFO] $($lang.ToUpper()) done: $msgCount messages" -ForegroundColor Cyan
}

$elapsed = (Get-Date) - $startTime
Write-Host "`n[DONE] Total: success=$totalSuccess fail=$totalFail elapsed=$([Math]::Round($elapsed.TotalSeconds))s" -ForegroundColor Green
