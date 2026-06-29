# Build batch WAV files from per-message WAVs for Sony recorder playback.
# Output: pilot_ko_batch.wav, pilot_en_batch.wav (44.1kHz stereo PCM16)
# Gap: 8s silence between messages, 2s prefix, 0.2s pre/post padding

param(
    [string]$OutDir = "assets\e2e_voice_inputs"
)

$ErrorActionPreference = "Stop"

$SAMPLE_RATE = 44100
$CHANNELS_OUT = 2        # stereo output
$BITS_PER_SAMPLE = 16
$BYTES_PER_SAMPLE = 2
$SILENCE_PREFIX_S = 2.0
$SILENCE_PRE_S = 0.2
$SILENCE_POST_S = 0.2
$SILENCE_GAP_S = 8.0

function New-SilenceBytes([double]$Seconds, [int]$Channels) {
    $frameCount = [int]($Seconds * $SAMPLE_RATE)
    $byteCount = $frameCount * $Channels * $BYTES_PER_SAMPLE
    return (New-Object byte[] $byteCount)
}

function Read-WavPcmData([string]$Path) {
    # Read WAV file, return PCM data bytes + metadata
    $bytes = [System.IO.File]::ReadAllBytes($Path)

    # Parse RIFF header
    $riff = [System.Text.Encoding]::ASCII.GetString($bytes, 0, 4)
    if ($riff -ne "RIFF") { throw "Not a RIFF file: $Path" }

    # Find 'fmt ' chunk
    $pos = 12
    $fmtFound = $false
    $audioFormat = 0
    $numChannels = 0
    $sampleRate = 0
    $bitsPerSample = 0

    while ($pos -lt ($bytes.Length - 8)) {
        $chunkId = [System.Text.Encoding]::ASCII.GetString($bytes, $pos, 4)
        $chunkSize = [BitConverter]::ToUInt32($bytes, $pos + 4)

        if ($chunkId -eq "fmt ") {
            $audioFormat = [BitConverter]::ToUInt16($bytes, $pos + 8)
            $numChannels = [BitConverter]::ToUInt16($bytes, $pos + 10)
            $sampleRate = [BitConverter]::ToUInt32($bytes, $pos + 12)
            $bitsPerSample = [BitConverter]::ToUInt16($bytes, $pos + 22)
            $fmtFound = $true
        }

        if ($chunkId -eq "data") {
            $dataStart = $pos + 8
            $dataLen = [int]$chunkSize
            if (($dataStart + $dataLen) -gt $bytes.Length) {
                $dataLen = $bytes.Length - $dataStart
            }
            $pcm = New-Object byte[] $dataLen
            [Array]::Copy($bytes, $dataStart, $pcm, 0, $dataLen)

            return @{
                Pcm = $pcm
                Channels = $numChannels
                SampleRate = $sampleRate
                BitsPerSample = $bitsPerSample
            }
        }

        $pos += 8 + [int]$chunkSize
        if ($pos % 2 -ne 0) { $pos++ }  # RIFF chunks are word-aligned
    }

    throw "No data chunk found in: $Path"
}

function ConvertTo-Stereo([byte[]]$MonoPcm) {
    # Duplicate mono samples to stereo (16-bit)
    $stereo = New-Object byte[] ($MonoPcm.Length * 2)
    for ($i = 0; $i -lt $MonoPcm.Length; $i += 2) {
        $offset = $i * 2
        $stereo[$offset]     = $MonoPcm[$i]
        $stereo[$offset + 1] = $MonoPcm[$i + 1]
        $stereo[$offset + 2] = $MonoPcm[$i]
        $stereo[$offset + 3] = $MonoPcm[$i + 1]
    }
    return $stereo
}

function Write-WavFile([string]$Path, [byte[]]$PcmData, [int]$Channels) {
    $dataSize = $PcmData.Length
    $byteRate = $SAMPLE_RATE * $Channels * $BYTES_PER_SAMPLE
    $blockAlign = $Channels * $BYTES_PER_SAMPLE
    $fileSize = 36 + $dataSize

    $ms = New-Object System.IO.MemoryStream
    $bw = New-Object System.IO.BinaryWriter($ms)

    # RIFF header
    $bw.Write([System.Text.Encoding]::ASCII.GetBytes("RIFF"))
    $bw.Write([uint32]$fileSize)
    $bw.Write([System.Text.Encoding]::ASCII.GetBytes("WAVE"))

    # fmt chunk
    $bw.Write([System.Text.Encoding]::ASCII.GetBytes("fmt "))
    $bw.Write([uint32]16)           # chunk size
    $bw.Write([uint16]1)            # PCM format
    $bw.Write([uint16]$Channels)
    $bw.Write([uint32]$SAMPLE_RATE)
    $bw.Write([uint32]$byteRate)
    $bw.Write([uint16]$blockAlign)
    $bw.Write([uint16]$BITS_PER_SAMPLE)

    # data chunk
    $bw.Write([System.Text.Encoding]::ASCII.GetBytes("data"))
    $bw.Write([uint32]$dataSize)
    $bw.Write($PcmData)

    $bw.Flush()
    [System.IO.File]::WriteAllBytes($Path, $ms.ToArray())
    $bw.Close()
    $ms.Close()
}

function Build-BatchWav([string]$Lang) {
    $langDir = Join-Path $OutDir $Lang
    $wavFiles = Get-ChildItem $langDir -Filter "*.wav" | Sort-Object Name
    $count = $wavFiles.Count
    Write-Host "[INFO] $($Lang.ToUpper()): $count WAV files found" -ForegroundColor Cyan

    # Collect all PCM segments
    $allSegments = New-Object System.Collections.ArrayList

    # Prefix silence (stereo)
    [void]$allSegments.Add((New-SilenceBytes $SILENCE_PREFIX_S $CHANNELS_OUT))

    for ($i = 0; $i -lt $count; $i++) {
        $wav = $wavFiles[$i]
        $info = Read-WavPcmData $wav.FullName

        if ($info.SampleRate -ne $SAMPLE_RATE) {
            Write-Host "[WARN] $($wav.Name): sample rate $($info.SampleRate) != $SAMPLE_RATE" -ForegroundColor Yellow
        }

        # Convert mono to stereo if needed
        $pcm = $info.Pcm
        if ($info.Channels -eq 1) {
            $pcm = ConvertTo-Stereo $pcm
        }

        # Pre-silence
        [void]$allSegments.Add((New-SilenceBytes $SILENCE_PRE_S $CHANNELS_OUT))
        # Message audio
        [void]$allSegments.Add($pcm)
        # Post-silence
        [void]$allSegments.Add((New-SilenceBytes $SILENCE_POST_S $CHANNELS_OUT))

        # Inter-message gap (except after last)
        if ($i -lt ($count - 1)) {
            [void]$allSegments.Add((New-SilenceBytes $SILENCE_GAP_S $CHANNELS_OUT))
        }

        Write-Host "[OK] $($wav.Name): $($info.Pcm.Length) bytes ($($info.Channels)ch)" -ForegroundColor Green
    }

    # Concatenate all segments
    $totalBytes = 0
    foreach ($seg in $allSegments) { $totalBytes += $seg.Length }

    $result = New-Object byte[] $totalBytes
    $offset = 0
    foreach ($seg in $allSegments) {
        [Array]::Copy($seg, 0, $result, $offset, $seg.Length)
        $offset += $seg.Length
    }

    # Write output
    $outPath = Join-Path $OutDir "pilot_${Lang}_batch.wav"
    Write-WavFile -Path $outPath -PcmData $result -Channels $CHANNELS_OUT

    $durationS = $totalBytes / ($SAMPLE_RATE * $CHANNELS_OUT * $BYTES_PER_SAMPLE)
    $fileSizeMB = [Math]::Round((Get-Item $outPath).Length / 1MB, 1)
    $minutes = [Math]::Floor($durationS / 60)
    $seconds = [Math]::Round($durationS % 60)
    $secStr = $seconds.ToString("00")
    Write-Host "[DONE] $outPath -> ${fileSizeMB}MB, ${minutes}:${secStr}" -ForegroundColor Green
}

Write-Host "`n=== Building batch WAV files ===" -ForegroundColor Cyan
$startTime = Get-Date

Build-BatchWav "ko"
Write-Host ""
Build-BatchWav "en"

$elapsed = [Math]::Round(((Get-Date) - $startTime).TotalSeconds)
Write-Host "`n[ALL DONE] elapsed=${elapsed}s" -ForegroundColor Green
