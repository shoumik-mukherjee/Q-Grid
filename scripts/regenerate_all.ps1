# Regenerate every result, replay file and generated doc, core (portal + headline docs) first.
#   powershell -ExecutionPolicy Bypass -File scripts\regenerate_all.ps1
# Logs each step with start/end timestamps to results\regenerate.log; stops at the first failing step.
# Run on an otherwise idle machine: the time-limited local MILPs depend on CPU load.

$ErrorActionPreference = "Stop"
$Root = "D:\QGrid"
$Py   = "D:\QGrid\.venv\Scripts\python.exe"
$Log  = Join-Path $Root "results\regenerate.log"
Set-Location $Root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$Root\backend;$Root"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

function Invoke-Step([string]$name, [string]$cmdline, [string]$dir = $Root) {
    Write-Log "START $name :: $cmdline"
    $t0 = Get-Date
    # cmd merges stderr into stdout, so PowerShell 5.1 never wraps stderr lines as errors
    & cmd.exe /d /c "cd /d $dir && $cmdline 2>&1" | ForEach-Object {
        Write-Host $_
        Add-Content -Path $Log -Value $_ -Encoding UTF8
    }
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -ne 0) {
        Write-Log "FAILED $name (exit $code after $mins min)"
        Write-Host ""
        Write-Host "Regeneration stopped. Failed step: $name" -ForegroundColor Red
        Write-Host "Command: $cmdline" -ForegroundColor Red
        exit $code
    }
    Write-Log "END   $name ($mins min)"
}

Write-Log "==== regenerate_all.ps1 started (git $(git rev-parse --short HEAD))"

# ---------------------------------------------------------------- CORE: portal + headline docs
# The hybrid-Benders schedules in results\v2 were computed on the old campus data. make_replay.py adds them to the
# replay as runs.qgrid_v2 whenever they exist, and the portal's story view then shows that run as "Q-Grid".
# Move them aside so the core replay uses only the current design; run_benchmark_v2 rebuilds results\v2 later.
if (Test-Path "$Root\results\v2") {
    $dest = "$Root\results\v2_previous_" + (Get-Date -Format "yyyyMMdd_HHmm")
    Move-Item "$Root\results\v2" $dest
    Write-Log "moved stale results\v2 -> $(Split-Path $dest -Leaf)"
}

Invoke-Step "01 processed data"            "$Py -m qgrid.data"
Invoke-Step "02 benchmark v3"              "$Py scripts\run_bench_v3.py"
Invoke-Step "03 ablation"                  "$Py scripts\run_ablation.py"
Invoke-Step "04 wear variant"              "$Py scripts\run_wear_variant.py --workers=8"
Invoke-Step "05 quantum placement"         "$Py scripts\quantum_placement.py"
Invoke-Step "06 QAOA parameters"           "$Py scripts\train_qaoa_params.py"
Invoke-Step "07 benchmark v1"              "$Py scripts\run_benchmark.py"
Invoke-Step "08 problem day"               "$Py scripts\problem_day.py"
Invoke-Step "09 QAOA study"                "$Py scripts\qaoa_study.py"
Invoke-Step "10 write bench v3"            "$Py scripts\write_bench_v3.py"
Invoke-Step "11 write consistency"         "$Py scripts\write_consistency.py"
Invoke-Step "12 write impact"              "$Py scripts\write_impact.py"
Invoke-Step "13 make replay"               "$Py scripts\make_replay.py"
Invoke-Step "14 make live"                 "$Py scripts\make_live.py"
Invoke-Step "15 make sankey"               "$Py scripts\make_sankey.py"
Invoke-Step "16 record operate"            "$Py scripts\record_operate.py"
Invoke-Step "17 write results md"          "$Py scripts\write_results_md.py"
Write-Log "CORE DONE"

# ---------------------------------------------------------------- LEGACY: earlier designs only
Invoke-Step "18 benchmark v2"              "$Py scripts\run_benchmark_v2.py"
Invoke-Step "19 benchmark v2 retry"        "$Py scripts\run_benchmark_v2.py --retry-failed"
Invoke-Step "20 hardware v2 (no tokens)"   "$Py scripts\hardware_v2.py"
Invoke-Step "21 write results md (again)"  "$Py scripts\write_results_md.py"

# ---------------------------------------------------------------- checks
Invoke-Step "22 manifest"                  "$Py scripts\make_manifest.py"
Invoke-Step "23 verify"                    "$Py scripts\verify.py"
Invoke-Step "24 verify --ci"               "$Py scripts\verify.py --ci"
Invoke-Step "25 pytest"                    "$Py -m pytest backend\tests -p no:warnings"
Invoke-Step "26 contrast test"             "npm run test:contrast" "$Root\frontend"
Write-Log "==== ALL DONE"
