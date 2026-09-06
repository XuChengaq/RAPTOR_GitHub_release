$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
$python = 'C:\ProgramData\anaconda3\envs\phyattentionnet\python.exe'
$stdout = Join-Path $project 'logs\linearno_formal.stdout.log'
$stderr = Join-Path $project 'logs\linearno_formal.stderr.log'
$done = @(
    (Join-Path $project 'outputs\linearno_formal\linearno_tiny\reynolds-linearno_tiny-s42\result.json'),
    (Join-Path $project 'outputs\linearno_formal\linearno_tiny\aoa-linearno_tiny-s42\result.json')
)

Set-Location -LiteralPath $project
New-Item -ItemType Directory -Force -Path (Join-Path $project 'logs') | Out-Null
"QUEUE_START $(Get-Date -Format o)" | Add-Content -LiteralPath $stdout

while ($true) {
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(.exe)?$' -and
        ($_.CommandLine -match 'run_aoa_pipeline.py' -or
         $_.CommandLine -match 'configs[/\\]aoa_.*formal.yaml')
    })
    if ($active.Count -eq 0) { break }
    Start-Sleep -Seconds 60
}

if (($done | Where-Object { Test-Path -LiteralPath $_ }).Count -eq $done.Count) {
    "QUEUE_SKIP already complete $(Get-Date -Format o)" | Add-Content -LiteralPath $stdout
    exit 0
}

"CUDA_SMOKE_START $(Get-Date -Format o)" | Add-Content -LiteralPath $stdout
& $python -s scripts\smoke_linearno.py 1>> $stdout 2>> $stderr
if ($LASTEXITCODE -ne 0) { throw "LinearNO CUDA smoke failed with exit code $LASTEXITCODE" }

"FORMAL_START $(Get-Date -Format o)" | Add-Content -LiteralPath $stdout
& $python -s run_modern_baselines.py --config configs\linearno_formal.yaml 1>> $stdout 2>> $stderr
if ($LASTEXITCODE -ne 0) { throw "LinearNO formal runner failed with exit code $LASTEXITCODE" }
"FORMAL_DONE $(Get-Date -Format o)" | Add-Content -LiteralPath $stdout
