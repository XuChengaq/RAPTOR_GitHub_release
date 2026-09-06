$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonExe = 'C:\ProgramData\anaconda3\envs\phyattentionnet\python.exe'
Set-Location -LiteralPath $projectRoot
& $pythonExe -s run_aoa_pipeline.py 2>&1 | Tee-Object -FilePath 'logs\aoa_pipeline.stdout.log' -Append
exit $LASTEXITCODE
