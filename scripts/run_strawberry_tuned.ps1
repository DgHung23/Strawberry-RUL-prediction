$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $PSScriptRoot
$python = "py"

Set-Location $root

$jobs = @(
    @{
        Name = "A"
        Script = "src\strawberry\stage4_training\model_A\train_tuned.py"
        Log = "notebooks\strawberry\results\train_A_tuned.log"
    }
    @{
        Name = "B"
        Script = "src\strawberry\stage4_training\model_B\train_tuned.py"
        Log = "notebooks\strawberry\results\train_B_tuned.log"
    }
    @{
        Name = "C"
        Script = "src\strawberry\stage4_training\model_C\train_tuned.py"
        Log = "notebooks\strawberry\results\train_C_tuned.log"
    }
    @{
        Name = "D"
        Script = "src\strawberry\stage4_training\model_D\train_tuned.py"
        Log = "notebooks\strawberry\results\train_D_tuned.log"
    }
)

foreach ($job in $jobs) {
    $logPath = Join-Path $root $job.Log
    New-Item -ItemType Directory -Force -Path (Split-Path $logPath) | Out-Null
    Write-Output "Starting tuned Model $($job.Name)"
    & $python -u $job.Script --seq-len 10 --batch-size 4 --epochs 10 --patience 10 --learning-rate 3e-4 --weight-decay 1e-4 --hidden-size 128 --num-layers 1 --dropout 0.35 --env-hidden-size 32 --fusion-mode late_env_branch --temporal-pooling last_mean_max --loss smooth_l1 *> $logPath
    if ($LASTEXITCODE -ne 0) {
        Write-Output "Model $($job.Name) failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Output "Finished tuned Model $($job.Name)"
}

Write-Output "All tuned strawberry model runs completed."
