# scripts/pre-commit 을 .git/hooks/pre-commit 으로 설치한다 (Windows/PowerShell용).
# .git/hooks/ 는 저장소에 포함되지 않으므로 클론한 사람마다 한 번씩 실행해야 한다.
# 훅 자체는 Git for Windows에 포함된 sh(#!/bin/sh)로 실행되므로 이 스크립트는
# 파일을 복사하기만 하면 된다.
$ErrorActionPreference = "Stop"
$repoRoot = (git rev-parse --show-toplevel).Trim()
Copy-Item -Path (Join-Path $repoRoot "scripts/pre-commit") -Destination (Join-Path $repoRoot ".git/hooks/pre-commit") -Force
Write-Host "pre-commit 훅을 설치했습니다: $repoRoot\.git\hooks\pre-commit"
