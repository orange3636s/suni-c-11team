#!/bin/sh
# scripts/pre-commit 을 .git/hooks/pre-commit 으로 설치한다.
# .git/hooks/ 는 저장소에 포함되지 않으므로 클론한 사람마다 한 번씩 실행해야 한다.
set -e
repo_root=$(git rev-parse --show-toplevel)
cp "$repo_root/scripts/pre-commit" "$repo_root/.git/hooks/pre-commit"
chmod +x "$repo_root/.git/hooks/pre-commit"
echo "pre-commit 훅을 설치했습니다: $repo_root/.git/hooks/pre-commit"
