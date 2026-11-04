#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export QA_PYTHON="$ROOT/.venv-qa/bin/python"
export PATH="$ROOT/.tools/node/bin:$PATH"
if [[ -d "$ROOT/.tools/host-libs/usr/lib/x86_64-linux-gnu" ]]; then
    export LD_LIBRARY_PATH="$ROOT/.tools/host-libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
for required in "$QA_PYTHON" "$ROOT/.tools/node/bin/node" "$ROOT/frontend/node_modules/@playwright/test/cli.js"; do
    [[ -f "$required" ]] || { printf 'Missing QA prerequisite: %s (run bash qa/setup.sh)\n' "$required" >&2; exit 1; }
done
[[ "$(bun --version)" == 1.4.4 ]] || { printf 'Bun 1.4.4 is required\n' >&2; exit 1; }
[[ "$(node --version)" == v22.23.4 ]] || { printf 'Local Node 22.23.4 is required\n' >&2; exit 1; }
cd "$ROOT"
mkdir -p qa/artifacts/logs
trap 'printf "QA gate failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR
"$QA_PYTHON" -m unittest discover -s qa/python_tests -v 2>&1 | tee qa/artifacts/logs/python.txt
# The exporter independently refuses to write answers if frozen regressions fail.
"$QA_PYTHON" -m qa.reference_cli --out qa/artifacts/reference 2>&1 | tee qa/artifacts/logs/reference.txt
bun run --cwd frontend test:unit 2>&1 | tee qa/artifacts/logs/unit.txt
ffmpeg -hide_banner -loglevel error -y -loop 1 -i qa/fixtures/images/astronaut.png \
    -vf scale=320:240 -r 15 -t 10 -pix_fmt yuv420p -f yuv4mpegpipe qa/artifacts/fake-camera.y4m
bun run --cwd frontend test:browser 2>&1 | tee qa/artifacts/logs/browser.txt
"$QA_PYTHON" -m qa.aggregate_ab 2>&1 | tee qa/artifacts/logs/aggregate.txt
bun run --cwd frontend test:production 2>&1 | tee qa/artifacts/logs/production.txt
printf 'All executable gates passed. Review required visual evidence in qa/artifacts/ab/index.html.\n'
