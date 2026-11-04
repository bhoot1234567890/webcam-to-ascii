#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/.tools"
NODE_VERSION=22.23.4
NODE_ARCHIVE="node-v${NODE_VERSION}-linux-x64.tar.xz"
NODE_SHA256=1805bcd35e6815559bd2eb97db438d3fdb0eb3a5e6a977b7c1badca76337e9ce
PYTHON_VERSION=3.12.16
export UV_PYTHON_INSTALL_DIR="$TOOLS/python"
export UV_CACHE_DIR="$TOOLS/uv-cache"
# Optional unprivileged host-library installation (see README).
if [[ -d "$TOOLS/host-libs/usr/lib/x86_64-linux-gnu" ]]; then
    export LD_LIBRARY_PATH="$TOOLS/host-libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

fail() { printf 'Setup failed: %s\n' "$*" >&2; exit 1; }
trap 'printf "Setup failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || \
    fail 'qa/setup.sh currently requires Linux x64 for the pinned Node artifact'
for command in bun curl tar sha256sum mktemp tee; do
    command -v "$command" >/dev/null || fail "Required command is unavailable: $command"
done
[[ "$(bun --version)" == 1.4.4 ]] || fail 'Bun 1.4.4 is required'
for path in qa/requirements.in qa/requirements.lock frontend/package.json frontend/bun.lock; do
    [[ -s "$ROOT/$path" ]] || fail "Missing or empty committed dependency file: $ROOT/$path"
done
mkdir -p "$TOOLS/downloads"
ARCHIVE="$TOOLS/downloads/$NODE_ARCHIVE"
if [[ ! -f "$ARCHIVE" ]]; then
    curl --fail --location --retry 2 \
        --output "$ARCHIVE.partial" "https://nodejs.org/dist/v${NODE_VERSION}/$NODE_ARCHIVE"
    printf '%s  %s\n' "$NODE_SHA256" "$ARCHIVE.partial" | sha256sum --check --status || \
        fail "SHA-256 mismatch: $ARCHIVE.partial"
    mv -- "$ARCHIVE.partial" "$ARCHIVE"
fi
printf '%s  %s\n' "$NODE_SHA256" "$ARCHIVE" | sha256sum --check --status || \
    fail "SHA-256 mismatch: $ARCHIVE"
if [[ ! -x "$TOOLS/node/bin/node" ]] || [[ "$("$TOOLS/node/bin/node" --version)" != "v$NODE_VERSION" ]]; then
    NODE_STAGE="$(mktemp -d "$TOOLS/node-install.XXXXXX")"
    tar -xJf "$ARCHIVE" --strip-components=1 -C "$NODE_STAGE"
    [[ "$("$NODE_STAGE/bin/node" --version)" == "v$NODE_VERSION" ]] || \
        fail "Incorrect Node version extracted from $ARCHIVE"
    if [[ -e "$TOOLS/node" ]]; then
        fail "Existing $TOOLS/node is invalid; remove that local installation explicitly and rerun"
    fi
    mv -- "$NODE_STAGE" "$TOOLS/node"
fi
"$TOOLS/node/bin/node" --version

if [[ ! -x "$TOOLS/uv/uv" ]]; then
    curl --fail --location --retry 2 --output "$TOOLS/downloads/uv-install.sh" https://astral.sh/uv/install.sh
    UV_UNMANAGED_INSTALL="$TOOLS/uv" sh "$TOOLS/downloads/uv-install.sh"
fi
"$TOOLS/uv/uv" --version
"$TOOLS/uv/uv" python install "$PYTHON_VERSION"
if [[ ! -e "$ROOT/.venv-qa" ]]; then
    "$TOOLS/uv/uv" venv --managed-python --python "$PYTHON_VERSION" "$ROOT/.venv-qa"
fi
[[ -x "$ROOT/.venv-qa/bin/python" ]] || fail "Invalid virtual environment: $ROOT/.venv-qa"
[[ "$("$ROOT/.venv-qa/bin/python" -c 'import platform; print(platform.python_version())')" == "$PYTHON_VERSION" ]] || \
    fail "Expected Python $PYTHON_VERSION in $ROOT/.venv-qa"
# The committed lock is resolved once, not silently regenerated during setup.
"$TOOLS/uv/uv" pip sync --python "$ROOT/.venv-qa/bin/python" --require-hashes "$ROOT/qa/requirements.lock"
"$ROOT/.venv-qa/bin/python" -c 'import sys, numpy, PIL, skimage, cv2; print("Python", sys.version.split()[0]); print("NumPy", numpy.__version__); print("Pillow", PIL.__version__); print("scikit-image", skimage.__version__); print("OpenCV", cv2.__version__)'

bun install --cwd "$ROOT/frontend" --frozen-lockfile
# Playwright's installer reports missing host libraries as warnings with exit 0.
# Retain its diagnostics and turn that condition into a hard setup failure.
INSTALL_LOG="$TOOLS/playwright-install.log"
env -u PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD -u PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS \
    PATH="$TOOLS/node/bin:$PATH" \
    "$TOOLS/node/bin/node" "$ROOT/frontend/node_modules/@playwright/test/cli.js" \
    install chromium firefox webkit 2>&1 | tee "$INSTALL_LOG"
if grep -Eq 'Playwright Host validation warning|Host system is missing dependencies' "$INSTALL_LOG"; then
    fail "Playwright host libraries are missing; see named prerequisites in $INSTALL_LOG (no sudo was attempted)"
fi
printf 'QA tooling ready at %s\n' "$ROOT"
