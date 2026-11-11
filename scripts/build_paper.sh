#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../paper"
mkdir -p build ../submission
TEX_ENGINE=${TEX_ENGINE:-tectonic}
args=(--keep-logs --keep-intermediates --outdir build)
if [[ -n "${TECTONIC_BUNDLE:-}" ]]; then args+=(--bundle "$TECTONIC_BUNDLE"); fi
"$TEX_ENGINE" "${args[@]}" main.tex
cp build/main.pdf ../submission/report.pdf
# Anonymous source includes main.tex but bypasses ignored local identity file.
printf '%s\n' '\def\publicversion{1}\input{main.tex}' > public.tex
"$TEX_ENGINE" "${args[@]}" public.tex
cp build/public.pdf ../submission/report_public.pdf
