#!/usr/bin/env bash
# Runs DEGAS2 on the converted test case (needs DEGAS2_SRC: DEGAS2 source dir
# with data/, and DEGAS2_BIN: dir with datasetup, problemsetup, defineback,
# tallysetup, flighttest). Usage: ci/degas2_acceptance.sh <convert2degas2>
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
cp -r "$here/tests/assets/simple_degas2_case/degas2-case/." "$work"
sed -i "s#/[^ ]*/degas2/data/#${DEGAS2_SRC}/data/#" "$work/degas2.in"
cd "$work"
"$1" "$here/tests/assets/simple_degas2_case/tagged-dg2mesh.osh" -o geometry.nc -a --force
"$DEGAS2_BIN/datasetup"
"$DEGAS2_BIN/problemsetup"
"$DEGAS2_BIN/defineback" db.in
"$DEGAS2_BIN/tallysetup"
"$DEGAS2_BIN/flighttest"
test -s output.nc
echo "DEGAS2 accepted geometry.nc ($work)"
