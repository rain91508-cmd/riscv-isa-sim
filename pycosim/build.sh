#!/usr/bin/env bash
# Build the pyspike_cosim CPython extension against the fork's libriscv.
#
# Deliberately NOT part of the spike autotools build: the .so is only imported
# when co-simulation is enabled (ADR-0013 D21), and keeping it out of `make`
# means a normal spike build (and therefore the CL regression) is untouched.
#
#   usage: ./build.sh [output-dir]        (default: .)
#
# Requires: spike configured+built out-of-tree (see spike-build/), gem5's
# vendored pybind11 headers, and CPython 3.12 dev headers.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # <fork>/pycosim
SRC="$(dirname "$HERE")"                                  # <fork>
OUT="${1:-$HERE}"

# Build dir: the out-of-tree spike build that produced libriscv.so.
BUILD="${SPIKE_BUILD:-$(cd "$SRC/.." && pwd)/spike-build}"
# pybind11: gem5 vendors it, no pip install needed.  The fork is a gem5
# submodule, so walk up from it looking for ext/pybind11/include.
default_pybind() {
  local d="$SRC"
  for _ in 1 2 3 4 5 6; do
    d="$(dirname "$d")"
    if [ -d "$d/ext/pybind11/include/pybind11" ]; then
      echo "$d/ext/pybind11/include"; return 0
    fi
  done
  return 1
}
PYBIND="${PYBIND11_INC:-$(default_pybind || true)}"
[ -n "$PYBIND" ] || { echo "cannot find gem5's vendored pybind11; set PYBIND11_INC" >&2; exit 1; }
PYINC="${PYTHON_INC:-$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')}"
PYLIB_EXT="${PYTHON_EXT_SUFFIX:-$(python3 -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')}"

# boost (pulled in transitively by sim.h -> socketif.h) is installed under
# ~/.local in this environment, the same prefix spike was configured with.
BOOST="${BOOST_INC:-$HOME/.local/include}"

for p in "$BUILD/libriscv.so" "$PYBIND/pybind11/pybind11.h" "$PYINC/Python.h"; do
  [ -e "$p" ] || { echo "missing: $p" >&2; exit 1; }
done

# -I"$SRC" is what makes spike's own <fesvr/...> style includes resolve.
g++ -std=c++2a -O2 -g -fPIC -shared \
  -I"$SRC" -I"$SRC/riscv" -I"$SRC/fdt" -I"$SRC/fesvr" -I"$SRC/softfloat" \
  -I"$BUILD" -I"$BOOST" -I"$PYBIND" -I"$PYINC" \
  -DHAVE_CONFIG_H \
  "$HERE/pyspike_cosim.cc" \
  -o "$OUT/pyspike_cosim$PYLIB_EXT" \
  -L"$BUILD" -Wl,-rpath,"$BUILD" \
  -lriscv -lsoftfloat -ldisasm -lfdt \
  "$BUILD/libfesvr.a"

echo "built: $OUT/pyspike_cosim$PYLIB_EXT"
