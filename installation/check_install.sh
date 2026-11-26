#!/usr/bin/env bash
# PyGLDA v1 installation check: serial part (check_install.py) + MPI part (mpiexec -n 4).
#   bash installation/check_install.sh [ENV_NAME]     (default: the active env, else pyglda_v1)
# Exit code 0 only if all required checks pass.
# no 'set -u': conda activation hooks of magics/metview read unset variables (MAGPLUS_HOME)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-${CONDA_DEFAULT_ENV:-pyglda_v1}}"
NP="${PYGLDA_CHECK_NP:-4}"

# activate the env (conda or mamba hook) unless it is already active
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
    PFX_BASE=""; case "$ENV_NAME" in /*) PFX_BASE="$(dirname "$ENV_NAME")/miniforge3" ;; esac   # --prefix layout
    for base in "$PFX_BASE" "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}" "$HOME/miniforge3" "$HOME/mambaforge" \
                "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$base/etc/profile.d/conda.sh"; break
        fi
    done
    conda activate "$ENV_NAME" || { echo "check_install: cannot activate env '$ENV_NAME'"; exit 2; }
fi

echo "=== machine ==="
echo "host $(hostname); nproc $(nproc) (on a container/UCloud job this is the HOST, see cgroup below)"
if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max
    if [ "$q" = "max" ]; then echo "cgroup cpu.max: max (no CPU limit)"; else echo "cgroup cpu.max: $q $p = $(awk -v q="$q" -v p="$p" 'BEGIN{printf "%.1f", q/p}') vCPU"; fi
fi
if [ -r /sys/fs/cgroup/memory.max ]; then
    m=$(cat /sys/fs/cgroup/memory.max)
    if [ "$m" = "max" ]; then echo "cgroup memory.max: max (no memory limit)"; else echo "cgroup memory.max: $m B = $(awk -v m="$m" 'BEGIN{printf "%.1f", m/1e9}') GB"; fi
fi
echo "mpiexec: $(command -v mpiexec)"
echo
echo "=== serial checks (env $ENV_NAME) ==="
python "$HERE/check_install.py"; rc_serial=$?

echo
echo "=== MPI checks: mpiexec -n $NP ==="
timeout 120 mpiexec -n "$NP" python -m mpi4py.bench helloworld; rc_hello=$?
timeout 120 mpiexec -n "$NP" python "$HERE/check_install.py" --mpi | command grep -v '^Hello'; rc_mpi=${PIPESTATUS[0]}

echo
if [ $rc_serial -eq 0 ] && [ $rc_hello -eq 0 ] && [ $rc_mpi -eq 0 ]; then
    echo "CHECK_INSTALL: PASS (serial $rc_serial, helloworld $rc_hello, allreduce $rc_mpi)"; exit 0
else
    echo "CHECK_INSTALL: FAIL (serial $rc_serial, helloworld $rc_hello, allreduce $rc_mpi)"; exit 1
fi
