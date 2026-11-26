#!/usr/bin/env bash
# PyGLDA v1 (py-w3ra) environment installer (Linux x86_64).
#
#   bash installation/install.sh                 # env 'pyglda_v1' from the explicit lock (reproducible, no solve)
#   bash installation/install.sh --solve         # solve installation/environment.yml instead (newer builds possible)
#   bash installation/install.sh --env NAME      # other env name
#   bash installation/install.sh --force         # remove and recreate an existing env
#   bash installation/install.sh --vis           # also the optional visualisation env 'pyglda_vis' (pygmt)
#   bash installation/install.sh --no-check      # skip check_install.sh at the end
#   bash installation/install.sh --prefix DIR    # HPC / UCloud: everything under DIR (persistent storage):
#        DIR/miniforge3   private Miniforge, bootstrapped (sha256-verified) if no conda is on PATH
#        DIR/<env>        the env(s), created by path
#        DIR/activate.sh  source this in every shell
#        DIR/ucloud_init.sh  UCloud job "Initialization" script (activate.sh in every shell, tmux, 2-rank MPI check)
#
# Idempotent: an existing env is verified, not rebuilt (hooks rewritten, checks run) unless --force.
# Log: installation/logs/install_<date>_<time>.log (git-ignored), or DIR/install_<date>_<time>.log with --prefix.
# No 'set -u': conda activation hooks of magics read unset variables.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="pyglda_v1"; MODE="lock"; FORCE=0; VIS=0; CHECK=1; PREFIX=""
# Miniforge for the --prefix bootstrap: a fixed release (its .sha256 exists only for the versioned file name)
MINIFORGE_VERSION="${PYGLDA_MINIFORGE_VERSION:-26.9.4-2}"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/$MINIFORGE_VERSION/Miniforge3-$MINIFORGE_VERSION-Linux-x86_64.sh"
while [ $# -gt 0 ]; do
    case "$1" in
        --env) ENV_NAME="$2"; shift 2 ;;
        --env=*) ENV_NAME="${1#*=}"; shift ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --prefix=*) PREFIX="${1#*=}"; shift ;;
        --solve) MODE="solve"; shift ;;
        --force) FORCE=1; shift ;;
        --vis) VIS=1; shift ;;
        --no-check) CHECK=0; shift ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "install.sh: unknown option $1 (see --help)"; exit 2 ;;
    esac
done

if [ -n "$PREFIX" ]; then
    mkdir -p "$PREFIX" || { echo "ERROR: cannot create prefix $PREFIX"; exit 1; }
    PREFIX="$(cd "$PREFIX" && pwd)"
    LOG="$PREFIX/install_$(date +%Y%m%d_%H%M%S).log"
else
    mkdir -p "$HERE/logs"
    LOG="$HERE/logs/install_$(date +%Y%m%d_%H%M%S).log"
fi
exec > >(tee -a "$LOG") 2>&1
echo "PyGLDA v1 install: env=$ENV_NAME mode=$MODE force=$FORCE vis=$VIS check=$CHECK prefix=${PREFIX:-<none>}  ($(date '+%Y-%m-%d %H:%M:%S'), host $(hostname))"

# ---------------------------------------------------------------- find (or bootstrap) conda / mamba
CONDA_BASE=""
if [ -n "$PREFIX" ] && [ -f "$PREFIX/miniforge3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$PREFIX/miniforge3"; echo "using the private Miniforge of the prefix"
fi
if [ -z "$CONDA_BASE" ] && [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
fi
if [ -z "$CONDA_BASE" ]; then
    c="$(command -v mamba || command -v conda || true)"
    [ -n "$c" ] && CONDA_BASE="$(dirname "$(dirname "$(readlink -f "$c")")")"
fi
if [ -z "$CONDA_BASE" ] && [ -z "$PREFIX" ]; then       # desktop mode only: usual install locations
    for b in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        [ -f "$b/etc/profile.d/conda.sh" ] && { CONDA_BASE="$b"; break; }
    done
fi
if [ -z "$CONDA_BASE" ]; then
    [ -n "$PREFIX" ] || { echo "ERROR: no conda/mamba found. Install Miniforge (https://github.com/conda-forge/miniforge), or use --prefix DIR to bootstrap a private one."; exit 1; }
    echo "no conda/mamba found: bootstrapping a private Miniforge into $PREFIX/miniforge3"
    command -v curl >/dev/null || { echo "ERROR: curl is needed for the Miniforge bootstrap"; exit 1; }
    INST="$PREFIX/Miniforge3-Linux-x86_64.sh"
    curl -fsSL -o "$INST" "$MINIFORGE_URL" && curl -fsSL -o "$INST.sha256" "$MINIFORGE_URL.sha256" \
        || { echo "ERROR: Miniforge download failed ($MINIFORGE_URL)"; exit 1; }
    want="$(awk '{print $1}' "$INST.sha256")"; have="$(sha256sum "$INST" | awk '{print $1}')"
    [ -n "$want" ] && [ "$want" = "$have" ] || { echo "ERROR: Miniforge sha256 mismatch (want $want, have $have)"; exit 1; }
    echo "Miniforge installer sha256 verified: $have"
    bash "$INST" -b -p "$PREFIX/miniforge3" || { echo "ERROR: Miniforge install failed"; exit 1; }
    rm -f "$INST" "$INST.sha256"
    CONDA_BASE="$PREFIX/miniforge3"
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
SOLVER="conda"; [ -x "$CONDA_BASE/bin/mamba" ] && SOLVER="$CONDA_BASE/bin/mamba"
echo "conda base: $CONDA_BASE   solver for --solve: $SOLVER   ($(conda --version))"

# env target: by name (desktop) or by path (prefix)
env_path() {   # name -> prefix path of the env, empty if it does not exist
    if [ -n "$PREFIX" ]; then
        [ -x "$PREFIX/$1/bin/python" ] && echo "$PREFIX/$1"
    else
        conda env list | awk -v n="$1" '$1==n {print $NF}'
    fi
}
target() { if [ -n "$PREFIX" ]; then echo "-p $PREFIX/$1"; else echo "-n $1"; fi; }

# ---------------------------------------------------------------- create one env
create_env() {   # name, lockfile, ymlfile
    local name="$1" lock="$2" yml="$3" t0 prefix
    prefix="$(env_path "$name")"
    if [ -n "$prefix" ] && [ $FORCE -eq 1 ]; then
        echo "removing existing env $name ($prefix) (--force)"
        # shellcheck disable=SC2046
        conda env remove -y $(target "$name") || return 1
        prefix=""
    fi
    if [ -n "$prefix" ]; then
        echo "env $name exists ($prefix): kept, will be verified (use --force to recreate)"
        return 0
    fi
    t0=$(date +%s)
    # shellcheck disable=SC2046
    if [ "$MODE" = "lock" ]; then
        [ -f "$lock" ] || { echo "ERROR: lock file $lock missing (use --solve)"; return 1; }
        echo "creating $name from $(basename "$lock") (explicit, no solve)"
        conda create -y $(target "$name") --file "$lock" || return 1
    else
        echo "creating $name by solving $(basename "$yml")"
        "$SOLVER" env create -y $(target "$name") -f "$yml" || return 1
    fi
    echo "created $name in $(( $(date +%s) - t0 )) s"
}

# ---------------------------------------------------------------- activation hooks
write_hooks() {   # name
    local prefix; prefix="$(env_path "$1")"
    mkdir -p "$prefix/etc/conda/activate.d" "$prefix/etc/conda/deactivate.d"
    cat > "$prefix/etc/conda/activate.d/zz_pyglda_v1.sh" <<'EOF'
# written by py-w3ra installation/install.sh: environment for PyGLDA v1 / metview
export _PYGLDA_OLD_PROJ_LIB="${PROJ_LIB-__unset__}"
export PROJ_LIB="$CONDA_PREFIX/share/proj"          # older GDAL/pyproj code reads PROJ_LIB (PROJ >= 9.1 reads PROJ_DATA)
export _PYGLDA_OLD_PROJ_DATA="${PROJ_DATA-__unset__}"
export PROJ_DATA="$CONDA_PREFIX/share/proj"
EOF
    cat > "$prefix/etc/conda/deactivate.d/zz_pyglda_v1.sh" <<'EOF'
# written by py-w3ra installation/install.sh
if [ "${_PYGLDA_OLD_PROJ_LIB-}" = "__unset__" ]; then unset PROJ_LIB; elif [ -n "${_PYGLDA_OLD_PROJ_LIB+x}" ]; then export PROJ_LIB="$_PYGLDA_OLD_PROJ_LIB"; fi
if [ "${_PYGLDA_OLD_PROJ_DATA-}" = "__unset__" ]; then unset PROJ_DATA; elif [ -n "${_PYGLDA_OLD_PROJ_DATA+x}" ]; then export PROJ_DATA="$_PYGLDA_OLD_PROJ_DATA"; fi
unset _PYGLDA_OLD_PROJ_LIB _PYGLDA_OLD_PROJ_DATA
EOF
    echo "activation hooks written: $prefix/etc/conda/{activate,deactivate}.d/zz_pyglda_v1.sh"
}

# ---------------------------------------------------------------- prefix mode: activate.sh + ucloud_init.sh
write_prefix_scripts() {
    cat > "$PREFIX/activate.sh" <<EOF
# PyGLDA v1 - source this in every shell:   source $PREFIX/activate.sh
# (written by py-w3ra installation/install.sh on $(date +%F); ucloud_init.sh adds it to ~/.bashrc)
# Puts the env first on PATH (its python, mpiexec and metview) and sets the env's activation variables.
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$PREFIX/$ENV_NAME"
EOF
    cat > "$PREFIX/ucloud_init.sh" <<EOF
#!/usr/bin/env bash
# PyGLDA v1 - select this file as the UCloud job "Initialization" script.
# (written by py-w3ra installation/install.sh on $(date +%F))
# Every new shell sources activate.sh; tmux gets mouse scrolling and a long scrollback; a 2-rank MPI check with the
# env's own mpiexec is written to $PREFIX/last_job_init.log.
LINE="source $PREFIX/activate.sh"
RC="\$HOME/.bashrc"
grep -qF "\$LINE" "\$RC" 2>/dev/null || printf '\n# PyGLDA v1 env (added by ucloud_init.sh)\n%s\n' "\$LINE" >> "\$RC"
# tmux: user overrides go to ~/.tmux.conf.local when it exists ("Oh my tmux" setups: ~/.tmux.conf must not be edited)
TC="\$HOME/.tmux.conf"; [ -f "\$HOME/.tmux.conf.local" ] && TC="\$HOME/.tmux.conf.local"
grep -q '^set -g mouse on' "\$TC" 2>/dev/null || printf '\n# PyGLDA v1 (added by ucloud_init.sh)\nset -g mouse on\nset -g history-limit 50000\n' >> "\$TC"
source "$PREFIX/activate.sh" >/dev/null 2>&1
{
  echo "job init \$(date '+%Y-%m-%d %H:%M:%S') on \$(hostname); mpiexec = \$(command -v mpiexec)"
  echo "cgroup cpu.max: \$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)   memory.max: \$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)"
  timeout 120 mpiexec -n 2 python -c "from mpi4py import MPI; c=MPI.COMM_WORLD; print('rank %d of %d' % (c.Get_rank(), c.Get_size()), flush=True)" 2>&1
} > "$PREFIX/last_job_init.log"
EOF
    chmod +x "$PREFIX/ucloud_init.sh"
    echo "written: $PREFIX/activate.sh and $PREFIX/ucloud_init.sh"
}

T0=$(date +%s)
create_env "$ENV_NAME" "$HERE/linux-64.lock.txt" "$HERE/environment.yml" || { echo "INSTALL FAILED ($ENV_NAME)"; exit 1; }
write_hooks "$ENV_NAME"
if [ $VIS -eq 1 ]; then
    create_env "pyglda_vis" "$HERE/linux-64-vis.lock.txt" "$HERE/environment-vis.yml" \
        || echo "WARNING: optional visualisation env pyglda_vis failed (core env is fine)"
fi
[ -n "$PREFIX" ] && write_prefix_scripts
echo "install time: $(( $(date +%s) - T0 )) s"

RC=0
ENV_REF="$ENV_NAME"; [ -n "$PREFIX" ] && ENV_REF="$PREFIX/$ENV_NAME"
if [ $CHECK -eq 1 ]; then
    echo; echo "running checks"
    conda activate "$ENV_REF" && bash "$HERE/check_install.sh" "$ENV_REF"; RC=$?
    conda deactivate 2>/dev/null
fi
echo
if [ -n "$PREFIX" ]; then
    echo "done (exit $RC). Use:  source $PREFIX/activate.sh ; cd demo ; mpiexec -n <ens+1> python <driver>.py"
else
    echo "done (exit $RC). Use:  conda activate $ENV_NAME ; cd demo ; mpiexec -n <ens+1> python <driver>.py"
fi
echo "log: $LOG"
exit $RC
