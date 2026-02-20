import os
import sys
import platform
from datetime import datetime


def env():
    mpi_available = False
    mpi_running = False

    try:
        from mpi4py import MPI
        mpi_available = True
        # Check if MPI was initialized
        if MPI.Is_initialized():
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
            mpi_running = size > 1
            # print(rank)
        else:
            # Not running under mpirun/mpiexec
            comm = MPI.COMM_SELF
            size = 1
            rank = -1
            mpi_running = False
    except ImportError:
        mpi_available = False
        comm = None
        size = 1
        rank = -1
        mpi_running = False

    # Usage
    # if mpi_running:
    #     print(f"Running with MPI: {size} processes, rank {rank}")
    # else:
    #     print("Running in serial mode")

    return rank


def print_pyglda_banner(version="1.0.0"):
    if env() != 0: return

    use_color = sys.stdout.isatty()

    def c(text, code):
        return f"\033[{code}m{text}\033[0m" if use_color else text

    BLUE = "34"
    CYAN = "36"
    GREEN = "32"
    GRAY = "90"
    BOLD = "1"

    line = "=" * 70
    subline = "-" * 70

    # Adjusted block ASCII logo — wider P and A
    logo = f"""
{c('█████   ██   ██   ██████   ██       ██████     █████ ', BLUE + ";" + BOLD)}
{c('█    █   ██ ██   ██        ██       ██   ██   █     █', BLUE + ";" + BOLD)}
{c('█████     ███    ██  ███   ██       ██   ██   ███████', BLUE + ";" + BOLD)}
{c('█         ██     ██   ██   ██       ██   ██   █     █', BLUE + ";" + BOLD)}
{c('█         ██      ██████   ██████   ██████    █     █', BLUE + ";" + BOLD)}
"""

    # Print banner
    print(c(line, CYAN))
    print(logo)
    print(c("Python Global Land Data Assimilation System", BLUE))
    print(c(line, CYAN))
    print()
    print(f"{c('Version', GREEN):14} : {version}")
    print(f"{c('Institution', GREEN):14} : Aalborg University – Geodesy Group")
    print(f"{c('DOI', GREEN):14} : 10.5194/gmd-2024-125")
    print(f"{c('Repository', GREEN):14} : github.com/AAUGeodesyGroup/PyGLDA")
    # print()
    # print("Ensemble-based global land water data assimilation")
    # print("Multi-source satellite integration (e.g., GRACE/GRACE-FO)")
    print(c(subline, CYAN))
    print()
    print(f"{c('Start Time', GRAY):14} : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{c('Python', GRAY):14} : {platform.python_version()}")
    print(f"{c('Platform', GRAY):14} : {platform.system()} {platform.release()}")
    print(c(line, CYAN))
    print('\n\n')
