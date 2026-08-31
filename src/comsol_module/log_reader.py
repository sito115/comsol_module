import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def process_solver_log(log_filepaths: Path | str | list[str | Path]) -> pd.DataFrame:
    """
    Parses COMSOL log files for datasets associated specifically with
    'Time-Dependent Solver' blocks.

    Parameters:
    -----------
    log_filepaths : str or list of str
        File path(s) to the .log file(s).

    Returns:
    --------
    pd.DataFrame
        Columns: Source, Run_ID, Step, Time, Stepsize, Reciprocal_Stepsize
    """
    if isinstance(log_filepaths, (str, Path)):
        log_filepaths = [log_filepaths]

    all_records = []

    for filepath in log_filepaths:
        records = []
        current_run_timestamp = None
        run_count = 0
        in_time_dependent_solver = False  # Track active solver type

        with open(filepath, "r") as f:
            lines = f.readlines()

        for line in lines:
            line_str = line.strip()
            line_lower = line_str.lower()

            # Detect new run boundary and extract timestamp
            if line_str.startswith("Started at"):
                match = re.search(r"Started at\s+(.*?)(?:\.|$)", line_str)
                run_count += 1
                current_run_timestamp = (
                    match.group(1).strip() if match else f"Run {run_count}"
                )
                in_time_dependent_solver = False  # Reset solver state on new run
                continue

            # Detect solver headers: enable parsing only if it is a Time-Dependent Solver
            if "<----" in line_str or "solver" in line_lower:
                if "time-dependent solver" in line_lower:
                    in_time_dependent_solver = True
                elif "<----" in line_str and "time-dependent" not in line_lower:
                    in_time_dependent_solver = False

            # Skip parsing if not currently inside a Time-Dependent Solver block
            if not in_time_dependent_solver:
                continue

            # Skip intermediate memory and progress lines
            if not line_str or any(
                kw in line_str for kw in ["Current Progress", "Memory:", "Ended at"]
            ):
                continue

            parts = line_str.split()
            # Match table rows starting with integer step index
            if parts and re.match(r"^\d+$", parts[0]):
                try:
                    step = int(parts[0])
                    time_val = float(parts[1])
                    stepsize_str = parts[2]

                    # Handle non-numeric initial step indicators ('-', 'out')
                    if stepsize_str in ["-", "out"] or not re.match(
                        r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", stepsize_str
                    ):
                        stepsize_val = np.nan
                    else:
                        stepsize_val = float(stepsize_str)

                    run_id = (
                        f"Run @ {current_run_timestamp}"
                        if current_run_timestamp
                        else f"Run {run_count}"
                    )
                    records.append(
                        {
                            "Source": Path(filepath).stem,
                            "Run_ID": run_id,
                            "Step": step,
                            "Time": time_val,
                            "Stepsize": stepsize_val,
                        }
                    )
                except (ValueError, IndexError):
                    continue

        all_records.extend(records)

    df = pd.DataFrame(all_records)
    if df.empty:
        return df

    df_valid = df.dropna(subset=["Stepsize"]).copy()
    df_valid = df_valid[df_valid["Stepsize"] > 0].copy()
    df_valid["Reciprocal_Stepsize"] = 1.0 / df_valid["Stepsize"]

    return df_valid


def plot_solver_log(df: pd.DataFrame) -> Figure:
    """
    Plots reciprocal step sizes vs. simulation time and vs. solver step number,
    one line per (Source, Run_ID) group.

    Parameters:
    -----------
    df : pd.DataFrame
        Output of process_solver_log().
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    if df.empty:
        return fig

    # Group by Source + Run_ID so multiple files/runs don't collide in the legend
    for (source, run_label), df_group in df.groupby(["Source", "Run_ID"], sort=False):
        label = f"{source} | {run_label}"

        # Plot 1: 1/dt vs Simulation Time
        axes[0].plot(
            df_group["Time"],
            df_group["Reciprocal_Stepsize"],
            "o-",
            label=label,
            alpha=0.8,
            markersize=4,
        )

        # Plot 2: 1/dt vs Solver Step Number
        axes[1].plot(
            df_group["Step"],
            df_group["Reciprocal_Stepsize"],
            "o-",
            label=label,
            alpha=0.8,
            markersize=4,
        )

    # Format Plot 1
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Simulation Time $t$ [s]", fontsize=11, fontweight="bold")
    axes[0].set_ylabel(
        "Reciprocal Step Size $1/\\Delta t$ [1/s]", fontsize=11, fontweight="bold"
    )
    axes[0].set_title(
        "Reciprocal Step Size vs. Simulation Time", fontsize=12, fontweight="bold"
    )
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].legend(frameon=True, fontsize=9)

    # Format Plot 2
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Solver Step Number", fontsize=11, fontweight="bold")
    axes[1].set_ylabel(
        "Reciprocal Step Size $1/\\Delta t$ [1/s]", fontsize=11, fontweight="bold"
    )
    axes[1].set_title(
        "Reciprocal Step Size vs. Step Number", fontsize=12, fontweight="bold"
    )
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True, fontsize=9)

    return fig
