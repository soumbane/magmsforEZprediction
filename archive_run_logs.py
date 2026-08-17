# Archive the logs of a training or evaluation run into a committable form
#
# The shards are launched with --show_verbose, so tqdm redraws its progress bar thousands of times
# per node and every redraw is kept in the log file. That inflates a run to hundreds of megabytes of
# which ~99% is superseded copies of the same progress line.
#
# This keeps only the final render of each line and gzips the result, which preserves every settings
# header, epoch summary and completion record while shrinking a run from ~750MB to ~7MB.
#
# Usage:
#   python archive_run_logs.py logs_add_root run_logs/addcohort_training
#   python archive_run_logs.py logs_add_eval run_logs/addcohort_evaluation
import argparse
import glob
import gzip
import os
import shutil


def collapse(raw: bytes) -> bytes:
    r"""
    Drop superseded progress-bar renders.

    A tqdm redraw rewrites the current line by emitting a carriage return, so only the text after
    the last `\r` of a line was ever visible.
    """
    return b"\n".join(line.split(b"\r")[-1] for line in raw.split(b"\n"))


def archive(src_dir: str, dest_dir: str) -> None:
    r"""
    Args:
        src_dir (str): The live log directory of a run.
        dest_dir (str): Where to write the archived copies.
    """
    os.makedirs(dest_dir, exist_ok=True)

    total_raw = total_out = 0

    for path in sorted(glob.glob(os.path.join(src_dir, "*.log"))):
        with open(path, "rb") as f:
            raw = f.read()

        out = collapse(raw)

        dest = os.path.join(dest_dir, os.path.basename(path) + ".gz")
        with gzip.open(dest, "wb") as f:
            f.write(out)

        total_raw += len(raw)
        total_out += os.path.getsize(dest)

        print(f"  {os.path.basename(path):<32}{len(raw)/1e6:8.1f} MB -> {os.path.getsize(dest)/1e6:6.2f} MB")

    # the PID file is tiny and records which processes produced the run
    pids = os.path.join(src_dir, "pids.txt")
    if os.path.exists(pids):
        shutil.copy2(pids, os.path.join(dest_dir, "pids.txt"))

    if total_raw:
        print(f"\n{src_dir} -> {dest_dir}: {total_raw/1e6:.0f} MB -> {total_out/1e6:.2f} MB ({100*total_out/total_raw:.2f}%)")
    else:
        print(f"No .log files found in {src_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archive a run's logs into a committable form.")
    parser.add_argument("src_dir", type=str, help="The live log directory of a run, e.g. logs_add_root.")
    parser.add_argument("dest_dir", type=str, help="Where to write the archived copies, e.g. run_logs/addcohort_training.")
    args = parser.parse_args()

    archive(args.src_dir, args.dest_dir)

    print("\nDone!")
