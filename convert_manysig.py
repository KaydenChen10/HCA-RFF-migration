"""Convert the Hugging Face ManySig_ZF mirror (bobo-Josef/ManySig_ZF)
into the per-receiver-per-day pkl layout expected by ASKNet's WiSig loader.

Usage:
    python convert_manysig.py --src /path/to/ManySig_zf.pkl \
        --dst ./WiSig/dataset/ManySig/non_equalized

Note: the ZF mirror only contains raw (non-equalized) signals for 4 of the
12 receivers, while the ZF-equalized signals (equalized_list index 1) are
complete for all 288 (tx, rx, date) slots. We therefore export the
ZF-equalized signals. Baseline and HCA runs in this project both use these
exact files, so the comparison is fair; absolute numbers are NOT comparable
to the ASKNet paper (which uses the original non-equalized data).
"""

import argparse
import os
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to ManySig_zf.pkl")
    ap.add_argument("--dst", required=True,
                    help="output root, e.g. WiSig/dataset/ManySig/non_equalized")
    ap.add_argument("--eq_index", type=int, default=1,
                    help="0 = raw, 1 = ZF-equalized (default: 1, complete)")
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        d = pickle.load(f)

    data, rx_list = d["data"], d["rx_list"]
    n_tx = len(d["tx_list"])
    n_dates = len(d["capture_date_list"])

    n_files = 0
    for di in range(n_dates):
        dirp = os.path.join(args.dst, f"date{di + 1}")
        os.makedirs(dirp, exist_ok=True)
        for ri, rx_name in enumerate(rx_list):
            per_tx = [np.asarray(data[ti][ri][di][args.eq_index],
                                 dtype=np.float32) for ti in range(n_tx)]
            out = os.path.join(dirp, f"rx_{rx_name}_data.pkl")
            with open(out, "wb") as f:
                pickle.dump({"data": per_tx}, f, protocol=4)
            n_files += 1
    print(f"written {n_files} files under {args.dst}")


if __name__ == "__main__":
    main()
