from __future__ import print_function, division

import subprocess as sproc
from shlex import split as shplit
import os
import argparse as ap
from glob import glob

from make_psrcat_db import main as mpd_main
from tempo2_test import main as ttest_main


def proc_args():
    pars = ap.ArgumentParser(description="Run all tests of par files "
                             "and write psrcat-style 'obs99.db' file")
    #pars.add_argument("pars", nargs="+", help="Par files to test/convert")
    pars.add_argument("-o", "--out", default="obs99.db",
                      help="Name of .db file to write out")
    pars.add_argument("-p", "--path", default="../ptuse_folding/",
                      help="Path to directories (<path>/raw, "
                      "<path>/converted, <path>/failure, <path>/success). "
                      "Reads pars from <path>/raw.")


def main(out, path):
    out_db = out
    test1_cmd = "python test_pars.py {} -u -m -d {}"
    test2_cmd = "python tempo2_test.py {} -r {} -d {}"

    # directories to expect in 'path': raw, converted,
    # failure(/fake, /dspsr), success
    dir_r = os.path.join(path, "raw")
    dir_c = os.path.join(path, "converted")
    dir_fd = os.path.join(path, "failure/dspsr")
    dir_ff = os.path.join(path, "failure/fake")
    dir_s = os.path.join(path, "success")
    for d in [dir_r, dir_c, dir_fd, dir_ff, dir_s]:
        if not os.path.exists(d):
            raise(RuntimeError("Path does not exist: "+d))

        if not os.access(d, os.R_OK):
            raise(RuntimeError("Necessary directories not read-able"))

        if not os.access(d, os.W_OK):
            raise(RuntimeError("Necessary directories not write-able"))

    # list of par files from /raw/
    raw_pars = os.path.join(dir_r, "*.par")
    par_files = sorted(glob(raw_pars))

    # convert par files to appropriate form
    pass

    # test par files for dspsr compatibility
    p = sproc.Popen(shplit(test1_cmd.format(raw_pars, 'dir')),
                    stdout=sproc.PIPE, stderr=sproc.PIPE)
    p.wait()
    # check errors and output? 

    # generate .db file
    conv_pars = os.path.join(dir_c, "*.par")
    par_files = sorted(glob(conv_pars))
    mpd_main(par_files, out_db, append=False)

    # check if conversion to .db files succeeded
    ## TODO: need to change tempo2_test.py to use correct pars and directories
    ttest_main(pars=raw_pars, root_dir='dir', db_file=out_db)

    # make new db file from successful pars
    good_pars = os.path.join(dir_s, "*.par")
    par_files = sorted(glob(good_pars))
    mpd_main(par_files, out_db, append=False)


if __name__ == "__main__":
    args = proc_args()
    main(**args)
