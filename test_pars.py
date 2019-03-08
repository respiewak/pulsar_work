from __future__ import print_function, division

import subprocess as sproc
import shlex
import os
import sys
from glob import glob
import argparse
import re

pars = argparse.ArgumentParser(description="Check par files for "
                               "compatability with dspsr/tempo2 "
                               "(and move to 'folding_ephems' dir.)")
pars.add_argument('pars', nargs="+", help="Par files to process; e.g., "
                  "`ls -t *.par` (sorting by time recommended)")
pars.add_argument('-u', '--update', action="store_true",
                  help="Flag to indicate updating old ephemerides")
args = vars(pars.parse_args())

sched_dir = "/fred/oz002/software/meersched/"
# check names to avoid repetition (assume first par is best)
check_names = []

log_file = sched_dir+'par_testing/bad_pars.log'
if len(glob(log_file)) > 0:
    os.remove(log_file)

for par in args['pars']:
    try:
        jname = sproc.Popen(shlex.split('grep PSRJ {}'.format(par)),
                            stdout=sproc.PIPE).communicate()[0].split()[1]
    except IndexError:
        with open(log_file, 'a') as f:
            f.write("ERROR: Could not find PSRJ in par file {}; skipping\n"
                    .format(par))
        continue

    if not jname in check_names:
        check_names.append(jname)
    else:
        with open(log_file, 'a') as f:
            f.write("WARNING: Multiple par files for {}; skipping {}\n"
                    .format(jname, par))
        continue

    if len(glob(sched_dir+'folding_ephems/'+os.path.split(par)[1])) > 0 \
       and not args['update']:
        continue

    user = sproc.Popen(shlex.split('whoami'), stdout=sproc.PIPE)\
                .communicate()[0].rstrip('\n')
    tempo_file = '/tmp/tempo2/{}/stdout.txt'.format(user)

    if len(glob(tempo_file)) > 0:
        os.remove(tempo_file)

    header = sched_dir+'par_testing/header.dada'
    with open(header, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sline = line.split()
            if len(sline) > 0 and sline[0] == 'SOURCE':
                old_name = sline[1]

    with open(header, 'w') as f:
        for line in lines:
            f.write(re.sub(old_name, jname, line))


    dm = None
    p0 = None
    with open(par, 'r') as f:
        for line in f.readlines():
            sline = line.split()
            if len(sline) > 0:
                if sline[0] == "P0":
                    p0 = float(sline[1].replace("D", "E"))

                elif sline[0] == "F0":
                    p0 = 1/float(sline[1].replace("D", "E"))

                elif sline[0] == "DM":
                    dm = float(sline[1])

    with open(log_file, 'a') as f:
        if p0 is None:
            p0 = 1
            f.write('WARNING: Could not get period from par file {}; '
                    'assuming 1s\n'.format(par))

        if dm is None:
            f.write("ERROR: Could not get DM from par file {}; skipping\n"
                    .format(par))
            continue

    kernel = None
    p = sproc.Popen(shlex.split("dmsmear -f 1070 -b 428 -n 512 -d {}"
                                .format(dm)), stderr=sproc.PIPE)\
        .communicate()[1]
    for pline in p.split('\n'):
        if "Minimum Kernel" in pline:
            kernel = int(pline.split()[3])

    if kernel is None:
        with open(log_file, 'a') as f:
            f.write("ERROR: Failed to get kernel length from dmsmear; "
                    "skipping {}\n".format(par))
        continue

    dspsr_cmd = 'dspsr {} -T {} -Q -minram 1024 -cuda 0 -no_dyn -x {} -E {}'\
        .format(header, int(2*p0 + 1), kernel, par)
    p = sproc.Popen(shlex.split(dspsr_cmd), stderr=sproc.PIPE)
    p.wait()
    err = p.communicate()[1]
    if "Error::message" in err:
        kernel *= 2
        dspsr_cmd = 'dspsr {} -T {} -Q -minram 1024 -cuda 0 -no_dyn -x '\
                    '{} -E {}'.format(header, int(2*p0 + 1), kernel, par)
        p = sproc.Popen(shlex.split(dspsr_cmd), stderr=sproc.PIPE)
        p.wait()
        err = p.communicate()[1]
        if "Error::message" in err:
            with open(log_file, 'a') as f:
                f.write("ERROR: Could not process with dspsr: {}\n".format(par))

            continue

    with open(tempo_file, 'r') as f:
        for line in f.readlines():
            sline = line.split()
            if len(sline) > 0:
                if sline[0] == "RMS":
                    rms = float(sline[3])

    with open(log_file, 'a') as f:
        if rms > p0:
            f.write('ERROR: RMS larger than period for {}\n'.format(par))
        else:
            sproc.call(shlex.split('cp {} {}folding_ephems/.'
                                   .format(par, sched_dir)))


