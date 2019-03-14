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
pars.add_argument('-m', '--move_bool', action='store_false',
                  help="Do not move par files into 'folding_ephems' dir.")
pars.add_argument('-d', '--src_dir', default="./",
                  help="specify directory for log and header.dada; "
                  "assume 'folding_ephems' dir. (if any) is one dir. up")
args = vars(pars.parse_args())

move_bool = args['move_bool']
src_dir = args['src_dir']
if src_dir[-1] != '/':
    src_dir +='/'

if not os.path.exists(src_dir):
    raise(RuntimeError("Not valid path: "+src_dir))

#sched_dir = "/fred/oz002/software/meersched/"
# check names to avoid repetition (assume first par is best)
check_names = []

if src_dir == './':
    fold_dir = "../"
else:
    fold_dir = os.path.split(os.path.split(src_dir)[0])[0]+'/'

if not os.path.exists(fold_dir+"folding_ephems") and move_bool:
    raise(RuntimeError("Could not find folding_ephems dir. in "+fold_dir))

fold_dir += "folding_ephems/"

log_file = src_dir+'bad_pars.log'
if len(glob(log_file)) > 0:
    os.remove(log_file)

for par in args['pars']:
    try:
        jname = sproc.Popen(shlex.split('grep PSRJ {0}'.format(par)),
                            stdout=sproc.PIPE).communicate()[0].split()[1]
    except IndexError:
        with open(log_file, 'a') as f:
            f.write("ERROR: Could not find PSRJ in par file {0}; skipping\n"
                    .format(par))
        continue

    if not jname in check_names:
        check_names.append(jname)
    else:
        with open(log_file, 'a') as f:
            f.write("WARNING: Multiple par files for {0}; skipping {}\n"
                    .format(jname, par))
        continue

    if len(glob(fold_dir+os.path.split(par)[1])) > 0 \
       and not args['update']:
        continue

    user = sproc.Popen(shlex.split('whoami'), stdout=sproc.PIPE)\
                .communicate()[0].rstrip('\n')
    tempo_file = '/tmp/tempo2/{0}/stdout.txt'.format(user)

    if len(glob(tempo_file)) > 0:
        os.remove(tempo_file)

    header = src_dir+'header.dada'
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
            f.write('WARNING: Could not get period from par file {0}; '
                    'assuming 1s\n'.format(par))

        if dm is None:
            f.write("ERROR: Could not get DM from par file {0}; skipping\n"
                    .format(par))
            continue

    kernel = None
    p = sproc.Popen(shlex.split("dmsmear -f 1070 -b 428 -n 512 -d {0}"
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

    dspsr_cmd = 'dspsr {0} -T {1} -Q -minram 1024 -cuda 0 -no_dyn -x {2} -E {3}'\
        .format(header, int(2*p0 + 1), kernel, par)
    p = sproc.Popen(shlex.split(dspsr_cmd), stderr=sproc.PIPE)
    p.wait()
    err = p.communicate()[1]
    if "Error::message" in err:
        kernel *= 2
        dspsr_cmd = 'dspsr {0} -T {1} -Q -minram 1024 -cuda 0 -no_dyn -x '\
                    '{2} -E {3}'.format(header, int(2*p0 + 1), kernel, par)
        p = sproc.Popen(shlex.split(dspsr_cmd), stderr=sproc.PIPE)
        p.wait()
        err = p.communicate()[1]
        if "Error::message" in err:
            with open(log_file, 'a') as f:
                f.write("ERROR: Could not process with dspsr: {0}\n".format(par))

            continue

    with open(tempo_file, 'r') as f:
        for line in f.readlines():
            sline = line.split()
            if len(sline) > 0:
                if sline[0] == "RMS":
                    rms = float(sline[3])

    with open(log_file, 'a') as f:
        if rms > p0:
            f.write('ERROR: RMS larger than period for {0}\n'.format(par))
        elif move_bool:
            sproc.call(shlex.split('cp {0} {1}.'.format(par, fold_dir)))


