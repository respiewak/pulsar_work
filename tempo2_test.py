from __future__ import print_function, division
import argparse, os
import numpy as np
import subprocess as sproc
from glob import glob
from shlex import split as shp
try:
    from astropy.time import Time
    apy_bool = True
except:
    apy_bool = False

pars = argparse.ArgumentParser()
pars.add_argument("psrs", nargs="+")
pars.add_argument("-r", "--root_dir", default="../")
pars.add_argument("-d", "--db_file", default="test.db")
args = vars(pars.parse_args())

root_dir = args['root_dir']
psrs = args['psrs']
db_file = args['db_file']

if root_dir[-1] != '/':
    root_dir += '/'

psrcat_call = "psrcat -db_file {} -E {}"
psrcat_call2 = "psrcat -c 'jname rajd decj p0 dm' {}"
temp_par = "tmp.par"
t2_call = "tempo2 -gr fake -ndobs 14 -nobsd 1 -randha y -ha 12 -start"\
              " 57000 -end 59000 -rms {} -f {}"
temp_sim = "tmp.simulate"
t2_call2 = "tempo2 -nofit -newpar -f tmp.par tmp.simulate"
mv_call = "mv {} {}"
cp_call = "cp {} {}"
fold_eph_dir = "second_ephems"

if apy_bool:
    t = Time.now()
    temp_log = "test_{:.4f}.log".format(t.mjd)
else:
    blob = glob("test*log")
    n = len(blob)+1
    temp_log = "test_{}.log".format(n)

if not os.path.exists(root_dir+db_file) \
   or not os.path.exists(root_dir+fold_eph_dir):
    raise RuntimeError("Invalid paths")

for psr in psrs:
    with open(temp_par, 'w') as f:
        p = sproc.Popen(shp(psrcat_call.format(root_dir+db_file, psr)),
                        stdout=sproc.PIPE)
        p.wait()
        f.write(p.communicate()[0])

    psr_par = glob(root_dir+fold_eph_dir+"/{}*par".format(psr))
    if len(psr_par) > 1:
        with open(temp_log, 'a') as f:
            print("Found more than one par file: {}{}/{}*par"
                  .format(root_dir, fold_eph_dir, psr))
            f.write(psr+"\t#FAILED\n")
            continue
        #raise RuntimeError("Found more than one par file: {}{}/"
        #                   "{}*par".format(root_dir, fold_eph_dir, psr))
    elif len(psr_par) == 0:
        with open(temp_log, 'a') as f:
            print("Could not find par file: {}{}/{}*par"
                  .format(root_dir, fold_eph_dir, psr))
            p = sproc.Popen(shp(psrcat_call2.format(psr)), stdout=sproc.PIPE,
                            stderr=sproc.PIPE)
            p.wait()
            err = p.communicate()[0].split('\n')[0]
            if "not in catalogue" in err:
                print("Pulsar {} not in psrcat".format(psr))
                f.write(psr+"\t#FAILED\n")
            else:
                f.write(psr+"\t#SKIPPED\n")

            continue
        #raise RuntimeError("Could not find par file: {}{}/{}*par"
        #                   .format(root_dir, fold_eph_dir, psr))
    else:
        psr_par = psr_par[0]

    with open(psr+".par", 'w') as f:
        p = sproc.Popen(shp("grep -v T2EFAC {}".format(psr_par)),
                        stdout=sproc.PIPE)
        p.wait()
        f.write(p.communicate()[0])

    psr_par = psr+".par"
    psr_p0 = None
    with open(psr_par, 'r') as f:
        for line in f.readlines():
            if line.split()[0] == "F0":
                psr_p0 = 1/float(line.split()[1])
            elif line.split()[0] == "P0":
                psr_p0 = float(line.split()[1])

    if psr_p0 is None:
        with open(temp_log, 'a') as f:
            print("Could not find P0 or F0 in "+psr_par)
            f.write(psr+"\t#FAILED\n")
            continue
        #raise RuntimeError("Could not find P0 or F0 in "+psr_par)
    else:
        psr_rms = "{:.6f}".format(psr_p0)

    p = sproc.Popen(shp(t2_call.format(psr_rms, psr_par)),
                    stdout=sproc.PIPE, stderr=sproc.PIPE)
    p.wait()
    sproc.call(shp(mv_call.format("{}.simulate".format(psr), temp_sim)))

    p = sproc.Popen(shp(t2_call2), stdout=sproc.PIPE, stderr=sproc.PIPE)
    p.wait()

    if not os.path.exists("new.par"):
        raise RuntimeError("No par file saved by tempo2")

    rms_out = None
    with open("new.par", 'r') as f:
        for line in f.readlines():
            if line.split()[0] == "TRES":
                rms_out = float(line.split()[1])

    with open(temp_log, 'a') as f:
        if rms_out is None:
            print("Could not read TRES from output par file: {}".format(psr))
            f.write(psr+"\t#FAILED\n")
            #raise RuntimeError("Could not read TRES from output par file")
        else:
            if rms_out/1e3 > 2*psr_p0:
                print("WARNING: RMS greater than period for {}".format(psr))
                f.write(psr+"\t#FAILED\n")
            else:
                f.write(psr+"\t#Passed\n")

    os.remove(psr_par)
    os.remove(temp_sim)
    os.remove(temp_par)
    os.remove('new.par')
