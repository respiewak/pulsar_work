from __future__ import division, print_function

import numpy as np
import argparse as ap
from decimal import Decimal
from collections import OrderedDict


def proc_args():
    pars = ap.ArgumentParser(description="Convert a tempo2 style par file to a "
                             "psrcat-style .db file")
    pars.add_argument("pars", nargs="+", help="Par file(s) to convert")
    pars.add_argument("-o", "--out", default="new.db", help="Filename for output")
    pars.add_argument("-a", "--append", action="store_true",
                      help="Append to existing file (instead of writing new file)")
    return vars(pars.parse_args())


def read_par(parf, skip_pars):
    """
    Read in a par file and return a dictionary.

    Example:
    PSRJ           J1234-5432
    RAJ             12:34:00.0021343         1  0.00533946672769897183   
    DECJ           -54:32:10.01294           1  0.17283794918830025291   
    F0             0.32112332147535372       1  0.00000000001438903159   
    F1             -1.004051639837601107e-17 1  2.2414854767351043599e-19
    PEPOCH         57317                       
    POSEPOCH       57317                       
    DMEPOCH        57317                       
    DM             10.014152403614174525        0.22823901573735372317   
    TZRMJD         58015.899047851605093       
    TZRFRQ         1352.6409909999999854    
    TZRSITE        pks   
    EPHVER         5                           
    CLK            TT(TAI)
    MODE 1
    UNITS          TCB
    """

    all_pars = OrderedDict()

    f = open(parf, 'r')
    for line in f.readlines():
        err = None
        f_type = None
        sline = line.split()
        if sline[0] in skip_pars:
            continue
        par = sline[0]
        val = sline[1]
        if len(sline) == 3:
            err = sline[2]
        elif len(sline) == 4:
            err = sline[3]

        try:
            val = int(val)
        except ValueError:
            try:
                val = Decimal(val)
                if 'e' in sline[1] or 'E' in sline[1]:
                    f_type = 'e'
                else:
                    f_type = 'f'
            except:
                pass

        all_pars[par] = val
        if err:
            all_pars[par+"_ERR"] = Decimal(err)
        if f_type:
            all_pars[par+"_TYPE"] = f_type

    f.close()

    return all_pars


def short_err(err):
    err = Decimal(err)
    err_mag = err.logb()
    a = round(err/10**err_mag,0)
    b = float(10**max(Decimal(0), err_mag))
    return int(a*b)


def short_float(val, err):
    err = Decimal(err)
    err_mag = int(err.logb())
    a = round(float(val), -1*err_mag)
    if err_mag < 0:
        return a
    else:
        return int(a)


def pos_fmt(val, err=0, max_digits=15):
    val = str(val)
    if ":" not in val:
        try:
            dd = int(val)
            return str(dd)
        except:
            raise RuntimeError("Can only format positions of "
                               "format DD:MM:SS.S...")
    else:
        dd = val.split(":")[0]
        mm = val.split(":")[1]
        ss = None
        if len(val.split(":")) == 3:
            ss = val.split(":")[2]

    if not isinstance(err, Decimal):
        try:
            err = Decimal(err)
        except:
            raise TypeError("Uncertainty must be float-like")

    if err == Decimal(0):
        err_mag = 0
    else:
        err_mag = int(err.logb())

    if ss:
        max_digits -= (len(dd)+len(mm)+2)
        ss = short_float(ss, err)
        if err == Decimal(0):
            return "{{}}:{{}}:{{:<{}d}}".format(max_digits)\
                                        .format(dd, mm, ss)
        else:
            ss_e = short_err(err)
            if err_mag > 0:
                return "{{}}:{{}}:{{:<{}d}}{{}}"\
                    .format(max_digits).format(dd, mm, ss, ss_e)
            else:
                return "{{}}:{{}}:{{:<{}.{}f}}{{}}"\
                    .format(max_digits, -1*err_mag)\
                    .format(dd, mm, ss, ss_e)

    else:
        max_digits -= (len(dd)+1)
        mm = short_float(mm, err)
        if err == Decimal(0):
            return "{{}}:{{:<{}d}}".format(max_digits).format(dd, mm)
        else:
            mm_e = short_err(err)
            return "{{}}:{{:<{}d}}{{}}"\
                .format(max_digits).format(dd, mm, mm_e)
        

def write_db(psr_pars, out_file, skip_pars=None, append=False):
    """
    Write out psrcat-style database file for given
    pulsar parameters. 

    Example:
    @------------------------------------
    PSRJ            J1234-5432
    RAJ             12:34:00.0       2
    DECJ            -54:32:10.0      4
    F0              0.321123321      4
    F1              -1.0e-17         5
    DM              10.0             2                     
    PEPOCH          57317                    
    POSEPOCH        57317                    
    DMEPOCH         57317                    
    TZRMJD          58219.348129996        
    TZRFRQ          1372.0350340000     
    TZRSITE         pks   
    EPHEM           DE421                    
    EPHVER          2                        
    UNITS           TDB                      
    @------------------------------------
    """

    sep_str = "@"+('-'*30)+"\n"
    max_digits = 15

    fmt_e = "{{:<{0}s}}{{:<{0}s}}{{}}\n".format(max_digits+2)
    fmt = "{{:<{0}s}}{{}}\n".format(max_digits+2)

    # convert dictionaries to lists
    if isinstance(psr_pars, dict):
        psr_pars = [psr_pars[A] for A in psr_pars]

    if append:
        mode = 'a'
    else:
        mode = 'w'

    f = open(out_file, mode)
    f.write(sep_str)
    for P in psr_pars:
        for par in P.keys():
            if "_ERR" in par or "_TYPE" in par:
                continue

            if par in ["RAJ", "DECJ"]:
                # use formatting function
                if par+"_ERR" in P.keys():
                    out_str = pos_fmt(P[par], P[par+"_ERR"],
                                      max_digits+2)
                else:
                    out_str = pos_fmt(P[par], 0, max_digits+2)
                line_fmt = fmt.format(par, out_str)
            
            elif par+"_ERR" in P.keys():
                # process with uncertainty
                val = P[par]
                val_mag = val.logb()
                err = P[par+"_ERR"]
                err_mag = err.logb()
                val = short_float(val, err)
                err = short_err(err)
                if err_mag < 0:
                    #if -1*err_mag > max_digits:
                    #    raise RuntimeError("Not enough digits for {}"
                    #                       .format(par))
                    if par+"_TYPE" in P.keys() and P[par+"_TYPE"] == 'e':
                        val_str = "{{:.{}e}}".format(val_mag-err_mag)\
                                             .format(val)
                    else:
                        val_str = "{{:.{}f}}".format(-1*err_mag)\
                                             .format(val)
                    line_fmt = fmt_e.format(par, val_str, err)
                else:
                    line_fmt = fmt_e.format(par, str(val), err)
            else:
                # no uncertainty
                val = P[par]
                if isinstance(val, float) or isinstance(val, Decimal):
                    val = "{{:<{}.8f}}".format(max_digits).format(val)
                line_fmt = fmt.format(par, val)

            f.write(line_fmt)
                
        f.write(sep_str)

    f.close()


def main(pars, out, append=False):
    skip_pars = ["TRES", "MODE", "TIMEEPH", "NITS",
                 "NTOA", "CHI2R", "JUMP", "DILATEFREQ",
                 "PLANET_SHAPIRO", "T2CMETHOD", "NE_SW",
                 "CORRECT_TROPOSPHERE", "START", "FINISH"]

    all_pars = []
    for parf in pars:
        all_pars.append(read_par(parf, skip_pars))

    write_db(all_pars, out, append)


if __name__ == "__main__":
    args = proc_args()
    main(**args)
