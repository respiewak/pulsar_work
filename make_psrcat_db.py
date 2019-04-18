from __future__ import division, print_function

#import numpy as np
import argparse as ap
from decimal import (Decimal, DivisionByZero, InvalidOperation)
from collections import OrderedDict


def proc_args():
    pars = ap.ArgumentParser(description="Convert a tempo2 style par file "
                             "to a psrcat-style .db file")
    pars.add_argument("pars", nargs="+", help="Par file(s) to convert")
    pars.add_argument("-o", "--out", default="new.db", help="Filename for "
                      "output")
    pars.add_argument("-a", "--append", action="store_true",
                      help="Append to existing file (instead of writing "
                      "new file)")
    pars.add_argument("-l", "--latex", action="store_true",
                      help="Format output as latex table(s); "
                      "uses .tex extension for output file by default")
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
        p_type = None
        sline = line.split()
        if len(sline) == 0 or sline[0] in skip_pars or line[0] == "#":
            continue

        par = sline[0]
        if par == "E":
            par = "ECC"
        elif par == "PSR":
            par += "J"

        val = sline[1]
        if len(sline) == 3 and sline[2] not in ['0', '1']:
            err = sline[2].replace('D', 'E')
        elif len(sline) == 4:
            err = sline[3].replace('D', 'E')

        try:
            val = int(val)
            p_type = 'd'
        except ValueError:
            try:
                val = Decimal(val.replace('D', 'E'))
                if 'e' in sline[1] or 'E' in sline[1].replace('D', 'E'):
                    p_type = 'e'
                else:
                    p_type = 'f'
            except InvalidOperation:
                p_type = 's'

        all_pars[par] = val
        if err:
            all_pars[par+"_ERR"] = Decimal(err)

        if p_type:
            all_pars[par+"_TYPE"] = p_type

    f.close()

    return all_pars


def short_err(err):
    err = Decimal(err)
    if err != 0:
        err_mag = err.logb()
    else:
        err_mag = Decimal(0)

    a = round(err/10**err_mag,0)
    b = float(10**max(Decimal(0), err_mag))
    return int(a*b)


def short_float(val, err, max_digits=24):
    err = Decimal(err)
    if err != 0:
        err_mag = int(err.logb())
    else:
        err_mag = 0

    try:
        if err_mag > int(Decimal(val).logb()):
            err_mag = int(Decimal(val).logb())-1
    except DivisionByZero:
        # only occurs when val == 0
        return 0

    if err_mag > max_digits:
        err_mag = max_digits
    elif err_mag < -max_digits:
        err_mag = -max_digits

    a = round(float(val), -1*err_mag)
    if -10 < err_mag < 0:
        return a
    elif err_mag < -9:
        return "{{0:.{0}f}}".format(-1*err_mag).format(Decimal(val))
    else:
        return int(a)


def short_val_err(par, val, err=None, v_type=None, max_digits=24):
    err_str = "  "
    # number of digits to keep in Sci. not. with no uncertainty
    num_exp = 3
    orbitals = ["PB", "A1", "T0", "TASC", "FB0", "SINI", "OM", "ECC",
                "E", "EPS1", "EPS2", "PBDOT", "XDOT", "OMDOT", "XPBDOT",
                "A1DOT", "M2", "MTOT", "GAMMA", "DR", "DTHETA", "DTH",
                "XOMDOT", "EPS1DOT", "EPS2DOT", "KOM", "KIN", "A0",
                "B0", "BP", "BPP", "SHAPMAX"]

    if par in orbitals:
        err = None
        num_exp = 10
    elif (par[:2] == "FB" or par[:2] == "PB") and len(par) > 2:
        try:
            num = int(par[2:])
            err = None
            num_exp = 10
        except: 
            pass

    if v_type and v_type in 'ef' and not isinstance(val, Decimal):
        raise RuntimeError("Wrong type for par: {0}".format(val))

    if v_type == 's' or (isinstance(val, str) and v_type is None):
        return ("{{0:<{0}s}}".format(max_digits).format(val), "")
    elif v_type == 'd' and val == 0 and err > val:
        return ("0", "")
    else:
        if v_type is None:
            if isinstance(val, Decimal):
                v_type = 'e' if 'e' in str(val) else 'f'
            elif isinstance(val, int):
                v_type = 'd'
            else:
                raise RuntimeError("Cannot determine value type: {0}"
                                   .format(val))

        if err is None:
            err_str = ""
            if v_type == 'f':
                # clean up trailing 0s (opt. followed by single 1) and 9s
                vs = str(val)
                if len(vs) > len(vs.rstrip('9')):
                    val_0, val_1 = vs.split('.')
                    val_1_d = Decimal(val_1)+Decimal(1)
                    if len(str(val_1_d)) > len(val_1):
                        val_0 = str(Decimal(val_0)+Decimal(1))
                        val_1_d = Decimal('0')

                    val_1_s = str(val_1_d).rstrip('0')
                    if val_1_s == '':
                        val_1_s = '0'

                    val_short = "{0}.{1}".format(val_0, val_1_s)
                elif vs[-2:] == '01' and \
                     len(vs)-2 > len(vs.rstrip('1').rstrip('0')):
                    val_0, val_1 = vs.split('.')
                    val_1_d = Decimal(val_1)-Decimal(1)
                    val_1_s = str(val_1_d).rstrip('0')
                    if val_1_s == '':
                        val_1_s = '0'

                    val_short = "{0}.{1}".format(val_0, val_1_s)
                else:
                    val_short = vs.rstrip('0')

                val = Decimal(val_short)

            if '.' in str(val) and v_type == 'f':
                err_mag = -min(len(str(val).split('.')[1]),
                               max_digits - (len(str(val).split('.')[0])+1))
            elif '.' in str(val):
                # e.g., -2.383043e-14
                val_mag = int(Decimal(val).logb())
                err_mag = val_mag-num_exp
            else:
                err_mag = 0

        elif err == 0:
            err_str += "0"
            err_mag = 0
        else:
            err_mag = Decimal(err).logb()
            if round(err/10**err_mag, 0) == 1:
                err_mag -= 1

            a = round(err/10**err_mag, 0)
            b = float(10**max(Decimal(0), err_mag))
            err_str += str(int(a*b))

        val_mag = 0 if val == 0 else int(Decimal(val).logb())
        rel_mag = max(0, val_mag - int(err_mag))

        if v_type == 'd' and err is not None:
            a = round(val/10**val_mag, rel_mag)
            b = int(10**val_mag)
            val_str = "{{0:<{0}d}}".format(max_digits).format(int(a*b))
        elif v_type == 'd':
            val_str = "{{0:<{0}d}}".format(max_digits).format(int(val))
        elif v_type == 'e':
            #if rel_mag >=6:
            #    max_digits -= 4

            val_str = "{{0:<{0}.{1}e}}".format(max_digits, rel_mag).format(val)
        elif v_type == 'f':
            val_str = "{{0:<{0}.{1}f}}".format(max_digits, int(abs(err_mag)))\
                                    .format(val)
        else:
            raise RuntimeError("Invalid value type: "+v_type)

        return (val_str, err_str)



def pos_fmt(par, val, err=None, max_digits=24):
    val = str(val)
    if ":" not in val:
        try:
            dd = int(val)
            dd_str = str(dd)
            if dd < 10:
                dd_str = "0"+dd_str

            return dd_str
        except:
            raise RuntimeError("Can only format positions of "
                               "format DD:MM:SS.S...")
    else:
        dd = val.split(":")[0]
        mm = val.split(":")[1]
        ss = None
        if len(val.split(":")) == 3:
            ss = val.split(":")[2]

    if not isinstance(err, Decimal) and err is not None:
        try:
            err = Decimal(err)
        except:
            raise TypeError("Uncertainty must be float-like")

    if err == Decimal(0):
        err_mag = 0
    elif err is None and ss is not None:
        try:
            err_mag = int(Decimal(ss.split('.')[1]).logb())+1
        except DivisionByZero:
            err_mag = 1
        except IndexError:
            err_mag = 0
    elif err is not None:
        err_mag = int(err.logb())

    if ss:
        max_digits -= len(dd)+4
        v_type = 'f' if '.' in ss else 'd'
        ss_s, err_str = short_val_err(par, Decimal(ss), err, v_type,
                                      max_digits)
        if float(ss) < 10:
            if ss_s[-1] == ' ':
                ss_short = ss_s[:-1]
            elif ss_s == '0':
                ss_short = "{{0:<{0}s}}".format(max_digits-1).format(ss_s)
            else:
                l = len(ss_s)-1
                ss_short = "{{0:<{0}f}}".format(l).format(Decimal(ss_s))

            ss_s = "0{0}".format(ss_short)

        return "{0}:{1}:{2}{3}".format(dd, mm, ss_s, err_str)

    else:
        max_digits -= (len(dd)+1)
        mm_s, mm_e = short_val_err(par, int(mm), err, 'd', max_digits)
        if err == Decimal(0):
            return "{{0}}:{{1:<{0}s}}".format(max_digits).format(dd, mm)
        else:
            return "{{0}}:{{1:<{0}s}}{{2}}"\
                .format(max_digits).format(dd, mm, mm_e)
        

def write_db(psr_pars, out_file, skip_pars=None, append=False,
             max_digits=24):
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

    sep_str = "@"+('-'*40)+"\n"

    fmt_e = "{{0:<{0}s}}{{1:<{0}s}}  {{2}}\n".format(max_digits)
    fmt = "{{0:<{0}s}}{{1}}\n".format(max_digits)

    # put dictionaries into lists
    if isinstance(psr_pars, dict):
        psr_pars = [psr_pars]

    if append:
        mode = 'a'
    else:
        mode = 'w'

    f = open(out_file, mode)
    f.write(sep_str)
    for P in psr_pars:
        for par in P.keys():
            if len(par) > 4 and ("_ERR" == par[-4:] or "_TYPE" == par[-5:])\
               or par in skip_pars:
                continue

            if par in ["RAJ", "DECJ"]:
                # use formatting function
                if par+"_ERR" in P.keys():
                    out_str = pos_fmt(par, P[par], P[par+"_ERR"],
                                      max_digits)
                else:
                    out_str = pos_fmt(par, P[par], max_digits=max_digits)

                line_fmt = fmt.format(par, out_str)
            else:
                v_type = P[par+"_TYPE"] if par+"_TYPE" in P.keys() else None
                error = P[par+"_ERR"] if par+"_ERR" in P.keys() else None
                val_str, err_str = short_val_err(par, P[par], error, v_type,
                                                 max_digits)
                line_fmt = "{{0:<{0}s}}".format(max_digits).format(par)
                line_fmt += val_str+err_str+"\n"

            f.write(line_fmt)
                
        f.write(sep_str)

    f.close()


def write_tex(psr_pars, out_file, skip_pars=None):
    pass


def main(pars, out, append=False, latex=False):
    skip_pars = ["MODE", "TIMEEPH", "NITS", "JUMP",
                 "DILATEFREQ", "PLANET_SHAPIRO",
                 "NE_SW", "START",
                 "FINISH", "T2EFAC", "TRACK"]

    all_pars = []
    for parf in pars:
        all_pars.append(read_par(parf, skip_pars))

    if latex:
        extras = ["TZRMJD", "TZRFRQ", "TZRSITE", "CLK",
                  "EPHVER"]
        write_tex(all_pars, out, skip_pars+extras)
    else:
        extras = ["TRES", "NTOA", "CHI2R", "TNRedAmp",
                  "TNEF", "TNEQ", "TNRedGam", "TNRedC"]
        write_db(all_pars, out, skip_pars+extras, append)


if __name__ == "__main__":
    args = proc_args()
    main(**args)
