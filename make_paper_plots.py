from __future__ import print_function, division

import os
import sys
import re
import numpy as np
import pandas as pd
import argparse as ap
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import psrchive
import subprocess as sproc
from shlex import split as shplit

sys.path.append("/fred/oz002/users/rspiewak/code/")
#sys.path.append("/fred/oz002/users/rspiewak/code/psrqpy/")
#import psrqpy
from flux_analysis import (get_flux_data as gfd, get_snr_data as gsd,
                           beam_corr as bc)
import ppdot_plot as ppd
from mt_pipe_results import make_subplots


__all__ = ['read_archive', 'get_weighted_data',
           'plot_pols', 'calc_pa', 'centre_phase', 'plot_Scyl',
           'get_stats', 'plot_waterfall', 'plot_single_prof',
           'plot_histos', 'plot_rrat_wfall', 'do_scrunch',
           'get_res_fromtim', 'plot_toas_fromarr', 'make_indiv_polprofs',
           'make_all_polprof', 'make_S_plots', 'calc_pol_fracs']


def _check_zoom(s):
    msg = "{} is not a valid zoom option".format(s)
    msg += "\nFormat options: 0,0.1; 0x0.1; m0.1,0.1; m0.1x0.1; m0.2,m0.1; m0.2xm0.1"
    if s == 'auto':
        return(s)
    elif 'x' in s:
        try:
            a = float(s.split('x')[0].replace('m', '-'))
            b = float(s.split('x')[1].replace('m', '0'))
        except ValueError:
            raise ap.ArgumentTypeError(msg)
    elif ',' in s:
        try:
            a = float(s.split(',')[0].replace('m', '-'))
            b = float(s.split(',')[1].replace('m', '-'))
        except ValueError:
            raise ap.ArgumentTypeError(msg)
    else:
        raise ap.ArgumentTypeError(msg)

    return(s)


def proc_args():
    pars = ap.ArgumentParser(description="Make plots for SUPERB V")
    pars.add_argument("-p1", "--ppdot", action="store_true",
                      help="Make P-Pdot diagram; requires path to db_file")
    pars.add_argument("-d", "--db_file", help="Path to db_file")
    pars.add_argument("-p2", "--profiles", action="store_true",
                      help="Plot grid of all pulsar profiles")
    pars.add_argument("-p3", "--histos", action="store_true",
                      help="Plot grid of all pulsar flux histograms")
    pars.add_argument("-r", "--rrat", dest="rrat_file",
                      help="Plot waterfall and profile of J1646-1910 "
                      "from given data file")
    pars.add_argument("-S", "--plot_pol", action="store_true")
    #pars.add_argument("-s", "--plot_onlypol", action="store_true")
    pars.add_argument("-A", "--archives", nargs="+")
    pars.add_argument("-T", "--tscrunch", nargs="?", const=True, default=False,
                      help="Scrunch in time (use just the flag to fully "
                      "T-scrunch; else give an integer)")
    pars.add_argument("-F", "--fscrunch", nargs="?", const=True, default=False,
                      help="Scrunch in freq. (use just the flag to fully "
                      "F-scrunch; else give an integer)")
    pars.add_argument("-b", "--bscrunch", type=int,
                      help="Scrunch bins by this factor")
    pars.add_argument("-p", "--pscrunch", action="store_true",
                      help="Polarisation scrunch")
    pars.add_argument("-g", "--output", default="1/xs",
                      help="Output file for plotting (1/xs -> plt.show)")
    pars.add_argument("-C", "--centre", action="store_true")
    pars.add_argument("-z", "--zoom_phase", type=_check_zoom,
                      help="Zoom on phase (specified as XxY or X,Y or "
                      "'auto' to automatically zoom on main pulse; "
                      "implies -C).")
    pars.add_argument("-y", "--zoom_y", type=_check_zoom,
                      help="Zoom the y-axis")
    pars.add_argument("-w", "--waterfall", action="store_true")
    pars.add_argument("-W", "--wfall_wprof", action="store_true",
                      help="Plot waterfall with profile in upper panel")
    pars.add_argument("-R", "--residuals", dest="toa_plot",
                      action="store_true")
    pars.add_argument("-t", "--toa_cfg", help="File with config. params "
                      "for ToA residual plot")
    pars.add_argument("-p4I", "--plot_pols_I", action="store_true",
                      help="Plot pol. profs individually")
    pars.add_argument("-p4S", "--plot_pols_S", action="store_true",
                      help="Plot pol. profs in single pages")
    pars.add_argument("-N", "--shape", type=_check_zoom, default="3x3",
                      help="Layout for pol. profs")
    pars.add_argument("-v", "--verbose", action="store_true",
                      help="Set verbose printing (default: warnings only)")
    pars.add_argument("-P", "--report_pols", action="store_true",
                      help="Print the polarisation fractions for given files")
    pars.add_argument('--census', help="Print period and DM on pol plots",
                      action='store_true')
    pars.add_argument("--widths", help="CSV file with source names and baseline widths")
    pars.add_argument("--pa_err", dest="pa_err_file", help="File containing "
                      "data for calculating PA uncertainties")
    pars.add_argument("--no_dedisp", action="store_true", help="Do not dedisperse.")
    pars.add_argument("--vmax", type=float)
    pars.add_argument("--high_res", help="Use 300dpi for images instead of 100",
                      action="store_true")
    pars.add_argument("--extras", help="Extra information in a csv file")
    pars.add_argument("--no_norm", action="store_true", help="Do not normalize")
    pars.add_argument("--rotate", help="CSV containing extra rotation values")
    pars.add_argument("--iquv", help="Plot IQUV instead of ILV", action="store_true")
    return(vars(pars.parse_args()))


def make_pretty(fontsize=14, high_res=False):
    plt.rc('font', **{'family': 'sans-serif',
                     'sans-serif': ['DejaVu Sans']})
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    #plt.rc('text', usetex=True)
    if high_res:
        plt.rc('figure', dpi=300)

    return(fontsize)


def config_parser(cfg_name):
    """
    Extract a dictionary of information from a given config. file
    """

    config_params = {}
    with open(str(cfg_name)) as cfile:
        for line in cfile.readlines():
            sline = line.split("=")
            attr = sline[0].rstrip()
            par = sline[1].rstrip().lstrip()
            if attr == 'flags':
                config_params['flags'] = par.split(',')
            else:
                config_params[attr] = par

    return(config_params)


def read_archive(filename, base_wid="normal", no_dedisp=False):
    """
    Returns psrchive.archive object. 
    Data are dedispersed and have state converted to Stokes.
    """

    if not type(filename) is str:
        raise(RuntimeError("Wrong type of variable: {}".format(filename)))
    elif not os.path.exists(filename):
        raise(RuntimeError("No such file: "+filename))

    a = psrchive.Archive_load(filename)
    if not no_dedisp:
        a.dedisperse()

    if base_wid is None:
        base_wid = "normal"

    if type(base_wid) is str:
        a.execute("edit off="+base_wid)
    elif type(base_wid) in (float, np.float64):
        a.execute("edit off:smooth:width={}".format(base_wid))

    a.remove_baseline()
    if a.get_npol() > 1:
        a.convert_state("Stokes")

    return(a)


def get_weighted_data(archive, ichn=0, isub=0):
    if type(archive) is not psrchive.Archive:
        raise(RuntimeError("Wrong type for archive: "+type(archive)))

    if (archive.get_nsubint() == 1 and isub == ':') or \
       (archive.get_nchan() == 1 and ichn == ':'):
        raise(RuntimeError("Check dimensions of data"))

    if isub == ':' and ichn != ':':
        data = archive.get_data()[:,0,ichn,:]
    elif ichn == ':' and isub != ':':
        data = archive.get_data()[isub,0,:,:]
    else:
        raise(RuntimeError("Not enough dimensions"))

    if ichn == 0 and isub != 0:
        mask = archive.get_weights()
    else:
        mask = archive.get_weights().transpose()

    return(np.multiply(mask * np.ones((1, np.shape(data)[1])), data))


def make_indiv_polprof(archives, out_dir="./", tscrunch=True, fscrunch=True, high_res=False):
    fs = make_pretty(14, high_res)
    for filename in archives:
        ar = read_archive(filename)
        name_ext = os.path.basename(filename)
        name_new = os.path.splitext(name_ext)[0]+".png"
        out_png = os.path.join(out_dir, name_new)

        plt.clf()
        plot_Scyl(ar, fscr=fscrunch, tscr=tscrunch, fs=fs, out=out_png)


def plot_pols(archive, ax=None, isub=0, ichn=0, tscr=None,
              fscr=None, bscr=None, phase_ref=None,
              xlims=None, arr_roll=None, norm=False,
              add_inset=False, inset_lims=None, base_width=None,
              plot_wide=True, plot_iquv=False):
    if type(archive) is not psrchive.Archive:
        raise(RuntimeError("Wrong type for archive: {}".format(type(archive))))

    axins = None
    if add_inset and inset_lims is None:
        if xlims is None:
            inset_lims = [0.0, 0.1]
        else:
            inset_lims = [xlims[0]-0.5, xlims[1]-0.5]

    a = do_scrunch(archive, tscr, fscr, bscr)
    if tscr is True:
        isub = 0

    if fscr is True:
        ichn = 0

    if a.get_state() != "Stokes":
        a.convert_state("Stokes")

    #if type(base_width) in [float, np.float64]:
    #    a.execute("edit off:smooth:width={}".format(base_width))
    #elif type(base_width) is str:
    #    a.execute("edit off="+base_width)

    #a.remove_baseline()
    bstats = a.get_Integration(isub).baseline_stats()
    if ax is None:
        ax = plt.gca()

    int_sI = a[isub].get_Profile(0, ichn).get_amps()#-bstats[0][0][ichn]
    phase = np.linspace(0, 1, len(int_sI), endpoint=False)
    if arr_roll is not None:
        int_sI = np.roll(int_sI, arr_roll, axis=0)

    #if phase_ref is not None:
    #    phase = centre_phase(int_sI, phase, phase_ref)[0]

    if xlims is None:
        lim = phase < 10
    else:
        # make sure to plot a bit extra on either end
        lim = np.logical_and(phase > xlims[0]*0.9, phase < xlims[1]*1.1)

    prof_I = int_sI[lim]
    if norm:
        I_norm = prof_I.max()
        prof_I /= I_norm
    else:
        I_norm = None

    if plot_wide and xlims is None:
        # lim.all() is True
        ax.plot(np.concatenate((phase-1, phase, phase+1)),
                np.concatenate((prof_I, prof_I, prof_I)), 'k-', lw=0.75)
        ax.set_xlim(-0.05, 1.05)
    else:
        ax.plot(phase[lim], prof_I, 'k-', lw=0.75)
        ax.set_xlim(xlims)

    if add_inset:
        # default location for inset axis is upper right
        axins = inset_axes(ax, width="30%", height="40%", borderpad=1.1)
        # check if the range to be plotted is in the phase range
        # roll a new array if not (by half?)
        if inset_lims[0] < phase.min() or inset_lims[1] > phase.max():
            roll_ins = int(len(phase)/2.)
            prof_I_ins = np.roll(prof_I, roll_ins, axis=0)
            phase_ins = phase-0.5
            if inset_lims[1] > phase.max():
                phase_ins += 1
        else:
            prof_I_ins = prof_I


        axins.plot(phase_ins, prof_I_ins, 'k-', lw=0.75)
        axins.set_xlim(inset_lims)

        prof_lim = prof_I_ins[np.logical_and(phase_ins > inset_lims[0],
                                             phase_ins < inset_lims[1])]
        max_ins = max(prof_lim)*1.1
        min_ins = min(prof_lim)*1.15 # negative value
        axins.set_ylim(min_ins, max_ins)

    if a.get_npol() == 1:
        print("Cannot plot Stokes parameters; only plotting I")
    else:
        prof_L, prof_V, _, prof_Q, prof_U = calc_pols(
            a, isub, ichn, I_norm, arr_roll, lim)

        if plot_wide and xlims is None:
            ax.plot(np.concatenate((phase-1, phase, phase+1)),
                    np.concatenate((prof_V, prof_V, prof_V)), 'b:', lw=0.75)
            if plot_iquv:
                ax.plot(np.concatenate((phase-1, phase, phase+1)),
                        np.concatenate((prof_Q, prof_Q, prof_Q)), 'r--', lw=0.75)
                ax.plot(np.concatenate((phase-1, phase, phase+1)),
                        np.concatenate((prof_U, prof_U, prof_U)), 'g-.', lw=0.75)
            else:
                ax.plot(np.concatenate((phase-1, phase, phase+1)),
                        np.concatenate((prof_L, prof_L, prof_L)), 'r--', lw=0.75)
        else:
            ax.plot(phase[lim], prof_V, 'b:', lw=0.75)
            if plot_iquv:
                ax.plot(phase[lim], prof_Q, 'r--', lw=0.75)
                ax.plot(phase[lim], prof_U, 'g-.', lw=0.75)
            else:
                ax.plot(phase[lim], prof_L, 'r--', lw=0.75)

        if add_inset:
            # axis already created above
            prof_V_ins = np.roll(prof_V, roll_ins, axis=0)
            axins.plot(phase_ins, prof_V_ins, 'b:', lw=0.75)
            if plot_iquv:
                prof_Q_ins = np.roll(prof_Q, roll_ins, axis=0)
                axins.plot(phase_ins, prof_Q_ins, 'r--', lw=0.75)
                prof_U_ins = np.roll(prof_U, roll_ins, axis=0)
                axins.plot(phase_ins, prof_U_ins, 'g-.', lw=0.75)
            else:
                prof_L_ins = np.roll(prof_L, roll_ins, axis=0)
                axins.plot(phase_ins, prof_L_ins, 'r--', lw=0.75)

    #ax.tick_params(axis='x', bottom=False, labelbottom=False)

    del(a)

    return(ax, axins, I_norm)


def calc_pols(archive, isub, ichn, norm=None, arr_roll=None, lims=None):
    bstats = archive.get_Integration(isub).baseline_stats()
    int_sI = archive[isub].get_Profile(0, ichn).get_amps()#-bstats[0][0][ichn]
    int_sQ = archive[isub].get_Profile(1, ichn).get_amps()
    int_sU = archive[isub].get_Profile(2, ichn).get_amps()
    #int_sL = archive[isub].get_linear()
    sq_sL = int_sQ**2 + int_sU**2
    int_sL = np.sqrt(sq_sL)
    #int_sL = np.sqrt(abs(sq_sL - (bstats[0][1][ichn]**2 + bstats[0][2][ichn]**2)))
    int_sV = archive[isub].get_Profile(3, ichn).get_amps()#-bstats[0][3][ichn]

    # de-bias the linear
    rms_I = np.sqrt(bstats[1][0][ichn])
    if "J2124-3358" in archive.get_filename():
        rms_I /= 5

    L_true = np.zeros(len(int_sI))
    for i, l_val in enumerate(int_sL):
        if l_val > rms_I:
            L_true[i] = np.sqrt(sq_sL[i] - rms_I**2)
        elif l_val < rms_I:
            L_true[i] = -np.sqrt(abs(sq_sL[i] - rms_I**2))

    # subtract the mean (assuming a Raleigh distribution and L_var=I_var)
    #L_true = np.zeros(len(int_sL))
    #for i, l_val in enumerate(int_sL):
    #    if int_sQ[i]**2 + int_sU[i]**2 >= rms_I*np.sqrt(np.pi/2):
    #        L_true[i] = np.sqrt(int_sQ[i]**2 + int_sU[i]**2 - rms_I**2*np.pi/2)
    #    else:
    #        L_true[i] = -np.sqrt(abs(int_sQ[i]**2 + int_sU[i]**2 - rms_I**2*np.pi/2))

    # check if any values are more than 1 std below 0 and clip
    #for i, l_val in enumerate(int_sL):
    #    if l_val < -1*np.sqrt((4 - np.pi)*rms_I**2/2):
    #        int_sL[i] = -1*np.sqrt((4 -np.pi)*rms_I**2/2)

    int_sL = L_true
    if arr_roll != 0:
        int_sI = np.roll(int_sI, arr_roll, axis=0)
        int_sL = np.roll(int_sL, arr_roll, axis=0)
        int_sV = np.roll(int_sV, arr_roll, axis=0)
        int_sQ = np.roll(int_sQ, arr_roll, axis=0)
        int_sU = np.roll(int_sU, arr_roll, axis=0)

    if lims is None:
        lims = np.array([True for a in int_sI])

    prof_I = int_sI[lims]
    if norm:
        I_norm = prof_I.max()
        prof_I /= I_norm

    prof_L = int_sL[lims]
    prof_V = int_sV[lims]
    prof_Q = int_sQ[lims]
    prof_U = int_sU[lims]
    if norm is not None and norm is not True:
        prof_L /= norm
        prof_V /= norm
        prof_Q /= norm
        prof_U /= norm
    elif norm is True:
        prof_L /= I_norm
        prof_V /= I_norm
        prof_Q /= I_norm
        prof_U /= I_norm

    return(prof_L, prof_V, prof_I, prof_Q, prof_U)


def calc_pa(archive, isub=0, ichn=0, tscr=None, fscr=None, bscr=None,
            return_bool=False, threshold=2.5, base_width=None,
            err_tab=None, err_file=None):
    """
    Return a tuple of (PA, PA_e):
        PA: np.ndarray of position angle as a function of pulse phase
        PA_e: np.ndarray of uncertainty in PA as a function of phase
        phase: 
        filter: boolean np.ndarray

    PA = 1/2 inv_tan(U/Q) (in radians)
    See equation 30 of Montier et al. 2015 for PA_e

    """

    if type(archive) is not psrchive.Archive:
        raise(RuntimeError("Wrong type for archive: {}".format(type(archive))))

    a = archive.clone()
    if type(base_width) is float:
        err = a.execute("edit off:smooth:width={}".format(base_width))
        if err != '':
            print(err)
    elif type(base_width) is str and base_width in ["normal", "iqr"]:
        err = a.execute("edit off="+base_width)
        if err != '':
            print(err)

    #a.remove_baseline()
    a = do_scrunch(a, tscr, fscr, bscr)
    if tscr is True:
        isub = 0
    if fscr is True:
        ichn = 0

    if a.get_state() != "Stokes":
        a.convert_state("Stokes")

    int_sI = a[isub].get_Profile(0, ichn).get_amps() # 0th pol, Stokes I
    phase = np.linspace(0, 1, len(int_sI), endpoint=False)

    int_sQ = a[isub].get_Profile(1, ichn).get_amps() # 1st pol, Stokes Q
    int_sU = a[isub].get_Profile(2, ichn).get_amps() # 2nd pol, Stokes U
    int_sL = np.sqrt(int_sQ**2 + int_sU**2)

    # psrchive method to get mean and variance
    bstats = a[isub].baseline_stats()
    I_var = bstats[1][0][ichn] # variance of I
    Q_var = bstats[1][1][ichn] # variance of Q
    U_var = bstats[1][2][ichn] # variance of U

    #L_true = np.zeros(len(int_sL))
    #for i, l_val in enumerate(int_sL):
    #    if l_val/np.sqrt(I_var) > 1.56:
    #        L_true[i] = np.sqrt(l_val**2 - I_var)
    #    else:
    #        L_true[i] = np.sqrt(I_var)/100

    # subtract the mean (assuming a Raleigh distribution and L_var=I_var)
    #int_sL = L_true #- np.sqrt(I_var*np.pi/2)
    # check if any values are more than 1 std below 0 and clip
    #for i, l_val in enumerate(int_sL):
    #    if l_val < -1*np.sqrt((4 - np.pi)*I_var/2):
    #        int_sL[i] = -1*np.sqrt((4 -np.pi)*I_var/2)

    #p = int_sL/int_sI
    #sig_p = np.sqrt((1/(p*int_sI**2)**2)
    #                * (int_sQ**2*Q_var + int_sU**2*U_var
    #                   + (p**2*int_sI)**2*I_var))

    PA = 0.5*np.arctan2(int_sU, int_sQ)*(180/np.pi)
    #PA_e = (180/np.pi)*(0.5*sig_p/p)\
    #       *np.sqrt((int_sQ**2*U_var + int_sU**2*Q_var)
    #                /(int_sQ**2*Q_var + int_sU**2*U_var))

    # Calculate uncertainties in PA following Everett & Weisberg 2001
    P0 = int_sL/np.sqrt(I_var)
    PA_e = calc_pa_err(P0, err_tab, err_file)

    # Make arrays longer to show wrapping when plotted over (-180, 180)
    PA = np.concatenate((PA, PA-180, PA+180))
    PA_e = np.concatenate((PA_e, PA_e, PA_e))
    phase = np.concatenate((phase, phase, phase))

    if return_bool:
        # Make other arrays longer as well
        int_sU = np.concatenate((int_sU, int_sU, int_sU))
        int_sQ = np.concatenate((int_sQ, int_sQ, int_sQ))
        int_sL = np.concatenate((int_sL, int_sL, int_sL))

        #sig_L = np.sqrt(Q_var*int_sQ**2 + U_var*int_sU**2)/int_sL
        lim = np.logical_or(np.abs(int_sU) > threshold*np.sqrt(U_var),
                            np.abs(int_sQ) > threshold*np.sqrt(Q_var))
        #lim = np.logical_and(np.logical_and(np.abs(PA_e) < 90, lim),
        #                     int_sL > threshold*np.sqrt(I_var))
        lim = np.logical_and(np.abs(PA_e) < 90, lim)
        return(PA, PA_e, phase, lim)
    else:
        return(PA, PA_e, phase)


def calc_pa_err(P0, err_tab=None, err_file=None):
    import scipy.interpolate as spp

    pa_errs = np.zeros(len(P0))
    # First check if direct calculation is possible for all points
    if (P0 > 10).all():
        return(28.64789/P0)
    if err_tab is not None:
        P0_steps, err_steps = zip(*err_tab)
        P0_steps = np.array(P0_steps)
        err_steps = np.array(err_steps)
    elif err_file is None or os.access(err_file, os.R_OK):
        P0_steps, err_steps = make_err_table(err_file)
    else:
        P0_steps, err_steps = np.loadtxt(err_file, unpack=True)

    # Now get the actual uncertainties through linear interpolation
    for num, p0 in enumerate(P0):
        if p0 == 0:
            pa_errs[num] = 90
        elif p0 < 10:
            lim = np.logical_and(P0_steps > p0-0.06, P0_steps < p0+0.07)
            if len(P0_steps[lim]) == 0:
                raise(RuntimeError("Length of array is 0"))

            cs = spp.CubicSpline(P0_steps[lim], err_steps[lim])
            pa_errs[num] = cs(p0)*180/np.pi
        else:
            pa_errs[num] = 28.64789/p0

    return(pa_errs)


def make_err_table(file_out):
    import scipy.integrate as spi

    # Make a table of integration results for p0 values for interpolation
    p_min = 0.01 #max(np.floor(100*(P0.min()-0.05))/100, 0)
    p_max = 10.07 #min(np.ceil(100*(P0.max()+0.06))/100, 10.06)
    p_count = 1006 #int(100*(p_max - p_min))
    P0_steps = np.linspace(p_min, p_max, p_count, endpoint=False)
    err_steps = np.zeros(p_count)
    for num, p0 in enumerate(P0_steps):
        # Adjust bounds of integral to find error for p0
        res = (0, 1)
        err = 0.5/p0
        stop_count = 0
        while res[0] < 0.682 or res[0] > 0.683:
            stop_count += 1
            if stop_count > 1e5:
                raise(RuntimeError("PA uncertainty calculation not converging!!"
                                   " Vals p0={}, err={}, res={}".format(p0, err, res)))

            res = spi.quad(_G, -err, err, args=(p0)) # do integration
            if res[0] < 0.682:
                err *= 1+(np.random.random()/2)
            elif res[0] > 0.683:
                err *= 0.5+(np.random.random()/2)
                if err == 0:
                    err = np.random.random()*(np.pi/2)

        err_steps[num] = err

    if file_out is not None:
        # Write the table to a file
        with open(file_out, 'w') as f:
            for P, e in zip(P0_steps, err_steps):
                f.write("{:<6.2f} {:.8f}\n".format(P, e))

    return(P0_steps, err_steps)


def _G(PA, P0):
    import scipy.special as spc

    eta0 = (P0/np.sqrt(2.))*np.cos(2*PA)
    term1 = 1/np.sqrt(np.pi)
    term2 = eta0*np.exp(eta0**2)
    term3 = (1 + spc.erf(eta0))
    term4 = np.exp(-P0**2/2.)
    return(term1*(term1 + term2*term3)*term4)


def calc_pol_fracs(filename):
    ar = read_archive(filename)
    ar = do_scrunch(ar, fscr=True, tscr=True)
    ar.remove_baseline()
    if ar.get_state() != "Stokes":
        ar.convert_state("Stokes")

    isub = 0
    ichn = 0
    int_sI = ar[isub].get_Profile(0, ichn).get_amps() # 0th pol, Stokes I
    phase = np.linspace(0, 1, len(int_sI), endpoint=False)

    int_sQ = ar[isub].get_Profile(1, ichn).get_amps() # 1st pol, Stokes Q
    int_sU = ar[isub].get_Profile(2, ichn).get_amps() # 2nd pol, Stokes U
    int_sV = ar[isub].get_Profile(3, ichn).get_amps() # 3rd pol, Stokes V
    int_sL = np.sqrt(int_sQ**2 + int_sU**2) # total linear

    _, _, width, _, cen, lim = get_stats(int_sI, return_lim=True)

    # psrchive method to get mean and variance
    bstats = ar[isub].baseline_stats()
    I_var = bstats[1][0] # variance of I
    Q_var = bstats[1][1] # variance of Q
    U_var = bstats[1][2] # variance of U
    V_var = bstats[1][3] # variance of V

    p = int_sL/int_sI
    sig_p = np.sqrt((1/(p*int_sI**2)**2)
                    * (int_sQ**2*Q_var + int_sU**2*U_var
                       + (p**2*int_sI)**2*I_var)) # sqrt(var_p)

    pulse = int_sI > 2.5*np.std(int_sI[lim]) # use std of off-pulse
    good_bins = np.logical_and(int_sI > 3*np.sqrt(I_var),
                               np.logical_and(int_sI > int_sL, int_sI > int_sV))
    good_bins = np.logical_and(pulse,
                               np.logical_and(int_sI > int_sL, int_sI > int_sV))
    good_p = np.logical_and(np.abs(p) > sig_p*2, good_bins)
    good_v = np.logical_and(int_sV > np.sqrt(V_var)*2, good_bins)

    print("{}\t{:.2f}%L, {:.2f}%V"
          .format(filename, 100*np.mean(p[good_bins]), 
                  100*np.mean((np.abs(int_sV)/int_sI)[good_bins])))


def centre_phase(profile, phase, ref=0.4, use_peak=True):
    """
    Find the centre of a given profile, rotate phase
    array to place centre of pulse at ref phase.
    Returns tuple of new phase array and a range centred
    on the pulse for plotting.

    """

    if ref is None:
        ref = 0.4

    stats = get_stats(profile, phase)
    width = stats[2]
    phase_cen = stats[1] if use_peak else stats[4]
    del_phase = ref-phase_cen
    new_phase = (phase + del_phase + 1) % 1
    print("Peak at {}, rotated to {}".format(phase_cen, new_phase[profile == profile.max()]))

    phase_range = [max(ref - width*1.5, 0), min(ref + width*1.5, 1)]
    return(new_phase, phase_range)


def roll_arrays(array, phase=None, ref=0):
    """
    Determine number that phase should be "rolled" to bring the peak
    of array to ref, and return the optimal x limits for plotting.

    """

    xmin = -0.5
    xmax = 0.5
    roll = 0
    if phase is None:
        phase = np.linspace(0, 1, len(array), endpoint=False)

    stats = get_stats(array)
    #print(stats)
    width = stats[2]
    phase_cen = stats[4]
    ref_in = len(phase)*ref
    roll = int(round(ref_in-phase_cen, 0))
    #print(roll)

    if roll < 0:
        phase_new = np.append(phase, phase-1)
    else:
        phase_new = np.append(phase, phase+1)

    phase_roll = np.roll(phase_new, -roll, axis=0)
    phase_new = phase_roll[:len(array)]
    phase_new = np.array([i % 1 for i in phase_new])

    array_new = np.roll(array, roll, axis=0)
    width_ext = max((width/len(array))*1.05, 0.075)
    xmin = ref-width_ext
    xmax = ref+width_ext

    if ref == 0: # Major TODO
    #    phase_new -= width_ext
        print("WARNING: Bugs in roll_arrays for ref==0; suggest 0.5")

    #print(xmin, xmax)

    return(array_new, phase_new, roll, (xmin, xmax))


def plot_Scyl(archive, isub=0, ichn=0, tscr=None, fscr=None, bscr=None,
              thresh=3, out=None, centre=False, zoom=None, zoomy=None,
              ref=0.5, fs=14, high_res=False):
    """
    Make a plot similar to `psrplot -pS` with PA in the top panel
    and the profile in the bottom panel, for a single archive. 

    """

    if out is not None and out != '1/xs':
        fs = make_pretty(fs, high_res)

    if type(archive) is not psrchive.Archive:
        raise(RuntimeError("Wrong type for archive: {}".format(type(archive))))
    else:
        if not archive.get_dedispersed():
            archive.dedisperse()

    a = do_scrunch(archive, tscr, fscr, bscr)
    if tscr is True:
        isub = 0
    if fscr is True:
        ichn = 0

    plt.clf()
    fig = plt.figure(num=1)
    fig.set_size_inches(5, 5)
    ax1 = fig.add_axes((0.2, 0.8, 0.7375, 0.15))
    ax2 = fig.add_axes((0.2, 0.125, 0.7375, 0.675), sharex=ax1)

    ax1, ax2, _ = make_S_plots(a, ax1, ax2, thresh, centre, zoom,
                               zoomy, ref)

    ax2.set_xlabel('Pulse Phase', fontsize=fs)
    ax2.tick_params(axis='x', bottom=True, labelbottom=True)
    ax2.set_ylabel('Flux (arb. units)', fontsize=fs)

    ax1.tick_params(axis='x', bottom=False, labelbottom=False)
    ax1.set_ylim(-100, 100)
    ax1.set_yticks([-90, -45, 0, 45, 90])
    ax1.set_ylabel("P.A. (deg.)", fontsize=fs)

    if out is None or out == '1/xs':
        plt.show()
    else:
        plt.savefig(out, bbox_inches='tight')


def get_stats(profile, phase=None, highsnr=False, return_lim=False,
              no_roll=True):
    """
    S/N, peak, width, mean_off, and centre (and lim); in units of bins
    if phase is not given (centre rounded down);
    otherwise, all have units of phase

    """

    # Assume profile is a np.ndarray of intensity vs phase
    # Check if profile has invalid data
    if np.any(np.isnan(profile)) or np.all(profile == profile[0]):
        if return_lim:
            return(0, 0, 0, 0, 0, np.zeros(len(profile)))
        else:
            return(0, 0, 0, 0, 0)

    i = 0
    val = profile
    num = np.arange(len(profile))
    start = 5 if highsnr else 2
    lim = val < (val.max()-val.min())/start + val.min()
    peak = 0
    cen = 10
    width = 2
    first = True
    roll = 0

    #while not (peak-width < cen and peak+width > cen):
    #    if first is False:
    #        val = np.roll(val, int(len(val)*0.9/4), axis=0)
    #        #print("Rolling profile to get better answer")
    #        if phase is None:
    #            roll += int(len(val)*0.9/4)
    #            #print(roll)
    #            if roll > 2*len(val):
    #                print("Roll wrapped twice! Using current answers")
    #                break
    #        else:
    #            roll += int(len(val)*0.9/4)/len(val)
    #            #print(roll)
    #            if roll > 2:
    #                print("Roll wrapped twice! Using current answers")
    #                break
    #    else:
    #        first = False

    while i < 3:
        lim2 = val < val[lim].mean()+2.5*val[lim].std()
        lim = val < val[lim2].mean()+2.5*val[lim2].std()
        if val[lim].std() == val[lim2].std():
            break

        i += 1

    width = float(max(len(val) - len(val[lim]), 1))
    # check that nothing unreasonable is included (or excluded)
    if width > 1:
        high_val = num[val == val.max()][0]
        sig = val[lim].std()
        moff = val[lim].mean()
        for i, V in enumerate(val):
            if i == high_val or (lim[i-1] == lim[i] and
                                 lim[i] == lim[(i+1)%len(lim)]):
                continue # ignore if surrounding points agree

            list_be = [val[A] for A in [i-1, i]]
            list_af = [val[A] for A in [i, (i+1)%len(lim)]]
            av_be = np.mean(list_be)
            av_af = np.mean(list_af)

            if av_be < V < av_af or av_be > V > av_af:
                if av_be > moff + 2*sig or \
                   av_af > moff + 2*sig:
                    # if neighboring point is signal, include point i
                    lim[i] = False
            elif lim[i] and (V < av_be and V < av_af):
                lim[i] = False # include in pulse if in a local min
            elif not lim[i] and (V > av_be and V > av_af):
                lim[i] = True # exclude if local max

            if val[i] < moff + sig:
                # make sure no low points pass through
                lim[i] = True

    width = float(max(len(val) - len(val[lim]), 1))
    if np.all(lim):
        # select a random point to be the "signal" to avoid errors
        lim[0] = False

    P = float(len(val))
    sig = val[lim].std()
    moff = val[lim].mean()
    snr = np.abs(np.array([(n - val[lim].mean()) for n in
                           val]).sum())/(sig*np.sqrt(width))

    if phase is None:
        peak = num[val == val[np.logical_not(lim)].max()][0]
        cen = int(np.median(num[np.logical_not(lim)]))
    else:
        peak = phase[val == val[np.logical_not(lim)].max()][0]
        cen = np.median(phase[np.logical_not(lim)])
        del_phase = [phase[i+1] - phase[i] for i in range(len(phase[:-1]))]
        width = width*np.min(del_phase)

    new_peak = (peak - roll) % len(val) if phase is None else (peak-roll)%1
    new_cen = (cen - roll) % len(val) if phase is None else (cen-roll)%1

    if return_lim:
        return(round(snr, 3), new_peak, width, round(moff, 4), new_cen, lim)
    else:
        return(round(snr, 3), new_peak, width, round(moff, 4), new_cen)


def plot_wfall_wprof(filename, ordinate='freq', fscr=False, tscr=True,
                     out=None, zoom=None, high_res=False):
    """
    ordinate must be either 'freq' or 'time'
    """

    fs = make_pretty(12, high_res)
    plt.rc('lines', linewidth=1)
    ar = read_archive(filename)

    # check what to plot on y axis and set scrunching values
    if ordinate == 'freq':
        if ar.get_nchan() <= 1:
            raise(RuntimeError("Not enough channels to plot"))

        ar = do_scrunch(ar, fscr=fscr, tscr=tscr, pscr=True)
        fscr = False
        tscr = True
    elif ordinate == 'time':
        if ar.get_nsubint() <= 1:
            raise(RuntimeError("Not enough subints to plot"))

        ar = do_scrunch(ar, fscr=fscr, tscr=tscr, pscr=True)
        fscr = True
        tscr = False
    else:
        raise(RuntimeError("Unrecognised 'ordinate': {}".format(ordinate)))

    # check what to do with zoom values
    if zoom is not None and zoom == "auto":
        print("Don't know what to zoom on; not zooming")
        xlims = [0, 1]
    elif zoom is not None:
        if ',' in zoom:
            xlims = [float(A) for A in zoom.split(',')]
        else:
            xlims = [float(A) for A in zoom.split('x')]
    else:
        xlims = [0, 1]

    # set up the figure with 2 panels
    fig = plt.figure(num=1)
    fig.set_size_inches(5, 9)
    ax1 = fig.add_axes([0.175, 0.1, 0.775, 0.625])
    ax2 = fig.add_axes([0.175, 0.725, 0.775, 0.225])

    # plot the waterfall on the bottom panel and profile on top
    ax1 = plot_waterfall(ar, ax=ax1, tscr=tscr, fscr=fscr, fs=fs,
                         vmax_frac=1.0, cmap='magma')
    ax2 = plot_single_prof(ar, fscr=True, tscr=True, ax=ax2, fs=fs,
                           clear=False, norm=True)

    # adjust the axis labels, limits, ticks, etc. 
    ax1.set_xlim(*xlims)
    ax2.set_xlim(*xlims)
    ax2.tick_params(axis='x', direction='in')
    ax2.set_xticklabels([])
    ax2.set_xlabel('')
    ax2.set_ylabel('Intensity (arb. units)', fontsize=fs)

    if out is None or out == '1/xs':
        plt.show()
    else:
        plt.savefig(out, bbox_inches='tight')


def plot_waterfall(archive, ax=None, ipol=0, tscr=True,
                   subint=0, fscr=False, chan=0, cmap='viridis',
                   vmax_frac=1.0, vmin_frac=0.0, fs=14):
    if type(archive) is not psrchive.Archive:
        raise(RuntimeError("Wrong type for archive: "+type(archive)))

    if ax is None:
        ax = plt.gca()

    a = do_scrunch(archive, tscr, fscr, pscr=True)
    a.remove_baseline()

    if fscr == False:
        isub = subint
        ichn = ':'
        freqs = a.get_frequencies()
        ymin = freqs.min()
        ymax = freqs.max()
        ylab = 'Frequency (MHz)'
        origin = 'upper'
    elif tscr == False:
        ichn = chan
        isub = ':'
        ymin = 0
        ymax = a.end_time().in_minutes() - a.start_time().in_minutes()
        ylab = 'Time (min.)'
        origin = "lower"
    else:
        raise(RuntimeError("Cannot manage dimensions"))

    wf_data = get_weighted_data(a, ichn, isub)
    #print(wf_data.shape)
    if ichn == ':':
        wf_data = np.flip(wf_data, axis=0)

    vmin = np.percentile(wf_data, max(vmin_frac*100, 0))
    vmax = np.percentile(wf_data, min(vmax_frac*100, 100))
    extent = (0, 1, ymin, ymax)

    ax.imshow(wf_data, vmin=vmin, vmax=vmax, cmap=cmap,
              origin=origin, extent=extent, aspect='auto',
              interpolation='nearest')
    ax.set_xlabel('Pulse Phase', fontsize=fs)
    ax.set_ylabel(ylab, fontsize=fs)

    del(a)

    return(ax)


def do_scrunch(archive, tscr=None, fscr=None, bscr=None, pscr=False,
               tscr_to=None, fscr_to=None):
    """
    Scrunch archive as indicated and return psrchive.Archive object.

    """

    a = archive.clone()
    if fscr is True:
        a.fscrunch()
    elif type(fscr) is int:
        a.fscrunch(fscr)
    elif type(fscr) is str:
        try:
            fscr = int(fscr)
        except ValueError:
            raise(ValueError("Incorrect type/value for fscr: {}".format(fscr)))

        a.fscrunch(fscr)
    elif fscr_to is not None:
        if type(fscr_to) not in [float, int]:
            raise(ValueError("Incorrect type for fscr_to: {}"
                             .format(fscr_to)))
        else:
            a.fscrunch_to_nchan(fscr_to)

    if tscr is True:
        a.tscrunch()
    elif type(tscr) is int:
        a.tscrunch(tscr)
    elif type(tscr) is str:
        try:
            tscr = int(tscr)
        except ValueError:
            raise(ValueError("Incorrect type/value for tscr: {}".format(tscr)))

        a.tscrunch(tscr)
    elif tscr_to is not None:
        if type(tscr_to) not in [float, int]:
            raise(ValueError("Incorrect type for tscr_to: {}"
                             .format(tscr_to)))
        else:
            a.tscrunch_to_nsub(tscr_to)

    if bscr:
        a.bscrunch(bscr)

    if pscr:
        a.pscrunch()

    return(a)


def plot_single_prof(archive, ipol=0, ichn=0, isub=0,
                     tscr=None, fscr=None, bscr=None,
                     ax=None, clear=True, norm=False):
    if type(archive) is not psrchive.Archive:
        raise(RuntimeError("Wrong type for archive: "+type(archive)))

    a = do_scrunch(archive, tscr, fscr, bscr)
    if tscr is True:
        isub = 0
    if fscr is True:
        ichn = 0

    if ax is None:
        ax = plt.gca()

    a.remove_baseline()

    prof = a[isub].get_Profile(ipol, ichn).get_amps()
    phase = np.linspace(0, 1, len(prof), endpoint=False)

    if norm == True:
        p_base = get_stats(prof)[3]
        prof -= p_base
        p_max = prof.max()
        prof /= p_max

    ax.plot(phase, prof, 'k-', linewidth=0.75)
    if clear:
        ax.set_yticks([-1000])
        ax.set_xticks([-1000])

    return(ax)


def plot_profiles(psr_list, high_res=False):
    for psr in psr_list:
        pass


def plot_histos(psr_list, high_res=False):
    fig = plt.figure(num=1)
    fig.set_size_inches(6, 9)
    npsrs = len(psr_list) + 0.0
    ncol = int(np.sqrt(npsrs))
    if npsrs/ncol % 1 < 0.5:
        ncol -= 1
    nrow = int(np.ceil(npsrs/ncol))
    axes = fig.add_subplots((nrow, ncol))


def plot_rrat_wfall(rrat_file, out=None, high_res=False):
    """
    Plot the Stokes I profile and phase-v-time plot of
    a given "RRAT" data file, centred and zoomed on pulse.

    """

    fs = make_pretty(16, high_res)

    archive = read_archive(rrat_file)
    #print("Read archive")
    a = do_scrunch(archive, fscr=True, pscr=True)
    a.remove_baseline()

    fig = plt.figure(num=1)
    fig.set_size_inches(8, 14)
    ax1 = fig.add_axes((0.2, 0.8, 0.7375, 0.15))
    ax2 = fig.add_axes((0.2, 0.125, 0.7375, 0.675))
    xlims = [0, 1]

    wf_data = get_weighted_data(a, 0, ':')
    """
    stats = []
    for f in wf_data:
        try:
            stats.append(np.array(get_stats(f)))
        except:
            stats.append(np.array((0, 0, 0, 0, 0)))

    stats = np.array(stats)
    num, bins, _ = plt.hist(stats[:, 1], range=(1, 1024), bins=40)
    bmins = bins[:-1]
    bmaxs = bins[1:]
    xmins = bmins[num > num.mean() + 2*num.std()]
    xmaxs = bmaxs[num > num.mean() + 2*num.std()]
    """

    off_std = np.std(wf_data[:, 150:], axis=1)
    lim = np.logical_and(off_std > 3, off_std < 3.6)

    xmins = [1024*0.03,]
    xmaxs = [1024*0.12,]
    xlims = (xmins[0], xmaxs[-1])
    #lim = np.logical_and(stats[:, 1] > xmins[0], stats[:, 1] < xmaxs[-1])
    vmax_frac = 1.0
    vmin_frac = 0.03
    vmin = np.percentile(wf_data, max(vmin_frac*100, 0))
    vmax = np.percentile(wf_data, min(vmax_frac*100, 100))
    vmin = -5
    vmax = 20

    mjds = a.get_mjds()
    lim2 = np.logical_and(mjds[lim] > 58218, mjds[lim] < 58219)
    nums = np.arange(0, len(mjds[lim][lim2]))
    lim3 = nums != 49

    ax2.imshow(wf_data[lim][lim2][lim3], cmap='viridis_r', origin='lower',
               aspect='auto', interpolation='nearest', vmin=vmin,
               vmax=vmax, extent=(0, 1, 0, len(mjds[lim][lim2][lim3])))
    ax2.set_xlabel('Pulse Phase', fontsize=20)
    ax2.set_ylabel('Sub-integration Index', fontsize=20)

    print("Plotted waterfall")

    prof = np.sum(wf_data[lim], axis=0)
    ax1.plot(np.linspace(0, 1, 1024), prof, 'k-')
    ax1.set_yticks([])
    ax1.set_xticks([])
    print("Plotted profile")

    xlims = (0.03, 0.12)
    ax1.set_xlim(*xlims)
    ax2.set_xlim(*xlims)

    if out is None or out == '1/xs':
        plt.show()
    else:
        plt.savefig(out, bbox_inches='tight')


def make_prof_gallery(file_list, out_dir="./", pscrunch=True,
                      bscrunch=None, zoom=None, shape="4x5", widths=None,
                      verb=False, high_res=False):
    print("Making pol. profile plots for {} files".format(len(file_list)))

    # extract the numbers of rows and columns from the shape
    m = re.match('(.*)[,x](.*)', shape.lstrip(' '))
    if m is None:
        raise(RuntimeError("Improper format for shape: "+shape))

    try:
        cols = int(m.groups()[0])
        rows = int(m.groups()[1])
    except ValueError:
        raise(RuntimeError("Improper values for shape: "+shape))

    fs = make_pretty(14, high_res)

    figx = 6.7*1.75
    figy = 8.5*1.75
    top = 0.99
    bottom = 0.04
    left = 0.07
    right = 0.99
    if zoom is None:
        ver_sep = 0
        hor_sep = 0
    else:
        ver_sep = 0.025*3./rows # some space between axes
        hor_sep = 0.045*3./rows

    height = (top - bottom - ver_sep*(rows-1))/rows # tot. height of each axis
    width = (right - left - hor_sep*(cols-1))/cols # width of each axis
        
    base_name = None
    base_wid = None
    if widths is not None and os.access(widths, os.R_OK):
        base_name, base_wid = np.loadtxt(widths, dtype=[('name', 'U14'), ('width', "U30")],
                                         delimiter=",", unpack=True)

    for ar_count, filename in enumerate(file_list):
        # find PSR Jname in filename
        m = re.match('(J[\d]{4}[-+][\d]{2,4}[A-Za-z]{0,2})[_\W]?.*',
                     os.path.split(filename)[-1])
        if m is None and verb:
            print("Could not match the pulsar name in the filename "+filename)
            psrname = None
            psr_wid = None
        elif m is not None:
            psrname = m.groups()[0]
            if base_name is not None and psrname in base_name:
                psr_wid = base_wid[base_name == psrname][0]
                try:
                    psr_wid = float(psr_wid)
                except ValueError:
                    psr_wid = psr_wid.strip("'").strip('"')
            else:
                psr_wid = None

        if verb:
            print("Starting on file number {}".format(ar_count))

        if ar_count % (rows*cols) == 0:
            if verb:
                print("Setting up a page")
            # set up the page for the next set of rows*cols plots
            if ar_count > 0:
                # if a figure is open, close it and save the figure
                print("Closing figure "+pdf_name)

                pp.savefig()
                plt.close(fig)
                pp.close()

            # make the PDF file for the set of plots
            pg_count = int(np.floor(float(ar_count)/(rows*cols)))
            pdf_name = os.path.join(out_dir, "profs_{}_{}.pdf"
                                    .format(shape, pg_count))
            pp = PdfPages(pdf_name)
            if verb:
                print("Opening figure "+pdf_name)

            plt.figure(num=pg_count)
            plt.clf()
            fig = plt.gcf()

            if ar_count > 0 and len(file_list)-ar_count < rows*cols:
                if verb:
                    print("Have {} files, but only {} left"
                          .format(len(file_list), len(file_list)-ar_count))

                # scale the figure down in height if fewer rows needed
                num_left = len(file_list)-ar_count
                rows_new = int(np.ceil(float(num_left)/cols))
                if verb:
                    print("New number of rows: {}".format(rows_new))

                ver_sep *= rows/float(rows_new) # fix space between axes
                bottom *= rows/float(rows_new)
                top = 1-((1-top)*rows/float(rows_new))
                rows = rows_new
                height_new = (top - bottom - ver_sep*(rows-1))/rows
                # scale the figure length by the ratio of heights
                # this should keep each axis identical *in theory*
                figy *= height/height_new
                height = height_new

            fig.set_size_inches(figx, figy)

            # initialise lists for the axes
            axes = []

            # iterate over the rows and columns to add all axes
            if verb:
                print("Creating the axes lists")

            ax_count = 0 # yes, I use this variable later
            # count the rows from the top
            for row_count in range(rows-1, -1, -1):
                for col_count in range(cols):
                    ax_count += 1
                    # don't make unnecessary axes
                    if ax_count > len(file_list)-ar_count:
                        break

                    ax = fig.add_axes([
                        left + (width+hor_sep)*col_count,
                        bottom +(height+ver_sep)*row_count,
                        width, height])
                    axes.append(ax)

            if verb:
                print("Have {} axes".format(len(axes)))

        if verb:
            ar_num = ar_count%(rows*cols)
            if ar_num % 10 == 1:
                mod1 = "st"
            elif ar_num % 10 == 2:
                mod1 = "nd"
            elif ar_num % 10 == 3:
                mod1 = "rd"
            else:
                mod1 = "th"

            if ar_count % 10 == 1:
                mod2 = "st"
            elif ar_count % 10 == 2:
                mod2 = "nd"
            elif ar_count % 10 == 3:
                mod2 = "rd"
            else:
                mod2 = "th"

            print("Using the {}-{} axes for the {}-{} file"
                  .format(ar_count%(rows*cols), mod1, ar_count, mod2))

        # for each archive, make the plots
        # axes lists are overwritten for each new page
        ax_count = ar_count%(rows*cols) # reset in each iteration
        ax2 = axes[ax_count]
        ar = read_archive(filename, base_wid)
        ar.centre()
        ar = do_scrunch(ar, tscr=True, fscr=True, bscr=bscrunch, pscr=pscrunch)

        # add some formatting and labels to the axes
        if ax_count%cols == 0: # the first axis in each row
            ax2.set_ylabel('Intensity (arb. units)', fontsize=fs)
            ax2.set_yticks([-0.5, 0.0, 0.5, 1.0])
        elif zoom is None:
            ax2.set_yticks([-0.5, 0.0, 0.5, 1.0])
            ax2.set_yticklabels([])
        else:
            ax2.set_yticks([-0.5, 0.0, 0.5, 1.0])

        # add x-labels to the bottom axes of each column
        ax2.tick_params(axis='x', top=True, direction='in')
        ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if ax_count >= len(axes)-cols or ar_count >= len(file_list)-cols:
            ax2.set_xlabel('Pulse Phase', fontsize=fs)
        elif zoom is None:
            ax2.set_xticklabels([])

        if psrname:
            ax2.text(0.035, 0.965, psrname.replace('-', '$-$'), fontsize=fs,
                     transform=ax2.transAxes, horizontalalignment='left',
                     verticalalignment='top')

        add_ins = False
        ins_lims = None

        ax2 = plot_single_prof(ar, ax=ax2, fs=fs, clear=False, norm=True)
        #ax1, ax2, axins, _ = make_S_plots(
        #        ar, ax1, ax2, zoom=zoom, centre=True, ref=0.7, norm=True,
        #        thresh=2.5, inset_bool=add_ins, inset_lims=ins_lims,
        #        base_width=psr_wid)

        if psrname: # add a bit of space at the top
            ylims = ax2.get_ylim()
            ax2.set_ylim([ylims[0], ylims[1]*1.05])

        # now adjust the xticks based on the plotted profile
        if zoom is None:
            ax2.set_xlim([0.0, 1.0])
            if ax_count%cols == 0:
                row_lim = ax2.get_ylim()
            else:
                row_lim = (min(row_lim[0], ax2.get_ylim()[0]),
                           max(row_lim[1], ax2.get_ylim()[1]))

            if ax_count%cols == cols-1:
                # set all axes for this row to same limits
                for num in range(ax_count-cols+1, ax_count+1):
                    axes[num].set_ylim(row_lim)

        elif zoom == 'auto':
            xlim = ax2.get_xlim()
            xticks = [0.4, 0.5, 0.6]
            if min(xlim) <= 0.22:
                xticks.insert(0, 0.2)
                xticks.insert(1, 0.3)
            elif min(xlim) <= 0.3:
                xticks.insert(0, 0.3)
            elif min(xlim) > 0.4:
                xticks.insert(1, 0.45)

            if max(xlim) >= 0.78:
                xticks.append(0.7)
                xticks.append(0.8)
            elif max(xlim) >= 0.7:
                xticks.append(0.7)
            elif max(xlim) < 0.6:
                xticks.insert(-1, 0.55)

            ax2.set_xticks(xticks)
            ax2.set_xlim(xlim)

    # save and close the final figure/page
    print("All done! Closing the final figure: "+pdf_name)

    pp.savefig()
    plt.close(fig)
    pp.close()


def make_all_polprof(file_list, out_dir="./", tscrunch=True, fscrunch=True,
                     bscrunch=None, zoom=None, shape="3x3", widths=None,
                     pae_file=None, extras=False, verb=False, db_file=None,
                     zoomy=None, high_res=False, extra_csv=None, no_norm=False,
                     rot_csv=None, plot_iquv=False):

    """
    Assemble polarisation plots for multiple archives into single-page PDFs
    with given shapes (default: 3 rows x 3 columns). Output are files written
    to the given directory 

    """

    save_extra_df = None
    if extra_csv:
        extras = True

    pae_tab = None
    print("Making pol. profile plots for {} files".format(len(file_list)))

    # extract the numbers of rows and columns from the shape
    m = re.match('(.*)[,x](.*)', shape.lstrip(' '))
    if m is None:
        raise(RuntimeError("Improper format for shape: "+shape))

    try:
        cols = int(m.groups()[0])
        rows = int(m.groups()[1])
    except ValueError:
        raise(RuntimeError("Improper values for shape: "+shape))

    fs = make_pretty(int(12*3./rows), high_res) # adjust fontsize according to plot size

    figx = 6.7*1.75
    figy = 8.5*1.75
    top = 0.99
    bottom = 0.04*3./rows
    left = 0.07*3./rows # room for labels
    right = 0.99
    if zoom is None:
        ver_sep = 0
        hor_sep = 0
    else:
        ver_sep = 0.025*3./rows # some space between axes
        hor_sep = 0.045*3./rows

    height = (top - bottom - ver_sep*(rows-1))/rows # tot. height of each axis
    height_top = height*0.135 # height of the PA axis
    height_bot = height*0.865 # height of the profile axis
    width = (right - left - hor_sep*(cols-1))/cols # width of each axis
        
    base_name = None
    base_wid = None
    if widths is not None and os.access(widths, os.R_OK):
        base_name, base_wid = np.loadtxt(widths, dtype=[('name', 'U14'), ('width', "U30")],
                                         delimiter=",", unpack=True)

    if extra_csv:
        extra_df = pd.read_csv(extra_csv)

    if rot_csv:
        rot_df = pd.read_csv(rot_csv, header=None, names=["PSRJ", "ROT"])
        if verb:
            print("Read in csv of phase shifts")

    for ar_count, filename in enumerate(file_list):
        # find PSR Jname in filename
        m = re.match('(J[\d]{4}[-+][\d]{2,4}[A-Za-z]{0,2})[_\W]?.*',
                     os.path.split(filename)[-1])
        if m is None and verb:
            print("Could not match the pulsar name in the filename "+filename)
            psrname = None
            psr_wid = None
        elif m is not None:
            psrname = m.groups()[0]
            if base_name is not None and psrname in base_name:
                psr_wid = base_wid[base_name == psrname][0]
                try:
                    psr_wid = float(psr_wid)
                except ValueError:
                    psr_wid = psr_wid.strip("'").strip('"')
            else:
                psr_wid = None

        if verb:
            print("Starting on file number {}".format(ar_count))

        if ar_count % (rows*cols) == 0:
            if verb:
                print("Setting up a page")
            # set up the page for the next set of rows*cols plots
            if ar_count > 0:
                # if a figure is open, close it and save the figure
                print("Closing figure "+pdf_name)

                pp.savefig()
                plt.close(fig)
                pp.close()

            # make the PDF file for the set of plots
            pg_count = int(np.floor(float(ar_count)/(rows*cols)))
            pdf_name = os.path.join(out_dir, "pols_{}_{}.pdf"
                                    .format(shape, pg_count))
            pp = PdfPages(pdf_name)
            if verb:
                print("Opening figure "+pdf_name)

            plt.figure(num=pg_count)
            plt.clf()
            fig = plt.gcf()

            if ar_count > 0 and len(file_list)-ar_count < rows*cols:
                if verb:
                    print("Have {} files, but only {} left"
                          .format(len(file_list), len(file_list)-ar_count))

                # scale the figure down in height if fewer rows needed
                num_left = len(file_list)-ar_count
                rows_new = int(np.ceil(float(num_left)/cols))
                if verb:
                    print("New number of rows: {}".format(rows_new))

                ver_sep *= rows/float(rows_new) # fix space between axes
                bottom *= rows/float(rows_new)
                top = 1-((1-top)*rows/float(rows_new))
                rows = rows_new
                height_new = (top - bottom - ver_sep*(rows-1))/rows
                # scale the figure length by the ratio of heights
                # this should keep each axis identical *in theory*
                figy *= height/height_new
                height_top *= height_new/height
                height_bot *= height_new/height
                height = height_new

            fig.set_size_inches(figx, figy)

            # initialise lists for the axes
            top_axes = []
            bottom_axes = []

            # iterate over the rows and columns to add all axes
            # two axes per "plot" - PA and profile
            if verb:
                print("Creating the axes lists")

            ax_count = 0 # yes, I use this variable later
            # count the rows from the top
            first_row = True
            first_col = True
            for row_count in range(rows-1, -1, -1):
                for col_count in range(cols):
                    ax_count += 1
                    # don't make unnecessary axes
                    if ax_count > len(file_list)-ar_count:
                        break

                    ax_bot = fig.add_axes([
                        left + (width+hor_sep)*col_count,
                        bottom +(height+ver_sep)*row_count,
                        width, height_bot])
                    ax_top = fig.add_axes([
                        left + (width+hor_sep)*col_count,
                        bottom + (height+ver_sep)*row_count + height_bot,
                        width, height_top])
                    top_axes.append(ax_top)
                    bottom_axes.append(ax_bot)

            #if verb:
            #    print("Have {} tops and {} bottoms".format(len(top_axes),
            #                                               len(bottom_axes)))

        # for each archive, make the plots
        # axes lists are overwritten for each new page
        if verb:
            ar_num = ar_count%(rows*cols)
            if ar_num % 10 == 1:
                mod1 = "st"
            elif ar_num % 10 == 2:
                mod1 = "nd"
            elif ar_num % 10 == 3:
                mod1 = "rd"
            else:
                mod1 = "th"

            if ar_count % 10 == 1:
                mod2 = "st"
            elif ar_count % 10 == 2:
                mod2 = "nd"
            elif ar_count % 10 == 3:
                mod2 = "rd"
            else:
                mod2 = "th"

            print("Using the {}-{} axes for the {}-{} file"
                  .format(ar_count%(rows*cols), mod1, ar_count, mod2))

        ax_count = ar_count%(rows*cols) # reset in each iteration
        ax1 = top_axes[ax_count]
        ax2 = bottom_axes[ax_count]
        ar = read_archive(filename, psr_wid)
        ar = do_scrunch(ar, tscrunch, fscrunch, bscr=bscrunch)

        # add some formatting and labels to the axes
        if ax_count%cols == 0: # the first axis in each row
            if no_norm:
                ax2.set_ylabel('Flux (data units)', fontsize=fs)
            else:
                ax2.set_ylabel('Intensity (arb. units)', fontsize=fs)

            ax1.set_ylabel("P.A. (deg.)", fontsize=fs)
        elif zoom is None:
            if not no_norm:
                ax2.set_yticklabels([])

            ax1.set_yticklabels([])

        ax1.set_yticks([-90, 0, 90])
        if zoomy is None and not no_norm:
            ax2.set_yticks([-0.5, 0.0, 0.5, 1.0])

        # add x-labels to the bottom axes of each column
        ax2.tick_params(axis='x', top=True, direction='in')
        ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if ax_count >= len(top_axes)-cols or ar_count >= len(file_list)-cols:
            ax2.set_xlabel('Pulse Phase', fontsize=fs)
        elif zoom is None:
            ax2.set_xticklabels([])

        ax1.tick_params(bottom=False, labelbottom=False)
        ax1.set_ylim(-135, 135)

        add_ins = False
        ins_lims = None
        if psrname and psrname == 'J1544-0713':
            add_ins = True
            ins_lims = [-0.05, 0.05]

        plot_wide = True if zoom is None else False
        if pae_tab is None and pae_file is not None and os.access(pae_file, os.R_OK):
            pae_tab = np.loadtxt(pae_file)

        if psrname and rot_csv and psrname in rot_df.PSRJ:
            row_num = rot_df[rot_df['PSRJ'] == psrname].first_valid_index()
            psr_rot = rot_df.ROT[row_num]
        else:
            psr_rot = None

        ax1, ax2, axins, I_norm = make_S_plots(
                ar, ax1, ax2, zoom=zoom, centre=True, ref=0.5, norm=not(no_norm),
                thresh=3.5, inset_bool=add_ins, inset_lims=ins_lims,
                base_width=psr_wid, plot_wide=plot_wide, pae_file=pae_file,
                pae_tab=pae_tab, zoomy=zoomy, ex_rot=psr_rot, plot_iquv=plot_iquv)

        if psrname and extras: # add a bit of space at the top
            ylims = ax2.get_ylim()
            ax2.set_ylim([ylims[0], ylims[1]*1.175])
        elif psrname:
            ylims = ax2.get_ylim()
            ax2.set_ylim([ylims[0], ylims[1]*1.05])

        if psrname:
            opt = ''
            right_opt = ''
            if extras and not extra_csv:
                db_print = ''
                if db_file:
                    db_print = ' -db_file '+db_file
                dm = sproc.Popen(shplit('psrcat -x -c dm{} {}'.format(db_print, psrname)),
                                 stdout=sproc.PIPE).communicate()[0]
                dm = round(float(dm.split()[0]), 2)
                p0 = sproc.Popen(shplit('psrcat -x -c p0{} {}'.format(db_print, psrname)),
                                 stdout=sproc.PIPE).communicate()[0]
                p0 = round(float(p0.split()[0])*1e3, 3)
                opt = '\nP={}\nDM={}'.format(p0, dm)

            elif extra_csv:
                # ignore the db_file
                row_num = extra_df[extra_df['Jname'] == psrname].first_valid_index()
                if "P0" in extra_df.columns:
                    p0 = extra_df.P0[row_num]*1e3
                    opt += "\nP={:.3f}".format(float(p0))

                if 'DM' in extra_df.columns:
                    dm = extra_df.DM[row_num]
                    opt += "\nDM={:.2f}".format(float(dm))

                if "N_bin" in extra_df.columns:
                    nbin = extra_df.N_bin[row_num]
                    right_opt += "\nNb={}".format(nbin)

                #if I_norm is not None:
                #    right_opt += "\nSp={:.2f}".format(I_norm)
                #    extra_df.at[row_num, 'S_peak'] = I_norm
                #    save_extra_df = True
                if "S_peak" in extra_df.columns:
                    s_peak = float(extra_df.S_peak[row_num])
                    s_peak_exp = np.floor(np.log10(s_peak))
                    s_peak_out = 10**s_peak_exp*round(s_peak/10**s_peak_exp, 2)
                    if int(s_peak_out) == s_peak_out:
                        s_peak_out = int(s_peak_out)

                    right_opt += "\nSp={}".format(s_peak_out)

            ax2.text(0.035, 0.965, '{}{}'.format(psrname.replace('-', '$-$'), opt), fontsize=fs-2,
                     transform=ax2.transAxes, horizontalalignment='left',
                     verticalalignment='top')
            if right_opt != '':
                ax2.text(0.965, 0.965, right_opt, fontsize=fs-2,
                         transform=ax2.transAxes, horizontalalignment='right',
                         verticalalignment='top')

        # now adjust the xticks based on the plotted profile
        if zoom is None:
            ax2.set_xlim([-0.05, 1.05])
            if ax_count%cols == 0:
                row_lim = ax2.get_ylim()
            else:
                row_lim = (min(row_lim[0], ax2.get_ylim()[0]),
                           max(row_lim[1], ax2.get_ylim()[1]))

            if ax_count%cols == cols-1:
                # set all axes for this row to same limits
                for num in range(ax_count-cols+1, ax_count+1):
                    bottom_axes[num].set_ylim(row_lim)

        elif zoom == 'auto':
            xlim = ax2.get_xlim()
            xticks = [0.4, 0.5, 0.6]
            if min(xlim) <= 0.22:
                xticks.insert(0, 0.2)
                xticks.insert(1, 0.3)
            elif min(xlim) <= 0.3:
                xticks.insert(0, 0.3)
            elif min(xlim) > 0.4:
                xticks.insert(1, 0.45)

            if max(xlim) >= 0.78:
                xticks.append(0.7)
                xticks.append(0.8)
            elif max(xlim) >= 0.7:
                xticks.append(0.7)
            elif max(xlim) < 0.6:
                xticks.insert(-1, 0.55)

            ax2.set_xticks(xticks)
            ax2.set_xlim(xlim)

            # fix J1421 plot because :-(
            if psrname == 'J1421-4409':
                # assume it's already been rolled to have the max at 0.5
                centre = 0.31
                w50 = 0.31
                ax1.set_xlim(centre-w50, centre+w50)
                ax2.set_xlim(centre-w50, centre+w50)
                ax2.set_xticks([0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61])
                ax2.set_xticklabels(["$0.2$", "$0.3$", "$0.4$", "$0.5$", 
                                     "$0.6$", "$0.7$", "$0.8$"])

            # adjust J1544 with inset
            if psrname == 'J1544-0713':
                ax1.set_xlim(xlim[0]+0.022, xlim[1]+0.022)
                ax2.set_xlim(xlim[0]+0.022, xlim[1]+0.022)
                ax2.set_xticks([0.4575, 0.5075, 0.5575])
                ax2.set_xticklabels(["$0.45$", "$0.5$", "$0.55$"])

            # also fix J1337-4441 (shift by half width and reduce range)
            if psrname == "J1337-4441":
                # centre is at leading edge
                w50 = 0.075
                centre = 0.5 + w50/2
                ax1.set_xlim(centre-0.07, centre+0.07)
                ax2.set_xlim(centre-0.07, centre+0.07)
                ax2.set_xticks([centre-0.05, centre, centre+0.05])
                ax2.set_xticklabels(["$0.45$", "$0.5$", "$0.55$"])

    # save and close the final figure/page
    print("All done! Closing the final figure: "+pdf_name)

    pp.savefig()
    plt.close(fig)
    pp.close()

    if save_extra_df:
        extra_df.to_csv(extra_csv, index=False)


def make_S_plots(archive, ax1, ax2, thresh=2.5, centre=False,
                 zoom=None, zoomy=None, ref=0.5, norm=False,
                 inset_bool=False, inset_lims=None,
                 base_width=None, plot_wide=False, pae_tab=None,
                 pae_file=None, ex_rot=None, plot_iquv=False):
    """
    Do all of the plotting for Scyl-type plots, taking axes as inputs
    and returning the axes (with no labels)

    """

    if type(archive) is not psrchive.Archive:
        raise(RuntimeError("Wrong type for archive: {}".format(type(archive))))
    else:
        if not archive.get_dedispersed():
            archive.dedisperse()

    a = archive # assuming this is F- and T-scrunched already
    roll = None
    xlims = [0, 1]
    ylims = None
    phase = None
    isub = 0
    ichn = 0
    phase_roll = None
    axins = None

    if zoom == 'auto':
        int_sI = a[isub].get_Profile(0, ichn).get_amps()
        int_sI_roll, phase_roll, roll, xlims = roll_arrays(int_sI, ref=ref)
    elif zoom is not None and ('x' in zoom or ',' in zoom):
        m = re.match('(.*)[,x](.*)', zoom.lstrip(' '))
        if m is None:
            raise(RuntimeError("Improper format for x-zoom: "+zoom))

        try:
            xlims[0] = float(m.groups()[0].replace('m', '-'))
            xlims[1] = float(m.groups()[1].replace('m', '-'))
        except ValueError:
            raise(RuntimeError("Improper values for x-zoom: "+zoom))

    elif zoom is not None:
        raise(RuntimeError("Improper type for zoom: "+zoom))

    if zoomy is not None and ('x' in zoomy or ',' in zoomy):
        ylims = [0, 0]
        m = re.match('(.*)[,x](.*)', zoomy.lstrip(' '))
        if m is None:
            raise(RuntimeError("Improper format for y-zoom: "+zoomy))

        try:
            ylims[0] = float(m.groups()[0].replace('m', '-'))
            ylims[1] = float(m.groups()[1].replace('m', '-'))
        except ValueError:
            raise(RuntimeError("Improper values for y-zoom: "+zoomy))

    if zoom != 'auto' and centre is True:
        int_sI = a[isub].get_Profile(0, ichn).get_amps()
        _, phase_roll, roll, _ = roll_arrays(int_sI, ref=ref)

    if ex_rot and roll:
        roll += ex_rot
    elif ex_rot:
        roll = ex_rot

    ax2, axins, I_norm = plot_pols(a, ax2, phase_ref=ref, arr_roll=roll,
                                   norm=norm, add_inset=inset_bool,
                                   inset_lims=inset_lims, base_width=base_width,
                                   plot_wide=plot_wide, plot_iquv=plot_iquv)
    ax2.set_xlim(*xlims)
    if ylims is not None:
        ax2.set_ylim(*ylims)

    pa, pa_e, phase, lim = calc_pa(a, return_bool=True, threshold=thresh,
                                   base_width=base_width, err_tab=pae_tab,
                                   err_file=pae_file)

    if phase is None:
        phase = np.linspace(0, 1, len(pa), endpoint=False)

    if phase_roll is None:
        phase_roll = phase

    if plot_wide:
        if len(phase) != len(phase_roll):
            phase_roll = np.concatenate((phase_roll, phase_roll, phase_roll))
        phase_roll = np.concatenate((phase_roll-1, phase_roll, phase_roll+1))
        lim = np.concatenate((lim, lim, lim))
        pa = np.concatenate((pa, pa, pa))
        pa_e = np.concatenate((pa_e, pa_e, pa_e))
        ax1.errorbar(phase_roll[lim], pa[lim], yerr=pa_e[lim], marker='x',
                     color='k', markersize=2, linestyle='')
        ax1.set_xlim(-0.05, 1.05)
    else:
        ax1.errorbar(phase_roll[lim], pa[lim], yerr=pa_e[lim], marker='x',
                     color='k', markersize=2, linestyle='')
        ax1.set_xlim(*xlims)


    return(ax1, ax2, axins, I_norm)


def main(ppdot=False, db_file=None, profiles=False, histos=False,
         rrat_file=None, plot_pol=False, plot_polonly=False, archives=None,
         tscrunch=False, fscrunch=False, bscrunch=None, pscrunch=False,
         output=None, centre=False, zoom_phase=None, zoom_y=None,
         waterfall=False, toa_plot=False, toa_cfg=None, plot_pols_I=False,
         plot_pols_S=False, report_pols=False, wfall_wprof=False,
         shape="3x3", widths=None, pa_err_file=None, no_dedisp=False,
         vmax=1.0, verbose=False, census=False, high_res=False, extras=None,
         no_norm=False, rotate=None, iquv=False):
    """
    The beast that is make_paper_plots.main(). This function is designed
    to deal with the many input options, including looping over lists
    of archives to make many plots (or calculations).
    Use of this function in an external script is not recommended.

    """

    psr_list = ["J0750-6846", "J1115-0956", "J1207-4508", "J1244-1812",
                "J1337-4441", "J1406-4233", "J1421-4409", "J1523-3235",
                "J1544-0713", "J1604-3142", "J1631-2609", "J1646-1910",
                "J1700-0954", "J1759-5505", "J1828+1221", "J1910-0556",
                "J1921-0510", "J1923-0408", "J1928-0108", "J1940-0902",
                "J1942-2019", "J2001-0349", "J2136-5046"]
    rrat = "J1646-1910"

    if profiles:
        make_prof_gallery(archives, output, pscrunch, bscrunch, zoom_phase, shape, widths,
                          verbose, high_res=high_res)

    if report_pols is True and archives is None:
        raise(RuntimeError("Need archive name(s)"))
    elif report_pols is True:
        for filename in archives:
            calc_pol_fracs(filename)

    if wfall_wprof:
        if archives is None:
            raise(RuntimeError("Need archive name(s)"))

        if fscrunch:
            ordinate = 'time'
        else:
            ordinate = 'freq'

        for filename in archives:
            plot_wfall_wprof(filename, ordinate, fscr=fscrunch, tscr=tscrunch,
                             out=output, zoom=zoom_phase, high_res=high_res)

    if plot_pols_I:
        if archives is None:
            raise(RuntimeError("Need archives to make plots"))

        if output is "1/xs":
            out_dir = "./"
        else:
            if os.path.isdir(output):
                out_dir = output
            else:
                print("Output not a directory; changing to last directory in path given")
                out_dir = "./" if "/" not in output else os.path.split(output)[0]

        if pscrunch:
            print("Warning: pol. scrunching ignored for pol. plots")

        make_indiv_polprof(archives, out_dir, tscrunch, fscrunch, high_res=high_res)

    if plot_pols_S:
        if archives is None:
            raise(RuntimeError("Need archives to make plots"))

        if output is "1/xs":
            out_dir = "./"
        else:
            if os.path.isdir(output):
                out_dir = output
            else:
                print("Output not a directory; changing to last directory in path given")
                out_dir = "./" if "/" not in output else os.path.split(output)[0]

        if pscrunch:
            print("Warning: pol. scrunching ignored for pol. plots")

        make_all_polprof(archives, out_dir, tscrunch, fscrunch, verb=verbose,
                         bscrunch=bscrunch, zoom=zoom_phase, shape=shape,
                         widths=widths, pae_file=pa_err_file, extras=census,
                         db_file=db_file, zoomy=zoom_y, high_res=high_res,
                         extra_csv=extras, no_norm=no_norm, rot_csv=rotate, 
                         plot_iquv=iquv)

    if toa_plot and toa_cfg is not None:
        try:
            toa_pars = config_parser(toa_cfg)
        except:
            raise(RuntimeError("Could not read parameters from "+toa_cfg))

        if 'tim_file' not in toa_pars.keys() or \
           'par_file' not in toa_pars.keys():
            raise(RuntimeError("Not enough information in "+toa_cfg))
        else:
            tim_file = toa_pars['tim_file']
            par_file = toa_pars['par_file']
            if 'sel_file' in toa_pars.keys():
                sel_file = toa_pars['sel_file']
            else:
                sel_file = None
            if 'out_dir' in toa_pars.keys():
                out_dir = toa_pars['out_dir']
            else:
                out_dir = './'
            if 'out_file' in toa_pars.keys():
                out_file = toa_pars['out_file']
            else:
                out_file = 'toas.png'
            if 'fontsize' in toa_pars.keys():
                fontsize = int(toa_pars['fontsize'])
            elif 'fs' in toa_pars.keys():
                fontsize = int(toa_pars['fs'])
            else:
                fontsize = 14
            if 'high_res' in toa_pars.keys():
                high_res = True
            else:
                high_res = False
            fs = make_pretty(fontsize, high_res)
            if 'sequential' in toa_pars.keys():
                sequential = toa_pars['sequential']
            else:
                sequential = True
            if 'title' in toa_pars.keys():
                title = toa_pars['title']
            else:
                title = None
            if 'bw' in toa_pars.keys():
                bw = float(toa_pars['bw'])
            else:
                bw = 320
            if 'cfrq' in toa_pars.keys():
                cfrq = float(toa_pars['cfrq'])
            else:
                cfrq = 1382
            if 'flag' in toa_pars.keys():
                flag = toa_pars['flag']
            else:
                flag = None
            if 'nchn' in toa_pars.keys():
                nchn = int(toa_pars['nchn'])
            else:
                nchn = 1
            if 'key' in toa_pars.keys():
                key = toa_pars['key']
            else:
                key = 'meerkat'

        toas = get_res_fromtim(tim_file, par_file, sel_file, out_dir,
                               verb=False, key=key)
        plot_toas_fromarr(toas, fs, out_file, out_dir, sequential,
                          title, verb=False, bw=bw, cfrq=cfrq,
                          flag=flag, nchn=nchn, high_res=high_res)

    if ppdot and db_file:
        plot_ppdot(db_file, psr_list, high_res=high_res)

    if profiles:
        plot_profiles(psr_list, high_res=high_res)

    if histos:
        plot_histos(psr_list, high_res=high_res)

    if rrat_file:
        plot_rrat_wfall(rrat_file, out=output, high_res=high_res)

    if plot_pol and archives is not None:
        archive = archives[0]
        ar_obj = read_archive(archive)
        plot_Scyl(ar_obj, tscr=tscrunch, fscr=fscrunch, bscr=bscrunch,
                  out=output, centre=centre, zoom=zoom_phase, zoomy=zoom_y,
                  high_res=high_res)

    if waterfall and archives is not None:
        archive = archives[0]
        ar_obj = read_archive(archive, no_dedisp=no_dedisp)
        plt.clf()
        fig = plt.gcf()
        ax = fig.add_axes((0.2, 0.125, 0.7375, 0.8))
        ax = plot_waterfall(ar_obj, ax=ax, tscr=tscrunch, fscr=fscrunch,
                            vmax_frac=vmax, vmin_frac=0.0, high_res=high_res)
        if output is not None:
            plt.savefig(output, bbox_inches='tight')
        else:
            plt.show()


def get_res_fromtim(tim_file, par_file, sel_file=None, out_dir="./",
                    verb=False, key='meerkat'):
    tempo2_call = "tempo2 -nofit -set START 40000 -set FINISH 99999 "\
                  "-output general2 -s \"{{bat}} {{post}} {{err}} "\
                  "{{freq}} BLAH\n\" -nobs 1000000 -npsr 1 -f {} {}"
    awk_cmd = "awk '{print $1,$2,$3*1e-6,$4}'"
    temp_file = os.path.basename(tim_file).replace('.tim', '_res.txt')
    temp_file = os.path.join(out_dir, temp_file)

    # if a select file is given, include it
    if sel_file is not None:
        tempo2_call += " -select {}".format(sel_file)

    # copy the tim file into a new file to filter out uncertainties=0
    new_tim = os.path.join(out_dir, "temp.tim")
    with open(new_tim, 'w') as f:
        p = sproc.Popen(shplit("grep -v {} {}".format(key, tim_file)),
                        stdout=f)
        p.communicate()
        p = sproc.Popen(shplit("awk '{if ($4 > 0) {print $0}}' "+tim_file),
                        stdout=f)
        p.communicate()

    # call tempo2 to produce residuals that can be read in
    with open(temp_file, 'w') as f:
        if verb:
            print("Running tempo2 command: {}"
                  .format(tempo2_call.format(par_file, new_tim)))

        p1 = sproc.Popen(shplit(tempo2_call.format(par_file, new_tim)),
                         stdout=sproc.PIPE)
        p2 = sproc.Popen(shplit("grep BLAH"), stdin=p1.stdout,
                         stdout=sproc.PIPE)
        p3 = sproc.Popen(shplit(awk_cmd), stdin=p2.stdout, stdout=f)
        p1.wait()
        p2.wait()
        p3.wait()

    os.remove(new_tim)

    toas = np.loadtxt(temp_file, usecols=(0, 1, 2, 3),
                      dtype=[('mjd', 'f8'), ('res', 'f4'), ('err', 'f4'),
                             ('freq', 'f4')])
    if toas.size == 1:
        if verb:
            print("Only one ToA from {}; skipping".format(tim_file))
        toas = np.array([])

    if len(toas) == 0:
        print("No ToAs from tempo2 for {}".format(tim_file))

    return(toas)


def plot_toas_fromarr(toas, fs=14, out_file="toas.png", out_dir=None,
                      sequential=True, title=None, verb=False, high_res=False,
                      bw=856, cfrq=1284, flag='16ch64s', nchn=None):
    fs = make_pretty(fs, high_res)
    if out_dir:
        out_file = os.path.join(out_dir, out_file)

    if flag is None and nchn is None:
        raise(RuntimeError("Not enough information given: need flag or num. of channels"))

    # use semi-fixed color normalisation
    f_min = cfrq-bw/2
    f_max = cfrq+bw/2

    norm = colors.Normalize(vmin=f_min, vmax=f_max)

    fig = plt.figure(num=1)
    fig.set_size_inches(6, 4.5)
    ax = fig.gca()

    if sequential == 'True' or sequential is True:
        if verb:
            print("Plotting serial ToAs")

        if flag is not None and 'ch' in flag:
            nchn = int(flag.split('ch')[0])

        if nchn > 1:
            chan = range(nchn)
            freq_mins = [f_min+(i*bw/nchn) for i in chan]
            freq_maxs = [f_min+((i+1)*bw/nchn) for i in chan]
            pulse = 0
            last_chan = -1
            num = []

            for f in toas['freq']:
                for i, mi, ma in zip(chan, freq_mins, freq_maxs):
                    if mi < f < ma:
                        if i <= last_chan:
                            pulse += nchn

                        num.append(pulse+i)

                        last_chan = i
                        break

            if len(num) != len(toas['freq']):
                print(num, toas['freq'], freq_mins, freq_maxs)
                raise(RuntimeError("Error determining ToA Number for {}"
                                   .format(out_file)))

            xdata = np.array(num)
        else:
            xdata = np.arange(len(toas['freq']))
    else:
        if verb:
            print("Plotting against MJD")

        xdata = toas['mjd']

    if nchn > 1 and len(toas['res']) > 1:
        p2 = ax.scatter(xdata, toas['res']*1e6, s=8, c=toas['freq'],
                        marker='s', norm=norm, cmap='viridis')
        cb = fig.colorbar(p2, ax=ax, fraction=0.1)

        lines = ax.errorbar(xdata, toas['res']*1e6, yerr=1e6*toas['err'],
                            ls='', marker='', ms=1, zorder=0)[2]
        lines[0].set_color(cb.to_rgba(toas['freq']))
    else:
        cb = None
        p2 = ax.scatter(xdata, toas['res']*1e6, s=8, c='blue',
                        marker='s')
        ax.errorbar(xdata, toas['res']*1e6, yerr=1e6*toas['err'],
                    ls='', marker='', c='blue', zorder=0)

    if len(xdata) > 1:
        spread = xdata.max()-xdata.min()
    else:
        spread = 1

    xmin = xdata.min()-0.05*spread
    xmax = xdata.max()+0.05*spread
    ax.plot([xmin-0.1*spread, xmax+0.1*spread], [0, 0], ls='--', color='0.5')
    ax.set_xlim(xmin, xmax)

    if cb is not None:
        cb.set_label("Observing frequency (MHz)", rotation=270,
                     size=fs, labelpad=16)

    if sequential is True or sequential == "True":
        ax.set_xlabel("ToA Number", fontsize=fs)
    else:
        ax.set_xlabel("MJD", fontsize=fs)

    ax.set_ylabel("residuals ($\mu$s)", fontsize=fs)
    if title is not None and type(title) is str:
        ax.set_title(title, fontsize=fs+2)

    plt.savefig(out_file, bbox_inches='tight')


def rrat_joy_division(wf_data, s_count=1000, color=False, high_res=False):
    fs = make_pretty(16, high_res)

    plt.clf()
    fig = plt.figure(num=1)
    fig.set_size_inches(4, 8)
    ax = fig.gca()
    norm = colors.Normalize(vmin=0, vmax=20)
    cmap = cm.get_cmap('plasma')
    p_count = int(np.ceil(len(wf_data)/float(s_count)))
    num = len(wf_data)/p_count
    for i in range(p_count):
        if i < p_count-1:
            p = np.sum(wf_data[i*num:(i+1)*num], axis=0)
        else:
            p = np.sum(wf_data[i*num:], axis=0)
        bl = np.median(p)
        p -= bl
        ht = np.max(p[trim])
        p = p/ht
        p += i
        if color is False:
            ax.plot(p, 'k-')
        else:
            ax.plot(p, ls='-', marker=None, color=cmap(norm(i)))
    ax.set_xlim(xmin, xmax)
    ax.set_yticks([])
    ax.set_xlabel('Phase Bins', fontsize=fs)
    c = 'bw' if color is False else 'col'
    plt.savefig('joy_division_{}_{}.png'.format(s_count, c),
                bbox_inches='tight')
    #plt.show(block=False)


if __name__ == "__main__":
    args = proc_args()

    if args['ppdot'] and not args['db_file']:
        raise(RuntimeError("Cannot make P-Pdot diagram without db_file"))

    if args['plot_pol'] and not args['archives']:
        raise(RuntimeError("Cannot make polarisation plot without archive"))

    main(**args)

