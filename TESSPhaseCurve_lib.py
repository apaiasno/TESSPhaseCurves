# TESSPhaseCurve_lib.py
#
# Library of routines for analyzing TESS phase curves.

# Imports
import numpy as np
import astropy.units as u
import lightkurve as lk

### Lightcurve component models ###
# Model functions adopted from  Wong et al. 2020

## Stellar pulsation model ##
zeta = lambda t, t_0_pulse, PI: ((t - t_0_pulse) % PI)/PI
Theta = lambda t, t_0_pulse, PI, alpha, beta: 1 + alpha*np.sin(2*np.pi*zeta(t, t_0_pulse, PI)) + \
beta*np.cos(2*np.pi*zeta(t, t_0_pulse, PI))
def Theta_func(params, t):
    return Theta(t, *params)

## Planet phase curve ##
phi = lambda t, t_0, P: ((t - t_0) % P)/P
psi_p = lambda t, t_0, P, fp, B1, delta: fp + B1 * np.cos(2*np.pi*phi(t, t_0, P) + delta)
def psi_p_func(params, t):
    return psi_p(t, *params)

## Stellar harmonics ##
# k = 1 harmonic
psi_star_1 = lambda t, t_0, P, A1: A1*np.sin(2*np.pi*phi(t, t_0, P)) 
# k = 2 harmonic
psi_star_2 = lambda t, t_0, P, A2, B2: A2*np.sin(2*np.pi*2*phi(t, t_0, P)) + B2*np.cos(2*np.pi*2*phi(t, t_0, P)) 
# k = 3 harmonic
psi_star_3 = lambda t, t_0, P, A3, B3: A3*np.sin(2*np.pi*3*phi(t, t_0, P)) + B3*np.cos(2*np.pi*3*phi(t, t_0, P))
# Combined stellar harmonics
psi_star_sum = lambda t, t_0, P, A1, A2, B2, A3, B3: 1 + psi_star_1(t, t_0, P, A1) + psi_star_2(t, t_0, P, A2, B2) + \
psi_star_3(t, t_0, P, A3, B3)
def psi_star_func(params, t):
    return psi_star(t, *params)

## Total lightcurve ##
psi_tot = lambda t, t_0, P, PI, alpha, beta, fp, delta, A1, B1, A2, B2, A3, B3: \
(psi_p(t, t_0, P, fp, B1, delta) + Theta(t, t_0, PI, alpha, beta) * psi_star_sum(t, t_0, P, A1, A2, B2, A3, B3))/(1. + fp)
def psi_tot_func(params, t):
    return psi_tot(t, *params)

### Data transformations to isolate specific components ###
def lc_transform_planet(lc, t_0, P, PI, alpha, beta, fp, A1, A2, B2, A3, B3):
    ''' Uses model parameters to transform TESS lightcurve to isolate planet phase curve.
        
        Parameters
        -----------
        lc : Lightkurve object
            Cleaned and unfolded TESS lightcurve with planet phase curve, stellar harmonics, and stellar pulsations.
        t_0 : float Quantity in units of days
            Mid-transit time
        P : float Quantity in units of days
            Orbital period
        PI : float Quantity in units of days
            Stellar pulsation period
        alpha : float
            Stellar pulsation sine amplitude
        beta : float
            Stellar pulsation cosine amplitude
        fp : float
            Mean planet atmospheric brightness (normalized to stellar flux)
        A1 : float
            Doppler beaming amplitude
        A2 : float
            Stellar k=2 harmonic sine amplitude
        B2 : float
            Ellipsoidal variation amplitude
        A3 : float
            Stellar k=3 harmonic sine amplitude
        B3 : float
            Stellar k=3 harmonic cosine amplitude

        Returns
        -------
        lc_planet : Lightkurve object
            Planet phase curve component extracted from TESS lightcurve
    '''
    lc_planet = lc.copy()
    pulse = Theta(lc_planet.time.value, t_0, PI, alpha, beta)
    star = psi_star_sum(lc_planet.time.value, t_0, P, A1, A2, B2, A3, B3)
    lc_planet.flux = (1+fp) * lc_planet.flux - star * pulse
    lc_planet.flux_err = (1+fp) * lc_planet.flux_err
    return lc_planet

def lc_transform_star(lc, t_0, P, PI, alpha, beta, fp, B1, delta):
    ''' Uses model parameters to transform TESS lightcurve to isolate stellar harmonics.
        
        Parameters
        -----------
        lc : Lightkurve object
            Cleaned and unfolded TESS lightcurve with planet phase curve, stellar harmonics, and stellar pulsations.
        t_0 : float Quantity in units of days
            Mid-transit time
        P : float Quantity in units of days
            Orbital period
        PI : float Quantity in units of days
            Stellar pulsation period
        alpha : float
            Stellar pulsation sine amplitude
        beta : float
            Stellar pulsation cosine amplitude
        fp : float
            Mean planet atmospheric brightness (normalized to stellar flux)
        B1 : float
            Planet phase curve amplitude
        delta : float
            Planet phase curve offset

        Returns
        -------
        lc_star : Lightkurve object
            Stellar harmonics component extracted from TESS lightcurve
    '''
    lc_star = lc.copy()
    pulse = Theta(lc_star.time.value, t_0, PI, alpha, beta)
    planet = psi_p(lc_star.time.value, t_0, P, fp, B1, delta)
    lc_star.flux = ((1+fp) * lc_star.flux - planet) / pulse
    lc_star.flux_err = (1+fp) * lc_star.flux_err / pulse
    return lc_star


def lc_transform_pulse(lc, t_0, P, fp, delta, A1, A2, B1, B2, A3, B3):
    ''' Uses model parameters to transform TESS lightcurve to isolate planet phase curve.
        
        Parameters
        -----------
        lc : Lightkurve object
            Cleaned and unfolded TESS lightcurve with planet phase curve, stellar harmonics, and stellar pulsations.
        t_0 : float Quantity in units of days
            Mid-transit time
        P : float Quantity in units of days
            Orbital period
        fp : float
            Mean planet atmospheric brightness (normalized to stellar flux)
        delta : float
            Planet phase curve offset
        A1 : float
            Doppler beaming amplitude
        A2 : float
            Stellar k=2 harmonic sine amplitude
        B1 : float
            Planet phase curve amplitude
        B2 : float
            Ellipsoidal variation amplitude
        A3 : float
            Stellar k=3 harmonic sine amplitude
        B3 : float
            Stellar k=3 harmonic cosine amplitude

        Returns
        -------
        lc_pulse : Lightkurve object
            Stellar pulsation component extracted from TESS lightcurve
    '''
    lc_pulse = lc.copy()
    planet = psi_p(lc_pulse.time.value, t_0, P, fp, B1, delta)
    star = psi_star_sum(lc_pulse.time.value, t_0, P, A1, A2, B2, A3, B3)
    lc_pulse.flux = ((1+fp) * lc_pulse.flux - planet) / star
    lc_pulse.flux_err = (1+fp) * lc_pulse.flux_err / star
    return lc_pulse

### Lightkurve helper routines ###
def fold_lk(lc, P, epoch_time):
    ''' Phase-folds a Lightkurve object and wraps the phase between 0 (transit) to 1 (0.5 is secondary eclipse).

        Parameters
        ----------
        lc : Lightkurve object
            A Lightkurve object with time in units of days
        P : float with time units
            Period to fold over
        epoch_time : float with time units
            Mid-transit reference time

        Returns
        -------
        lc_fold : Lightkurve object
            Lightkurve object with orbital phase (unitless) given by time attribute
    '''
    lc_fold = lc.fold(P, epoch_time=epoch_time, wrap_phase=1*u.dimensionless_unscaled, 
                                        normalize_phase=True)
    ind_order = np.argsort(lc_fold.time)

    lc_dict = {'time': lc_fold.time[ind_order] * u.day,
               'flux': lc_fold.flux[ind_order],
               'flux_err': lc_fold.flux_err[ind_order]
              }
    lc_fold = lk.LightCurve(lc_dict)
    return lc_fold



