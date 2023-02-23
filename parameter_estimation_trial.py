from precession import Precession
from lensing import Lensing
import numpy as np
from pycbc.filter import match
from helper_funcs import Sn
from cobaya.run import run

solar_mass = 4.92624076 * 1e-6 #[solar_mass] = sec
giga_parsec = 1.02927125 * 1e17 #[giga_parsec] = sec
year = 31557600 #[year] = sec


def lnlike(y, ML, theta_tilde, omega_tilde, gamma_P):

    angle_params = {
                'theta_J': 0.,
                'phi_J': 0.,
                'theta_S': np.pi/3,
                'phi_S': 0.
                }

    common_params = { 
                'mcz': 20 * solar_mass, 'dist': 1.5 * giga_parsec, 
                'eta': 0.25, 'tc': 0.0, 'phi_c': 0.0
                }

    default_precession_params = {
    'theta_S' : angle_params['theta_S'], 
    'phi_S' : angle_params['phi_S'], 
    'theta_J' : angle_params['theta_J'], 
    'phi_J' : angle_params['phi_J'], 
    'mcz' : common_params['mcz'], 
    'dist': common_params['dist'], 
    'eta' : common_params['eta'], 
    'tc' : common_params['tc'], 
    'phi_c' : common_params['phi_c'],
    'theta_tilde': theta_tilde,
    'omega_tilde': omega_tilde,
    'gamma_P': gamma_P
    }

    default_lensing_params = {
    'theta_S' : angle_params['theta_S'], 
    'phi_S' : angle_params['phi_S'], 
    'theta_L' : angle_params['theta_J'], 
    'phi_L' : angle_params['phi_J'], 
    'mcz' : common_params['mcz'], 
    'dist': common_params['dist'], 
    'eta' : common_params['eta'], 
    'tc' : common_params['tc'], 
    'phi_c' : common_params['phi_c'],
    'y': y,
    'MLz': ML
    }

    precession_init = Precession(default_precession_params)
    lensing_int = Lensing(default_lensing_params)

    f_cut = Precession(default_precession_params).get_f_cut()
    f_min = 20
    delta_f = 0.25
    f_range = np.arange(f_min, f_cut, delta_f)
    psd = Sn(f_range)

    hP = precession_init.precessing_strain(f_range, delta_f=delta_f)
    hL = lensing_int.lensing_strain(f_range, delta_f=delta_f)

    Match = match(hP, hL, psd)[0]
    return 1 - Match

info = {
    "likelihood": {
        "external": lnlike},
    "params": dict([
        ("y", {
            "prior": {"min": .1, "max": 1.},
            "latex": r"y"}),
	    ("ML", {
            "prior": {"min": 1e2 * solar_mass, "max": 1e5 * solar_mass},
            "latex": r"a"}),
	    ("theta_tilde", {
            "prior": {"min": 1., "max": 8.},
            "latex": r"\tilde{\theta}"}),
	    ("omega_tilde", {
            "prior": {"min": 1., "max": 6.},
            "latex": r"\tilde{\omega}"}),
	    ("gamma_P", {
            "prior": {"min": 0., "max": np.pi},
            "latex": r"\gamma_P"}),
	    
         ] ),
    
    "sampler": {
        "mcmc": {'max_tries':40000}}, 'output':'/Users/saifali/Desktop/Lensing_precession/chains/' }

updated_info, sampler = run(info)