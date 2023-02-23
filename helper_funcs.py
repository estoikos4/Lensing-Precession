import numpy as np
from pycbc.types import FrequencySeries
# from scipy.integrate import quad
# from precession import Precession
# from lensing import Lensing

def Sn(f, delta_f=0.25, frequencySeries=True):
        """ ALIGO noise curve from arXiv:0903.0338
        """
        Sn_val = np.zeros_like(f)
        fs = 20
        for i in range(len(f)):
            if f[i] < fs:
                Sn_val[i] = np.inf
            else:
                S0 = 1E-49
                f0 = 215
                Sn_temp = np.power(f[i]/f0, -4.14) - 5 * np.power(f[i]/f0, -2) + 111 * ((1 - np.power(f[i]/f0, 2) + 0.5 * np.power(f[i]/f0, 4)) / (1 + 0.5 * np.power(f[i]/f0, 2)))
                Sn_val[i] = Sn_temp * S0
        if frequencySeries:
            return FrequencySeries(Sn_val, delta_f=delta_f)
        return Sn_val
'''
def integrand(f, precession_params, lensing_params):
    lensing_ini = Lensing(lensing_params)
    precessing_ini = Precession(precession_params)
    hL = lensing_ini.hI(f) * lensing_ini.F(f)
    hP = precessing_ini.precessing_strain(f) 
    return hL * np.conjugate(hP) / Sn(f)

def inner_product(precessing_params, lensing_params):
    FMIN = 20
    FCUT = Precession(precessing_params).get_f_cut()
    integral, _ = quad(integrand, FMIN, FCUT, args=(precessing_params, lensing_params))
    return integral
'''
