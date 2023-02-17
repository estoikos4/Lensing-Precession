import numpy as np
import scipy.special as sc
import mpmath as mp

class Lensing():

    def __init__(self, params = None):

        self.params = params

        assert type(self.params == dict), "Parameters should be a dictionary"

        # unlensed parameters
        self.theta_s = params['theta_S']
        self.phi_s = params['phi_S']
        self.theta_l = params['theta_L']
        self.phi_l = params['phi_L']
        self.mcz = params['mcz']
        self.dist = params['dist']
        self.eta = params['eta']
        self.tc = params['tc'] 
        self.phi_c = params['phi_c']

        # lensed parameters
        self.M_lz = params['MLz']
        self.y = params['y']

    def mass_conv(self):
        """Converts chirp mass to total mass. M = mcz/eta^(3/5)
        """
        M_val = self.mcz/(self.eta**(3/5))
        return M_val

    def l_dot_n(self):
        """TODO
        """
        cos_term = np.cos(self.theta_s) * np.cos(self.theta_l)
        sin_term = np.sin(self.theta_s) * np.sin(self.theta_l) * np.cos(self.phi_s - self.phi_l)
        inner_prod = cos_term + sin_term
        return inner_prod

    def amp(self):
        """TODO
        """
        amplitude = np.sqrt(5/96) * np.pi**(-2/3) * self.mcz**(5/6) / (self.dist)
        return amplitude

    def psi(self, f):
        """eqn 3.13 in Cutler-Flanaghan 1994
        """
        x = (np.pi*self.mass_conv()*f)**(2/3)
        term1 = 2*np.pi*f*self.tc - self.phi_c - np.pi/4
        prefactor = (3/4)*(8*np.pi*self.mcz*f)**(-5/3)
        term2 = 1 + (20/9)*(743/336 + (11/4)*self.eta)*x - 16*np.pi*x**(3/2)
        Psi = term1 + prefactor * term2
        return Psi

    def psi_s(self):

        numerator = np.cos(self.theta_l)-np.cos(self.theta_s)*(self.l_dot_n())
        denominator = np.sin(self.theta_s)*np.sin(self.theta_l)*np.sin(self.phi_l-self.phi_s)

        psi_s_val = np.arctan2(numerator, denominator)
        return psi_s_val


    def fIp(self):
        """TODO
        """

        term_1 = (1 / 2 * (1 + np.power(np.cos(self.theta_s), 2)) * np.cos(2*self.phi_s)* np.cos(2*self.psi_s()))
        term_2 = (np.cos(self.theta_s) * np.sin(2*self.phi_s)* np.sin(2*self.psi_s()))

        fIp_val = term_1 - term_2
        return fIp_val

    def fIc(self):
        """TODO
        """

        term_1 = (1 / 2 * (1 + np.power(np.cos(self.theta_s), 2)) * np.cos(2*self.phi_s)
                    * np.sin(2*self.psi_s()))
        term_2 = (np.cos(self.theta_s) * np.sin(2*self.phi_s)
                    * np.cos(2*self.psi_s()))

        fIc_val = term_1 + term_2
        return fIc_val

    def lambdaI(self):
        """TODO
        """

        term_1 = np.power(2 * self.l_dot_n() * self.fIc(), 2)
        term_2 = np.power((1 + np.power(self.l_dot_n(), 2)) * self.fIp(), 2)
        lambdaI_val = np.sqrt(term_1 + term_2)
        return lambdaI_val
 
    def phi_pI(self):
        """TODO
        """

        numerator = (2 * self.l_dot_n() * self.fIc())
        denominator = ((1 + np.power(self.l_dot_n(), 2)) * self.fIp())

        phi_pI_val = np.arctan2(numerator, denominator)
        return phi_pI_val

    def hI(self, f):
        """TODO
        """

        term_1 = self.lambdaI()
        term_2 = (np.exp(-1j * self.phi_pI()))
        term_3 = self.amp() * f**(-7/6)
        term_4 = np.exp(1j * self.psi(f))

        signal_I = term_1 * term_2 * term_3 * term_4
        
        return signal_I

    def F(self, f):
        """ PM amplification factor
        """
        self.w = 8 * np.pi * self.M_lz * f
        x_m = 0.5 * (self.y + np.sqrt(self.y**2 + 4))
        phi_m = np.power((x_m - self.y) , 2) / 2 - np.log(x_m)

        first_term = np.exp(np.pi * self.w / 4 + 1j * (self.w / 2) * (np.log(self.w / 2) - 2 * phi_m)) 
        second_term = sc.gamma(1 - 1j * (self.w / 2))
        # broadcasting mp hyp1f1 function to NumPy ufunc
        hyp1f1_np = np.frompyfunc(mp.hyp1f1, 3, 1)
        third_term = hyp1f1_np(1j * self.w / 2, 1, 1j * (self.w / 2) * (self.y**2))

        F_val = np.complex128(first_term * second_term * third_term)
        return F_val
