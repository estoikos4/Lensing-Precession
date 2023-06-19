import numpy as np
import pandas as pd
from scipy.integrate import odeint
from pycbc.types import FrequencySeries

class Precession():

    def __init__(self, params=None) -> None:
        
        self.params = params
        
        # non-precession/unlensed parameters
        self.theta_S = params['theta_S']
        self.phi_S = params['phi_S']
        self.theta_J = params['theta_J']
        self.phi_J = params['phi_J']
        self.mcz = params['mcz']
        self.dist = params['dist']
        self.eta = params['eta']
        self.tc = params['tc']
        self.phi_c = params['phi_c']

        # precession parameters
        self.theta_tilde = params['theta_tilde']
        self.omega_tilde = params['omega_tilde']
        self.gamma_P = params['gamma_P']

        # some converters/constants
        
        self.SOLMASS2SEC = 4.92624076 * 1e-6 # solar mass -> seconds
        self.GIGAPC2SEC = 1.02927125 * 1e17 # gigaparsec -> seconds
        self.FMIN = 20 # lower frequency of the detector sensitivity band [Hz]

    def get_total_mass(self):
        """ Total mass from chirp mass [seconds]
        """
        return self.mcz/(self.eta**(3/5))

    def get_f_cut(self):
        """ f_cut
        """
        return 1/(6**(3/2)*np.pi*self.get_total_mass())

    def get_theta_LJ(self, f):
        """ theta_LJ 
        """
        return 0.1*self.theta_tilde*(f/self.get_f_cut())**(1/3)

    def get_phi_LJ(self, f):
        """ phi_LJ
        """
        num = 52.083 * self.omega_tilde
        deno = (self.get_total_mass()/self.SOLMASS2SEC) * np.pi**(8/3) * self.mcz**(5/3) * self.get_f_cut()**(5/3)
        phi_LJ_amp = num/deno
        return phi_LJ_amp * (1/self.FMIN - 1/f) + self.gamma_P

    def amp_prefactor(self) -> float:
        """ amplitude prefactor calculated using chirp mass and distance 
        """
        amp_prefactor = np.sqrt(5/96)*np.pi**(-2/3)*self.mcz**(5/6)/self.dist
        return amp_prefactor

    def precession_angles(self):
        """ some angles
        """
        Cie = np.cos(self.theta_S) * np.cos(self.theta_J) + np.sin(self.theta_S) * np.sin(self.theta_J) * np.cos(self.phi_S - self.phi_J)
        Sie = np.sqrt(1-Cie**2.)
        if Sie == 0:
            COe = 1
            SOe = 0
        else:
            COe = (np.cos(self.theta_S)*np.sin(self.theta_J)*np.cos(self.phi_J - self.phi_S)-np.sin(self.theta_S)*np.cos(self.theta_J))/(Sie)
            SOe = (np.sin(self.theta_J)*np.sin(self.phi_J - self.phi_S))/(Sie)
        return Cie, Sie, COe, SOe
    
    def LdotN(self, f):
        Cie, Sie, COe, SOe = self.precession_angles()
        LdotN = Cie*np.cos(self.get_theta_LJ(f))+Sie*np.sin(self.get_theta_LJ(f))*np.sin(self.get_phi_LJ(f))
        return LdotN

    def polarization_angles(self, f):
        Cie, Sie, COe, SOe = self.precession_angles()
        # polarization angle for precession

        # define cospsitest
        term01 = np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * COe * np.sin(self.theta_S)
        term02 = np.cos(self.get_theta_LJ(f)) * np.sin(self.theta_S) * np.sin(self.theta_J) * np.sin(self.phi_J - self.phi_S)
        term03 = np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * SOe * Cie * np.sin(self.theta_S)
        cospsitest0 = term01 + term02 - term03

        # define sinpsitest
        term11 = np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * SOe * np.sin(self.theta_S)
        term12 = np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * COe * np.sin(self.theta_S) * Cie
        term13 = np.cos(self.get_theta_LJ(f)) * np.cos(self.theta_J) 
        term14 = np.cos(self.get_theta_LJ(f)) * np.cos(self.theta_S) * Cie  
        sinpsitest0 = term11 + term12 + term13 - term14
        
        #normalizing cospsi and sinpsi
        if Sie == 0:
            cospsitest = cospsitest0
            sinpsitest = sinpsitest0
        else:
            cospsitest = (cospsitest0)/(np.sqrt(cospsitest0**2+sinpsitest0**2))
            sinpsitest = (sinpsitest0)/(np.sqrt(cospsitest0**2+sinpsitest0**2))
        
        cos2psi = cospsitest**2.-sinpsitest**2.
        sin2psi = 2*cospsitest*sinpsitest
        return cos2psi, sin2psi


    ### get the amplitude
    def amplitude(self, f) -> np.array:
        """ Non-precessin/unlensed amplitude
        """
        LdotN = self.LdotN(f)
        cos2psi, sin2psi = self.polarization_angles(f)

        # beam patterns
        Fp = (1./2.)*(1+np.cos(self.theta_S)**2)*np.cos(2*self.phi_S)*cos2psi - np.cos(self.theta_S)*np.sin(2*self.phi_S)*sin2psi
        Fc = (1./2.)*(1+np.cos(self.theta_S)**2)*np.cos(2*self.phi_S)*sin2psi + np.cos(self.theta_S)*np.sin(2*self.phi_S)*cos2psi

        amp = self.amp_prefactor()*f**(-7/6)*np.sqrt(4*Fc**2*LdotN**2+Fp**2*(1+LdotN**2)**2)
        return amp

    ### get the phase phi_P
    def phase_phi_P(self, f):
        
        LdotN = self.LdotN(f)
        cos2psi, sin2psi = self.polarization_angles(f)

        #beam patterns
        Fp = (1./2.)*(1+np.cos(self.theta_S)**2)*np.cos(2*self.phi_S)*cos2psi - np.cos(self.theta_S)*np.sin(2*self.phi_S)*sin2psi
        Fc = (1./2.)*(1+np.cos(self.theta_S)**2)*np.cos(2*self.phi_S)*sin2psi + np.cos(self.theta_S)*np.sin(2*self.phi_S)*cos2psi

        phi_p_temp = np.arctan2(2*LdotN*Fc, (1+LdotN**2)*Fp)
        phi_p = np.unwrap(phi_p_temp, discont=np.pi)
        return phi_p

    ### get the delta phi_P

    def Lcomps(self, f) -> np.array:
        """ L components
        """
        Cie, Sie, COe, SOe = self.precession_angles()
        # define Ldots
        Lx1 = np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * COe * np.sin(self.phi_S)
        Lx2 = np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * SOe * np.cos(self.phi_S) * np.cos(self.theta_S)
        Lx3 = np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * SOe * np.sin(self.phi_S) * Cie
        Lx4 = np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * COe * np.cos(self.phi_S) * Cie * np.cos(self.theta_S)
        Lx5 = np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * np.cos(self.phi_S) * Sie * np.sin(self.theta_S)
        Lx6 = np.cos(self.get_theta_LJ(f)) * np.sin(self.theta_J) * np.cos(self.phi_J)
        Lx  = -Lx1 - Lx2 + Lx3 - Lx4 + Lx5 + Lx6
        
        Ly1 = np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * COe * np.cos(self.phi_S)
        Ly2 = np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * SOe * np.sin(self.phi_S) * np.cos(self.theta_S)
        Ly3 = np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * SOe * np.cos(self.phi_S) * Cie
        Ly4 = np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * COe * np.sin(self.phi_S) * Cie * np.cos(self.theta_S)
        Ly5 = np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * np.sin(self.phi_S) * Sie * np.sin(self.theta_S)
        Ly6 = np.cos(self.get_theta_LJ(f)) * np.sin(self.theta_J) * np.sin(self.phi_J)
        Ly  = Ly1 - Ly2 - Ly3 - Ly4 + Ly5 + Ly6
        
        Lz1 = np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * SOe * np.sin(self.theta_S)
        Lz2 = np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * COe * np.sin(self.theta_S) * Cie
        Lz3 = np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * Sie * np.cos(self.theta_S)
        Lz4 = np.cos(self.get_theta_LJ(f)) * np.cos(self.theta_J)
        Lz  = Lz1 + Lz2 + Lz3 + Lz4
        return Lx, Ly, Lz
    
    def N_comps(self) -> float:
        """ N components
        """
        Nx = np.sin(self.theta_S)*np.cos(self.phi_S)
        Ny = np.sin(self.theta_S)*np.sin(self.phi_S)
        Nz = np.cos(self.theta_S)
        return Nx, Ny, Nz
    
    def f_dot(self, f):
        """ df/dt from Cutler Flanaghan 1994
        """
        prefactor = (96/5)*np.pi**(8/3)*self.mcz**(5/3)*f**(11/3)
        term1 = 1
        term2 = (743/336 + (11/4)*self.eta)*(np.pi*self.get_total_mass()*f)**(2/3)
        term3 = 4*np.pi*(np.pi*self.get_total_mass()*f)
        return prefactor*(term1-term2+term3)

    def integrand_delta_phi(self, y, f):
        """ integrand for delta phi p (equation in Apostolatos 1994)
        """
        Cie, Sie, COe, SOe = self.precession_angles()
        dfdt_inv = self.f_dot(f)**(-1)
        Lx, Ly, Lz = self.Lcomps(f)
        Nx, Ny, Nz = self.N_comps()
        LdotN = self.LdotN(f)

        derivf_omega = (1e3 * self.omega_tilde * (f/self.get_f_cut())**(5/3)) / (self.get_total_mass()/self.SOLMASS2SEC)
        derivf_theta = (0.1/3) * self.theta_tilde * (f**(-2)/self.get_f_cut())**(1/3)

        Lx1_dot = -derivf_omega * dfdt_inv * np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * COe * np.sin(self.phi_S) + derivf_theta * np.cos(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * COe * np.sin(self.phi_S)
        Lx2_dot = -derivf_omega * dfdt_inv * np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * SOe * np.cos(self.phi_S) * np.cos(self.theta_S) + derivf_theta * np.cos(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * SOe * np.cos(self.phi_S) * np.cos(self.theta_S)
        Lx3_dot =  derivf_omega * dfdt_inv * np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * SOe * np.sin(self.phi_S) * Cie + derivf_theta * np.cos(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * SOe * np.sin(self.phi_S) * Cie
        Lx4_dot =  derivf_omega * dfdt_inv * np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * COe * np.cos(self.phi_S) * Cie * np.cos(self.theta_S) + derivf_theta *  np.cos(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * COe * np.cos(self.phi_S) * Cie * np.cos(self.theta_S)
        Lx5_dot =  derivf_omega * dfdt_inv * np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * np.cos(self.phi_S) * Sie * np.sin(self.theta_S) + derivf_theta * np.cos(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * np.cos(self.phi_S) * Sie * np.sin(self.theta_S)
        Lx6_dot = -derivf_theta * np.sin(self.get_theta_LJ(f)) * np.sin(self.theta_J) * np.cos(self.phi_J)
        Lx_dot  = -Lx1_dot - Lx2_dot + Lx3_dot - Lx4_dot + Lx5_dot + Lx6_dot

        Ly1_dot = -derivf_omega * dfdt_inv * np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * COe * np.cos(self.phi_S) + derivf_theta * np.cos(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * COe * np.cos(self.phi_S)
        Ly2_dot = -derivf_omega * dfdt_inv * np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * SOe * np.sin(self.phi_S) * np.cos(self.theta_S) + derivf_theta * np.cos(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * SOe * np.sin(self.phi_S) * np.cos(self.theta_S)
        Ly3_dot =  derivf_omega * dfdt_inv * np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * SOe * np.cos(self.phi_S) * Cie + derivf_theta * np.cos(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * SOe * np.cos(self.phi_S) * Cie
        Ly4_dot =  derivf_omega * dfdt_inv * np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * COe * np.sin(self.phi_S) * Cie * np.cos(self.theta_S) + derivf_theta * np.cos(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * COe * np.sin(self.phi_S) * Cie * np.cos(self.theta_S)
        Ly5_dot =  derivf_omega * dfdt_inv * np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * np.sin(self.phi_S) * Sie * np.sin(self.theta_S) + derivf_theta * np.cos(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * np.sin(self.phi_S) * Sie * np.sin(self.theta_S)
        Ly6_dot = -derivf_theta * np.sin(self.get_theta_LJ(f)) * np.sin(self.theta_J) * np.sin(self.phi_J)
        Ly_dot  = Ly1_dot - Ly2_dot - Ly3_dot - Ly4_dot + Ly5_dot + Ly6_dot

        Lz1_dot = -derivf_omega * dfdt_inv * np.sin(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * SOe * np.sin(self.theta_S) + derivf_theta * np.cos(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * SOe * np.sin(self.theta_S)
        Lz2_dot =  derivf_omega * dfdt_inv * np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * COe * np.sin(self.theta_S) * Cie + derivf_theta * np.cos(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * COe * np.sin(self.theta_S) * Cie
        Lz3_dot =  derivf_omega * dfdt_inv * np.sin(self.get_theta_LJ(f)) * np.cos(self.get_phi_LJ(f)) * Sie * np.cos(self.theta_S) + derivf_theta * np.cos(self.get_theta_LJ(f)) * np.sin(self.get_phi_LJ(f)) * Sie * np.cos(self.theta_S)
        Lz4_dot = -derivf_theta * np.sin(self.get_theta_LJ(f)) * np.cos(self.theta_J)
        Lz_dot = Lz1_dot + Lz2_dot + Lz3_dot + Lz4_dot

        deltaphi_x_P = Lx_dot * (Ly*Nz-Lz*Ny)
        deltaphi_y_P = Ly_dot * (-Lx*Nz+Lz*Nx)
        deltaphi_z_P = Lz_dot * (Lx*Ny-Ly*Nx)

        k = 0
        integrand_delta_phi = (deltaphi_z_P+deltaphi_y_P+deltaphi_x_P)*LdotN/(1-LdotN**2) + k*y
        return integrand_delta_phi

    def phase_delta_phi(self, f):
        """ integrate the delta_phi integrand 
        """
        integral = odeint(self.integrand_delta_phi, 0, f)
        return np.squeeze(integral)

    def Psi(self, f):
        """ GW phase
        """
        x = (np.pi*self.get_total_mass()*f)**(2/3)
        term1 = 2*np.pi*f*self.tc - self.phi_c - np.pi/4
        prefactor = (3/4)*(8*np.pi*self.mcz*f)**(-5/3)
        term2 = 1 + (20/9)*(743/336 + (11/4)*self.eta)*x - 16*np.pi*x**(3/2)
        Psi = term1 + prefactor * term2
        return Psi
    
    def precessing_strain(self, f, delta_f=0.25, frequencySeries=True):
        """ precessing GW
        """
        precessing_strain = self.amplitude(f) * np.exp(1j*(self.Psi(f) - self.phase_phi_P(f) + 2*self.phase_delta_phi(f)))
        if frequencySeries:
            return FrequencySeries(precessing_strain, delta_f, delta_f)
        return precessing_strain
