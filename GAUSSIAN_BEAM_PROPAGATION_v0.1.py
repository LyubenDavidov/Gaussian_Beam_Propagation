# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 22:24:55 2023

@author: ljubo
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


#==============================================================================
# DEFINE CMAP OF CHOICE

plt.style.use('dark_background')

#==============================================================================
# DEFINE VARIABLES 

n2      = 10**(-20)         # [m^2 / V^2]   NONLINEARITY: INDEX OF REFRACTION
lamda   = 10**(-6)          # [m]           WAVELENGTH
etha    = 8.854*10**(-12)   # [C/Vm]        ELECTRIC PERMITTIVITY
w0      = 10**(-4)          # [m]           BEAM WAIST
z_min   = 0.0001            # [m]           DISTANCE TO THE BEAM WAIST
k       = 2*np.pi/lamda     # [m^-1]        WAVENUMBER


def waist(z):
    w_z = w0*np.sqrt(1+((2*(z-z_min))/(k*(w0**2)))**2);
    return w_z;



#==============================================================================
# RADIUS OF THE GAUSSIAN BEAM AT THE ENTRANCE z=0

d = waist(0)          # [m] 



#==============================================================================
# DEFINE X Y Z COORDINATES

STEPS       = 300

MIN_Z       = 0.0
MAX_Z       = 0.2
STEPS_Z     = 1000

x = np.linspace(-w0*10, w0*10, STEPS)
y = np.linspace(-w0*10, w0*10, STEPS)

xx,yy = np.meshgrid(x,y);

r = np.sqrt(xx*xx + yy*yy)
print(r.shape)
z = np.linspace(MIN_Z, MAX_Z, STEPS_Z)



#==============================================================================
# DEFINE R

def R(z):
    R_z = z*(1+((k*(w0**2))/(2*z))**2)
    return R_z;



#==============================================================================
# DEFINE INCIDENT FIELD

#argument = -1j*k*n2*z_min + 1j*np.arctan2(2*z_min, k*n2*(w0*w0)) + 1j*k*n2*((r*r)/(2*R(-z_min)))
#U_in = (d/w0)*np.exp(argument)*np.exp((-(r*r))/(waist(z_min)**2));
    
argument = -1j*k*n2*z_min + 1j*np.arctan2(2*z_min, k*n2*(w0*w0)) + 1j*k*n2*((r*r)/(2*R(-z_min)))
U_in = (d/w0)*np.exp(argument)*np.exp((-(r*r))/(w0**2));
    


#==============================================================================
# DEFINE PLOT TICKS

x_ticks = np.linspace(-1, 1, 5)
y_ticks = np.linspace(-1, 1, 5)
z_ticks = np.linspace(MIN_Z, MAX_Z, 5)



#==============================================================================
# PLOT THE INCIDENT GAUSSIAN BEAM

plt.imshow((np.abs(U_in))**2, cmap = 'jet', origin = 'lower', extent=[-1, 1, -1, 1])
bar_U = plt.colorbar()
bar_U.ax.set_title('[-]', fontsize=9)
plt.title("Intensity of the Incident Gaussian Beam \n (U$^{in}$ from the notes)")
plt.xlabel("x-direction [mm]")
plt.ylabel("y-direction [mm]")
plt.xticks(x_ticks);
plt.yticks(y_ticks);
plt.savefig("Incident_Gaussian_Beam_Cross_Section.png", dpi=190)
plt.show()



#==============================================================================
# CROSS SECTION OF THE GAUSSIAN BEAM

x_coor = int(U_in.shape[0]/2)
cross_section = np.transpose(abs(U_in[x_coor, :]));



#==============================================================================
# PLOT THE CROSS SECTION

x_ticks = np.linspace(-w0*10, w0*10, 11)

plt.plot(x*10**3, cross_section)
plt.xticks(x_ticks*10**3);
plt.xlabel("y-direction [mm]")
plt.ylabel("Intensity [-]")
plt.title("Cross Section Along x=0 of the Incident Gaussian Beam \n (U$^{in}$ from the notes)")
plt.savefig("1D_Cross_Section_Incident_Gaussian_Beam.png", dpi=190)
plt.show()



#==============================================================================
# SOLUTION OF LINEAR PARAXIAL CASE

U_RES = np.empty((STEPS, STEPS, STEPS_Z))

def U_PARAXIAL(z):
    
    argument = 1j*k*n2*(z-z_min) - 1j*np.arctan2(2*(z-z_min), k*(w0*w0)) + 1j*k*n2*0.5*((r*r)/(2*R(z-z_min)));
    U_P = (d/waist(z-z_min))*np.exp(argument)*np.exp((-(r*r))/(waist(z-z_min)**2));
    
    
    return U_P;
    
for i in range(len(z)):
    U_RES[:,:, i] = np.abs(U_PARAXIAL(z[i]))
    
    
    
#==============================================================================
# PLOT THE PARAXIAL SOLUTION

plt.imshow((U_RES[int(U_RES.shape[0]/2),:,:])**2, cmap = 'jet', origin = 'lower', extent=[MIN_Z, MAX_Z, -1, 1], interpolation='nearest', aspect='auto')
bar_U = plt.colorbar()
bar_U.ax.set_title('[-]', fontsize=9)
plt.title("Intensity of the Gaussian Beam \n Paraxial Solution (Analytical)")
plt.xlabel("Propagation distance z [m]")
plt.ylabel("Radial position [mm]")
plt.xticks(z_ticks);
y_ticks = np.linspace(-1, 1, 11)
plt.yticks(y_ticks)

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places

plt.savefig("Gaussian_Beam_Paraxial_Solution.png", dpi=190)
plt.show()
    


#==============================================================================
# DEFINE HANKEL TRANSFORM OF 2D DATA


from pyhank import HankelTransform
# noinspection PyUnresolvedReferences


# 1D Gaussian function ACCORDING TO THE EXAMPLE
#def gauss1d(x, x0, fwhm):
    #return np.exp(-2 * np.log(2) * ((x - x0) / fwhm) ** 2)
    
# MY DEFINITION OF THE INPUT FIELD AS PER THE NOTES, DIFFERENT FROM THE EXAMPLE
def gauss1d(r):    
    argument = -1j*k*n2*z_min + 1j*np.arctan2(2*(z_min), k*n2*(w0*w0)) + 1j*k*n2*((r*r)/(2*R(-z_min)))
    U_in = (d/waist(z_min))*np.exp(argument)*np.exp((-(r*r))/(w0*w0));
    
    return U_in



# Plotting function equivalent to Matlab's imagesc !!!NO NEED TO USE IT!!!
def imagesc(x: np.ndarray, y: np.ndarray, intensity: np.ndarray, axes=None, **kwargs):
    assert x.ndim == 1 and y.ndim == 1, "Both x and y must be 1d arrays"
    assert intensity.ndim == 2, "Intensity must be a 2d array"
    extent = (x[0], x[-1], y[-1], y[0])
    if axes is None:
        img = plt.imshow(intensity, extent=extent, **kwargs, aspect='auto', cmap='jet')
    else:
        img = axes.imshow(intensity, extent=extent, **kwargs, aspect='auto', cmap='jet')
    img.axes.invert_yaxis()
    return img


nr      = 1024                          # Number of sample points
r_max   = 10e-4                         # Maximum radius (0.1mm) 
r       = np.linspace(0, r_max, nr)



Nz      = 300                           # Number of z positions
z_max   = 0.2                           # Maximum propagation distance
z       = np.linspace(0, z_max, Nz)     # Propagation axis


Dr      = 1e-4                  # Beam radius (100um)
lambda_ = lamda                 # wavelength 
k0      = 2 * np.pi / lambda_   # Vacuum k vector


H = HankelTransform(order=0, radial_grid=r)


#Er = gauss1d(r, 0, Dr)     # Initial field as in the example
Er  = gauss1d(r)            # Initial field as per the notes 
ErH = H.to_transform_r(Er)  # Resampled field



# Convert from physical field to physical wavevector
EkrH = H.qdht(ErH)


# Pre-allocate an array for field as a function of r and z
Erz = np.zeros((nr, Nz), dtype=complex)
kz = np.sqrt(k0 ** 2 - H.kr ** 2)
for i, z_loop in enumerate(z):
    phi_z = kz * z_loop  # Propagation phase
    EkrHz = EkrH * np.exp(1j * phi_z)  # Apply propagation
    ErHz = H.iqdht(EkrHz)  # iQDHT
    Erz[:, i] = H.to_original_r(ErHz)  # Interpolate output
Irz = np.abs(Erz) ** 2


# PLOT THE INTIAL FIELD 
plt.figure()
plt.plot(r * 1e3, np.abs(Er) ** 2, r * 1e3, np.unwrap(np.angle(Er)),
         H.r * 1e3, np.abs(ErH) ** 2, H.r * 1e3, np.unwrap(np.angle(ErH)), '+')
plt.title('Initial electric field distribution')
plt.xlabel('Radial co-ordinate (r) /mm')
plt.ylabel('Field intensity /arb.')
plt.legend(['$|E(r)|^2$', '$\\phi(r)$', '$|E(H.r)|^2$', '$\\phi(H.r)$'])
plt.axis([0, 1, 0, 1])
plt.savefig("Initial_Field_Distribution.png", dpi=190)

# PLOT THE RADIAL WAVEVECTOR DISTRIBUTION
plt.figure()
plt.plot(H.kr, np.abs(EkrH) ** 2)
plt.title('Radial wave-vector distribution')
plt.xlabel(r'Radial wave-vector ($k_r$) /rad $m^{-1}$')
plt.ylabel('Field intensity /arb.')
plt.axis([0, 3e4, 0, np.max(np.abs(EkrH) ** 2)])
plt.savefig("Radial_Wavevector_Distribution.png", dpi=190)



#==============================================================================
# PLOT OF THE PROPAGATED GAUSSIAN BEAM

plt.figure()

# MIRROR THE 1D PROPAGATED BEAM
Irz_mirrored = np.flip(Irz, axis=0);
Irz = np.concatenate((Irz_mirrored,Irz),0)

# DEFINE TICKS FOR THE TWO AXES
x_ticks = np.linspace(0, 0.2, 5)
y_ticks = np.linspace(-1, 1, 11)

# PLOT THE PROPAGATED GAUSSIAN BEAM
plt.imshow(Irz, cmap = 'jet', origin = 'lower', extent=[0.0, 0.2, -1, 1], interpolation='nearest', aspect='auto')
bar_U = plt.colorbar()
bar_U.ax.set_title('[-]', fontsize=9)
plt.title("Radial Field Intensity \n Propagation of U$^{in}$ Using DHT")
plt.xlabel("Propagation distance z [m]")
plt.ylabel("Radial position [mm]")

plt.xticks(x_ticks);
plt.yticks(y_ticks);

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places

plt.savefig('Field_Intensity_Propagated_Incident_Field.png', dpi=190);

plt.show()



#==============================================================================
# DEFINE SOURCE: TO BE USED IN THE PICARD ITERATION

#etha_NL     = ...                          # CHOOSE A VALUE

'''
def F(U):
    f = etha_NL*(np.abs(U)**2)*U
    return f;

'''