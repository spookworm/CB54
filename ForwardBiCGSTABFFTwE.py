from IPython import get_ipython
import numpy as np
import sys
import time
from lib import custom_functions_EM

# Clear workspace
get_ipython().run_line_magic('clear', '-sf')
# get_ipython().run_line_magic('reset', '-sf')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=20)


# START OF ForwardBiCGSTABFFTwE
Edata2D, Hdata2D = custom_functions_EM.EMsctCircle()

c_0, eps_sct, mu_sct, gamma_0, xS, NR, rcvr_phi, xR, N1, N2, dx, X1, X2, FFTG, a, CHI_eps, CHI_mu, Errcri = custom_functions_EM.initEM()
custom_functions_EM.plotEMContrast(CHI_eps, CHI_mu, X1, X2)
E_inc, ZH_inc = custom_functions_EM.IncEMwave(gamma_0, xS, dx, X1, X2)

tic0 = time.time()
w_E, exit_code, information = custom_functions_EM.ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, N1, N2, dx, FFTG, gamma_0, x0=None)
toc0 = time.time() - tic0
print("toc", toc0)

E_sct = custom_functions_EM.KopE(w_E, gamma_0, N1, N2, dx, FFTG)
E_val = custom_functions_EM.E(E_inc, E_sct)

# Test Initial Guess as final answer
N = np.shape(CHI_eps)[0]*np.shape(CHI_eps)[1]
tic1 = time.time()
N = np.shape(CHI_eps.flatten('F'))[0]
x0 = np.zeros((2*N, 1), dtype=np.complex128, order='F')
x0[0:N, 0] = w_E[0, :].flatten('F')
x0[N:2*N, 0] = w_E[1, :].flatten('F')
w_E_m, exit_code_m, information_m = custom_functions_EM.ITERBiCGSTABwE(E_inc, CHI_eps, Errcri, N1, N2, dx, FFTG, gamma_0, x0=x0)
toc1 = time.time() - tic1
print("toc", toc1)

custom_functions_EM.plotContrastSourcewE(w_E, X1, X2)
E_sct = custom_functions_EM.KopE(w_E, gamma_0, N1, N2, dx, FFTG)
E_val = custom_functions_EM.E(E_inc, E_sct)
custom_functions_EM.plotEtotalwavefield(E_inc, a, X1, X2, N1, N2)

# Drop the first and last columns and rows due to finite differences at border
custom_functions_EM.plotEtotalwavefield(E_sct[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)
custom_functions_EM.plotEtotalwavefield(E_val[:, 1:-1, 1:-1], a, X1[1:-1, 1:-1], X2[1:-1, 1:-1], N1-1, N2-1)

Edata, Hdata = custom_functions_EM.DOPwE(w_E, gamma_0, dx, xR, NR, X1, X2)
angle = rcvr_phi * 180 / np.pi
# displayDataBesselApproach(Edata, angle)
# displayDataBesselApproach(Hdata, angle)
custom_functions_EM.displayDataCompareApproachs(Edata2D, Edata, angle)
custom_functions_EM.displayDataCompareApproachs(Hdata2D, Hdata, angle)
