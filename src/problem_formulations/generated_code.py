import numpy as np
log = np.log

def h(Sv, Nv):
    return  1/4*(2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)
def dh_dS(Sv, Nv):
    return  -Sv*log(Sv)/(Nv - Sv) + 1/4*(2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**2
def dh_dN(Sv, Nv):
    return  Nv*log(Nv)/(Nv - Sv) - 1/4*(2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**2
def d2h_dS2(Sv, Nv):
    return  -log(Sv)/(Nv - Sv) - 2*Sv*log(Sv)/(Nv - Sv)**2 - 1/(Nv - Sv) + 1/2*(2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3
def d2h_dN2(Sv, Nv):
    return  log(Nv)/(Nv - Sv) - 2*Nv*log(Nv)/(Nv - Sv)**2 + 1/(Nv - Sv) + 1/2*(2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3
def d2h_dSdN(Sv, Nv):
    return  Nv*log(Nv)/(Nv - Sv)**2 + Sv*log(Sv)/(Nv - Sv)**2 - 1/2*(2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3
def hessian_upper_diag(Sv, Nv):
    return  2/(2*log(Nv)/(Nv - Sv) - 4*Nv*log(Nv)/(Nv - Sv)**2 + 2/(Nv - Sv) + (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3) - 2*(2*Nv*log(Nv)/(Nv - Sv)**2 + 2*Sv*log(Sv)/(Nv - Sv)**2 - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)**2/(((2*Nv*log(Nv)/(Nv - Sv)**2 + 2*Sv*log(Sv)/(Nv - Sv)**2 - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)**2/(2*log(Nv)/(Nv - Sv) - 4*Nv*log(Nv)/(Nv - Sv)**2 + 2/(Nv - Sv) + (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3) + 2*log(Sv)/(Nv - Sv) + 4*Sv*log(Sv)/(Nv - Sv)**2 + 2/(Nv - Sv) - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)*(2*log(Nv)/(Nv - Sv) - 4*Nv*log(Nv)/(Nv - Sv)**2 + 2/(Nv - Sv) + (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)**2)
def hessian_lower_diag(Sv, Nv):
    return  -2/((2*Nv*log(Nv)/(Nv - Sv)**2 + 2*Sv*log(Sv)/(Nv - Sv)**2 - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)**2/(2*log(Nv)/(Nv - Sv) - 4*Nv*log(Nv)/(Nv - Sv)**2 + 2/(Nv - Sv) + (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3) + 2*log(Sv)/(Nv - Sv) + 4*Sv*log(Sv)/(Nv - Sv)**2 + 2/(Nv - Sv) - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)
def hessian_off_diag(Sv, Nv):
    return  2*(2*Nv*log(Nv)/(Nv - Sv)**2 + 2*Sv*log(Sv)/(Nv - Sv)**2 - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)/(((2*Nv*log(Nv)/(Nv - Sv)**2 + 2*Sv*log(Sv)/(Nv - Sv)**2 - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)**2/(2*log(Nv)/(Nv - Sv) - 4*Nv*log(Nv)/(Nv - Sv)**2 + 2/(Nv - Sv) + (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3) + 2*log(Sv)/(Nv - Sv) + 4*Sv*log(Sv)/(Nv - Sv)**2 + 2/(Nv - Sv) - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)*(2*log(Nv)/(Nv - Sv) - 4*Nv*log(Nv)/(Nv - Sv)**2 + 2/(Nv - Sv) + (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3))

S = 5.0
N = 2.0
A = np.array([
    [d2h_dN2(S, N), d2h_dSdN(S, N)],
    [d2h_dSdN(S, N), d2h_dS2(S, N)]
])
B = np.array([
    [hessian_upper_diag(S, N), hessian_off_diag(S, N)],
    [hessian_off_diag(S, N), hessian_lower_diag(S, N)]
])
print(A @ B)
