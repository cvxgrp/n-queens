#!/usr/bin/sage

r"""
    This code computes closed form expressions for:
        - h(s, n) = int_0^1 g((1 - y)s + y n) dy
            where s and n are real numbers and g(x) = x log(x)
        - dh/ds
        - dh/dn
        - dh2f/ds^2
        - dh2f/dn^2
        - dh2f/dsdn

    The output can be copy-and-pasted into Python code by running
        sage generate_closed_form_xqueenon_integral.sage | sed 's/\^/**/g'
    
"""

Sv, Nv, y = var('Sv, Nv, y')

assume(Sv > 0)
assume(Nv > 0)
assume(Sv < 1)
assume(Nv < 1)

g(x) = x * log(x)

objective_term = integral(g((1 - y) * Sv + y * Nv), y, 0, 1)
print("def h(Sv, Nv):\n    return ", objective_term)

print("def dh_dS(Sv, Nv):\n    return ", diff(objective_term, Sv))
print("def dh_dN(Sv, Nv):\n    return ", diff(objective_term, Nv))
print("def d2h_dS2(Sv, Nv):\n    return ", diff(objective_term, Sv, 2))
print("def d2h_dN2(Sv, Nv):\n    return ", diff(objective_term, Nv, 2))
print("def d2h_dSdN(Sv, Nv):\n    return ", diff(objective_term, Sv, Nv))
