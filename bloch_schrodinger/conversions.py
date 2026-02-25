from scipy.constants import h, hbar, m_u, c, eV


## Functions used to make conversions between a natural units to SI units
## We assume hbar = m = 1, with m the atomic species used. Its value can be changed here

m_kg = 6.941 * m_u # Mass of lithium in atomic unit
alpha_SI = hbar**2 / 2 / m_kg # Alpha in J^2 s^2 / m

def Hz_to_J(freq):
    """Converts a frequency in hertz to an energy in Joules, using E = h * freq
    """
    return h * freq

def J_to_Hertz(E):
    """Converts an energy in Joules to a frequency in Hertz, using freq = E / h
    """
    return E / h

def nm_J(wavelength):
    """Converts a wavelength in nm to an energy in Joules, using E = h * c / wavelength. Works in both directions
    """
    return h * c / (wavelength / 1e9)

def eV_to_J(E):
    """Go from eV to Joules.
    """
    return E * eV

def J_to_eV(E):
    """Go from Joules to eV
    """
    return E / eV

def fromE(energy, unit:str = "J"):
    """Converts energy to natural units.

    Args:
        energy: The energy to convert
        unit (str): Unit used, between Joules (J), electronVolts (eV), namometers (nm) and Hertz (Hz). 
        If nanometers or Hertz is used, the energy return is that of a photon with the corresponding wavelength or frequency.
        Default to "J"
    """
    
    if unit == 'J':
        fac = energy
    if unit == 'eV':
        fac = eV_to_J(energy)
    if unit == 'nm':
        fac = nm_J(energy)
    if unit == 'Hz':
        fac = Hz_to_J(energy)
    
    return fac / hbar
    