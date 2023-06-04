####################################################################################################
# This file contains the code for the disk compression simulation. The simulation is based on the
# paper "Diametral compression test with composite disk for dentin bond strength measurement
# – Finite element analysis" by Shih-Hao Huanga, Lian-Shan Linb, Alex S.L. Fokb, Chun-Pin Lina.
# The simulation is done using the finite element method.
####################################################################################################

import math

#######################################
##### sigma equations for the disk ####
#######################################


def sigma_x(x: int, y: int, R: float, T: float, p: float) -> float:
    """Calculates the stress state at the point (x, y) in the disk.
    Args:
        x (int): x coordinate
        y (int): y coordinate
        R (float): radius of the disk
        T (float): thickness of the disk
        p (float): pressure applied to the disk

    Returns:
        float: stress states x at the point (x, y).
    """

    if (R == y or R == -y) and x == 0:
        return 0

    return (
        (-2 * p)
        * (2.25 * 0.01)
        * (
            ((x**2) * (R - y) / ((((R - y) ** 2) + x**2) ** 2))
            + ((x**2) * (R + y) / ((((R + y) ** 2) + x**2) ** 2))
            - 1 / (2 * R)
        )
    )


def sigma_y(x: int, y: int, R: float, T: float, p: float) -> float:
    """Calculates the stress state at the point (x, y) in the disk.
    Args:
        x (int): x coordinate
        y (int): y coordinate
        R (float): radius of the disk
        T (float): thickness of the disk
        p (float): pressure applied to the disk

    Returns:
        float: stress states y at the point (x, y).
    """
    if (R == y or R == -y) and x == 0:
        return 0
    return (-2 * p) * (
        ((R - y) ** 3) / ((((R - y) ** 2) + x**2) ** 2)
        + ((R + y) ** 3) / ((((R + y) ** 2) + x**2) ** 2)
        - 1 / (2 * R)
    )


def tau(x: int, y: int, R: float, T: float, p: float) -> float:
    """Calculates the stress state at the point (x, y) in the disk.
    Args:
        x (int): x coordinate
        y (int): y coordinate
        R (float): radius of the disk
        T (float): thickness of the disk
        p (float): pressure applied to the disk

    Returns:
        float: stress states tau at the point (x, y).
    """
    if (R == y or R == -y) and x == 0:
        return 0
    return (
        (-2 * p)
        # * (2.25 * 0.01)
        * (
            ((x) * ((R + y) ** 2) / ((((R + y) ** 2) + x**2) ** 2))
            - ((x) * ((R - y) ** 2) / ((((R - y) ** 2) + x**2) ** 2))
        )
    )


###########################################
##### power law equations for the disk ####
###########################################


def phase_sigma(
    sigma_x: float,
    sigma_y: float,
    T: float,
    w_length: int,
    stress_const: float,
) -> float:
    """Calculates the phase angle for the given stress state.
    Args:
        sigma_x (float): stress state x
        sigma_y (float): stress state y
        T (float): thickness of the disk
        w_length (int): wave length (for red green and blue)
        stress_const (float): stress constant
    Returns:
        float: phase angle
    """
    return (2 * stress_const * (sigma_x - sigma_y)) / w_length


def stress_angle(
    sigma_x: float,
    sigma_y: float,
    tau: float,
    polar_angle: int = math.pi * (0.25),
) -> float:
    """Calculates the stress angle for the given stress state.
        We calculate it by calulating the angle of the diagonized vector of

        | sigma_x   tau     |      -->      | sigma_x - tau**2/sigma_y  0      |
        | tau       sigma_y |      -->      | 0                        sigma_y|

    Args:
        sigma_x (float): stress state x
        sigma_y (float): stress state y
        tau (float): stress state tau
        polar_able (int): polarizing angle
    Returns:
        float: max stress angle for each point
    """

    # calculate the angle of the diagonized vector
    if sigma_y == 0:
        return 0

    return (1 / 2) * math.atan(2 * tau / (sigma_x - sigma_y)) - polar_angle


def intensity(
    x: int, y: int, R: float, T: float, p: float, w_length: int, c: float
) -> float:
    """Calculates the intensity for the given stress state.


    Args:
        x (int): x coordinate
        y (int): y coordinate
        R (float): radius of the disk
        T (float): thickness of the disk
        p (float): pressure applied to the disk
        w_length (int): wave length (for red green and blue)
        c (float): stress constant
    Returns:
        float: intensity for each point in disk
    """

    # calculate the stress state for each point
    i_sigma_x = sigma_x(x, y, R, T, p)
    i_sigma_y = sigma_y(x, y, R, T, p)
    i_tau = tau(x, y, R, T, p)
    i_stress_angle = stress_angle(i_sigma_x, i_sigma_y, i_tau)
    # i_stress_angle = 1
    i_phase_sigma = phase_sigma(i_sigma_x, i_sigma_y, T, w_length, c)
    # calculate the magnitude of the diagonized vector
    return (math.sin(2 * i_stress_angle) ** 2) * (
        math.sin(i_phase_sigma / 2) ** 2
    )
