import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq

font = {"size": 14}
matplotlib.rc("font", **font)


def fit_circle(x, y):
    """
    Fits a circle to a two-dimensional data set.
    Circle function to fit: (x-a)^2 + (y-b)^2 = r^2

    Parameters:
    x: Array containing samples of the first dimension.
    y: Array containing samples of the second dimension.

    Returns:
    xc: x-coordinate of the center of the fitted circle.
    yc: y-coordinate of the center of the fitted circle.
    R:  Radius of the fitted circle
    """

    def calc_R(xc, yc):
        """Calculate the distance of each 2D points from the center (xc, yc)"""
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f(c):
        """Calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc)"""
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    # Initial guess for the center of the circle
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m

    # Perform the least-squares fitting
    center, _ = leastsq(f, center_estimate)

    xc, yc = center
    Ri = calc_R(xc, yc)
    R = Ri.mean()
    return xc, yc, R


def calculate_tangent(xc, yc, xp, yp):
    """
    Calculate the equation of the tangent line to a circle at a given point.

    Parameters:
    center (tuple): The (x, y) coordinates of the center of the circle.
    radius (float): The radius of the circle.
    point (tuple): The (x, y) coordinates of the point on the circle.

    Returns:
    angle: Angle of the tangent on the resonance circle in degree
    """

    # Edge case: Vertical line
    if yp == yc:
        return float("inf"), float("inf")

    # Calculate slope of the tangent
    slope_normal = (yp - yc) / (xp - xc)
    slope_tangent = -1 / slope_normal

    # Calculate the y-intercept of the tangent line
    y_intercept = yp - slope_tangent * xp

    return slope_tangent, y_intercept


def rotate_circle(data_real, data_imag, angle):
    z = [real + 1j * imag for real, imag in zip(data_real, data_imag)]

    r = np.abs(z)  # Magnitude
    theta = np.angle(z)  # Angle

    new_theta = theta + angle

    # Convert back to rectangular form
    new_z = r * (np.cos(new_theta) + 1j * np.sin(new_theta))

    new_data_real = np.real(new_z)
    new_data_imag = np.imag(new_z)

    return new_data_real, new_data_imag


def plot_result(data_real, data_imag, xc, yc, xr, yr, R, filepath):

    slope, intercept = calculate_tangent(xc, yc, xr, yr)
    angle = np.arctan(slope)

    # Plot input data
    plt.plot(data_real[0::2], data_imag[0::2], ".")

    # Plot fitted circle
    circle = plt.Circle((xc, yc), R, fill=False, edgecolor="red", linewidth=2)
    plt.gca().add_patch(circle)

    # Mark circle center
    plt.plot(xc, yc, "x", markersize=10, color="red")

    # Mark readout point
    plt.plot(xr, yr, "*", markersize=10, linewidth=3, color="black")

    # Plot tangent
    # xt = np.linspace(np.min(data_real), np.max(data_real), num=2)
    xt = np.linspace(-1000, 1000, num=2)
    yt = slope * xt + intercept
    plt.plot(xt, yt, color="green")

    # Plot angle
    arc_radius = 50
    theta = np.linspace(min(angle, 0), max(0, angle), 100)
    x_arc = xr + arc_radius * np.cos(theta)
    y_arc = yr + arc_radius * np.sin(theta)
    plt.plot(x_arc, y_arc, color="black")
    plt.text(
        xr + arc_radius * np.cos(angle / 2) + 5,
        yr + arc_radius * np.sin(angle / 2),
        f"{np.abs(np.degrees(angle)):.2f}°",
        color="black",
        ha="left",
    )

    # Plot rotated tangent
    plt.axhline(y=yr, color="black", linestyle=":")

    # Define xlim and ylim
    x_diff = np.max(data_real) - np.min(data_real)
    y_diff = np.max(data_imag) - np.min(data_imag)
    diff = max(x_diff, y_diff) * 1.2
    plt.xlim([xc - diff / 2, xc + diff / 2])
    plt.ylim([yc - diff / 2, yc + diff / 2])

    plt.gca().set_aspect("equal")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid()
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")

    if not filepath is None:
        plt.savefig(f"{filepath}/circle_prerotation.png", bbox_inches="tight")
        plt.show()
    plt.show()


def plot_rotated_circle(
    data_real, data_imag, data_real_rot, data_imag_rot, xc, yc, xr, yr
):

    # Plot rotated circle
    plt.plot(data_real_rot[0::2], data_imag_rot[0::2], ".")

    # Mark readout point
    plt.plot(xr, yr, "*", markersize=10, color="black")

    # Plot rotated tangent
    plt.axhline(y=yr, color="green")

    x_diff = np.max(data_real) - np.min(data_real)
    y_diff = np.max(data_imag) - np.min(data_imag)
    diff = max(x_diff, y_diff) * 1.2
    plt.xlim([xc - diff / 2, xc + diff / 2])
    plt.ylim([yc - diff / 2, yc + diff / 2])

    plt.gca().set_aspect("equal")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid()
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")


def calculate_resonator_angle(
    data, freqs, resonator, plot=False, verbose=False, filepath=None
):
    """
    Calculate the tangent angle of a readout tone.

    Parameters:
    data: complex samples of a resonator sweep.
    freqs: Array containing the frequency values of the resonator sweep.
    resonator: actual frequency the readout tone is configured.
    angle_unit: ["rad", "deg"] specifies the unit of the return value
    plot: Enable plotting
    verbose: Enable additional messages

    Returns:
    tuple: The slope and y-intercept of the tangent line.
    """

    # Extract real and imaginary part of complex data
    data_real = [sample.real for sample in data]
    data_imag = [sample.imag for sample in data]

    # Fit circle
    xc, yc, R = fit_circle(data_real, data_imag)
    if verbose:
        print(f"Fitted Circle -> Center: {xc}, {yc}, Radius: {R}")

    # Find index of the actual resonator
    res_index = np.argmin(np.abs(freqs - resonator))
    if verbose:
        print(f"Resonator index: {res_index}")

    # Calculate tangent at actual resonator point
    xr = data_real[res_index]
    yr = data_imag[res_index]
    slope, intercept = calculate_tangent(xc, yc, xr, yr)
    if verbose:
        print(f"Calculated tangent -> Slope: {slope}, intercept_point: {intercept}")

    angle = np.arctan(slope)

    if plot:
        plot_result(data_real, data_imag, xc, yc, xr, yr, R, filepath)

        new_real, new_imag = rotate_circle(data_real, data_imag, -angle)
        xc, yc, R = fit_circle(new_real, new_imag)
        xr = new_real[res_index]
        yr = new_imag[res_index]
        plot_rotated_circle(data_real, data_imag, new_real, new_imag, xc, yc, xr, yr)

        if not filepath is None:
            plt.savefig(f"{filepath}/circle_postrotation.png", bbox_inches="tight")
        plt.show()

    return angle
