import numpy as np
from diffusion_equation.fit import GEOMETRY, fun_residual, preprocess
from scipy.optimize import least_squares

# Define a parameter dictionary (can be populated from GUI) THESE ARE DEFAULTS
default_non_changing_residual_params = {
    'n1': 1.0,        # Refractive index 1
    'n2': 1.41,       # Refractive index 2
    'rho': 0.0,       # Source-detector separation (mm)
    's': 18e3,        # Slab thickness (microns)

    #TODO: dynamically calculate fit_start and fit_end based on the length of the time array, 10-100 is for 128 bins
    'fit_start': 10,  # Start bin for fitting
    'fit_end': 100,   # End bin for fitting

    'phantom': 'slab',# Geometry type
    'mua_independent': True,  # Diffusion coefficient flag
    'm': 400,         # Number of imaginary sources
    'geometry': GEOMETRY.REFLECTANCE,  # Measurement geometry
}


def fit_mua_musp_live(irf, measured, time_arr):

    pre_meas, pre_irf = preprocess(
        measured, irf,
        meas_noise_win=(0, 10),  # adjust as needed
        irf_noise_win=(0, 10),
        meas_roi=(0, len(measured)),
        irf_roi=(0, len(irf)),
        meas_avg_w=3,
        irf_avg_w=3
    )

    x0 = np.array([0.1e-4, 5e-4])  # initial guess
    # Fitting call
    fit = least_squares(
        fun_residual_fixed_params,  # Our parameterized function
        x0,               # Initial guess [mua, musp] 1st param of the fun_residual_fixed_params function
        method='lm',
        args=(time_arr, pre_irf, pre_meas, default_non_changing_residual_params),  # these are the other parameters following the x0 parameter in the fun_residual_fixed_params function
        verbose=1
    )

    return fit.x  # fitted [mua, musp]

def fun_residual_fixed_params(x, time, irf, measured, params):
    """Residual function using parameter dictionary"""
    return fun_residual(
        x, 
        time, 
        irf, 
        measured,
        n1=params['n1'],
        n2=params['n2'],
        rho=params['rho'],
        s=params['s'],
        fit_start=params['fit_start'],
        fit_end=params['fit_end'],
        phantom=params['phantom'],
        mua_independent=params['mua_independent'],
        m=params['m'],
        geometry=params['geometry']
    )

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage
    # Load IRF data
    with open('example csv/test_irf.csv', 'r') as f_irf:
        irf = np.array([float(val) for val in f_irf.read().strip().split(",")])

    # Load measured data
    with open('example csv/test_meas.csv', 'r') as f_meas:
        measured = np.array([float(val) for val in f_meas.read().strip().split(",")])
    # time bin 0 should not be used because it can't divide by zero
    
    time_step = 0.19  # Example time step in nanoseconds
    num_bins = 128
    time_arr = [1e-9 if i == 0 else time_step * i for i in range(num_bins)] #dont use 0 for time bin 0


    fitted_params = fit_mua_musp_live(irf, measured, time_arr)

    # Plotting the results for debugging
    pre_meas, pre_irf = preprocess(
        measured, irf,
        meas_noise_win=(0, 10),  # adjust as needed
        irf_noise_win=(0, 10),
        meas_roi=(0, len(measured)),
        irf_roi=(0, len(irf)),
        meas_avg_w=3,
        irf_avg_w=3
    )

    # plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    # plt.plot(time_arr, irf, label='IRF')
    # plt.plot(time_arr, measured, label='Measured')
    # plt.legend()
    # plt.title("Raw Data")

    
    # plt.subplot(122)
    # # plt.plot(time_arr, pre_irf, label='Processed IRF')
    # # plt.plot(time_arr, pre_meas, label='Processed Measured')
    # plt.plot(time_arr[0:30], pre_irf[0:30], label='Processed IRF (30-128)', marker='o', linestyle='-')
    # plt.plot(time_arr[0:30], pre_meas[0:30], label='Processed Measured (30-128)', marker='o', linestyle='-')
    # plt.legend()
    # plt.title("Processed Data")
    # plt.show()



    # testing multiple initial guesses
    initial_guesses = [
        [0.05e-4, 1e-4],   # Low scattering
        [0.1e-4, 5e-4],     # Original
        [0.2e-4, 10e-4],    # Higher scattering
        [0.5e-4, 20e-4]     # Very high scattering
    ]

    for i, x0 in enumerate(initial_guesses):
        print(f"\nTesting initial guess {i+1}: {x0}")
        fit = least_squares(
            fun_residual_fixed_params,
            x0,
            method='trf',
            bounds=([1e-6, 1e-5], [0.1, 10.0]),
            args=(time_arr, pre_irf, pre_meas, default_non_changing_residual_params),
            verbose=1
        )
        print(f"Result: {fit.x}")
        print("Debug:", fit.success, fit.message)