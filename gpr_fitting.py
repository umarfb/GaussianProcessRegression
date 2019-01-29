import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor
import os

# Method to get lightcurve data
def get_lightcurve(filename):
    lc_data = ascii.read(filename)

    filter_id = lc_data['fid'][0]
    epoch = lc_data['obsmjd']
    mag = lc_data['mag_autocorr']
    mag_err = lc_data['magerr_auto']
    sn_type = lc_data['type_01'][0]
    epoch = np.asarray(epoch)
    mag = np.asarray(mag)
    mag_err = np.asarray(mag_err)

    if filter_id == 1:
        mag_filter = 'g'
    elif filter_id == 2:
        mag_filter = 'R'
    
    return epoch, mag, mag_err, sn_type, mag_filter

# Method to plot lightcurve
def plot_lightcurve(x, y, yerr, sn_type, y_label):
    plt.figure(figsize=(8,5.5))
    plt.errorbar(x, y, yerr = yerr, lw=1, capsize=1, color='#000099', linestyle='None', marker='.')
    plt.gca().invert_yaxis()
    plt.xlabel('MJD')
    plt.title('SN2011dh: type ' + str(sn_type))
    plt.ylabel(y_label)

# Def. method to iteratively run GPR process
# Arguments:
# - x_in - independent variable
# - y_in - dependent variable
# - kernel - kernel function to use for GPR
#
# Returns:
# hyper_vector - array containing log likelihood and hyperparameters

def do_GPR(x_in, y_in):
    # Define range of input space to predict over
    x_min = x_in.min() - 20
    x_max = x_in.max() + 20
    
    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    x_space = np.atleast_2d(np.linspace(x_min, x_max, 100)).T
    x_fit = np.atleast_2d(x_in).T
    
    k_RBF = RBF(length_scale=1e2, length_scale_bounds=(1e-2, 1e5))
    k_exp = (Matern(length_scale=1e2, length_scale_bounds=(1e-2, 1e6), nu=0.5))
    k_sine = ExpSineSquared(length_scale=1e2, length_scale_bounds=(1e-2, 1e5), periodicity=1e4, periodicity_bounds=(1e3, 5e5))
    k_noise = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e5))

    # Matern kernel with nu = 0.5 is equivalent to the exponential kernel
    # Define kernel function
    kernel = 1.0 * k_RBF + 1.0*(k_exp*k_sine) + k_noise
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10, normalize_y=True)
    
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gpr.fit(x_fit, y_in)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, y_pred_sigma = gpr.predict(x_space, return_std=True)
    
    # Get log likelihood and hyperparameters
    log_likelihood = gpr.log_marginal_likelihood()
    hyper_params = gpr.kernel_
    
    hyper_vector = []
    hyper_vector.append(log_likelihood)
    params = hyper_params.get_params()
    for i, key in enumerate(sorted(params)):
        if i in (3,6,10,14,18,20,23):
            hyper_vector.append(params[key])
    
    return x_space, y_pred, y_pred_sigma, hyper_vector

# Method to write .csv file containing GPR fitted curve
def tabulate_GPR(path, x_space, y_pred, y_pred_sigma, sn_name):
    # Flatten arrays to create dataframe
    x_space = x_space.flatten()
    y_pred = y_pred.flatten()
    y_pred_sigma = y_pred_sigma.flatten()

    # Create pandas dataframe for GPR fit values
    gp_data = {'MJD' : x_space, 'mag_pred' : y_pred, 'mag_pred_sigma' : y_pred_sigma}
    gp_df = pd.DataFrame(data=gp_data)
    filename = sn_name + '_gpr_fit.csv'
    gp_df.to_csv(path + filename, header=True, index=False)

# Method to write GPR fit parameters, SN type and filter used to .csv file
def tabulate_params(path, theta, sn_name, sn_type, mag_filter):
    filename = sn_name + '_fit_params.csv'
    theta_labels = ['log_likelihood', 'w1', 'l1', 'w2', 'l2', 'l3', 'P', 'sigma_noise', 'sn_type', 'filter']
    theta.append(sn_type)
    theta.append(mag_filter)
    param_data = {'parameter' : theta_labels, 'value' : theta}
    param_df = pd.DataFrame(data=param_data)
    param_df.to_csv(path + filename, header=True, index=False)

# Method to plot GPR fit
def plot_GPR(x_in, y_in, y_err, x_space, y_pred, y_pred_sigma, mag_label, sn_name, theta):
    plt.figure(figsize=(8, 5.5))
    plt.title('{0}, type {1}, log likelihood={2}'.format(sn_name, theta[-2], theta[0]))
    plt.errorbar(x_in, y_in, yerr = y_err, lw=1, capsize=1, color='#b30000', linestyle='None', marker='.', label='Data')
    plt.plot(x_space, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x_space, x_space[::-1]]),
             np.concatenate([y_pred - 1.9600 * y_pred_sigma,
                            (y_pred + 1.9600 * y_pred_sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('MJD')
    plt.ylabel('$m_{0}$'.format(mag_label))
    plt.gca().invert_yaxis()
    plt.legend(loc='lower left')
    plt.savefig('{0}_gprfit.pdf'.format(sn_name), format='pdf', bbox_inches='tight')
    #plt.show()

# path to PTF lightcurves
lc_path = '/local/php18ufb/backed_up_on_astro3/PTF_classification/lightcurve_GPR/PTF_lightcurves_db/'

# path for GPR fits
gpr_path = '/local/php18ufb/backed_up_on_astro3/PTF_classification/lightcurve_GPR/GPR_fits/'

# path for GPR plots
gpr_plots_path = '/local/php18ufb/backed_up_on_astro3/PTF_classification/lightcurve_GPR/GPR_plots/'

# Get list of SN lightcurves
lc_list = os.listdir(lc_path)

for sn in lc_list:
    sn_name = sn.strip('.tbl')
    print('GPR fitting: {0}'.format(sn_name))

    # Get lightcurve parameters
    epoch, mag, mag_err, sn_type, mag_filter = get_lightcurve(lc_path + sn)

    # Do GPR for lightcurve
    x_space, mag_pred, mag_pred_sigma, theta = do_GPR(epoch, mag)

    # Write GPR fit curve and parameters to file
    tabulate_GPR(gpr_path, x_space, mag_pred, mag_pred_sigma, sn_name)
    tabulate_params(gpr_path, theta, sn_name, sn_type, mag_filter)

    # Save GPR fit plot
    plot_GPR(epoch, mag, mag_err, x_space, mag_pred, mag_pred_sigma, mag_filter, sn_name, theta)