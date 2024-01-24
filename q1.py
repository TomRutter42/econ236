# Tomer Fidelman, Tom Rutter, Tomas Tapak

# We used GitHub Copilot to help with this assignment. 



import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.formula.api as smf
import numpy as np

## 
from warnings import simplefilter 
simplefilter(action="ignore", category=pd.errors.PerformanceWarning) # F
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# set seed 
np.random.seed(123)

# Read in the data 
## Data is from Heathcote et al 2023
## Sample B seems more relevant for us, so use that
## read in the PSID_sampleB.dta file from the same folder as this script
## surely there is a better way to do paths in python

data = pd.read_stata("C:/Users/tomru/Dropbox (Personal)/GitHub/econ236/PSID_sampleB.dta")

## They focus on log wages 

## print how many zeroes there are in the data

print("Number of zeroes in the data: ", np.sum(data["hwage"] == 0)) # 0
## Assert the number of zeros in hwage is zero, breaking the code if not
assert np.sum(data["hwage"] == 0) == 0  

# log
data["lhwage"] = np.log(data["hwage"])

## for predicting log income 

## dummy for at least 16 years of schooling 

data["hedu"] = np.where(data["hyrsed"] >= 16, 1, 0)

## couldn't find occupation data 

## or gender data 

## race is hrace 
## generate dummy for each hrace, by looping over the values of hrace 

## drop if race is nan 
## count how many race nans 

print("Number of NaNs in hrace: ", np.sum(data["hrace"].isnull())) # 177

data = data.dropna(subset = ["hrace"])

data["hrace1"] = np.where(data["hrace"] == 1, 1, 0)
data["hrace2"] = np.where(data["hrace"] == 2, 1, 0)
data["hrace3"] = np.where(data["hrace"] == 3, 1, 0)

## print the names of all the new variables starting with hrace 
    
# print(data.columns[data.columns.str.startswith('hrace')])

## Index(['hrace', 'hrace1.0', 'hrace2.0', 'hrace3.0'], dtype='object')

## Project log wages onto our dummies 

## age is continuous hage

## generate squared age 

data["hage2"] = data["hage"]**2

## address the warning 
data = data.copy()

## run a regression of log wages on the regressors 
## data time is at the year level, split by year

## New dataset to store the residuals, with three variables: year, id68, resid

residuals_data = pd.DataFrame(columns = ["year", "id68", "hage", "hpersno", "resid"])

for year in data.year.unique():
    ## subset the data to the year
    data_year = data[data.year == year]
    ## run the regression. You could add more variables if you want, but hopefully this is enough to make the method clear
    reg = smf.ols("lhwage ~ hage + hage2 + hedu + hrace1 + hrace2 + hrace3", data = data_year).fit()
    ## print the results 
    # print(reg.summary())
    ## Store the residuals in residuals_data
    ## concat axis = 0 means you are pasting together the columns, i.e., you get a row reading across for each entry
    data_to_append = pd.concat([data_year[["year", "id68", "hage", "hpersno"]], reg.resid], axis = 1)

    ## axis = 1 means you are pasting together the columns, i.e., you are stacking entries on top of each other. 
    residuals_data = pd.concat([residuals_data, data_to_append], axis = 0)


## 6-8 is done. 
    
## Simulate 
    
N = 5000
T = 15
rho = 0.95 
sigma_a = 0.1
sigma_v = 0.2
sigma_n = 0.15


## Set up vectors 

epsilon = np.zeros((N, T))
y = np.zeros((N, T))
person = np.zeros((N, T))
time = np.zeros((N, T))

## Draw shocks

alpha = np.random.normal(0, np.sqrt(sigma_a), N)
v = np.random.normal(0, np.sqrt(sigma_v), (N, T))
eta = np.random.normal(0, np.sqrt(sigma_n), (N, T))

for i in range(N):
    person[i, :] = i
    for t in range(T):
        time[i, t] = t
        ## initial period is special
        if t == 0:
            epsilon[i, t] = eta[i, t]
            y[i, t] = alpha[i] + epsilon[i, t] + v[i, t]
        else:
            epsilon[i, t] = rho * epsilon[i, t - 1] + eta[i, t]
            y[i, t] = alpha[i] + epsilon[i, t] + v[i, t]


## Create a dataframe with columns y, person, time
            
sim_data = pd.DataFrame({"y": y.flatten(), "person": person.flatten(), "time": time.flatten()})

## Print the first 10 rows of the data as a check

print(sim_data.head(1000))

## Plot the simulations for the first 10 people 

import matplotlib.pyplot as plt

for i in range(10):
    plt.plot(sim_data[sim_data.person == i].time, sim_data[sim_data.person == i].y)
# plt.show()

# Save the plot to the folder outlined above 

plt.savefig("C:/Users/tomru/Dropbox (Personal)/GitHub/econ236/incomesims.png")

# Drop 5% of the observations at random from sim_data 

# sim_data = sim_data.sample(frac = 0.95)

## Do everything above to get sim_data but in a loop 20 times 

for run in range(20):

    run_number = run
    ## Set up vectors 

    epsilon = np.zeros((N, T))
    y = np.zeros((N, T))
    person = np.zeros((N, T))
    time = np.zeros((N, T))

    ## Draw shocks

    alpha = np.random.normal(0, np.sqrt(sigma_a), N)
    v = np.random.normal(0, np.sqrt(sigma_v), (N, T))
    eta = np.random.normal(0, np.sqrt(sigma_n), (N, T))

    for i in range(N):
        person[i, :] = i
        for t in range(T):
            time[i, t] = t
            ## initial period is special
            if t == 0:
                epsilon[i, t] = eta[i, t]
                y[i, t] = alpha[i] + epsilon[i, t] + v[i, t]
            else:
                epsilon[i, t] = rho * epsilon[i, t - 1] + eta[i, t]
                y[i, t] = alpha[i] + epsilon[i, t] + v[i, t]

    sim_data = pd.DataFrame({"y": y.flatten(), "person": person.flatten(), "time": time.flatten(), "run": run_number})
    # sim_data = sim_data.sample(frac = 0.95, replace = False)
    if run == 0:
        sim_data_all = sim_data
    else:
        sim_data_all = pd.concat([sim_data_all, sim_data], axis = 0)


print(sim_data_all.head(10))

## Theory 

# List all the combinations between 0 and T-1 

## import itertools
import itertools

time_pairs = list(itertools.combinations(np.arange(0, T), 2))  + [(a,a) for a in np.arange(0, T)]

# Print the list of time pairs 

print(time_pairs)

# Initialize the covariance matrix, which will be T x T

cov_mat = np.empty((T, T))

for tp in time_pairs:
    
    # get the distance between the two time periods
    dist = np.abs(tp[0] - tp[1])

    if dist == 0:
        cov = sigma_a + sigma_v + sigma_n * sum(rho**(2 * (t - s)) for s in range(1, tp[0] + 1)) # don't forget eta shock is not relaized in period 0
    else:
        cov = sigma_a + rho**dist * sigma_n * sum(rho**(2 * (t - s)) for s in range(1, tp[0] + 1))
    # total = sum(v.amount for ob in self.oblist for v in ob.anotherob)
    # put cov in the right place in the covariance matrix
    cov_mat[tp[0], tp[1]] = cov
    cov_mat[tp[1], tp[0]] = cov
    ## symmetry 
    # cov_mat[tp[1], tp[0]] = cov


# That's (1) done
    
# (3): Calculate the covariance matrix of the sim_data.

## For each person-run combination, calculate the individual covariance matrix

## Initialize the covariance matrix, which will be T x T
    
cov_mat_sim = np.empty((T, T))

print("NaN values in sim_data['y']:", np.isnan(sim_data['y']).any())

for tp in time_pairs:

    # Produce a vector containing the product of y at tp[0] and y at tp[1] for every person 
    # so the first entry is y at tp[0] for person 0 times y at tp[1] for person 0
    # the second entry is y at tp[0] for person 1 times y at tp[1] for person 1
    # etc
    y_prod = np.multiply(sim_data.loc[sim_data['time'] == tp[0], 'y'].values, sim_data.loc[sim_data['time'] == tp[1], 'y'].values)
    # print(y_prod)
    # take the mean of y_prod
    y_prod_mean = np.mean(y_prod)

    # put cov in the right place in the covariance matrix
    # cov_mat_sim[tp[0], tp[1]] = y_prod_mean
    cov_mat_sim[tp[1], tp[0]] = y_prod_mean
    




    
# Check that cov_mat_sim is the same as cov_mat 
    
print(cov_mat_sim - cov_mat)
print(cov_mat)
print(cov_mat_sim)
# looks good 


# ==============================================================================

# GMM

## Define Helper Functions

### Take your parameter vector, the empirical covariance matrix (to size the theoretical matrix), and the time pairs

def calculate_m_theoretical(theta, m_empirical, age_combinations):
    # Create m theoretical
    m_theoretical = np.empty_like(m_empirical)
    m_theoretical[:] = np.nan

    # Population Theoretical matrix
    for x in age_combinations:
        j = np.min(x)
        h = np.max(x) - j # distance between ages

        # Calculate E(e_ij ^2)
        if j > 0:
            var_eps_ij = (theta[0]**(2*j)) * 0 + np.sum([theta[0]**(2*(j - k)) * theta[2] for k in np.arange(1, j+1) ])
        else:
            var_eps_ij = 0

        # Covariance
        if x[0] != x[1]:    
            covar = theta[3] + (theta[0]**h) *var_eps_ij
            m_theoretical[x[0], x[1]] =  m_theoretical[x[1], x[0]] = covar
        # Now the variance
        else:
            m_theoretical[j, j] = theta[3] + theta[1]  + var_eps_ij

    return m_theoretical

# Take your parameter vector, the empirical covariance matrix, the time pairs
# returns the GMM objective, which is the sum of squared differences between the empirical and theoretical covariance matrices
# and we'll try to minimize

def gmm(theta, m_empirical, ages):

    # All combinations + doubles
    age_combinations = list(itertools.combinations(ages, 2))  + [(a,a) for a in ages]

    # Calculate theoretical moments
    m_theoretical = calculate_m_theoretical(theta, m_empirical, age_combinations)

    # Extract moments for GMM
    m_theoretical_vec = np.column_stack(m_theoretical[np.tril_indices(m_theoretical.shape[0])]).transpose()
    m_empirical_vec =  np.column_stack(m_empirical[np.tril_indices(m_empirical.shape[0])]).transpose()

    # Make weighting matrix:

    if type(weight) == str  and weight == "identity":
        weight = np.identity(len(m_theoretical_vec))
    elif type(weight) is np.ndarray:
        weight = weight # matrix given
    else: 
        weight = np.identity(len(m_theoretical_vec))

    # Take difference between moments
    mt_me = m_theoretical_vec - m_empirical_vec
    objective = mt_me.transpose() @ mt_me

    return objective

# # Remove NA
# residuals_data = resid_income_df.query("resid.notnull()", engine = 'python')
# Add effective age
residuals_data["effective_age"] = residuals_data["hage"] - 21
residuals_data["effective_age_norm"]= residuals_data["effective_age"] - np.min(residuals_data["effective_age"]) # normalize min age to 0

ages = np.sort(residuals_data.effective_age_norm.unique().astype(int))

ages_min = np.min(ages)
ages_max = np.max(ages)

# Initialize
rho = 0.9
sigma_sq_alpha = 0.05
sigma_sq_eta = 0.2
sigma_sq_ups = 0.1
theta0 = np.array([rho, sigma_sq_ups, sigma_sq_eta, sigma_sq_alpha])

# Empty covariance matrix
m_empirical = np.empty( [ages_max - ages_min + 1, ages_max - ages_min + 1] )
m_empirical[:] = np.nan


# All combinations + doubles
age_combinations = list(itertools.combinations(ages, 2))  + [(a,a) for a in ages]

for x in age_combinations:
    # Covariance
    if x[0] != x[1]:             
        temp = residuals_data.query("effective_age_norm == @x[0] | effective_age_norm == @x[1]")

        # Covariance calculation
        temp_covar = temp.groupby(["id68", "hpersno" ]).filter(lambda x: x["hage"].count() == 2).groupby(["id68", "hpersno" ]).agg( cov = ("resid",np.prod) ).mean()
        # Populate matrix (both entries)
        m_empirical[x[0], x[1]] = temp_covar
        m_empirical[x[1], x[0]] = temp_covar
    else:
        temp = residuals_data.query("effective_age_norm == @x[0]")
        temp_var = temp.groupby(["id68", "hpersno" ]).agg( var = ("resid",  lambda x: x**2 ) ).mean()
        m_empirical[x[0], x[0]] = temp_var

print(m_empirical)

# Run GMM
gmm_estimate = minimize(gmm, theta0,  method='nelder-mead', args=(m_empirical, ages))

# Extract esimates and print
params = gmm_estimate.x

param_names = ["rho", "sigma_sq_ups", "sigma_sq_eta", "sigma_sq_alpha"]
output_text = "GMM Estimates:\n"
for i,j in enumerate(param_names):
    output_text +=  j +": " + str(np.round(params[i], 4) ) +  "\n"

print(output_text)