
DECIMAL = 2

m = 2  # numbe of nodes to connect

parameters = {
    'sigma_1_I': 1.5,  # given it is inactive it becomes active after exponential time with rate sigma_1_i
    'sigma_2_I': 2.5,  # given it is active it becomes inactive after exponential time with rate sigma_2_i
    'sigma_1_H': 3.5,  # given it is inactive it becomes active after exponential time with rate sigma_1_i
    'sigma_2_H': 4.5,  # given it is active it becomes inactive after exponential time with rate sigma_2_i
    'beta_I': 3.0,  # transmit I->S (susceptible becomes carrier)
    'beta_C': 5.7,  # transmit C->S  (susceptible becomes carrier)
    'gamma': 5.9,  # cure rate I-> R
    'nu_I': 0.75,  # C->I
    'nu_R': 7.9  # C->R
}