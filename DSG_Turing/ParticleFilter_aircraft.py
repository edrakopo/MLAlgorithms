#modified code baed on: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
from numpy.random import uniform
import numpy as np
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
from matplotlib import pyplot as plt
from random import random

def create_uniform_particles(x_range, y_range, z_range, hdg_range, N):
    particles = np.empty((N, 4))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(z_range[0], z_range[1], size=N)
    particles[:, 3] = uniform(hdg_range[0], hdg_range[1], size=N) #heading
    particles[:, 3] %= 2 * np.pi
    return particles

#def create_gaussian_particles(mean, std, N):
#    particles = np.empty((N, 3))
#    particles[:, 0] = mean[0] + (randn(N) * std[0])
#    particles[:, 1] = mean[1] + (randn(N) * std[1])
#    particles[:, 2] = mean[2] + (randn(N) * std[2])
#    particles[:, 2] %= 2 * np.pi
#    return particles

def predict(particles, u, std, dt=1.): #in our case we keep 0 elem, and add 123 for xyz
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""

    N = len(particles)
    # update heading
    particles[:, 3] += u[0] + (randn(N) * std[0])
    particles[:, 3] %= 2 * np.pi

    # move in the (noisy) commanded direction
    #dist = (u[1] * dt) + (randn(N) * std[1])
    #particles[:, 0] += np.cos(particles[:, 2]) * dist
    #particles[:, 1] += np.sin(particles[:, 2]) * dist
    distx = (u[1] * dt) + (randn(N) * std[1])
    disty = (u[2] * dt) + (randn(N) * std[2])
    distz = (u[3] * dt) + (randn(N) * std[3])
    particles[:, 0] += np.cos(particles[:, 2]) * distx     
    particles[:, 1] += np.sin(particles[:, 2]) * disty
    particles[:, 2] += distz

def update(particles, weights, z, R, landmarks): #z is the distance from robot to each landmark->
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:3] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize

def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:3]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)

def neff(weights):
    return 1. / np.sum(np.square(weights))

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))

def systematic_resample(weights):
    """ Performs the systemic resampling algorithm used by particle filters.
    This algorithm separates the sample space into N divisions. A single random
    offset is used to to choose where to sample from for all divisions. This
    guarantees that every sample is exactly 1/N apart.
    Parameters
    ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (random() + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


#from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats

def run_pf1(N, iters=18, sensor_std_err=.1, 
            do_plot=True, plot_particles=True,
            xlim=(0, 20), ylim=(0, 20), zlim=(0,20),
            initial_x=None):
    landmarks = np.array([[-1, 2, 1], [5, 10, 10], [12,14, 14], [18,21,21]])
    NL = len(landmarks)
    
    plt.figure()
   
    # create particles and weights
    if initial_x is not None:
        particles = create_gaussian_particles(
            mean=initial_x, std=(5, 5, np.pi/4), N=N)
    else:
        particles = create_uniform_particles((0,20), (0,20), (0,20), (0, 6.28), N)
    weights = np.ones(N) / N

    if plot_particles:
        alpha = .20
        if N > 5000:
            alpha *= np.sqrt(5000)/np.sqrt(N)           
        plt.scatter(particles[:, 0], particles[:, 1], 
                    alpha=alpha, color='g')
    
    xs = []
    robot_pos = np.array([0., 0., 0.])
    for x in range(iters):
        robot_pos += (1, 1, 1)

        # distance from robot to each landmark
        zs = (norm(landmarks - robot_pos, axis=1) + 
              (randn(NL) * sensor_std_err))

        # move diagonally forward to (x+1, x+1)
        #predict(particles, u=(0.00, 1.414), std=(.2, .05))
        predict(particles, u=(0.00, 1.414, 1.414, 1.414), std=(.2, .05, .05, .05))        

        # incorporate measurements
        update(particles, weights, z=zs, R=sensor_std_err, 
               landmarks=landmarks)
        
        # resample if too few effective particles
        if neff(weights) < N/2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
            assert np.allclose(weights, 1/N)
        mu, var = estimate(particles, weights)
        xs.append(mu)

        if plot_particles:
            plt.scatter(particles[:, 0], particles[:, 1], 
                        color='k', marker=',', s=1)
        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+',
                         color='k', s=180, lw=3)
        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')
    
    xs = np.array(xs)
    #plt.plot(xs[:, 0], xs[:, 1])
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    print('final position error, variance:\n\t', mu - np.array([iters, iters, iters]), var)
    plt.show()

from numpy.random import seed
seed(2) 
run_pf1(N=5000, plot_particles=False)
