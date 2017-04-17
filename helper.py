
# coding: utf-8

# In[10]:

'''
# Planar curves generation - Cube Method

Divide a unit cube into 4 adjecent sub-cubes, pick a point randomly 
from each sub-unit within a pre-specified planar volume. 

The area within each sub-unit can be chosen based on
the desired expanse of the curve. 

Join the chosen point by spline interpolation.
'''
# Helper functions for Spline Processing 
from __future__ import division, print_function, absolute_import

import sys
import time
import math
import scipy as sp
from scipy.interpolate import splprep, splev
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import random as random
import sklearn.datasets, sklearn.decomposition
import numpy.linalg as linalg
get_ipython().magic('matplotlib notebook')
    
# Helper functions for Spline Processing 
def get_3d_points(X,Y,Z):
    pts = np.concatenate((X,Y,Z), axis=0)
    pts = pts.reshape(3,len(X))
    return pts
def add_curve_to_array(x, y, z):
    inputs = np.concatenate((x,y,z), axis=0)
    len(inputs)
    inputs = inputs.reshape(3,300)
    return inputs
# Spline Generation
def spline_generate(pts):
    #pts = np.unique(pts)
    tck, u = splprep(pts, u=None, s=0.0) 
    u_new = np.linspace(u.min(), u.max(), 300)
    x_new, y_new, z_new = splev(u_new, tck, der=0)
    return x_new, y_new, z_new
def get_rot_angle(theta=0):
    if(theta == 0): theta = np.random.uniform(0,1)*2*np.pi
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return cos_t, sin_t
def random_rotate(x,y,z,a=0,b=0,g=0):
    cos_t, sin_t = get_rot_angle(a)
    r_x = np.matrix([[1, 0, 0], [0, cos_t, -sin_t], [0, sin_t, cos_t]])
    
    cos_t, sin_t = get_rot_angle(b)
    r_y = np.matrix([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t,0, cos_t]])
    
    cos_t, sin_t = get_rot_angle(g)
    r_z = np.matrix([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
    
    r = np.dot((np.dot(r_x, r_y)), r_z)
    
    rot_v = np.dot(r,np.matrix([[x],[y],[z]]))
    
    return rot_v.item(0),rot_v.item(1),rot_v.item(2)
       
def draw_cube(b):
    for s, e in combinations(np.array(b), 2):
        if np.sum(np.abs(s-e)) == 1:
            ax.plot3D(*zip(s, e), color="r")  
def create_show_p_curve():
    boxes = [
            [(0.5, 0.9), (0.9, 0.9), (0.5, 0.9)],    #[(0.1, 0.5), (0.9, 0.9), (0.1, 0.5)],    
             [(-0.9, -0.5), (0.9, 0.9), (0.5, 0.9)], #[(-0.5, -0.1), (0.9, 0.9), (0.1, 0.5)],  
            [(0.5, 0.9), (0.9, 0.9), (-0.9, -0.5)],  #[(0.1, 0.5), (0.9, 0.9), (-0.5, -0.1)],   
            [(-0.9, -0.5), (0.9, 0.9), (-0.9, -0.5)] #[(-0.5, -0.1), (0.9, 0.9), (-0.5, -0.1)], 
        ]
    X_raw=[]
    Y_raw=[]
    Z_raw=[]

    N=1

    data = {}
    data['planar_curves'] = []

    startTime = time.time()

    for i in range(N):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')   
        pts=[]
        X_raw=[]
        Y_raw=[]
        Z_raw=[]

        #for all points in this curve
        x_theta = np.random.uniform(0,1)*2*np.pi
        y_theta = np.random.uniform(0,1)*2*np.pi
        z_theta = np.random.uniform(0,1)*2*np.pi

        for b in boxes:
            x = random.uniform(b[0][0]/1, b[0][1]/1)
            y = random.uniform(b[1][0]/1, b[1][1]/1)
            z = random.uniform(b[2][0]/1, b[2][1]/1)
            x,y,z = random_rotate(x,y,z, x_theta, y_theta, z_theta)
            X_raw.append(x)
            Y_raw.append(y)
            Z_raw.append(z)    

        # draw cube
        r = [-1, 1]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s, e), color="b")

        pts = get_3d_points(X_raw,Y_raw,Z_raw)
        ax.plot(X_raw, Y_raw, Z_raw, 'ro')
        X, Y, Z = spline_generate(pts)

        curve = add_curve_to_array(X, Y, Z)

        ax.plot(X, Y, Z, 'b--')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.show()    
        data['planar_curves'].append(curve.tolist())
#create_show_p_curve()


# In[12]:

'''
# Non-planar curves generation - Cube Method

Divide a unit cube into 8 sub-cubes, pick a point randomly 
from each sub-unit. Join the chosen point by spline
interpolation.

The area within each sub-unit can be chosen based on
the desired expanse of the curve. 
'''

# Create Test NP Curve
def create_plot_new_np_curve():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # draw cube
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="b")

    #plt.show()        

    boxes = [[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], 
     [(-1.0, 0.0), (0.0, 1.0), (0.0, 1.0)], 
     [(0.0, 1.0), (-1.0, 0.0), (0.0, 1.0)], 
     [(0.0, 1.0), (0.0, 1.0), (-1.0, 0.0)], 
     [(-1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0)], 
     [(0.0, 1.0), (-1.0, 0.0), (-1.0, 0.0)], 
     [(-1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)], 
     [(-1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]]

    import random as random

    X_raw=[]
    Y_raw=[]
    Z_raw=[]
    N=1



    startTime = time.time()

    for i in range(N):
        X_raw=[]
        Y_raw=[]
        Z_raw=[]
        for b in boxes:
            x = random.uniform(b[0][0]/1, b[0][1]/1)
            y = random.uniform(b[1][0]/1, b[1][1]/1)
            z = random.uniform(b[2][0]/1, b[2][1]/1)
            #print(x,y,z)
            X_raw.append(x)
            Y_raw.append(y)
            Z_raw.append(z)
        pts = get_3d_points(X_raw,Y_raw,Z_raw)
        X, Y, Z = spline_generate(pts)
        curve = add_curve_to_array(X, Y, Z)
        ax.plot(X, Y, Z, 'b--')
        ax.plot(X_raw, Y_raw, Z_raw, 'ro')
        plt.show()    
#create_plot_new_np_curve()


# In[38]:

# PCA analysis and plot
with open('np_curve_data_cube_method_1489680586.46.json') as infile:
    c = json.load(infile)
    n_planar_curves_array = np.asarray(c['non_planar_curves'])
    
with open('p_curve_data_cube_method_1489173944.8.json') as infile:
    c = json.load(infile)
    planar_curves_array = np.asarray(c['planar_curves'])
data = {}
data['planar_curves_error'] = []
data['non_planar_curves_error'] = []
import numpy as np
def pca_err(curves_array):
    errors=[]
    im = 0
    for i in range(len(curves_array[:])):  
        X = curves_array[i].T 
        mu = np.mean(X, axis=0)
        #print("X: ", X.shape)
        #print("mu: ", mu)
        pca = sklearn.decomposition.PCA()
        pca.fit(X)

        #ax1.plot(curves_array[i][0], curves_array[i][1], curves_array[i][2], 'ro') 

        nComp = 2
        #print("Transfomed: ", pca.transform(X)[:,:nComp].shape)
        #print("EV: ", pca.components_[:,:][:,:nComp])

        transformed = pca.transform(X)[:,:nComp].T
        if (im < 1):
            fig = plt.figure()
            fig.suptitle('Top Left - Original Curve | Top Right - PCA | Bottom Left - Reconstucted Curve', fontsize=10)
            ax1 = fig.add_subplot(221, projection='3d')   
            ax2 = fig.add_subplot(222, projection='3d') 
            ax3 = fig.add_subplot(223, projection='3d') 
            ax1.plot(curves_array[0][0], curves_array[0][1], curves_array[0][2], 'ro') 
            ax2.plot(transformed[0], transformed[1], 'ro') 

        Xhat = np.dot(pca.transform(X)[:,:nComp], pca.components_[:nComp,:])
        Xhat += mu

        reconstructed_curve = Xhat.T
        if (im < 1):
            ax3.plot(reconstructed_curve[0], reconstructed_curve[1], reconstructed_curve[2], 'ro')
            plt.show()

        
        #print(Xhat.shape)
        
        
        err = 0.5*sum((X-Xhat)**2)
        errors.append(sum(err))
        im = im+1
        
        #print("Err: ", err)
        
    return np.asarray(errors)
def plot_PCA_errors():
    np_pca_err = pca_err(n_planar_curves_array)
    p_pca_err = pca_err(planar_curves_array)

    get_ipython().magic('matplotlib inline')
    bins = np.linspace(0, 50, 50)

    plt.hist(np_pca_err, bins, alpha=0.35, label='NPE')
    plt.hist(p_pca_err, bins, alpha=0.35, label='PE')
    plt.legend(loc='upper right')
    plt.title('Reconstruction Errors Histogram')
    plt.show()
#plot_PCA_errors()   


# In[37]:

# PCA weigths initialized auto-encoder



# In[20]:

#Non-Planar Errors

def ae_with_pca_wt_np_errors():
    with open('planarity_errors_1490807988.67.json') as infile:
        c = json.load(infile)
    np_errors = np.asarray(c['non_planar_curves_error'])
    p_errors = np.asarray(c['planar_curves_error'])
    
    NPE = np.insert(np_errors, 1, 1, axis=2)
    PE = np.insert(p_errors, 1, 0, axis=2)
    X = np.concatenate((NPE, PE), axis=0)
    X = X.reshape(200,2)
    hist, bins = np.histogram(X[0:100,0], bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


# In[21]:

#Planar Errors

def ae_with_pca_wt_p_errors():
    with open('planarity_errors_1490807988.67.json') as infile:
        c = json.load(infile)
    np_errors = np.asarray(c['non_planar_curves_error'])
    p_errors = np.asarray(c['planar_curves_error'])
    
    NPE = np.insert(np_errors, 1, 1, axis=2)
    PE = np.insert(p_errors, 1, 0, axis=2)
    X = np.concatenate((NPE, PE), axis=0)
    X = X.reshape(200,2)
    
    hist, bins = np.histogram(X[100:200,0], bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


# In[42]:

#Autoencoder
_debug_verbose = False 
class AutoEncoder(object):
    def __init__(self, arch):        
        self.num_layers = len(arch)
        self.input_layer_size = arch[0]
        self.output_layer_size = arch[-1]
        self.num_hidden_layers = len(arch)-2
        self.costs = []
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(arch[:-1], arch[1:])]        
        self.biases = [np.random.randn(y, 1) for y in arch[1:]]

    def getParams(self):
        #Get weights and biases unrolled into vector:
        params = [(x.ravel(), y.ravel()) for x, y in zip(self.weights, self.biases)]
        return params
    
    def forward(self, X):
        for b, w in zip(self.biases, self.weights):
            if (_debug_verbose): print("weights: ", w)
            if (_debug_verbose): print("biases: ", b)
            if (_debug_verbose): print("inputs :", X)
            if (_debug_verbose): print("dot product :", np.dot(w, X))
            #print("matrix dot product :", w.dot(X))
            X = self.unit_step(np.dot(w, X) + b)
            if (_debug_verbose): print("result :", X)
        return X.reshape(3,1)
    
    def unit_step(self, z):
        #return (lambda x: 0 if (x) < 0 else 1, z)[1]
        return z
    
    def unit_step_prime(self, z):
        return (1)

    def cost_function(self, X):
        self.yHat = self.forward(X)
        if (_debug_verbose): print ("squared error of X:{0} - Xhat:{1} is {2} & sum is {3}\n".format(X, self.yHat, ((X-self.yHat)**2), sum((X-self.yHat)**2)))
        J = 0.5*sum((X-self.yHat)**2)
        #self.costs.append(J) 
        return J
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)   
    
    def cost_function_by_epoch(self, test_data, n_test):
        y_hat = [(self.forward(y)) for (y) in test_data[0:n_test]]
        y = [(y) for (y) in test_data[0:n_test]]
        #print([float(a[0][0]) for a in y])
        np.seterr( over='ignore' )
        #costs = []
        costs = [0.5*((a - b)**2) for a, b in zip(y, y_hat)]
        #costs.append([max(math.sqrt(0.5*(round(a[0][0],2) - round(b[0][0],2))**2),1000) for a, b in zip(y, y_hat)])
        #costs.append([0.5*math.sqrt((float(a[1][0]) - float(b[1][0]))**2) for a, b in zip(y, y_hat)])
        #costs.append([0.5*math.sqrt((float(a[2][0]) - float(b[2][0]))**2) for a, b in zip(y, y_hat)])
        self.costs.append(sum(costs)) #/n_test)
        #self.costs.append(sum(costs[:][:]))
        #self.costs.append([sum(costs[0]),sum(costs[1]),sum(costs[2])])
        if (_debug_verbose): print ("Total Cost {1} for Epoch {0} complete".format(len(self.costs), sum(self.costs[-1])))
        if (_debug_verbose): print ("Axis-wise Cost is {0} ".format((self.costs[-1])))
        return self.costs[-1]

 
    def GD(self, training_data, epochs, learning_rate, test_data=None):
        """Train the neural network using batch-wise 
        gradient descent. If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            self.process_batch(training_data, learning_rate)
            if test_data:
                result = self.evaluate(test_data, n_test)
                if (_debug_verbose): print ("Epoch {0}: Score {1} / {2}".format(j, result, n_test))
            else:
                if (_debug_verbose): print ("Epoch {0} complete".format(j))

                        
    def process_batch(self, batch, learning_rate):
        """Update the network's weights by applying
        gradient descent using backpropagation to a single batch.
        """
        base_w = [np.zeros(w.shape) for w in self.weights]
        base_b = [np.zeros(b.shape) for b in self.biases]
        count=0
        for x in batch:
            delta_error_b , delta_error_w = self.backprop(x)
            updated_b = [nb+dnb for nb, dnb in zip(base_b, delta_error_b)]
            updated_w = [nw+dnw for nw, dnw in zip(base_w, delta_error_w)]
            count=count+1
        #print ("Process {0} inputs backprop ".format(count))    
            eta=learning_rate   
            self.weights = [w-(eta/len(batch))*nw
                        for w, nw in zip(self.weights, updated_w)]
            self.biases = [b-(eta/len(batch))*nb
                       for b, nb in zip(self.biases, updated_b)]
        
    def backprop(self, x):
        """Return ``( delta_w)`` representing the
        gradient for the cost function C_x. """
        
        if (_debug_verbose): print ("input: ", x)
        
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the activation (z) vectors, layer by layer
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.unit_step(z)
            activations.append(activation)
        
        if (_debug_verbose): print ("activations: ", activations)
        
        # backward pass
        delta = self.cost_derivative(activations[-1], x) *             self.unit_step_prime(zs[-1]) 
        delta_b[-1] = delta    
        delta_w[-1] = np.dot(delta, activations[-2].transpose())
        
        if (_debug_verbose): print ("cost derivative: ", self.cost_derivative(activations[-1], x))
        if (_debug_verbose): print ("unit step: ", self.unit_step_prime(zs[-1]))
        if (_debug_verbose): print("delta: ",delta)
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            step1 = np.dot(self.weights[-l+1].transpose(), delta)
            delta = step1 * z
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, activations[-l-1].transpose())
            if (_debug_verbose): print ("delta b updated: ", delta_b)
            if (_debug_verbose): print ("delta w updated:", delta_w)
            
        #print ("delta b: ", delta_b) 
        #print ("delta w:", delta_w)    
            
        return (delta_b, delta_w)

    def evaluate(self, test_data, n_test):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        self.cost_function_by_epoch(test_data, n_test)
        test_results = [self.forward(x)
                        for (x) in test_data]
        return sum(((x) - (x_hat))**2 for (x, x_hat) in zip(test_data, test_results))/n_test      
    
    def reconstruct(self, inputs):
        return [self.forward(x) for (x) in inputs]


# In[19]:

def rdm_wt_ae_errors():
    import numpy as np
    import matplotlib.pyplot as plt
    with open('planarity_errors_1489714415.76.json') as infile:
        c = json.load(infile)
        np_errors = np.asarray(c['non_planar_curves_error'])
        p_errors = np.asarray(c['planar_curves_error'])
    
    # clean data
    NPE = np.insert(np_errors, 1, 1, axis=2)
    PE = np.insert(p_errors, 1, 0, axis=2)
    X = np.concatenate((NPE, PE), axis=0)
    X = X.reshape(200,2)
    nan_idx = [i for i, x in enumerate(X) if (math.isnan(x[0]) == True)]
    print(nan_idx)
    X_cleaned = np.delete(X, nan_idx, axis=0)
    X_cleaned.shape


    bins = np.linspace(0, 100, 100)
    plt.hist(X_cleaned[0:100,0], bins, alpha=0.25, label='NPE')
    plt.hist(X_cleaned[100:198,0], bins, alpha=0.25, label='PE')
    plt.legend(loc='upper right')
    plt.show()



# In[32]:

def rdm_p_errors():
    # planar curves
    import numpy as np
    import matplotlib.pyplot as plt
    with open('planarity_errors_1488999893.39.json') as infile:
        c = json.load(infile)
        np_errors = np.asarray(c['non_planar_curves_error'])
        p_errors = np.asarray(c['planar_curves_error'])
    
    # clean data
    NPE = np.insert(np_errors, 1, 1, axis=2)
    PE = np.insert(p_errors, 1, 0, axis=2)
    X = np.concatenate((NPE, PE), axis=0)
    X = X.reshape(200,2)
    nan_idx = [i for i, x in enumerate(X) if (math.isnan(x[0]) == True)]
    print(nan_idx)
    X_cleaned = np.delete(X, nan_idx, axis=0)
    X_cleaned.shape
    
    hist, bins = np.histogram(X_cleaned[100:197,0], bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

def rdm_np_errors():
    # planar curves
    import numpy as np
    import matplotlib.pyplot as plt
    with open('planarity_errors_1488999893.39.json') as infile:
        c = json.load(infile)
        np_errors = np.asarray(c['non_planar_curves_error'])
        p_errors = np.asarray(c['planar_curves_error'])
    
    # clean data
    NPE = np.insert(np_errors, 1, 1, axis=2)
    PE = np.insert(p_errors, 1, 0, axis=2)
    X = np.concatenate((NPE, PE), axis=0)
    X = X.reshape(200,2)
    nan_idx = [i for i, x in enumerate(X) if (math.isnan(x[0]) == True)]
    print(nan_idx)
    X_cleaned = np.delete(X, nan_idx, axis=0)
    X_cleaned.shape
    
    hist, bins = np.histogram(X_cleaned[0:100,0], bins=70)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()
# In[36]:

# non-planar curves
#hist, bins = np.histogram(X_cleaned[100:199,0], bins=50)
#width = 0.7 * (bins[1] - bins[0])
#center = (bins[:-1] + bins[1:]) / 2
#plt.bar(center, hist, align='center', width=width)
#plt.show()

