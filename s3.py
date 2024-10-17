import numpy as np

# Data Extraction
X = np.array([[10],
     [15],
     [7],
     [4]])
y = np.array([11, 12, 15, 16]).reshape(-1,1)


# Linear regression
sample, feature = X.shape




o = np.ones((feature, 1))


theta = [0,0]

X_bias = np.c_[o, X]
sample, feature = X_bias.shape


print(X_bias)

theta = np.zeros((feature,1))

print(X_bias) # 4*2
print(theta) # 2*1

# Hyper parameters
learning_rate = 0.01
max_iter = 100

for i in range(max_iter):
     # Hypothesis (predict)

     h = np.dot(X_bias, theta)
     error = np.sqrt(np.mean(np.power(h - y, 2)))

     # gradient Descent
     gradient = np.dot(X_bias.T, error)/ sample

     # Updating Weights and bias
     theta = theta - learning_rate * gradient


