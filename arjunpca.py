import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

data = [
	(6,6),
	(4,8),
	(2,6),
	(2,3),
	(4,4)	
		]

x = np.array([e[0] for e in data])
y = np.array([e[1] for e in data])
x_mean = np.mean(x)
y_mean = np.mean(y)

def shift_to_center(data):
	global x_mean
	global y_mean
	new_data = []
	for i in range(len(data)):
		new_data.append((data[i][0]-x_mean, data[i][1]-y_mean))
	return new_data	

def variance(array):
	mean = np.mean(array)
	sums = [(s - mean)**2 for s in array]
	return np.mean(sums)

def covariance(data, index1=0, index2=1):
	x_mean = np.mean([e[0] for e in data])
	y_mean = np.mean([e[1] for e in data])
	product_of_coors = [(point[index1] - x_mean)*(point[index2] - y_mean) for point in data]
	return np.mean(product_of_coors)

normalized_data = shift_to_center(data)

norm_x = np.array([e[0] for e in normalized_data])
norm_y = np.array([e[1] for e in normalized_data])

''' construct covariance matrix '''
covariance_matrix = np.array([
[covariance(normalized_data, index1=0, index2=0), covariance(normalized_data)],
[covariance(normalized_data)                    , covariance(normalized_data, index1=1, index2=1)]])


''' linear transformations 
(x,y) -> (covariance_matrix[0][0] * x + covariance_matrix[0][1] * y, covariance_matrix[1][0] * x + covariance_matrix[1][1] * y)

'''

transformed_data = np.matmul(normalized_data, covariance_matrix)


tran_x = np.array([e[0] for e in transformed_data])
tran_y = np.array([e[1] for e in transformed_data])

''' eigen stuff '''

eignvalues, eigenvectors= eig(covariance_matrix)

''' sort by PC values '''

if eignvalues[1] > eignvalues[0]:
	eigenvectors[0], eigenvectors[1] = eigenvectors[1], eigenvectors[0]
	eignvalues[0], eignvalues[1] = eignvalues[1], eignvalues[0]

''' rotations - arjun being an ideiot '''

angle = 2*np.pi - np.arctan2(eigenvectors[0][0], -eigenvectors[1][0])
cos = np.cos(angle) 
sin = np.sin(angle)

norm_rotated_data = []
for point in normalized_data:
	nx = point[0]*cos - point[1]*sin
	ny = point[0]*sin + point[1]*cos
	norm_rotated_data.append([nx, ny])

norm_rotatedx = np.array([e[0] for e in norm_rotated_data])
norm_rotatedy = np.array([e[1] for e in norm_rotated_data])

transformed_rotated_data = []
for point in transformed_data:
	nx = point[0]*cos - point[1]*sin
	ny = point[0]*sin + point[1]*cos
	transformed_rotated_data.append([nx, ny])



transformed_rotated_datax = np.array([e[0] for e in transformed_rotated_data])
transformed_rotated_datay = np.array([e[1] for e in transformed_rotated_data])



''' show data '''
fig = plt.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(236)
ax6 = fig.add_subplot(235)

ax1.scatter(x,y)
ax1.scatter(x_mean, y_mean)
ax1.set_title("starting data")

ax2.scatter(norm_x, norm_y)
ax2.axhline(y=0, color='k')
ax2.axvline(x=0, color='k')
ax2.set_title("eigenvectors (w/o linear transformation)")

ax3.scatter(tran_x, tran_y)
ax3.set_title("eigenvectors with linear transformation")
ax2.arrow(0,0,eigenvectors[0][0]*eignvalues[0], eigenvectors[1][0]*eignvalues[0], color="green")
ax2.arrow(0,0,eigenvectors[0][1]*eignvalues[1], eigenvectors[1][1]*eignvalues[1], color="green")
ax3.arrow(0,0,eigenvectors[0][0]*eignvalues[0], eigenvectors[1][0]*eignvalues[0], color="green")
ax3.arrow(0,0,eigenvectors[0][1]*eignvalues[1], eigenvectors[1][1]*eignvalues[1], color="green")
ax3.axhline(y=0, color='k')
ax3.axvline(x=0, color='k')

ax4.set_title("new principle components (w/o linear transformation)")
ax4.scatter(norm_rotatedx, norm_rotatedy, color="magenta")
ax4.axhline(y=0, color='k')
ax4.axvline(x=0, color='k')

ax5.set_title("new principle components with linear transformation")
# ax5.set_xlim([-6,6])
# ax5.set_ylim([-6,6])
ax5.scatter(transformed_rotated_datax, transformed_rotated_datay, color="magenta")
ax5.axhline(y=0, color='k')
ax5.axvline(x=0, color='k')


ax6.text(0.5,1, f"Principle Component analysis\n eigenvalues: [{round(eignvalues[0], 3)}, {round(eignvalues[1], 3)}]", ha='center', va='top')

# print(tran_x, tran_y)

plt.show()



