import cv2
import numpy

import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def disp_norm(img):
	img = img.astype(numpy.float128)
	img -= img.min()
	img /= img.max()
	img *= 255
	return img.astype(numpy.uint8)

def norm_range01(img):
	img = img.astype(numpy.float64)
	img -= img.min()
	img /= img.max()

	return img

def inspect(array):
	""" Look at some values and statistics of a numpy array.
	Useful for checking type and value ranges.
	"""
	if type(array) == numpy.ndarray:
		info = {'min':array.min(), 'max':array.max(), 'mean':array.mean(), 'var':array.var(), 'dtype':array.dtype, 'shape':array.shape}
		
		return ' '.join(['{0}: {1}'.format(key, info[key]) for key in info])

def show(img, name=None, mode='gray', *args, **kwargs):
	""" Show an image in a matplotlib window.
	Designed for use from python shell.

	args:
		img: image to be shown
		name: optional, name of figure. can be used to modify an existing figure
		rest: matplotlib.imshow args
	"""
	plt.ion()
	if name:
		fig = plt.figure(name)
	else:
		fig = plt.figure()

	plt.imshow(img, mode, args, kwargs)
	#return fig

def close(fig='all'):
	""" destroy a figure or by default all figures
	"""
	plt.close(fig)

def read(file):
	""" Wrap cv2.imread with warnings.
	"""
	img = cv2.imread(file)

	if img is None:
		warnings.warn('Image did not load from path: {0}'.format(file), UserWarning)

	return img

def show_complex_filter(filter):

	fig = plt.figure()
 	ax = fig.add_subplot(211, projection='3d')
 	x, y = numpy.indices(filter.shape)
 	ax.plot_wireframe(x, y, filter.real)
 	ax.set_title('Real')
 	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')

 	ax = fig.add_subplot(212, projection='3d')
 	ax.plot_wireframe(x, y, filter.imag)
 	ax.set_title('Imaginary')
 	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')

 	return fig

def imshow(image, mode='gray'):
	fig = plt.figure()
	size = fig.get_size_inches() * fig.dpi

	plt.imshow(image, mode)
	fig.tight_layout()
	plt.colorbar()

	return fig
