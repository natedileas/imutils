import cv2
import numpy
import math

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


def rescale(img, new_min=0, new_max=1, type='linear'):
	img = img.astype(numpy.float64)
	img -= img.min() - new_min
	img *= new_max / img.max()

	return img
	
def inspect(array):
	if type(array) == numpy.ndarray:
		info = {'min':array.min(), 'max':array.max(), 'mean':array.mean(), 'var':array.var(), 'dtype':array.dtype, 'shape':array.shape}
		
		return ' '.join(['{0}: {1}'.format(key, info[key]) for key in info])


def rotate_block(angle, x, y, new_type=numpy.int32):
	block = numpy.asarray([x.flatten().astype(numpy.float64), y.flatten().astype(numpy.float64)])

	c, s = numpy.cos(angle), numpy.sin(angle)

	rot_block = numpy.matrix([[c, s], [-s, c]]) * block

	r_x, r_y = numpy.round(numpy.asarray(rot_block)).astype(new_type)

	return r_x, r_y


def block(i, j, theta, block_shape, image_shape):
	""" Project a rotated block into an image.

	Rotated around center pixel i, j.

	i, j: pixel coordinates
	theta: radians, angle to rotate block to from x-axis
	block_shape: shape of the block to generate
	image_shape: shape of the image
	"""
	theta -= numpy.pi / 2.

	y, x = numpy.indices(block_shape).astype(numpy.float64)

	y -= math.floor(block_shape[0] / 2.)
	x -= math.floor(block_shape[1] / 2.)

	c, r = rotate_block(theta, x, y)

	r += i
	c += j

	if (r >= image_shape[0]).any() or (c >= image_shape[1]).any() or (r < 0).any() or (c < 0).any():
		raise ValueError('New Indices out of range.')

	return r, c
