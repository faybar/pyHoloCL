#
# vGnL
#
# Element wise addition of two vectors (c = a + b)
# Asks the user to select a device at runtime
#
# History: C version written by Tim Mattson, December 2009
#          C version Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
#          Ported to Python by Tom Deakin, July 2013
#--------------------------------
import pyopencl as cl
from numpy import *
import deviceinfo
import time
#------------------------------------------------------------------------------
TOL = 0.001     # tolerance used in floating point comparisons
# length of vectors a, b and c
LENGTH = 800 #1024
#LCD settings
nw = 800 #1024           #width
nh = 800 #768            #height
#optical setup settings
wave = 600          #wavelength
f = 10              #focal length
k = (2*pi)/wave     #k = 2pi / lambda
#trap location
trap1x = 5
trap1y = 5
trap1z = 0
#print settings
print '--settings--'
print 'wavelength:' + str(wave)
print 'focal length:' + str(f)
#print traps
print '--traps--'
print 'trap1x:' + str(trap1x)
print 'trap1y:' + str(trap1y)
print 'trap1z:' + str(trap1z)
#[X, Y] = meshgrid(np.linspace(-1,1,nw), np.linspace(-1,1,nh))
#lens1 = -trap1z * (X**2 + Y**2) / (2*f)
#grating = (np.dot(trap1x,X) + np.dot(trap1y,Y))
#data = np.empty([w,h],dtype = float)   #meshgrid(widthvect,widthvect);
#data = np.mod(f*(grating + lens1),(2*pi))
#------------------------------------------------------------------------------
# grating and lenses
#
# Input: a and b float vectors of length count
# Output c float vector of length count holding the sum a + b
kernelsource = open('GnL.cl').read()
#------------------------------------------------------------------------------
# Main procedure
# Create a compute context
# Ask the user to select a platform/device on the CLI
context = cl.create_some_context()
# Print out device info
deviceinfo.output_device_info(context.devices[0])
# Create a command queue
queue = cl.CommandQueue(context)
# Create the compute program from the source buffer
# and build it
program = cl.Program(context, kernelsource).build()
# Create a and b vectors and fill with random float values
h_X = numpy.random.rand(LENGTH).astype(numpy.float32)
h_Y = numpy.random.rand(LENGTH).astype(numpy.float32)
# Create an empty c vector (a+b) to be returned from the compute device
h_data = numpy.empty(LENGTH).astype(numpy.float32)
# Create the input (a, b) arrays in device memory and copy data from host
d_X = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_X)
d_Y = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Y)
# Create the output (c) array in device memory
d_data = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_data.nbytes)
# Start the timer
#rtime = time()
# Execute the kernel over the entire range of our 1d input
# allowing OpenCL runtime to select the work group items for the device
vGnL = program.vGnL
vGnL.set_scalar_arg_dtypes([None, None, None, None, None, numpy.float, numpy.float, numpy.uint32, numpy.uint32, numpy.uint32, numpy.uint32, numpy.uint32])
vGnL(queue, h_X.shape, None, d_X, d_Y, d_data, LENGTH)
# Wait for the commands to finish before reading back
queue.finish()
#rtime = time() - rtime
#print "The kernel ran in", rtime, "seconds"
# Read back the results from the compute device
cl.enqueue_copy(queue, h_data, d_data)
# Test the results
#correct = 0;
#for a, b, c in zip(h_X, h_Y, h_data):
#    # assign element i of a+b to tmp
#    tmp = a + b
#    # compute the deviation of expected and output result
#    tmp -= c
#    # correct if square deviation is less than tolerance squared
#    if tmp*tmp < TOL*TOL:
#        correct += 1
#    else:
#        print "tmp", tmp, "h_X", a, "h_Y", b, "h_c", c
## Summarize results
#print "C = A+B:", correct, "out of", LENGTH, "results were correct."
