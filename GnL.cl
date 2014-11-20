__kernel void vGnL(
    __global float* X,
    __global float* Y,
    __global float* lens1,
    __global float* grating,
    __global float* data,
    const float f,
    const float k,
    const unsigned int nw,
    const unsigned int nh,
    const unsigned int trap1x,
    const unsigned int trap1y,
    const unsigned int trap1z)
{
    int i = get_global_id(0);
    //if (i < nw)
        //c[i] = a[i] + b[i];
        //lens1 = -trap1z * (pow(X[i],2) + pow(Y[i],2)) / (2*f);
        //grating = (trap1x * X[i]) + (trap1y * Y[i]);
        //data = (f * (grating + lens1)) % (2 * M_PI_F);
}