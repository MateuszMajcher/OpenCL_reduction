
float f1(float x) { return (2*(x*x*x)+(4*x)); }

__kernel void vecAdd(__global float* a,
  float x,
   float  y,
	const unsigned int  size)
{
	local float h;
    unsigned int i = get_global_id(0);
    h =(y-x) / (float)size;
	if ( i < size )  {
	
    a[i] = f1(x + (i+1)*h)  *h;

	}

}