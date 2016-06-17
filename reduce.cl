__kernel void vecMinReduce(__global float *vec, __global float *results,
                           __constant uint *size, __local float *l_data)
{
    uint g_id = get_global_id(0);
    uint l_id = get_local_id(0);
   uint group_size = get_local_size(0);
   
        l_data[l_id] = vec[g_id];

   
	
    for (uint stride = group_size  / 2; stride > 0; stride >>= 1)
    {
		 barrier(CLK_LOCAL_MEM_FENCE);
        if (l_id < stride)
            l_data[l_id] += l_data[l_id + stride];
        
    }
    if (0 == l_id)
        results[get_group_id(0)] = l_data[0];
}
