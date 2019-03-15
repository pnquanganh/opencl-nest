#define WARP_SIZE 32



#define OPENCL_DBL_MEMCPY(dest, src, n, tid, nthreads)		\
    for (int i = 0; i < n; i++)					\
      {								\
	int idx = nthreads * i + tid;				\
	dest[idx] = src[idx];					\
      }

#define GSL_MAX_DBL(a,b) ((a) > (b) ? (a) : (b))

double ring_buffer_get_value(double *ring_buffer, int ring_buffer_size, int num_nodes, int tid, long lag)
{
  return ring_buffer[num_nodes * (lag % ring_buffer_size) + tid];
}

__kernel void update(int num_nodes, int dimension,
		  long lag,
		  int ring_buffer_size,
		  ////
__global double *P__a_,
__global double *P__c_,
__global double *P__V_th_,
__global double *S__I_,
__global double *P__I_e_,
__global double *P__V_min_,
__global bool *P__consistent_integration_,
__global double *S__u_,
__global double *P__b_,
__global double *S__v_,
__global double *P__d_,
__global double *currents_,
__global double *spikes_,		  
		  ///
		  __global unsigned int *spike_count)
{
    unsigned int tid = get_global_id(0);

    if (tid >= num_nodes)
	return;

  {
    // neuron is never refractory
    // use standard forward Euler numerics in this case
    if ( P__consistent_integration_[tid] )
    {
      v_old = S__v_[tid];
      u_old = S__u_[tid];
      S__v_[tid] += h * ( 0.04 * v_old * v_old + 5.0 * v_old + 140.0 - u_old + S__I_[tid]
                     + P__I_e_[tid] )
        + ring_buffer_get_value(spikes_, ring_buffer_size, num_nodes, tid, lag);
      S__u_[tid] += h * P__a_[tid] * ( P__b_[tid] * v_old - u_old );
    }
    // use numerics published in Izhikevich (2003) in this case (not
    // recommended)
    else
    {
      double I_syn = ring_buffer_get_value(spikes_, ring_buffer_size, num_nodes, tid, lag);
      S__v_[tid] += h * 0.5 * ( 0.04 * S__v_[tid] * S__v_[tid] + 5.0 * S__v_[tid] + 140.0 - S__u_[tid]
                           + S__I_[tid] + P__I_e_[tid] + I_syn );
      S__v_[tid] += h * 0.5 * ( 0.04 * S__v_[tid] * S__v_[tid] + 5.0 * S__v_[tid] + 140.0 - S__u_[tid]
                           + S__I_[tid] + P__I_e_[tid] + I_syn );
      S__u_[tid] += h * P__a_[tid] * ( P__b_[tid] * S__v_[tid] - S__u_[tid] );
    }

    // lower bound of membrane potential
    S__v_[tid] = ( S__v_[tid] < P__V_min_[tid] ? P__V_min_[tid] : S__v_[tid] );

    // threshold crossing
    if ( S__v_[tid] >= P__V_th_[tid] )
    {
      S__v_[tid] = P__c_[tid];
      S__u_[tid] = S__u_[tid] + P__d_[tid];

      // compute spike time
      spike_count[tid]++;

      /*set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

      SpikeEvent se;
      kernel().event_delivery_manager.send( *this, se, lag )*/
      ;
    }

    // set new input current
    S__I_[tid] = ring_buffer_get_value(currents_, ring_buffer_size, num_nodes, tid, lag);

    // voltage logging
    /*LOG DATA*/
    //B_.logger_.record_data( origin.get_steps() + lag );
  }
}

__kernel void deliver_events(int num_nodes,
			     int event_size,
			     __global int *connections_ptr,
			     __global int *connections,
			     __global double *event_buffer_in,
			     __global double *event_buffer_out)
{
  unsigned int tid = get_global_id(0);
  unsigned int wid = tid / WARP_SIZE;
  unsigned int lane = tid & (WARP_SIZE - 1);
  unsigned int total_threads = get_global_size(0);
  unsigned int total_warps = total_threads / WARP_SIZE;

  for (int tgt_id = wid; tgt_id < num_nodes; tgt_id += total_warps)
    {
      //__global double *output = event_buffer_out + event_size * tgt_id;
      int src_start = connections_ptr[tgt_id];
      int src_end = connections_ptr[tgt_id + 1];
      
      for (int l = lane; l < event_size; l += WARP_SIZE)
	{
	  double tmp = 0.0;
	  for (int i = src_start; i < src_end; i++)
	    {
	      int src_id = connections[i];
	      __global double *input = event_buffer_in + event_size * src_id;
	      tmp += input[l];
	    }
	  event_buffer_out[num_nodes * l + tgt_id] += tmp;
	}
    }
  
  /* for (int tgt_id = wid; tgt_id < num_nodes; tgt_id += total_warps) */
  /*   { */
  /*     if (lane == 0) */
  /* 	sumj = B_sumj_g_ij_[tgt_id]; */
      
  /*     __global double *tgt_coeff_buffer = coeff_buffer + event_size * tgt_id; */
      
  /*     int src_start = connections_ptr[tgt_id]; */
  /*     int src_end = connections_ptr[tgt_id + 1]; */
  /*     for (int i = src_start; i < src_end; i++) */
  /* 	{ */
  /* 	  int src_id = connections[i]; */
  /* 	  double src_weight = event_weight[src_id]; */

  /* 	  sumj += src_weight; */

  /* 	  __global double *src_event_buffer = event_buffer + event_size * src_id; */
	  
  /* 	  for (int l = lane; l < event_size; l += WARP_SIZE) */
  /* 	    { */
  /* 	      tgt_coeff_buffer[l] += src_event_buffer[l]; */
  /* 	    } */
  /* 	} */

  /*     if (lane == 0) */
  /* 	B_sumj_g_ij_[tgt_id] = sumj; */
  /*   } */
}
