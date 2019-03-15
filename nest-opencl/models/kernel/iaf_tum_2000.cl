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
__global double *S__i_0_,
__global int *V__RefractoryCountsAbs_,
__global double *P__Theta_,
__global double *S__V_m_,
__global double *V__P21ex_,
__global double *P__V_reset_,
__global double *P__I_e_,
__global double *V__P21in_,
__global double *V__P20_,
__global int *S__r_tot_,
__global double *V__P11in_,
__global double *S__i_syn_ex_,
__global double *V__P11ex_,
__global int *V__RefractoryCountsTot_,
__global int *S__r_abs_,
__global double *V__P22_,
__global double *S__i_syn_in_,
__global double *currents_,
__global double *spikes_ex_,
__global double *spikes_in_,		  
		  ///
		  __global unsigned int *spike_count)
{
    unsigned int tid = get_global_id(0);

    if (tid >= num_nodes)
	return;

  {

    if ( S__r_abs_[tid] == 0 ) // neuron not refractory, so evolve V
    {
      S__V_m_[tid] = S__V_m_[tid] * V__P22_[tid] + S__i_syn_ex_[tid] * V__P21ex_[tid]
        + S__i_syn_in_[tid] * V__P21in_[tid] + ( P__I_e_[tid] + S__i_0_[tid] ) * V__P20_[tid];
    }
    else
    {
      --S__r_abs_[tid];
    } // neuron is absolute refractory

    // exponential decaying PSCs
    S__i_syn_ex_[tid] *= V__P11ex_[tid];
    S__i_syn_in_[tid] *= V__P11in_[tid];
    // the spikes arriving at T+1 have an immediate effect on the
    // state of the neuron
    S__i_syn_ex_[tid] += ring_buffer_get_value(spikes_ex_, ring_buffer_size, num_nodes, tid, lag);
    S__i_syn_in_[tid] += ring_buffer_get_value(spikes_in_, ring_buffer_size, num_nodes, tid, lag);

    if ( S__r_tot_[tid] == 0 )
    {
      if ( S__V_m_[tid] >= P__Theta_[tid] ) // threshold crossing
      {
        S__r_abs_[tid] = V__RefractoryCountsAbs_[tid];
        S__r_tot_[tid] = V__RefractoryCountsTot_[tid];
        S__V_m_[tid] = P__V_reset_[tid];

        spike_count[tid]++;

        /*set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

        SpikeEvent se;
        kernel().event_delivery_manager.send( *this, se, lag )*/
        ;
      }
    }
    else
    {
      --S__r_tot_[tid];
    } // neuron is totally refractory (cannot generate spikes)


    // set new input current
    S__i_0_[tid] = ring_buffer_get_value(currents_, ring_buffer_size, num_nodes, tid, lag);

    // logging
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
