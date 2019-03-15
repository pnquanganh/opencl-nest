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
__global double *V__P33_,
__global double *V__P11_,
__global double *P__I_e_,
__global double *P__omega_,
__global double *S__V_th_v_,
__global int *V__RefractoryCountsTot_,
__global double *S__I_syn_in_,
__global double *S__V_th_2_,
__global int *S__r_,
__global double *S__I_syn_ex_,
__global double *S__V_m_,
__global double *V__P62_,
__global double *V__P66_,
__global double *V__P55_,
__global double *V__P76_,
__global double *V__P60_,
__global double *V__P72_,
__global double *S__V_th_dv_,
__global double *V__P31_,
__global double *V__P70_,
__global double *V__P32_,
__global double *V__P77_,
__global double *P__alpha_1_,
__global double *S__V_th_1_,
__global double *V__P44_,
__global double *V__P63_,
__global double *V__P71_,
__global double *P__alpha_2_,
__global double *V__P61_,
__global double *V__P22_,
__global double *V__P30_,
__global double *V__P73_,
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

    // evolve voltage dependency (6,7)
    S__V_th_v_[tid] = ( P__I_e_[tid] + S__i_0_[tid] ) * V__P70_[tid] + S__I_syn_ex_[tid] * V__P71_[tid]
      + S__I_syn_in_[tid] * V__P72_[tid] + S__V_m_[tid] * V__P73_[tid] + S__V_th_dv_[tid] * V__P76_[tid]
      + S__V_th_v_[tid] * V__P77_[tid];

    S__V_th_dv_[tid] = ( P__I_e_[tid] + S__i_0_[tid] ) * V__P60_[tid] + S__I_syn_ex_[tid] * V__P61_[tid]
      + S__I_syn_in_[tid] * V__P62_[tid] + S__V_m_[tid] * V__P63_[tid] + S__V_th_dv_[tid] * V__P66_[tid];


    // evolve membrane potential (3)
    S__V_m_[tid] = ( P__I_e_[tid] + S__i_0_[tid] ) * V__P30_[tid] + S__I_syn_ex_[tid] * V__P31_[tid]
      + S__I_syn_in_[tid] * V__P32_[tid] + S__V_m_[tid] * V__P33_[tid];


    // evolve adaptive threshold (4,5)
    S__V_th_1_[tid] *= V__P44_[tid];
    S__V_th_2_[tid] *= V__P55_[tid];

    // exponential decaying PSCs (1,2)
    S__I_syn_ex_[tid] *= V__P11_[tid];
    S__I_syn_in_[tid] *= V__P22_[tid];
    S__I_syn_ex_[tid] +=
      ring_buffer_get_value(spikes_ex_, ring_buffer_size, num_nodes, tid, lag); // the spikes arriving at T+1 have an
    S__I_syn_in_[tid] +=
      ring_buffer_get_value(spikes_in_, ring_buffer_size, num_nodes, tid, lag); // the spikes arriving at T+1 have an


    if ( S__r_[tid] == 0 ) // neuron is allowed to fire
    {
      if ( S__V_m_[tid] >= P__omega_[tid] + S__V_th_2_[tid] + S__V_th_1_[tid]
          + S__V_th_v_[tid] ) // threshold crossing
      {
        S__r_[tid] = V__RefractoryCountsTot_[tid];

        // procedure for adaptive potential
        S__V_th_1_[tid] += P__alpha_1_[tid]; // short time
        S__V_th_2_[tid] += P__alpha_2_[tid]; // long time

        spike_count[tid]++;

        /*set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

        SpikeEvent se;
        kernel().event_delivery_manager.send( *this, se, lag )*/
        ;
      }
    }
    else
    {
      --S__r_[tid];
    } // neuron is totally refractory (cannot generate spikes)

    // set new input current
    S__i_0_[tid] = ring_buffer_get_value(currents_, ring_buffer_size, num_nodes, tid, lag);

    // log state data
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
