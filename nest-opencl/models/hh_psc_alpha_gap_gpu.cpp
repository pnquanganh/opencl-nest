#include "hh_psc_alpha_gap_gpu.h"
#include "hh_psc_alpha_gap.h"

#ifdef PROFILING
#include <sys/time.h>
#endif

nest::hh_psc_alpha_gap_gpu::clContext_ nest::hh_psc_alpha_gap_gpu::gpu_context;
bool nest::hh_psc_alpha_gap_gpu::is_gpu_initialized = false;

nest::hh_psc_alpha_gap_gpu::hh_psc_alpha_gap_gpu()
  : graph_size( 0 )
  , h_connections_ptr( NULL )
  , h_connections( NULL )
  , event_size( 0 )
  , h_event_buffer( NULL )
  , h_coeff_buffer( NULL )
  , h_event_weight( NULL )
  , h_B_sumj( NULL )
  , is_initialized( false )
  , h_e_y0( NULL )
  , h_e_yerr( NULL )
  , h_e_dydt_in( NULL )
  , h_e_dydt_out( NULL )
  , h_con_state_eps_abs( NULL )
  , h_con_state_eps_rel( NULL )
  , h_con_state_a_y( NULL )
  , h_con_state_a_dydt( NULL )
  , h_rk_state_k1( NULL )
  , h_rk_state_k2( NULL )
  , h_rk_state_k3( NULL )
  , h_rk_state_k4( NULL )
  , h_rk_state_k5( NULL )
  , h_rk_state_k6( NULL )
  , h_rk_state_y0( NULL )
  , h_rk_state_ytmp( NULL )
  , h_P_g_Na( NULL )
  , h_P_g_Kv1( NULL )
  , h_P_g_Kv3( NULL )
  , h_P_g_L( NULL )
  , h_P_C_m( NULL )
  , h_P_E_Na( NULL )
  , h_P_E_K( NULL )
  , h_P_E_L( NULL ) 
  , h_P_tau_synE( NULL )
  , h_P_tau_synI( NULL )
  , h_P_I_e( NULL )
  , h_B_step_( NULL )
  , h_B_lag_( NULL )
  , h_B_sumj_g_ij_( NULL )
  , h_B_interpolation_coefficients( NULL )
  , h_new_coefficients( NULL )
  , h_B_I_stim_( NULL )
  , h_B_IntegrationStep_( NULL )
  , h_S_y_( NULL )
  , h_y_i( NULL )
  , h_f_temp( NULL )
  , h_hf_i( NULL )
  , h_hf_ip1( NULL )
  , h_B_spike_exc_( NULL )
  , h_B_spike_inh_( NULL )
  , h_y_ip1( NULL)
{
}

nest::hh_psc_alpha_gap_gpu::~hh_psc_alpha_gap_gpu()
{
  if (h_event_buffer) delete[] h_event_buffer;
  if (h_coeff_buffer) delete[] h_coeff_buffer;
  if (h_event_weight) delete[] h_event_weight;
  if (h_B_sumj) delete[] h_B_sumj;
  if (h_e_y0) delete[] h_e_y0;
  if (h_e_yerr) delete[] h_e_yerr;
  if (h_e_dydt_in) delete[] h_e_dydt_in;
  if (h_e_dydt_out) delete[] h_e_dydt_out;
  if (h_con_state_eps_abs) delete[] h_con_state_eps_abs;
  if (h_con_state_eps_rel) delete[] h_con_state_eps_rel;
  if (h_con_state_a_y) delete[] h_con_state_a_y;
  if (h_con_state_a_dydt) delete[] h_con_state_a_dydt;
  if (h_rk_state_k1) delete[] h_rk_state_k1;
  if (h_rk_state_k2) delete[] h_rk_state_k2;
  if (h_rk_state_k3) delete[] h_rk_state_k3;
  if (h_rk_state_k4) delete[] h_rk_state_k4;
  if (h_rk_state_k5) delete[] h_rk_state_k5;
  if (h_rk_state_k6) delete[] h_rk_state_k6;
  if (h_rk_state_y0) delete[] h_rk_state_y0;
  if (h_rk_state_ytmp) delete[] h_rk_state_ytmp;
  if (h_P_g_Na) delete[] h_P_g_Na;
  if (h_P_g_Kv1) delete[] h_P_g_Kv1;
  if (h_P_g_Kv3) delete[] h_P_g_Kv3;
  if (h_P_g_L) delete[] h_P_g_L;
  if (h_P_C_m) delete[] h_P_C_m;
  if (h_P_E_Na) delete[] h_P_E_Na;
  if (h_P_E_K) delete[] h_P_E_K;
  if (h_P_E_L) delete[] h_P_E_L;
  if (h_P_tau_synE) delete[] h_P_tau_synE;
  if (h_P_tau_synI) delete[] h_P_tau_synI;
  if (h_P_I_e) delete[] h_P_I_e;
  if (h_B_step_) delete[] h_B_step_;
  if (h_B_lag_) delete[] h_B_lag_;
  if (h_B_sumj_g_ij_) delete[] h_B_sumj_g_ij_;
  if (h_y_i) delete[] h_y_i;
  if (h_f_temp) delete[] h_f_temp;
  if (h_hf_i) delete[] h_hf_i;
  if (h_hf_ip1) delete[] h_hf_ip1;
  if (h_B_spike_exc_) delete[] h_B_spike_exc_;
  if (h_B_spike_inh_) delete[] h_B_spike_inh_;
  if (h_y_ip1) delete[] h_y_ip1;
  if (h_B_interpolation_coefficients) delete[] h_B_interpolation_coefficients;
  if (h_new_coefficients) delete[] h_new_coefficients;
  if (h_B_I_stim_) delete[] h_B_I_stim_;
  if (h_B_IntegrationStep_) delete[] h_B_IntegrationStep_;
  if (h_S_y_) delete[] h_S_y_;
}

void
nest::hh_psc_alpha_gap_gpu::initialize_gpu()
{
  if (is_gpu_initialized)
    return;

  if (initialize_opencl_context())
    return;

  // if (getKernel("gsl", "deliver_events", &gpu_context))
  //   return;
  
  is_gpu_initialized = true;
}

void
nest::hh_psc_alpha_gap_gpu::mass_update(std::vector< Node* > nodes, Time const& origin,
					const long from,
					const long to )
{
  mass_update_(nodes, origin, from, to, false );
}

bool
nest::hh_psc_alpha_gap_gpu::mass_wfr_update(std::vector< Node* > nodes, Time const& origin,
					    const long from,
					    const long to )
{
  return mass_update_(nodes, origin, from, to, true );
}

// TODO: the for loops will later be kernel calls
bool
nest::hh_psc_alpha_gap_gpu::mass_update_( std::vector<Node *> &nodes,
					  Time const& origin,
					  const long from,
					  const long to,
					  const bool called_from_wfr_update ) // TODO: don't know yet whether we need to cover both cases here
{
  // TODO: do AoS for now, SoA will come later on
  int total_num_nodes = kernel().node_manager.size();
  int num_local_nodes = nodes.size();
  int thrd_id = kernel().vp_manager.get_thread_id();
  
  initialize_device(total_num_nodes, num_local_nodes, hh_psc_alpha_gap::State_::STATE_VEC_SIZE);
  set_kernel_args(this->gpu_kernel, num_local_nodes, hh_psc_alpha_gap::State_::STATE_VEC_SIZE, kernel().simulation_manager.get_wfr_interpolation_order(), called_from_wfr_update);

  // TODO: for now, I assume that these are the same for all nodes
  const size_t interpolation_order =
    kernel().simulation_manager.get_wfr_interpolation_order();
  const double wfr_tol = kernel().simulation_manager.get_wfr_tol();
  // allocate memory to store the new interpolation coefficients
  // to be sent by gap event
  const size_t buffer_size =
    kernel().connection_manager.get_min_delay() * ( interpolation_order + 1 );

  bool global_wfr_tol_not_exceeded = true;
 
  std::vector<hh_psc_alpha_gap::State_> old_state;
  // old_state.resize(nodes.size());

  for ( std::vector<Node*>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); nodeIt++ )
    {
      nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;
      // printf("node %p, before update_, state is %.2f\n", node, node->S_.y_[State_::V_M]);
      if ( called_from_wfr_update )
        old_state.push_back(node->S_); // save old state

      // std::vector< double > new_coefficients( buffer_size, 0.0 );
      node->new_coefficients = std::vector < double >( buffer_size, 0.0); // TODO: is the above equivalent to clear()?
  
      // parameters needed for piecewise interpolation
      node->hf_i = 0.0;
      node->hf_ip1 = 0.0;
      node->y_i = 0.0;
      node->y_ip1 = 0.0;
      // double f_temp[ State_::STATE_VEC_SIZE ];

    }

  bool first_loop = true;
  if (from < to)
    prepare_copy_to_device(nodes, called_from_wfr_update, from);
  for ( long lag = from; lag < to; ++lag, first_loop = false )
    {

#ifdef PROFILING
      struct timeval start_time, end_time, diff_time;
      double diff;
      gettimeofday(&start_time, NULL);
#endif
      // Start of GPU section
#ifdef PROFILING
      struct timeval start_time_h, end_time_h, diff_time_h;
      double diff_h;
      gettimeofday(&start_time_h, NULL);
#endif
    
      copy_data_to_device(nodes, hh_psc_alpha_gap::State_::STATE_VEC_SIZE, first_loop, called_from_wfr_update, lag);
      synchronize();

#ifdef PROFILING
      gettimeofday(&end_time_h, NULL);
      timersub(&end_time_h, &start_time_h, &diff_time_h);
      diff_h = (double)diff_time_h.tv_sec*1000 + (double)diff_time_h.tv_usec/1000;
      printf("HtD %d: %0.3f\n", thrd_id, diff_h);
#endif

#ifdef PROFILING
      struct timeval start_time_k, end_time_k, diff_time_k;
    
      gettimeofday(&start_time_k, NULL);
#endif

      execute_kernel(this->gpu_kernel, &gpu_context, num_local_nodes);

      if (lag + 1 < to)
	prepare_copy_to_device(nodes, called_from_wfr_update, lag + 1);

      synchronize();
    
#ifdef PROFILING
      gettimeofday(&end_time_k, NULL);
      timersub(&end_time_k, &start_time_k, &diff_time_k);
      double diff_k = (double)diff_time_k.tv_sec*1000 + (double)diff_time_k.tv_usec/1000;
      printf("Execute kernel %d: %0.3f\n", thrd_id, diff_k);
#endif

      // #ifdef PROFILING
      //     gettimeofday(&start_time, NULL);
      // #endif
#ifdef PROFILING
      struct timeval start_time_d, end_time_d, diff_time_d;
      double diff_d;
      gettimeofday(&start_time_d, NULL);
#endif
    
      copy_data_from_device(nodes, hh_psc_alpha_gap::State_::STATE_VEC_SIZE, false, called_from_wfr_update);

      // #ifdef PROFILING
      //     gettimeofday(&end_time, NULL);
      //     timersub(&end_time, &start_time, &diff_time);
      //     diff = (double)diff_time.tv_sec*1000 + (double)diff_time.tv_usec/1000;
      //     printf("Phase 2: %0.3f\n", diff);
      // #endif

      // #ifdef PROFILING
      //     gettimeofday(&start_time, NULL);
      // #endif

#ifdef PROFILING
      gettimeofday(&end_time_d, NULL);
      timersub(&end_time_d, &start_time_d, &diff_time_d);
      diff_d = (double)diff_time_d.tv_sec*1000 + (double)diff_time_d.tv_usec/1000;
      printf("DtH %d: %0.3f\n", thrd_id, diff_d);
#endif

#ifdef PROFILING
      gettimeofday(&end_time, NULL);
      timersub(&end_time, &start_time, &diff_time);
      diff = (double)diff_time.tv_sec*1000 + (double)diff_time.tv_usec/1000;
      printf("GPU %d: %0.3f\n", thrd_id, diff);
#endif

      // End of GPU section

      // #ifdef PROFILING
      //     gettimeofday(&start_time, NULL);
      // #endif

      //int nodeid = 0;
      // XXX: post gsl
      for ( std::vector<Node *>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); nodeIt++)
	{
	  nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;
	  bool wfr_tol_exceeded = false;

	  // State_ old_state = node->S_; // save state

	  // gettimeofday( &t_slice_end_, NULL );
	  // if ( t_slice_end_.tv_sec != 0 )
	  // {
	  //   // usec
	  //   long t_real_ = 0;
	  //   long t_real_s = ( t_slice_end_.tv_sec - t_slice_begin_.tv_sec ) * 1e6;
	  //   // usec
	  //   t_real_ += t_real_s + ( t_slice_end_.tv_usec - t_slice_begin_.tv_usec );
	  //   printf("\nwhile loop took %.2fms\n", t_real_ / 1e3);
	  // }

	  if ( not called_from_wfr_update )
	    {
	      node->S_.y_[ hh_psc_alpha_gap::State_::DI_EXC ] +=
		node->B_.spike_exc_.get_value( lag ) * node->V_.PSCurrInit_E_;
	      node->S_.y_[ hh_psc_alpha_gap::State_::DI_INH ] +=
		node->B_.spike_inh_.get_value( lag ) * node->V_.PSCurrInit_I_;
	      // sending spikes: crossing 0 mV, pseudo-refractoriness and local
	      // maximum...
	      // refractory?
	      if ( node->S_.r_ > 0 )
		{
		  --node->S_.r_;
		}
	      else
		// (    threshold    &&     maximum       )
		if ( node->S_.y_[ hh_psc_alpha_gap::State_::V_M ] >= 0 && node->U_old > node->S_.y_[ hh_psc_alpha_gap::State_::V_M ] )
		  {
		    node->S_.r_ = node->V_.RefractoryCounts_;

		    node->set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

		    SpikeEvent se;

		    // printf("node %p sending spike with lag %d: ", node, lag);

		    kernel().event_delivery_manager.send( *node, se, lag );
		  }

	      // log state data
	      node->B_.logger_.record_data( origin.get_steps() + lag );

	      // set new input current
	      node->B_.I_stim_ = node->B_.currents_.get_value( lag );
	    }
	  else // if(called_from_wfr_update)
	    {
	      // node->S_.y_[ State_::DI_EXC ] +=
	      //   node->B_.spike_exc_.get_value_wfr_update( lag ) * node->V_.PSCurrInit_E_;
	      // node->S_.y_[ State_::DI_INH ] +=
	      //   node->B_.spike_inh_.get_value_wfr_update( lag ) * node->V_.PSCurrInit_I_;

	      //int nodeid = node->get_gid();


	      // check if deviation from last iteration exceeds wfr_tol
      
	      wfr_tol_exceeded = wfr_tol_exceeded
		or fabs( node->S_.y_[ hh_psc_alpha_gap::State_::V_M ] - node->B_.last_y_values[ lag ] ) > wfr_tol;

	      node->B_.last_y_values[ lag ] = node->S_.y_[ hh_psc_alpha_gap::State_::V_M ];

	      // // update different interpolations

	      // // constant term is the same for each interpolation order
	      // node->new_coefficients[ lag * ( interpolation_order + 1 ) + 0 ] = node->y_i;

	      // switch ( interpolation_order )
	      // {
	      // case 0:
	      //   break;

	      // case 1:
	      //   node->y_ip1 = node->S_.y_[ State_::V_M ];

	      //   node->new_coefficients[ lag * ( interpolation_order + 1 ) + 1 ] = node->y_ip1 - node->y_i;

	      //   break;

	      // case 3:
	      //   node->y_ip1 = node->S_.y_[ State_::V_M ];

	      //   hh_psc_alpha_gap_dynamics(
	      //     node->B_.step_, node->S_.y_, node->f_temp, node); // reinterpret_cast< void* >( this ) );
	      //   node->hf_ip1 = node->B_.step_ * node->f_temp[ State_::V_M ];

	      //   node->new_coefficients[ lag * ( interpolation_order + 1 ) + 1 ] = node->hf_i;
	      //   node->new_coefficients[ lag * ( interpolation_order + 1 ) + 2 ] =
	      //     -3 * node->y_i + 3 * node->y_ip1 - 2 * node->hf_i - node->hf_ip1;
	      //   node->new_coefficients[ lag * ( interpolation_order + 1 ) + 3 ] =
	      //     2 * node->y_i - 2 * node->y_ip1 + node->hf_i + node->hf_ip1;

	      //   break;
	      // default:
	      //   throw BadProperty( "Interpolation order must be 0, 1, or 3." );
	      //   }

	    }


	  global_wfr_tol_not_exceeded = global_wfr_tol_not_exceeded && not wfr_tol_exceeded;

	}

      // #ifdef PROFILING
      //     gettimeofday(&end_time, NULL);
      //     timersub(&end_time, &start_time, &diff_time);
      //     diff = (double)diff_time.tv_sec*1000 + (double)diff_time.tv_usec/1000;
      //     printf("cpu: %0.3f\n", diff);
      // #endif

    }

  copy_data_from_device(nodes, hh_psc_alpha_gap::State_::STATE_VEC_SIZE, true, called_from_wfr_update);

  for ( std::vector<Node*>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); nodeIt++ )
    {
      nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;
      // if not called_from_wfr_update perform constant extrapolation
      // and reset last_y_values
      if ( not called_from_wfr_update )
	{
	  // printf("resetting last y values\n");
	  for ( long temp = from; temp < to; ++temp )
	    {
	      node->new_coefficients[ temp * ( interpolation_order + 1 ) + 0 ] =
		node->S_.y_[ hh_psc_alpha_gap::State_::V_M ];
	    }

	  std::vector< double >( kernel().connection_manager.get_min_delay(), 0.0 )
	    .swap( node->B_.last_y_values );
	}

      // Send gap-event
      GapJunctionEvent ge;
      ge.set_coeffarray( node->new_coefficients );
      // printf("node %p send_secondary with coeffs: ", node);
      // for(int i = 0; i < node->new_coefficients.size(); i++)
      // {
      //   printf("%.2f ", node->new_coefficients[i]);
      // }
      // printf("\n");

      kernel().event_delivery_manager.send_secondary( *node, ge );

      // Reset variables
      node->B_.sumj_g_ij_ = 0.0;
      std::vector< double >( buffer_size, 0.0 )
	.swap( node->B_.interpolation_coefficients );

      //    for(int i = 0; i < buffer_size; i++)
      //    {
      //      printf("B_.interpolation_coefficients[%d]: %.2f\n", i, node->B_.interpolation_coefficients[i]);
      //    }
    }

  for ( std::vector<Node*>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); nodeIt++ )
    {
      nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;

      if ( called_from_wfr_update )
        node->S_ = old_state[nodeIt - nodes.begin()]; // restore old state
    }

  return global_wfr_tol_not_exceeded;
}

int
nest::hh_psc_alpha_gap_gpu::initialize_opencl_context()
{
  int thrd_id = kernel().vp_manager.get_thread_id();
  printf("Thread %d\n", thrd_id);

  try
    {
      // Get list of OpenCL platforms.
      std::vector<cl::Platform> list_platform;
      cl::Platform::get(&list_platform);

      if (list_platform.empty()) {
	std::cerr << "OpenCL platforms not found." << std::endl;
	return 1;
      }

      gpu_context.platform = list_platform[0];
      
      gpu_context.platform.getDevices(CL_DEVICE_TYPE_GPU, &gpu_context.list_device);

      //gpu_context.list_device.push_back(devices[thrd_id]);
      
      //gpu_context.context = cl::Context(gpu_context.list_device);

    }
   catch (const cl::Error &err) {
	std::cerr
	    << "OpenCL error: "
	    << err.what() << "(" << err.err() << ")"
	    << std::endl;
	return 1;
    }

  return 0;
}

int
nest::hh_psc_alpha_gap_gpu::initialize_command_queue(clContext_ *clCxt)
{
  int thrd_id = kernel().vp_manager.get_thread_id();

  try
    {
      this->devices.push_back(gpu_context.list_device[thrd_id]);

      this->context = cl::Context(this->devices);
  
      this->command_queue = cl::CommandQueue(this->context, this->devices[0]);

    }
  catch (const cl::Error &err) {
	std::cerr
	    << "OpenCL error: "
	    << err.what() << "(" << err.err() << ")"
	    << std::endl;
	return 1;
    }

  return 0;
}

void
nest::hh_psc_alpha_gap_gpu::initialize_device(int total_num_nodes, int num_local_nodes, int dimension)
{
  if (!is_initialized)
    {
      printf("initialize_device %d\n", num_local_nodes);
      connections.resize(total_num_nodes);
      event_size = kernel().connection_manager.get_min_delay()
	* ( kernel().simulation_manager.get_wfr_interpolation_order() + 1 );

      h_event_buffer = new double[total_num_nodes * event_size];
      h_coeff_buffer = new double[total_num_nodes * event_size];
      h_event_weight = new double[total_num_nodes];
      h_B_sumj = new double[total_num_nodes];
      
      if (initialize_command_queue(&gpu_context))
	return;

      getKernel("hh_psc_alpha_gap", "gsl", "deliver_events", &gpu_context);
            
      int len1 = dimension*num_local_nodes;
      int len2 = num_local_nodes;

      h_e_y0 = new double[len1];
      h_e_yerr = new double[len1];
      h_e_dydt_in = new double[len1];
      h_e_dydt_out = new double[len1];
      h_con_state_eps_abs = new double[len2];
      h_con_state_eps_rel = new double[len2];
      h_con_state_a_y = new double[len2];
      h_con_state_a_dydt = new double[len2];
      h_rk_state_k1 = new double[len1];
      h_rk_state_k2 = new double[len1];
      h_rk_state_k3 = new double[len1];
      h_rk_state_k4 = new double[len1];
      h_rk_state_k5 = new double[len1];
      h_rk_state_k6 = new double[len1];
      h_rk_state_y0 = new double[len1];
      h_rk_state_ytmp = new double[len1];
      h_P_g_Na = new double[len2];
      h_P_g_Kv1 = new double[len2];
      h_P_g_Kv3 = new double[len2];
      h_P_g_L = new double[len2];
      h_P_C_m = new double[len2];
      h_P_E_Na = new double[len2];
      h_P_E_K = new double[len2];
      h_P_E_L = new double[len2];
      h_P_tau_synE = new double[len2];
      h_P_tau_synI = new double[len2];
      h_P_I_e = new double[len2];
      h_B_step_ = new double[len2];
      h_B_lag_ = new long[len2];
      h_B_sumj_g_ij_ = new double[len2];
      h_y_i = new double[len2];
      h_f_temp = new double[len1];
      h_hf_i = new double[len2];
      h_hf_ip1 = new double[len2];
      h_B_spike_exc_ = new double[len2];
      h_B_spike_inh_ = new double[len2];
      //h_B_last_y_values = new double[len2];
      h_y_ip1 = new double[len2];

      create(&gpu_context, &e_y0, len1*sizeof(double));
      create(&gpu_context, &e_yerr, len1*sizeof(double));
      create(&gpu_context, &e_dydt_in, len1*sizeof(double));
      create(&gpu_context, &e_dydt_out, len1*sizeof(double));
      create(&gpu_context, &con_state_eps_abs, len2*sizeof(double));
      create(&gpu_context, &con_state_eps_rel, len2*sizeof(double));
      create(&gpu_context, &con_state_a_y, len2*sizeof(double));
      create(&gpu_context, &con_state_a_dydt, len2*sizeof(double));
      create(&gpu_context, &rk_state_k1, len1*sizeof(double));
      create(&gpu_context, &rk_state_k2, len1*sizeof(double));
      create(&gpu_context, &rk_state_k3, len1*sizeof(double));
      create(&gpu_context, &rk_state_k4, len1*sizeof(double));
      create(&gpu_context, &rk_state_k5, len1*sizeof(double));
      create(&gpu_context, &rk_state_k6, len1*sizeof(double));
      create(&gpu_context, &rk_state_y0, len1*sizeof(double));
      create(&gpu_context, &rk_state_ytmp, len1*sizeof(double));
      create(&gpu_context, &P_g_Na, len2*sizeof(double));
      create(&gpu_context, &P_g_Kv1, len2*sizeof(double));
      create(&gpu_context, &P_g_Kv3, len2*sizeof(double));
      create(&gpu_context, &P_g_L, len2*sizeof(double));
      create(&gpu_context, &P_C_m, len2*sizeof(double));
      create(&gpu_context, &P_E_Na, len2*sizeof(double));
      create(&gpu_context, &P_E_K, len2*sizeof(double));
      create(&gpu_context, &P_E_L, len2*sizeof(double ));
      create(&gpu_context, &P_tau_synE, len2*sizeof(double));
      create(&gpu_context, &P_tau_synI, len2*sizeof(double));
      create(&gpu_context, &P_I_e, len2*sizeof(double));
      create(&gpu_context, &B_step_, len2*sizeof(double));
      create(&gpu_context, &B_lag_, len2*sizeof(long));
      create(&gpu_context, &B_sumj_g_ij_, len2*sizeof(double));

      int len3 = /*buffer_size*/ 4 * num_local_nodes;

      h_B_interpolation_coefficients = new double[len3];
      h_new_coefficients = new double[len3];
      h_B_I_stim_ = new double[len2];
      h_B_IntegrationStep_ = new double[len2];
      h_S_y_ = new double[len1];

      create(&gpu_context, &B_interpolation_coefficients, len3*sizeof(double));
      create(&gpu_context, &d_new_coefficients, len3*sizeof(double));
      create(&gpu_context, &B_I_stim_, len2*sizeof(double));
      create(&gpu_context, &B_IntegrationStep_, len2*sizeof(double));
      create(&gpu_context, &S_y_, len1*sizeof(double));
      create(&gpu_context, &d_y_i, len2*sizeof(double));
      create(&gpu_context, &d_f_temp, len1*sizeof(double));
      create(&gpu_context, &d_hf_i, len2*sizeof(double));
      create(&gpu_context, &d_hf_ip1, len2*sizeof(double));
      create(&gpu_context, &B_spike_exc_, len2*sizeof(double));
      create(&gpu_context, &B_spike_inh_, len2*sizeof(double));
      create(&gpu_context, &d_y_ip1, len2*sizeof(double));

      is_initialized = true;
    }
}

#define UPLOAD_2D_DATA(buf, src)		\
  if (node->src != NULL)			\
    for (int j = 0; j < dimension; j++)		\
      buf[j*num_nodes + i] = node->src[j];				

#define UPLOAD_1D_DATA(buf, src)		\
  buf[i] = node->src;							

#define UPLOAD_1D_DATA_VALUE(buf, val)		\
  buf[i] = val;							

#define FINISH_2D_UPLOAD(dst, buf, dim, type)				\
  upload(&gpu_context, (void *)buf, dst, dim*num_nodes*sizeof(type));

#define FINISH_1D_UPLOAD(dst, buf, type)				\
  upload(&gpu_context, (void *)buf, dst, num_nodes*sizeof(type));

void
nest::hh_psc_alpha_gap_gpu::prepare_copy_to_device(std::vector< Node* > &nodes, bool called_from_wfr_update, long lag_)
{
  int num_nodes = nodes.size();

  int wfr_interpolation_order = kernel().simulation_manager.get_wfr_interpolation_order();

  //int len3 = /*buffer_size*/ 4 * num_nodes;
	
  std::vector<Node *>::iterator nodeIt = nodes.begin();

  for (int i = 0; nodeIt != nodes.end(); nodeIt++, i++ )
    {
      nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;
      
      double *tmp = h_B_interpolation_coefficients;// + len3;

      switch (wfr_interpolation_order)
	{
	case 0:
	  tmp[num_nodes*0 + i] = node->B_.interpolation_coefficients[ lag_ ];
	  break;
	case 1:
	  tmp[num_nodes*0 + i] = node->B_.interpolation_coefficients[ lag_ * 2 + 0 ];
	  tmp[num_nodes*1 + i] = node->B_.interpolation_coefficients[ lag_ * 2 + 1 ];
	  break;
	case 3:
	  tmp[num_nodes*0 + i] = node->B_.interpolation_coefficients[ lag_ * 4 + 0 ];
	  tmp[num_nodes*1 + i] = node->B_.interpolation_coefficients[ lag_ * 4 + 1 ];
	  tmp[num_nodes*2 + i] = node->B_.interpolation_coefficients[ lag_ * 4 + 2 ];
	  tmp[num_nodes*3 + i] = node->B_.interpolation_coefficients[ lag_ * 4 + 3 ];
	  break;
	default:
	  break;
	}

      if (called_from_wfr_update)
	{
	  double val = node->B_.spike_exc_.get_value_wfr_update( lag_ ) * node->V_.PSCurrInit_E_;
	  UPLOAD_1D_DATA_VALUE(h_B_spike_exc_, val);
	  val = node->B_.spike_inh_.get_value_wfr_update( lag_ ) * node->V_.PSCurrInit_I_;
	  UPLOAD_1D_DATA_VALUE(h_B_spike_inh_, val);
	}
    }
}

void
nest::hh_psc_alpha_gap_gpu::copy_data_to_device(std::vector< Node* > &nodes, int dimension, bool first_loop, bool called_from_wfr_update, long lag_)
{
  int num_nodes = nodes.size();

  int wfr_interpolation_order = kernel().simulation_manager.get_wfr_interpolation_order();

  //int len3 = /*buffer_size*/ 4 * num_nodes;
	
  std::vector<Node *>::iterator nodeIt = nodes.begin();

  for (int i = 0; nodeIt != nodes.end(); nodeIt++, i++ )
    {

      nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;

      node->B_.lag_ = lag_;
      node->U_old = node->S_.y_[ hh_psc_alpha_gap::State_::V_M ];

      if (first_loop)
	{
	  // UPLOAD_1D_DATA(h_con_state_eps_abs, B_.c_->eps_abs);
	  // UPLOAD_1D_DATA(h_con_state_eps_rel, B_.c_->eps_rel);
	  // UPLOAD_1D_DATA(h_con_state_a_y, B_.c_->a_y);
	  // UPLOAD_1D_DATA(h_con_state_a_dydt, B_.c_->a_dydt);
	  UPLOAD_1D_DATA(h_P_g_Na, P_.g_Na);
	  UPLOAD_1D_DATA(h_P_g_Kv1, P_.g_Kv1);
	  UPLOAD_1D_DATA(h_P_g_Kv3, P_.g_Kv3);
	  UPLOAD_1D_DATA(h_P_g_L, P_.g_L);
	  UPLOAD_1D_DATA(h_P_C_m, P_.C_m);
	  UPLOAD_1D_DATA(h_P_E_Na, P_.E_Na);
	  UPLOAD_1D_DATA(h_P_E_K, P_.E_K);
	  UPLOAD_1D_DATA(h_P_E_L, P_.E_L);
	  UPLOAD_1D_DATA(h_P_tau_synE, P_.tau_synE);
	  UPLOAD_1D_DATA(h_P_tau_synI, P_.tau_synI);
	  UPLOAD_1D_DATA(h_P_I_e, P_.I_e);
	  UPLOAD_1D_DATA(h_B_step_, B_.step_);
	  UPLOAD_1D_DATA(h_B_I_stim_, B_.I_stim_);
	  UPLOAD_1D_DATA(h_B_sumj_g_ij_, B_.sumj_g_ij_);
	  UPLOAD_1D_DATA(h_B_IntegrationStep_, B_.IntegrationStep_);
	}


      if (not called_from_wfr_update || first_loop)
	UPLOAD_2D_DATA(h_S_y_, S_.y_);

    }
  
  if (first_loop)
    {
      FINISH_1D_UPLOAD(con_state_eps_abs, h_con_state_eps_abs, double);
      FINISH_1D_UPLOAD(con_state_eps_rel, h_con_state_eps_rel, double);
      FINISH_1D_UPLOAD(con_state_a_y, h_con_state_a_y, double);
      FINISH_1D_UPLOAD(con_state_a_dydt, h_con_state_a_dydt, double);
      FINISH_1D_UPLOAD(P_g_Na, h_P_g_Na, double);
      FINISH_1D_UPLOAD(P_g_Kv1, h_P_g_Kv1, double);
      FINISH_1D_UPLOAD(P_g_Kv3, h_P_g_Kv3, double);
      FINISH_1D_UPLOAD(P_g_L, h_P_g_L, double);
      FINISH_1D_UPLOAD(P_C_m, h_P_C_m, double);
      FINISH_1D_UPLOAD(P_E_Na, h_P_E_Na, double);
      FINISH_1D_UPLOAD(P_E_K, h_P_E_K, double);
      FINISH_1D_UPLOAD(P_E_L, h_P_E_L, double);
      FINISH_1D_UPLOAD(P_tau_synE, h_P_tau_synE, double);
      FINISH_1D_UPLOAD(P_tau_synI, h_P_tau_synI, double);
      FINISH_1D_UPLOAD(P_I_e, h_P_I_e, double);
      FINISH_1D_UPLOAD(B_step_, h_B_step_, double);
      FINISH_1D_UPLOAD(B_I_stim_, h_B_I_stim_, double);
      FINISH_1D_UPLOAD(B_sumj_g_ij_, h_B_sumj_g_ij_, double);
      FINISH_1D_UPLOAD(B_IntegrationStep_, h_B_IntegrationStep_, double);
    }

  int inter_coeff_size = 0;
  switch (wfr_interpolation_order)
    {
    case 0:
      inter_coeff_size = 1; break;
    case 1:
      inter_coeff_size = 2; break;
    case 3:
      inter_coeff_size = 4; break;
    default:
      break;
    }

  FINISH_2D_UPLOAD(B_interpolation_coefficients, h_B_interpolation_coefficients, inter_coeff_size, double);

  if (not called_from_wfr_update || first_loop)
    FINISH_2D_UPLOAD(S_y_, h_S_y_, dimension, double);

  if (called_from_wfr_update)
    {
      FINISH_1D_UPLOAD(B_spike_exc_, h_B_spike_exc_, double);
      FINISH_1D_UPLOAD(B_spike_inh_, h_B_spike_inh_, double);
    }
}

#define START_2D_DOWNLOAD(src, buf, dim, type)				\
  download(&gpu_context, src, (void *)buf, dim*num_nodes*sizeof(type));

#define DOWNLOAD_2D_DATA(dst, buf)		\
  for (int j = 0; j < dimension; j++)		\
    node->dst[j] = buf[j*num_nodes + i];

#define START_1D_DOWNLOAD(src, buf, type)				\
  download(&gpu_context, src, (void *)buf, num_nodes*sizeof(type));

#define START_1D_DOWNLOAD_OFF(src, buf, type, off)				\
  download(&gpu_context, src, (void *)buf, num_nodes*sizeof(type), off);

#define DOWNLOAD_1D_DATA(dst, buf)		\
  node->dst = buf[i];							

void
nest::hh_psc_alpha_gap_gpu::copy_data_from_device(std::vector< Node* > &nodes, int dimension, bool last_copy, bool called_from_wfr_update)
{
  int num_nodes = nodes.size();

  int wfr_interpolation_order = kernel().simulation_manager.get_wfr_interpolation_order();
  
  std::vector<Node *>::iterator nodeIt = nodes.begin();

  if (last_copy)
    {
      START_1D_DOWNLOAD(B_IntegrationStep_, h_B_IntegrationStep_, double);
    }
  
  int offset = num_nodes * hh_psc_alpha_gap::State_::V_M;
  double *h_S_y_off_ = h_S_y_ + offset;
  if (called_from_wfr_update && !last_copy)
    {
      START_1D_DOWNLOAD_OFF(S_y_, h_S_y_off_, double, offset*sizeof(double));
    }
  else
    {
      START_2D_DOWNLOAD(S_y_, h_S_y_, dimension, double);
    }

  int inter_coeff_size = 0;
  switch (wfr_interpolation_order)
    {
    case 0:
      inter_coeff_size = 1; break;
    case 1:
      inter_coeff_size = 2; break;
    case 3:
      inter_coeff_size = 4; break;
    default:
      break;
    }

  if (called_from_wfr_update)
    {
      START_2D_DOWNLOAD(d_new_coefficients, h_new_coefficients, inter_coeff_size, double);
    }

  synchronize();
  
  for (int i = 0; nodeIt != nodes.end(); nodeIt++, i++)
    {
      nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;

      if (last_copy)
	{
	  DOWNLOAD_1D_DATA(B_.IntegrationStep_, h_B_IntegrationStep_);
	}

      if (called_from_wfr_update && !last_copy)
	{
	  DOWNLOAD_1D_DATA(S_.y_[hh_psc_alpha_gap::State_::V_M], h_S_y_off_);
	}
      else
	{
	  DOWNLOAD_2D_DATA(S_.y_, h_S_y_);
	}
      
      if (called_from_wfr_update)
	{
	  double *tmp = h_new_coefficients;// + len3;

	  node->new_coefficients[ node->B_.lag_ * ( wfr_interpolation_order + 1 ) + 0 ] = tmp[num_nodes*0 + i];
	  switch (wfr_interpolation_order)
	    {
	    case 0:
	      break;
	    case 1:
	      node->new_coefficients[ node->B_.lag_ * ( wfr_interpolation_order + 1 ) + 1 ] = tmp[num_nodes*1 + i];
	      break;
	    case 3:
	      node->new_coefficients[ node->B_.lag_ * ( wfr_interpolation_order + 1 ) + 1 ] = tmp[num_nodes*1 + i];
	      node->new_coefficients[ node->B_.lag_ * ( wfr_interpolation_order + 1 ) + 2 ] = tmp[num_nodes*2 + i];
	      node->new_coefficients[ node->B_.lag_ * ( wfr_interpolation_order + 1 ) + 3 ] = tmp[num_nodes*3 + i];
	      break;
	    default:
	      break;
	    }
	}
    }
}

void
nest::hh_psc_alpha_gap_gpu::create(clContext_ *clCxt, cl::Buffer *mem, int len)
{
  cl_int ret;
  //printf("len %d\n", len);
  if (len == 0)
    return;
  *mem = cl::Buffer(this->context, CL_MEM_READ_WRITE, len, NULL, &ret);
  if(ret != CL_SUCCESS){
    printf("Failed to create buffer on GPU. %d\n", ret);
    return ;
  }
}

void
nest::hh_psc_alpha_gap_gpu::upload(clContext_ *clCxt, void *data, cl::Buffer &gdata, int datalen)
{
  //write data to buffer
  cl_int ret;
  //printf("write datalen %d\n", datalen);
  if (datalen == 0)
    return;
  ret = this->command_queue.enqueueWriteBuffer(gdata, CL_FALSE, 0, datalen, (void *)data, NULL, NULL);
  if(ret != CL_SUCCESS){
    printf("clEnqueueWriteBuffer failed. %d\n", ret);
    return ;
  }
}

void
nest::hh_psc_alpha_gap_gpu::download(clContext_ *clCxt, cl::Buffer &gdata,void *data,int data_len, int offset)
{
  cl_int ret;
  //printf("read datalen %d\n", data_len);
  if (data_len == 0)
    return;
  ret = this->command_queue.enqueueReadBuffer(gdata, CL_FALSE, offset, data_len, (void *)data, NULL,NULL);
  if(ret != CL_SUCCESS){
    printf("clEnqueueReadBuffer failed. error code: %d\n", ret);
    return ;
  }
}

void
nest::hh_psc_alpha_gap_gpu::synchronize()
{
  this->command_queue.finish();
}

int
nest::hh_psc_alpha_gap_gpu::getKernel(string source, string kernelName, string deliver_kernel_name, clContext_ *clCxt)
{
  string kernel_path = "kernel/";
  string src_file = kernel_path + source + ".cl";
  FILE *fs = fopen(src_file.c_str(), "r");
  if (!fs)
    {
      printf("Failed to load kernel file.\n");
      return 1;
    }

  char *source_str = (char *)malloc(MAX_SOURCE_SIZE);
  size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fs);
  fclose(fs);

  // Compile OpenCL program for found device.
  this->program = cl::Program(this->context,
				    cl::Program::Sources(1, std::make_pair(source_str, source_size)));

  try {
    this->program.build(this->devices, "-cl-nv-maxrregcount=200 -cl-nv-verbose");
  } catch (const cl::Error&) {
    std::cerr
      << "OpenCL compilation error" << std::endl
      << this->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->devices[0])
      << std::endl;
    return 1;
  }

  this->gpu_kernel = new cl::Kernel(this->program, kernelName.c_str());

  this->deliver_kernel = new cl::Kernel(this->program, deliver_kernel_name.c_str());

  return 0;
}

// int
// nest::hh_psc_alpha_gap::savebinary(cl_program &program, const char *fileName)
// {
//   size_t binarySize;
//   cl_int ret = clGetProgramInfo(program,
// 				CL_PROGRAM_BINARY_SIZES,
// 				sizeof(size_t),
// 				&binarySize, NULL);
//   if(ret != CL_SUCCESS){
//     printf("Failed to get binary size. error code: %d\n", ret);
//     return 1;
//   }
//   char* binary = (char*)malloc(binarySize);
//   ret = clGetProgramInfo(program,
// 			 CL_PROGRAM_BINARIES,
// 			 sizeof(char *),
// 			 &binary,
// 			 NULL);
//   if(ret != CL_SUCCESS){
//     printf("Failed to get binary. error code: %d\n", ret);
//     return 1;
//   }

//   FILE *fp = fopen(fileName, "wb+");
//   if(fp != NULL)
//     {
//       fwrite(binary, binarySize, 1, fp);
//       fclose(fp);
//     }
//   free(binary);
//   return 0;
// }

#define set_kernel(value)						\
  ret = kernel->setArg(arg_idx, value); \
  if (ret != CL_SUCCESS) {						\
    printf("Failed to set arg %d, error code %d\n", arg_idx, ret);	\
    return 1;								\
  }									\
  arg_idx++;

int
nest::hh_psc_alpha_gap_gpu::set_kernel_args(cl::Kernel *kernel, int num_nodes, int dimension, int wfr_interpolation_order, bool called_from_wfr_update)
{
  cl_int ret;
  int arg_idx = 0;

  ret = kernel->setArg(arg_idx, static_cast<cl_int>(num_nodes)); 
  if (ret != CL_SUCCESS) {						
    printf("Failed to set arg %d, error code %dn", arg_idx, ret);	
    return 1;								
  }									
  arg_idx++;

  ret = kernel->setArg(arg_idx, static_cast<cl_int>(dimension)); 
  if (ret != CL_SUCCESS) {						
    printf("Failed to set arg %d, error code %dn", arg_idx, ret);	
    return 1;								
  }									
  arg_idx++;

  int wfr_update = called_from_wfr_update ? 1 : 0;
  ret = kernel->setArg(arg_idx, static_cast<cl_int>(wfr_update)); 
  if (ret != CL_SUCCESS) {						
    printf("Failed to set arg %d, error code %dn", arg_idx, ret);	
    return 1;								
  }									
  arg_idx++;

  set_kernel(e_y0);
  set_kernel(e_yerr);
  set_kernel(e_dydt_in);
  set_kernel(e_dydt_out);
  // set_kernel(e_last_step);
  // set_kernel(e_count);
  // set_kernel(e_failed_steps);
  set_kernel(con_state_eps_abs);
  set_kernel(con_state_eps_rel);
  set_kernel(con_state_a_y);
  set_kernel(con_state_a_dydt);
  set_kernel(rk_state_k1);
  set_kernel(rk_state_k2);
  set_kernel(rk_state_k3);
  set_kernel(rk_state_k4);
  set_kernel(rk_state_k5);
  set_kernel(rk_state_k6);
  set_kernel(rk_state_y0);
  set_kernel(rk_state_ytmp);
  set_kernel(P_g_Na);
  set_kernel(P_g_Kv1);
  set_kernel(P_g_Kv3);
  set_kernel(P_g_L);
  set_kernel(P_C_m);
  set_kernel(P_E_Na);
  set_kernel(P_E_K);
  set_kernel(P_E_L);
  set_kernel(P_tau_synE);
  set_kernel(P_tau_synI);
  set_kernel(P_I_e);
  //set_kernel(cl_int, wfr_interpolation_order);
  ret = kernel->setArg(arg_idx, static_cast<cl_int>(wfr_interpolation_order)); 
  if (ret != CL_SUCCESS) {						
    printf("Failed to set arg %d, error code %dn", arg_idx, ret);	
    return 1;								
  }									
  arg_idx++;

  set_kernel(B_step_);
  //set_kernel(B_lag_);
  set_kernel(B_sumj_g_ij_);
  set_kernel(B_interpolation_coefficients);
  set_kernel(d_new_coefficients);
  set_kernel(B_I_stim_);
  //set_kernel(d_y_i);
  set_kernel(d_f_temp);
  // set_kernel(d_hf_i);
  // set_kernel(d_hf_ip1);
  // set_kernel(d_y_ip1);
  set_kernel(B_spike_exc_);
  set_kernel(B_spike_inh_);
  set_kernel(B_IntegrationStep_);
  set_kernel(S_y_);
  //set_kernel(d_U_old);

  return 0;
}

int
nest::hh_psc_alpha_gap_gpu::set_deliver_kernel_args(cl::Kernel *kernel, int num_nodes)
{
  cl_int ret;
  int arg_idx = 0;

  ret = kernel->setArg(arg_idx, static_cast<cl_int>(num_nodes)); 
  if (ret != CL_SUCCESS) {						
    printf("Failed to set arg %d, error code %dn", arg_idx, ret);	
    return 1;								
  }									
  arg_idx++;

  ret = kernel->setArg(arg_idx, static_cast<cl_int>(event_size)); 
  if (ret != CL_SUCCESS) {						
    printf("Failed to set arg %d, error code %dn", arg_idx, ret);	
    return 1;								
  }									
  arg_idx++;

  set_kernel(d_connections_ptr);
  set_kernel(d_connections);
  set_kernel(d_event_buffer);
  set_kernel(d_event_weight);
  set_kernel(d_coeff_buffer);
  set_kernel(B_sumj);

  return 0;
}

void
nest::hh_psc_alpha_gap_gpu::execute_kernel(cl::Kernel *kernel, clContext_ *clCxt, size_t num_nodes)
{
  cl_int ret;
  size_t local_work_size = 128;
  //num_nodes /= 2;
  size_t global_work_size = num_nodes % local_work_size == 0 ? num_nodes : (num_nodes / local_work_size + 1) * local_work_size;
  
  // size_t globalthreads[1] = {global_work_size};
  // size_t localthreads[1] = {local_work_size};

  ret = this->command_queue.enqueueNDRangeKernel(*(kernel), cl::NullRange, global_work_size, local_work_size);

  if (ret != CL_SUCCESS)
    {
      printf("Failed to EnqueueNDRangeKernel. error code: %d\n", ret);
      return;
    }

  //clFinish(this->command_queue);
}

void
nest::hh_psc_alpha_gap_gpu::fill_event_buffer( SecondaryEvent& e)
{
  GapJunctionEvent *gap_event = (GapJunctionEvent*)&e;
  index sgid = gap_event->get_sender_gid() - 1;
  double *h_sgid_event_buffer = h_event_buffer + event_size * sgid;

  h_event_weight[sgid] = e.get_weight();

  size_t i = 0;
  std::vector< unsigned int >::iterator it = gap_event->begin();
  // The call to get_coeffvalue( it ) in this loop also advances the iterator it
  while ( it != gap_event->end() )
    {
      h_sgid_event_buffer[ i ] =
	gap_event->get_weight() * gap_event->get_coeffvalue( it );
      ++i;
    }
}

void
nest::hh_psc_alpha_gap_gpu::fill_buffer_zero(clContext_ *clCxt, cl::Buffer &buffer, int size)
{
  cl_double zero = 0.0;
  cl_int ret;

  ret = this->command_queue.enqueueFillBuffer(buffer, zero, 0, size);
  
  if (ret != CL_SUCCESS)
    {
      printf("Failed to EnqueueNDRangeKernel. error code: %d\n", ret);
      return;
    }
}

void
nest::hh_psc_alpha_gap_gpu::initialize_graph()
{
  for (vector<vector<size_t> >::iterator tgt_it = connections.begin();
       tgt_it != connections.end();
       tgt_it++)
    {
      graph_size += (*tgt_it).size();
    }

  size_t num_nodes = kernel().node_manager.size();
  h_connections_ptr = new int[num_nodes + 1];
  h_connections = new int[graph_size];
  create(&gpu_context, &d_connections_ptr, (num_nodes + 1) * sizeof(int));
  create(&gpu_context, &d_connections, graph_size * sizeof(int));

  create(&gpu_context, &d_event_buffer, num_nodes * event_size * sizeof(double));
  create(&gpu_context, &d_coeff_buffer, num_nodes * event_size * sizeof(double));
  create(&gpu_context, &d_event_weight, num_nodes * sizeof(double));
  create(&gpu_context, &B_sumj, num_nodes*sizeof(double));

  int i = 0;
  int node_id = 0;
  for (vector<vector<size_t> >::iterator tgt_it = connections.begin();
       tgt_it != connections.end();
       tgt_it++, node_id++)
    {
      h_connections_ptr[node_id] = i;

      for (vector<size_t>::iterator src_it = (*tgt_it).begin();
	   src_it != (*tgt_it).end();
	   src_it++, i++)
	h_connections[i] = *src_it;
    }

  for (size_t i = connections.size(); i <= num_nodes; i++)
    h_connections_ptr[i] = graph_size;
  // h_connections_ptr[connections.size()] = graph_size;
  // h_connections_ptr[num_nodes] = graph_size;
  synchronize();

  upload(&gpu_context, h_connections_ptr, d_connections_ptr, (num_nodes + 1) * sizeof(int));
  upload(&gpu_context, h_connections, d_connections, graph_size * sizeof(int));

  synchronize();

}

void
nest::hh_psc_alpha_gap_gpu::upload_event_data(int num_nodes, int buffer_size)
{
  fill_buffer_zero(&gpu_context, d_coeff_buffer, buffer_size);
  fill_buffer_zero(&gpu_context, B_sumj, num_nodes * sizeof(double));
  upload(&gpu_context, (void*)h_event_buffer, d_event_buffer, buffer_size);
  upload(&gpu_context, (void*)h_event_weight, d_event_weight, num_nodes * sizeof(double));
}

void
nest::hh_psc_alpha_gap_gpu::download_event_data(int num_nodes, int buffer_size)
{
  download(&gpu_context, d_coeff_buffer, (void*)h_coeff_buffer, buffer_size);
  download(&gpu_context, B_sumj, (void*)h_B_sumj, num_nodes * sizeof(double));
}

void
nest::hh_psc_alpha_gap_gpu::deliver_events()
{
  size_t num_nodes = kernel().node_manager.size();
  int buffer_size = num_nodes * event_size * sizeof(double);
  
  set_deliver_kernel_args(this->deliver_kernel, num_nodes);

  upload_event_data(num_nodes, buffer_size);
  synchronize();

#ifdef PROFILING
  struct timeval start_time, end_time, diff_time;
  gettimeofday(&start_time, NULL);
#endif

  execute_kernel(this->deliver_kernel, &gpu_context, num_nodes);
  synchronize();

#ifdef PROFILING
  gettimeofday(&end_time, NULL);
  timersub(&end_time, &start_time, &diff_time);
  double diff = (double)diff_time.tv_sec*1000 + (double)diff_time.tv_usec/1000;
  printf("Deliver kernel: %0.3f\n", diff);
#endif

  download_event_data(num_nodes, buffer_size);
  synchronize();
}

void
nest::hh_psc_alpha_gap_gpu::copy_event_data(std::vector<Node *> nodes)
{
  
  int nodeid = 0;
  double *coeff_ptr = h_coeff_buffer;
  for ( std::vector<Node*>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); nodeIt++)
    {
      nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;
      nodeid = node->get_gid() - 1;
      coeff_ptr = h_coeff_buffer + event_size * nodeid;

      node->B_.sumj_g_ij_ = h_B_sumj[nodeid];
      for (size_t i = 0; i < event_size; i++)
	{
	  node->B_.interpolation_coefficients[i] = coeff_ptr[i];
	}
    
      //coeff_ptr += event_size;
    }
}

void
nest::hh_psc_alpha_gap_gpu::handle(int sgid, int tgid)
{
  connections[tgid].push_back(sgid);
}
