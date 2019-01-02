#include "hh_cond_exp_traub_gpu.h"
#include "hh_cond_exp_traub.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

#ifdef PROFILING
#include <sys/time.h>
#endif

nest::hh_cond_exp_traub_gpu::clContext_ nest::hh_cond_exp_traub_gpu::gpu_context;
bool nest::hh_cond_exp_traub_gpu::is_gpu_initialized = false;

nest::hh_cond_exp_traub_gpu::hh_cond_exp_traub_gpu()
  : is_data_ready( false )
  , event_size( 0 )
  , ring_buffer_size( 0 )
  , is_initialized( false )
  , is_ring_buffer_ready( false )
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
 , h_P__E_in( NULL )
 , h_P__tau_synE( NULL )
 , h_P__E_Na( NULL )
 , h_P__C_m( NULL )
 , h_B__step_( NULL )
 , h_P__tau_synI( NULL )
 , h_B__I_stim_( NULL )
 , h_B__IntegrationStep_( NULL )
 , h_P__g_K( NULL )
 , h_P__E_ex( NULL )
 , h_P__V_T( NULL )
 , h_P__g_L( NULL )
 , h_V__U_old_( NULL )
 , h_S__r_( NULL )
 , h_V__refractory_counts_( NULL )
 , h_P__E_L( NULL )
 , h_P__E_K( NULL )
 , h_S__y_( NULL )
 , h_P__I_e( NULL )
 , h_P__g_Na( NULL )
  , h_B_step_( NULL )
  , h_B_IntegrationStep_( NULL )
    , h_currents_( NULL )
, h_spike_exc_( NULL )
, h_spike_inh_( NULL )
  , h_spike_count (NULL)
  , h_history_ptr (NULL)
  , h_Kminus_ (NULL)
  , h_tau_minus_inv_ (NULL)

{
}

nest::hh_cond_exp_traub_gpu::~hh_cond_exp_traub_gpu()
{
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
if (h_P__E_in) delete[] h_P__E_in;
if (h_P__tau_synE) delete[] h_P__tau_synE;
if (h_P__E_Na) delete[] h_P__E_Na;
if (h_P__C_m) delete[] h_P__C_m;
if (h_B__step_) delete[] h_B__step_;
if (h_P__tau_synI) delete[] h_P__tau_synI;
if (h_B__I_stim_) delete[] h_B__I_stim_;
if (h_B__IntegrationStep_) delete[] h_B__IntegrationStep_;
if (h_P__g_K) delete[] h_P__g_K;
if (h_P__E_ex) delete[] h_P__E_ex;
if (h_P__V_T) delete[] h_P__V_T;
if (h_P__g_L) delete[] h_P__g_L;
if (h_V__U_old_) delete[] h_V__U_old_;
if (h_S__r_) delete[] h_S__r_;
if (h_V__refractory_counts_) delete[] h_V__refractory_counts_;
if (h_P__E_L) delete[] h_P__E_L;
if (h_P__E_K) delete[] h_P__E_K;
if (h_S__y_) delete[] h_S__y_;
if (h_P__I_e) delete[] h_P__I_e;
if (h_P__g_Na) delete[] h_P__g_Na;
  if (h_B_step_) delete[] h_B_step_;
  if (h_B_IntegrationStep_) delete[] h_B_IntegrationStep_;
  if (h_spike_count) delete[] h_spike_count;
  if (h_currents_) delete[] h_currents_;
if (h_spike_exc_) delete[] h_spike_exc_;
if (h_spike_inh_) delete[] h_spike_inh_;
}

void
nest::hh_cond_exp_traub_gpu::initialize_gpu()
{
  if (not is_gpu_initialized)
    {
      if (initialize_opencl_context())
	return;
      is_gpu_initialized = true;
    }
}

void
nest::hh_cond_exp_traub_gpu::mass_update(const std::vector< Node* > &nodes, Time const& origin,
					const long from,
					const long to )
{
  mass_update_(nodes, origin, from, to, false );
}

bool
nest::hh_cond_exp_traub_gpu::mass_wfr_update(const std::vector< Node* > &nodes, Time const& origin,
					    const long from,
					    const long to )
{
  return mass_update_(nodes, origin, from, to, true );
}

// TODO: the for loops will later be kernel calls
bool
nest::hh_cond_exp_traub_gpu::mass_update_( const std::vector<Node *> &nodes,
					  Time const& origin,
					  const long from,
					  const long to,
					  const bool called_from_wfr_update ) // TODO: don't know yet whether we need to cover both cases here
{
  // TODO: do AoS for now, SoA will come later on
  int total_num_nodes = kernel().node_manager.size();
  int num_local_nodes = nodes.size();

#ifdef PROFILING
  int thrd_id = kernel().vp_manager.get_thread_id();
#endif  

  initialize_device(hh_cond_exp_traub::State_::STATE_VEC_SIZE);
  set_kernel_args(this->gpu_kernel, num_local_nodes, hh_cond_exp_traub::State_::STATE_VEC_SIZE);

  // TODO: for now, I assume that these are the same for all nodes

  if (not is_data_ready)
    {
      copy_data_to_device(nodes, hh_cond_exp_traub::State_::STATE_VEC_SIZE);
      is_data_ready = true;
    }

  // if (from < to)
  //   prepare_copy_to_device(nodes, called_from_wfr_update, from);
  for ( long lag = from; lag < to; ++lag)
    {
  //     for ( std::vector<Node*>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); nodeIt++ )
  // 	{
  // 	  nest::hh_cond_exp_traub* node = (nest::hh_cond_exp_traub*)*nodeIt;
  // 	  node->pre_gsl(origin, lag);
  // 	}

      set_lag_args(this->gpu_kernel, lag);
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
    
      fill_buffer_zero_uint(&gpu_context, d_spike_count, this->num_local_nodes*sizeof(unsigned int));      
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

      execute_kernel(this->gpu_kernel, &gpu_context, this->num_local_nodes);

      // if (lag + 1 < to)
      // 	prepare_copy_to_device(nodes, called_from_wfr_update, lag + 1);

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
    
      copy_data_from_device(nodes, hh_cond_exp_traub::State_::STATE_VEC_SIZE, false);

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

      int node_id = 0;
      for ( std::vector<Node*>::const_iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); nodeIt++, node_id++ )
	{
	  nest::hh_cond_exp_traub* node = (nest::hh_cond_exp_traub*)*nodeIt;
	  //node->post_gsl(origin, lag);

	  if (node_id >= this->num_local_nodes - 2)
	    {
	      break;
	    }
	  for (size_t i = 0; i < h_spike_count[node_id]; i++)
	    {
	      node->set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );
	      SpikeEvent se;
	      kernel().event_delivery_manager.send( *node, se, lag );
	    }
	}
    }

  for (int nodeid = this->num_local_nodes - 2; nodeid < nodes.size(); nodeid++)
    if (called_from_wfr_update)
      nodes[nodeid]->wfr_update(origin, from, to);
    else
      nodes[nodeid]->update(origin, from, to);

  copy_data_from_device(nodes, hh_cond_exp_traub::State_::STATE_VEC_SIZE, true);
  return true;
}

int
nest::hh_cond_exp_traub_gpu::initialize_opencl_context()
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
nest::hh_cond_exp_traub_gpu::initialize_command_queue()
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
nest::hh_cond_exp_traub_gpu::initialize_device(int dimension)
{
  if (!is_initialized)
    {
      printf("initialize_device %d\n", this->num_local_nodes);
      
      int len1 = this->num_local_nodes;
      int len2 = dimension*this->num_local_nodes;
      
      h_e_y0 = new double[len2];
      h_e_yerr = new double[len2];
      h_e_dydt_in = new double[len2];
      h_e_dydt_out = new double[len2];
      h_con_state_eps_abs = new double[len1];
      h_con_state_eps_rel = new double[len1];
      h_con_state_a_y = new double[len1];
      h_con_state_a_dydt = new double[len1];
      h_rk_state_k1 = new double[len2];
      h_rk_state_k2 = new double[len2];
      h_rk_state_k3 = new double[len2];
      h_rk_state_k4 = new double[len2];
      h_rk_state_k5 = new double[len2];
      h_rk_state_k6 = new double[len2];
      h_rk_state_y0 = new double[len2];
      h_rk_state_ytmp = new double[len2];
      h_P__E_in = new double[len1];
      h_P__tau_synE = new double[len1];
      h_P__E_Na = new double[len1];
      h_P__C_m = new double[len1];
      h_B__step_ = new double[len1];
      h_P__tau_synI = new double[len1];
      h_B__I_stim_ = new double[len1];
      h_B__IntegrationStep_ = new double[len1];
      h_P__g_K = new double[len1];
      h_P__E_ex = new double[len1];
      h_P__V_T = new double[len1];
      h_P__g_L = new double[len1];
      h_V__U_old_ = new double[len1];
      h_S__r_ = new int[len1];
      h_V__refractory_counts_ = new int[len1];
      h_P__E_L = new double[len1];
      h_P__E_K = new double[len1];
      h_S__y_ = new double[len1 *6];
      h_P__I_e = new double[len1];
      h_P__g_Na = new double[len1];
      h_B_step_ = new double[len1];
      h_B_IntegrationStep_ = new double[len1];
      h_spike_count = new unsigned int[len1];

      create(&gpu_context, &e_y0, len2*sizeof(double));
      create(&gpu_context, &e_yerr, len2*sizeof(double));
      create(&gpu_context, &e_dydt_in, len2*sizeof(double));
      create(&gpu_context, &e_dydt_out, len2*sizeof(double));
      create(&gpu_context, &con_state_eps_abs, len1*sizeof(double));
      create(&gpu_context, &con_state_eps_rel, len1*sizeof(double));
      create(&gpu_context, &con_state_a_y, len1*sizeof(double));
      create(&gpu_context, &con_state_a_dydt, len1*sizeof(double));
      create(&gpu_context, &rk_state_k1, len2*sizeof(double));
      create(&gpu_context, &rk_state_k2, len2*sizeof(double));
      create(&gpu_context, &rk_state_k3, len2*sizeof(double));
      create(&gpu_context, &rk_state_k4, len2*sizeof(double));
      create(&gpu_context, &rk_state_k5, len2*sizeof(double));
      create(&gpu_context, &rk_state_k6, len2*sizeof(double));
      create(&gpu_context, &rk_state_y0, len2*sizeof(double));
      create(&gpu_context, &rk_state_ytmp, len2*sizeof(double));
      create(&gpu_context, &P__E_in, len1*sizeof(double));
      create(&gpu_context, &P__tau_synE, len1*sizeof(double));
      create(&gpu_context, &P__E_Na, len1*sizeof(double));
      create(&gpu_context, &P__C_m, len1*sizeof(double));
      create(&gpu_context, &B__step_, len1*sizeof(double));
      create(&gpu_context, &P__tau_synI, len1*sizeof(double));
      create(&gpu_context, &B__I_stim_, len1*sizeof(double));
      create(&gpu_context, &B__IntegrationStep_, len1*sizeof(double));
      create(&gpu_context, &P__g_K, len1*sizeof(double));
      create(&gpu_context, &P__E_ex, len1*sizeof(double));
      create(&gpu_context, &P__V_T, len1*sizeof(double));
      create(&gpu_context, &P__g_L, len1*sizeof(double));
      create(&gpu_context, &V__U_old_, len1*sizeof(double));
      create(&gpu_context, &S__r_, len1*sizeof(int));
      create(&gpu_context, &V__refractory_counts_, len1*sizeof(int));
      create(&gpu_context, &P__E_L, len1*sizeof(double));
      create(&gpu_context, &P__E_K, len1*sizeof(double));
      create(&gpu_context, &S__y_, len1*6*sizeof(double));
      create(&gpu_context, &P__I_e, len1*sizeof(double));
      create(&gpu_context, &P__g_Na, len1*sizeof(double));
      create(&gpu_context, &B_step_, len1*sizeof(double));
      create(&gpu_context, &d_spike_count, len1*sizeof(double));
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

// void
// nest::hh_cond_exp_traub_gpu::prepare_copy_to_device(std::vector< Node* > &nodes, bool called_from_wfr_update, long lag_)
// {
  // int num_nodes = nodes.size();

  // int wfr_interpolation_order = kernel().simulation_manager.get_wfr_interpolation_order();

  // //int len3 = /*buffer_size*/ 4 * num_nodes;
	
  // std::vector<Node *>::iterator nodeIt = nodes.begin();

  // for (int i = 0; nodeIt != nodes.end(); nodeIt++, i++ )
  //   {
  //     nest::hh_cond_exp_traub* node = (nest::hh_cond_exp_traub*)*nodeIt;
      
  //     double *tmp = h_B_interpolation_coefficients;// + len3;

  //     switch (wfr_interpolation_order)
  // 	{
  // 	case 0:
  // 	  tmp[num_nodes*0 + i] = node->B_.interpolation_coefficients[ lag_ ];
  // 	  break;
  // 	case 1:
  // 	  tmp[num_nodes*0 + i] = node->B_.interpolation_coefficients[ lag_ * 2 + 0 ];
  // 	  tmp[num_nodes*1 + i] = node->B_.interpolation_coefficients[ lag_ * 2 + 1 ];
  // 	  break;
  // 	case 3:
  // 	  tmp[num_nodes*0 + i] = node->B_.interpolation_coefficients[ lag_ * 4 + 0 ];
  // 	  tmp[num_nodes*1 + i] = node->B_.interpolation_coefficients[ lag_ * 4 + 1 ];
  // 	  tmp[num_nodes*2 + i] = node->B_.interpolation_coefficients[ lag_ * 4 + 2 ];
  // 	  tmp[num_nodes*3 + i] = node->B_.interpolation_coefficients[ lag_ * 4 + 3 ];
  // 	  break;
  // 	default:
  // 	  break;
  // 	}

  //     if (called_from_wfr_update)
  // 	{
  // 	  double val = node->B_.spike_exc_.get_value_wfr_update( lag_ ) * node->V_.PSCurrInit_E_;
  // 	  UPLOAD_1D_DATA_VALUE(h_B_spike_exc_, val);
  // 	  val = node->B_.spike_inh_.get_value_wfr_update( lag_ ) * node->V_.PSCurrInit_I_;
  // 	  UPLOAD_1D_DATA_VALUE(h_B_spike_inh_, val);
  // 	}
  //   }
// }

void
nest::hh_cond_exp_traub_gpu::copy_data_to_device(const std::vector< Node* > &nodes, int dimension)
{
  int num_nodes = nodes.size();

  std::vector<Node *>::const_iterator nodeIt = nodes.begin();

  for (int i = 0; nodeIt != nodes.end(); nodeIt++, i++ )
    {

      nest::hh_cond_exp_traub* node = (nest::hh_cond_exp_traub*)*nodeIt;

UPLOAD_1D_DATA(h_P__E_in, P_.E_in);
UPLOAD_1D_DATA(h_P__tau_synE, P_.tau_synE);
UPLOAD_1D_DATA(h_P__E_Na, P_.E_Na);
UPLOAD_1D_DATA(h_P__C_m, P_.C_m);
UPLOAD_1D_DATA(h_B__step_, B_.step_);
UPLOAD_1D_DATA(h_P__tau_synI, P_.tau_synI);
UPLOAD_1D_DATA(h_B__I_stim_, B_.I_stim_);
UPLOAD_1D_DATA(h_B__IntegrationStep_, B_.IntegrationStep_);
UPLOAD_1D_DATA(h_P__g_K, P_.g_K);
UPLOAD_1D_DATA(h_P__E_ex, P_.E_ex);
UPLOAD_1D_DATA(h_P__V_T, P_.V_T);
UPLOAD_1D_DATA(h_P__g_L, P_.g_L);
UPLOAD_1D_DATA(h_V__U_old_, V_.U_old_);
UPLOAD_1D_DATA(h_S__r_, S_.r_);
UPLOAD_1D_DATA(h_V__refractory_counts_, V_.refractory_counts_);
UPLOAD_1D_DATA(h_P__E_L, P_.E_L);
UPLOAD_1D_DATA(h_P__E_K, P_.E_K);
UPLOAD_2D_DATA(h_S__y_, S_.y_);
UPLOAD_1D_DATA(h_P__I_e, P_.I_e);
UPLOAD_1D_DATA(h_P__g_Na, P_.g_Na);
      UPLOAD_1D_DATA(h_B_step_, B_.step_);
      UPLOAD_1D_DATA(h_B_IntegrationStep_, B_.IntegrationStep_);
      
      /* DEVICE OUTPUT VAR UPLOAD */

    }
  
FINISH_1D_UPLOAD(P__E_in, h_P__E_in, double);
FINISH_1D_UPLOAD(P__tau_synE, h_P__tau_synE, double);
FINISH_1D_UPLOAD(P__E_Na, h_P__E_Na, double);
FINISH_1D_UPLOAD(P__C_m, h_P__C_m, double);
FINISH_1D_UPLOAD(B__step_, h_B__step_, double);
FINISH_1D_UPLOAD(P__tau_synI, h_P__tau_synI, double);
FINISH_1D_UPLOAD(B__I_stim_, h_B__I_stim_, double);
FINISH_1D_UPLOAD(B__IntegrationStep_, h_B__IntegrationStep_, double);
FINISH_1D_UPLOAD(P__g_K, h_P__g_K, double);
FINISH_1D_UPLOAD(P__E_ex, h_P__E_ex, double);
FINISH_1D_UPLOAD(P__V_T, h_P__V_T, double);
FINISH_1D_UPLOAD(P__g_L, h_P__g_L, double);
FINISH_1D_UPLOAD(V__U_old_, h_V__U_old_, double);
FINISH_1D_UPLOAD(S__r_, h_S__r_, int);
FINISH_1D_UPLOAD(V__refractory_counts_, h_V__refractory_counts_, int);
FINISH_1D_UPLOAD(P__E_L, h_P__E_L, double);
FINISH_1D_UPLOAD(P__E_K, h_P__E_K, double);
FINISH_2D_UPLOAD(S__y_, h_S__y_, 6, double);
FINISH_1D_UPLOAD(P__I_e, h_P__I_e, double);
FINISH_1D_UPLOAD(P__g_Na, h_P__g_Na, double);
      FINISH_1D_UPLOAD(B_step_, h_B_step_, double);
      FINISH_1D_UPLOAD(B_IntegrationStep_, h_B_IntegrationStep_, double);

    /* DEVICE OUTPUT VAR FINISH UPLOAD */
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
nest::hh_cond_exp_traub_gpu::copy_data_from_device(const std::vector< Node* > &nodes, int dimension, bool last_copy)
{
  int num_nodes = nodes.size();

  std::vector<Node *>::const_iterator nodeIt = nodes.begin();

  // if (last_copy)
  //   {
  //     START_1D_DOWNLOAD(B_IntegrationStep_, h_B_IntegrationStep_, double);
  //     /* DEVICE OUTPUT VAR DOWNLOAD */
  //   }
    
  START_1D_DOWNLOAD(d_spike_count, h_spike_count, unsigned int);
  
  synchronize();
  
  // for (int i = 0; nodeIt != nodes.end(); nodeIt++, i++)
  //   {
  //     nest::hh_cond_exp_traub* node = (nest::hh_cond_exp_traub*)*nodeIt;

  //     if (last_copy)
  // 	{
  // 	  DOWNLOAD_1D_DATA(B_.IntegrationStep_, h_B_IntegrationStep_);
  // 	}

  //     /* DEVICE OUTPUT VAR FINISH DOWNLOAD */
  //   }
}

void
nest::hh_cond_exp_traub_gpu::create(clContext_ *clCxt, cl::Buffer *mem, int len)
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
nest::hh_cond_exp_traub_gpu::upload(clContext_ *clCxt, void *data, cl::Buffer &gdata, int datalen)
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
nest::hh_cond_exp_traub_gpu::download(clContext_ *clCxt, cl::Buffer &gdata,void *data,int data_len, int offset)
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
nest::hh_cond_exp_traub_gpu::synchronize()
{
  this->command_queue.finish();
}

int
nest::hh_cond_exp_traub_gpu::getKernel(string source, string kernelName, string deliver_kernel_name, clContext_ *clCxt)
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


#define set_kernel(value)						\
  ret = kernel->setArg(arg_idx, value);					\
  if (ret != CL_SUCCESS) {						\
    printf("Failed to set arg %d, error code %d\n", arg_idx, ret);	\
    return 1;								\
  }									\
  arg_idx++;

#define set_kernel_prim(value, type)					\
  ret = kernel->setArg(arg_idx, static_cast<type>(value));		\
  if (ret != CL_SUCCESS) {						\
    printf("Failed to set arg %d, error code %d\n", arg_idx, ret);	\
    return 1;								\
  }									\
  arg_idx++;

int
nest::hh_cond_exp_traub_gpu::set_lag_args(cl::Kernel *kernel, long lag)
{
  cl_int ret;
  ret = kernel->setArg(2, static_cast<cl_long>(lag));
  if (ret != CL_SUCCESS) {						
    printf("Failed to set lag arg, error code %dn", ret);	
    return 1;								
  }									
  return 0;
}

int
nest::hh_cond_exp_traub_gpu::set_kernel_args(cl::Kernel *kernel, int num_nodes, int dimension)
{
  cl_int ret;
  int arg_idx = 0;

  set_kernel_prim(num_nodes, cl_int);
  set_kernel_prim(dimension, cl_int);

  arg_idx++; //lag

  set_kernel_prim(event_size, cl_int);
  set_kernel(e_y0);
  set_kernel(e_yerr);
  set_kernel(e_dydt_in);
  set_kernel(e_dydt_out);
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
  set_kernel(P__E_in);
  set_kernel(P__tau_synE);
  set_kernel(P__E_Na);
  set_kernel(P__C_m);
  set_kernel(B__step_);
  set_kernel(P__tau_synI);
  set_kernel(B__I_stim_);
  set_kernel(B__IntegrationStep_);
  set_kernel(P__g_K);
  set_kernel(P__E_ex);
  set_kernel(P__V_T);
  set_kernel(P__g_L);
  set_kernel(V__U_old_);
  set_kernel(S__r_);
  set_kernel(V__refractory_counts_);
  set_kernel(P__E_L);
  set_kernel(P__E_K);
  set_kernel(P__I_e);
  set_kernel(P__g_Na);
  set_kernel(d_currents_);
  set_kernel(d_spike_exc_);
  set_kernel(d_spike_inh_);
  set_kernel(B_step_);
  set_kernel(B_IntegrationStep_);
  set_kernel(d_spike_count);
    set_kernel(S__y_);
  set_kernel_prim(time_index, cl_int);

  return 0;
}

int
nest::hh_cond_exp_traub_gpu::set_deliver_kernel_args(cl::Kernel *kernel, int num_nodes, int batch_size)
{
  cl_int ret;
  int arg_idx = 0;

  set_kernel_prim(num_nodes, cl_int);
  set_kernel_prim(batch_size, cl_int);
  set_kernel_prim(event_size, cl_int);
  set_kernel(d_history_ptr);
  set_kernel(d_history_Kminus_);
  set_kernel(d_history_t_);
  set_kernel(d_history_access_counter_);
  set_kernel(d_Kminus_);
  set_kernel(d_tau_minus_inv_);
  set_kernel(d_spike_tgid);
  set_kernel_prim(event_t_spike, cl_double);
  set_kernel_prim(event_dendritic_delay, cl_double);
  set_kernel(d_weight_);
  set_kernel(d_Kplus_);
  set_kernel(d_conn_type_);
  set_kernel(d_t_lastspike);
  set_kernel(d_pos);
  set_kernel(d_multiplicity);
  set_kernel_prim(event_multiplicity, cl_int);
  int update_type = nest::kernel().simulation_manager.update_type;
  set_kernel_prim(update_type, cl_int);
  set_kernel_prim(conn_type, cl_int);
  set_kernel_prim(cp_lambda_, cl_double);
  set_kernel_prim(cp_mu_, cl_double);
  set_kernel_prim(cp_alpha_, cl_double);
  set_kernel_prim(cp_tau_plus_inv_, cl_double);
  set_kernel(d_spike_exc_);
  set_kernel(d_spike_inh_);
  set_kernel_prim(time_index, cl_int);
  
  return 0;
}

void
nest::hh_cond_exp_traub_gpu::execute_kernel(cl::Kernel *kernel, clContext_ *clCxt, size_t num_nodes)
{
  cl_int ret;
  size_t local_work_size = 128;
  size_t global_work_size = num_nodes % local_work_size == 0 ? num_nodes : (num_nodes / local_work_size + 1) * local_work_size;
  
  ret = this->command_queue.enqueueNDRangeKernel(*(kernel), cl::NullRange, global_work_size, local_work_size);

  if (ret != CL_SUCCESS)
    {
      printf("Failed to EnqueueNDRangeKernel. error code: %d\n", ret);
      return;
    }

  //clFinish(this->command_queue);
}

void
nest::hh_cond_exp_traub_gpu::fill_spike_event_buffer( Event& e)
{
}

void
nest::hh_cond_exp_traub_gpu::fill_event_buffer( SecondaryEvent& e)
{
  // GapJunctionEvent *gap_event = (GapJunctionEvent*)&e;
  // index sgid = gap_event->get_sender_gid() - 1;
  // double *h_sgid_event_buffer = h_event_buffer + event_size * sgid;

  // h_event_weight[sgid] = e.get_weight();

  // size_t i = 0;
  // std::vector< unsigned int >::iterator it = gap_event->begin();
  // // The call to get_coeffvalue( it ) in this loop also advances the iterator it
  // while ( it != gap_event->end() )
  //   {
  //     h_sgid_event_buffer[ i ] =
  // 	gap_event->get_weight() * gap_event->get_coeffvalue( it );
  //     ++i;
  //   }
}

void
nest::hh_cond_exp_traub_gpu::fill_buffer_zero_double(clContext_ *clCxt, cl::Buffer &buffer, int size)
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
nest::hh_cond_exp_traub_gpu::fill_buffer_zero_uint(clContext_ *clCxt, cl::Buffer &buffer, int size)
{
  cl_uint zero = 0;
  cl_int ret;

  ret = this->command_queue.enqueueFillBuffer(buffer, zero, 0, size);
  
  if (ret != CL_SUCCESS)
    {
      printf("Failed to enqueueFillBuffer. error code: %d\n", ret);
      return;
    }
}


void
nest::hh_cond_exp_traub_gpu::initialize()
{
  if (not is_ring_buffer_ready)
    {

      if (initialize_command_queue())
	return;

      int total_num_nodes = kernel().node_manager.size();

      event_size = kernel().connection_manager.get_min_delay()
        + kernel().connection_manager.get_max_delay();
      ring_buffer_size = total_num_nodes * event_size;

      h_currents_ = new double[ring_buffer_size];
h_spike_exc_ = new double[ring_buffer_size];
h_spike_inh_ = new double[ring_buffer_size];

      create(&gpu_context, &d_currents__buf, ring_buffer_size*sizeof(double));
create(&gpu_context, &d_spike_exc__buf, ring_buffer_size*sizeof(double));
create(&gpu_context, &d_spike_inh__buf, ring_buffer_size*sizeof(double));
create(&gpu_context, &d_currents_, ring_buffer_size*sizeof(double));
create(&gpu_context, &d_spike_exc_, ring_buffer_size*sizeof(double));
create(&gpu_context, &d_spike_inh_, ring_buffer_size*sizeof(double));
fill_buffer_zero_double(&gpu_context, d_currents__buf, ring_buffer_size * sizeof(double));
fill_buffer_zero_double(&gpu_context, d_spike_exc__buf, ring_buffer_size * sizeof(double));
fill_buffer_zero_double(&gpu_context, d_spike_inh__buf, ring_buffer_size * sizeof(double));
fill_buffer_zero_double(&gpu_context, d_currents_, ring_buffer_size * sizeof(double));
fill_buffer_zero_double(&gpu_context, d_spike_exc_, ring_buffer_size * sizeof(double));
fill_buffer_zero_double(&gpu_context, d_spike_inh_, ring_buffer_size * sizeof(double));
      is_ring_buffer_ready = true;
    }
    
  getKernel("hh_cond_exp_traub", "update", "deliver_events_stdp_pl", &gpu_context);
}

void
nest::hh_cond_exp_traub_gpu::clear_buffer()
{
  for (unsigned int i = 0; i < ring_buffer_size; i++)
    {
      h_currents_[i] = 0.0;
h_spike_exc_[i] = 0.0;
h_spike_inh_[i] = 0.0;
    }
  //   h_event_buffer[i] = 0.0;
  
  //fill_buffer_zero_double(&gpu_context, d_ring_buffer, ring_buffer_size);
}

void
nest::hh_cond_exp_traub_gpu::deliver_events()
{
  int batch_size = list_spikes.size();
  if (batch_size == 0)
    return;
  
  h_spike_tgid = new int[batch_size];
  // h_t_spike = new double[batch_size];
  // h_dendritic_delay = new double[batch_size];
  h_weight_ = new double[batch_size];
  h_pos = new long[batch_size];
  h_t_lastspike = new double[batch_size];
  h_Kplus_ = new double[batch_size];
  h_conn_type_ = new int[batch_size];

  int update_type = kernel().simulation_manager.update_type;
  if (update_type == 2)
    {
      h_multiplicity = new int[batch_size];
    }

  create(&gpu_context, &d_spike_tgid, batch_size*sizeof(int));
  // create(&gpu_context, &d_t_spike, batch_size*sizeof(double));
  // create(&gpu_context, &d_dendritic_delay, batch_size*sizeof(double));
  create(&gpu_context, &d_weight_, batch_size*sizeof(double));
  create(&gpu_context, &d_pos, batch_size*sizeof(long));
  create(&gpu_context, &d_t_lastspike, batch_size*sizeof(double));
  create(&gpu_context, &d_Kplus_, batch_size*sizeof(double));
  create(&gpu_context, &d_conn_type_, batch_size*sizeof(int));

  if (update_type == 2)
    {
      create(&gpu_context, &d_multiplicity, batch_size*sizeof(int));
    }

  synchronize();

  int type_count = 0;
  int ind = 0;
  for (vector< synapse_info >::iterator it = list_spikes.begin();
       it != list_spikes.end(); it++, ind++)
    {
      synapse_info& entry = *it;
      h_spike_tgid[ind] = entry.target_node;
      // h_t_spike[ind] = entry.t_spike;
      // h_dendritic_delay[ind] = entry.dendritic_delay;
      h_weight_[ind] = entry.weight;
      
      h_conn_type_[ind] = entry.type;
      h_pos[ind] = entry.pos;
      if (update_type == 2)
	{
	  h_multiplicity[ind] = entry.multiplicity;
	  
	}

      if (entry.type == 2)
	{
	  type_count++;
	  h_Kplus_[ind] = entry.Kplus;	  
	  h_t_lastspike[ind] = entry.last_t_spike;
	}
    }
  
  upload(&gpu_context, (void*)h_spike_tgid, d_spike_tgid, batch_size*sizeof(int));
  // upload(&gpu_context, (void*)h_t_spike, d_t_spike, batch_size*sizeof(double));
  // upload(&gpu_context, (void*)h_dendritic_delay, d_dendritic_delay, batch_size*sizeof(double));
  upload(&gpu_context, (void*)h_weight_, d_weight_, batch_size*sizeof(double));
  upload(&gpu_context, (void*)h_pos, d_pos, batch_size*sizeof(long));

  if (update_type == 2)
    {
      upload(&gpu_context, (void*)h_multiplicity, d_multiplicity, batch_size*sizeof(int));
    }
  
  if (type_count > 0)
    {
      upload(&gpu_context, (void*)h_t_lastspike, d_t_lastspike, batch_size*sizeof(double));
      upload(&gpu_context, (void*)h_Kplus_, d_Kplus_, batch_size*sizeof(double));
      upload(&gpu_context, (void*)h_conn_type_, d_conn_type_, batch_size*sizeof(int));
      conn_type = 0;
    }
  else
    {
      conn_type = 1;
    }
  
  synchronize();

  set_deliver_kernel_args(this->deliver_kernel, this->num_local_nodes, batch_size);
  execute_kernel(this->deliver_kernel, &gpu_context, batch_size);

  synchronize();

  if (type_count > 0)
    {
      download(&gpu_context, d_weight_, (void*)h_weight_, batch_size * sizeof(double));
      download(&gpu_context, d_Kplus_, (void*)h_Kplus_, batch_size * sizeof(double));
  
  
      synchronize();
      ind = 0;
      for (vector< synapse_info >::iterator it = list_spikes.begin();
	   it != list_spikes.end(); it++, ind++)
	{
	  synapse_info& entry = *it;
	  if (entry.type == 1)
	    {
	      // STDPConnection< TargetIdentifierIndex > *connection = entry.connection;
	      // connection->weight_ = h_weight_[ind];
	      // connection->Kplus_ = h_Kplus_[ind];

	    }
	  else if (entry.type == 2)
	    {
	      STDPPLConnectionHom< TargetIdentifierIndex > *connection = entry.connection;
	      connection->weight_ = h_weight_[ind];
	      connection->Kplus_ = h_Kplus_[ind];
	    }
	}

    }
  delete[] h_spike_tgid;
  // delete[] h_t_spike;
  // delete[] h_dendritic_delay;
  delete[] h_weight_;
  delete[] h_pos;
  delete[] h_t_lastspike;
  if (update_type == 2)
    {
      delete[] h_multiplicity;
    }
  delete[] h_Kplus_;
  delete[] h_conn_type_;
  
  list_spikes.clear();
  
//   size_t num_nodes = kernel().node_manager.size();
//   int buffer_size = num_nodes * event_size * sizeof(double);
  
//   set_deliver_kernel_args(this->deliver_kernel, num_nodes);

//   upload_event_data(num_nodes, buffer_size);
//   synchronize();

// #ifdef PROFILING
//   struct timeval start_time, end_time, diff_time;
//   gettimeofday(&start_time, NULL);
// #endif

//   execute_kernel(this->deliver_kernel, &gpu_context, num_nodes);
//   synchronize();

// #ifdef PROFILING
//   gettimeofday(&end_time, NULL);
//   timersub(&end_time, &start_time, &diff_time);
//   double diff = (double)diff_time.tv_sec*1000 + (double)diff_time.tv_usec/1000;
//   printf("Deliver kernel: %0.3f\n", diff);
// #endif

//   download_event_data(num_nodes, buffer_size);
//   synchronize();
}

void
nest::hh_cond_exp_traub_gpu::copy_event_data(std::vector<Node *> nodes)
{
  // int nodeid = 0;
  // double *coeff_ptr = h_coeff_buffer;
  // for ( std::vector<Node*>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); nodeIt++)
  //   {
  //     nest::hh_cond_exp_traub* node = (nest::hh_cond_exp_traub*)*nodeIt;
  //     nodeid = node->get_gid() - 1;
  //     coeff_ptr = h_coeff_buffer + event_size * nodeid;

  //     node->B_.sumj_g_ij_ = h_B_sumj[nodeid];
  //     for (size_t i = 0; i < event_size; i++)
  // 	{
  // 	  node->B_.interpolation_coefficients[i] = coeff_ptr[i];
  // 	}
    
  //     //coeff_ptr += event_size;
  //   }
}

void
nest::hh_cond_exp_traub_gpu::handle(index sgid, index tgid)
{

}

void nest::hh_cond_exp_traub_gpu::handle( SpikeEvent& e )
{};
void nest::hh_cond_exp_traub_gpu::handle( CurrentEvent& e )
{};

void nest::hh_cond_exp_traub_gpu::pre_deliver_event(const std::vector< Node* > &nodes)
{
  size_t len = nodes.size();
  
  if (h_history_ptr == NULL)
    {
      h_history_ptr = new int[len + 1];
      create(&gpu_context, &d_history_ptr, (len + 1)*sizeof(int));
    }
  
  if (h_Kminus_ == NULL)
    {
      h_Kminus_ = new double[len];
      create(&gpu_context, &d_Kminus_, len*sizeof(double));
    }
  
  if (h_tau_minus_inv_ == NULL)
    {
      h_tau_minus_inv_ = new double[len];
      create(&gpu_context, &d_tau_minus_inv_, len*sizeof(double));
    }

  synchronize();
  
  typedef std::deque< histentry > hist_queue;
  hist_queue nodes_history;
  
  int nodeid = 0;

  
  for (std::vector< Node* >::const_iterator it = nodes.begin(); it != nodes.end();
       it++, nodeid++)
    {
      if (nodeid >= this->num_local_nodes - 2)
	break;

      nest::hh_cond_exp_traub* node = (nest::hh_cond_exp_traub*)*it;
      
      h_history_ptr[nodeid] = nodes_history.size();
      //history_size = nodes_history.size();
      hist_queue &h_ = node->get_all_history();
      hist_queue::iterator last_it = nodes_history.end();
      nodes_history.insert(last_it, h_.begin(), h_.end());
      h_Kminus_[nodeid] = node->get_Kminus_();
      h_tau_minus_inv_[nodeid] = node->get_tau_minus_inv_();
    }
  
  history_size = nodes_history.size();
  
  for (; nodeid < len + 1; nodeid++)
    h_history_ptr[nodeid] = history_size;

  upload(&gpu_context, (void*)h_history_ptr, d_history_ptr, (len + 1) * sizeof(int));
  upload(&gpu_context, (void*)h_Kminus_, d_Kminus_, len*sizeof(double));
  upload(&gpu_context, (void*)h_tau_minus_inv_, d_tau_minus_inv_, len*sizeof(double));

  synchronize();
  
  h_history_Kminus_ = new double[history_size];
  h_history_t_ = new double[history_size];
  h_history_access_counter_ = new int[history_size];

  create(&gpu_context, &d_history_Kminus_, history_size*sizeof(double));
  create(&gpu_context, &d_history_t_, history_size*sizeof(double));
  create(&gpu_context, &d_history_access_counter_, history_size*sizeof(int));

  synchronize();
  
  int index = 0;
  for (hist_queue::iterator hist_it = nodes_history.begin();
       hist_it != nodes_history.end();
       hist_it++, index++)
    {
      histentry& entry = *hist_it;
      h_history_Kminus_[index] = entry.Kminus_;
      h_history_t_[index] = entry.t_;
      h_history_access_counter_[index] = entry.access_counter_;
    }

  upload(&gpu_context, (void*)h_history_Kminus_, d_history_Kminus_, history_size*sizeof(double));
  upload(&gpu_context, (void*)h_history_t_, d_history_t_, history_size*sizeof(double));
  upload(&gpu_context, (void*)h_history_access_counter_, d_history_access_counter_, history_size*sizeof(int));
  //fill_buffer_zero_uint(&gpu_context, d_history_access_counter_, history_size*sizeof(int));      

  synchronize();
}

void nest::hh_cond_exp_traub_gpu::post_deliver_event(const std::vector< Node* > &nodes)
{
  typedef std::deque< histentry > hist_queue;
  download(&gpu_context, d_history_access_counter_, (void*)h_history_access_counter_, history_size * sizeof(int));

  synchronize();
  
  int hist_it = 0;
  int nodeid = 0;
  for (std::vector< Node* >::const_iterator it = nodes.begin(); it != nodes.end();
       it++, nodeid++)
    {
      if (nodeid >= this->num_local_nodes - 2)
	break;

      nest::hh_cond_exp_traub* node = (nest::hh_cond_exp_traub*)*it;
      hist_queue h_ = node->get_all_history();
      for (hist_queue::iterator h_it = h_.begin(); h_it != h_.end(); h_it++, hist_it++)
	{
	  h_it->access_counter_ = h_history_access_counter_[hist_it];
	}
    }

  delete[] h_history_Kminus_;
  delete[] h_history_t_;
  delete[] h_history_access_counter_;
  // delete d_history_Kminus_;
  // delete d_history_t_;
  // delete d_history_access_counter_;
}

void nest::hh_cond_exp_traub_gpu::handle(Event& ev, double last_t_spike, const CommonSynapseProperties *csp, void *conn, int conn_type)
{
  SpikeEvent *e = (SpikeEvent *)&ev;

  if (conn_type == 2)
    {
      const STDPPLHomCommonProperties *cp = (STDPPLHomCommonProperties *)csp;
      int sgid = e->get_sender_gid();
      int tgid = e->get_receiver().get_thread_lid();

      cp_lambda_ = cp->lambda_;
      cp_mu_ = cp->mu_;
      cp_alpha_ = cp->alpha_;
      cp_tau_plus_inv_ = cp->tau_plus_inv_;

      double t_spike = e->get_stamp().get_ms();

      STDPPLConnectionHom< TargetIdentifierIndex > *connection = (STDPPLConnectionHom< TargetIdentifierIndex > *) conn;

      synapse_info conn_info;
      conn_info.source_node = sgid;
      conn_info.target_node = tgid;
      conn_info.connection = connection;
      //conn_info.t_spike = t_spike;
      conn_info.last_t_spike = last_t_spike;
      //conn_info.dendritic_delay = connection->get_delay();
      conn_info.weight = connection->weight_;
      conn_info.Kplus = connection->Kplus_;
      conn_info.pos = e->get_rel_delivery_steps(
						kernel().simulation_manager.get_slice_origin() );
      conn_info.multiplicity = e->get_multiplicity();
      conn_info.type = conn_type;
      list_spikes.push_back(conn_info);

      event_t_spike = t_spike;
      event_dendritic_delay = connection->get_delay();

      event_multiplicity = e->get_multiplicity();
    }
  else if (conn_type == 1)
    {
      int sgid = e->get_sender_gid();
      int tgid = e->get_receiver().get_thread_lid();

      synapse_info conn_info;
      conn_info.source_node = sgid;
      conn_info.target_node = tgid;
      conn_info.weight = e->get_weight();
      conn_info.pos = e->get_rel_delivery_steps(
      						kernel().simulation_manager.get_slice_origin() );
      conn_info.multiplicity = e->get_multiplicity();
      conn_info.type = conn_type;
      list_spikes.push_back(conn_info);

      event_multiplicity = e->get_multiplicity();
    }
}

void nest::hh_cond_exp_traub_gpu::insert_event(SpikeEvent& e)
{
  //SpikeEvent *e = (SpikeEvent *)&ev;

  int sgid = e.get_sender_gid();
  int tgid = e.get_receiver().get_thread_lid();

  synapse_info conn_info;
  conn_info.source_node = sgid;
  conn_info.target_node = tgid;
  conn_info.pos = e.get_rel_delivery_steps(
					   kernel().simulation_manager.get_slice_origin() );
  conn_info.weight = e.get_weight();
  conn_info.multiplicity = e.get_multiplicity();
  conn_info.type = 1;
  list_spikes.push_back(conn_info);
}

void
nest::hh_cond_exp_traub_gpu::advance_time()
{
  time_index = (time_index + kernel().connection_manager.get_min_delay()) % event_size;
}
