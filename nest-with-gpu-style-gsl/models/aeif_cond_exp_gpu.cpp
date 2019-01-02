#include "aeif_cond_exp_gpu.h"
#include "aeif_cond_exp.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

#ifdef PROFILING
#include <sys/time.h>
#endif

nest::aeif_cond_exp_gpu::clContext_ nest::aeif_cond_exp_gpu::gpu_context;
bool nest::aeif_cond_exp_gpu::is_gpu_initialized = false;

nest::aeif_cond_exp_gpu::aeif_cond_exp_gpu()
  : graph_size( 0 )
  , h_connections_ptr( NULL )
  , h_connections( NULL )
  , event_size( 0 )
  , ring_buffer_size( 0 )
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
 , h_P__a( NULL )
 , h_B__I_stim_( NULL )
 , h_P__tau_syn_in( NULL )
 , h_P__E_L( NULL )
 , h_S__y_( NULL )
 , h_P__I_e( NULL )
 , h_V__V_peak( NULL )
 , h_B__IntegrationStep_( NULL )
 , h_S__r_( NULL )
 , h_V__refractory_counts_( NULL )
 , h_P__E_in( NULL )
 , h_P__C_m( NULL )
 , h_P__V_peak_( NULL )
 , h_P__g_L( NULL )
 , h_P__Delta_T( NULL )
 , h_P__V_th( NULL )
 , h_P__b( NULL )
 , h_P__tau_syn_ex( NULL )
 , h_P__E_ex( NULL )
 , h_B__step_( NULL )
 , h_P__V_reset_( NULL )
 , h_P__tau_w( NULL )
  , h_B_step_( NULL )
  , h_B_IntegrationStep_( NULL )
    , h_currents_( NULL )
, h_spike_exc_( NULL )
, h_spike_inh_( NULL )
    , h_spike_count (NULL)

{
}

nest::aeif_cond_exp_gpu::~aeif_cond_exp_gpu()
{
  if (h_connections_ptr) delete[] h_connections_ptr;
  if (h_connections) delete[] h_connections;
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
if (h_P__a) delete[] h_P__a;
if (h_B__I_stim_) delete[] h_B__I_stim_;
if (h_P__tau_syn_in) delete[] h_P__tau_syn_in;
if (h_P__E_L) delete[] h_P__E_L;
if (h_S__y_) delete[] h_S__y_;
if (h_P__I_e) delete[] h_P__I_e;
if (h_V__V_peak) delete[] h_V__V_peak;
if (h_B__IntegrationStep_) delete[] h_B__IntegrationStep_;
if (h_S__r_) delete[] h_S__r_;
if (h_V__refractory_counts_) delete[] h_V__refractory_counts_;
if (h_P__E_in) delete[] h_P__E_in;
if (h_P__C_m) delete[] h_P__C_m;
if (h_P__V_peak_) delete[] h_P__V_peak_;
if (h_P__g_L) delete[] h_P__g_L;
if (h_P__Delta_T) delete[] h_P__Delta_T;
if (h_P__V_th) delete[] h_P__V_th;
if (h_P__b) delete[] h_P__b;
if (h_P__tau_syn_ex) delete[] h_P__tau_syn_ex;
if (h_P__E_ex) delete[] h_P__E_ex;
if (h_B__step_) delete[] h_B__step_;
if (h_P__V_reset_) delete[] h_P__V_reset_;
if (h_P__tau_w) delete[] h_P__tau_w;
  if (h_B_step_) delete[] h_B_step_;
  if (h_B_IntegrationStep_) delete[] h_B_IntegrationStep_;
  if (h_spike_count) delete[] h_spike_count;
  if (h_currents_) delete[] h_currents_;
if (h_spike_exc_) delete[] h_spike_exc_;
if (h_spike_inh_) delete[] h_spike_inh_;
}

void
nest::aeif_cond_exp_gpu::initialize_gpu()
{
  if (is_gpu_initialized)
    return;

  if (initialize_opencl_context())
    return;

  is_gpu_initialized = true;
}

void
nest::aeif_cond_exp_gpu::mass_update(std::vector< Node* > nodes, Time const& origin,
					const long from,
					const long to )
{
  mass_update_(nodes, origin, from, to, false );
}

bool
nest::aeif_cond_exp_gpu::mass_wfr_update(std::vector< Node* > nodes, Time const& origin,
					    const long from,
					    const long to )
{
  return mass_update_(nodes, origin, from, to, true );
}

// TODO: the for loops will later be kernel calls
bool
nest::aeif_cond_exp_gpu::mass_update_( std::vector<Node *> &nodes,
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

  initialize_device(total_num_nodes, num_local_nodes, aeif_cond_exp::State_::STATE_VEC_SIZE);
  set_kernel_args(this->gpu_kernel, num_local_nodes, aeif_cond_exp::State_::STATE_VEC_SIZE);

  // TODO: for now, I assume that these are the same for all nodes

  bool first_loop = true;
  // if (from < to)
  //   prepare_copy_to_device(nodes, called_from_wfr_update, from);
  for ( long lag = from; lag < to; ++lag, first_loop = false )
    {
  //     for ( std::vector<Node*>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); nodeIt++ )
  // 	{
  // 	  nest::aeif_cond_exp* node = (nest::aeif_cond_exp*)*nodeIt;
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
    
      copy_data_to_device(nodes, aeif_cond_exp::State_::STATE_VEC_SIZE, first_loop, lag);
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
    
      copy_data_from_device(nodes, aeif_cond_exp::State_::STATE_VEC_SIZE, false);

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
      for ( std::vector<Node*>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); nodeIt++, node_id++ )
	{
	  nest::aeif_cond_exp* node = (nest::aeif_cond_exp*)*nodeIt;
	  //node->post_gsl(origin, lag);

	  for (size_t i = 0; i < h_spike_count[node_id]; i++)
	    {
	      node->set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );
	      SpikeEvent se;
	      kernel().event_delivery_manager.send( *node, se, lag );
	    }
	}
    }

  copy_data_from_device(nodes, aeif_cond_exp::State_::STATE_VEC_SIZE, true);
  return true;
}

int
nest::aeif_cond_exp_gpu::initialize_opencl_context()
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
nest::aeif_cond_exp_gpu::initialize_command_queue(clContext_ *clCxt)
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
nest::aeif_cond_exp_gpu::initialize_device(int total_num_nodes, int num_local_nodes, int dimension)
{
  if (!is_initialized)
    {
      printf("initialize_device %d\n", num_local_nodes);
      connections.resize(total_num_nodes);
      event_size = kernel().connection_manager.get_min_delay()
        + kernel().connection_manager.get_max_delay();
      ring_buffer_size = total_num_nodes * event_size;

      h_currents_ = new double[ring_buffer_size];
h_spike_exc_ = new double[ring_buffer_size];
h_spike_inh_ = new double[ring_buffer_size];
      
      if (initialize_command_queue(&gpu_context))
	return;

      getKernel("aeif_cond_exp", "gsl", "deliver_events", &gpu_context);

      int len1 = num_local_nodes;
      int len2 = dimension*num_local_nodes;
      
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
      h_P__a = new double[len1];
      h_B__I_stim_ = new double[len1];
      h_P__tau_syn_in = new double[len1];
      h_P__E_L = new double[len1];
      h_S__y_ = new double[len1 *4];
      h_P__I_e = new double[len1];
      h_V__V_peak = new double[len1];
      h_B__IntegrationStep_ = new double[len1];
      h_S__r_ = new unsigned int[len1];
      h_V__refractory_counts_ = new unsigned int[len1];
      h_P__E_in = new double[len1];
      h_P__C_m = new double[len1];
      h_P__V_peak_ = new double[len1];
      h_P__g_L = new double[len1];
      h_P__Delta_T = new double[len1];
      h_P__V_th = new double[len1];
      h_P__b = new double[len1];
      h_P__tau_syn_ex = new double[len1];
      h_P__E_ex = new double[len1];
      h_B__step_ = new double[len1];
      h_P__V_reset_ = new double[len1];
      h_P__tau_w = new double[len1];
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
      create(&gpu_context, &P__a, len1*sizeof(double));
      create(&gpu_context, &B__I_stim_, len1*sizeof(double));
      create(&gpu_context, &P__tau_syn_in, len1*sizeof(double));
      create(&gpu_context, &P__E_L, len1*sizeof(double));
      create(&gpu_context, &S__y_, len1*4*sizeof(double));
      create(&gpu_context, &P__I_e, len1*sizeof(double));
      create(&gpu_context, &V__V_peak, len1*sizeof(double));
      create(&gpu_context, &B__IntegrationStep_, len1*sizeof(double));
      create(&gpu_context, &S__r_, len1*sizeof(unsigned int));
      create(&gpu_context, &V__refractory_counts_, len1*sizeof(unsigned int));
      create(&gpu_context, &P__E_in, len1*sizeof(double));
      create(&gpu_context, &P__C_m, len1*sizeof(double));
      create(&gpu_context, &P__V_peak_, len1*sizeof(double));
      create(&gpu_context, &P__g_L, len1*sizeof(double));
      create(&gpu_context, &P__Delta_T, len1*sizeof(double));
      create(&gpu_context, &P__V_th, len1*sizeof(double));
      create(&gpu_context, &P__b, len1*sizeof(double));
      create(&gpu_context, &P__tau_syn_ex, len1*sizeof(double));
      create(&gpu_context, &P__E_ex, len1*sizeof(double));
      create(&gpu_context, &B__step_, len1*sizeof(double));
      create(&gpu_context, &P__V_reset_, len1*sizeof(double));
      create(&gpu_context, &P__tau_w, len1*sizeof(double));
      create(&gpu_context, &B_step_, len1*sizeof(double));
      create(&gpu_context, &d_spike_count, len1*sizeof(double));
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
// nest::aeif_cond_exp_gpu::prepare_copy_to_device(std::vector< Node* > &nodes, bool called_from_wfr_update, long lag_)
// {
  // int num_nodes = nodes.size();

  // int wfr_interpolation_order = kernel().simulation_manager.get_wfr_interpolation_order();

  // //int len3 = /*buffer_size*/ 4 * num_nodes;
	
  // std::vector<Node *>::iterator nodeIt = nodes.begin();

  // for (int i = 0; nodeIt != nodes.end(); nodeIt++, i++ )
  //   {
  //     nest::aeif_cond_exp* node = (nest::aeif_cond_exp*)*nodeIt;
      
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
nest::aeif_cond_exp_gpu::copy_data_to_device(std::vector< Node* > &nodes, int dimension, bool first_loop, long lag_)
{
  int num_nodes = nodes.size();

  std::vector<Node *>::iterator nodeIt = nodes.begin();

  for (int i = 0; nodeIt != nodes.end(); nodeIt++, i++ )
    {

      nest::aeif_cond_exp* node = (nest::aeif_cond_exp*)*nodeIt;

      if (first_loop)
	{
	  // UPLOAD_1D_DATA(h_con_state_eps_abs, B_.c_->eps_abs);
	  // UPLOAD_1D_DATA(h_con_state_eps_rel, B_.c_->eps_rel);
	  // UPLOAD_1D_DATA(h_con_state_a_y, B_.c_->a_y);
	  // UPLOAD_1D_DATA(h_con_state_a_dydt, B_.c_->a_dydt);
UPLOAD_1D_DATA(h_P__a, P_.a);
UPLOAD_1D_DATA(h_B__I_stim_, B_.I_stim_);
UPLOAD_1D_DATA(h_P__tau_syn_in, P_.tau_syn_in);
UPLOAD_1D_DATA(h_P__E_L, P_.E_L);
UPLOAD_2D_DATA(h_S__y_, S_.y_);
UPLOAD_1D_DATA(h_P__I_e, P_.I_e);
UPLOAD_1D_DATA(h_V__V_peak, V_.V_peak);
UPLOAD_1D_DATA(h_B__IntegrationStep_, B_.IntegrationStep_);
UPLOAD_1D_DATA(h_S__r_, S_.r_);
UPLOAD_1D_DATA(h_V__refractory_counts_, V_.refractory_counts_);
UPLOAD_1D_DATA(h_P__E_in, P_.E_in);
UPLOAD_1D_DATA(h_P__C_m, P_.C_m);
UPLOAD_1D_DATA(h_P__V_peak_, P_.V_peak_);
UPLOAD_1D_DATA(h_P__g_L, P_.g_L);
UPLOAD_1D_DATA(h_P__Delta_T, P_.Delta_T);
UPLOAD_1D_DATA(h_P__V_th, P_.V_th);
UPLOAD_1D_DATA(h_P__b, P_.b);
UPLOAD_1D_DATA(h_P__tau_syn_ex, P_.tau_syn_ex);
UPLOAD_1D_DATA(h_P__E_ex, P_.E_ex);
UPLOAD_1D_DATA(h_B__step_, B_.step_);
UPLOAD_1D_DATA(h_P__V_reset_, P_.V_reset_);
UPLOAD_1D_DATA(h_P__tau_w, P_.tau_w);
	  UPLOAD_1D_DATA(h_B_step_, B_.step_);
	  UPLOAD_1D_DATA(h_B_IntegrationStep_, B_.IntegrationStep_);
	}


      /* DEVICE OUTPUT VAR UPLOAD */

    }
  
  if (first_loop)
    {
      FINISH_1D_UPLOAD(con_state_eps_abs, h_con_state_eps_abs, double);
      FINISH_1D_UPLOAD(con_state_eps_rel, h_con_state_eps_rel, double);
      FINISH_1D_UPLOAD(con_state_a_y, h_con_state_a_y, double);
      FINISH_1D_UPLOAD(con_state_a_dydt, h_con_state_a_dydt, double);
FINISH_1D_UPLOAD(P__a, h_P__a, double);
FINISH_1D_UPLOAD(B__I_stim_, h_B__I_stim_, double);
FINISH_1D_UPLOAD(P__tau_syn_in, h_P__tau_syn_in, double);
FINISH_1D_UPLOAD(P__E_L, h_P__E_L, double);
FINISH_2D_UPLOAD(S__y_, h_S__y_, 4, double);
FINISH_1D_UPLOAD(P__I_e, h_P__I_e, double);
FINISH_1D_UPLOAD(V__V_peak, h_V__V_peak, double);
FINISH_1D_UPLOAD(B__IntegrationStep_, h_B__IntegrationStep_, double);
FINISH_1D_UPLOAD(S__r_, h_S__r_, unsigned int);
FINISH_1D_UPLOAD(V__refractory_counts_, h_V__refractory_counts_, unsigned int);
FINISH_1D_UPLOAD(P__E_in, h_P__E_in, double);
FINISH_1D_UPLOAD(P__C_m, h_P__C_m, double);
FINISH_1D_UPLOAD(P__V_peak_, h_P__V_peak_, double);
FINISH_1D_UPLOAD(P__g_L, h_P__g_L, double);
FINISH_1D_UPLOAD(P__Delta_T, h_P__Delta_T, double);
FINISH_1D_UPLOAD(P__V_th, h_P__V_th, double);
FINISH_1D_UPLOAD(P__b, h_P__b, double);
FINISH_1D_UPLOAD(P__tau_syn_ex, h_P__tau_syn_ex, double);
FINISH_1D_UPLOAD(P__E_ex, h_P__E_ex, double);
FINISH_1D_UPLOAD(B__step_, h_B__step_, double);
FINISH_1D_UPLOAD(P__V_reset_, h_P__V_reset_, double);
FINISH_1D_UPLOAD(P__tau_w, h_P__tau_w, double);
      FINISH_1D_UPLOAD(B_step_, h_B_step_, double);
      FINISH_1D_UPLOAD(B_IntegrationStep_, h_B_IntegrationStep_, double);
    }

    /* DEVICE OUTPUT VAR FINISH UPLOAD */

  fill_buffer_zero_uint(&gpu_context, d_spike_count, num_nodes*sizeof(unsigned int));
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
nest::aeif_cond_exp_gpu::copy_data_from_device(std::vector< Node* > &nodes, int dimension, bool last_copy)
{
  int num_nodes = nodes.size();

  std::vector<Node *>::iterator nodeIt = nodes.begin();

  // if (last_copy)
  //   {
  //     START_1D_DOWNLOAD(B_IntegrationStep_, h_B_IntegrationStep_, double);
  //     /* DEVICE OUTPUT VAR DOWNLOAD */
  //   }
    
  START_1D_DOWNLOAD(d_spike_count, h_spike_count, unsigned int);
  
  synchronize();
  
  // for (int i = 0; nodeIt != nodes.end(); nodeIt++, i++)
  //   {
  //     nest::aeif_cond_exp* node = (nest::aeif_cond_exp*)*nodeIt;

  //     if (last_copy)
  // 	{
  // 	  DOWNLOAD_1D_DATA(B_.IntegrationStep_, h_B_IntegrationStep_);
  // 	}

  //     /* DEVICE OUTPUT VAR FINISH DOWNLOAD */
  //   }
}

void
nest::aeif_cond_exp_gpu::create(clContext_ *clCxt, cl::Buffer *mem, int len)
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
nest::aeif_cond_exp_gpu::upload(clContext_ *clCxt, void *data, cl::Buffer &gdata, int datalen)
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
nest::aeif_cond_exp_gpu::download(clContext_ *clCxt, cl::Buffer &gdata,void *data,int data_len, int offset)
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
nest::aeif_cond_exp_gpu::synchronize()
{
  this->command_queue.finish();
}

int
nest::aeif_cond_exp_gpu::getKernel(string source, string kernelName, string deliver_kernel_name, clContext_ *clCxt)
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
  ret = kernel->setArg(arg_idx, value); \
  if (ret != CL_SUCCESS) {						\
    printf("Failed to set arg %d, error code %d\n", arg_idx, ret);	\
    return 1;								\
  }									\
  arg_idx++;

int
nest::aeif_cond_exp_gpu::set_lag_args(cl::Kernel *kernel, long lag)
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
nest::aeif_cond_exp_gpu::set_kernel_args(cl::Kernel *kernel, int num_nodes, int dimension)
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

  arg_idx++; //lag

  ret = kernel->setArg(arg_idx, static_cast<cl_int>(ring_buffer_size)); 
  if (ret != CL_SUCCESS) {						
    printf("Failed to set arg %d, error code %dn", arg_idx, ret);	
    return 1;								
  }									
  arg_idx++;

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
  set_kernel(P__a);
  set_kernel(B__I_stim_);
  set_kernel(P__tau_syn_in);
  set_kernel(P__E_L);
  set_kernel(P__I_e);
  set_kernel(V__V_peak);
  set_kernel(B__IntegrationStep_);
  set_kernel(S__r_);
  set_kernel(V__refractory_counts_);
  set_kernel(P__E_in);
  set_kernel(P__C_m);
  set_kernel(P__V_peak_);
  set_kernel(P__g_L);
  set_kernel(P__Delta_T);
  set_kernel(P__V_th);
  set_kernel(P__b);
  set_kernel(P__tau_syn_ex);
  set_kernel(P__E_ex);
  set_kernel(B__step_);
  set_kernel(P__V_reset_);
  set_kernel(P__tau_w);
  set_kernel(d_currents_);
  set_kernel(d_spike_exc_);
  set_kernel(d_spike_inh_);
  set_kernel(B_step_);
  set_kernel(B_IntegrationStep_);
  set_kernel(d_spike_count);
    set_kernel(S__y_);

  return 0;
}

int
nest::aeif_cond_exp_gpu::set_deliver_kernel_args(cl::Kernel *kernel, int num_nodes, cl::Buffer &d_event_buffer_in, cl::Buffer &d_event_buffer_out)
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
  set_kernel(d_event_buffer_in);
  set_kernel(d_event_buffer_out);
  // set_kernel(d_coeff_buffer);
  // set_kernel(B_sumj);

  return 0;
}

void
nest::aeif_cond_exp_gpu::execute_kernel(cl::Kernel *kernel, clContext_ *clCxt, size_t num_nodes)
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
nest::aeif_cond_exp_gpu::fill_spike_event_buffer( Event& e)
{
}

void
nest::aeif_cond_exp_gpu::fill_event_buffer( SecondaryEvent& e)
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
nest::aeif_cond_exp_gpu::fill_buffer_zero_double(clContext_ *clCxt, cl::Buffer &buffer, int size)
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
nest::aeif_cond_exp_gpu::fill_buffer_zero_uint(clContext_ *clCxt, cl::Buffer &buffer, int size)
{
  cl_uint zero = 0.0;
  cl_int ret;

  ret = this->command_queue.enqueueFillBuffer(buffer, zero, 0, size);
  
  if (ret != CL_SUCCESS)
    {
      printf("Failed to EnqueueNDRangeKernel. error code: %d\n", ret);
      return;
    }
}


void
nest::aeif_cond_exp_gpu::initialize_graph()
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

  // create(&gpu_context, &d_event_buffer, num_nodes * event_size * sizeof(double));
  // create(&gpu_context, &d_coeff_buffer, num_nodes * event_size * sizeof(double));
  // create(&gpu_context, &d_event_weight, num_nodes * sizeof(double));
  // create(&gpu_context, &B_sumj, num_nodes*sizeof(double));

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
nest::aeif_cond_exp_gpu::upload_event_data(double *h_event_buffer, cl::Buffer &d_event_buffer_in, cl::Buffer &d_event_buffer_out)
{
  upload(&gpu_context, (void*)h_event_buffer, d_event_buffer_in, ring_buffer_size * sizeof(double));
  // upload(&gpu_context, (void*)h_event_weight, d_event_weight, num_nodes * sizeof(double));
}

void
nest::aeif_cond_exp_gpu::download_event_data(int num_nodes, int buffer_size)
{
  // download(&gpu_context, d_coeff_buffer, (void*)h_coeff_buffer, buffer_size);
  // download(&gpu_context, B_sumj, (void*)h_B_sumj, num_nodes * sizeof(double));
}

void
nest::aeif_cond_exp_gpu::clear_buffer()
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
nest::aeif_cond_exp_gpu::deliver_events(double *h_event_buffer, cl::Buffer &d_event_buffer_in, cl::Buffer &d_event_buffer_out)
{
  size_t num_nodes = kernel().node_manager.size();
  
  set_deliver_kernel_args(this->deliver_kernel, num_nodes, d_event_buffer_in, d_event_buffer_out);

  upload_event_data(h_event_buffer, d_event_buffer_in, d_event_buffer_out);
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

  synchronize();

}

void
nest::aeif_cond_exp_gpu::deliver_events()
{
  deliver_events(h_currents_, d_currents__buf, d_currents_);
deliver_events(h_spike_exc_, d_spike_exc__buf, d_spike_exc_);
deliver_events(h_spike_inh_, d_spike_inh__buf, d_spike_inh_);
  
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
nest::aeif_cond_exp_gpu::copy_event_data(std::vector<Node *> nodes)
{
  // int nodeid = 0;
  // double *coeff_ptr = h_coeff_buffer;
  // for ( std::vector<Node*>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); nodeIt++)
  //   {
  //     nest::aeif_cond_exp* node = (nest::aeif_cond_exp*)*nodeIt;
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
nest::aeif_cond_exp_gpu::handle(index sgid, index tgid)
{
  
  connections[tgid].push_back(sgid);
}

void nest::aeif_cond_exp_gpu::handle( SpikeEvent& e )
{
  assert( e.get_delay() > 0 );

  if ( e.get_weight() > 0.0 )
  {
    ring_buffer_add_value(h_spike_exc_, e.get_rel_delivery_steps(
                               kernel().simulation_manager.get_slice_origin() ), e.get_weight() * e.get_multiplicity());
  }
  else
  {
    ring_buffer_add_value(h_spike_inh_, e.get_rel_delivery_steps(
                               kernel().simulation_manager.get_slice_origin() ), -e.get_weight() * e.get_multiplicity());
  } // keep conductances positive
};
void nest::aeif_cond_exp_gpu::handle( CurrentEvent& e )
{
  assert( e.get_delay() > 0 );

  const double c = e.get_current();
  const double w = e.get_weight();

  // add weighted current; HEP 2002-10-04
  ring_buffer_add_value(h_currents_, e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), w * c);
};

void nest::aeif_cond_exp_gpu::ring_buffer_add_value(double *h_ring_buffer, long pos, double val)
{
  h_ring_buffer[pos % ring_buffer_size] += val;
}
