#include "model_name_gpu.h"
#include "model_name.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"
#include "vp_manager_impl.h"

#ifdef PROFILING
#include <sys/time.h>
#endif

nest::model_name_gpu::clContext_ nest::model_name_gpu::gpu_context;
cl::Context nest::iaf_psc_alpha_gpu::context;
cl::Program nest::iaf_psc_alpha_gpu::program;


nest::model_name_gpu::model_name_gpu()
  : is_data_ready( false )
  , graph_size( 0 )
  , h_connections_ptr( NULL )
  , h_connections( NULL )
  , event_size( 0 )
  , ring_buffer_size( 0 )
  , event_multiplicity( 1 )
  , is_initialized( false )
  , is_ring_buffer_ready( false )
  , is_gpu_initialized( false )
/* HOST BUFFERS INIT */
    /* HOST RING BUFFERS INIT */
  , h_spike_count (NULL)
  , h_history_ptr (NULL)
  , h_Kminus_ (NULL)
  , h_tau_minus_inv_ (NULL)

{
}

nest::model_name_gpu::~model_name_gpu()
{
  if (h_connections_ptr) delete[] h_connections_ptr;
  if (h_connections) delete[] h_connections;
/* HOST BUFFERS FREE */
  if (h_spike_count) delete[] h_spike_count;
  /* HOST RING BUFFERS FREE */
}

void
nest::model_name_gpu::initialize_gpu()
{
  if (not is_gpu_initialized)
    {
      int thrd = kernel().vp_manager.get_thread_id();
      if (thrd == 0)
	{
	  if (initialize_opencl_context())
	    return;
	}

      int total_num_nodes = kernel().node_manager.size();
      connections.resize(total_num_nodes);
      is_gpu_initialized = true;
    }
}

void
nest::model_name_gpu::mass_update(const std::vector< Node* > &nodes, Time const& origin,
					const long from,
					const long to )
{
  mass_update_(nodes, origin, from, to, false );
}

bool
nest::model_name_gpu::mass_wfr_update(const std::vector< Node* > &nodes, Time const& origin,
					    const long from,
					    const long to )
{
  return mass_update_(nodes, origin, from, to, true );
}

// TODO: the for loops will later be kernel calls
bool
nest::model_name_gpu::mass_update_( const std::vector<Node *> &nodes,
					  Time const& origin,
					  const long from,
					  const long to,
					  const bool called_from_wfr_update ) // TODO: don't know yet whether we need to cover both cases here
{
  // TODO: do AoS for now, SoA will come later on

#ifdef PROFILING
  int thrd_id = kernel().vp_manager.get_thread_id();
#endif  

  
  initialize_device();
  
  // TODO: for now, I assume that these are the same for all nodes

  if (not is_data_ready)
    {
      copy_data_to_device(nodes);
      is_data_ready = true;
    }
  // if (from < to)
  //   prepare_copy_to_device(nodes, called_from_wfr_update, from);
  
  set_kernel_args(this->gpu_kernel, this->num_local_nodes);
  
  for ( long lag = from; lag < to; ++lag)
    {
  //     for ( std::vector<Node*>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); nodeIt++ )
  // 	{
  // 	  nest::model_name* node = (nest::model_name*)*nodeIt;
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
    
      copy_data_from_device(nodes, false);

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
	  nest::model_name* node = (nest::model_name*)*nodeIt;
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

  copy_data_from_device(nodes, true);
  return true;
}

int
nest::model_name_gpu::initialize_opencl_context()
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

      context = cl::Context(gpu_context.list_device);

      string source = "model_name";
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
      program = cl::Program(context,
			    cl::Program::Sources(1, std::make_pair(source_str, source_size)));

      try {
	program.build(gpu_context.list_device, "-cl-nv-maxrregcount=200 -cl-nv-verbose");
      } catch (const cl::Error&) {
	std::cerr
	  << "OpenCL compilation error" << std::endl
	  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(gpu_context.list_device[0])
	  << std::endl;
	return 1;
      }
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
nest::model_name_gpu::initialize_command_queue()
{
  int thrd_id = kernel().vp_manager.get_thread_id();
  const int vp_id = kernel().vp_manager.thread_to_vp(thrd_id);

  try
    {
      int num_devices = gpu_context.list_device.size();
      
      this->command_queue = cl::CommandQueue(context, gpu_context.list_device[vp_id % num_devices]);
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
nest::model_name_gpu::initialize_device()
{
  if (!is_initialized)
    {
      printf("initialize_device %d\n", this->num_local_nodes);

      int len = this->num_local_nodes;
      
/* HOST BUFFERS ALLOC */
      h_spike_count = new unsigned int[len];

/* DEVICE BUFFERS ALLOC */
      create(&gpu_context, &d_spike_count, len*sizeof(unsigned int));

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
// nest::model_name_gpu::prepare_copy_to_device(std::vector< Node* > &nodes, bool called_from_wfr_update, long lag_)
// {
  // int num_nodes = nodes.size();

  // int wfr_interpolation_order = kernel().simulation_manager.get_wfr_interpolation_order();

  // //int len3 = /*buffer_size*/ 4 * num_nodes;
	
  // std::vector<Node *>::iterator nodeIt = nodes.begin();

  // for (int i = 0; nodeIt != nodes.end(); nodeIt++, i++ )
  //   {
  //     nest::model_name* node = (nest::model_name*)*nodeIt;
      
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
nest::model_name_gpu::copy_data_to_device(const std::vector< Node* > &nodes)
{
  int num_nodes = nodes.size();

  std::vector<Node *>::const_iterator nodeIt = nodes.begin();

  for (int i = 0; nodeIt != nodes.end(); nodeIt++, i++ )
    {

      nest::model_name* node = (nest::model_name*)*nodeIt;

/* DEVICE BUFFERS UPLOAD */


      /* DEVICE OUTPUT VAR UPLOAD */

    }
  

/* DEVICE BUFFERS FINISH UPLOAD */

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
nest::model_name_gpu::copy_data_from_device(const std::vector< Node* > &nodes, bool last_copy)
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
  //     nest::model_name* node = (nest::model_name*)*nodeIt;

  //     if (last_copy)
  // 	{
  // 	  DOWNLOAD_1D_DATA(B_.IntegrationStep_, h_B_IntegrationStep_);
  // 	}

  //     /* DEVICE OUTPUT VAR FINISH DOWNLOAD */
  //   }
}

void
nest::model_name_gpu::create(clContext_ *clCxt, cl::Buffer *mem, int len)
{
  cl_int ret;
  //printf("len %d\n", len);
  if (len == 0)
    return;
  *mem = cl::Buffer(context, CL_MEM_READ_WRITE, len, NULL, &ret);
  if(ret != CL_SUCCESS){
    printf("Failed to create buffer on GPU. %d\n", ret);
    return ;
  }
}

void
nest::model_name_gpu::upload(clContext_ *clCxt, void *data, cl::Buffer &gdata, int datalen)
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
nest::model_name_gpu::download(clContext_ *clCxt, cl::Buffer &gdata,void *data,int data_len, int offset)
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
nest::model_name_gpu::synchronize()
{
  this->command_queue.finish();
}

int
nest::model_name_gpu::getKernel(string kernelName, string deliver_kernel_name, string static_deliver_kernel_name, clContext_ *clCxt)
{
  this->gpu_kernel = new cl::Kernel(program, kernelName.c_str());

  this->deliver_kernel = new cl::Kernel(program, deliver_kernel_name.c_str());

  this->static_deliver_kernel = new cl::Kernel(program, static_deliver_kernel_name.c_str());
  
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
nest::model_name_gpu::set_lag_args(cl::Kernel *kernel, long lag)
{
  cl_int ret;
  ret = kernel->setArg(1, static_cast<cl_long>(lag));
  if (ret != CL_SUCCESS) {						
    printf("Failed to set lag arg, error code %dn", ret);	
    return 1;								
  }									
  return 0;
}

int
nest::model_name_gpu::set_kernel_args(cl::Kernel *kernel, int num_nodes)
{
  cl_int ret;
  int arg_idx = 0;

  set_kernel_prim(num_nodes, cl_int);

  //set_kernel_prim(lag, cl_long);
  arg_idx++;

  set_kernel_prim(event_size, cl_int);

/* MODEL BUFFERS SET ARG */
/* RING BUFFERS SET ARG */
  set_kernel(d_spike_count);
  /* OUTPUT VAR SET ARG */
  set_kernel_prim(time_index, cl_int);

  return 0;
}

int
nest::model_name_gpu::set_deliver_kernel_args(cl::Kernel *kernel, int num_nodes, int batch_size)
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
  set_kernel_prim(this->update_type, cl_int);
  set_kernel_prim(conn_type, cl_int);
  set_kernel_prim(cp_lambda_, cl_double);
  set_kernel_prim(cp_mu_, cl_double);
  set_kernel_prim(cp_alpha_, cl_double);
  set_kernel_prim(cp_tau_plus_inv_, cl_double);
  set_kernel(d_ex_spikes_);
  set_kernel(d_in_spikes_);
  set_kernel_prim(time_index, cl_int);
  
  return 0;
}

int
nest::model_name_gpu::set_static_deliver_kernel_args(cl::Kernel *kernel, int num_nodes, int batch_size)
{
  cl_int ret;
  int arg_idx = 0;

  set_kernel_prim(num_nodes, cl_int);
  set_kernel_prim(event_size, cl_int);
  set_kernel(d_connections_ptr);
  set_kernel(d_connections);
  set_kernel(d_connections_weight);
  set_kernel_prim(batch_size, cl_int);
  set_kernel(d_spike_src);
  set_kernel(d_spike_multiplicity);
  set_kernel(d_spike_pos);
  set_kernel(d_ex_spikes_);
  set_kernel(d_in_spikes_);
  set_kernel_prim(time_index, cl_int);

  return 0;
}

void
nest::model_name_gpu::execute_kernel(cl::Kernel *kernel, clContext_ *clCxt, size_t num_nodes)
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
nest::model_name_gpu::fill_spike_event_buffer( Event& e)
{
}

void
nest::model_name_gpu::fill_event_buffer( SecondaryEvent& e)
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
nest::model_name_gpu::fill_buffer_zero_double(clContext_ *clCxt, cl::Buffer &buffer, int size)
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
nest::model_name_gpu::fill_buffer_zero_uint(clContext_ *clCxt, cl::Buffer &buffer, int size)
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
nest::model_name_gpu::initialize()
{
  if (not is_ring_buffer_ready)
    {

      if (initialize_command_queue())
	return;

      
      event_size = kernel().connection_manager.get_min_delay()
        + kernel().connection_manager.get_max_delay();
      ring_buffer_size = this->num_local_nodes * event_size;

      /* HOST RING BUFFERS ALLOC */

      /* DEVICE RING BUFFERS ALLOC */
      is_ring_buffer_ready = true;
    }

  for (vector<vector<connection_info> >::iterator tgt_it = connections.begin();
       tgt_it != connections.end();
       tgt_it++)
    {
      graph_size += (*tgt_it).size();
    }

  h_connections_ptr = new int[this->total_num_nodes + 1];
  h_connections = new int[graph_size];
  h_connections_weight = new double[graph_size];
  create(&gpu_context, &d_connections_ptr, (this->total_num_nodes + 1) * sizeof(int));
  create(&gpu_context, &d_connections, graph_size * sizeof(int));
  create(&gpu_context, &d_connections_weight, graph_size * sizeof(double));

  int i = 0;
  int node_id = 0;
  for (vector<vector<connection_info> >::iterator src_it = connections.begin();
       src_it != connections.end();
       src_it++, node_id++)
    {
      h_connections_ptr[node_id] = i;

      for (vector<connection_info>::iterator tgt_it = (*src_it).begin();
	   tgt_it != (*src_it).end();
	   tgt_it++, i++)
	{
	  connection_info& info = *tgt_it;
	  h_connections[i] = info.tgt_id;
	  h_connections_weight[i] = info.weight;
	}
    }

  for (size_t i = connections.size(); i <= this->total_num_nodes; i++)
    h_connections_ptr[i] = graph_size;
  synchronize();

  upload(&gpu_context, h_connections_ptr, d_connections_ptr, (this->total_num_nodes + 1) * sizeof(int));
  upload(&gpu_context, h_connections, d_connections, graph_size * sizeof(int));
  upload(&gpu_context, h_connections_weight, d_connections_weight, graph_size * sizeof(double));

  synchronize();

  getKernel("update", "deliver_events_stdp_pl", "deliver_events", &gpu_context);

  delete[] h_connections_ptr; h_connections_ptr = NULL;
  delete[] h_connections; h_connections = NULL;
  delete[] h_connections_weight; h_connections_weight = NULL;
}

void
nest::model_name_gpu::clear_buffer()
{
  for (unsigned int i = 0; i < ring_buffer_size; i++)
    {
      /* HOST RING BUFFERS CLEAR */
    }
  //   h_event_buffer[i] = 0.0;
  
  //fill_buffer_zero_double(&gpu_context, d_ring_buffer, ring_buffer_size);
}

void
nest::model_name_gpu::deliver_events()
{
  int batch_size = list_spikes.size();
  
  if (batch_size != 0)
    {
      h_spike_tgid = new int[batch_size];
      // h_t_spike = new double[batch_size];
      // h_dendritic_delay = new double[batch_size];
      h_weight_ = new double[batch_size];
      h_pos = new long[batch_size];
      h_t_lastspike = new double[batch_size];
      h_Kplus_ = new double[batch_size];
      h_conn_type_ = new int[batch_size];

      if (this->update_type == 2)
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

      if (this->update_type == 2)
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
	  if (this->update_type == 2)
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

      if (this->update_type == 2)
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
      if (this->update_type == 2)
	{
	  delete[] h_multiplicity;
	}
      delete[] h_Kplus_;
      delete[] h_conn_type_;
  
      list_spikes.clear();
    }
}

void
nest::model_name_gpu::deliver_static_events()
{
  int static_batch_size = list_sgid.size();

  if (static_batch_size != 0)
    {
      h_spike_src = new int[static_batch_size];
      h_spike_multiplicity = new int[static_batch_size];
      h_spike_pos = new long[static_batch_size];

      create(&gpu_context, &d_spike_src, static_batch_size*sizeof(int));
      create(&gpu_context, &d_spike_multiplicity, static_batch_size*sizeof(int));
      create(&gpu_context, &d_spike_pos, static_batch_size*sizeof(long));

      int i = 0;
      for (vector< SpikeEvent >::iterator it = list_sgid.begin();
	   it != list_sgid.end();
	   it++, i++)
	{
	  SpikeEvent &e = *it;
	  long pos = e.get_rel_delivery_steps(
					      kernel().simulation_manager.get_slice_origin() );
	  h_spike_src[i] = e.get_sender_gid();
	  h_spike_multiplicity[i] = e.get_multiplicity();
	  h_spike_pos[i] = pos;
	}
      upload(&gpu_context, (void*)h_spike_src, d_spike_src, static_batch_size*sizeof(int));
      upload(&gpu_context, (void*)h_spike_multiplicity, d_spike_multiplicity, static_batch_size*sizeof(int));
      upload(&gpu_context, (void*)h_spike_pos, d_spike_pos, static_batch_size*sizeof(long));

      synchronize();
      set_static_deliver_kernel_args(this->static_deliver_kernel, this->num_local_nodes, static_batch_size);
      execute_kernel(this->static_deliver_kernel, &gpu_context, this->num_local_nodes);

      synchronize();

      delete[] h_spike_src;
      delete[] h_spike_multiplicity;
      delete[] h_spike_pos;
      list_sgid.clear();
    }
}

void
nest::model_name_gpu::copy_event_data(std::vector<Node *> nodes)
{
}

void
nest::model_name_gpu::handle(index sgid, index tgid, double weight_)
{
  connection_info info;
  info.tgt_id = tgid;
  info.weight = weight_;
  connections[sgid].push_back(info);
}

/*HANDLE DEFINITION*/

void nest::model_name_gpu::pre_deliver_event(const std::vector< Node* > &nodes)
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

      nest::model_name* node = (nest::model_name*)*it;
      
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

void nest::model_name_gpu::post_deliver_event(const std::vector< Node* > &nodes)
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

      nest::model_name* node = (nest::model_name*)*it;
      hist_queue h_ = node->get_all_history();
      for (hist_queue::iterator h_it = h_.begin(); h_it != h_.end(); h_it++, hist_it++)
	{
	  h_it->access_counter_ = h_history_access_counter_[hist_it];
	}
    }

  delete[] h_history_Kminus_;
  delete[] h_history_t_;
  delete[] h_history_access_counter_;
}

void nest::model_name_gpu::handle(Event& ev, double last_t_spike, const CommonSynapseProperties *csp, void *conn, int conn_type)
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

void nest::model_name_gpu::insert_static_event(SpikeEvent& e)
{
  //SpikeEvent *e = (SpikeEvent *)&ev;

  /*int sgid = e.get_sender_gid();
  int tgid = e.get_receiver().get_thread_lid();

  synapse_info conn_info;
  conn_info.source_node = sgid;
  conn_info.target_node = tgid;
  conn_info.pos = e.get_rel_delivery_steps(
					   kernel().simulation_manager.get_slice_origin() );
  conn_info.weight = e.get_weight();
  conn_info.multiplicity = e.get_multiplicity();
  conn_info.type = 1;
  list_spikes.push_back(conn_info);*/
  list_sgid.push_back(e);
}

void nest::model_name_gpu::insert_event(SpikeEvent& e)
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
nest::model_name_gpu::advance_time()
{
  time_index = (time_index + kernel().connection_manager.get_min_delay()) % event_size;
}
