#ifndef MODEL_NAME_GPU_H
#define MODEL_NAME_GPU_H

#include "model_gpu.h"

#include "../gsl/gsl_errno.h"
  // #include "gsl_matrix.h"
#include "../gsl/gsl_odeiv.h"
  // #include "gsl_sf_exp.h"

#include "stdp_pl_connection_hom.h"
#include "../nestkernel/target_identifier.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

#include <string>
using namespace std;

#define MAX_SOURCE_SIZE (0x100000)


namespace nest
{

  class model_name_gpu : public model_gpu
  {
  public:

    model_name_gpu();
    ~model_name_gpu();

    void initialize_gpu();
    bool mass_wfr_update(const std::vector< Node* > &nodes, Time const& origin, const long from, const long to );
    void mass_update(const std::vector< Node* > &nodes, Time const& origin, const long from, const long to );
    void clear_buffer();
    void initialize();
    void fill_event_buffer(SecondaryEvent& e);
    void fill_spike_event_buffer(Event& e);
    
    void deliver_events();
    void deliver_static_events();
    void copy_event_data(std::vector< Node* > nodes);

    void handle(index sgid, index tgid);

    void pre_deliver_event(const std::vector< Node* > &nodes);
    void post_deliver_event(const std::vector< Node* > &nodes);

    void handle(Event &e, double last_t_spike, const CommonSynapseProperties *cp, void *conn, int type);
    void handle(index sgid, index tgid, double weight);
    void insert_event(SpikeEvent &e);
    void insert_static_event(SpikeEvent &e);
    
    /*HANDLE FUNCTIONS*/
    void advance_time();  
  private:

    struct clContext_
    {
      cl::Platform platform;
      std::vector<cl::Device> list_device;
    };

    static clContext_ gpu_context;
    bool is_data_ready;

    static cl::Context context;
    static cl::Program program;
    cl::CommandQueue command_queue;
    cl::Kernel *gpu_kernel;
    cl::Kernel *deliver_kernel;
    cl::Kernel *static_deliver_kernel;

    typedef struct connection_info {
      int tgt_id;
      double weight;      
    } connection_info;
    
    vector< vector<connection_info> > connections;
    vector< SpikeEvent > list_sgid;
    
    size_t graph_size;
    int *h_connections_ptr;
    int *h_connections;
    double *h_connections_weight;
    cl::Buffer d_connections_ptr;
    cl::Buffer d_connections;
    cl::Buffer d_connections_weight;
 
    size_t event_size;
    size_t ring_buffer_size;
    /*HOST RING BUFFER*/

    /*DEVICE RING BUFFER*/

    bool is_initialized;
    bool is_gpu_initialized;
    bool is_ring_buffer_ready;
  
/* DEVICE BUFFERS */

    cl::Buffer d_spike_count;

/* HOST BUFFERS */
    
    unsigned int* h_spike_count;

    int *h_history_ptr;
    double *h_history_Kminus_;
    double *h_history_t_;
    int *h_history_access_counter_;
    double *h_Kminus_;
    double *h_tau_minus_inv_;

    int *h_spike_tgid;
    double *h_t_spike;
    double *h_dendritic_delay;
    double *h_weight_;
    double *h_Kplus_;
    int *h_conn_type_;
    double *h_t_lastspike;
    long *h_pos;
    int *h_multiplicity;
    
    double cp_lambda_;
    double cp_mu_;
    double cp_alpha_;
    double cp_tau_plus_inv_;
    
    cl::Buffer d_history_ptr;
    cl::Buffer d_history_Kminus_;
    cl::Buffer d_history_t_;
    cl::Buffer d_history_access_counter_;
    cl::Buffer d_Kminus_;
    cl::Buffer d_tau_minus_inv_;

    cl::Buffer d_spike_tgid;
    cl::Buffer d_t_spike;
    cl::Buffer d_dendritic_delay;
    cl::Buffer d_weight_;
    cl::Buffer d_Kplus_;
    cl::Buffer d_conn_type_;
    cl::Buffer d_t_lastspike;
    cl::Buffer d_pos;
    cl::Buffer d_multiplicity;

    int *h_spike_src;
    int *h_spike_multiplicity;
    long *h_spike_pos;

    cl::Buffer d_spike_src;
    cl::Buffer d_spike_multiplicity;
    cl::Buffer d_spike_pos;
    
    struct synapse_info
    {
      int source_node;
      int target_node;
      double t_spike;
      double last_t_spike;
      double dendritic_delay;
      double weight;
      double Kplus;
      long pos;
      int multiplicity;
      int type;
      nest::STDPPLConnectionHom< TargetIdentifierIndex > *connection;
    };

    double event_t_spike;
    double event_dendritic_delay;
    int event_multiplicity;
    int conn_type;
    
    vector< synapse_info > list_spikes;

    int time_index;
    /* typedef std::vector< histentry > hist_queue; */
    /* hist_queue nodes_history; */

    size_t history_size;

    bool mass_update_( const std::vector< Node* > &nodes,
		       Time const& origin,
		       const long from,
		       const long to,
		       const bool called_from_wfr_update );

    int initialize_opencl_context();
    void initialize_device();
    int initialize_command_queue();
    //void prepare_copy_to_device(std::vector< Node* > &nodes, bool called_from_wfr_update, long lag_);
    void copy_data_to_device(const std::vector< Node* > &nodes);
    void copy_data_from_device(const std::vector< Node* > &nodes, bool last_copy);
    //int check_data(std::vector< Node* > &nodes, int dimension);
    void create(clContext_ *clCxt, cl::Buffer *mem, int len);
    void upload(clContext_ *clCxt, void *data,cl::Buffer &gdata,int datalen);
    void download(clContext_ *clCxt, cl::Buffer &gdata,void *data,int data_len, int offset = 0);
    int getKernel(string kernelName, string deliver_kernel_name, string static_deliver_kernel_name, clContext_ *clCxt);
    //int savebinary(cl_program &program, const char *fileName);
    int set_kernel_args(cl::Kernel *kernel, int num_nodes);
    int set_lag_args(cl::Kernel *kernel, long lag);

    int set_deliver_kernel_args(cl::Kernel *kernel, int num_nodes, int batch_size);
    int set_static_deliver_kernel_args(cl::Kernel *kernel, int num_nodes, int batch_size);
    void execute_kernel(cl::Kernel *kernel, clContext_ *clCxt, size_t num_nodes);
    void synchronize();

    void fill_buffer_zero_double(clContext_ *clCxt, cl::Buffer &buffer, int size);
    void fill_buffer_zero_uint(clContext_ *clCxt, cl::Buffer &buffer, int size);

  };
}

#endif // MODEL_NAME_GPU_H
