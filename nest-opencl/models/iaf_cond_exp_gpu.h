#ifndef IAF_COND_EXP_GPU_H
#define IAF_COND_EXP_GPU_H

#include "model_gpu.h"

#include "../gsl/gsl_errno.h"
  // #include "gsl_matrix.h"
#include "../gsl/gsl_odeiv.h"
  // #include "gsl_sf_exp.h"

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

  class iaf_cond_exp_gpu : public model_gpu
  {
  public:

    iaf_cond_exp_gpu();
    ~iaf_cond_exp_gpu();

    void initialize_gpu();
    bool mass_wfr_update(std::vector< Node* > nodes, Time const& origin, const long from, const long to );
    void mass_update(std::vector< Node* > nodes, Time const& origin, const long from, const long to );

    void initialize_graph();
    void fill_event_buffer(SecondaryEvent& e);
    void fill_spike_event_buffer(Event& e);
    void clear_buffer();
    void deliver_events();
    void copy_event_data(std::vector< Node* > nodes);

    void handle(index sgid, index tgid);

    void handle( SpikeEvent& e );
void handle( CurrentEvent& e );
    
  private:
    struct clContext_
    {
      cl::Platform platform;
      std::vector<cl::Device> list_device;
    };

    static clContext_ gpu_context;
    static bool is_gpu_initialized;

    cl::Context context;
    cl::Program program;
    std::vector<cl::Device> devices;
    cl::CommandQueue command_queue;
    cl::Kernel *gpu_kernel;
    cl::Kernel *deliver_kernel;

    vector< vector<size_t> > connections;
    size_t graph_size;

    int *h_connections_ptr;
    int *h_connections;
    cl::Buffer d_connections_ptr;
    cl::Buffer d_connections;

    size_t event_size;
    size_t ring_buffer_size;
    double* h_currents_;
double* h_spike_exc_;
double* h_spike_inh_;

    cl::Buffer d_currents__buf;
cl::Buffer d_spike_exc__buf;
cl::Buffer d_spike_inh__buf;
cl::Buffer d_currents_;
cl::Buffer d_spike_exc_;
cl::Buffer d_spike_inh_;

    bool is_initialized;
  
    cl::Buffer e_y0;
    cl::Buffer e_yerr;
    cl::Buffer e_dydt_in;
    cl::Buffer e_dydt_out;
    cl::Buffer con_state_eps_abs;
    cl::Buffer con_state_eps_rel;
    cl::Buffer con_state_a_y;
    cl::Buffer con_state_a_dydt;
    cl::Buffer rk_state_k1;
    cl::Buffer rk_state_k2;
    cl::Buffer rk_state_k3;
    cl::Buffer rk_state_k4;
    cl::Buffer rk_state_k5;
    cl::Buffer rk_state_k6;
    cl::Buffer rk_state_y0;
    cl::Buffer rk_state_ytmp;

cl::Buffer P__E_in;
cl::Buffer P__V_th_;
cl::Buffer P__E_ex;
cl::Buffer V__RefractoryCounts_;
cl::Buffer P__C_m;
cl::Buffer B__step_;
cl::Buffer P__tau_synE;
cl::Buffer P__V_reset_;
cl::Buffer P__tau_synI;
cl::Buffer B__I_stim_;
cl::Buffer B__IntegrationStep_;
cl::Buffer P__g_L;
cl::Buffer S__r_;
cl::Buffer P__E_L;
cl::Buffer S__y_;
cl::Buffer P__I_e;

    cl::Buffer B_step_;
    cl::Buffer B_IntegrationStep_;

    cl::Buffer d_spike_count;

    double* h_e_y0;
    double* h_e_yerr;
    double* h_e_dydt_in;
    double* h_e_dydt_out;
    double* h_con_state_eps_abs;
    double* h_con_state_eps_rel;
    double* h_con_state_a_y;
    double* h_con_state_a_dydt;
    double* h_rk_state_k1;
    double* h_rk_state_k2;
    double* h_rk_state_k3;
    double* h_rk_state_k4;
    double* h_rk_state_k5;
    double* h_rk_state_k6;
    double* h_rk_state_y0;
    double* h_rk_state_ytmp;

double* h_P__E_in;
double* h_P__V_th_;
double* h_P__E_ex;
int* h_V__RefractoryCounts_;
double* h_P__C_m;
double* h_B__step_;
double* h_P__tau_synE;
double* h_P__V_reset_;
double* h_P__tau_synI;
double* h_B__I_stim_;
double* h_B__IntegrationStep_;
double* h_P__g_L;
int* h_S__r_;
double* h_P__E_L;
double* h_S__y_;
double* h_P__I_e;
    
    double* h_B_step_;
    double* h_B_IntegrationStep_;

    unsigned int* h_spike_count;
    
    bool mass_update_( std::vector< Node* > &nodes,
		       Time const& origin,
		       const long from,
		       const long to,
		       const bool called_from_wfr_update );

    int initialize_opencl_context();
    void initialize_device(int total_num_nodes, int num_local_nodes, int dimension);
    int initialize_command_queue(clContext_ *clCxt);
    //void prepare_copy_to_device(std::vector< Node* > &nodes, bool called_from_wfr_update, long lag_);
    void copy_data_to_device(std::vector< Node* > &nodes, int dimension, bool first_loop, long lag);
    void copy_data_from_device(std::vector< Node* > &nodes, int dimension, bool last_copy);
    //int check_data(std::vector< Node* > &nodes, int dimension);
    void create(clContext_ *clCxt, cl::Buffer *mem, int len);
    void upload(clContext_ *clCxt, void *data,cl::Buffer &gdata,int datalen);
    void download(clContext_ *clCxt, cl::Buffer &gdata,void *data,int data_len, int offset = 0);
    int getKernel(string source, string kernelName, string deliver_kernel_name, clContext_ *clCxt);
    //int savebinary(cl_program &program, const char *fileName);
    int set_kernel_args(cl::Kernel *kernel, int num_nodes, int dimension);
    int set_lag_args(cl::Kernel *kernel, long lag);

    int set_deliver_kernel_args(cl::Kernel *kernel, int num_nodes, cl::Buffer &d_event_buffer_in, cl::Buffer &d_event_buffer_out);
    void execute_kernel(cl::Kernel *kernel, clContext_ *clCxt, size_t num_nodes);
    void synchronize();

    void fill_buffer_zero_double(clContext_ *clCxt, cl::Buffer &buffer, int size);
    void fill_buffer_zero_uint(clContext_ *clCxt, cl::Buffer &buffer, int size);
    void upload_event_data(double *h_event_buffer, cl::Buffer &d_event_buffer_in, cl::Buffer &d_event_buffer_out);
    void download_event_data(int num_nodes, int buffer_size);

    void ring_buffer_add_value(double *h_ring_buffer, long pos, double val);

    void deliver_events(double *h_event_buffer, cl::Buffer &d_event_buffer_in, cl::Buffer &d_event_buffer_out);
  };
}

#endif // IAF_COND_EXP_GPU_H
