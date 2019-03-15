#ifndef AMAT2_PSC_EXP_GPU_H
#define AMAT2_PSC_EXP_GPU_H

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

  class amat2_psc_exp_gpu : public model_gpu
  {
  public:

    amat2_psc_exp_gpu();
    ~amat2_psc_exp_gpu();

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
double* h_spikes_ex_;
double* h_spikes_in_;

    cl::Buffer d_currents__buf;
cl::Buffer d_spikes_ex__buf;
cl::Buffer d_spikes_in__buf;
cl::Buffer d_currents_;
cl::Buffer d_spikes_ex_;
cl::Buffer d_spikes_in_;

    bool is_initialized;
  
cl::Buffer S__i_0_;
cl::Buffer V__P33_;
cl::Buffer V__P11_;
cl::Buffer P__I_e_;
cl::Buffer P__omega_;
cl::Buffer S__V_th_v_;
cl::Buffer V__RefractoryCountsTot_;
cl::Buffer S__I_syn_in_;
cl::Buffer S__V_th_2_;
cl::Buffer S__r_;
cl::Buffer S__I_syn_ex_;
cl::Buffer S__V_m_;
cl::Buffer V__P62_;
cl::Buffer V__P66_;
cl::Buffer V__P55_;
cl::Buffer V__P76_;
cl::Buffer V__P60_;
cl::Buffer V__P72_;
cl::Buffer S__V_th_dv_;
cl::Buffer V__P31_;
cl::Buffer V__P70_;
cl::Buffer V__P32_;
cl::Buffer V__P77_;
cl::Buffer P__alpha_1_;
cl::Buffer S__V_th_1_;
cl::Buffer V__P44_;
cl::Buffer V__P63_;
cl::Buffer V__P71_;
cl::Buffer P__alpha_2_;
cl::Buffer V__P61_;
cl::Buffer V__P22_;
cl::Buffer V__P30_;
cl::Buffer V__P73_;

    cl::Buffer d_spike_count;

double* h_S__i_0_;
double* h_V__P33_;
double* h_V__P11_;
double* h_P__I_e_;
double* h_P__omega_;
double* h_S__V_th_v_;
int* h_V__RefractoryCountsTot_;
double* h_S__I_syn_in_;
double* h_S__V_th_2_;
int* h_S__r_;
double* h_S__I_syn_ex_;
double* h_S__V_m_;
double* h_V__P62_;
double* h_V__P66_;
double* h_V__P55_;
double* h_V__P76_;
double* h_V__P60_;
double* h_V__P72_;
double* h_S__V_th_dv_;
double* h_V__P31_;
double* h_V__P70_;
double* h_V__P32_;
double* h_V__P77_;
double* h_P__alpha_1_;
double* h_S__V_th_1_;
double* h_V__P44_;
double* h_V__P63_;
double* h_V__P71_;
double* h_P__alpha_2_;
double* h_V__P61_;
double* h_V__P22_;
double* h_V__P30_;
double* h_V__P73_;
    
    unsigned int* h_spike_count;
    
    bool mass_update_( std::vector< Node* > &nodes,
		       Time const& origin,
		       const long from,
		       const long to,
		       const bool called_from_wfr_update );

    int initialize_opencl_context();
    void initialize_device(int total_num_nodes, int num_local_nodes);
    int initialize_command_queue(clContext_ *clCxt);
    //void prepare_copy_to_device(std::vector< Node* > &nodes, bool called_from_wfr_update, long lag_);
    void copy_data_to_device(std::vector< Node* > &nodes, bool first_loop, long lag);
    void copy_data_from_device(std::vector< Node* > &nodes, bool last_copy);
    //int check_data(std::vector< Node* > &nodes, int dimension);
    void create(clContext_ *clCxt, cl::Buffer *mem, int len);
    void upload(clContext_ *clCxt, void *data,cl::Buffer &gdata,int datalen);
    void download(clContext_ *clCxt, cl::Buffer &gdata,void *data,int data_len, int offset = 0);
    int getKernel(string source, string kernelName, string deliver_kernel_name, clContext_ *clCxt);
    //int savebinary(cl_program &program, const char *fileName);
    int set_kernel_args(cl::Kernel *kernel, int num_nodes);
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

#endif // AMAT2_PSC_EXP_GPU_H
