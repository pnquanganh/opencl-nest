#ifndef HH_PSC_ALPHA_GAP_GPU_H
#define HH_PSC_ALPHA_GAP_GPU_H

#include "model_gpu.h"

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

  class hh_psc_alpha_gap_gpu : public model_gpu
  {
  public:

    hh_psc_alpha_gap_gpu();
    ~hh_psc_alpha_gap_gpu();
    
    //bool built_connections;
    void initialize_gpu();
    bool mass_wfr_update(std::vector< Node* > nodes, Time const& origin, const long from, const long to );
    void mass_update(std::vector< Node* > nodes, Time const& origin, const long from, const long to );

    void initialize_graph();
    void fill_event_buffer(SecondaryEvent& e);
    void deliver_events();
    void copy_event_data(std::vector< Node* > nodes);
    //void check_event_data(std::vector< Node* > nodes);

    //std::vector< Node *> updated_nodes;

    void handle(int sgid, int tgid);
    
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
    double *h_event_buffer;
    double *h_coeff_buffer;
    double *h_event_weight;
    double *h_B_sumj;

    cl::Buffer d_event_buffer;
    cl::Buffer d_coeff_buffer;
    cl::Buffer d_event_weight;

    bool is_initialized;
  
    cl::Buffer e_y0;
    cl::Buffer e_yerr;
    cl::Buffer e_dydt_in;
    cl::Buffer e_dydt_out;
    /* cl::Buffer e_last_step; */
    /* cl::Buffer e_count; */
    /* cl::Buffer e_failed_steps; */
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
    cl::Buffer P_g_Na;
    cl::Buffer P_g_Kv1;
    cl::Buffer P_g_Kv3;
    cl::Buffer P_g_L;
    cl::Buffer P_C_m;
    cl::Buffer P_E_Na;
    cl::Buffer P_E_K;
    cl::Buffer P_E_L; 
    cl::Buffer P_tau_synE;
    cl::Buffer P_tau_synI;
    cl::Buffer P_I_e;
    cl::Buffer B_step_;
    cl::Buffer B_lag_;
    cl::Buffer B_sumj_g_ij_;
    cl::Buffer B_interpolation_coefficients;
    cl::Buffer d_new_coefficients;
    cl::Buffer B_I_stim_;
    cl::Buffer B_IntegrationStep_;
    cl::Buffer S_y_;
    cl::Buffer d_y_i;
    cl::Buffer d_f_temp;
    cl::Buffer d_hf_i;
    cl::Buffer d_hf_ip1;
    cl::Buffer B_spike_exc_;
    cl::Buffer B_spike_inh_;
    //cl::Buffer B_last_y_values;
    cl::Buffer d_y_ip1;

    cl::Buffer B_sumj;
    //cl::Buffer d_U_old;

    double* h_e_y0;
    double* h_e_yerr;
    double* h_e_dydt_in;
    double* h_e_dydt_out;
    /* double* h_e_last_step; */
    /* unsigned long int* h_e_count; */
    /* unsigned long int* h_e_failed_steps; */
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
    double* h_P_g_Na;
    double* h_P_g_Kv1;
    double* h_P_g_Kv3;
    double* h_P_g_L;
    double* h_P_C_m;
    double* h_P_E_Na;
    double* h_P_E_K;
    double* h_P_E_L; 
    double* h_P_tau_synE;
    double* h_P_tau_synI;
    double* h_P_I_e;
    double* h_B_step_;
    long* h_B_lag_;
    double* h_B_sumj_g_ij_;
    double* h_B_interpolation_coefficients;
    double* h_new_coefficients;
    double* h_B_I_stim_;
    double* h_B_IntegrationStep_;
    double* h_S_y_;
    double* h_y_i;
    double* h_f_temp;
    double* h_hf_i;
    double* h_hf_ip1;
    double* h_B_spike_exc_;
    double* h_B_spike_inh_;
    //double* h_B_last_y_values;
    double* h_y_ip1;

    //double* h_d_U_old;

    bool mass_update_( std::vector< Node* > &nodes,
		       Time const& origin,
		       const long from,
		       const long to,
		       const bool called_from_wfr_update );

    int initialize_opencl_context();
    void initialize_device(int total_num_nodes, int num_local_nodes, int dimension);
    int initialize_command_queue(clContext_ *clCxt);
    void prepare_copy_to_device(std::vector< Node* > &nodes, bool called_from_wfr_update, long lag_);
    void copy_data_to_device(std::vector< Node* > &nodes, int dimension, bool first_loop, bool called_from_wfr_update, long lag);
    void copy_data_from_device(std::vector< Node* > &nodes, int dimension, bool last_copy, bool called_from_wfr_update);
    //int check_data(std::vector< Node* > &nodes, int dimension);
    void create(clContext_ *clCxt, cl::Buffer *mem, int len);
    void upload(clContext_ *clCxt, void *data,cl::Buffer &gdata,int datalen);
    void download(clContext_ *clCxt, cl::Buffer &gdata,void *data,int data_len, int offset = 0);
    int getKernel(string source, string kernelName, string deliver_kernel_name, clContext_ *clCxt);
    //int savebinary(cl_program &program, const char *fileName);
    int set_kernel_args(cl::Kernel *kernel, int num_nodes, int dimension, int wfr_interpolation_order, bool called_from_wfr_update);

    int set_deliver_kernel_args(cl::Kernel *kernel, int num_nodes);
    void execute_kernel(cl::Kernel *kernel, clContext_ *clCxt, size_t num_nodes);
    void synchronize();

    void fill_buffer_zero(clContext_ *clCxt, cl::Buffer &buffer, int size);
    void upload_event_data(int num_nodes, int buffer_size);
    void download_event_data(int num_nodes, int buffer_size);

    
  };
}

#endif // HH_PSC_ALPHA_GAP_GPU_H
