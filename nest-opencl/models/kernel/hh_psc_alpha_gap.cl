#define WARP_SIZE 32

#define GSL_SUCCESS  0 
#define GSL_FAILURE  -1
#define GSL_CONTINUE -2  /* iteration has not converged */
#define GSL_EDOM     1   /* input domain error e.g sqrt(-1) */
#define GSL_ERANGE   2   /* output range error e.g. exp(1e100) */
#define GSL_EFAULT   3   /* invalid pointer */
#define GSL_EINVAL   4   /* invalid argument supplied by user */
#define GSL_EFAILED  5   /* generic failure */
#define GSL_EFACTOR  6   /* factorization failed */
#define GSL_ESANITY  7   /* sanity check failed - shouldn't happen */
#define GSL_ENOMEM   8   /* malloc failed */
#define GSL_EBADFUNC 9   /* problem with user-supplied function */
#define GSL_ERUNAWAY 10  /* iterative process is out of control */
#define GSL_EMAXITER 11  /* exceeded max number of iterations */
#define GSL_EZERODIV 12  /* tried to divide by zero */
#define GSL_EBADTOL  13  /* user specified an invalid tolerance */
#define GSL_ETOL     14  /* failed to reach the specified tolerance */
#define GSL_EUNDRFLW 15  /* underflow */
#define GSL_EOVRFLW  16  /* overflow  */
#define GSL_ELOSS    17  /* loss of accuracy */
#define GSL_EROUND   18  /* failed because of roundoff error */
#define GSL_EBADLEN  19  /* matrix vector lengths are not conformant */
#define GSL_ENOTSQR  20  /* matrix not square */
#define GSL_ESING    21  /* apparent singularity detected */
#define GSL_EDIVERGE 22  /* integral or series is divergent */
#define GSL_EUNSUP   23  /* requested feature is not supported by the hardware */
#define GSL_EUNIMPL  24  /* requested feature not (yet) implemented */
#define GSL_ECACHE   25  /* cache limit exceeded */
#define GSL_ETABLE   26  /* table limit exceeded */
#define GSL_ENOPROG  27  /* iteration is not making progress towards solution */
#define GSL_ENOPROGJ 28  /* jacobian evaluations are not improving the solution */
#define GSL_ETOLF    29  /* cannot reach the specified tolerance in F */
#define GSL_ETOLX    30  /* cannot reach the specified tolerance in X */
#define GSL_ETOLG    31  /* cannot reach the specified tolerance in gradient */
#define GSL_EOF      32   /* end of file */

#define V_M 0
#define HH_M 1
#define HH_H 2
#define HH_N 3
#define HH_P 4
#define DI_EXC 5
#define I_EXC 6
#define DI_INH 7
#define I_INH 8
#define STATE_VEC_SIZE 9

#define GSL_ODEIV_HADJ_INC   1  /* step was increased */
#define GSL_ODEIV_HADJ_NIL   0  /* step unchanged     */
#define GSL_ODEIV_HADJ_DEC (-1) /* step decreased     */

#define OPENCL_DBL_MEMCPY(dest, src, n, tid, nthreads)		\
    for (int i = 0; i < n; i++)					\
      {								\
	int idx = nthreads * i + tid;				\
	dest[idx] = src[idx];					\
      }

#define GSL_MAX_DBL(a,b) ((a) > (b) ? (a) : (b))

__constant double ah[] = { 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0 };
__constant double b3[] = { 3.0/32.0, 9.0/32.0 };
__constant double b4[] = { 1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0};
__constant double b5[] = { 8341.0/4104.0, -32832.0/4104.0, 29440.0/4104.0, -845.0/4104.0};
__constant double b6[] = { -6080.0/20520.0, 41040.0/20520.0, -28352.0/20520.0, 9295.0/20520.0, -5643.0/20520.0};

__constant double c1 = 902880.0/7618050.0;
__constant double c3 = 3953664.0/7618050.0;
__constant double c4 = 3855735.0/7618050.0;
__constant double c5 = -1371249.0/7618050.0;
__constant double c6 = 277020.0/7618050.0;

__constant double ec[] = { 0.0,
			   1.0 / 360.0,
			   0.0,
			   -128.0 / 4275.0,
			   -2197.0 / 75240.0,
			   1.0 / 50.0,
			   2.0 / 55.0
};

unsigned int
rkf45_order ()
{
    return 5;
}

void std_control_hadjust(int num_nodes,
			 double r_con_state_eps_abs,
			 double r_con_state_eps_rel,
			 double r_con_state_a_y,
			 double r_con_state_a_dydt,
			 int dimension,
			 unsigned int ord,
			 __global double *y,
			 __global double *yerr,
			 __global double *yp,
			 /*__private*/ double *h,
			 /*__private*/ int *ret)
{
    unsigned int tid = get_global_id(0);

    double eps_abs = r_con_state_eps_abs;
    double eps_rel = r_con_state_eps_rel;
    double a_y     = r_con_state_a_y;
    double a_dydt  = r_con_state_a_dydt;

    double S = 0.9;
    double h_old = *h;

    double rmax = DBL_MIN;
    int i;

    for(i=0; i<dimension; i++) {
	int idx = num_nodes * i + tid;
	double D0 = 
	    eps_rel * (a_y * fabs(y[idx]) + a_dydt * fabs(h_old * yp[idx])) + eps_abs;
	double r  = fabs(yerr[idx]) / fabs(D0);
	rmax = GSL_MAX_DBL(r, rmax);
    }

    if(rmax > 1.1) {
	/* decrease step, no more than factor of 5, but a fraction S more
	   than scaling suggests (for better accuracy) */
	double r =  S / pow(rmax, 1.0/ord);
    
	if (r < 0.2)
	    r = 0.2;

	*h = r * h_old;

	*ret = GSL_ODEIV_HADJ_DEC;
	return; 
    }
    else if(rmax < 0.5) {
	/* increase step, no more than factor of 5 */
	double r = S / pow(rmax, 1.0/(ord+1.0));

	if (r > 5.0)
	    r = 5.0;

	if (r < 1.0)  /* don't allow any decrease caused by S<1 */
	    r = 1.0;
        
	*h = r * h_old;

	*ret = GSL_ODEIV_HADJ_INC;
	return; 
    }
    else {
	/* no change */
	*ret = GSL_ODEIV_HADJ_NIL;
	return; 
    }
}

int hh_psc_alpha_gap_dynamics(int num_nodes,
			       double time, __global double *y, __global double *f,
			       double r_P_g_Na,
			       double r_P_g_Kv1,
			       double r_P_g_Kv3,
			       double r_P_g_L,
			       double r_P_C_m,
			       double r_P_E_Na,
			       double r_P_E_K,
			       double r_P_E_L, 
			       double r_P_tau_synE,
			       double r_P_tau_synI,
			       double r_P_I_e,
			       int wfr_interpolation_order,
			       double B_step_,
			      //long rB_lag_,
			       double r_B_sumj_g_ij_,
			       __global const double *B_interpolation_coefficients,
			       double r_B_I_stim_)
{
    unsigned int tid = get_global_id(0);

    double V = y[num_nodes * V_M + tid];
    double m = y[num_nodes * HH_M + tid];
    double h = y[num_nodes * HH_H + tid];
    double n = y[num_nodes * HH_N + tid];
    double p = y[num_nodes * HH_P + tid];
    double dI_ex = y[num_nodes * DI_EXC + tid];
    double I_ex = y[num_nodes * I_EXC + tid];
    double dI_in = y[num_nodes * DI_INH + tid];
    double I_in = y[num_nodes * I_INH + tid];

    double alpha_m =
	40. * ( V - 75.5 ) / ( 1. - exp( -( V - 75.5 ) / 13.5 ) );
    double beta_m = 1.2262 / exp( V / 42.248 );
    double alpha_h = 0.0035 / exp( V / 24.186 );
    double beta_h =
	0.017 * ( 51.25 + V ) / ( 1. - exp( -( 51.25 + V ) / 5.2 ) );
    double alpha_p = ( V - 95. ) / ( 1. - exp( -( V - 95. ) / 11.8 ) );
    double beta_p = 0.025 / exp( V / 22.222 );
    double alpha_n =
	0.014 * ( V + 44. ) / ( 1. - exp( -( V + 44. ) / 2.3 ) );
    double beta_n = 0.0043 / exp( ( V + 44. ) / 34. );
    double I_Na = r_P_g_Na * m * m * m * h * ( V - r_P_E_Na );
    double I_K = ( r_P_g_Kv1 * n * n * n * n + r_P_g_Kv3 * p * p )
	* ( V - r_P_E_K );
    double I_L = r_P_g_L * ( V - r_P_E_L );

    double gap = 0.0;

    const double t = time / B_step_;

    switch (wfr_interpolation_order)
    {
    case 0:
	gap = -r_B_sumj_g_ij_ * V
	  + B_interpolation_coefficients[ num_nodes * 0 + tid ];
	  //+ B_interpolation_coefficients[ num_nodes * B_lag_[tid] + tid ];
	break;

    case 1:
	gap = -r_B_sumj_g_ij_ * V
	    + B_interpolation_coefficients[ num_nodes * 0 + tid ]
	    + B_interpolation_coefficients[ num_nodes * 1 + tid ] * t;
	    /* + B_interpolation_coefficients[ num_nodes * (B_lag_[tid] * 2 + 0) + tid ] */
	    /* + B_interpolation_coefficients[ num_nodes * (B_lag_[tid] * 2 + 1) + tid ] * t; */
	break;

    case 3:
	gap = -r_B_sumj_g_ij_ * V
	    + B_interpolation_coefficients[ num_nodes * 0 + tid ]
	    + B_interpolation_coefficients[ num_nodes * 1 + tid ] * t
	    + B_interpolation_coefficients[ num_nodes * 2 + tid ] * t * t
	    + B_interpolation_coefficients[ num_nodes * 3 + tid ] * t * t * t;
	    /* + B_interpolation_coefficients[ num_nodes * (B_lag_[tid] * 4 + 0) + tid ] */
	    /* + B_interpolation_coefficients[ num_nodes * (B_lag_[tid] * 4 + 1) + tid ] * t */
	    /* + B_interpolation_coefficients[ num_nodes * (B_lag_[tid] * 4 + 2) + tid ] * t * t */
	    /* + B_interpolation_coefficients[ num_nodes * (B_lag_[tid] * 4 + 3) + tid ] * t * t * t; */
	break;

    default:
	return GSL_FAILURE;
	// default:
	//   throw BadProperty( "Interpolation order must be 0, 1, or 3." );
    }

    const double I_gap = gap;

    // V dot -- synaptic input are currents, inhib current is negative
    f[ num_nodes * V_M + tid] = ( -( I_Na + I_K + I_L ) + r_B_I_stim_ + r_P_I_e + I_ex
				  + I_in + I_gap ) / r_P_C_m;

    // channel dynamics
    int idx = num_nodes * HH_M + tid;
    double tmp = y[idx];
    f[idx] =
	alpha_m * ( 1 - tmp ) - beta_m * tmp; // m-variable

    idx = num_nodes * HH_H + tid;
    tmp = y[idx];
    f[idx] =
	alpha_h * ( 1 - tmp ) - beta_h * tmp; // h-variable

    idx = num_nodes * HH_P + tid;
    tmp = y[idx];
    f[idx] =
	alpha_p * ( 1 - tmp ) - beta_p * tmp; // p-variable

    idx = num_nodes * HH_N + tid;
    tmp = y[idx];
    f[idx] =
	alpha_n * ( 1 - tmp ) - beta_n * tmp; // n-variable

    // synapses: alpha functions
    f[num_nodes * DI_EXC + tid] = -dI_ex / r_P_tau_synE;
    f[num_nodes * I_EXC + tid] = dI_ex - ( I_ex / r_P_tau_synE );
    f[num_nodes * DI_INH + tid] = -dI_in / r_P_tau_synI;
    f[num_nodes * I_INH + tid] = dI_in - ( I_in / r_P_tau_synI);

    return GSL_SUCCESS;
}

int rkf45_apply(int num_nodes,
		int dimension,
		/*__private*/ double t,
		/*__private*/ double h,
		__global double *y,
		__global double *yerr,
		__global double *rk_state_k1,
		__global double *rk_state_k2,
		__global double *rk_state_k3,
		__global double *rk_state_k4,
		__global double *rk_state_k5,
		__global double *rk_state_k6,
		__global double *rk_state_y0,
		__global double *rk_state_ytmp,
		__global double *e_dydt_in,
		__global double *e_dydt_out,
		///
		double r_P_g_Na,
		double r_P_g_Kv1,
		double r_P_g_Kv3,
		double r_P_g_L,
		double r_P_C_m,
		double r_P_E_Na,
		double r_P_E_K,
		double r_P_E_L,
		double r_P_tau_synE,
		double r_P_tau_synI,
		double r_P_I_e,
		int wfr_interpolation_order,
		double B_step_,
		//long r_B_lag_,
		double r_B_sumj_g_ij_,
		__global const double *B_interpolation_coefficients,
		double r_B_I_stim_)
{
    unsigned int tid = get_global_id(0);

    int i;

    __global double *k1 = rk_state_k1;
    __global double *k2 = rk_state_k2;
    __global double *k3 = rk_state_k3;
    __global double *k4 = rk_state_k4;
    __global double *k5 = rk_state_k5;
    __global double *k6 = rk_state_k6;
    __global double *ytmp = rk_state_ytmp;
    //__global double *y0 = rk_state_y0;

    //OPENCL_DBL_MEMCPY(y0, y, dimension, tid, num_nodes);
	
    // for (int i = 0; i < dimension; i++)
    // 	y0[num_nodes * i + tid] = y[num_nodes * i + tid];

    /* k1 step */
    if (e_dydt_in != NULL)
    {
	OPENCL_DBL_MEMCPY(k1, e_dydt_in, dimension, tid, num_nodes);
    }
    else
    {
	int s = hh_psc_alpha_gap_dynamics(num_nodes,
					  t, y, k1,
					  r_P_g_Na,
					  r_P_g_Kv1,
					  r_P_g_Kv3,
					  r_P_g_L,
					  r_P_C_m,
					  r_P_E_Na,
					  r_P_E_K,
					  r_P_E_L, 
					  r_P_tau_synE,
					  r_P_tau_synI,
					  r_P_I_e,
					  wfr_interpolation_order,
					  B_step_,
					  //r_B_lag_,
					  r_B_sumj_g_ij_,
					  B_interpolation_coefficients,
					  r_B_I_stim_);
	if (s != GSL_SUCCESS)
	    return s;
    }


    for (i = 0; i < dimension; i++)
    {
	int idx = num_nodes * i + tid;
	ytmp[idx] = y[idx] + ah[0] * h * k1[idx];
    }

    /* k2 step */
    {
	int s = hh_psc_alpha_gap_dynamics(num_nodes,
					  t + ah[0] * h, ytmp, k2,
					  r_P_g_Na,
					  r_P_g_Kv1,
					  r_P_g_Kv3,
					  r_P_g_L,
					  r_P_C_m,
					  r_P_E_Na,
					  r_P_E_K,
					  r_P_E_L, 
					  r_P_tau_synE,
					  r_P_tau_synI,
					  r_P_I_e,
					  wfr_interpolation_order,
					  B_step_,
					  //r_B_lag_,
					  r_B_sumj_g_ij_,
					  B_interpolation_coefficients,
					  r_B_I_stim_);
	if (s != GSL_SUCCESS)
	    return s;
    }

    for (i = 0; i < dimension; i++)
    {
	int idx = num_nodes * i + tid;
	ytmp[idx] = y[idx] + h * (b3[0] * k1[idx] + b3[1] * k2[idx]);
    }

    /* k3 step */

    {
	int s = hh_psc_alpha_gap_dynamics(num_nodes,
					  t + ah[1] * h, ytmp, k3,
					  r_P_g_Na,
					  r_P_g_Kv1,
					  r_P_g_Kv3,
					  r_P_g_L,
					  r_P_C_m,
					  r_P_E_Na,
					  r_P_E_K,
					  r_P_E_L, 
					  r_P_tau_synE,
					  r_P_tau_synI,
					  r_P_I_e,
					  wfr_interpolation_order,
					  B_step_,
					  //r_B_lag_,
					  r_B_sumj_g_ij_,
					  B_interpolation_coefficients,
					  r_B_I_stim_);

	if (s != GSL_SUCCESS)
	    return s;
    }
    
    for (i = 0; i < dimension; i++)
    {
	int idx = num_nodes * i + tid;
	ytmp[idx] = y[idx] + h * (b4[0] * k1[idx] + b4[1] * k2[idx] + b4[2] * k3[idx]);
    }

    /* k4 step */
    {
	int s = hh_psc_alpha_gap_dynamics(num_nodes,
					  t + ah[2] * h, ytmp, k4,
					  r_P_g_Na,
					  r_P_g_Kv1,
					  r_P_g_Kv3,
					  r_P_g_L,
					  r_P_C_m,
					  r_P_E_Na,
					  r_P_E_K,
					  r_P_E_L, 
					  r_P_tau_synE,
					  r_P_tau_synI,
					  r_P_I_e,
					  wfr_interpolation_order,
					  B_step_,
					  //r_B_lag_,
					  r_B_sumj_g_ij_,
					  B_interpolation_coefficients,
					  r_B_I_stim_);

	if (s != GSL_SUCCESS)
	    return s;
    }


    for (i = 0; i < dimension; i++)
    {
	int idx = num_nodes * i + tid;
	ytmp[idx] = y[idx] + h * (b5[0] * k1[idx] + b5[1] * k2[idx] +
				  b5[2] * k3[idx] + b5[3] * k4[idx]);
    }

    /* k5 step */
    {
	int s = hh_psc_alpha_gap_dynamics(num_nodes,
					  t + ah[3] * h, ytmp, k5,
					  r_P_g_Na,
					  r_P_g_Kv1,
					  r_P_g_Kv3,
					  r_P_g_L,
					  r_P_C_m,
					  r_P_E_Na,
					  r_P_E_K,
					  r_P_E_L, 
					  r_P_tau_synE,
					  r_P_tau_synI,
					  r_P_I_e,
					  wfr_interpolation_order,
					  B_step_,
					  //r_B_lag_,
					  r_B_sumj_g_ij_,
					  B_interpolation_coefficients,
					  r_B_I_stim_);
	if (s != GSL_SUCCESS)
	    return s;
    }

    for (i = 0; i < dimension; i++)
    {
	int idx = num_nodes * i + tid;
	ytmp[idx] = y[idx] + h * (b6[0] * k1[idx] + b6[1] * k2[idx] + b6[2] * k3[idx] +
				  b6[3] * k4[idx] + b6[4] * k5[idx]);
    }

    /* k6 step and final sum */
    {
	int s = hh_psc_alpha_gap_dynamics(num_nodes,
					  t + ah[4] * h, ytmp, k6,
					  r_P_g_Na,
					  r_P_g_Kv1,
					  r_P_g_Kv3,
					  r_P_g_L,
					  r_P_C_m,
					  r_P_E_Na,
					  r_P_E_K,
					  r_P_E_L, 
					  r_P_tau_synE,
					  r_P_tau_synI,
					  r_P_I_e,
					  wfr_interpolation_order,
					  B_step_,
					  //r_B_lag_,
					  r_B_sumj_g_ij_,
					  B_interpolation_coefficients,
					  r_B_I_stim_);
	if (s != GSL_SUCCESS)
	    return s;
    }


    for (i = 0; i < dimension; i++)
    {
	int idx = num_nodes * i + tid;
	double d_i = c1 * k1[idx] + c3 * k3[idx] + c4 * k4[idx] + c5 * k5[idx] + c6 * k6[idx];
	y[idx] += h * d_i;
    }



    /* Derivatives at output */
    if (e_dydt_out != NULL)
    {
	int s = hh_psc_alpha_gap_dynamics(num_nodes,
					  t + h, y, e_dydt_out,
					  r_P_g_Na,
					  r_P_g_Kv1,
					  r_P_g_Kv3,
					  r_P_g_L,
					  r_P_C_m,
					  r_P_E_Na,
					  r_P_E_K,
					  r_P_E_L, 
					  r_P_tau_synE,
					  r_P_tau_synI,
					  r_P_I_e,
					  wfr_interpolation_order,
					  B_step_,
					  //r_B_lag_,
					  r_B_sumj_g_ij_,
					  B_interpolation_coefficients,
					  r_B_I_stim_);
	if (s != GSL_SUCCESS)
	    return s;
    }

    /* difference between 4th and 5th order */
    for (i = 0; i < dimension; i++)
    {
	int idx = num_nodes * i + tid;
	yerr[idx] = h * (ec[1] * k1[idx] + ec[3] * k3[idx] + ec[4] * k4[idx] 
			 + ec[5] * k5[idx] + ec[6] * k6[idx]);
    }
    return GSL_SUCCESS;
}

int gsl_odeiv_evolve_apply(int num_nodes, int dimension, __global double *e_y0,
			   __global double *e_yerr, __global double *e_dydt_in,
			   __global double *e_dydt_out,
			   /* __global double *e_last_step, */
			   /* __global unsigned long int *e_count, */
			   /* __global unsigned long int *e_failed_steps, */
			   double r_con_state_eps_abs,
			   double r_con_state_eps_rel,
			   double r_con_state_a_y,
			   double r_con_state_a_dydt,
			   __global double *rk_state_k1,
			   __global double *rk_state_k2,
			   __global double *rk_state_k3,
			   __global double *rk_state_k4,
			   __global double *rk_state_k5,
			   __global double *rk_state_k6,
			   __global double *rk_state_y0,
			   __global double *rk_state_ytmp,
			   ////
			   double r_P_g_Na,
			   double r_P_g_Kv1,
			   double r_P_g_Kv3,
			   double r_P_g_L,
			   double r_P_C_m,
			   double r_P_E_Na,
			   double r_P_E_K,
			   double r_P_E_L, 
			   double r_P_tau_synE,
			   double r_P_tau_synI,
			   double r_P_I_e,
			   int wfr_interpolation_order,
			   double B_step_,
			   //long r_B_lag_,
			   double r_B_sumj_g_ij_,
			   __global const double *B_interpolation_coefficients,
			   double r_B_I_stim_,
			   ///
			   /*__private*/ double *t,
			   __global double *h,
			   __global double *y)
{
    unsigned int tid = get_global_id(0);
    
    double t0 = *t;
    double h0 = h[tid];
    int step_status;
    int final_step = 0;
    double t1 = B_step_;
    double dt = t1 - t0;

    OPENCL_DBL_MEMCPY(e_y0, y, dimension, tid, num_nodes);
    // for (int i = 0; i < dimension; i++)
    //   e_y0[num_nodes * i + tid] = y[num_nodes * i + tid];

    int status = hh_psc_alpha_gap_dynamics(num_nodes,
					   t0, y, e_dydt_in,
					   r_P_g_Na,
					   r_P_g_Kv1,
					   r_P_g_Kv3,
					   r_P_g_L,
					   r_P_C_m,
					   r_P_E_Na,
					   r_P_E_K,
					   r_P_E_L, 
					   r_P_tau_synE,
					   r_P_tau_synI,
					   r_P_I_e,
					   wfr_interpolation_order,
					   B_step_,
					   //r_B_lag_,
					   r_B_sumj_g_ij_,
					   B_interpolation_coefficients,
					   r_B_I_stim_);

    if (status)
	return status;
    
    while (true)
    {
	if ((dt >= 0.0 && h0 > dt) || (dt < 0.0 && h0 < dt))
	{
	    h0 = dt;
	    final_step = 1;
	}
	else
	{
	    final_step = 0;
	}

	step_status = rkf45_apply(num_nodes,
				  dimension, t0, h0,
				  y, e_yerr,
				  rk_state_k1,
				  rk_state_k2,
				  rk_state_k3,
				  rk_state_k4,
				  rk_state_k5,
				  rk_state_k6,
				  rk_state_y0,
				  rk_state_ytmp,
				  e_dydt_in,
				  e_dydt_out,
///
				  r_P_g_Na,
				  r_P_g_Kv1,
				  r_P_g_Kv3,
				  r_P_g_L,
				  r_P_C_m,
				  r_P_E_Na,
				  r_P_E_K,
				  r_P_E_L, 
				  r_P_tau_synE,
				  r_P_tau_synI,
				  r_P_I_e,
				  wfr_interpolation_order,
				  B_step_,
				  //r_B_lag_,
				  r_B_sumj_g_ij_,
				  B_interpolation_coefficients,
				  r_B_I_stim_);


	//   /* Check for stepper internal failure */

	if (step_status != GSL_SUCCESS) 
	  {
	    *h = h0;  /* notify user of step-size which caused the failure */
	    *t = t0;  /* restore original t value */
	    return step_status;
	  }

	/* e_count[tid]++; */
	/* e_last_step[tid] = h0; */

	if (final_step)
	{
	    *t = t1;
	}
	else
	{
	    *t = t0 + h0;
	}

	double h_old = h0;

	int hadjust_status;
	std_control_hadjust(num_nodes,
			    r_con_state_eps_abs,
			    r_con_state_eps_rel,
			    r_con_state_a_y,
			    r_con_state_a_dydt,
			    dimension,
			    rkf45_order(),
			    y, e_yerr, e_dydt_out, &h0,
			    &hadjust_status);


	if (hadjust_status == GSL_ODEIV_HADJ_DEC)
	{
	    /* Check that the reported status is correct (i.e. an actual
	       decrease in h0 occured) and the suggested h0 will change
	       the time by at least 1 ulp */

	    double t_curr = *t;
	    double t_next = *t + h0;

	    if (fabs(h0) < fabs(h_old) && t_next != t_curr) 
	    {
		/* Step was decreased. Undo step, and try again with new h0. */
		//DBL_MEMCPY (y, e->y0, dydt->dimension);
		OPENCL_DBL_MEMCPY(y, e_y0, dimension, tid, num_nodes);
		//e_failed_steps[tid]++;

		//goto try_step;
	    }
	    else
	    {
		h0 = h_old; /* keep current step size */
		break;
	    }
	}
	else
	{
	    break;
	}
    }
    h[tid] = h0;  /* suggest step size for next time-step */
    return step_status;
}

__kernel void gsl(int num_nodes, int dimension,
		  int called_from_wfr_update, __global double *e_y0,
		  __global double *e_yerr, __global double *e_dydt_in,
		  __global double *e_dydt_out,
		  /* __global double *e_last_step, */
		  /* __global unsigned long int *e_count, */
		  /* __global unsigned long int *e_failed_steps, */
		  __global double *con_state_eps_abs,
		  __global double *con_state_eps_rel,
		  __global double *con_state_a_y,
		  __global double *con_state_a_dydt,
		  __global double *rk_state_k1,
		  __global double *rk_state_k2,
		  __global double *rk_state_k3,
		  __global double *rk_state_k4,
		  __global double *rk_state_k5,
		  __global double *rk_state_k6,
		  __global double *rk_state_y0,
		  __global double *rk_state_ytmp,
		  ////
		  __global const double *P_g_Na,
		  __global const double *P_g_Kv1,
		  __global const double *P_g_Kv3,
		  __global const double *P_g_L,
		  __global const double *P_C_m,
		  __global const double *P_E_Na,
		  __global const double *P_E_K,
		  __global const double *P_E_L, 
		  __global const double *P_tau_synE,
		  __global const double *P_tau_synI,
		  __global const double *P_I_e,
		  int wfr_interpolation_order,
		  __global double *B_step_,
		  //__global long *B_lag_,
		  __global const double *B_sumj_g_ij_,
		  __global const double *B_interpolation_coefficients,
		  __global double *new_coefficients,
		  __global const double *B_I_stim_,
		  //__global double *y_i,
		  __global double *f_temp,
		  /* __global double *hf_i, */
		  /* __global double *hf_ip1, */
		  /* __global double *y_ip1, */
		  __global double *B_spike_exc_,
		  __global double *B_spike_inh_,
		  //__global double *B_last_y_values,
		  ///
		  __global double *h,
		  __global double *y)
{
    unsigned int tid = get_global_id(0);
    double t = 0.0;

    if (tid >= num_nodes)
	return;

    double r_con_state_eps_abs = con_state_eps_abs[tid];
    double r_con_state_eps_rel = con_state_eps_rel[tid];
    double r_con_state_a_y = con_state_a_y[tid];
    double r_con_state_a_dydt = con_state_a_dydt[tid];

    double r_P_g_Na = P_g_Na[tid];
    double r_P_g_Kv1 = P_g_Kv1[tid];
    double r_P_g_Kv3 = P_g_Kv3[tid];
    double r_P_g_L = P_g_L[tid];
    double r_P_C_m = P_C_m[tid];
    double r_P_E_Na = P_E_Na[tid];
    double r_P_E_K = P_E_K[tid];
    double r_P_E_L = P_E_L[tid];
    double r_P_tau_synE = P_tau_synE[tid];
    double r_P_tau_synI = P_tau_synI[tid];
    double r_P_I_e = P_I_e[tid];
    double step = B_step_[tid];
    //long r_B_lag_ = B_lag_[tid];
    double r_B_sumj_g_ij_ = B_sumj_g_ij_[tid];
    double r_B_I_stim_ = B_I_stim_[tid];

    //U_old[tid] = y[num_nodes * V_M + tid];


    double r_y_i, r_hf_i, r_hf_ip1, r_y_ip1;
    
    if (called_from_wfr_update)
      {
	r_y_i = y[num_nodes * V_M + tid];
	
	if (wfr_interpolation_order == 3)
	  {
	    hh_psc_alpha_gap_dynamics(num_nodes,
				      0, y, f_temp,
				      r_P_g_Na,
				      r_P_g_Kv1,
				      r_P_g_Kv3,
				      r_P_g_L,
				      r_P_C_m,
				      r_P_E_Na,
				      r_P_E_K,
				      r_P_E_L, 
				      r_P_tau_synE,
				      r_P_tau_synI,
				      r_P_I_e,
				      wfr_interpolation_order,
				      step,
				      //r_B_lag_,
				      r_B_sumj_g_ij_,
				      B_interpolation_coefficients,
				      r_B_I_stim_);

	    r_hf_i = step * f_temp[num_nodes * V_M + tid];
	  }
      }
 
    while (t < step)
    {
	gsl_odeiv_evolve_apply(num_nodes, dimension, e_y0,
			       e_yerr, e_dydt_in,
			       e_dydt_out,
			       /* e_last_step, */
			       /* e_count, */
			       /* e_failed_steps, */
			       r_con_state_eps_abs,
			       r_con_state_eps_rel,
			       r_con_state_a_y,
			       r_con_state_a_dydt,
			       rk_state_k1,
			       rk_state_k2,
			       rk_state_k3,
			       rk_state_k4,
			       rk_state_k5,
			       rk_state_k6,
			       rk_state_y0,
			       rk_state_ytmp,
			       r_P_g_Na,
			       r_P_g_Kv1,
			       r_P_g_Kv3,
			       r_P_g_L,
			       r_P_C_m,
			       r_P_E_Na,
			       r_P_E_K,
			       r_P_E_L, 
			       r_P_tau_synE,
			       r_P_tau_synI,
			       r_P_I_e,
			       wfr_interpolation_order,
			       step,
			       //r_B_lag_,
			       r_B_sumj_g_ij_,
			       B_interpolation_coefficients,
			       r_B_I_stim_,
			       &t,
			       h,
			       y);
    }

    if (called_from_wfr_update)
      {
	y[num_nodes * DI_EXC + tid] += B_spike_exc_[tid];
	y[num_nodes * DI_INH + tid] += B_spike_inh_[tid];

	new_coefficients[ num_nodes * 0 + tid ] = r_y_i;

	switch (wfr_interpolation_order)
	  {
	  case 0:
	    break;
	
	  case 1:
	    r_y_ip1 = y[num_nodes * V_M + tid];
	    new_coefficients[num_nodes * 1 + tid] = r_y_ip1 - r_y_i;
	    break;
	
	  case 3:
	    r_y_ip1 = y[num_nodes * V_M + tid];
	
	    hh_psc_alpha_gap_dynamics(num_nodes,
				      step, y, f_temp,
				      r_P_g_Na,
				      r_P_g_Kv1,
				      r_P_g_Kv3,
				      r_P_g_L,
				      r_P_C_m,
				      r_P_E_Na,
				      r_P_E_K,
				      r_P_E_L, 
				      r_P_tau_synE,
				      r_P_tau_synI,
				      r_P_I_e,
				      wfr_interpolation_order,
				      step,
				      //r_B_lag_,
				      r_B_sumj_g_ij_,
				      B_interpolation_coefficients,
				      r_B_I_stim_);

	    r_hf_ip1 = step * f_temp[num_nodes * V_M + tid];

	    new_coefficients[num_nodes * 1 + tid] = r_hf_i;
	    new_coefficients[num_nodes * 2 + tid] =
	      -3 * r_y_i + 3 * r_y_ip1 - 2 * r_hf_i - r_hf_ip1;
	    new_coefficients[ num_nodes * 3 + tid] =
	      2 * r_y_i - 2 * r_y_ip1 + r_hf_i + r_hf_ip1;

	    break;
	  default:
	    break;
	  }
      }
    
}

__kernel void deliver_events(int num_nodes,
			     int event_size,
			     __global int *connections_ptr,
			     __global int *connections,
			     __global double *event_buffer,
			     __global double *event_weight,
			     __global double *coeff_buffer,
			     __global double *B_sumj_g_ij_)
{
  unsigned int tid = get_global_id(0);
  unsigned int wid = tid / WARP_SIZE;
  unsigned int lane = tid & (WARP_SIZE - 1);
  unsigned int total_threads = get_global_size(0);
  unsigned int total_warps = total_threads / WARP_SIZE;
  double sumj;

  for (int tgt_id = wid; tgt_id < num_nodes; tgt_id += total_warps)
    {
      if (lane == 0)
	sumj = B_sumj_g_ij_[tgt_id];
      
      __global double *tgt_coeff_buffer = coeff_buffer + event_size * tgt_id;
      
      int src_start = connections_ptr[tgt_id];
      int src_end = connections_ptr[tgt_id + 1];
      for (int i = src_start; i < src_end; i++)
	{
	  int src_id = connections[i];
	  double src_weight = event_weight[src_id];

	  sumj += src_weight;

	  __global double *src_event_buffer = event_buffer + event_size * src_id;
	  
	  for (int l = lane; l < event_size; l += WARP_SIZE)
	    {
	      tgt_coeff_buffer[l] += src_event_buffer[l];
	    }
	}

      if (lane == 0)
	B_sumj_g_ij_[tgt_id] = sumj;
    }
}
