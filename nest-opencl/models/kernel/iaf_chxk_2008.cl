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
#define DG_EXC 1
#define G_EXC 2
#define DG_INH 3
#define G_INH 4
#define DG_AHP 5
#define G_AHP 6
#define STATE_VEC_SIZE 7

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


int dynamic_function(int num_nodes,
			      double time, __global double *y, __global double *f,
double *P__tau_ahp,
double *P__E_in,
double *P__tau_synE,
double *P__E_ex,
double *P__C_m,
double *B__step_,
double *P__tau_synI,
double *B__I_stim_,
double *P__E_ahp,
double *B__IntegrationStep_,
double *V__PSConInit_AHP,
bool *P__ahp_bug,
double *V__PSConInit_E,
double *P__V_th,
double *V__PSConInit_I,
double *P__E_L,
double *P__I_e,
double *P__g_L,
		     double B_step_)
{
    unsigned int tid = get_global_id(0);

{
  // a shorthand
  /*typedef nest::iaf_chxk_2008::State_ S;

  // get access to node so we can almost work as in a member function
  assert( pnode );
  const nest::iaf_chxk_2008& node =
    *( reinterpret_cast< nest::iaf_chxk_2008* >( pnode ) );*/
    

  // y[] here is---and must be---the state vector supplied by the integrator,
  // not the state vector in the node, node.S_.y[].

  // The following code is verbose for the sake of clarity. We assume that a
  // good compiler will optimize the verbosity away ...
  const double I_syn_exc = y[ G_EXC*num_nodes + tid ] * ( y[ V_M*num_nodes + tid ] - P__E_ex[tid] );
  const double I_syn_inh = y[ G_INH*num_nodes + tid ] * ( y[ V_M*num_nodes + tid ] - P__E_in[tid] );
  const double I_ahp = y[ G_AHP*num_nodes + tid ] * ( y[ V_M*num_nodes + tid ] - P__E_ahp[tid] );
  const double I_leak = P__g_L[tid] * ( y[ V_M*num_nodes + tid ] - P__E_L[tid] );

  // dV_m/dt
  f[ V_M*num_nodes + tid ] = ( -I_leak - I_syn_exc - I_syn_inh - I_ahp + B__I_stim_[tid]
                  + P__I_e[tid] ) / P__C_m[tid];

  // d dg_exc/dt, dg_exc/dt
  f[ DG_EXC*num_nodes + tid ] = -y[ DG_EXC*num_nodes + tid ] / P__tau_synE[tid];
  f[ G_EXC*num_nodes + tid ] = y[ DG_EXC*num_nodes + tid ] - ( y[ G_EXC*num_nodes + tid ] / P__tau_synE[tid] );

  // d dg_inh/dt, dg_inh/dt
  f[ DG_INH*num_nodes + tid ] = -y[ DG_INH*num_nodes + tid ] / P__tau_synI[tid];
  f[ G_INH*num_nodes + tid ] = y[ DG_INH*num_nodes + tid ] - ( y[ G_INH*num_nodes + tid ] / P__tau_synI[tid] );

  // d dg_ahp/dt, dg_ahp/dt
  f[ DG_AHP*num_nodes + tid ] = -y[ DG_AHP*num_nodes + tid ] / P__tau_ahp[tid];
  f[ G_AHP*num_nodes + tid ] = y[ DG_AHP*num_nodes + tid ] - ( y[ G_AHP*num_nodes + tid ] / P__tau_ahp[tid] );

  return GSL_SUCCESS;
}
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
		
double *P__tau_ahp,
double *P__E_in,
double *P__tau_synE,
double *P__E_ex,
double *P__C_m,
double *B__step_,
double *P__tau_synI,
double *B__I_stim_,
double *P__E_ahp,
double *B__IntegrationStep_,
double *V__PSConInit_AHP,
bool *P__ahp_bug,
double *V__PSConInit_E,
double *P__V_th,
double *V__PSConInit_I,
double *P__E_L,
double *P__I_e,
double *P__g_L,
		double B_step_)
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
	int s = dynamic_function(num_nodes,
				 t, y, k1,
					  
P__tau_ahp,
P__E_in,
P__tau_synE,
P__E_ex,
P__C_m,
B__step_,
P__tau_synI,
B__I_stim_,
P__E_ahp,
B__IntegrationStep_,
V__PSConInit_AHP,
P__ahp_bug,
V__PSConInit_E,
P__V_th,
V__PSConInit_I,
P__E_L,
P__I_e,
P__g_L,
				 B_step_);
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
	int s = dynamic_function(num_nodes,
				 t + ah[0] * h, ytmp, k2,
					  
P__tau_ahp,
P__E_in,
P__tau_synE,
P__E_ex,
P__C_m,
B__step_,
P__tau_synI,
B__I_stim_,
P__E_ahp,
B__IntegrationStep_,
V__PSConInit_AHP,
P__ahp_bug,
V__PSConInit_E,
P__V_th,
V__PSConInit_I,
P__E_L,
P__I_e,
P__g_L,					  
				 B_step_);
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
	int s = dynamic_function(num_nodes,
					  t + ah[1] * h, ytmp, k3,
					  
P__tau_ahp,
P__E_in,
P__tau_synE,
P__E_ex,
P__C_m,
B__step_,
P__tau_synI,
B__I_stim_,
P__E_ahp,
B__IntegrationStep_,
V__PSConInit_AHP,
P__ahp_bug,
V__PSConInit_E,
P__V_th,
V__PSConInit_I,
P__E_L,
P__I_e,
P__g_L,
				 B_step_);

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
	int s = dynamic_function(num_nodes,
					  t + ah[2] * h, ytmp, k4,
					  
P__tau_ahp,
P__E_in,
P__tau_synE,
P__E_ex,
P__C_m,
B__step_,
P__tau_synI,
B__I_stim_,
P__E_ahp,
B__IntegrationStep_,
V__PSConInit_AHP,
P__ahp_bug,
V__PSConInit_E,
P__V_th,
V__PSConInit_I,
P__E_L,
P__I_e,
P__g_L,
				 B_step_);

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
	int s = dynamic_function(num_nodes,
					  t + ah[3] * h, ytmp, k5,
					  
P__tau_ahp,
P__E_in,
P__tau_synE,
P__E_ex,
P__C_m,
B__step_,
P__tau_synI,
B__I_stim_,
P__E_ahp,
B__IntegrationStep_,
V__PSConInit_AHP,
P__ahp_bug,
V__PSConInit_E,
P__V_th,
V__PSConInit_I,
P__E_L,
P__I_e,
P__g_L,
				 B_step_);
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
	int s = dynamic_function(num_nodes,
					  t + ah[4] * h, ytmp, k6,
					  
P__tau_ahp,
P__E_in,
P__tau_synE,
P__E_ex,
P__C_m,
B__step_,
P__tau_synI,
B__I_stim_,
P__E_ahp,
B__IntegrationStep_,
V__PSConInit_AHP,
P__ahp_bug,
V__PSConInit_E,
P__V_th,
V__PSConInit_I,
P__E_L,
P__I_e,
P__g_L,
				 B_step_);
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
	int s = dynamic_function(num_nodes,
					  t + h, y, e_dydt_out,
					  
P__tau_ahp,
P__E_in,
P__tau_synE,
P__E_ex,
P__C_m,
B__step_,
P__tau_synI,
B__I_stim_,
P__E_ahp,
B__IntegrationStep_,
V__PSConInit_AHP,
P__ahp_bug,
V__PSConInit_E,
P__V_th,
V__PSConInit_I,
P__E_L,
P__I_e,
P__g_L,
				 B_step_);
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
double *P__tau_ahp,
double *P__E_in,
double *P__tau_synE,
double *P__E_ex,
double *P__C_m,
double *B__step_,
double *P__tau_synI,
double *B__I_stim_,
double *P__E_ahp,
double *B__IntegrationStep_,
double *V__PSConInit_AHP,
bool *P__ahp_bug,
double *V__PSConInit_E,
double *P__V_th,
double *V__PSConInit_I,
double *P__E_L,
double *P__I_e,
double *P__g_L,
			   double B_step_,
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

    int status = dynamic_function(num_nodes,
					   t0, y, e_dydt_in,
					   
P__tau_ahp,
P__E_in,
P__tau_synE,
P__E_ex,
P__C_m,
B__step_,
P__tau_synI,
B__I_stim_,
P__E_ahp,
B__IntegrationStep_,
V__PSConInit_AHP,
P__ahp_bug,
V__PSConInit_E,
P__V_th,
V__PSConInit_I,
P__E_L,
P__I_e,
P__g_L,
				  B_step_);

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
				  
P__tau_ahp,
P__E_in,
P__tau_synE,
P__E_ex,
P__C_m,
B__step_,
P__tau_synI,
B__I_stim_,
P__E_ahp,
B__IntegrationStep_,
V__PSConInit_AHP,
P__ahp_bug,
V__PSConInit_E,
P__V_th,
V__PSConInit_I,
P__E_L,
P__I_e,
P__g_L,
				  B_step_);


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

double ring_buffer_get_value(double *ring_buffer, int ring_buffer_size, int num_nodes, int tid, long lag)
{
  return ring_buffer[num_nodes * (lag % ring_buffer_size) + tid];
}

__kernel void gsl(int num_nodes, int dimension,
		  long lag,
		  int ring_buffer_size,
		  __global double *e_y0,
		  __global double *e_yerr,
		  __global double *e_dydt_in,
		  __global double *e_dydt_out,
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
__global double *P__tau_ahp,
__global double *P__E_in,
__global double *P__tau_synE,
__global double *P__E_ex,
__global double *P__C_m,
__global double *B__step_,
__global double *P__tau_synI,
__global double *B__I_stim_,
__global double *P__E_ahp,
__global double *B__IntegrationStep_,
__global double *V__PSConInit_AHP,
__global bool *P__ahp_bug,
__global double *V__PSConInit_E,
__global double *P__V_th,
__global double *V__PSConInit_I,
__global double *P__E_L,
__global double *P__I_e,
__global double *P__g_L,
__global double *currents_,
__global double *spike_exc_,
__global double *spike_inh_,		  
		  ///
		  __global double *B_step_,
		  __global double *h,
		  __global unsigned int *spike_count,
		  __global double *y)
{
    unsigned int tid = get_global_id(0);

    if (tid >= num_nodes)
	return;

    double r_con_state_eps_abs = con_state_eps_abs[tid];
    double r_con_state_eps_rel = con_state_eps_rel[tid];
    double r_con_state_a_y = con_state_a_y[tid];
    double r_con_state_a_dydt = con_state_a_dydt[tid];
    double step = B_step_[tid];


  {

    double t = 0.0;

    // remember membrane potential at beginning of step
    // to check for *crossing*
    const double vm_prev = S__y[ V_M*num_nodes + tid ];

    // numerical integration with adaptive step size control:
    // ------------------------------------------------------
    // gsl_odeiv_evolve_apply performs only a single numerical
    // integration step, starting from t and bounded by step;
    // the while-loop ensures integration over the whole simulation
    // step (0, step] if more than one integration step is needed due
    // to a small integration step size;
    // note that (t+IntegrationStep > step) leads to integration over
    // (t, step] and afterwards setting t to step, but it does not
    // enforce setting IntegrationStep to step-t; this is of advantage
    // for a consistent and efficient integration across subsequent
    // simulation intervals
    while ( t < B__step_[tid] )
    {
      const int status =  gsl_odeiv_evolve_apply(num_nodes, dimension, e_y0,
						  e_yerr, e_dydt_in,
						  e_dydt_out,
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
P__tau_ahp,
P__E_in,
P__tau_synE,
P__E_ex,
P__C_m,
B__step_,
P__tau_synI,
B__I_stim_,
P__E_ahp,
B__IntegrationStep_,
V__PSConInit_AHP,
P__ahp_bug,
V__PSConInit_E,
P__V_th,
V__PSConInit_I,
P__E_L,
P__I_e,
P__g_L,
						  step,
						  &t,
						  h,
						  y);

      /*gsl_odeiv_evolve_apply( B_.e_,
        B_.c_,
        B_.s_,
        &B_.sys_,             // system of ODE
        &t,                   // from t
        B__step_[tid],             // to t <= step
        &B__IntegrationStep_[tid], // integration step size
        S__y )*/
        ;               // neuronal state
      if ( status != GSL_SUCCESS )
      {
        return;;
      }
    }
    // neuron should spike on threshold crossing only.
    if ( vm_prev < P__V_th[tid] && S__y[ V_M*num_nodes + tid ] >= P__V_th[tid] )
    {
      // neuron is not absolute refractory

      // Find precise spike time using linear interpolation
      double sigma = ( S__y[ V_M*num_nodes + tid ] - P__V_th[tid] ) * B__step_[tid]
        / ( S__y[ V_M*num_nodes + tid ] - vm_prev );

      double alpha = exp( -sigma / P__tau_ahp[tid] );

      double delta_g_ahp = V__PSConInit_AHP[tid] * sigma * alpha;
      double delta_dg_ahp = V__PSConInit_AHP[tid] * alpha;

      if ( P__ahp_bug[tid] == true )
      {
        // Bug in original code ignores AHP conductance from previous spikes
        S__y[ G_AHP*num_nodes + tid ] = delta_g_ahp;
        S__y[ DG_AHP*num_nodes + tid ] = delta_dg_ahp;
      }
      else
      {
        S__y[ G_AHP*num_nodes + tid ] += delta_g_ahp;
        S__y[ DG_AHP*num_nodes + tid ] += delta_dg_ahp;
      }

      // log spike with Archiving_Node
      spike_count[tid]++;

      /*set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

      SpikeEvent se;
      se.set_offset( sigma );
      kernel().event_delivery_manager.send( *this, se, lag )*/
      ;
    }

    // add incoming spikes
    S__y[ DG_EXC*num_nodes + tid ] += ring_buffer_get_value(spike_exc_, ring_buffer_size, num_nodes, tid, lag) * V__PSConInit_E[tid];
    S__y[ DG_INH*num_nodes + tid ] += ring_buffer_get_value(spike_inh_, ring_buffer_size, num_nodes, tid, lag) * V__PSConInit_I[tid];

    // set new input current
    B__I_stim_[tid] = ring_buffer_get_value(currents_, ring_buffer_size, num_nodes, tid, lag);

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
