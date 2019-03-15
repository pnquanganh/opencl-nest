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

/* DEFINE STATES */

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

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
void AtomicAdd(__global double *val, double delta) {
  union {
    double f;
    ulong  i;
  } old;
  union {
    double f;
    ulong  i;
  } new;
  do {
    old.f = *val;
    new.f = old.f + delta;
  } while (atom_cmpxchg ( (volatile __global ulong *)val, old.i, new.i) != old.i);
}

double ring_buffer_get_value(__global double *ring_buffer, int ring_buffer_size, int num_nodes, int tid, long lag, int time_index)
{
  int ind = num_nodes * ((time_index + lag + ring_buffer_size) % ring_buffer_size) + tid;
  double val = ring_buffer[ind];
  ring_buffer[ind] = 0.0;
  return val;
}

void ring_buffer_add_value(__global double *ring_buffer, int ring_buffer_size, int num_nodes, int tgtid, long pos, double val, int time_index)
{
  int ind = num_nodes * ((time_index + pos + ring_buffer_size) % ring_buffer_size) + tgtid;
  ring_buffer[ind] += val;
}

void ring_buffer_atomic_add_value(__global double *ring_buffer, int ring_buffer_size, int num_nodes, int tgtid, long pos, double val, int time_index)
{
  int ind = num_nodes * ((time_index + pos + ring_buffer_size) % ring_buffer_size) + tgtid;
  AtomicAdd(ring_buffer + ind, val);
}

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
/* MODEL PARAMETER INPUT DECL */
		     double B_step_)
{
    unsigned int tid = get_global_id(0);

/* DEFINE DYNAMICS FUNCTION */
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
		
/* MODEL PARAMETER INPUT DECL */
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
					  
/* MODEL PARAMETER INPUT PASS */
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
					  
/* MODEL PARAMETER INPUT PASS */					  
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
					  
/* MODEL PARAMETER INPUT PASS */
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
					  
/* MODEL PARAMETER INPUT PASS */
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
					  
/* MODEL PARAMETER INPUT PASS */
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
					  
/* MODEL PARAMETER INPUT PASS */
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
					  
/* MODEL PARAMETER INPUT PASS */
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
/* MODEL PARAMETER INPUT DECL */
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
					   
/* MODEL PARAMETER INPUT PASS */
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
				  
/* MODEL PARAMETER INPUT PASS */
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
/* MODEL PARAMETER GSL INPUT DECL */
/* RING BUFFER INPUT DECL */		  
		  ///
		  __global double *B_step_,
		  __global double *h,
		  __global unsigned int *spike_count,
		  __global double *y,
		  int time_index)
{
    unsigned int tid = get_global_id(0);

    if (tid >= num_nodes)
	return;

    double r_con_state_eps_abs = con_state_eps_abs[tid];
    double r_con_state_eps_rel = con_state_eps_rel[tid];
    double r_con_state_a_y = con_state_a_y[tid];
    double r_con_state_a_dydt = con_state_a_dydt[tid];
    double step = B_step_[tid];

/* GSL BODY */
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

void get_history(double t1, double t2, int history_size, __global double *history_t_,
		 __global int *history_access_counter_, int *start, int *finish)
{
  *finish = history_size;

  if (history_size == 0)
    {
      *start = *finish;
      return;
    }
  else
    {
      int runner = 0;
      while ((runner != history_size) && (history_t_[runner] <= t1))
	{
	  ++runner;
	}
      *start = runner;
      while ((runner != history_size) && (history_t_[runner] <= t2))
	{
	  history_access_counter_[runner]++;
	  ++runner;
	}
      *finish = runner;
    }
}

double facilitate_(double w, double kplus, double lambda_, double mu_plus_, double Wmax_)
{
  double norm_w = ( w / Wmax_ )
    + ( lambda_ * pow( 1.0 - ( w / Wmax_ ), mu_plus_ ) * kplus );
  return norm_w < 1.0 ? norm_w * Wmax_ : Wmax_;
}

double depress_( double w, double kminus, double lambda_, double alpha_, double mu_minus_, double Wmax_ )
{
  double norm_w = ( w / Wmax_ )
    - ( alpha_ * lambda_ * pow( w / Wmax_, mu_minus_ ) * kminus );
  return norm_w > 0.0 ? norm_w * Wmax_ : 0.0;
}

double facilitate_pl_(double w, double kplus, double cp_lambda_, double cp_mu_)
{
  return w + ( cp_lambda_ * pow( w, cp_mu_ ) * kplus );
}

double depress_pl_( double w, double kminus, double cp_lambda_, double cp_alpha_ )
{
  double new_w = w - ( cp_lambda_ * cp_alpha_ * w * kminus );
  return new_w > 0.0 ? new_w : 0.0;
}

double get_K_value(double t, int history_size,
		   double K_minus_,
		   __global double *history_Kminus_,
		   __global double *history_t_,
		   double tau_minus_inv_)
{
  if (history_size == 0)
    {
      return K_minus_;
    }
  int i = history_size - 1;
  while (i >= 0)
    {
      if (t > history_t_[i])
	{
	  return (history_Kminus_[i] *
		  exp((history_t_[i] - t) * tau_minus_inv_));
	}
      i--;
    }
  return 0;
}

void insert_into_ring_buffer(int num_nodes,
			     int ring_buffer_size,
			     int tgtid,
			     long pos,
			     double weight,
			     int multiplicity,
			     __global double *ex_spikes_,
			     __global double *in_spikes_,
			     int time_index)
{
  const double s = weight * multiplicity;

  if ( weight > 0.0 )
    {
      ring_buffer_atomic_add_value(ex_spikes_, ring_buffer_size, num_nodes, tgtid, pos, s, time_index);
    }
  else
    {
      ring_buffer_atomic_add_value(in_spikes_, ring_buffer_size, num_nodes, tgtid, pos, s, time_index);
    }

}

void send_stdp(int num_nodes,
	       int ring_buffer_size,
	       int tgtid,
	       double t_spike,
	       double dendritic_delay,
	       double *weight_,
	       double *Kplus_,
	       int conn_type_,
	       double t_lastspike,
	       long pos,
	       int history_size,
	       __global double *history_Kminus_,
	       __global double *history_t_,
	       __global int *history_access_counter_,
	       double K_minus_,
	       double tau_minus_inv_,
	       double cp_lambda,
	       double cp_mu,
	       double cp_alpha,
	       double cp_tau_plus_inv,
	       __global double *ex_spikes_,
	       __global double *in_spikes_,
	       int multiplicity,
	       int time_index)
{
  if (conn_type_ == 2)
    {
      int start, finish;
      get_history(t_lastspike - dendritic_delay, t_spike - dendritic_delay, history_size,
		  history_t_, history_access_counter_, &start, &finish);

      double minus_dt;
      while (start != finish)
	{
	  minus_dt = t_lastspike - (history_t_[start] + dendritic_delay);
	  start++;
	  if (minus_dt == 0)
	    {
	      continue;
	    }
	  *weight_ = facilitate_pl_(*weight_, *Kplus_ * exp(minus_dt * cp_tau_plus_inv), cp_lambda, cp_mu);
	}

      double kminus = get_K_value(t_spike - dendritic_delay, history_size,
				  K_minus_,
				  history_Kminus_,
				  history_t_,
				  tau_minus_inv_);
      *weight_ = depress_pl_( *weight_, kminus, cp_lambda, cp_alpha );

      insert_into_ring_buffer(num_nodes,
			      ring_buffer_size,
			      tgtid,
			      pos,
			      *weight_,
			      multiplicity,
			      ex_spikes_,
			      in_spikes_,
			      time_index);
      *Kplus_ = *Kplus_ * exp( ( t_lastspike - t_spike ) * cp_tau_plus_inv ) + 1.0;
    }
  else if (conn_type_ == 1)
    {
      insert_into_ring_buffer(num_nodes,
      			      ring_buffer_size,
      			      tgtid,
      			      pos,
      			      *weight_,
      			      multiplicity,
      			      ex_spikes_,
      			      in_spikes_,
			      time_index);
    }
}

__kernel void deliver_events_stdp_pl(int num_nodes, int batch_size,
				     int ring_buffer_size,
				     __global int *d_history_ptr,
				     __global double *d_history_Kminus_,
				     __global double *d_history_t_,
				     __global int *d_history_access_counter_,
				     __global double *d_Kminus_,
				     __global double *d_tau_minus_inv_,
				     __global int *d_spike_tgid,
				     double d_t_spike,
				     double d_dendritic_delay,
				     __global double *d_weight_,
				     __global double *d_Kplus_,
				     __global int *d_conn_type_,
				     __global double *d_t_lastspike,
				     __global long *d_pos,
				     __global int *d_multiplicity,
				     int event_multiplicity,
				     int update_type,
				     int d_conn_type,
				     double cp_lambda,
				     double cp_mu,
				     double cp_alpha,
				     double cp_tau_plus_inv,
				     __global double *ex_spikes_,
				     __global double *in_spikes_,
				     int time_index)
{
  unsigned int tid = get_global_id(0);

  if (tid >= batch_size)
    return;
  
  int synapse_id = tid;
  int tgid = d_spike_tgid[synapse_id];
  double t_spike = d_t_spike;//[synapse_id];
  double dendritic_delay = d_dendritic_delay;//[synapse_id];
  double weight_ = d_weight_[synapse_id];
  double Kplus_ = d_Kplus_[synapse_id];
  int conn_type_ = d_conn_type == 1 ? 1 : d_conn_type_[synapse_id];
  double t_lastspike = d_t_lastspike[synapse_id];
  long pos = d_pos[synapse_id];
  int multiplicity = update_type == 2 ? d_multiplicity[synapse_id] : event_multiplicity;

  int history_index = d_history_ptr[tgid];
  int history_size = d_history_ptr[tgid + 1] - history_index;

  __global double *tgt_history_Kminus_ = d_history_Kminus_ + history_index;
  __global double *tgt_history_t_ = d_history_t_ + history_index;
  __global int *tgt_history_access_counter_ = d_history_access_counter_ + history_index;
  double K_minus_ = d_Kminus_[tgid];
  double tau_minus_inv_ = d_tau_minus_inv_[tgid];

  send_stdp(num_nodes,
	    ring_buffer_size,
	    tgid,
	    t_spike,
	    dendritic_delay,
	    &weight_,
	    &Kplus_,
	    conn_type_,
	    t_lastspike,
	    pos,
	    history_size,
	    tgt_history_Kminus_,
	    tgt_history_t_,
	    tgt_history_access_counter_,
	    K_minus_,
	    tau_minus_inv_,
	    cp_lambda,
	    cp_mu,
	    cp_alpha,
	    cp_tau_plus_inv,
	    ex_spikes_,
	    in_spikes_,
	    multiplicity,
	    time_index);
  //t_lastspike = t_spike;

  if (conn_type_ == 2)
    {
      d_weight_[synapse_id] = weight_;
      d_Kplus_[synapse_id] = Kplus_;
    }
}
