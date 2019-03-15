#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "gsl_errno.h"
#include "gsl_odeiv.h"

#define GSL_MAX_DBL(a,b) ((a) > (b) ? (a) : (b))

static void *
std_control_alloc (void)
{
  std_control_state_t * s = 
    (std_control_state_t *) malloc (sizeof(std_control_state_t));

  return s;
}

static int
std_control_init (void * vstate, 
                  double eps_abs, double eps_rel, double a_y, double a_dydt)
{
  std_control_state_t * s = (std_control_state_t *) vstate;
  
  s->eps_rel = eps_rel;
  s->eps_abs = eps_abs;
  s->a_y = a_y;
  s->a_dydt = a_dydt;

  return GSL_SUCCESS;
}

int
std_control_hadjust(void * vstate, size_t dim, unsigned int ord, const double y[], const double yerr[], const double yp[], double * h)
{
  std_control_state_t *state = (std_control_state_t *) vstate;

  const double eps_abs = state->eps_abs;
  const double eps_rel = state->eps_rel;
  const double a_y     = state->a_y;
  const double a_dydt  = state->a_dydt;

  const double S = 0.9;
  const double h_old = *h;

  double rmax = DBL_MIN;
  size_t i;

  for(i=0; i<dim; i++) {
    const double D0 = 
      eps_rel * (a_y * fabs(y[i]) + a_dydt * fabs(h_old * yp[i])) + eps_abs;
    const double r  = fabs(yerr[i]) / fabs(D0);
    rmax = GSL_MAX_DBL(r, rmax);
  }

  if(rmax > 1.1) {
    /* decrease step, no more than factor of 5, but a fraction S more
       than scaling suggests (for better accuracy) */
    double r =  S / pow(rmax, 1.0/ord);
    
    if (r < 0.2)
      r = 0.2;

    *h = r * h_old;

    return GSL_ODEIV_HADJ_DEC;
  }
  else if(rmax < 0.5) {
    /* increase step, no more than factor of 5 */
    double r = S / pow(rmax, 1.0/(ord+1.0));

    if (r > 5.0)
      r = 5.0;

    if (r < 1.0)  /* don't allow any decrease caused by S<1 */
      r = 1.0;
        
    *h = r * h_old;

    return GSL_ODEIV_HADJ_INC;
  }
  else {
    /* no change */
    return GSL_ODEIV_HADJ_NIL;
  }
}


static void
std_control_free (void * vstate)
{
}

static const gsl_odeiv_control_type std_control_type =
{"standard",                    /* name */
 &std_control_alloc,
 &std_control_init,
 &std_control_hadjust,
 &std_control_free};

const gsl_odeiv_control_type *gsl_odeiv_control_standard = &std_control_type;


std_control_state_t *
gsl_odeiv_control_standard_new(double eps_abs, double eps_rel,
                               double a_y, double a_dydt)
{
  std_control_state_t * s = 
    std_control_alloc ();
  
  std_control_init (s, eps_abs, eps_rel, a_y, a_dydt);

  return s;
}

std_control_state_t *
gsl_odeiv_control_y_new(double eps_abs, double eps_rel)
{
  return gsl_odeiv_control_standard_new(eps_abs, eps_rel, 1.0, 0.0);
}


std_control_state_t *
gsl_odeiv_control_yp_new(double eps_abs, double eps_rel)
{
  return gsl_odeiv_control_standard_new(eps_abs, eps_rel, 0.0, 1.0);
}
