#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "gsl_errno.h"
#include "gsl_odeiv.h"

#include "odeiv_util.h"

gsl_odeiv_evolve *
gsl_odeiv_evolve_alloc (size_t dim)
{
  gsl_odeiv_evolve *e =
    (gsl_odeiv_evolve *) malloc (sizeof (gsl_odeiv_evolve));
  e->y0 = (double *) malloc (dim * sizeof (double));
  e->yerr = (double *) malloc (dim * sizeof (double));
  e->dydt_in = (double *) malloc (dim * sizeof (double));
  e->dydt_out = (double *) malloc (dim * sizeof (double));
  e->dimension = dim;
  e->count = 0;
  e->failed_steps = 0;
  e->last_step = 0.0;

  return e;
}

int
gsl_odeiv_evolve_reset (gsl_odeiv_evolve * e)
{
  e->count = 0;
  e->failed_steps = 0;
  e->last_step = 0.0;
  return GSL_SUCCESS;
}

void
gsl_odeiv_evolve_free (gsl_odeiv_evolve * e)
{
  RETURN_IF_NULL (e);
  free (e->dydt_out);
  free (e->dydt_in);
  free (e->yerr);
  free (e->y0);
  free (e);
}

/* Evolution framework method.
 *
 * Uses an adaptive step control object
 */
int
gsl_odeiv_evolve_apply (gsl_odeiv_evolve * e,
                        std_control_state_t * con_state,
                        rkf45_state_t * step, int dim,
                        const gsl_odeiv_system * dydt,
                        double *t, double t1, double *h, double y[])
{
  const double t0 = *t;
  double h0 = *h;
  int step_status;
  int final_step = 0;
  double dt = t1 - t0;  /* remaining time, possibly less than h */

  // printf("gsl_odeiv_evolve_apply called with args %d %.2f %.2f %.2f %.2f\n", dim, *t, t1, *h, y[0]);

  /* No need to copy if we cannot control the step size. */

  DBL_MEMCPY (e->y0, y, e->dimension);

  /* Calculate initial dydt once if the method can benefit. */

  int status = GSL_ODEIV_FN_EVAL (dydt, t0, y, e->dydt_in);

  if (status) 
    {
      return status;
    }

try_step:
    
  if ((dt >= 0.0 && h0 > dt) || (dt < 0.0 && h0 < dt))
    {
      h0 = dt;
      final_step = 1;
    }
  else
    {
      final_step = 0;
    }

    step_status = rkf45_apply(step, dim, t0, h0, y, e->yerr, e->dydt_in, e->dydt_out, dydt);

  /* Check for stepper internal failure */

  if (step_status != GSL_SUCCESS) 
    {
      *h = h0;  /* notify user of step-size which caused the failure */
      *t = t0;  /* restore original t value */
      return step_status;
    }

  e->count++;
  e->last_step = h0;

  if (final_step)
    {
      *t = t1;
    }
  else
    {
      *t = t0 + h0;
    }

   /* Check error and attempt to adjust the step. */

   double h_old = h0;

   const int hadjust_status 
     = std_control_hadjust (con_state, dim, rkf45_order(), y, e->yerr, e->dydt_out, &h0);

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
           DBL_MEMCPY (y, e->y0, dydt->dimension);
           e->failed_steps++;
           goto try_step;
         }
       else
         {
           h0 = h_old; /* keep current step size */
         }
     }

  *h = h0;  /* suggest step size for next time-step */

  return step_status;
}
