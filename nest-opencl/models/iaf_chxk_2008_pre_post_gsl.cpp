#include "iaf_chxk_2008.h"
// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

void nest::iaf_chxk_2008::pre_gsl(Time const& origin, long lag)
{
    double t = 0.0;

    // remember membrane potential at beginning of step
    // to check for *crossing*
    vm_prev = S_.y[ State_::V_M ];
   
}

void nest::iaf_chxk_2008::post_gsl(Time const& origin, long lag)
{
    // neuron should spike on threshold crossing only.
    if ( vm_prev < P_.V_th && S_.y[ State_::V_M ] >= P_.V_th )
    {
      // neuron is not absolute refractory

      // Find precise spike time using linear interpolation
      double sigma = ( S_.y[ State_::V_M ] - P_.V_th ) * B_.step_
        / ( S_.y[ State_::V_M ] - vm_prev );

      double alpha = exp( -sigma / P_.tau_ahp );

      double delta_g_ahp = V_.PSConInit_AHP * sigma * alpha;
      double delta_dg_ahp = V_.PSConInit_AHP * alpha;

      if ( P_.ahp_bug == true )
      {
        // Bug in original code ignores AHP conductance from previous spikes
        S_.y[ State_::G_AHP ] = delta_g_ahp;
        S_.y[ State_::DG_AHP ] = delta_dg_ahp;
      }
      else
      {
        S_.y[ State_::G_AHP ] += delta_g_ahp;
        S_.y[ State_::DG_AHP ] += delta_dg_ahp;
      }

      // log spike with Archiving_Node
      set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

      SpikeEvent se;
      se.set_offset( sigma );
      kernel().event_delivery_manager.send( *this, se, lag );
    }

    // add incoming spikes
    S_.y[ State_::DG_EXC ] += B_.spike_exc_.get_value( lag ) * V_.PSConInit_E;
    S_.y[ State_::DG_INH ] += B_.spike_inh_.get_value( lag ) * V_.PSConInit_I;

    // set new input current
    B_.I_stim_ = B_.currents_.get_value( lag );

    // log state data
    B_.logger_.record_data( origin.get_steps() + lag );
   
}
