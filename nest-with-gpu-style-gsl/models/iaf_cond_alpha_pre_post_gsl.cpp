#include "iaf_cond_alpha.h"
// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

void nest::iaf_cond_alpha::pre_gsl(Time const& origin, long lag)
{
   
}

void nest::iaf_cond_alpha::post_gsl(Time const& origin, long lag)
{
    // refractoriness and spike generation
    if ( S_.r )
    { // neuron is absolute refractory
      --S_.r;
      S_.y[ State_::V_M ] = P_.V_reset; // clamp potential
    }
    else
      // neuron is not absolute refractory
      if ( S_.y[ State_::V_M ] >= P_.V_th )
    {
      S_.r = V_.RefractoryCounts;
      S_.y[ State_::V_M ] = P_.V_reset;

      // log spike with Archiving_Node
      set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

      SpikeEvent se;
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
