#include "hh_psc_alpha.h"
// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

void nest::hh_psc_alpha::pre_gsl(Time const& origin, long lag)
{
    double t = 0.0;
    U_old = S_.y_[ State_::V_M ];
   
}

void nest::hh_psc_alpha::post_gsl(Time const& origin, long lag)
{
    S_.y_[ State_::DI_EXC ] +=
      B_.spike_exc_.get_value( lag ) * V_.PSCurrInit_E_;
    S_.y_[ State_::DI_INH ] +=
      B_.spike_inh_.get_value( lag ) * V_.PSCurrInit_I_;

    // sending spikes: crossing 0 mV, pseudo-refractoriness and local maximum...
    // refractory?
    if ( S_.r_ > 0 )
    {
      --S_.r_;
    }
    else
      // (    threshold    &&     maximum       )
      if ( S_.y_[ State_::V_M ] >= 0 && U_old > S_.y_[ State_::V_M ] )
    {
      S_.r_ = V_.RefractoryCounts_;

      set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

      SpikeEvent se;
      kernel().event_delivery_manager.send( *this, se, lag );
    }

    // log state data
    B_.logger_.record_data( origin.get_steps() + lag );

    // set new input current
    B_.I_stim_ = B_.currents_.get_value( lag );
   
}
