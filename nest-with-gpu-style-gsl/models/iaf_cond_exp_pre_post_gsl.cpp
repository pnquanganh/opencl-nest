#include "iaf_cond_exp.h"
// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

void nest::iaf_cond_exp::pre_gsl(Time const& origin, long lag)
{

}

void nest::iaf_cond_exp::post_gsl(Time const& origin, long lag)
{
    S_.y_[ State_::G_EXC ] += B_.spike_exc_.get_value( lag );
    S_.y_[ State_::G_INH ] += B_.spike_inh_.get_value( lag );

    // absolute refractory period
    if ( S_.r_ )
    { // neuron is absolute refractory
      --S_.r_;
      S_.y_[ State_::V_M ] = P_.V_reset_;
    }
    else
      // neuron is not absolute refractory
      if ( S_.y_[ State_::V_M ] >= P_.V_th_ )
    {
      S_.r_ = V_.RefractoryCounts_;
      S_.y_[ State_::V_M ] = P_.V_reset_;

      set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

      SpikeEvent se;
      kernel().event_delivery_manager.send( *this, se, lag );
    }

    // set new input current
    B_.I_stim_ = B_.currents_.get_value( lag );

    // log state data
    B_.logger_.record_data( origin.get_steps() + lag );
   
}
