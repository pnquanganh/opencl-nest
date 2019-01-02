#include "hh_cond_exp_traub.h"
// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

void nest::hh_cond_exp_traub::pre_gsl(Time const& origin, long lag)
{
    double tt = 0.0; // it's all relative!
    V_.U_old_ = S_.y_[ State_::V_M ];
}

void nest::hh_cond_exp_traub::post_gsl(Time const& origin, long lag)
{
    S_.y_[ State_::G_EXC ] += B_.spike_exc_.get_value( lag );
    S_.y_[ State_::G_INH ] += B_.spike_inh_.get_value( lag );

    // sending spikes: crossing 0 mV, pseudo-refractoriness and local maximum...
    // refractory?
    if ( S_.r_ )
    {
      --S_.r_;
    }
    else
    {
      // (threshold   &&    maximum    )
      if ( S_.y_[ State_::V_M ] >= P_.V_T + 30.
        && V_.U_old_ > S_.y_[ State_::V_M ] )
      {
        S_.r_ = V_.refractory_counts_;

        set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );

        SpikeEvent se;
        kernel().event_delivery_manager.send( *this, se, lag );
      }
    }

    // set new input current
    B_.I_stim_ = B_.currents_.get_value( lag );

    // log state data
    B_.logger_.record_data( origin.get_steps() + lag );
}
