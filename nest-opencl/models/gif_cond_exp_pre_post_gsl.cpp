#include "gif_cond_exp.h"
// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

void nest::gif_cond_exp::pre_gsl(Time const& origin, long lag)
{
    // exponential decaying stc and sfa elements
    S_.stc_ = 0.0;
    for ( size_t i = 0; i < S_.stc_elems_.size(); i++ )
    {
      S_.stc_ += S_.stc_elems_[ i ];
      S_.stc_elems_[ i ] = V_.P_stc_[ i ] * S_.stc_elems_[ i ];
    }

    S_.sfa_ = P_.V_T_star_;
    for ( size_t i = 0; i < S_.sfa_elems_.size(); i++ )
    {
      S_.sfa_ += S_.sfa_elems_[ i ];
      S_.sfa_elems_[ i ] = V_.P_sfa_[ i ] * S_.sfa_elems_[ i ];
    }
   
}

void nest::gif_cond_exp::post_gsl(Time const& origin, long lag)
{
    S_.neuron_state_[ State_::G_EXC ] += B_.spike_exc_.get_value( lag );
    S_.neuron_state_[ State_::G_INH ] += B_.spike_inh_.get_value( lag );

    if ( S_.r_ref_ == 0 ) // neuron is not in refractory period
    {

      const double lambda =
        P_.lambda_0_ * std::exp( ( S_.neuron_state_[ State_::V_M ] - S_.sfa_ )
                         / P_.Delta_V_ );

      if ( lambda > 0.0 )
      {

        // Draw random number and compare to prob to have a spike
        // hazard function is computed by 1 - exp(- lambda * dt)
        if ( V_.rng_->drand()
          < -numerics::expm1( -lambda * Time::get_resolution().get_ms() ) )
        {

          for ( size_t i = 0; i < S_.stc_elems_.size(); i++ )
          {
            S_.stc_elems_[ i ] += P_.q_stc_[ i ];
          }

          for ( size_t i = 0; i < S_.sfa_elems_.size(); i++ )
          {
            S_.sfa_elems_[ i ] += P_.q_sfa_[ i ];
          }

          S_.r_ref_ = V_.RefractoryCounts_;

          // And send the spike event
          set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );
          SpikeEvent se;
          kernel().event_delivery_manager.send( *this, se, lag );
        }
      }
    }
    else
    { // neuron is absolute refractory
      --S_.r_ref_;
      S_.neuron_state_[ State_::V_M ] = P_.V_reset_;
    }

    // Set new input current
    S_.I_stim_ = B_.currents_.get_value( lag );

    // Voltage logging
    B_.logger_.record_data( origin.get_steps() + lag );
   
}
