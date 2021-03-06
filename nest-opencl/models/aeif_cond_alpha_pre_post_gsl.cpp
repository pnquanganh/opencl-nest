#include "aeif_cond_alpha.h"
// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

void nest::aeif_cond_alpha::pre_gsl(Time const& origin, long lag)
{

}

void nest::aeif_cond_alpha::post_gsl(Time const& origin, long lag)
{

    if ( S_.r_ > 0 )
    {
      --S_.r_;
    }

    S_.y_[ State_::DG_EXC ] += B_.spike_exc_.get_value( lag ) * V_.g0_ex_;
    S_.y_[ State_::DG_INH ] += B_.spike_inh_.get_value( lag ) * V_.g0_in_;

    B_.I_stim_ = B_.currents_.get_value( lag );

    B_.logger_.record_data( origin.get_steps() + lag );
   
}
