/*
 *  hh_psc_alpha_gap.cpp
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

// #include "nestkernel/gsl/"
#include "hh_psc_alpha_gap.h"

// #ifdef HAVE_GSL

// C++ includes:
#include <cmath> // in case we need isnan() // fabs
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"

nest::RecordablesMap< nest::hh_psc_alpha_gap >
  nest::hh_psc_alpha_gap::recordablesMap_;

namespace nest
{
// Override the create() method with one call to RecordablesMap::insert_()
// for each quantity to be recorded.
template <>
void
RecordablesMap< hh_psc_alpha_gap >::create()
{
  // use standard names whereever you can for consistency!
  insert_( names::V_m,
    &hh_psc_alpha_gap::get_y_elem_< hh_psc_alpha_gap::State_::V_M > );
  insert_( names::I_syn_ex,
    &hh_psc_alpha_gap::get_y_elem_< hh_psc_alpha_gap::State_::I_EXC > );
  insert_( names::I_syn_in,
    &hh_psc_alpha_gap::get_y_elem_< hh_psc_alpha_gap::State_::I_INH > );
  insert_( names::Act_m,
    &hh_psc_alpha_gap::get_y_elem_< hh_psc_alpha_gap::State_::HH_M > );
  insert_( names::Act_h,
    &hh_psc_alpha_gap::get_y_elem_< hh_psc_alpha_gap::State_::HH_H > );
  insert_( names::Inact_n,
    &hh_psc_alpha_gap::get_y_elem_< hh_psc_alpha_gap::State_::HH_N > );
  insert_( names::Inact_p,
    &hh_psc_alpha_gap::get_y_elem_< hh_psc_alpha_gap::State_::HH_P > );
}

extern "C" int
hh_psc_alpha_gap_dynamics( double time,
  const double y[],
  double f[],
  void* pnode )
{
  // printf("\nentering hh_psc_alpha_gap_dynamics\n");
  static int call_count = 0;
  call_count++;
  // if(call_count % 1000000 == 0) {
  //   printf("\n%d calls to hh_psc_alpha_gap_dynamics\n", call_count);
  // }

  // a shorthand
  typedef nest::hh_psc_alpha_gap::State_ S;

  // get access to node so we can almost work as in a member function
  assert( pnode );
  const nest::hh_psc_alpha_gap& node =
    *( reinterpret_cast< nest::hh_psc_alpha_gap* >( pnode ) );

  // y[] here is---and must be---the state vector supplied by the integrator,
  // not the state vector in the node, node.S_.y[].

  // The following code is verbose for the sake of clarity. We assume that a
  // good compiler will optimize the verbosity away ...

  // shorthand for state variables
  const double& V = y[ S::V_M ];
  const double& m = y[ S::HH_M ];
  const double& h = y[ S::HH_H ];
  const double& n = y[ S::HH_N ];
  const double& p = y[ S::HH_P ];
  const double& dI_ex = y[ S::DI_EXC ];
  const double& I_ex = y[ S::I_EXC ];
  const double& dI_in = y[ S::DI_INH ];
  const double& I_in = y[ S::I_INH ];

  const double alpha_m =
    40. * ( V - 75.5 ) / ( 1. - std::exp( -( V - 75.5 ) / 13.5 ) );
  const double beta_m = 1.2262 / std::exp( V / 42.248 );
  const double alpha_h = 0.0035 / std::exp( V / 24.186 );
  const double beta_h =
    0.017 * ( 51.25 + V ) / ( 1. - std::exp( -( 51.25 + V ) / 5.2 ) );
  const double alpha_p = ( V - 95. ) / ( 1. - std::exp( -( V - 95. ) / 11.8 ) );
  const double beta_p = 0.025 / std::exp( V / 22.222 );
  const double alpha_n =
    0.014 * ( V + 44. ) / ( 1. - std::exp( -( V + 44. ) / 2.3 ) );
  const double beta_n = 0.0043 / std::exp( ( V + 44. ) / 34. );
  const double I_Na = node.P_.g_Na * m * m * m * h * ( V - node.P_.E_Na );
  const double I_K = ( node.P_.g_Kv1 * n * n * n * n + node.P_.g_Kv3 * p * p )
    * ( V - node.P_.E_K );
  const double I_L = node.P_.g_L * ( V - node.P_.E_L );

  // set I_gap depending on interpolation order
  double gap = 0.0;

  const double t = time / node.B_.step_;

  switch ( kernel().simulation_manager.get_wfr_interpolation_order() )
  {
  case 0:
    gap = -node.B_.sumj_g_ij_ * V
      + node.B_.interpolation_coefficients[ node.B_.lag_ ];
    break;

  case 1:
    gap = -node.B_.sumj_g_ij_ * V
      + node.B_.interpolation_coefficients[ node.B_.lag_ * 2 + 0 ]
      + node.B_.interpolation_coefficients[ node.B_.lag_ * 2 + 1 ] * t;
    break;

  case 3:
    gap = -node.B_.sumj_g_ij_ * V
      + node.B_.interpolation_coefficients[ node.B_.lag_ * 4 + 0 ]
      + node.B_.interpolation_coefficients[ node.B_.lag_ * 4 + 1 ] * t
      + node.B_.interpolation_coefficients[ node.B_.lag_ * 4 + 2 ] * t * t
      + node.B_.interpolation_coefficients[ node.B_.lag_ * 4 + 3 ] * t * t * t;
    break;

  default:
    throw BadProperty( "Interpolation order must be 0, 1, or 3." );
  }

  const double I_gap = gap;

  // V dot -- synaptic input are currents, inhib current is negative
  f[ S::V_M ] = ( -( I_Na + I_K + I_L ) + node.B_.I_stim_ + node.P_.I_e + I_ex
                  + I_in + I_gap ) / node.P_.C_m;

  // channel dynamics
  f[ S::HH_M ] =
    alpha_m * ( 1 - y[ S::HH_M ] ) - beta_m * y[ S::HH_M ]; // m-variable
  f[ S::HH_H ] =
    alpha_h * ( 1 - y[ S::HH_H ] ) - beta_h * y[ S::HH_H ]; // h-variable
  f[ S::HH_P ] =
    alpha_p * ( 1 - y[ S::HH_P ] ) - beta_p * y[ S::HH_P ]; // p-variable
  f[ S::HH_N ] =
    alpha_n * ( 1 - y[ S::HH_N ] ) - beta_n * y[ S::HH_N ]; // n-variable

  // synapses: alpha functions
  f[ S::DI_EXC ] = -dI_ex / node.P_.tau_synE;
  f[ S::I_EXC ] = dI_ex - ( I_ex / node.P_.tau_synE );
  f[ S::DI_INH ] = -dI_in / node.P_.tau_synI;
  f[ S::I_INH ] = dI_in - ( I_in / node.P_.tau_synI );

  return GSL_SUCCESS;
}
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

nest::hh_psc_alpha_gap::Parameters_::Parameters_()
  : t_ref_( 2.0 )   // ms
  , g_Na( 4500. )   // nS
  , g_Kv1( 9.0 )    // nS
  , g_Kv3( 9000.0 ) // nS
  , g_L( 10.0 )     // nS
  , C_m( 40.0 )     // pF
  , E_Na( 74.0 )    // mV
  , E_K( -90.0 )    // mV
  , E_L( -70. )     // mV
  , tau_synE( 0.2 ) // ms
  , tau_synI( 2.0 ) // ms
  , I_e( 0.0 )      // pA
{
}

nest::hh_psc_alpha_gap::State_::State_( const Parameters_& )
  : r_( 0 )
{
  y_[ 0 ] = -69.60401191631222; // p.E_L;
  //'Inact_n': 0.0005741576228359798, 'Inact_p': 0.00025113182271506364
  //'Act_h': 0.8684620412943986,
  for ( size_t i = 1; i < STATE_VEC_SIZE; ++i )
  {
    y_[ i ] = 0;
  }

  // equilibrium values for (in)activation variables
  const double alpha_m =
    40. * ( y_[ 0 ] - 75.5 ) / ( 1. - std::exp( -( y_[ 0 ] - 75.5 ) / 13.5 ) );
  const double beta_m = 1.2262 / std::exp( y_[ 0 ] / 42.248 );
  const double alpha_h = 0.0035 / std::exp( y_[ 0 ] / 24.186 );
  const double beta_h = 0.017 * ( 51.25 + y_[ 0 ] )
    / ( 1. - std::exp( -( 51.25 + y_[ 0 ] ) / 5.2 ) );
  const double alpha_p =
    ( y_[ 0 ] - 95. ) / ( 1. - std::exp( -( y_[ 0 ] - 95. ) / 11.8 ) );
  const double beta_p = 0.025 / std::exp( y_[ 0 ] / 22.222 );
  const double alpha_n =
    0.014 * ( y_[ 0 ] + 44. ) / ( 1. - std::exp( -( y_[ 0 ] + 44. ) / 2.3 ) );
  const double beta_n = 0.0043 / std::exp( ( y_[ 0 ] + 44. ) / 34. );

  y_[ HH_H ] = alpha_h / ( alpha_h + beta_h );
  y_[ HH_N ] = alpha_n / ( alpha_n + beta_n );
  y_[ HH_M ] = alpha_m / ( alpha_m + beta_m );
  y_[ HH_P ] = alpha_p / ( alpha_p + beta_p );
}

nest::hh_psc_alpha_gap::State_::State_( const State_& s )
  : r_( s.r_ )
{
  for ( size_t i = 0; i < STATE_VEC_SIZE; ++i )
  {
    y_[ i ] = s.y_[ i ];
  }
}

nest::hh_psc_alpha_gap::State_& nest::hh_psc_alpha_gap::State_::operator=(
  const State_& s )
{
  assert( this != &s ); // would be bad logical error in program
  for ( size_t i = 0; i < STATE_VEC_SIZE; ++i )
  {
    y_[ i ] = s.y_[ i ];
  }
  r_ = s.r_;
  return *this;
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
nest::hh_psc_alpha_gap::Parameters_::get( DictionaryDatum& d ) const
{
  def< double >( d, names::t_ref, t_ref_ );
  def< double >( d, names::g_Na, g_Na );
  def< double >( d, names::g_Kv1, g_Kv1 );
  def< double >( d, names::g_Kv3, g_Kv3 );
  def< double >( d, names::g_L, g_L );
  def< double >( d, names::E_Na, E_Na );
  def< double >( d, names::E_K, E_K );
  def< double >( d, names::E_L, E_L );
  def< double >( d, names::C_m, C_m );
  def< double >( d, names::tau_syn_ex, tau_synE );
  def< double >( d, names::tau_syn_in, tau_synI );
  def< double >( d, names::I_e, I_e );
}

void
nest::hh_psc_alpha_gap::Parameters_::set( const DictionaryDatum& d )
{
  updateValue< double >( d, names::t_ref, t_ref_ );
  updateValue< double >( d, names::C_m, C_m );
  updateValue< double >( d, names::g_Na, g_Na );
  updateValue< double >( d, names::E_Na, E_Na );
  updateValue< double >( d, names::g_Kv1, g_Kv1 );
  updateValue< double >( d, names::g_Kv3, g_Kv3 );
  updateValue< double >( d, names::E_K, E_K );
  updateValue< double >( d, names::g_L, g_L );
  updateValue< double >( d, names::E_L, E_L );

  updateValue< double >( d, names::tau_syn_ex, tau_synE );
  updateValue< double >( d, names::tau_syn_in, tau_synI );

  updateValue< double >( d, names::I_e, I_e );
  if ( C_m <= 0 )
  {
    throw BadProperty( "Capacitance must be strictly positive." );
  }
  if ( t_ref_ < 0 )
  {
    throw BadProperty( "Refractory time cannot be negative." );
  }
  if ( tau_synE <= 0 || tau_synI <= 0 )
  {
    throw BadProperty( "All time constants must be strictly positive." );
  }
  if ( g_Kv1 < 0 || g_Kv3 < 0 || g_Na < 0 || g_L < 0 )
  {
    throw BadProperty( "All conductances must be non-negative." );
  }
}

void
nest::hh_psc_alpha_gap::State_::get( DictionaryDatum& d ) const
{
  def< double >( d, names::V_m, y_[ V_M ] );
  def< double >( d, names::Act_m, y_[ HH_M ] );
  def< double >( d, names::Act_h, y_[ HH_H ] );
  def< double >( d, names::Inact_n, y_[ HH_N ] );
  def< double >( d, names::Inact_p, y_[ HH_P ] );
}

void
nest::hh_psc_alpha_gap::State_::set( const DictionaryDatum& d )
{
  updateValue< double >( d, names::V_m, y_[ V_M ] );
  updateValue< double >( d, names::Act_m, y_[ HH_M ] );
  updateValue< double >( d, names::Act_h, y_[ HH_H ] );
  updateValue< double >( d, names::Inact_n, y_[ HH_N ] );
  updateValue< double >( d, names::Inact_p, y_[ HH_P ] );
  if ( y_[ HH_M ] < 0 || y_[ HH_H ] < 0 || y_[ HH_N ] < 0 || y_[ HH_P ] < 0 )
  {
    throw BadProperty( "All (in)activation variables must be non-negative." );
  }
}

nest::hh_psc_alpha_gap::Buffers_::Buffers_( hh_psc_alpha_gap& n )
  : logger_( n )
  , s_( 0 )
  , c_( 0 )
  , e_( 0 )
{
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

nest::hh_psc_alpha_gap::Buffers_::Buffers_( const Buffers_&,
  hh_psc_alpha_gap& n )
  : logger_( n )
  , s_( 0 )
  , c_( 0 )
  , e_( 0 )
{
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */

nest::hh_psc_alpha_gap::hh_psc_alpha_gap()
  : Archiving_Node()
  , P_()
  , S_( P_ )
  , B_( *this )
{
  recordablesMap_.create();
  Node::set_node_uses_wfr( kernel().simulation_manager.use_wfr() );
}

nest::hh_psc_alpha_gap::hh_psc_alpha_gap( const hh_psc_alpha_gap& n )
  : Archiving_Node( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
  Node::set_node_uses_wfr( kernel().simulation_manager.use_wfr() );
}

nest::hh_psc_alpha_gap::~hh_psc_alpha_gap()
{
  // GSL structs may not have been allocated, so we need to protect destruction
  if ( B_.s_ )
  {
    gsl_odeiv_step_free( B_.s_ );
  }
  if ( B_.c_ )
  {
    gsl_odeiv_control_free( B_.c_ );
  }
  if ( B_.e_ )
  {
    gsl_odeiv_evolve_free( B_.e_ );
  }
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
nest::hh_psc_alpha_gap::init_state_( const Node& proto )
{
  const hh_psc_alpha_gap& pr = downcast< hh_psc_alpha_gap >( proto );
  S_ = pr.S_;
}

void
nest::hh_psc_alpha_gap::init_buffers_()
{
  B_.spike_exc_.clear(); // includes resize
  B_.spike_inh_.clear(); // includes resize
  B_.currents_.clear();  // includes resize

  // allocate strucure for gap events here
  // function is called from Scheduler::prepare_nodes() before the
  // first call to update
  // so we already know which interpolation scheme to use according
  // to the properties of this neurons
  // determine size of structure depending on interpolation scheme
  // and unsigned int Scheduler::min_delay() (number of simulation time steps
  // per min_delay step)

  // resize interpolation_coefficients depending on interpolation order
  const size_t buffer_size = kernel().connection_manager.get_min_delay()
    * ( kernel().simulation_manager.get_wfr_interpolation_order() + 1 );

  B_.interpolation_coefficients.resize( buffer_size, 0.0 );

  B_.last_y_values.resize( kernel().connection_manager.get_min_delay(), 0.0 );

  B_.sumj_g_ij_ = 0.0;

  Archiving_Node::clear_history();

  B_.logger_.reset();

  B_.step_ = Time::get_resolution().get_ms();
  B_.IntegrationStep_ = B_.step_;

  if ( B_.s_ == 0 )
  {
    B_.s_ =
      gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, State_::STATE_VEC_SIZE );
  }
  else
  {
    gsl_odeiv_step_reset( B_.s_ );
  }

  if ( B_.c_ == 0 )
  {
    B_.c_ = gsl_odeiv_control_y_new( 1e-6, 0.0 );
  }
  else
  {
    gsl_odeiv_control_init( B_.c_, 1e-6, 0.0, 1.0, 0.0 );
  }

  if ( B_.e_ == 0 )
  {
    B_.e_ = gsl_odeiv_evolve_alloc( State_::STATE_VEC_SIZE );
  }
  else
  {
    gsl_odeiv_evolve_reset( B_.e_ );
  }

  B_.sys_.function = hh_psc_alpha_gap_dynamics;
  B_.sys_.jacobian = NULL;
  B_.sys_.dimension = State_::STATE_VEC_SIZE;
  B_.sys_.params = reinterpret_cast< void* >( this );

  B_.I_stim_ = 0.0;
}

void
nest::hh_psc_alpha_gap::calibrate()
{
  // ensures initialization in case mm connected after Simulate
  B_.logger_.init();

  V_.PSCurrInit_E_ = 1.0 * numerics::e / P_.tau_synE;
  V_.PSCurrInit_I_ = 1.0 * numerics::e / P_.tau_synI;
  V_.RefractoryCounts_ = Time( Time::ms( P_.t_ref_ ) ).get_steps();
  // since t_ref_ >= 0, this can only fail in error
  assert( V_.RefractoryCounts_ >= 0 );
}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 * ---------------------------------------------------------------- */



// #define CHECK_2D_DATA_DBL(dst, src, buf, dim)				\
//   download(&update_context, src, (void *)buf, dim*num_nodes*sizeof(double)); \
//   for (nodeIt = nodes.begin(), i = 0; nodeIt != nodes.end(); nodeIt++, i++ ) \
//     {									\
//       nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;	\
//       for (int j = 0; j < dim; j++)					\
// 	{								\
// 	  double tmp = fabs(node->dst[j] - buf[j*nodes.size() + i])/node->dst[j]; \
// 	  if (tmp > 1e-3)						\
// 	    {								\
// 	      printf("\ncheck 2d data error: %d %d %0.5f %0.5f %0.5f\n", j, i, tmp, node->dst[j], buf[j*nodes.size() + i]); \
// 	      return 1;							\
// 	    }								\
// 	}								\
//     }

// #define CHECK_2D_DATA_DBL(dst, buf)					\
//   for (int j = 0; j < dimension; j++)					\
//     {									\
//       printf("\n%d %d %0.5f %0.5f\n", j, i, node->dst[j], buf[j*hlf_num_nodes + i]); \
//       if (fabs(buf[j*hlf_num_nodes + i]) > 1e-4)			\
// 	{								\
// 	  double tmp = fabs(node->dst[j] - buf[j*hlf_num_nodes + i])/node->dst[j]; \
// 	  if (tmp > 1e-3)						\
// 	    {								\
// 	      printf("\ncheck 2d data error: %d %d %0.5f %0.5f %0.5f\n", j, i, tmp, node->dst[j], buf[j*hlf_num_nodes + i]); \
// 	      return 1;							\
// 	    }								\
// 	}								\
//     }

// #define CHECK_2D_DATA_LONG(dst, src, buf, dim) \
//   download(&update_context, src, (void *)buf, dim*num_nodes*sizeof(long)); \
//   for (nodeIt = nodes.begin(), i = 0; nodeIt != nodes.end(); nodeIt++, i++ ) \
//     {									\
//       nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;	\
//       for (int j = 0; j < dim; j++)					\
// 	{								\
// 	  long tmp = abs(node->dst[j] - buf[j*nodes.size() + i]);	\
// 	  if (tmp != 0)							\
// 	    {								\
// 	      printf("\ncheck 2d data error: %d %d %lld\n", j, i, tmp);	\
// 	      return 1;							\
// 	    }								\
// 	}								\
//     }

// //#define CHECK_2D_DATA(dst, src, buf, len, (long long int)) CHECK_2D_DATA(dst, src, buf, len, long)

// #define CHECK_1D_DATA_DBL(dst, src, buf)				\
//   download(&update_context, src, (void *)buf, num_nodes*sizeof(double)); \
//   for (nodeIt = nodes.begin(), i = 0; nodeIt != nodes.end(); nodeIt++, i++ ) \
//     {									\
//       nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;	\
//       {									\
// 	double tmp = fabs(node->dst - buf[i])/node->dst;		\
// 	if (tmp > 1e-3)							\
// 	  {								\
// 	    printf("\ncheck 1d data error: %d %0.5f\n", i, tmp);	\
// 	    return 1;							\
// 	  }								\
//       }									\
//     }

// #define CHECK_1D_DATA_LONG(dst, src, buf)				\
//   download(&update_context, src, (void *)buf, num_nodes*sizeof(long));	\
//   for (nodeIt = nodes.begin(), i = 0; nodeIt != nodes.end(); nodeIt++, i++ ) \
//     {									\
//       nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;	\
//       {									\
// 	long tmp = abs(node->dst - buf[i]);				\
// 	if (tmp != 0)							\
// 	{								\
// 	  printf("\ncheck 1d data error: %d %ld\n", i, tmp);		\
// 	  return 1;							\
// 	}								\
//       }									\
//     }

// int
// nest::hh_psc_alpha_gap::check_data(std::vector< Node* > &nodes, int dimension, int queue_index)
// // {
// //   return 0;
// // }
// {
//   int num_nodes = nodes.size();

//   int hlf_num_nodes = num_nodes/2;
//   int len1 = dimension*hlf_num_nodes*queue_index;
//   //int len2 = hlf_num_nodes*queue_index;

//   // const size_t buffer_size = kernel().connection_manager.get_min_delay()
//   //   * ( kernel().simulation_manager.get_wfr_interpolation_order() + 1 );

//   // size_t buf_len = dimension > buffer_size ? dimension : buffer_size;
//   // double *dbl_buffer = (double *)malloc(buf_len*num_nodes*sizeof(double));
//   // long int *long_buffer = (long int *)malloc(buf_len*num_nodes*sizeof(long int));

//   int start_nodes = hlf_num_nodes * queue_index;
//   int end_nodes = std::min(start_nodes + hlf_num_nodes, num_nodes);

//   int nodeid = 0;
//   std::vector<Node *>::iterator nodeIt = nodes.begin();
//   for (; nodeIt != nodes.end() && nodeid < start_nodes; nodeIt++, nodeid++)
//     {}

//   //printf("\nnum_nodes: %d dimension: %d buffer_size %d\n", num_nodes, dimension, buffer_size);

//   printf("\nS_.y_\n");
//   START_2D_DOWNLOAD(S_y_, (h_S_y_ + len1), dimension, double);
//   clFinish(update_context.command_queue[queue_index]);

//   for (int i = 0; nodeIt != nodes.end() && nodeid < end_nodes; nodeIt++, nodeid++, i++)
//     {
//       nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;
//       CHECK_2D_DATA_DBL(S_.y_, (h_S_y_ + len1));
//     }

//   //CHECK_2D_DATA_DBL(S_.y_, S_y_, dbl_buffer, dimension);
//   // printf("\nB_.e_->y0\n");CHECK_2D_DATA_DBL(B_.e_->y0, e_y0, dbl_buffer, dimension);
//   // printf("\nB_.e_->yerr\n");CHECK_2D_DATA_DBL(B_.e_->yerr, e_yerr, dbl_buffer, dimension);
//   // printf("\nB_.e_->dydt_in\n");CHECK_2D_DATA_DBL(B_.e_->dydt_in, e_dydt_in, dbl_buffer, dimension);
//   // printf("\nB_.e_->dydt_out\n");CHECK_2D_DATA_DBL(B_.e_->dydt_out, e_dydt_out, dbl_buffer, dimension);
//   // printf("\nB_.e_->last_step\n");CHECK_1D_DATA_DBL(B_.e_->last_step, e_last_step, dbl_buffer);
//   // printf("\nB_.e_->count\n");CHECK_1D_DATA_LONG(B_.e_->count, e_count, long_buffer); //unsigned long int);
//   // printf("\nB_.e_->failed_steps\n");CHECK_1D_DATA_LONG(B_.e_->failed_steps, e_failed_steps, long_buffer); //unsigned long int);
//   // printf("\nB_.c_->eps_abs\n");CHECK_1D_DATA_DBL(B_.c_->eps_abs, con_state_eps_abs, dbl_buffer);
//   // printf("\nB_.c_->eps_rel\n");CHECK_1D_DATA_DBL(B_.c_->eps_rel, con_state_eps_rel, dbl_buffer);
//   // printf("\nB_.c_->a_y\n");CHECK_1D_DATA_DBL(B_.c_->a_y, con_state_a_y, dbl_buffer);
//   // printf("\nB_.c_->a_dydt\n");CHECK_1D_DATA_DBL(B_.c_->a_dydt, con_state_a_dydt, dbl_buffer);
//   // printf("\nB_.s_->k1\n");CHECK_2D_DATA_DBL(B_.s_->k1, rk_state_k1, dbl_buffer, dimension);
//   // printf("\nB_.s_->k2\n");CHECK_2D_DATA_DBL(B_.s_->k2, rk_state_k2, dbl_buffer, dimension);
//   // printf("\nB_.s_->k3\n");CHECK_2D_DATA_DBL(B_.s_->k3, rk_state_k3, dbl_buffer, dimension);
//   // printf("\nB_.s_->k4\n");CHECK_2D_DATA_DBL(B_.s_->k4, rk_state_k4, dbl_buffer, dimension);
//   // printf("\nB_.s_->k5\n");CHECK_2D_DATA_DBL(B_.s_->k5, rk_state_k5, dbl_buffer, dimension);
//   // printf("\nB_.s_->k6\n");CHECK_2D_DATA_DBL(B_.s_->k6, rk_state_k6, dbl_buffer, dimension);
//   // printf("\nB_.s_->y0\n");CHECK_2D_DATA_DBL(B_.s_->y0, rk_state_y0, dbl_buffer, dimension);
//   // printf("\nB_.s_->ytmp\n");CHECK_2D_DATA_DBL(B_.s_->ytmp, rk_state_ytmp, dbl_buffer, dimension);
//   // printf("\nP_.g_Na\n");CHECK_1D_DATA_DBL(P_.g_Na, P_g_Na, dbl_buffer);
//   // printf("\nP_.g_Kv1\n");CHECK_1D_DATA_DBL(P_.g_Kv1, P_g_Kv1, dbl_buffer);
//   // printf("\nP_.g_Kv3\n");CHECK_1D_DATA_DBL(P_.g_Kv3, P_g_Kv3, dbl_buffer);
//   // printf("\nP_.g_L\n");CHECK_1D_DATA_DBL(P_.g_L, P_g_L, dbl_buffer);
//   // printf("\nP_.C_m\n");CHECK_1D_DATA_DBL(P_.C_m, P_C_m, dbl_buffer);
//   // printf("\nP_.E_Na\n");CHECK_1D_DATA_DBL(P_.E_Na, P_E_Na, dbl_buffer);
//   // printf("\nP_.E_K\n");CHECK_1D_DATA_DBL(P_.E_K, P_E_K, dbl_buffer);
//   // printf("\nP_.E_L\n");CHECK_1D_DATA_DBL(P_.E_L, P_E_L, dbl_buffer);
//   // printf("\nP_.tau_synE\n");CHECK_1D_DATA_DBL(P_.tau_synE, P_tau_synE, dbl_buffer);
//   // printf("\nP_.tau_synI\n");CHECK_1D_DATA_DBL(P_.tau_synI, P_tau_synI, dbl_buffer);
//   // printf("\nP_.I_e\n");CHECK_1D_DATA_DBL(P_.I_e, P_I_e, dbl_buffer);
//   // printf("\nB_.step_\n");CHECK_1D_DATA_DBL(B_.step_, B_step_, dbl_buffer);
//   // printf("\nB_.lag_\n");CHECK_1D_DATA_LONG(B_.lag_, B_lag_, long_buffer);
//   // printf("\nB_.sumj_g_ij_\n");CHECK_1D_DATA_DBL(B_.sumj_g_ij_, B_sumj_g_ij_, dbl_buffer);
//   // //

//   // //COPY_2D_DATA(B_interpolation_coefficients, B_.interpolation_coefficients, dbl_buffer, buffer_size, double);
  
//   // download(&update_context, B_interpolation_coefficients, (void *)dbl_buffer, buffer_size*num_nodes*sizeof(double));
//   // for (nodeIt = nodes.begin(), i = 0; nodeIt != nodes.end(); nodeIt++, i++ )
//   //   {
//   //     nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;
//   //     std::vector<double>::iterator it = node->B_.interpolation_coefficients.begin();
//   //     int c = 0;
//   //     for (; it != node->B_.interpolation_coefficients.end(); it++, c++)
//   // 	{
//   // 	  double tmp = fabs(*it - dbl_buffer[num_nodes*c + i])/(*it);
//   // 	  if (tmp > 1e-3)
//   // 	    {
//   // 	      printf("\ncheck error B_interpolation_coefficients %d %0.2f\n", num_nodes*c + i, tmp);
//   // 	      return 1;
//   // 	    }
//   // 	}
//   //   }
  
//   // //
//   // printf("\nB_.I_stim_\n");CHECK_1D_DATA_DBL(B_.I_stim_, B_I_stim_, dbl_buffer);
//   // printf("\nB_.IntegrationStep_\n");CHECK_1D_DATA_DBL(B_.IntegrationStep_, B_IntegrationStep_, dbl_buffer);

//   //printf("\nU_old\n");CHECK_1D_DATA_DBL(U_old, d_U_old, dbl_buffer);
 
//   // free(dbl_buffer);
//   // free(long_buffer);

//   return 0;
// }

bool
nest::hh_psc_alpha_gap::update_( Time const& origin,
  const long from,
  const long to,
  const bool called_from_wfr_update )
{
  // printf("hh_psc_alpha_gap::update_(): %d\n", called_from_wfr_update);
  // note: the pattern is "a bunch of calls from wfr_update, then a bunch of calls from update".
  //       and the numbers vary over the course of a run, dunno why/how yet.

  //printf("update node %d\n", this->get_gid());
  assert(
    to >= 0 && ( delay ) from < kernel().connection_manager.get_min_delay() );
  assert( from < to );

  const size_t interpolation_order =
    kernel().simulation_manager.get_wfr_interpolation_order();
  const double wfr_tol = kernel().simulation_manager.get_wfr_tol();
  bool wfr_tol_exceeded = false;

  // allocate memory to store the new interpolation coefficients
  // to be sent by gap event
  const size_t buffer_size =
    kernel().connection_manager.get_min_delay() * ( interpolation_order + 1 );
  std::vector< double > new_coefficients( buffer_size, 0.0 );

  // parameters needed for piecewise interpolation
  double y_i = 0.0, y_ip1 = 0.0, hf_i = 0.0, hf_ip1 = 0.0;
  // double f_temp[ State_::STATE_VEC_SIZE ];
  
  for ( long lag = from; lag < to; ++lag )
  {

    // B_.lag is needed by hh_psc_alpha_gap_dynamics to
    // determine the current section
    B_.lag_ = lag;

    if ( called_from_wfr_update )
    {
      y_i = S_.y_[ State_::V_M ];
      if ( interpolation_order == 3 )
      {
        // printf("before hh_psc_alpha_gap_dynamics, S_.y_[V_M] is %.2f\n", S_.y_[State_::V_M]);
        hh_psc_alpha_gap_dynamics(
          0, S_.y_, f_temp, reinterpret_cast< void* >( this ) );
        // printf("after hh_psc_alpha_gap_dynamics, S_.y_[V_M] is %.2f\n", S_.y_[State_::V_M]);
        hf_i = B_.step_ * f_temp[ State_::V_M ];
      }
    }

    double t = 0.0;
    const double U_old = S_.y_[ State_::V_M ];

    // numerical integration with adaptive step size control:
    // ------------------------------------------------------
    // gsl_odeiv_evolve_apply performs only a single numerical
    // integration step, starting from t and bounded by step;
    // the while-loop ensures integration over the whole simulation
    // step (0, step] if more than one integration step is needed due
    // to a small integration step size;
    // note that (t+IntegrationStep > step) leads to integration over
    // (t, step] and afterwards setting t to step, but it does not
    // enforce setting IntegrationStep to step-t; this is of advantage
    // for a consistent and efficient integration across subsequent
    // simulation intervals


    struct timeval t_slice_begin_, t_slice_end_;
    gettimeofday( &t_slice_begin_, NULL );
    while ( t < B_.step_ )
    {
      // printf("\ncalling gsl_odeiv_evolve_apply\n");
      // printf("before gsl_odeiv_evolve_apply, t is %.5f, value is %.2f\n", t, S_.y_[State_::V_M]);
      const int status = gsl_odeiv_evolve_apply( B_.e_,
        B_.c_,
        B_.s_,
        &B_.sys_,             // system of ODE
        &t,                   // from t
        B_.step_,             // to t <= step
        &B_.IntegrationStep_, // integration step size
        S_.y_ );              // neuronal state

      // printf("after gsl_odeiv_evolve_apply, t is %.5f, value is %.2f\n", t, S_.y_[State_::V_M]);

      static int call_count = 0;
      call_count++;
      // if(call_count % 1000000 == 0) {
      //   printf("\n%d calls to gsl_odeiv_evolve_apply\n", call_count);
      // }

      if ( status != GSL_SUCCESS )
      {
        throw GSLSolverFailure( get_name(), status );
      }
    }
    gettimeofday( &t_slice_end_, NULL );
    if ( t_slice_end_.tv_sec != 0 )
    {
      // usec
      long t_real_ = 0;
      long t_real_s = ( t_slice_end_.tv_sec - t_slice_begin_.tv_sec ) * 1e6;
      // usec
      t_real_ += t_real_s + ( t_slice_end_.tv_usec - t_slice_begin_.tv_usec );
      // printf("\nwhile loop took %.2fms\n", t_real_ / 1e3);
    }

    if ( not called_from_wfr_update )
    {
      S_.y_[ State_::DI_EXC ] +=
        B_.spike_exc_.get_value( lag ) * V_.PSCurrInit_E_;
      S_.y_[ State_::DI_INH ] +=
        B_.spike_inh_.get_value( lag ) * V_.PSCurrInit_I_;
      // sending spikes: crossing 0 mV, pseudo-refractoriness and local
      // maximum...
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

        // printf("node %p sending spike with lag %d: ", this, lag);

        kernel().event_delivery_manager.send( *this, se, lag );
      }

      // log state data
      B_.logger_.record_data( origin.get_steps() + lag );

      // set new input current
      B_.I_stim_ = B_.currents_.get_value( lag );
    }
    else // if(called_from_wfr_update)
    {
      S_.y_[ State_::DI_EXC ] +=
        B_.spike_exc_.get_value_wfr_update( lag ) * V_.PSCurrInit_E_;
      S_.y_[ State_::DI_INH ] +=
        B_.spike_inh_.get_value_wfr_update( lag ) * V_.PSCurrInit_I_;
      // check if deviation from last iteration exceeds wfr_tol
      wfr_tol_exceeded = wfr_tol_exceeded
        or fabs( S_.y_[ State_::V_M ] - B_.last_y_values[ lag ] ) > wfr_tol;
      // printf("node %p, checking fabs(%.2f - %.2f) > %.2f, lag is %d, to is %d, wfr_tol_exceeded is %d\n", this, S_.y_[ State_::V_M ], B_.last_y_values[ lag ], wfr_tol, lag, to, wfr_tol_exceeded);
      B_.last_y_values[ lag ] = S_.y_[ State_::V_M ];

      // update different interpolations

      // constant term is the same for each interpolation order
      new_coefficients[ lag * ( interpolation_order + 1 ) + 0 ] = y_i;

      switch ( interpolation_order )
      {
      case 0:
        break;

      case 1:
        y_ip1 = S_.y_[ State_::V_M ];

        new_coefficients[ lag * ( interpolation_order + 1 ) + 1 ] = y_ip1 - y_i;
        break;

      case 3:
        y_ip1 = S_.y_[ State_::V_M ];

        hh_psc_alpha_gap_dynamics(
          B_.step_, S_.y_, f_temp, reinterpret_cast< void* >( this ) );
        hf_ip1 = B_.step_ * f_temp[ State_::V_M ];

        new_coefficients[ lag * ( interpolation_order + 1 ) + 1 ] = hf_i;
        new_coefficients[ lag * ( interpolation_order + 1 ) + 2 ] =
          -3 * y_i + 3 * y_ip1 - 2 * hf_i - hf_ip1;
        new_coefficients[ lag * ( interpolation_order + 1 ) + 3 ] =
          2 * y_i - 2 * y_ip1 + hf_i + hf_ip1;
        break;

      default:
        throw BadProperty( "Interpolation order must be 0, 1, or 3." );
      }
    }


  } // end for-loop

  // if not called_from_wfr_update perform constant extrapolation
  // and reset last_y_values
  if ( not called_from_wfr_update )
  {
    // printf("non-mass update, resetting last y values\n");
    for ( long temp = from; temp < to; ++temp )
    {
      new_coefficients[ temp * ( interpolation_order + 1 ) + 0 ] =
        S_.y_[ State_::V_M ];
    }

    std::vector< double >( kernel().connection_manager.get_min_delay(), 0.0 )
      .swap( B_.last_y_values );
  }

  // Send gap-event
  GapJunctionEvent ge;
  ge.set_coeffarray( new_coefficients );

  // printf("node %p send_secondary with coeffs: ", this);
  // for(int i = 0; i < new_coefficients.size(); i++)
  // {
  //   printf("%.2f ", new_coefficients[i]);
  // }
  // printf("\n");

  kernel().event_delivery_manager.send_secondary( *this, ge );

  // Reset variables
  B_.sumj_g_ij_ = 0.0;
  std::vector< double >( buffer_size, 0.0 )
    .swap( B_.interpolation_coefficients );

  // printf("S_.y_[ State_::V_M ]");

  return wfr_tol_exceeded;
}

void
nest::hh_psc_alpha_gap::handle( SpikeEvent& e )
{
  assert( e.get_delay() > 0 );

  if ( e.get_weight() > 0.0 )
  {
    B_.spike_exc_.add_value( e.get_rel_delivery_steps(
                               kernel().simulation_manager.get_slice_origin() ),
      e.get_weight() * e.get_multiplicity() );
  }
  else
  {
    B_.spike_inh_.add_value( e.get_rel_delivery_steps(
                               kernel().simulation_manager.get_slice_origin() ),
      e.get_weight() * e.get_multiplicity() );
  } // current input, keep negative weight
}

void
nest::hh_psc_alpha_gap::handle( CurrentEvent& e )
{
  assert( e.get_delay() > 0 );

  const double c = e.get_current();
  const double w = e.get_weight();

  // add weighted current; HEP 2002-10-04
  B_.currents_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    w * c );
}

void
nest::hh_psc_alpha_gap::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}

void
nest::hh_psc_alpha_gap::handle( GapJunctionEvent& e )
{
  index sgid = e.get_sender_gid() - 1;
  index tgid = this->get_gid() - 1;
  double weight = e.get_weight();
  
  int thrd = kernel().vp_manager.get_thread_id();
  kernel().simulation_manager.gpu_execution[thrd]->handle(sgid, tgid, weight);
  // B_.sumj_g_ij_ += e.get_weight();

  // size_t i = 0;
  // std::vector< unsigned int >::iterator it = e.begin();
  // // The call to get_coeffvalue( it ) in this loop also advances the iterator it
  // while ( it != e.end() )
  // {
  //   B_.interpolation_coefficients[ i ] +=
  //     e.get_weight() * e.get_coeffvalue( it );
  //   ++i;
  // }
}


// void
// nest::hh_psc_alpha_gap::check_event_data(std::vector<Node *> nodes)
// {
//   int nodeid = 0;
//   double *coeff_ptr = h_coeff_buffer;
//   for ( std::vector<Node*>::iterator nodeIt = nodes.begin(); nodeIt != nodes.end(); nodeIt++)
//   {
//     nest::hh_psc_alpha_gap* node = (nest::hh_psc_alpha_gap*)*nodeIt;

//     nodeid = node->get_gid() - 1;
//     coeff_ptr = h_coeff_buffer + event_size * nodeid;
    
//     double tmp1 = node->B_.sumj_g_ij_;
//     double tmp2 = h_B_sumj[nodeid];

//     double fa = fabs(tmp1 - tmp2);
//     double ch = fa/tmp1;

//     //printf("\nB_.sumj_g_ij_ %0.5f %0.5f %d %0.5f\n", tmp1, tmp2, nodeid, ch);
//     if (fa > 1e-3 || ch > 1e-3)
//       {
// 	printf("\ncheck B_.sumj_g_ij_ error: %0.5f %0.5f %d %0.5f\n", tmp1, tmp2, nodeid, ch);
// 	return;
//       }
    
//     node->B_.sumj_g_ij_ = h_B_sumj[nodeid];

//     //printf("\nB_.interpolation_coefficients");
//     for (size_t i = 0; i < event_size; i++)
//       {
//     	tmp1 = node->B_.interpolation_coefficients[i];
//     	tmp2 = coeff_ptr[i];
//     	fa = fabs(tmp1 - tmp2);
//     	ch = fa/tmp1;
//     	//printf("\n%0.5f %0.5f %d %d", tmp1, tmp2, nodeid, i);
//     	if (fa > 1e-3 || ch > 1e-3)
//     	  {
//     	    printf("\ncheck B_.interpolation_coefficients[i] error: %0.5f %0.5f %d %d %0.5f\n", tmp1, tmp2, nodeid, i, ch);
//     	    return;
//     	  }

// 	//node->B_.interpolation_coefficients[i] = coeff_ptr[i];
//     }
    
//     //coeff_ptr += event_size;
//   }
// }


// #endif // HAVE_GSL
