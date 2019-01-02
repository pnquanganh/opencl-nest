#ifndef MODEL_GPU_H
#define MODEL_GPU_H

#include "../nestkernel/archiving_node.h"
#include "../nestkernel/event.h"
#include "../nestkernel/common_synapse_properties.h"
//#include "stdp_pl_connection_hom.h"

namespace nest
{
  class model_gpu
  {
  public:
    
  model_gpu(): init_device(false)
      {};
    virtual ~model_gpu() {};

    int total_num_nodes;
    int num_local_nodes;
    
    bool init_device;
    int update_type;
    virtual void initialize_gpu() = 0;
    virtual bool mass_wfr_update(const std::vector< Node* > &nodes, Time const& origin, const long from, const long to ) = 0;
    virtual void mass_update(const std::vector< Node* > &nodes, Time const& origin, const long from, const long to ) = 0;

    virtual void initialize() = 0;
    virtual void fill_event_buffer(SecondaryEvent& e) = 0;
    virtual void fill_spike_event_buffer(Event& e) = 0;
    virtual void clear_buffer() = 0;
    virtual void deliver_events() = 0;
    virtual void deliver_static_events() = 0;
    virtual void copy_event_data(std::vector< Node* > nodes) = 0;
    //void check_event_data(std::vector< Node* > nodes);

    virtual void handle(Event& e, double last_t_spike, const CommonSynapseProperties *cp, void *conn, int type) = 0;
    virtual void handle(CurrentEvent& e) = 0;

    std::vector< Node *> updated_nodes;

    virtual void handle(index sgid, index tgid, double weight) = 0;
    virtual void pre_deliver_event(const std::vector< Node* > &nodes) = 0;
    virtual void post_deliver_event(const std::vector< Node* > &nodes) = 0;
    virtual void insert_event(SpikeEvent &e) = 0;
    virtual void insert_static_event(SpikeEvent &e) = 0;
    
    virtual void advance_time() = 0;
  };
}

#endif // MODEL_GPU_H
