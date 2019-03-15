#include "model_name.h"
#include "model_gpu.h"
#include "kernel_manager.h"

void nest::model_name::handle(BuildGraphEvent &e)
{
  index sgid = e.get_sender_gid();
  index tgid = this->get_thread_lid();
  double weight = e.get_weight();
  int thrd = kernel().vp_manager.get_thread_id();

  kernel().simulation_manager.gpu_execution[thrd]->handle(sgid, tgid, weight);
}
