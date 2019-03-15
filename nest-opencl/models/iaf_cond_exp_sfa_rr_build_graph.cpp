#include "iaf_cond_exp_sfa_rr.h"
#include "model_gpu.h"
#include "kernel_manager.h"

void nest::iaf_cond_exp_sfa_rr::handle(BuildGraphEvent &e)
{
  index sgid = e.get_sender_gid() - 1;
  index tgid = this->get_gid() - 1;
  double weight = e.get_weight();
  int thrd = kernel().vp_manager.get_thread_id();

  kernel().simulation_manager.gpu_execution[thrd]->handle(sgid, tgid, weight);
}
