#include "iaf_psc_alpha.h"
#include "model_gpu.h"
#include "kernel_manager.h"

void nest::iaf_psc_alpha::handle(BuildGraphEvent &e)
{
  index sgid = e.get_sender_gid();
  index tgid = this->get_thread_lid();
  double weight = e.get_weight();
  int thrd = kernel().vp_manager.get_thread_id();

    // if (sgid == 68 && tgid < 100)
    // {
    //   std::cout << sgid << " " << tgid << " " << weight << std::endl;
    //   getchar();
    // }
  kernel().simulation_manager.gpu_execution[thrd]->handle(sgid, tgid, weight);
}
