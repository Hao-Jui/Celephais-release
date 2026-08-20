#include "bns_gr_nosym_policy.hpp"
#include "Apps/Workflow/bns_app_workflow.hpp"

int main(int argc, char** argv) {
  return KadathApps::run_bns_app<KadathApps::BnsGrNosymPolicy>(argc, argv);
}
