#include "ns_gr_policy.hpp"
#include "Apps/Workflow/ns_app_workflow.hpp"

int main(int argc, char** argv) {
    return KadathApps::run_ns_app<KadathApps::NsGrPolicy>(argc, argv);
}
