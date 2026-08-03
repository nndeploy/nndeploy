
#ifndef _NNDEPLOY_DAG_OPTIMIZER_H_
#define _NNDEPLOY_DAG_OPTIMIZER_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"

namespace nndeploy {
namespace dag {

class Graph;

class NNDEPLOY_CC_API DagOptPass {
 public:
  DagOptPass() = default;
  virtual ~DagOptPass() = default;

  virtual std::string getName() const = 0;

  /**
   * @brief Run the pass on the DAG.
   *
   * The pass operates on the graph's node/edge repositories and may
   * alter the DAG structure.  It MUST NOT initialise/deinitialise nodes.
   *
   * @return kStatusCodeOk on success.
   */
  virtual base::Status optimize(Graph* graph) = 0;
};

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

using DagOptPassCreator = std::function<std::shared_ptr<DagOptPass>()>;

NNDEPLOY_CC_API void registerDagOptPass(const std::string& name,
                                         DagOptPassCreator creator);
NNDEPLOY_CC_API std::shared_ptr<DagOptPass> createDagOptPass(
    const std::string& name);
NNDEPLOY_CC_API std::vector<std::string> getAllRegisteredDagOptPasses();

/**
 * @brief Static-registration macro.
 *
 *   REGISTER_DAG_OPT_PASS("rga_fuse_pass", RgaFusePass);
 */
#define REGISTER_DAG_OPT_PASS(name, pass_class)                             \
  namespace {                                                               \
  struct DagOptPassReg_##pass_class {                                       \
    DagOptPassReg_##pass_class() {                                          \
      ::nndeploy::dag::registerDagOptPass(                                  \
          name,                                                             \
          []() -> std::shared_ptr<::nndeploy::dag::DagOptPass> {            \
            return std::make_shared<pass_class>();                          \
          });                                                               \
    }                                                                       \
  };                                                                        \
  static DagOptPassReg_##pass_class g_dag_opt_pass_reg_##pass_class;        \
  }

// ---------------------------------------------------------------------------
// DagOptimizer
// ---------------------------------------------------------------------------

class NNDEPLOY_CC_API DagOptimizer {
 public:
  DagOptimizer() = default;
  virtual ~DagOptimizer() = default;

  base::Status init(const std::vector<std::string>& enabled_passes,
                     const std::vector<std::string>& disabled_passes = {});
  base::Status optimize(Graph* graph);
  base::Status deinit();

  bool hasPasses() const { return !passes_.empty(); }

 private:
  std::vector<std::shared_ptr<DagOptPass>> passes_;
};

}  // namespace dag
}  // namespace nndeploy

#endif  // _NNDEPLOY_DAG_OPTIMIZER_H_
