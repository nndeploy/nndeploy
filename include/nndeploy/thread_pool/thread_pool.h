
#ifndef _NNDEPLOY_THREAD_POOL_THREAD_POOL_H_
#define _NNDEPLOY_THREAD_POOL_THREAD_POOL_H_

#include <thread>
#include <future>
#include <atomic>

#include "nndeploy/base/status.h"
#include "runnable_task.h"
#include "safe_ws_queue.h"
#include "local_thread.h"

namespace nndeploy {
namespace thread_pool {
class ThreadPool {

public:
    explicit ThreadPool(int size = 4) {
        max_thread_size_ = size;
    }

    Status init() {
        for (int i = 0; i < max_thread_size_; i++) {
            auto ptr = new LocalThread();    // 创建核心线程数
            ptr->setThreadPoolInfo(i, &threads_);
            ptr->init();
            threads_.emplace_back(ptr);
        }

        return Status();
    }

    Status destroy() {
        for (auto &pt : threads_) {
            pt->destroy();
        }

        return Status();
    }

    template<typename FunctionType>
    auto commit(const FunctionType& func)
    -> std::future<decltype(std::declval<FunctionType>()())> {
        using ResultType = decltype(std::declval<FunctionType>()());
        std::packaged_task<ResultType()> task(func);
        std::future<ResultType> result(task.get_future());

        cur_index_++;
        if (cur_index_ >= max_thread_size_ || cur_index_ < 0) {
            cur_index_ = 0;
        }
        threads_[cur_index_]->pushTask(std::move(task));
        return result;
    }

private:
    std::atomic<int> cur_index_ {0};
    int max_thread_size_ = 0;
    std::vector<LocalThread *> threads_;
};

}
}

#endif //_NNDEPLOY_THREAD_POOL_THREAD_POOL_H_
