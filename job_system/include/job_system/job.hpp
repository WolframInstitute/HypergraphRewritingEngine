#ifndef JOB_SYSTEM_JOB_HPP
#define JOB_SYSTEM_JOB_HPP

#include <functional>
#include <memory>
#include <type_traits>

namespace job_system {

template<typename JobType>
class Job {
public:
    virtual ~Job() = default;
    
    virtual void execute() = 0;
    virtual JobType get_type() const = 0;
    virtual int get_priority() const { return 0; }
};

template<typename JobType, typename Func>
class FunctionJob : public Job<JobType> {
private:
    Func function_;
    JobType type_;
    int priority_;

public:
    template<typename F>
    FunctionJob(F&& func, JobType type, int priority = 0)
        : function_(std::forward<F>(func)), type_(type), priority_(priority) {}
    
    void execute() override {
        if constexpr (std::is_invocable_v<Func>) {
            function_();
        } else {
            static_assert(std::is_invocable_v<Func>, "Function must be callable");
        }
    }
    
    JobType get_type() const override {
        return type_;
    }
    
    int get_priority() const override {
        return priority_;
    }
};

template<typename JobType, typename Func>
auto make_job(Func&& func, JobType type, int priority = 0) {
    return std::make_unique<FunctionJob<JobType, std::decay_t<Func>>>(
        std::forward<Func>(func), type, priority);
}

template<typename JobType>
using JobPtr = std::unique_ptr<Job<JobType>>;

enum class ScheduleMode {
    LIFO,  // Last In, First Out (cache-friendly for related tasks)
    FIFO   // First In, First Out (fair scheduling)
};

template<typename JobType>
class CompatibilityAwareJob : public Job<JobType> {
private:
    JobPtr<JobType> wrapped_job_;
    std::function<bool(JobType, JobType)> compatibility_check_;

public:
    template<typename F>
    CompatibilityAwareJob(JobPtr<JobType> job, F&& compatibility_func)
        : wrapped_job_(std::move(job))
        , compatibility_check_(std::forward<F>(compatibility_func)) {}
    
    void execute() override {
        wrapped_job_->execute();
    }
    
    JobType get_type() const override {
        return wrapped_job_->get_type();
    }
    
    bool is_compatible_with(JobType other_type) const override {
        return compatibility_check_(get_type(), other_type);
    }
    
    int get_priority() const override {
        return wrapped_job_->get_priority();
    }
};

template<typename JobType, typename F>
auto make_compatible_job(JobPtr<JobType> job, F&& compatibility_func) {
    return std::make_unique<CompatibilityAwareJob<JobType>>(
        std::move(job), std::forward<F>(compatibility_func));
}

} // namespace job_system

#endif // JOB_SYSTEM_JOB_HPP