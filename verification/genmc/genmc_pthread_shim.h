// Declarations GenMC's pthread.h leaves out, force-included into every translation unit the
// checker compiles.
//
// WHY IT IS NEEDED. GenMC ships its own pthread.h so the checker can interpret thread creation
// and joining rather than execute the real ones. That header covers what a harness written
// directly against a single structure uses, and no more: the condition-variable entry points are
// commented out, and so are the once/key families and the affinity calls. libstdc++ does not know
// that -- bits/gthr-default.h builds weak aliases for the whole pthread surface as soon as a
// translation unit reaches <memory>, which anything holding a unique_ptr does, so a TU that never
// touches a condition variable still fails to compile without these.
//
// NOTHING HERE IS CALLED BY THE ENGINE. The declarations exist so the alias machinery resolves;
// the engine's blocking primitive is hgcommon/park.hpp, which under HG_PARK_VERIFICATION is a
// spin the checker can see. If one of these were ever reached, GenMC would report it as an
// unknown external function rather than execute something unmodelled.
#pragma once

#include <pthread.h>
#include <sched.h>   // cpu_set_t, for the affinity declarations below; glibc's, not shimmed

extern "C" {

// The condition-variable TYPES. GenMC comments these out along with the functions, and run.sh
// empties glibc's bits/pthreadtypes.h so its definitions cannot stand in either -- under the
// checker nothing declares them at all.
//
// Guarded on _BITS_PTHREADTYPES_COMMON_H, glibc's own guard for the header that defines them:
// undefined under the checker because run.sh empties that header, defined in any ordinary build,
// so this file is inert outside the checker. NOT on PTHREAD_COND_INITIALIZER -- GenMC's pthread.h
// defines that macro while leaving the typedefs commented out, so keying on it skips exactly the
// case this exists for.
#ifndef _BITS_PTHREADTYPES_COMMON_H
typedef struct { long __genmc_cond[6]; } pthread_cond_t;
typedef struct { long __genmc_condattr; } pthread_condattr_t;
// The once and key types are commented out in the same way, and gthr-default.h typedefs
// __gthread_once_t and __gthread_key_t from them before it aliases the functions.
typedef int pthread_once_t;
typedef unsigned int pthread_key_t;
#endif

// The condition-variable family. GenMC comments all of these out.
int pthread_cond_init(pthread_cond_t *, const pthread_condattr_t *);
int pthread_cond_destroy(pthread_cond_t *);
int pthread_cond_signal(pthread_cond_t *);
int pthread_cond_broadcast(pthread_cond_t *);
int pthread_cond_wait(pthread_cond_t *, pthread_mutex_t *);
int pthread_cond_timedwait(pthread_cond_t *, pthread_mutex_t *, const struct timespec *);

// Thread identity and scheduling, named by gthr-default.h.
int pthread_detach(pthread_t);
int pthread_equal(pthread_t, pthread_t);
int sched_yield(void);

// One-time init and thread-local storage, named by gthr-default.h.
int pthread_once(pthread_once_t *, void (*)(void));
int pthread_key_create(pthread_key_t *, void (*)(void *));
int pthread_key_delete(pthread_key_t);
void *pthread_getspecific(pthread_key_t);
int pthread_setspecific(pthread_key_t, const void *);

// Mutex entry points gthr-default.h aliases. The engine takes no lock; these are here for the
// same reason as the rest.
int pthread_mutex_init(pthread_mutex_t *, const pthread_mutexattr_t *);
int pthread_mutex_destroy(pthread_mutex_t *);
int pthread_mutex_trylock(pthread_mutex_t *);
int pthread_mutex_timedlock(pthread_mutex_t *, const struct timespec *);
int pthread_mutexattr_init(pthread_mutexattr_t *);
int pthread_mutexattr_destroy(pthread_mutexattr_t *);
int pthread_mutexattr_settype(pthread_mutexattr_t *, int);

// Affinity, used by job_system/src/affinity.cpp to pin workers. Under the checker there is one
// scheduler and pinning decides nothing, but the call still has to resolve.
// cpu_set_t comes from glibc's <sched.h>, included above: GenMC shims pthread.h but not sched.h,
// so the real definition is available and inventing one here collides with it the moment a
// translation unit includes sched.h for itself, which job_system/src/affinity.cpp does.
int pthread_setaffinity_np(pthread_t, size_t, const cpu_set_t *);
int pthread_getaffinity_np(pthread_t, size_t, cpu_set_t *);

}  // extern "C"
