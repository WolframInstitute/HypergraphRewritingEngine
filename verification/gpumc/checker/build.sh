#!/usr/bin/env bash
# Build gpumc_docker:hg-fixes, the image verification/gpumc/run.sh uses: the CAV 2025 GPUMC artifact
# (gpumc_docker:latest, loaded by `docker load`) with gpumc-fixes.patch applied.
#
# gpumc-fixes.patch carries seven changes. Each has a test in this directory, run by
# `verification/gpumc/run.sh cells cas_width memcpy_promotion race_in_header struct_return
# barrier_divergence float_math loop_jump_threading loop_body_phi`.
#
# Compare-exchange width (Execution.cpp, GenMCDriver.cpp, SVal.hpp). GPUMC 0.9 compared a
# compare-exchange's read value against the expected value sign-extended from the access width,
# while a store keeps its value zero-extended. A 32-bit compare-exchange whose expected value has
# bit 31 set (0xFFFFFFFF, INVALID_ID, every all-ones sentinel the device uses) read the expected
# value, reported failure, and the execution blocked. The interpreter and the two driver sites that
# decide whether a read completes a compare-exchange compare the bottom access-width bits.
# checker/cas_width.cpp fails on gpumc_docker:latest and passes on hg-fixes.
#
# Memory intrinsic promotion (PromoteMemIntrinsicPass.cpp). GPUMC rewrites a constant-length
# memcpy or memset into per-field loads and stores over one object type, and executing one it
# could not rewrite is an error. Its rewrite indexed the source with the destination's field path,
# so a source of another type made an invalid GEP and aborted the checker on gpu/src/persistent.cu;
# it ignored the length, so a copy of a prefix copied the whole object; it found no type for an
# operand that was an argument, a loaded pointer or a malloc'd object; and a memset stored the fill
# byte's value in each integer field (0x01 for 0x01010101) and aborted on a pointer or double
# field. The rewrite runs over the destination's type, or the source's when only the source's size
# is the length, with the other operand cast to it; covers the fields inside the length and is
# refused when a field straddles it; takes an operand's type from the typed pointer under its
# casts; and fills each integer field with the fill byte repeated and any other field with zero.
# Each refusal names its function. checker/memcpy_promotion.cpp checks the cases.
#
# Race recording (GenMCDriver.cpp, GenMCDriver.hpp, main.cpp). GPUMC recorded a data race only
# when the access that detected it was in the input file, and on any other race returned without
# restoring the interpreter state it had saved to describe the race: a race in an included engine
# header was dropped, and the rest of the execution ran on a corrupted state (later assertions
# unreported, accesses to unallocated memory, crashes). Every race is recorded under the file and
# line of both accesses, and the state is restored. checker/race_in_header.cpp checks it.
#
# Stack aggregates (LLVMModule.cpp). The x86-64 ABI returns a struct of 16 bytes or less in
# registers: the callee stores its fields to a stack temporary and loads it as one integer, which
# GPUMC reported as a mixed-size access (a 128-bit load of a 64-bit store) on every map insert
# (gpu/include/hg_gpu/hash_table.hpp, InsertResult). SROA runs before mem2reg, so such temporaries
# become registers. checker/struct_return.cpp checks it.
#
# Barrier divergence (GenMCDriver.cpp, isExecutionBlocked). GPUMC reported a divergence whenever an
# execution ended with a thread waiting at a barrier, including executions it had cut short itself
# (a block-mate blocked by its spin-loop assume). A thread waiting at a barrier is reported when a
# thread of its work group has finished, since the barrier can then never complete.
# checker/barrier_divergence.cpp checks both cases.
#
# Floating point (IntrinsicLoweringPass.cpp, ExternalFunctions.cpp). GPUMC aborted on llvm.fma,
# llvm.fmuladd and llvm.fabs ("Code generator does not support intrinsic function"), and stopped on
# a call to sqrt, floor or any other C math function. fma and fmuladd become a multiply and an
# add; fabs, minnum and maxnum become calls to fabs, fmin and fmax; and the interpreter computes the
# C library's math functions (sqrt, cbrt, floor, ceil, trunc, round, rint, nearbyint, log, log2,
# log10, log1p, exp, exp2, expm1, sin, cos, fabs, pow, copysign, fmin, fmax, atan2, hypot) on
# double and float. An external call's declared non-pointer argument is passed as it is: the
# interpreter took any argument whose bits read as a static address for a pointer, which a
# negative double does (-2.25 is 0xC002...), and stopped at getStaticAddr. checker/float_math.cpp
# checks both.
#
# Loop inversion (LoopJumpThreadingPass.cpp). When the header's condition is true on entry, the
# pass sends the preheader straight into the loop body, and the header then runs only from the
# latch. It checked uses of the header's PHIs inside the loop only. A header value read after the
# loop, as in `bool ok = true; while (ok && full()) ok = grow(); if (ok) ...`
# (gpu/src/persistent.cu held_select_step), had no definition on the path that leaves the body on
# the first iteration, and the checker aborted in transformLLVMModule with "Instruction does not
# dominate all uses!". The pass inverts a loop only when every use of a header value is in the
# header, on an edge out of it, or is a use of the critical PHI in the body.
# checker/loop_jump_threading.cpp checks it. The body block the preheader jumps to had no PHI entry
# for the new edge when it merges two paths ("PHINode should have one entry for each predecessor"):
# each of its PHIs takes, from the preheader, the value it takes from the header, with the critical
# PHI replaced by its initial value, and a loop whose value is defined in the header is not
# inverted. checker/loop_body_phi.cpp checks it. Both defects are also in GenMC v0.20.0.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
docker image inspect gpumc_docker:latest >/dev/null
nice -n 10 docker build -t gpumc_docker:hg-fixes "$HERE"
