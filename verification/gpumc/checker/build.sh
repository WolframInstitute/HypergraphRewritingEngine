#!/usr/bin/env bash
# Build gpumc_docker:hg-fixes, the image verification/gpumc/run.sh uses: the CAV 2025 GPUMC artifact
# (gpumc_docker:latest, loaded by `docker load`) with gpumc-fixes.patch applied.
#
# gpumc-fixes.patch carries three fixes. Each has a test in this directory, run by
# `verification/gpumc/run.sh cells cas_width memcpy_promotion race_in_header`.
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
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
docker image inspect gpumc_docker:latest >/dev/null
nice -n 10 docker build -t gpumc_docker:hg-fixes "$HERE"
