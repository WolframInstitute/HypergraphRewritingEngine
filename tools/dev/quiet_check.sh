#!/bin/bash
# Reports whether the machine is quiet enough for wall-clock timing. It reads the Windows CPU
# (which includes WSL and any game), the GPU and the WSL run queue, sampling for about 10 s.
# Exit 0 when every reading is under its limit, 1 otherwise. The readings are printed either way.
# Limits: Windows CPU average 5%, GPU utilisation 5% in every sample, GPU power 100 W, WSL load 1.0.
# Runs under WSL: typeperf.exe and nvidia-smi are read through their WSL paths.
win=$(timeout 40 /mnt/c/Windows/System32/typeperf.exe "\Processor(_Total)\% Processor Time" -sc 10 2>/dev/null \
      | tr -d '\r' | awk -F'"' 'NF>=4 && $4 ~ /^[0-9.]+$/ {s+=$4; n++} END {if (n) printf "%.1f", s/n; else print "NA"}')
gpu=$(for i in 1 2 3 4 5; do /usr/lib/wsl/lib/nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits; done \
      | awk -F', ' '{if ($1>u) u=$1; if ($2>p) p=$2} END {printf "%d %.0f", u, p}')
gpu_util=${gpu% *}; gpu_pow=${gpu#* }
load=$(cut -d' ' -f1 /proc/loadavg)
echo "windows_cpu_avg=${win}% gpu_util_max=${gpu_util}% gpu_power_max=${gpu_pow}W wsl_load1=${load}"
quiet=1
[[ "$win" == "NA" ]] && { echo "not quiet: Windows CPU unreadable"; quiet=0; }
[[ "$win" != "NA" ]] && awk -v w="$win" 'BEGIN{exit !(w>5)}' && { echo "not quiet: Windows CPU above 5%"; quiet=0; }
(( gpu_util > 5 )) && { echo "not quiet: GPU utilisation above 5%"; quiet=0; }
(( gpu_pow > 100 )) && { echo "not quiet: GPU power above 100 W"; quiet=0; }
awk -v l="$load" 'BEGIN{exit !(l>1.0)}' && { echo "not quiet: WSL load above 1.0"; quiet=0; }
(( quiet )) && { echo "quiet"; exit 0; }
exit 1
