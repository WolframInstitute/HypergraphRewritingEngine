# Run wolframscript on the CPUs of a hexadecimal affinity mask. The mask is set on this
# PowerShell process, and the kernel and the engine worker it starts inherit it. Used by
# tools/dev/paper_tables.py --wl-affinity, so the Windows Wolfram kernel runs on the cores the
# table names: on a machine with performance and efficiency cores, Windows can otherwise place it
# on an efficiency core (measured on the i9-14900K: MultiwaySystem at depth 5 in 2330 ms on the
# performance cores, 3840 ms on the efficiency cores).
#
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File pinned_wolframscript.ps1 -Mask FFFF -Script <path> [args...]
param([string]$Mask, [string]$Script, [Parameter(ValueFromRemainingArguments = $true)] $Rest)
[System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr][Convert]::ToInt64($Mask, 16)
$ws = (Get-ChildItem 'C:\Program Files\Wolfram Research\Wolfram\*\wolframscript.exe' | Sort-Object FullName | Select-Object -Last 1).FullName
& $ws -file $Script @Rest
exit $LASTEXITCODE
