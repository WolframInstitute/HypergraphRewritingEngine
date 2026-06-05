#!/usr/bin/env bash
# Build the Hypergraph Rewriting paclet libraries for every platform this host
# can target, then assemble the multi-platform paclet archive.
#
# Design goals:
#   * Declarative target SCHEMA - every platform lists the toolchain it needs,
#     how to auto-install that toolchain, and how to configure its build.
#   * Fully automatic - missing-but-installable toolchains are installed for you
#     (disable with --no-install).
#   * No "crap-shoot" failures - a platform the current host architecture simply
#     cannot target (e.g. Windows-ARM64 on macOS, which needs the Windows SDK) is
#     SKIPPED with a clear reason. The run only FAILS when a build we actually
#     attempted errors out.
#
# Run `./build_all_platforms.sh --help` for options.

set -uo pipefail   # deliberately NOT -e: per-target errors are handled inline

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

HOST_OS="$(uname -s)"      # Darwin | Linux
HOST_ARCH="$(uname -m)"    # arm64 | x86_64 | aarch64
BUILD_JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
TOOLCHAINS="$PROJECT_ROOT/cmake/toolchains"

AUTO_INSTALL=true
WANT_LINUX=true; WANT_WINDOWS=true; WANT_MACOS=true
WANT_ARCHIVE=true

for arg in "$@"; do
    case "$arg" in
        --no-install)   AUTO_INSTALL=false ;;
        --no-archive)   WANT_ARCHIVE=false ;;
        --linux-only)   WANT_LINUX=true;  WANT_WINDOWS=false; WANT_MACOS=false ;;
        --windows-only) WANT_LINUX=false; WANT_WINDOWS=true;  WANT_MACOS=false ;;
        --macos-only)   WANT_LINUX=false; WANT_WINDOWS=false; WANT_MACOS=true ;;
        --no-linux)     WANT_LINUX=false ;;
        --no-windows)   WANT_WINDOWS=false ;;
        --no-macos)     WANT_MACOS=false ;;
        --help)
            cat <<'EOF'
Usage: ./build_all_platforms.sh [OPTIONS]

Builds every platform library this host can target, auto-installing the
required cross-toolchains, then assembles the .paclet archive.

Platform selection:
  --linux-only / --windows-only / --macos-only   Build only that OS family
  --no-linux / --no-windows / --no-macos          Skip that OS family

Behaviour:
  --no-install   Do not auto-install missing toolchains (skip instead)
  --no-archive   Build libraries only; do not assemble the .paclet archive

Environment:
  BUILD_JOBS     Parallel build jobs (default: auto-detect)

Toolchains are described by a schema inside this script; platforms whose
toolchain cannot exist on this host architecture are reported as SKIPPED,
not as failures.
EOF
            exit 0 ;;
        *) echo -e "${YELLOW}Unknown option: $arg (try --help)${NC}" ;;
    esac
done

# ---------------------------------------------------------------------------
# Toolchain installer helpers (macOS / Homebrew)
# ---------------------------------------------------------------------------
brew_install() {
    command -v brew >/dev/null 2>&1 || { echo "Homebrew not available"; return 1; }
    brew install "$@"
}

# Homebrew's messense cross-toolchains expose binaries as <arch>-unknown-linux-gnu-*,
# but the CMake toolchain (and convention) expect the <arch>-linux-gnu-* prefix.
# Symlink the short names into the Homebrew bin so detection just works.
link_linux_cross() {
    local arch="$1" formula="$2" keg dest t
    keg="$(brew --prefix "$formula" 2>/dev/null)" || return 1
    dest="$(brew --prefix)/bin"
    for t in gcc g++ ar ranlib; do
        [ -e "$keg/bin/${arch}-linux-gnu-$t" ] && ln -sf "$keg/bin/${arch}-linux-gnu-$t" "$dest/${arch}-linux-gnu-$t"
    done
}

# ---------------------------------------------------------------------------
# Target SCHEMA
# Parallel arrays, one entry per platform. (Bash 3.2 compatible - no assoc arrays.)
#   NAME    platform directory under paclet/LibraryResources
#   DIR     build directory
#   OUT     artifact (relative to paclet/LibraryResources) used to confirm success
#   PROBE   shell expression, true when the toolchain is present
#   INSTALL command to auto-install the toolchain ("" = not auto-installable here)
#   CONFIG  cmake configure command (run from inside DIR)
#   HINT    human reason / requirement, shown when a target is skipped
# ---------------------------------------------------------------------------
T_NAME=(); T_DIR=(); T_OUT=(); T_PROBE=(); T_INSTALL=(); T_CONFIG=(); T_HINT=()
add_target() {
    T_NAME+=("$1"); T_DIR+=("$2"); T_OUT+=("$3"); T_PROBE+=("$4")
    T_INSTALL+=("$5"); T_CONFIG+=("$6"); T_HINT+=("$7")
}

COMMON="-DCMAKE_BUILD_TYPE=Release -DBUILD_MATHEMATICA_PACLET=ON"

define_targets_darwin() {
    # Native macOS builds - Apple clang targets either arch from the universal SDK.
    add_target "MacOSX-ARM64" "build_macos_arm64" "MacOSX-ARM64/libHypergraphRewriting.dylib" \
        "true" "" \
        "cmake .. $COMMON -DCMAKE_OSX_ARCHITECTURES=arm64" \
        "native (Apple clang)"

    add_target "MacOSX-x86-64" "build_macos_x64" "MacOSX-x86-64/libHypergraphRewriting.dylib" \
        "true" "" \
        "cmake .. $COMMON -DCMAKE_OSX_ARCHITECTURES=x86_64" \
        "native (Apple clang, x86_64 slice of universal SDK)"

    # Linux cross-toolchains via the messense Homebrew tap.
    add_target "Linux-x86-64" "build_linux_x64" "Linux-x86-64/libHypergraphRewriting.so" \
        "command -v x86_64-linux-gnu-gcc" \
        "brew_install messense/macos-cross-toolchains/x86_64-unknown-linux-gnu && link_linux_cross x86_64 x86_64-unknown-linux-gnu" \
        "cmake .. $COMMON -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAINS/linux-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=x86_64" \
        "Linux x86_64 cross-toolchain (brew messense tap)"

    add_target "Linux-ARM64" "build_linux_arm64" "Linux-ARM64/libHypergraphRewriting.so" \
        "command -v aarch64-linux-gnu-gcc" \
        "brew_install messense/macos-cross-toolchains/aarch64-unknown-linux-gnu && link_linux_cross aarch64 aarch64-unknown-linux-gnu" \
        "cmake .. $COMMON -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAINS/linux-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=aarch64" \
        "Linux ARM64 cross-toolchain (brew messense tap)"

    # Windows x86_64 via MinGW-w64.
    add_target "Windows-x86-64" "build_windows_x64" "Windows-x86-64/HypergraphRewriting.dll" \
        "command -v x86_64-w64-mingw32-gcc" \
        "brew_install mingw-w64" \
        "cmake .. $COMMON -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAINS/windows-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=x86_64" \
        "MinGW-w64 (brew install mingw-w64)"

    # Windows ARM64 needs the Windows SDK + MSVC ARM64 import libs + a resource
    # compiler, which only exist on a Windows/WSL host. Not installable on macOS.
    add_target "Windows-ARM64" "build_windows_arm64" "Windows-ARM64/HypergraphRewriting.dll" \
        "[ -d '/mnt/c/Program Files (x86)/Windows Kits' ] && command -v llvm-rc" \
        "" \
        "cmake .. $COMMON -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAINS/windows-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DWINDOWS_COMPILER=clang" \
        "Windows SDK + MSVC ARM64 libraries (WSL host only)"
}

define_targets_linux() {
    # Native Linux host build.
    add_target "Linux-x86-64" "build_linux_x64" "Linux-x86-64/libHypergraphRewriting.so" \
        "command -v gcc" \
        "" \
        "cmake .. $COMMON" \
        "native gcc"

    add_target "Linux-ARM64" "build_linux_arm64" "Linux-ARM64/libHypergraphRewriting.so" \
        "command -v aarch64-linux-gnu-gcc" \
        "sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu" \
        "cmake .. $COMMON -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAINS/linux-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=aarch64" \
        "gcc-aarch64-linux-gnu"

    add_target "Windows-x86-64" "build_windows_x64" "Windows-x86-64/HypergraphRewriting.dll" \
        "command -v x86_64-w64-mingw32-gcc" \
        "sudo apt-get install -y gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64" \
        "cmake .. $COMMON -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAINS/windows-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=x86_64" \
        "gcc-mingw-w64-x86-64"

    add_target "Windows-ARM64" "build_windows_arm64" "Windows-ARM64/HypergraphRewriting.dll" \
        "[ -d '/mnt/c/Program Files (x86)/Windows Kits' ] && command -v llvm-rc" \
        "" \
        "cmake .. $COMMON -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAINS/windows-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DWINDOWS_COMPILER=clang" \
        "Windows SDK + MSVC ARM64 libraries (WSL host only)"

    # macOS targets from Linux require the OSXCross SDK; left to the maintainer.
    local osx="${OSXCROSS_ROOT:-$HOME/osxcross}"
    add_target "MacOSX-x86-64" "build_macos_x64" "MacOSX-x86-64/libHypergraphRewriting.dylib" \
        "[ -d '$osx' ]" "" \
        "cmake .. $COMMON -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAINS/macos-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=x86_64" \
        "OSXCross SDK at \$OSXCROSS_ROOT"
    add_target "MacOSX-ARM64" "build_macos_arm64" "MacOSX-ARM64/libHypergraphRewriting.dylib" \
        "[ -d '$osx' ]" "" \
        "cmake .. $COMMON -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAINS/macos-cross.cmake -DCMAKE_SYSTEM_PROCESSOR=arm64" \
        "OSXCross SDK at \$OSXCROSS_ROOT"
}

case "$HOST_OS" in
    Darwin) define_targets_darwin ;;
    Linux)  define_targets_linux ;;
    *) echo -e "${RED}Unsupported host OS: $HOST_OS${NC}"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Build loop
# ---------------------------------------------------------------------------
echo -e "${GREEN}=== Hypergraph Rewriting - multi-platform build ===${NC}"
echo -e "Host: ${BLUE}${HOST_OS} ${HOST_ARCH}${NC}   Jobs: ${BUILD_JOBS}   Auto-install: ${AUTO_INSTALL}\n"

command -v cmake >/dev/null 2>&1 || { echo -e "${RED}cmake not found - install it first${NC}"; exit 1; }

echo -e "${GREEN}Cleaning existing library files...${NC}"
rm -f paclet/LibraryResources/*/*.so paclet/LibraryResources/*/*.dll paclet/LibraryResources/*/*.dylib
echo

RESULT_NAMES=(); RESULT_STATUS=()   # status: BUILT | SKIPPED | FAILED
record() { RESULT_NAMES+=("$1"); RESULT_STATUS+=("$2"); }

family_wanted() {
    case "$1" in
        MacOSX-*)  $WANT_MACOS ;;
        Linux-*)   $WANT_LINUX ;;
        Windows-*) $WANT_WINDOWS ;;
    esac
}

for ((i=0; i<${#T_NAME[@]}; i++)); do
    name="${T_NAME[i]}"; dir="${T_DIR[i]}"; out="${T_OUT[i]}"
    probe="${T_PROBE[i]}"; install="${T_INSTALL[i]}"; config="${T_CONFIG[i]}"; hint="${T_HINT[i]}"

    if ! family_wanted "$name"; then continue; fi

    echo -e "${GREEN}=== ${name} ===${NC}"

    # Toolchain availability
    if ! eval "$probe" >/dev/null 2>&1; then
        if [ -n "$install" ] && $AUTO_INSTALL; then
            echo -e "${YELLOW}Toolchain missing - auto-installing ($hint)...${NC}"
            if eval "$install" >/dev/null 2>&1 && eval "$probe" >/dev/null 2>&1; then
                echo -e "${GREEN}Toolchain installed.${NC}"
            else
                echo -e "${YELLOW}Auto-install did not produce a usable toolchain - skipping ${name}.${NC}\n"
                record "$name" "SKIPPED"; continue
            fi
        else
            if [ -z "$install" ]; then
                echo -e "${YELLOW}Skipping ${name} - not supported on this host: ${hint}.${NC}\n"
            else
                echo -e "${YELLOW}Skipping ${name} - toolchain missing (${hint}); re-run without --no-install to add it.${NC}\n"
            fi
            record "$name" "SKIPPED"; continue
        fi
    fi

    # Configure + build
    mkdir -p "$dir"; ( cd "$dir" \
        && eval "$config" >/dev/null 2>&1 \
        && make -j"$BUILD_JOBS" paclet >/dev/null 2>&1 )
    if [ $? -eq 0 ] && [ -f "paclet/LibraryResources/$out" ]; then
        echo -e "${GREEN}✓ ${name} built${NC}\n"
        record "$name" "BUILT"
    else
        echo -e "${RED}✗ ${name} build FAILED (see $dir for logs)${NC}\n"
        record "$name" "FAILED"
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo -e "${GREEN}=== Build Summary ===${NC}"
any_failed=false; any_built=false
for ((i=0; i<${#RESULT_NAMES[@]}; i++)); do
    case "${RESULT_STATUS[i]}" in
        BUILT)   echo -e "${GREEN}✓ ${RESULT_NAMES[i]}${NC}"; any_built=true ;;
        SKIPPED) echo -e "${YELLOW}- ${RESULT_NAMES[i]} (skipped - not buildable on this host)${NC}" ;;
        FAILED)  echo -e "${RED}✗ ${RESULT_NAMES[i]} (build error)${NC}"; any_failed=true ;;
    esac
done
echo -e "\nPaclet libraries in: ${BLUE}paclet/LibraryResources/${NC}"

if $any_failed; then
    echo -e "\n${RED}A build that was attempted failed. Fix it and re-run.${NC}"
    exit 1
fi
if ! $any_built; then
    echo -e "\n${RED}Nothing was built.${NC}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Assemble the paclet archive (everything that built is packed in)
# ---------------------------------------------------------------------------
if $WANT_ARCHIVE; then
    echo -e "\n${GREEN}=== Assembling paclet archive ===${NC}"
    native_dir="build_linux"
    [ "$HOST_OS" = "Darwin" ] && native_dir="build_macos_arm64"
    if [ -d "$native_dir" ]; then
        ( cd "$native_dir" && make create_paclet_archive ) \
            && echo -e "${GREEN}✓ Paclet archive in paclet_archive/${NC}" \
            || echo -e "${YELLOW}Archive assembly failed (libraries are still in paclet/LibraryResources/).${NC}"
    else
        echo -e "${YELLOW}No native build dir to drive archive creation; run 'make create_paclet_archive' in any build dir.${NC}"
    fi
fi

echo -e "\n${GREEN}✓ Done.${NC}"
