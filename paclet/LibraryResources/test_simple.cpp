#include "WolframLibrary.h"

EXTERN_C DLLEXPORT int WolframLibrary_getVersion() {
    return WolframLibraryVersion;
}

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) {
}

EXTERN_C DLLEXPORT int getVersionSimple(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    MArgument_setUTF8String(res, "Test 1.0.0");
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int addTwoNumbers(WolframLibraryData libData, mint argc, MArgument *argv, MArgument res) {
    mint a = MArgument_getInteger(argv[0]);
    mint b = MArgument_getInteger(argv[1]);
    MArgument_setInteger(res, a + b);
    return LIBRARY_NO_ERROR;
}