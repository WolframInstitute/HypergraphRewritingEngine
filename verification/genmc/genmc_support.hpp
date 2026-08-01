#pragma once

// Glue every GenMC harness in this directory needs, kept in one place so a harness reads as the
// property it checks and nothing else.
//
// __dso_handle is emitted by clang whenever a translation unit registers a static object's
// destructor through __cxa_atexit. The symbol is normally supplied by the C runtime startup
// files, which GenMC's interpreter does not load -- it runs the module's IR directly -- so the
// address cannot be resolved and the interpreter aborts before the first thread starts. Defining
// it here gives the registration something to name. Nothing reads the value: GenMC terminates the
// program at the end of main rather than running atexit handlers.

extern "C" {
void* __dso_handle = nullptr;
}
