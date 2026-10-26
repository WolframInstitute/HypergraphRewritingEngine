#pragma once
// The listening socket of `hg_evolve --serve-socket`, and the socket type both sides use.
//
// open_loopback_listener binds a TCP socket to 127.0.0.1 on a port the operating system picks,
// puts it in the listening state, and only then calls `publish(port)`. The paclet reads the port
// from the file `publish` writes and connects at once, so the socket must already be listening:
// a connection to a bound socket that is not listening is refused.

#include <cstring>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
#define HG_CLOSESOCK closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif
#define HG_CLOSESOCK ::close
#endif

// The listening socket, or INVALID_SOCKET with `*error` naming the call that failed. On Windows
// the caller has run WSAStartup.
template <class Publish>
inline socket_t open_loopback_listener(Publish&& publish, const char** error) {
    socket_t listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener == INVALID_SOCKET) { *error = "socket()"; return INVALID_SOCKET; }
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        *error = "bind()";
        HG_CLOSESOCK(listener);
        return INVALID_SOCKET;
    }
    if (listen(listener, 1) != 0) {
        *error = "listen()";
        HG_CLOSESOCK(listener);
        return INVALID_SOCKET;
    }
    socklen_t alen = sizeof(addr);
    getsockname(listener, reinterpret_cast<sockaddr*>(&addr), &alen);
    publish(static_cast<unsigned>(ntohs(addr.sin_port)));
    return listener;
}
