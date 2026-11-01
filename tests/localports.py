"""Ports for the tests that start servers, safe beside every other program of the machine.

A port number read from a socket that is then closed is free for anyone until the server it was taken
for binds it: ci-local.sh runs the workflow's jobs side by side on one machine, three of them run the
same test files, and any program there can be handed that port meanwhile (code review, 2026-10-28).
run_suite.py ran those files one at a time, which kept them from each other inside one job, not from the
other jobs. Three answers, each proven in test_localports.py:

- held(test): a port kept for the test, bound and never listening, until the test's cleanup. A
  connection to it on 127.0.0.1 is refused, as to a lane that is gone, and no other program can bind it
  there: neither one that binds port 0 (the kernel never hands out a bound port, SO_REUSEPORT or not)
  nor one that asks for the number, unless it sets SO_REUSEPORT itself under the same user, as nothing
  here does. An IPv6-only socket may take the same number on ::1, which no IPv4 client reaches.
- serve(port, ...): a server of this process on a held port, shared with the socket that keeps it by
  SO_REUSEPORT: an engine that was gone comes up there, goes down and comes up again, and the port stays
  the test's in between. A second server of the test's own on a port that listens is refused, as a bind
  would be, since SO_REUSEPORT would have the kernel share the connections between the two.
- spawn(start): a server in a process of its own, started on port 0, the kernel's own free port, which
  is read back from the socket the process listens on (in /proc): no number is ever free between its
  choice and its bind (code review of the first version, which chose a number and tried again). The
  server is the process start() returns: a wrapper execs it.
"""
import errno
import os
import socket
import subprocess
import time

HOST = "127.0.0.1"


def held(test, host=HOST):
    """A port kept until the test's cleanup closes the socket that keeps it, which it also keeps alive."""
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, 0))
    test.addCleanup(sock.close)
    return sock.getsockname()[1]


def _listening(tables=("/proc/net/tcp", "/proc/net/tcp6")):
    """{inode: (address, port)} of the TCP sockets of the machine in the listening state, the address
    as /proc writes it (0100007F is 127.0.0.1, 00000000 is any)."""
    out = {}
    for table in tables:
        try:
            with open(table) as f:
                rows = f.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            cols = row.split()
            if len(cols) > 9 and cols[3] == "0A":
                addr, port = cols[1].rsplit(":", 1)
                out[cols[9]] = (addr, int(port, 16))
    return out


def serve(port, server_class, handler, host=HOST):
    """A server_class (socketserver) listening on a held port, beside the socket that keeps it."""
    if any(p == port and a in ("0100007F", "00000000") for a, p in _listening(("/proc/net/tcp",)).values()):
        raise OSError(errno.EADDRINUSE, f"something listens on {host}:{port} already")

    class Shared(server_class):
        def server_bind(self):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            super().server_bind()
    Shared.__name__ = Shared.__qualname__ = server_class.__name__
    return Shared((host, port), handler)


def _ports_of(pid):
    """The ports the process listens on: its own sockets first, the machine's tables only when it has
    one, since a process starting up has none."""
    inodes = set()
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return []
    for fd in fds:
        try:
            link = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if link.startswith("socket:["):
            inodes.add(link[8:-1])
    if not inodes:
        return []
    return sorted({port for inode, (_, port) in _listening().items() if inode in inodes})


def _end(proc):
    """The server's process, once it has ended."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _why(proc):
    """The end of what the process wrote on its stderr, when start() kept it in a pipe."""
    if proc.stderr is None:
        return ""
    try:
        tail = proc.stderr.read()[-1500:]
    except (OSError, ValueError):
        return ""
    return ": " + (tail.decode(errors="replace") if isinstance(tail, bytes) else tail).strip()


def spawn(start, within=20.0):
    """(process, port), start(port) being a Popen of a server that binds port: it is given 0 and the
    port is read back once the process listens. A process that ends first, does not listen within the
    time or listens on more than one port is said as such: no port can be the cause any more."""
    proc = start(0)
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        ports = _ports_of(proc.pid)
        if len(ports) == 1:
            return proc, ports[0]
        if ports:
            _end(proc)
            raise RuntimeError(f"the server listens on {len(ports)} ports, {ports}: which one is meant")
        if proc.poll() is not None:
            raise RuntimeError(f"the server ended with {proc.returncode} before it listened{_why(proc)}")
        time.sleep(0.05)
    _end(proc)
    raise RuntimeError(f"the server did not listen within {within:g} s")
