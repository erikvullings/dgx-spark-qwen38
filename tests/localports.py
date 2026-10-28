"""Ports for the tests that start servers, safe beside every other program of the machine.

A port number read from a socket that is then closed is free for anyone until the server it was taken
for binds it: ci-local.sh runs the workflow's jobs side by side on one machine, three of them run the
same test files, and any program there can be handed that port meanwhile (code review, 2026-10-28).
run_suite.py ran those files one at a time, which kept them from each other inside one job, not from the
other jobs. Three answers, each proven in test_localports.py:

- held(test): a port kept for the test, bound and never listening, until the test's cleanup. A
  connection to it is refused, as to a lane that is gone, and no other program can bind it: neither a
  program that binds port 0 (the kernel never hands out a bound port, SO_REUSEPORT or not) nor one that
  asks for the number, unless that one sets SO_REUSEPORT itself under the same user, as nothing here does.
- serve(port, ...): a server of this process on a held port, shared with the socket that keeps it by
  SO_REUSEPORT: an engine that was gone comes up there, goes down and comes up again, and the port stays
  the test's in between. A port something already listens on is refused, as a bind would be.
- spawn(start): a server in a process of its own, started on port 0, the kernel's own free port, which
  is read back from the socket the process listens on (in /proc): no number is ever free between its
  choice and its bind (code review of the first version, which chose a number and tried again).
"""
import errno
import os
import signal
import socket
import subprocess
import time

HOST = "127.0.0.1"


def held(test, host=HOST):
    """(socket, port): a port kept until the test's cleanup closes the socket, which it also keeps alive."""
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, 0))
    test.addCleanup(sock.close)
    return sock, sock.getsockname()[1]


def _listening():
    """{inode: port} of every TCP socket of the machine in the listening state."""
    out = {}
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table) as f:
                rows = f.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            cols = row.split()
            if len(cols) > 9 and cols[3] == "0A":
                out[cols[9]] = int(cols[1].rsplit(":", 1)[1], 16)
    return out


def serve(port, server_class, handler, host=HOST):
    """A server_class (socketserver) listening on a held port, beside the socket that keeps it."""
    if port in _listening().values():
        raise OSError(errno.EADDRINUSE, f"something listens on {port} already")

    class Shared(server_class):
        def server_bind(self):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            super().server_bind()
    Shared.__name__ = Shared.__qualname__ = server_class.__name__
    return Shared((host, port), handler)


def _family(pid):
    """The process and every process under it: a server started through a wrapper listens in a child."""
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            kids.setdefault(ppid, []).append(int(d))
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += kids.get(p, [])
    return out


def _port_of(pid):
    """The port the process, or one under it, listens on, or None."""
    listening = _listening()
    for p in _family(pid):
        try:
            fds = os.listdir(f"/proc/{p}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(f"/proc/{p}/fd/{fd}")
            except OSError:
                continue
            if link.startswith("socket:[") and link[8:-1] in listening:
                return listening[link[8:-1]]
    return None


def _gone(pid):
    """Whether the process has ended (a zombie has: only its parent's wait is left)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


def _end(proc):
    """The process and every process under it, which a wrapper would leave behind, once they have ended."""
    kids = _family(proc.pid)[1:]
    for p in reversed(kids):
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not all(_gone(p) for p in kids):
        time.sleep(0.02)


def spawn(start, within=20.0):
    """(process, port), start(port) being a Popen of a server that binds port: it is given 0 and the
    port is read back once the process listens. A process that ends first, or does not listen within the
    time, is said as such: no port can be the cause any more."""
    proc = start(0)
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        port = _port_of(proc.pid)
        if port:
            return proc, port
        if proc.poll() is not None:
            raise RuntimeError(f"the server ended with {proc.returncode} before it listened")
        time.sleep(0.02)
    _end(proc)
    raise RuntimeError(f"the server did not listen within {within:g} s")
