"""Ports for the tests that start servers, safe beside every other program of the machine.

A port number read from a socket that is then closed is free for anyone until the server it was taken
for binds it: ci-local.sh runs the workflow's jobs side by side on one machine, three of them run the
same test files, and any program there can be handed that port meanwhile (code review, 2026-10-28).
run_suite.py ran those files one at a time, which kept them from each other inside one job, not from the
other jobs. Three answers, each proven in test_localports.py:

- held(): a port kept for the test, bound and never listening. A connection to it is refused, as to a
  lane that is gone, and no other program can bind it: the socket has no SO_REUSEADDR, with which
  another socket could share the port while this one is not listening.
- serve(port, ...): a server of this process on a held port, shared with the socket that keeps it by
  SO_REUSEPORT: an engine that was gone comes up there, goes down and comes up again, and the port stays
  the test's in between.
- spawn(start): a server in a process of its own, which cannot share a held port. It starts on a port
  that was free a moment ago and is kept once the socket listening there is its own (read in /proc),
  else it starts again on another port.
"""
import os
import socket
import subprocess
import time

HOST = "127.0.0.1"


def held(test=None, host=HOST):
    """(socket, port): a port kept until the socket closes, at the test's cleanup when the test is given."""
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, 0))
    if test is not None:
        test.addCleanup(sock.close)
    return sock, sock.getsockname()[1]


def serve(port, server_class, handler, host=HOST):
    """A server_class (socketserver) listening on a held port, beside the socket that keeps it."""
    class Shared(server_class):
        def server_bind(self):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            super().server_bind()
    Shared.__name__ = Shared.__qualname__ = server_class.__name__
    return Shared((host, port), handler)


def _listener(port):
    """The inode of the socket listening on this TCP port, whatever its address, or None."""
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table) as f:
                rows = f.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            cols = row.split()
            if len(cols) > 9 and cols[3] == "0A" and int(cols[1].rsplit(":", 1)[1], 16) == port:
                return cols[9]
    return None


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


def _holds(pid, inode):
    """Whether the process, or one under it, holds the socket of that inode."""
    for p in _family(pid):
        try:
            fds = os.listdir(f"/proc/{p}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(f"/proc/{p}/fd/{fd}") == f"socket:[{inode}]":
                    return True
            except OSError:
                continue
    return False


def _end(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def spawn(start, tries=5, within=20.0, host=HOST, pick=None):
    """(process, port), start(port) being a Popen of a server that binds port: kept once the socket
    listening on the port is the process's own, started again on another port when the process ended
    first (someone had the port) or someone else listens there. pick(), for this module's tests, names
    the port to try, None for one that was free a moment ago."""
    tried = []
    for _ in range(tries):
        port = pick() if pick else None
        if port is None:
            probe = socket.socket()
            probe.bind((host, 0))
            port = probe.getsockname()[1]
            probe.close()
        proc = start(port)
        deadline, why = time.monotonic() + within, f"nothing listened in {within:g} s"
        while time.monotonic() < deadline:
            inode = _listener(port)
            if inode and _holds(proc.pid, inode):
                return proc, port
            if inode:
                why = "someone else listens there"
                break
            if proc.poll() is not None:
                why = f"it ended with {proc.returncode}"
                break
            time.sleep(0.02)
        tried.append(f"{port}: {why}")
        _end(proc)
    raise RuntimeError(f"no port of its own after {tries} tries ({'; '.join(tried)})")
