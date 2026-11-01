// The port of a server a browser check starts, taken the way tests/localports.py spawn() takes it: the server is
// started on port 0, the kernel's own free port, and the port is read back from the socket its process listens on
// (/proc). A number read from a socket that is then closed is free for anyone until the server binds it, and
// ci-local.sh runs the workflow's jobs side by side on one machine (review of PR #60, 2026-10-29).
import { readFileSync, readdirSync, readlinkSync } from 'node:fs';

// {inode: port} of the machine's TCP sockets in the listening state
function listening() {
  const out = {};
  for (const table of ['/proc/net/tcp', '/proc/net/tcp6']) {
    let rows = [];
    try { rows = readFileSync(table, 'utf8').split('\n').slice(1); } catch { continue; }
    for (const row of rows) {
      const c = row.trim().split(/\s+/);
      if (c.length > 9 && c[3] === '0A') out[c[9]] = parseInt(c[1].split(':').pop(), 16);
    }
  }
  return out;
}

// The ports the process listens on, from its own sockets
export function portsOf(pid) {
  const inodes = new Set();
  let fds = [];
  try { fds = readdirSync(`/proc/${pid}/fd`); } catch { return []; }
  for (const fd of fds) {
    let link = '';
    try { link = readlinkSync(`/proc/${pid}/fd/${fd}`); } catch { continue; }
    if (link.startsWith('socket:[')) inodes.add(link.slice(8, -1));
  }
  if (!inodes.size) return [];
  const all = listening();
  return [...new Set(Object.entries(all).filter(([inode]) => inodes.has(inode)).map(([, port]) => port))].sort((a, b) => a - b);
}

// The one port proc listens on once it listens, `skip` left out (a port it was given on purpose). Throws when it
// ends first, never listens within the time, or listens on more than one: no port can be the cause any more.
export async function portOf(proc, { within = 25000, skip = [] } = {}) {
  const t0 = Date.now();
  for (;;) {
    const ports = portsOf(proc.pid).filter(p => !skip.includes(p));
    if (ports.length === 1) return ports[0];
    if (ports.length > 1) throw new Error(`the server listens on ${ports.length} ports, ${ports}: which one is meant`);
    if (proc.exitCode !== null || proc.signalCode !== null) throw new Error(`the server ended (${proc.exitCode ?? proc.signalCode}) before it listened`);
    if (Date.now() - t0 > within) throw new Error(`the server did not listen within ${within / 1000} s`);
    await new Promise(r => setTimeout(r, 100));
  }
}
