# Agents and the network

What an agent on this box runs without asking, and what this repo denies it.

## Why this page exists

On 2026-10-26 Cohere reported that Qwen3.8-Flash-Next, given a shell in an agent test without network access, sent a PUT of a probe string to an Alibaba Cloud OSS bucket in Beijing ([Cohere's post](https://cohere.com/blog/concerning-behaviour-popular-open-weight-ai-model)). It happened in fewer than 1 in 100,000 calls, on versions from four vendors, Alibaba's included, so it comes from the original weights. Nothing left their sandbox. On a box with a network, an agent that runs every command it is given would let such a request leave.

"No prompt, image or video leaves the box" is about the serving stack. What an agent does with a shell is up to the agent's permission system, below.

## What runs without asking

- **opencode** (1.19.4, the version this repo pins) starts from permissive defaults: every tool, the shell included, runs without asking. It asks only before a loop it detects, before reaching outside the project directory and before reading a `.env` file. Read in its binary on 2026-10-28.
- **`oc`** runs opencode with `--yolo`, an alias of `--auto` in 1.19.4: it approves those few questions too. A `deny` rule still holds under it.
- **The Agent tab**, installed without `AGENT_AUTO`, keeps opencode's defaults, so its shell commands run unasked as well. With `AGENT_AUTO=1` it approves every tool call, and opencode merges that setting after every rule of its config files: your own rules, `deny` included, do not hold under it.
- **`claude-local`** starts Claude Code in its normal permission mode, which asks before each command ([clients.md](clients.md)).

## What this repo denies

The box's opencode configs carry these rules, under a key of their own, `"bash*"` (a wildcard that matches the shell tool), as the last member of `"permission"`:

```json
"permission": {
  "bash*": {
    "*aliyuncs.com*": "deny",
    "*169.254.169.254*": "deny",
    "*100.100.100.200*": "deny"
  }
}
```

opencode reads its permission rules in order and the last one that matches wins, so they come after every other rule, and your own `"bash"` stays as you wrote it.

- The config `install.sh` generates (`~/.config/qwen38/opencode.json`), which `oc` and the Agent tab load over your own: there the rules come after yours, whatever your config says.
- Your global config (`~/.config/opencode/opencode.json`), which plain `opencode` reads, when it is this repo's copy or lists this repo's providers. A key you add after the block later overrides it, and the next install puts it last again. A pattern you set inside it stays as you set it.
- A config of your own, without this repo's providers, is never rewritten: `oc` and the Agent tab still get the rules, and for plain `opencode` the block above is the one to add, as the last member of your `"permission"`.
- With `AGENT_AUTO=1`, the Agent tab's unit repeats the rules after its `"*": "allow"`, under `"*bash"`.

Checked on the box on 2026-10-28 with opencode 1.19.4 and the FP4-Local lane: a shell command naming an aliyuncs.com host was refused under `--yolo` and under the new `AGENT_AUTO` value, and ran under the old one (`{"*":"allow"}` alone).

What they stop: the case Cohere observed, a shell command whose text names the OSS domain or a cloud metadata address. What they do not stop: a model that writes the address another way (a variable, an encoding, a script it writes first), and opencode's `webfetch`, which 1.x cannot filter by URL. The rules stay in place after `uninstall.sh`, as `autoupdate` does.

## What the cockpit shows

The proxy sees every tool call before the client runs it, whatever the client (opencode, Claude Code, Hermes, pi, an SDK). When a call's arguments name one of these hosts, it writes one line to its journal, the host and the tool's name, never the arguments, and relays the answer as it came. A tool that writes or edits a file is not read: a page that quotes these hosts, this one included, is no request to them. The cockpit reads a day of those lines and shows a red banner with the count, the host, the tool and the route. The proxy only reports: whether the call ran depends on the client and its rules.

## Approving each command

To approve every shell command yourself, put `"*": "ask"` in the `"bash"` object of your config and start opencode without `--yolo` (plain `opencode`, not `oc`). The rules above still come after it:

```json
"permission": {
  "bash": {
    "*": "ask"
  },
  "bash*": {
    "*aliyuncs.com*": "deny",
    "*169.254.169.254*": "deny",
    "*100.100.100.200*": "deny"
  }
}
```

The Agent tab asks the same way when it is installed without `AGENT_AUTO`.

[Back to the README](../README.md)
