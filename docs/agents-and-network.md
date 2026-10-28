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

`install.sh` adds these rules to the opencode config the box owns (`~/.config/opencode/opencode.json`, when it is this repo's copy or lists this repo's providers):

```json
"permission": {
  "bash": {
    "*aliyuncs.com*": "deny",
    "*169.254.169.254*": "deny",
    "*100.100.100.200*": "deny"
  }
}
```

opencode reads its permission rules in order and the last one that matches wins, so they go after every shell rule the file already has. A `"bash"` you gave as one action (`"ask"`) becomes `{"*": "ask"}` followed by these rules, so your choice for every other command stands. A pattern you set yourself stays as you set it. The Agent tab's unit repeats the rules after the `"*"` of `AGENT_AUTO=1`.

Checked on the box on 2026-10-28 with opencode 1.19.4 and the FP4-Local lane: a shell command naming an aliyuncs.com host was refused under `--yolo` and under the new `AGENT_AUTO` value, and ran under the old one (`{"*":"allow"}` alone).

What they stop: the case Cohere observed, a shell command whose text names the OSS domain or a cloud metadata address. What they do not stop: a model that writes the address another way (a variable, an encoding, a script it writes first), and opencode's `webfetch`, which 1.x cannot filter by URL.

A config of your own, without this repo's providers, is never rewritten. `install.sh` says so, and the block above is the one to add, after your own shell rules. The rules stay in place after `uninstall.sh`, as `autoupdate` does.

## Approving each command

To approve every shell command yourself, put `"*": "ask"` first in the `"bash"` object of your config, keep the rules above after it, and start opencode without `--yolo` (plain `opencode`, not `oc`):

```json
"permission": {
  "bash": {
    "*": "ask",
    "*aliyuncs.com*": "deny",
    "*169.254.169.254*": "deny",
    "*100.100.100.200*": "deny"
  }
}
```

The Agent tab asks the same way when it is installed without `AGENT_AUTO`.

[Back to the README](../README.md)
