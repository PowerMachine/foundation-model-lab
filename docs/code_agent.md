# Codex-style coding agent

The agent alternates between one model-produced JSON action and one bounded tool result.
It can inspect files, create a new file, make an exact replacement patch, and run an
argument-vector command. It never invokes a shell.

## Safety boundary

`PodmanRunner` is the default boundary for model-generated code. It uses an existing local
image with `--pull=never`, disables networking, mounts only a disposable task workspace,
uses a read-only root filesystem, drops every capability, and enables no-new-privileges.
Commands are also constrained by an executable allowlist, timeout, CPU, memory, output,
file-size, and process-count limits. See `docs/agent_security.md` for the complete boundary.

`bwrap` remains an experimental runner, but namespace creation fails on this host. `process`
mode has resource and path limits but **does not isolate the network**. Both modes must not
be used with untrusted model output on this machine.
The generated HTML trace records which mode was used and whether network isolation was on.

## Providers

- `TransformersProvider`: directly loads the local Qwen Coder checkpoint.
- `LocalOpenAIProvider`: talks only to an OpenAI-compatible localhost endpoint such as vLLM.
- `ScriptedProvider`: deterministic end-to-end testing without a model.

The smoke demo intentionally writes a faulty implementation, observes the failing test,
patches the implementation, reruns it, and renders every model/tool event to HTML.

