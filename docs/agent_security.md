# Coding-agent container boundary

`PodmanRunner` is the execution boundary for model-generated Python code. It accepts an
argument vector, never a command string, and invokes Podman with `shell=False`. The only
host path visible in the container is the task workspace at `/workspace`.

## Enforced controls

- The image must already exist locally because every run uses `--pull=never`.
- Networking is disabled with `--network=none`; proxy variables are not inherited.
- The image root filesystem is read-only. A size-limited, `noexec`, `nosuid`, `nodev`
  tmpfs is provided only at `/tmp`.
- All Linux capabilities are dropped and `no-new-privileges` is enabled. Podman retains
  its default seccomp confinement.
- PID, memory, swap, CPU rate, CPU time, output, file size, and wall-time bounds are set.
- Image-declared volumes are ignored. Exactly one host bind mount is added: the selected
  workspace, read/write at `/workspace`.
- The environment is cleared and rebuilt from fixed values. Host credentials, model
  directories, SSH agents, container sockets, and the user's home are never mounted.
- Executables use a bare-name allowlist. Python `-c`, interactive/stdin execution,
  unapproved `-m` modules, path traversal, and absolute paths outside `/workspace` are
  rejected before Podman starts.
- Attached output is drained into bounded in-memory buffers. A host timeout kills the
  Podman process group and force-removes the named container as a fallback.

The trusted local image should contain Python and pytest only. Do not add a container
socket, privileged mode, host networking, extra mounts, or a shell entrypoint.

## One-time local image preparation

The runner never downloads an image. An administrator or trusted operator may prepare it
once, outside an agent session. For example, after obtaining the official Python image by
an approved process:

```bash
podman tag docker.io/library/python:3.12-slim localhost/fmlab-python:3.12
podman image exists localhost/fmlab-python:3.12
```

Installing pytest into a purpose-built image is preferable to modifying a running
container. Pin the base image by digest and record the build recipe before using this
boundary for untrusted output.

`PodmanRunner.preflight()` checks the Podman client, workspace permissions, and local
image presence. It does not pull or run the image. `build_command()` is a pure dry-run
builder and is covered by tests that require neither Podman nor an image.

## Residual risks

Containers are not virtual machines. Kernel vulnerabilities and Podman/runtime defects
remain in scope, and the workspace is intentionally writable, so generated code can
destroy files placed there. Create a fresh per-task workspace, copy in only disposable
inputs, inspect the recorded diff, and delete the workspace after retaining the report.
The network namespace blocks ordinary network access but is not a data-loss control for
the mounted workspace.

For stronger isolation, run Podman rootless under a dedicated account and place the whole
agent host inside a VM. Never point the workspace at `/home/user`, `/data`, a model store,
or a repository containing irreplaceable uncommitted work.
