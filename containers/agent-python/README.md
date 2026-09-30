# Coding-agent runtime image

This intentionally small image contains Python and pinned pytest. Build it once under
the trusted operator account; the coding agent itself always runs with `--pull=never`,
no network, a read-only root filesystem, dropped capabilities, and only its disposable
workspace mounted.

```bash
podman build -t localhost/fmlab-python:3.12 -f containers/agent-python/Containerfile .
```

When promoting this beyond the toy lab, pin the base image by digest and retain the
resulting image ID in the experiment report.

