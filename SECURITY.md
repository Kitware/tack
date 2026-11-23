# Security policy

## Reporting a vulnerability

Report a vulnerability privately, not in a public issue, pull request or
discussion: open the repository's **Security** tab and choose **Report a
vulnerability** ([direct link](https://github.com/Kitware/tack/security/advisories/new)).
The report stays private between you and the project's maintainers.

Say what is affected (the version or commit, the package, the backend), how
to reproduce it, and what an attacker could do with it. A small reproduction
helps most.

## What happens next

A maintainer acknowledges the report within a week and keeps you informed
from then on. The fix is worked out in a private advisory with you. We agree
with you on when to disclose, and publish the advisory with the release that
fixes the problem, crediting you unless you would rather not be named.

## Supported versions

Tack has not reached 1.0 and releases often. Fixes go into the next release:
only the latest minor version (the newest of the
[releases](https://github.com/Kitware/tack/releases)) is supported, and
earlier versions are not patched.

## What counts

Tack compiles and runs the code its caller gives it, on the caller's machine
and devices, with the caller's permissions. A kernel that does what its
author wrote is not a vulnerability, even when what it does is harmful. These
are, among others:

- data from outside the program (a file read through `tack.dataset` or its
  interop, a DLPack tensor, a message from another MPI rank) that makes Tack
  corrupt memory, read outside an allocation or run code;
- the disk cache (`TACK_CACHE_DIR`) loading an entry that another user could
  have written or changed: its entries are compiled code;
- valid use of the public API that reads or writes outside the memory it was
  given.

A vulnerability in a dependency (LLVM, a GPU runtime, VTK, NumPy) belongs to
that project; tell us too if Tack needs a change because of it.
