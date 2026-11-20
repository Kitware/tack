# Governance

Tack is a young project, started at Kitware. This document says who makes
decisions, how they are made, and how people come to share in them. It is
deliberately small and will grow with the project (see the last section).

## Roles

**Contributors** are everyone who opens an issue, submits a pull request or
reviews one. [CONTRIBUTING.md](CONTRIBUTING.md) says how, and the
[Code of Conduct](CODE_OF_CONDUCT.md) applies to everyone.

**Maintainers** review and merge pull requests, triage issues, look after
CI, and plan releases with the lead. They act for the project, not for
their employers.

**The project lead** is a maintainer who sets the project's direction,
decides when maintainers cannot agree, and is responsible for releases.

| Role         | Name            | Organization | GitHub                                              |
|--------------|-----------------|--------------|-----------------------------------------------------|
| Project lead | Berk Geveci     | Kitware      | [berkgeveci](https://github.com/berkgeveci)         |
| Maintainer   | Jeff Lee        | Kitware      | [jlee-kitware](https://github.com/jlee-kitware)     |
| Maintainer   | Utkarsh Ayachit | NVIDIA       | [utkarshayachit](https://github.com/utkarshayachit) |

## How decisions are made

Most decisions are made in pull requests. A pull request is merged when CI
passes, a maintainer other than its author has approved it, and no
maintainer's objection is left unanswered. A change to documentation or CI
configuration alone may be merged by its author once CI passes. While the
lead is the only maintainer taking part, the lead merges their own pull
requests once CI passes.

Larger changes start with an issue, before the code: a new backend, a
change to the public API that breaks existing callers, a new required
dependency, or a change to the license or to this document. That lets the
other maintainers weigh in while changing course is still cheap. Longer
design notes go in `docs/design/`.

Maintainers try to reach agreement in the issue or pull request. When they
cannot, the lead decides and says why in the same thread.

## Reviewing a pull request

A review asks whether the change is right and whether it is ready:

- **It does what it says.** It does what its issue asks, or its plan (a
  page in `docs/design/plans/`, accepted when its own pull request merged),
  and says where it departs. A plan's pull requests are reviewed against the
  plan: is this what it called for, and is the risk acceptable?
- **It is tested.** New behaviour has tests that fail without the change,
  and a fix the test that shows the bug. Tests check behaviour and values
  rather than coverage (a kernel's body never runs as Python) and use the
  shared backend fixtures, so every backend runs them. A change that CI's
  hardware cannot exercise (a GPU backend other than Metal, MPI across
  machines) says where it was run.
- **Claims are measured.** A claim about speed or memory comes with its
  numbers: the machine, the data, before and after, in one unit.
- **It is complete.** Public names are in `__all__` and the reference
  pages; the guides and examples that show the behaviour are updated; the
  change has its `changelog.d/` entry; a removal is deprecated first
  (`tack._deprecation`); new files carry their SPDX header, and code from
  elsewhere keeps its license.
- **It does nothing else.** No new network access, credentials or secrets,
  and no unrelated changes. A change to what [SECURITY.md](SECURITY.md)
  covers (the disk cache, memory handed outside Tack, readers of outside
  data) gets a careful read.

How maintainers approve:

- A maintainer approves with GitHub's **Approve**. **Request changes** marks
  what must change before merging; a plain comment may wait for a later
  pull request. The author answers every comment, with a change or a reply.
- A change made after approval that goes beyond what the review asked for
  is reviewed again.
- A review request unanswered for a week may go to another maintainer.
- A pull request merges through the merge queue once every check passes;
  no one merges around the checks. One stacked on another merges after it.

## Roadmap and releases

The roadmap is the issue tracker. Each planned release is a milestone, and
the work planned for it is the issues and pull requests assigned to that
milestone. Anyone may propose work by opening an issue; maintainers decide
what goes into a milestone, and the lead decides when a release is made.

The lead, or a maintainer the lead asks, makes a release: gathers the
CHANGELOG with `python tools/changelog.py release VERSION DATE`, tags the
commit, and publishes a GitHub Release, which publishes that version's
documentation.

## Becoming a maintainer

Anyone with a record of good contributions (code, reviews, documentation,
answers to other people's questions) can become a maintainer. A maintainer
nominates them in an issue; they are added if no maintainer objects within
a week, and the lead decides if one does.

Maintainership belongs to the person, not their employer: it does not pass
to a colleague, and it stays with someone who changes jobs. Maintainers who
can no longer take part step down by a pull request to this file, with the
project's thanks.

## Copyright

Contributors, or their employers, keep the copyright to their work, and
license it under the files' license, as [CONTRIBUTING.md](CONTRIBUTING.md)
describes. A file's header names the holders of its copyright: Kitware's
line by default, and an author from another organization adds their own
line to the files they write or substantially change.

## Changing this document

By a pull request, approved by the lead. This model suits a handful of
maintainers. It should be revisited when maintainers come from a third
organization, or before Tack 1.0, whichever comes first; a more formal
model, such as a steering group or a foundation, may suit the project then.
