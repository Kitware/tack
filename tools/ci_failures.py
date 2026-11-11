# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Which tests failed in CI lately, and which of them look flaky.

Reads the CI workflow's runs of the last DAYS days through GitHub's API, every
attempt of a re-run run included, takes the failing tests from the failed
jobs' logs (pytest's ``FAILED``/``ERROR`` lines) and the timing job's flake
warnings, and prints a Markdown report, a row per test::

    python3 tools/ci_failures.py [--days 14] [--repo Kitware/tack]

A failure is flagged when it says more than "this pull request broke
something":

- **main**: on a push to ``main``, where nothing should fail;
- **re-run**: the same job passed on the same commit, in another attempt or
  another run (a re-run that went green);
- **branches**: it failed on more than one branch in the window;
- **flake**: the timing job ran it again and it passed.

The token comes from ``GITHUB_TOKEN`` or ``GH_TOKEN``, else from ``gh auth
token``. The weekly ``ci-failures`` workflow writes the report to its run's
summary. A test flagged here is fixed or explained, not hidden.
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

API = "https://api.github.com"
WORKFLOW = "ci.yml"

# pytest -q's short summary: "FAILED path::test - message", "ERROR path::test".
# Job logs prefix each line with a timestamp.
_FAILED = re.compile(r"\b(?:FAILED|ERROR) (\S+::\S+)")
# The timing job's warning (ci.yml): "failed, then passed when run again: a b".
_FLAKY = re.compile(r"failed, then passed when run again: (.+)$")


@dataclass
class Failure:
    test: str
    run: int
    attempt: int
    job: str
    event: str
    branch: str
    sha: str
    created: str
    flake: bool = False         # the timing job passed it on a second try


@dataclass
class Row:
    test: str
    failures: list = field(default_factory=list)
    flags: set = field(default_factory=set)


def failing_tests(log):
    """The test ids a job's log reports failed, in order, once each."""
    seen = {}
    for line in log.splitlines():
        match = _FAILED.search(line)
        if match:
            seen.setdefault(match.group(1).rstrip(","), None)
    return list(seen)


def flaky_tests(log):
    """The test ids the timing job's warning names (passed on a second try)."""
    found = {}
    for line in log.splitlines():
        match = _FLAKY.search(line.strip())
        if match:
            for test in match.group(1).split():
                found.setdefault(test, None)
    return list(found)


def classify(failures, jobs):
    """Rows of failures by test, flagged. ``jobs`` is every job seen:
    ``(sha, job name, conclusion)`` tuples, for the re-run flag."""
    passed = {(sha, name) for sha, name, conclusion in jobs if conclusion == "success"}
    rows = {}
    for failure in failures:
        row = rows.setdefault(failure.test, Row(failure.test))
        row.failures.append(failure)
        if failure.event == "push" and failure.branch == "main":
            row.flags.add("main")
        if failure.flake:
            row.flags.add("flake")
        elif (failure.sha, failure.job) in passed:
            row.flags.add("re-run")
    for row in rows.values():
        if len({f.branch for f in row.failures}) > 1:
            row.flags.add("branches")
    order = ("main", "flake", "re-run", "branches")
    return sorted(rows.values(), key=lambda r: (-sum(f in r.flags for f in order),
                                                -len(r.failures), r.test))


def _row(row, repo):
    """One table row: the test, its failures, flags, branches, the last run."""
    last = max(row.failures, key=lambda f: f.created)
    branches = sorted({f.branch for f in row.failures})
    shown = ", ".join(f"`{b}`" for b in branches[:4]) + (" ..." if len(branches) > 4 else "")
    link = (f"[{last.created[:10]}](https://github.com/{repo}/actions/runs/"
            f"{last.run}/attempts/{last.attempt})")
    return (f"| `{row.test}` | {len(row.failures)} | {', '.join(sorted(row.flags)) or '-'} "
            f"| {shown} | {link} |")


def report(rows, repo, days, runs, today, unread=0):
    """The Markdown report."""
    total = sum(len(r.failures) for r in rows)
    missing = f" ({unread} more could not be read)" if unread else ""
    summary = (f"{runs} runs of `{WORKFLOW}` in {repo}{missing}; "
               f"{total} failures of {len(rows)} tests.")
    lines = [f"# CI failures, {days} days to {today}", "", summary, ""]
    if not rows:
        return "\n".join([*lines, "No test failed."]) + "\n"
    legend = ("Flags: **main** failed on a push to main; **re-run** the same job passed "
              "on the same commit; **branches** failed on more than one branch; "
              "**flake** the timing job passed it when run again.")
    flagged = [row for row in rows if row.flags]
    plain = [row for row in rows if not row.flags]
    header = ["| Test | Failures | Flags | Branches | Last |", "|---|---|---|---|---|"]
    lines += [legend, ""]
    lines += [*header, *(_row(row, repo) for row in flagged)] if flagged else ["No failure is flagged."]
    if plain:
        summary = f"{len(plain)} tests failed on one branch alone, unflagged"
        lines += ["", f"<details><summary>{summary}</summary>", "", *header,
                  *(_row(row, repo) for row in plain), "", "</details>"]
    return "\n".join(lines) + "\n"


# ── GitHub ──────────────────────────────────────────────────────────

def _token():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True,
                          check=True).stdout.strip()


class _SameHostAuth(urllib.request.HTTPRedirectHandler):
    """Drop the token on a redirect to another host: a job's log redirects to
    blob storage, which refuses a request that carries GitHub's token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_SameHostAuth)


class GitHub:
    def __init__(self, repo, token):
        self.repo, self.token = repo, token

    def get(self, path, raw=False, **params):
        """GET a path under the repository, retrying a server error twice."""
        url = f"{API}/repos/{self.repo}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"})
        for wait in (2, 8, None):
            try:
                with _OPENER.open(request, timeout=60) as response:
                    body = response.read()
                break
            except urllib.error.HTTPError as error:
                if error.code < 500 or wait is None:
                    raise
            except urllib.error.URLError:
                if wait is None:
                    raise
            time.sleep(wait)
        return body.decode("utf-8", "replace") if raw else json.loads(body)

    def pages(self, path, key, **params):
        page = 1
        while True:
            items = self.get(path, per_page=100, page=page, **params)[key]
            yield from items
            if len(items) < 100:
                return
            page += 1

    def log(self, job):
        try:
            return self.get(f"actions/jobs/{job}/logs", raw=True)
        except urllib.error.HTTPError as error:    # expired, or never written
            print(f"no log for job {job}: {error.code}", file=sys.stderr)
            return ""


# The timing job's step that runs a failure again (ci.yml): it runs, and its
# log names a flake, only when the first try failed.
_RUN_AGAIN = "Run the failures once more"


def _run_failures(github, run):
    """A run's failures and jobs, every attempt's: one call for the jobs, a
    log only for a failed job or a timing job that ran a failure again."""
    failures, jobs = [], []
    for job in github.pages(f"actions/runs/{run['id']}/jobs", "jobs", filter="all"):
        jobs.append((run["head_sha"], job["name"], job["conclusion"]))
        failed = job["conclusion"] == "failure"
        again = job["name"] == "timing" and any(
            step["name"] == _RUN_AGAIN and step["conclusion"] == "success"
            for step in job.get("steps") or ())
        if not (failed or again):
            continue
        log = github.log(job["id"])
        found = [(t, False) for t in failing_tests(log)] if failed else []
        found += [(t, True) for t in flaky_tests(log)] if again else []
        for test, flake in found:
            failures.append(Failure(test, run["id"], job.get("run_attempt", 1), job["name"],
                                    run["event"], run["head_branch"], run["head_sha"],
                                    run["created_at"], flake))
    return failures, jobs


def collect(github, since, workers=4):
    """Every failure and every job of the CI runs created since ``since``,
    and how many runs were read and how many could not be."""
    runs = [run for run in github.pages(f"actions/workflows/{WORKFLOW}/runs", "workflow_runs",
                                        created=f">={since}")
            if run["status"] == "completed"]
    failures, jobs, unread = [], [], 0
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_run_failures, github, run): run for run in runs}
        for future in concurrent.futures.as_completed(futures):
            try:
                found, seen = future.result()
            except (urllib.error.URLError, OSError, ValueError) as error:
                print(f"run {futures[future]['id']} not read: {error}", file=sys.stderr)
                unread += 1
                continue
            failures += found
            jobs += seen
    return failures, jobs, len(runs) - unread, unread


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "Kitware/tack"))
    args = parser.parse_args(argv)
    today = datetime.date.today()
    since = (today - datetime.timedelta(days=args.days)).isoformat()
    failures, jobs, runs, unread = collect(GitHub(args.repo, _token()), since)
    sys.stdout.write(report(classify(failures, jobs), args.repo, args.days, runs, today, unread))


if __name__ == "__main__":
    main()
