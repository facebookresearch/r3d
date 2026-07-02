# Contributing to R3D

We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

R3D is developed internally at Meta and mirrored to this public repository.
Changes are reviewed internally and synced out; pull requests are imported and
land through the same internal review process.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes (`bash scripts/run_tests.sh`).
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style

* Follow [PEP 8](https://peps.python.org/pep-0008/) and format with
  [Black](https://github.com/psf/black).
* Keep functions focused and add type hints where practical.
* Match the conventions of the surrounding code.

## License

By contributing to R3D, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
