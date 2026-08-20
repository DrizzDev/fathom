# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for security vulnerabilities.

Report privately through GitHub's
[private vulnerability reporting](https://github.com/DrizzDev/fathom/security/advisories/new),
or email **security@drizz.dev**.

Please include:

- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- the affected version and your environment (OS, Python version).

We aim to acknowledge a report within three business days and to keep you updated as we
investigate and prepare a fix. Please give us a reasonable window to release a fix before
any public disclosure.

## Data handling

To plan actions, Fathom captures screenshots and the on-screen view hierarchy of the app
under automation. These artifacts are written to the local machine by default; cloud upload
is off unless you explicitly enable a cloud storage backend and provide a bucket. Only run
Fathom against apps and data you are authorized to automate, and never commit captured
screenshots, hierarchies, device serials, or `.env` files.

## Supported versions

Security fixes are provided for the latest released version.
