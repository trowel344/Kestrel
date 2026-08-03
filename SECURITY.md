# Security policy

## Supported versions

Security fixes are applied to the latest released Kestrel version.

## Reporting a vulnerability

Do not publish API keys, model-provider credentials, private model paths or
security-sensitive logs in a public issue. Open a private GitHub security
advisory in the repository that distributed your copy of Kestrel. Include the
affected version, reproduction steps, impact and the smallest useful log with
secrets removed.

Kestrel never needs a Kimi key on its command line. Remote credentials are read
from `KIMI_API_KEY` or `MOONSHOT_API_KEY`; avoid placing them in tracked files.
