# Security and data handling

Report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/EmreKaplaner/rag-jev/security/advisories/new).
Include affected version, reproduction steps using synthetic data, impact and any suggested
fix. Do not put credentials, private chunks, or exploit details into a public issue.
The latest tagged release is the supported version. This is a maintained beta without
a contractual response-time guarantee.

Live scoring sends the query (or explicit standalone retrieval query), passage text, and
relevance guidance to TypeSafe. Arbitrary metadata and evaluation labels are not sent.
Optional answer generation sends the question and selected/baseline passage text to the
configured model endpoint. Provider retention and billing policies are separate from this
MIT-licensed integration. The application does not persist documents on the server.

Replay exports, calibration traces and evaluation artifacts include source text and may
include answers. Treat them like the underlying documents. Never commit private replays.
The key-free `--replay-only` server constructs no scoring/generation provider and refuses
live inference endpoints, even when credentials exist in the environment.

Keep the service on loopback by default. Non-loopback CLI binding requires a service
bearer token. Run behind TLS and your own access controls when deployed. Apply document
authorization before retrieval results reach this service. Selection is not an access-
control system or a prompt-injection defense. Fail-open mode returns all original chunks;
it does not promise that a downstream token budget will be met.
