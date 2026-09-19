# Releases and package publishing

The canonical repository is `EmreKaplaner/rag-jev`. Code is MIT licensed. Public registry
publishing requires the package owner's configured account/trusted publisher. Version
0.2.0 is published on both PyPI and npm, with clean registry installations verified.
The PyPI trusted publisher is configured. npm 0.2.0 used authenticated CLI publishing;
its future trusted-publisher setup still needs account confirmation.

1. Update versions in `pyproject.toml`, the HTTP schema, the TypeScript package and Dify
   manifest; regenerate locks, OpenAPI and TypeScript types. Update the changelog.
2. Run `make check-all`, `make schema`, and `uv build`. Test the wheel and npm tarball from
   clean environments. Inspect distribution contents and scan for actual configured secrets.
3. Push the reviewed commit. Require the GitHub Python 3.11/3.13 jobs to pass for that commit.
4. Tag the verified commit, create the GitHub release, and attach wheel, source archive,
   npm tarball, frozen research bundle and checksums. Keep previous study archives immutable.
5. Dispatch `publish.yml` for the release tag and desired registry after account setup.
   Verify installation from the published registry version before announcing availability.

## PyPI

Create a pending trusted publisher for `rag-jev`, GitHub owner `EmreKaplaner`, repository
`rag-jev`, workflow `publish.yml`, environment `pypi`. This supports the first publication
without putting a PyPI token in this repository. Workflow authentication uses short-lived OIDC.

## npm

The package is `rag-jev-client`. First publish requires an authenticated owner and any
required npm account verification/2FA. For later releases configure its GitHub trusted
publisher to owner `EmreKaplaner`, repository `rag-jev`, workflow `publish.yml`, environment
`npm`. The workflow uses npm 11 and provenance. Do not disable account protections or paste
tokens into issues, logs, or source files to work around authentication.

Sources: [PyPI pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/),
[npm trusted publishing](https://docs.npmjs.com/trusted-publishers/).
