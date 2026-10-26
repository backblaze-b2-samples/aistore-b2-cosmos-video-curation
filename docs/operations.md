# Operations

This runbook covers moving the sample beyond the local Compose stack and validating it on real B2 and GPU hosts.

## Deployment boundary

The Compose stack is a single-host development environment: one proxy and one target, unauthenticated HTTP, local volumes, and placeholder client credentials. Keep its loopback bindings. For a shared GPU cluster, deploy AIS with [ais-k8s](https://github.com/NVIDIA/ais-k8s) (then `make attach` with `AIS_ENDPOINT` pointing at it), enable [AuthN](https://docs.nvidia.com/aistore/authn), and terminate TLS on proxies and target redirects. The sample's clients use placeholder S3 credentials, so switching them to an authenticated cluster requires adapting both the native SDK and S3 client authentication. Test listing, prefetch, reads, multipart writes, and redirects against the real gateway.

## Credentials

Keep B2 keys in a secret manager or orchestrator secrets, scoped to the needed buckets with `listBuckets`, `listFiles`, `readFiles`, `writeFiles`. Only AIS targets and the direct-to-B2 tools need them; Curator subprocesses never receive them. To rotate a key, update the secret, restart the targets, and check one cold read and one write.

## Artifacts and recovery

Publishing follows the [publication protocol](../README.md#publication-protocol). Give each run a unique remote prefix and each dataset revision an immutable prefix. The sample has no distributed locking, so the scheduler must enforce one writer per prefix across machines.

Clean up abandoned prefixes with a separate maintenance key under an explicit retention policy; the sample's key needs no delete permission. The protocol is not a backup, so set B2 retention to match your recovery objective. Verify B2's durable copy before restoring from it:

```bash
make verify PREFIX=curated/<run-id> ARGS="--direct"
```

Losing the AIS cache costs only a refetch from B2. `make down` keeps cache and proxy identity volumes; `down -v` discards both.

## Monitoring

Size AIS mountpaths for the working set and node-local disk for Curator work directories and interrupted runs. Track cache hits and misses, backend GET/PUT bytes and errors, prefetch failures, disk and inode usage, GPU peak memory, and pending publications. Alert on backend access loss, disk pressure, failed final sync, and pending artifacts older than a run. AIS exports metrics as described in its [observability docs](https://github.com/NVIDIA/aistore/tree/main/docs).

The AIS target's `aws.get.n` / `aws.get.size` counters measure GETs AIS issued to B2 (prefetch and cold reads); reconcile them with B2 usage reports for billing. When benchmarking, record region, link bandwidth, object sizes, cache state, versions, node count, and page-cache handling, and do not drop caches on shared hosts.

## Upgrades

CI runs lint and unit tests with Action SHAs pinned and `uv.lock` enforced; it never touches real B2. Dependabot proposes Python, Action, and Compose image updates. It skips the `aistore` SDK: bump the SDK with the image digest in one change and rerun `make test-integration`. Review upstream code before changing commit or checksum pins.

The Curator environment has its own lock (`deploy/curator/uv.lock`, Linux x86_64 / Python 3.12). Record the driver, CUDA toolkit, model revisions and hashes, effective arguments, and the publication manifest for each experiment. The wrapper owns its local Curator process group; external or detached Ray services need their own shutdown procedure.

## GPU smoke test on real B2

On a Linux x86_64 GPU host with FFmpeg 8.0+ and `.env` filled in:

```bash
make setup-curator
make up
make seed      # default 50-video subset, ~60 MB
make run       # unlimited run: prefetches raw/ first
make verify PREFIX=curated/<run-id> ARGS="--direct"
```

Run without `--video-limit`: bounded runs skip prefetch and read cold. The run must print `Curator stage: 0 B2 GETs, 0 bytes (AIS backend counters)`, meaning every Curator read was served by AIS. Then check that clips are readable, embedding dimensions are correct, and metadata matches the downstream dataset contract.

Record results with driver, CUDA, and model versions. Unit tests and stub processes verify orchestration only.
