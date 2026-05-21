# NexusGrid Platform — System Specification (v3.2)

## Executive Summary

NexusGrid is a multi-tenant edge orchestration platform. The **Control Plane** (service `nexus-api`) coordinates **Worker Nodes** (`nexus-agent`) across regions. Data flows through **EventBridge** into **Analytics Lake** (S3 + Athena). Authentication is centralized in **AuthForge** (OAuth2 + JWT, RS256).

---

## Service Topology

| Service ID       | Role                          | Depends On              | Port  |
|------------------|-------------------------------|-------------------------|-------|
| nexus-api        | REST + GraphQL control plane  | AuthForge, Redis-Cluster| 8443  |
| nexus-agent      | Edge workload executor        | nexus-api, Vault        | 9100  |
| EventBridge      | Async event bus (Kafka)       | Schema Registry         | 9092  |
| Analytics Lake   | Batch + stream analytics      | EventBridge, S3         | —     |
| AuthForge        | Identity provider             | Postgres-Primary        | 443   |
| Redis-Cluster    | Session + rate-limit cache    | —                       | 6379  |
| Vault            | Secret distribution           | HSM-KMS                 | 8200  |

### nexus-api

- **Owner team:** Platform Core (`team-platform@nexusgrid.io`)
- **SLA:** 99.95% monthly uptime
- **Scaling:** HPA on CPU > 70%, min 3 / max 24 pods per region
- **Critical path:** All `nexus-agent` heartbeats route through `nexus-api` before persistence in **Redis-Cluster** keyspace `agent:hb:{node_id}`

### nexus-agent

- Registers with `nexus-api` using mTLS client certs issued by **Vault** path `pki/agent/{region}`
- Publishes workload metrics to **EventBridge** topic `metrics.workload.v1`
- **Failure mode:** If heartbeat missed for 90s, `nexus-api` marks node `DEGRADED` and triggers alert `ALERT-NODE-STALE`

### EventBridge

- Retention: 7 days hot, 90 days cold (S3 tier)
- **Consumers:** Analytics Lake (Flink job `lake-ingest-primary`), audit logger `audit-sink`
- Schema enforced via **Schema Registry** subject `nexus.events.{domain}`

### Analytics Lake

- Ingestion SLA: events visible in Athena within **5 minutes** (p95)
- **Query federation:** Trino coordinator `trino-coord-01` joins S3 parquet with Postgres dimension tables
- Cost guardrail: queries > 10 TB scanned require approval workflow `WF-COST-OVERRIDE`

### AuthForge

- Issues JWT with claims: `sub`, `tenant_id`, `roles[]`
- Token TTL: access 15m, refresh 7d
- **Integration:** `nexus-api` validates JWT via JWKS endpoint `/.well-known/jwks.json`
- Break-glass admin role `ROLE_BREAK_GLASS` requires MFA + ticket `INC-*`

---

## Data Flow: Deployment Pipeline

1. Developer pushes to `main` → **CI Runner** (`ci-runner-gcp`) executes pipeline `deploy-v3`
2. Pipeline calls `nexus-api` endpoint `POST /v1/releases` with artifact digest
3. `nexus-api` writes release record to **Postgres-Primary** table `releases`
4. `nexus-api` emits `ReleaseCreated` event to **EventBridge**
5. **nexus-agent** nodes subscribe via SSE channel `/stream/releases`
6. Agents pull artifacts from **Artifact Registry** (`registry.nexusgrid.io`)
7. **Analytics Lake** Flink job correlates deploy events with error-rate metrics

**Rollback policy:** Automatic rollback if error rate > 2% for 10 minutes post-deploy (monitored by `SLO-WATCHER`).

---

## Security & Compliance

- **Encryption at rest:** AES-256-GCM via **HSM-KMS** key `alias/nexus-data`
- **Encryption in transit:** TLS 1.3 minimum; mTLS between agents and API
- **Audit:** All admin actions logged to `audit-sink` with immutable hash chain
- **Compliance frameworks:** SOC2 Type II, ISO 27001 (annual audit Q4)

### Incident Response

| Severity | Response Time | Escalation            |
|----------|---------------|------------------------|
| SEV-1    | 15 minutes    | PagerDuty + VP Eng     |
| SEV-2    | 1 hour        | On-call lead           |
| SEV-3    | 4 hours       | Team channel           |

**SEV-1 triggers:** AuthForge outage, regional `nexus-api` total failure, Vault seal breach.

---

## Observability Stack

- **Metrics:** Prometheus (`prometheus-federation`) scrapes all services every 15s
- **Traces:** OpenTelemetry → Tempo (`tempo-distributor`)
- **Logs:** Vector agent → Loki (`loki-ingest`)
- **Dashboards:** Grafana folder `NexusGrid/Production`

**Golden signals** monitored per service: latency (p99), traffic (RPS), errors (5xx %), saturation (CPU/memory).

---

## Disaster Recovery

- **RPO:** 15 minutes (Postgres streaming replication to `dr-postgres-east`)
- **RTO:** 60 minutes for full region failover (`us-east-1` → `us-west-2`)
- **Neo4j dependency note (internal CMDB):** Service dependency graph mirrored in CMDB Neo4j instance for impact analysis — not production data path.

---

## Evaluation Reference Facts (Benchmark Ground Truth)

Use these facts when validating RAG answers:

1. **nexus-agent** heartbeat timeout before `DEGRADED` status: **90 seconds**.
2. **AuthForge** access token TTL: **15 minutes**.
3. **Analytics Lake** p95 ingestion SLA to Athena: **5 minutes**.
4. **nexus-api** monthly uptime SLA: **99.95%**.
5. Automatic deploy rollback triggers when error rate exceeds **2%** for **10 minutes**.
6. **EventBridge** hot retention period: **7 days**.
7. **SEV-1** incident response time: **15 minutes**.
8. DR **RPO** target: **15 minutes**.
