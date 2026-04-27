# Engineering Handbook

This handbook captures the engineering practices that the Acme Robotics
software division agreed on at the start of 2024.

## Languages and frameworks

The Conductor backend is written in Python 3.12. New services use FastAPI
with asynchronous handlers; legacy services are still on Flask but are being
migrated. The simulation layer uses C++ 20 and ROS 2 Humble.

For machine-learning code we standardise on PyTorch 2.x. Inference services
that need to run on the customer site use ONNX Runtime to avoid carrying the
PyTorch dependency.

## Databases

The system of record for fleet state is PostgreSQL 16. We use the pgvector
extension for similarity search over textual data — for example, mapping
free-form support tickets to known root causes. Indexes are HNSW with the
default parameters of pgvector 0.7+. Read-only analytical workloads run
against a daily DuckDB export.

For ephemeral state (job queues, ETA caches) we use Redis 7.

## Testing

Every backend service must reach 80% line coverage measured by ``pytest --cov``.
We rely on three layers of tests:

* **Unit** — pure functions, run on every push.
* **Integration** — spun up against a Postgres + Redis testcontainer.
* **End-to-end** — runs once a night against a synthetic warehouse simulation.

LLM-touching code is tested with mocked clients; we maintain a golden set of
50 support-ticket examples that gates releases of the ticket-routing service.

## Deployment

We deploy to Kubernetes via ArgoCD. CI runs on GitHub Actions and produces
container images that are signed with cosign. SonarQube quality gates block
merges if new critical issues are introduced.
