# Support Runbook — Common Issues

A short field guide used by the Acme support team. Each section describes a
symptom, the typical root cause, and the recommended remediation.

## AMR loses localisation indoors

Symptom: an AMR-100 or AMR-300 reports "localisation lost" and stops in place.

Most common root causes:

* Reflective floor (freshly polished concrete or metal sheets covering the
  warehouse floor) — the LIDAR returns are saturated.
* Cardboard maze: too many identical cardboard pallets create a feature-poor
  environment for the scan matcher.
* Drift after a forklift bumps into the robot.

Remediation:

1. Trigger a re-localisation by sending the robot back to a known dock.
2. If the warehouse layout changed significantly, re-run the SLAM mapping
   procedure documented in the operations manual.
3. For chronic cases at one site, install the optional ceiling-marker package.

## Conductor REST API returns 503 on /jobs

The Conductor service uses a Postgres-backed job queue. A 503 on ``/jobs``
usually means the Postgres connection pool is exhausted.

Remediation:

1. Check the ``pgbouncer`` connection pool gauge in Grafana.
2. Restart the offending Conductor pod; ArgoCD will recreate it.
3. If it recurs more than twice a day, escalate to the platform team — it's
   probably a leaking long-running query.

## Pick-Arm reports "force limit exceeded"

The Pick-Arm S1 stops with a force-limit error when an unexpected obstacle is
in the workspace. Verify the workspace is clear, then send a ``recover``
command via the Conductor API. If the error repeats with no obstacle, the
torque sensor likely needs recalibration — schedule a hardware visit.
