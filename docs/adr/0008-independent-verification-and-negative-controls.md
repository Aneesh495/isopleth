# ADR 0008: Independent Verification and Negative Controls

## Status
Accepted

## Context
Self-verifying machine learning pipelines often suffer from circular confirmation bias: the code checking whether an invariant holds shares the same tensor classes, helper utilities, or loss functions as the training model. Furthermore, verification suites without negative controls cannot confirm whether the verifier is capable of detecting realistic corruptions (e.g. data leakage, mass injection, gradient mismatch).

## Decision
We decouple evaluation into an independent zero-model verifier coupled with a mandatory negative controls suite:
1. Zero-Model Verifier (`isopleth/evaluation/independent_check.py`): Recomputes error norms, physical mass drifts, and positivity metrics using raw NumPy arrays without importing any neural network architectures or training routines.
2. Cryptographic Evidence Bundles: The acceptance runner produces a certified JSON evidence bundle containing gate execution diagnostics and a SHA-256 bundle digest.
3. Battery of 12 Negative Controls: The verifier executes 12 negative controls (including 5 semantically invalid bundles having valid cryptographic hashes):
   - Mismatched cryptographic hashes.
   - Missing mandatory gates.
   - Arithmetic contradictions in gate counters.
   - Physical threshold violations (e.g. mass drift > 1e-4).
   - Gradient check discrepancies (> 1e-4).
   - Manifest partition count deficits and leakage.
   - Violated lake-at-rest hydrostatic equilibria.
   - Inverted autocatalytic reaction signs.
4. Acceptance Gate IA12: The release is only certified if the verifier passes valid bundles and decisively rejects all 12 negative controls.

## Consequences
### Positive:
- Independent verification prevents accidental regressions or silent bugs from being masked by shared utility code.
- Negative controls prove the verifier has high discriminative power and does not act as a passive rubber stamp.

### Negative:
- Running the full negative controls suite adds a short delay (under 1 second) to the verification campaign.
