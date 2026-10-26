# ADR 0004: Local inference with Ollama, with CPU feasibility measured

**Status:** accepted (2026-10-25)

## Context
Patient data can't go to cloud LLMs. The target practice has no GPU. Discharge volume is low (a few to about 15 a
day), and the coordinator calls the day after discharge.

## Decision
- All inference runs through local Ollama. Patients are processed one at a time, with one model on one device.
- The milestone that adds real models (M3) also benchmarks latency per discharge on GPU and on CPU only, for the
  extraction and drafting models (7-8B, 4-bit).
- Deployment guidance comes from that measurement: a morning CPU batch if it fits, otherwise a single GPU
  workstation.

## Consequences
The claim that a practice without a GPU can run it is backed by data, not asserted.
