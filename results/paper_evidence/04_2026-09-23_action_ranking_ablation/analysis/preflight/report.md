# Action-ranking preflight

Validation-only inspection; these states are never inserted into training.

- Captured dynamic-collision states: 5
- Existing demo-action margin covers: 0/5
- Full-action margin covers: 5/5
- Risky greedy actions on identical observations: seed 0 = 0/5, seed 1 = 5/5, seed 2 = 0/5

The preflight verifies label timing and scope only. It does not establish
training benefit and does not read or select on the test split.
