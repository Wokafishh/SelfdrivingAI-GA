"""
Self-Supervised / Semi-Supervised Learning (SSL) — base script.

Workflow (as confirmed):
  1. Small labeled batch  -> student predicts -> compare to true label -> supervised loss (= BC).
  2. Large unlabeled batch (+noise) -> student predicts on noisy input;
     teacher (EMA copy of student, no grad) predicts on clean input ->
     compare -> consistency loss.
  3. loss = supervised + ramp * consistency -> backward -> STUDENT weights update only.
  4. teacher = decay * teacher + (1 - decay) * student   (no gradient, pure averaging)
"""
