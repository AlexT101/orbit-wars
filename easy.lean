example (P Q : Prop) : (P ∨ Q) ∧ ¬P → Q := by
  intro h
  have hpq := h.left
  have hnp := h.right
  rcases hpq with x|y
  cases hnp x
  exact y
