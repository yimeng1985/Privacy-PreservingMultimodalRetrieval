"""Model: Image local privacy protection via reconstruction sensitivity
analysis and VLM-guided semantic privacy judgment.

Pipeline:
  Phase 1: Shadow reconstructor training
  Phase 2: Block-level occlusion sensitivity analysis
  Phase 3: Candidate region screening
  Phase 4: VLM semantic privacy judgment
  Phase 5: Score fusion (reconstruction sensitivity × semantic privacy)
  Phase 6: Local adaptive protection
"""
