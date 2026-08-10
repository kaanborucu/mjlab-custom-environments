"""Model configuration for the dedicated forward-flight bird."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH

BIRD_5DOF_FORWARD_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "bird_5dof_forward"
  / "xmls"
  / "bird_5dof_crow_forward.xml"
)
assert BIRD_5DOF_FORWARD_XML.exists()


def get_forward_spec() -> mujoco.MjSpec:
  """Load a scene-attachable copy of the forward bird model."""
  spec = mujoco.MjSpec.from_file(str(BIRD_5DOF_FORWARD_XML))
  spec.option.density = 0.0
  spec.option.viscosity = 0.0
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
  return spec
