Project structure
=================

This workspace is based on mjlab ``v1.6.0`` and keeps robot environments in
the main source tree. The goal is to make an environment's assets, task
configuration, and tests easy to locate without introducing separate package
layers.

Environment roadmap
-------------------

The project has three first-class environments:

* **Bird**: flapping-flight and velocity-control experiments. The migrated
  task IDs are ``Mjlab-Velocity-Bird-5DoF``,
  ``Mjlab-Velocity-Bird-Forward-3D``,
  ``Mjlab-Velocity-Bird-5DoF-Original``, and
  ``Mjlab-Velocity-Bird-5DoF-Original-NewXML``.
* **Crawler**: crawling locomotion on flat and rough terrain. The migrated
  task IDs use the ``Mjlab-Crawl-*`` prefix and include flat, robust, and
  teacher/student variants.
* **Quad Mini**: joystick velocity control on flat terrain. This is planned
  after Bird and Crawler validation.

Bird and Crawler are now part of the main source tree. Quad Mini will be added
after both migrations are working and validated.

Where files belong
------------------

Use the existing mjlab layout for each environment:

.. code-block:: text

   src/mjlab/
   ├── asset_zoo/robots/<robot_name>/
   │   ├── __init__.py
   │   └── xmls/                 # MJCF and mesh assets when needed
   └── tasks/<task_family>/
       ├── config/<robot_name>/  # scene, MDP, and runner configuration
       ├── mdp/                  # reusable task terms
       └── rl/                   # runner and policy configuration

   tests/                        # focused tests for each environment
   scripts/                      # development and evaluation utilities
   docs/                         # user and project documentation

Keep a robot's asset directory and task configuration directory aligned by
name. Put reusable behavior in the existing task-family modules; keep
environment-specific configuration in that robot's ``config`` directory.

Simple rules
------------

* Bird, crawler, and quad are ordinary mjlab environments, not separate
  extension packages.
* Each environment gets a clear task ID, a focused configuration directory,
  and targeted tests.
* Shared framework code stays in ``src/mjlab`` and is changed only when the
  behavior is genuinely reusable.
* New environment work should not modify the original ``/home/kaan/mjlab``
  workspace.
