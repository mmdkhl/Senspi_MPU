# -*- coding: utf-8 -*-
"""Gravity analysis for the OpenSees frame model."""

import openseespy.opensees as ops


def run_gravity_analysis(ctx, show_info=False):
    """
    Run a static gravity analysis on the current OpenSees model.

    Parameters
    ----------
    ctx : dict
        Model context returned by build_model().
    show_info : bool
        Print analysis status to console.
    """
    story_node_tags = ctx["story_node_tags"]
    gravity_corner_masses_by_story = ctx.get("gravity_corner_masses_by_story", {})
    floor_weights = ctx.get("floor_weights", [])
    nStory = len(story_node_tags)

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    # Gravity loads follow the same physical mass definition used for dynamics.
    for story in range(1, nStory + 1):
        corner_masses = gravity_corner_masses_by_story.get(story, None)

        # Fallback for older saved input dictionaries.
        if corner_masses is None:
            Pcol = floor_weights[story - 1] / 4.0
            for nd in story_node_tags[story]:
                ops.load(nd, 0.0, 0.0, -Pcol, 0.0, 0.0, 0.0)
            continue

        for nd, m_corner in zip(story_node_tags[story], corner_masses):
            ops.load(nd, 0.0, 0.0, -m_corner * 9.81, 0.0, 0.0, 0.0)

    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-12, 50, 0)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')

    ok = ops.analyze(1)
    if ok != 0:
        raise RuntimeError("Static gravity analysis failed.")

    if show_info:
        print("Gravity analysis completed successfully.")

    ops.loadConst('-time', 0.0)
