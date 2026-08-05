"""core/skills -- the agent's higher-level skill orchestration layer.

A **skill** (in Locus) lives only in core and is a higher-level unit of agent
behavior that COMPOSES one or more tools (built-in tools in :mod:`core.tools`,
or ad-hoc endpoint tools invoked over the WS link). The provider never sees
skills directly; skills orchestrate tools -- deciding which skill to run,
which tools to expose to the provider, and how to react to tool results.

This package is intentionally a placeholder for now: the skill layer is built
in Step 7 (the agent loop), alongside the in-process event bus. Step 6 only
landed the **tool** layer (:mod:`core.tools` + the ``tool_defs`` memory table
seeded from :mod:`core.tools.toml`) on top of which skills will compose.
"""
