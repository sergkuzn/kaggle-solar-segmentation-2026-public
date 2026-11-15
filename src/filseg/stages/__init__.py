"""Pipeline stages, one module per ``filseg`` command group.

Stages are orchestration: they read inputs, call the library (`filseg.data`, `model`,
`training`, `postprocess`, ...) and write the artifacts in `ExperimentPaths`. Logic that changes
results belongs in the library, where tests cover it.
"""
