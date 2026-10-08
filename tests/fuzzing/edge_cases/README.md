# Minimized Fuzz Regression Corpus

This directory keeps one representative input for each historical
`(hull, dimension, strategy)` failure class: six hull implementations across
dimensions 2–6, for 30 JSON cases in total.

The original 1,096 files all recorded the same fixed exception-handling defect:
`cdd.Error` was not a `BaseException`, so attempting to catch it raised a raw
`TypeError`. Retaining every random reproduction added test count and repository
weight without adding a distinct contract.

The regression test ignores the stale `exception` and `message` fields. Each
representative must either return constraints or fail through a sanctioned
validation, degeneracy, or convergence exception. New fuzz findings belong here
only when they represent a distinct hull, input shape, strategy, or failure mode.
