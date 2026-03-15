Changelog
=========

Unreleased
----------

Changed
-------

- ``MaxVolAlgorithm`` now emits a warning instead of raising a ``RuntimeError`` when
  the MaxVol algorithm does not converge within ``maxiter`` iterations.
- Changed the default of ``LossSetting.stress_times_volume`` from ``False`` to ``True``.
  This affects how stress residuals are weighted in the loss (stress contributions are
  scaled by the configuration volume), so training results may differ for existing
  configs unless ``stress_times_volume`` is set explicitly.
