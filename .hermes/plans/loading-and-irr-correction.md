# PFTrackr loading states and IRR correction

## Problem and goal

PFTrackr gives no visible feedback while price refreshes and report calculations run. In addition, position IRR is negative for profitable foreign-currency holdings because the terminal value is not converted into the portfolio currency before IRR calculation.

## Scope

- Add a reusable loading state to the existing HTML form actions for:
  - Home portfolio Refresh.
  - Utilities: Tax Reporting submit/export.
  - Portfolio: Monthly Summary submit/export, NAV View NAV, and Force NAV Recalculation.
- Disable the initiating submit control while the request is in progress and show a spinner with an action-specific label.
- Convert terminal position value into the portfolio currency before calculating position IRR.
- Add regression and template contract tests.

## Non-goals

- No backend API or database schema changes.
- No changes to trade, price, dividend, or financial records.
- No redesign of report pages or asynchronous job architecture.
- No live deployment.

## Success criteria

1. Every scoped form action provides immediate visual loading feedback and prevents duplicate submission.
2. Loading controls use accessible status text and restore normal behaviour on a new page load/error response.
3. A foreign-currency position with positive AUD total return cannot receive a falsely negative IRR solely because its terminal value is unconverted.
4. Existing IRR and application tests pass.

## Constraints and dependencies

- Existing Flask/Jinja2 templates, Bootstrap 4 and native browser JavaScript; avoid a new frontend dependency.
- Preserve current form POST/download behaviour.
- Existing IRR uses date-based cash flows and portfolio-currency historical CF values.

## Current-state findings

- `app/templates/base.jinja2` has no shared loading-state script or spinner CSS.
- Scoped forms are in `home.jinja2`, `monthly.jinja2`, `tax.jinja2`, and `nav.jinja2`.
- `User.info_date()` passes `curr_df` to `calc_IRR()` but only passes `Date`, `Ticker`, and `Close`; `calc_IRR()` calculates terminal CF as `CumQuan * Close`, omitting FX.
- `calculate_gains()` already applies `Fx` to historical cash flows and the main table applies current `Fx` to `CurrVal`.
- Live read-only API evidence on 16 Sep 2026: NTR and CF have positive current value/total gain but negative IRR; both are CAD holdings.

## Technical plan

1. Add a small shared loading helper in `base.jinja2` and mark scoped forms/buttons with data attributes and action labels.
2. Extend `calc_IRR()` current-price input to carry FX and multiply terminal value by FX, defaulting to 1 only where FX is absent/invalid under the existing currency rules.
3. Add a focused IRR regression test for a foreign-currency terminal value and template tests for loading markers/labels.
4. Run focused tests, full pytest suite, and inspect final diff/status.

## Test matrix

- `pytest tests/test_irr.py -q`
- Regression test for `User.calc_IRR()` with non-1 FX and positive converted return.
- Template/source contract tests for all scoped forms and the shared loading script.
- `pytest -q`

## Risks and approval gates

- Calculation change affects displayed performance metrics but not stored financial records. Verify with deterministic cash-flow tests before considering delivery.
- Live deployment/rebuild is explicitly out of scope and requires Gerard's separate approval.
- No irreversible operation or external-state write is planned.
