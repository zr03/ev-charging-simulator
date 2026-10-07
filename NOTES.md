# Notes on the approach

Full detail is in [README.md](README.md) and [PLAN.md](PLAN.md). Assumptions are
numbered A1 to A14 in the README and tagged `ASSUMPTION A<n>` in the code where
they are applied.

## Decisions made along the way

- **Fit to published data, then check against data not used for fitting.**
  Plug-in and plug-out times come from Fig 4 of the Octopus report, target
  state of charge (SoC) from Fig 2, and how often drivers plug in from Table 2.
  Figs 5 to 9 were kept back and used only to check the results. Their bar
  heights were read exactly from the PDF's vector graphics.
- **How many times a driver plugs in each day.** A driver charges overnight
  with probability 0.74 (from Table 2). If they do, they may also plug in once
  more during the day. So on any given day a driver has 0, 1 or 2 sessions
  (26%, 48% and 26% of days), and over many days they average one a day,
  matching the spreadsheet's plug-in frequency.
- **Overnight and daytime sessions have different timings.** Fig 4 is split
  into sessions that run through 03:00 (overnight) and those that don't, and
  each kind is sampled from its own pattern.
- **Daily driving varies from day to day, and is right-skewed.** Each day's SoC
  use is drawn from a gamma distribution around the driver's average. Studies
  of daily mileage find it right-skewed (most days are short, a few are long),
  and the gamma has that shape.
- **Heavy drivers can't go many nights without charging.** In the first
  version, plug-in days were random regardless of mileage, so heavy drivers
  sometimes skipped several nights and ran down to the 5% minimum. That cut
  their delivered energy by 12 to 27% below the spreadsheet. Now each driver
  has a maximum number of nights between charges: how many days of their
  average driving fit before SoC would drop below 20%. Their nightly plug-in
  probability is raised to match, so a heavy driver plugs in every night.
  Energy delivered is now within 6% of the spreadsheet for every archetype
  that drives.
- **Charging is computed directly, not stepped through time.** Power is flat
  up to 80% SoC and then tapers, which has an exact formula. That makes 1,000
  drivers over 200 days run in seconds.
- **Each simulated day is independent,** so the 24 h window wraps around: a
  session still running at the window start continues at its left edge.
  Population plots run noon to noon so the overnight block stays whole;
  single-driver plots run 06:00 to 06:00 so a day reads in order (morning
  unplug, any daytime session, then the evening plug-in).

## Key assumptions (full list in the README)

- Spreadsheet values are averages. Annual mileage varies between drivers
  (lognormal, standard deviation 40% of the mean); daily use varies between
  days (gamma, as above).
- The Octopus charge limits in Fig 2 apply to every driver.
- **Scheduled charging** is read as a driver who plugs in late and unplugs
  late (about 22:00 to 09:00). If it really means a timer on a car plugged in
  earlier, its flexibility is understated.
- **Always plugged-in** has no energy demand, because its spreadsheet row
  can't be both always connected and driving 9,435 miles a year.
- Charging is unmanaged: it starts as soon as the car is plugged in and runs
  at the charger's rate (tapering above 80%) until the target SoC is reached
  or the car is unplugged. This
  is the baseline that flexibility is measured against.

## Where time went, and what was left out

- **Most time:** the behaviour model, fitting it and checking it. This is
  what decides whether the flexibility numbers can be trusted.
- **Also prioritised:** the outputs that matter for flexibility (charger power
  connected, energy still needed, time until unplugging) and tests for each
  module (25, including a check that delivered energy matches the spreadsheet).
- **Left out as less important for flexibility:** SoC while the car is
  unplugged (only SoC at plug-in and while plugged in is modelled); smart and
  bump charging; links between consecutive days; weather and seasonality;
  public charging; uncertainty in the model's own parameters.
- **Known gaps:** simulated plug-in SoC is higher than the report's (median 63%
  against 52%); about 7% of Octopus-average sessions still start at the 5%
  minimum; plug duration (Fig 6) fits less well since heavy drivers were made
  to charge every night.

## Designing for the end use

- **Built around the questions a flexibility planner asks:** how many cars are
  plugged in, how much energy they still need, and how long until they leave.
  These are main outputs, not by-products. The overview also shows the
  combined SoC of plugged-in cars (kWh stored over kWh capacity), a measure
  of how much headroom the connected fleet has.
- **Shows how much demand can be moved, not just how much there is.** The
  population overview plots unmanaged demand next to the same sessions
  charged as late as possible (each car starting at the last moment that
  still finishes by plug-out), and the energy that could be shifted later at
  each time: about 9 MWh for 1,000 EVs at 01:30. Both are fleet totals,
  since grid flexibility is managed in aggregate.
- **Two kinds of spread:** across drivers on one day, and across days for the
  same fleet. The second is the range a planner should expect day to day.
- **Easy to change:** every setting lives in one validated config, a new
  standard archetype is a new spreadsheet row, not new code, and the app's
  population mix can be set to match the chargers in a particular area.
- **Fast enough to explore:** calculations run on whole arrays of drivers at
  once, so the app updates as sliders move. A command-line script produces the
  same outputs in batch.
- **Reproducible:** seeded random numbers, a locked environment (`uv`), and one
  command each for the app, a population run, the validation and the tests.
