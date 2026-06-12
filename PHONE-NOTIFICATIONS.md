# Phone Study Notifications

The Windows scheduler publishes DSA and SQL reminders to this private ntfy topic:

`utsav-study-7e2bc39fd6af49f0b7871ad8e6a018b0fa81b4fd138ab7d8`

Subscription URL:

https://ntfy.sh/utsav-study-7e2bc39fd6af49f0b7871ad8e6a018b0fa81b4fd138ab7d8

## Phone Setup

1. Install the `ntfy` app from Google Play or the iOS App Store.
2. Tap the plus button to subscribe.
3. Enter the full topic name shown above.
4. Allow notifications and disable battery optimization for ntfy if Android delays alerts.

## Schedule

- DSA: every day at 12:00 PM
- SQL: every day at 4:00 PM
- Study cycle: Day 1 is June 12, 2026; the plan repeats after Day 30.

The Windows computer must be on, connected to the internet, and signed in when
the scheduled task runs. Windows Task Scheduler is configured to run a missed
notification as soon as possible after the computer becomes available.

The topic is intentionally long and unguessable. Anyone who knows it can
subscribe, so do not post it publicly.
