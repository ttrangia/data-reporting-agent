# Database reporting agent

Natural-language reporting over the **Pagila** sample database — a fictional DVD rental chain with customer, rental, payment, and film data spanning **February to August 2022**.

Ask anything you'd ask a data analyst:

- **Single questions** — "How many customers do we have?" → SQL runs, summary streams back, chart attached when one fits.
- **Multi-part asks** — "Top films and revenue trend" → splits into parallel sub-queries, composes a tight report with one section per ask.
- **Open-ended overviews** — "Quarterly review" → fans out into a 4-5 section report (headline metric, trend, top entities, breakdown, observations).
- **Follow-ups** — "Show that as a pie chart" reuses the prior result without re-querying. Memory works across turns.

The agent has **read-only** database access and a SQL guard that rejects any non-SELECT query. Chart code runs in an AST-validated sandbox.

Pick one of the example bubbles to see it in action, or type your own question.
