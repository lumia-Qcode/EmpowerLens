# D26: App and model-output test cases

Rows from the *Test cases* tab of `D3/tasks_D3_2026-11-15.xlsx` whose *Task* is D26: A-01 to A-19, D-01, D-02, D-03,
D-05 and D-06. Evidence is in `docs/test_evidence/<ID>.png|txt`.

**Setup.** The demo app was `empowerlens-demo` at commit `570b82a` (D25), serving the D12 model from `models/d12`:
MentalRoBERTa, version `mentalroberta-d12-s42-20261120`, architecture `classifier_pair`, on CPU. The backend ran with
uvicorn on port 8000 and the frontend with Vite on port 5173. The browser steps were driven with Playwright (Chromium),
and the API, database and load checks used curl and Python.

**Data isolation.** `EMPOWERLENS_DB` pointed the backend at a throwaway SQLite file in the session scratchpad. That file
was seeded with `backend/seed_demo.py --reset` (hand-written demo reflections and demo accounts), plus one test trainer
account (`trainerb@test.local`) and three `TEST …` cohorts. No real survey responses were used. The app's default database
(`backend/empowerlens.db`) did not exist before or after the run. `git status` in `empowerlens-demo` was clean before
and after, and no file there changed. The throwaway database was deleted at the end.

**Summary: 19 Pass, 5 Fail (A-04, A-06, A-08, A-16, D-02), 0 Not run.** None of the failures were fixed in this run.

| ID | What is checked | Actual | Pass / Fail | Evidence file | Tested by |
|---|---|---|---|---|---|
| A-01 | Prompts shown in order; each answer stored against its own prompt | The prompt set has only 1 prompt (the single reflection question). It appeared once, and the answer "answer to prompt 1" was stored as reflection 34 with `prompt_id` 1. The "different answer under each prompt" step can't be exercised with a single prompt. | Pass | A-01.png, A-01.txt | Claude agent, 2026-11-20 |
| A-02 | Consent recorded before anything is stored | Before consent, cohort 6 had 0 participant rows. After consent, P-2F62AB50 had `consented_at` 12:35:17. The reflection was stored at `submitted_at` 12:35:45, which is later. Both times are stored to the second only. | Pass | A-02.txt, A-02_1_consent.png, A-02_2_received.png | Claude agent, 2026-11-20 |
| A-03 | Declining consent stores nothing | The screen said "Nothing was stored". No request was sent. `POST consent {consent:false}` returned `{"stored":false}`. The participant and reflection counts stayed at 34/34. | Pass | A-03.png, A-03.txt | Claude agent, 2026-11-20 |
| A-04 | A blank answer is accepted and recorded as unanswered | There is no "prompt 2" (single prompt). With the answer left empty, no "required" message appears, but **Submit is disabled**, so a blank answer can't be submitted from the UI. Sent through the API, the blank answer is accepted and stored as `text=""` with **no unanswered flag**: the `reflection` table has no such column. Abandoning after consent also leaves a consented participant with no reflection. Not fixed. | Fail | A-04.png, A-04.txt | Claude agent, 2026-11-20 |
| A-05 | The participant is never shown a classification | After submitting "Everyone will laugh at my pitch, I always fail", only "Responses received" was shown. The page received 4 API responses (cohorts, prompts, `{"stored":true,"participant_token":…}`, `{"received":true}`), and none held a label, score, type or sentence. | Pass | A-05.png, A-05.txt | Claude agent, 2026-11-20 |
| A-06 | A finished link accepts nothing more | Resubmitting with the finished token returned **409 "This exercise has already been submitted"**, and no new row was stored. But **re-opening the link (`/participant`) shows a fresh consent screen**, not "exercise closed". The link is per workshop and the token lives only in page state, so the same person can submit again under a new token. A closed cohort does refuse consent (400 "This exercise is not open"). Not fixed. | Fail | A-06.png, A-06.txt | Claude agent, 2026-11-20 |
| A-07 | The cohort report needs a signed-in trainer | After Sign out, pasting the report URL (`/api/trainer/cohorts/1/report`) returned 401 `{"detail":"Sign in required"}` with no report content, and `/trainer` shows "Sign in as trainer". There is no per-cohort page URL, so the 401 isn't a redirect to sign-in. Bug found: after sign-out, the old bearer token still returns the full report (200), because the server session is never revoked. | Pass | A-07.png, A-07_trainer_page.png, A-07.txt | Claude agent, 2026-11-20 |
| A-08 | A trainer cannot open another trainer's cohort, and the attempt is logged | Trainer A opening trainer B's cohort 4 (report and export) got **403 "You do not own this cohort"**. But **no `refused` row was written to `access_log`**. In `_report`, the log insert and the `HTTPException` are in the same `with db.connect()` block, so the exception skips the commit. Not fixed. | Fail | A-08.txt | Claude agent, 2026-11-20 |
| A-09 | A successful report view is logged | Trainer A's own cohort 1 (18 submissions, threshold 3) showed the report: Personalization 3 · 17%, plus 1 example sentence. The access log got `log_id 4`: user 1, cohort 1, `view_report`, `allowed`, 12:39:52. | Pass | A-09.png, A-09.txt | Claude agent, 2026-11-20 |
| A-10 | No report below the reporting threshold | Cohort 5 has threshold 5 and 2 submissions. The page said "Not enough submissions… 3 more needed", and 3 = 5 − 2. No report was shown and Export was disabled. | Pass | A-10.png, A-10.txt | Claude agent, 2026-11-20 |
| A-11 | A rare pattern is suppressed | In cohort 1 (threshold 3), the per-reflection predictions gave Labeling 2, All-or-nothing 1 and Should statements 1. None of those names appear in the report page, its JSON or the example sentences (0 hits). Only the count is shown: "3 rarer type(s) are hidden". | Pass | A-11.png, A-11.txt | Claude agent, 2026-11-20 |
| A-12 | The report's counts match the individual predictions | The hand count from the researcher view's 18 predictions gave Personalization 3, and the report shows 3 (17%). The other 3 types are suppressed (A-11). Reflection 4 is distorted with no type (see D-02). | Pass | A-12.txt | Claude agent, 2026-11-20 |
| A-13 | The report holds no identifier or traceable reflection | Searched the page DOM, the page source, and the report JSON for all 36 participant tokens, the `P-…` pattern, account names and emails, reflection ID keys and submission timestamps. There were no participant identifiers. The only hits were the signed-in trainer's own name in the header and the cohort's `created_at`. | Pass | A-13.txt | Claude agent, 2026-11-20 |
| A-14 | The exported report holds the same aggregated content as the screen | Exported the A-12 report as PDF, XLSX, DOCX and JSON. Each has the same types and counts as the screen (Personalization 3/17%, 3 hidden, 18 submissions, 28%). The exports also include cohort metadata (status, created/closed dates; in JSON, `cohort_id` and `trainer_id`) but nothing per participant. The A-13 search found nothing. | Pass | A-14.txt | Claude agent, 2026-11-20 |
| A-15 | A trainer cannot reach the researcher view | In the browser, signing in to `/researcher` as trainer B is refused ("This account has the trainer role, not researcher"). This check is client-side and writes no log row. Called with trainer B's token, the researcher endpoints (cohorts, reflections, export, analyze) return 403, and each call logs `researcher_surface` / `refused` (log 15–18). | Pass | A-15.png, A-15.txt | Claude agent, 2026-11-20 |
| A-16 | The researcher view shows the prediction, and each access is logged | The researcher view shows the text, the highlighted selected sentence and the predicted labels with scores (All-or-nothing 0.69, Should statements 0.63, p = 0.978). The log row is `open_researcher_review` with user 2 and the time, but it is **per cohort (`cohort_id`), with no reflection ID**: `access_log` has no `reflection_id` column, and opening the next reflection logs nothing. Not fixed. | Fail | A-16.png, A-16.txt | Claude agent, 2026-11-20 |
| A-17 | The annotation export keeps the text exactly as written | Submitted 3 reflections through the UI, each with a curly apostrophe or quote, an em-dash, an emoji and a line break. The XLSX (default), JSON and DOCX exports are identical to the typed text under `==`. The JSON is valid UTF-8 with no mojibake. Side findings: the PDF drops the emoji, and empty text is written as "–" in XLSX, DOCX and PDF. | Pass | A-17.txt | Claude agent, 2026-11-20 |
| A-18 | The participant flow works on a phone browser | At a 375×812 viewport, ran consent, one answer and submit. There was no sideways scroll on any screen (`scrollWidth` 360 ≤ 375), every button was in view and at least 48 px tall, and the submission was stored (reflection 40). This was an emulated viewport in desktop Chromium, not a real phone. | Pass | A-18.png, A-18_1_consent.png, A-18_2_question.png, A-18_3_received.png, A-18.txt | Claude agent, 2026-11-20 |
| A-19 | 50 people can submit at the same moment | A Python script started 50 threads together. First, 50 simultaneous consents: 50/50 returned 200 in 1.85 s. Then 50 simultaneous submits, each with its own token and a distinct answer: 50/50 returned 200 in 1.18 s. The database gained exactly 50 rows, none stored under the wrong participant and no duplicates. Run on one machine over loopback. | Pass | A-19.txt | Claude agent, 2026-11-20 |
| D-01 | Inference returns all three outputs | Ran `empowerlens.inference.Predictor(models, "auto")`, the same call the app makes, on `val.csv` rows 0–4. Every row has `is_distorted`, 0–2 `labels`, and `sentences` with a `selected` flag. Results: row 1 is distorted with Personalization and Should statements and 3 selected sentences; rows 0, 2, 3 and 4 are not distorted. | Pass | D-01.txt | Claude agent, 2026-11-20 |
| D-02 | The types agree with the yes/no decision | On the D-01 rows: the 1 distorted row has 2 types and the 4 non-distorted rows have 0. The binary and type models are separate. When the binary model says no, the app drops any types the type model fires (val rows 2–4), and it caps the types at 2. **But a "distorted" decision with no type passing its threshold is not handled.** Seed cohort-1 reflection 4 (p = 0.985) comes back `is_distorted=true, labels=[]`, which breaks "distorted gives 1 or 2 types" (1 of 23 texts checked). Not fixed. | Fail | D-02.txt | Claude agent, 2026-11-20 |
| D-03 | The selected sentences really come from the text | For every D-01 row, every sentence appears word for word, and `text[start:end]` equals the sentence. The distorted row has 3 selected sentences. The 5 distorted cohort-1 texts, checked as extra cases, each have at least 1 selected sentence, all verbatim. | Pass | D-03.txt | Claude agent, 2026-11-20 |
| D-05 | The app runs the chosen model | The researcher view shows MentalRoBERTa, version `mentalroberta-d12-s42-20261120`, which matches `docs/model_selection.md`. `val.csv` row 1 (Id 4522) was run through the app (`/api/researcher/analyze`) and through `src.app_parity`. Both gave p = 0.9840, the same 10 type probabilities at 3 dp, and the same types (Personalization, Should statements). | Pass | D-05.png, D-05.txt | Claude agent, 2026-11-20 |
| D-06 | The same input always gives the same output | Ran D-01 twice in two separate processes. The output JSON is byte-identical (same sha256). The models are in `.eval()` and run under `torch.no_grad()`. | Pass | D-06.txt | Claude agent, 2026-11-20 |

## Bugs found (not fixed)

1. **Refused cohort access is never logged** (A-08). In `backend/app.py` `_report`, `db.log_access(..., "refused")` and the
   `raise HTTPException(403)` sit in the same `with db.connect()` block. The exception skips `conn.commit()`, so the row
   is rolled back. This affects both `/report` and `/export`.
2. **Sign out does not end the session** (A-07). The Sign out button only clears `sessionStorage`. There is no logout
   endpoint and `session` rows never expire, so an old bearer token still returns cohort reports.
3. **Researcher review is logged per cohort, not per reflection** (A-16). `access_log` has no `reflection_id`. One
   `open_researcher_review` row is written when a cohort's reflections load, and browsing individual reflections logs
   nothing.
4. **Blank answers are not supported as "unanswered"** (A-04). The UI disables Submit when the answer is empty. Through
   the API, a blank is stored as `""` with no unanswered flag. A participant who consents and then leaves remains a
   consented row with no reflection.
5. **A finished participant can start again from the same link** (A-06). The link is per workshop and the token lives only
   in page state, so re-opening the link, or clicking "Start a new exercise", allows another submission under a new token.
   This looks like a design choice, but it doesn't meet the test.
6. **Distorted with no type** (D-02). When P(distorted) passes the binary threshold but no type passes its own threshold,
   the app reports "distorted" with no types. The trainer report then counts the text in "share with distorted
   reasoning" but under no type.
7. **Exports alter some text** (A-17, minor). XLSX, DOCX and PDF write an empty reflection as "–" (`exports._cell`), and
   the PDF drops emoji. The XLSX and JSON annotation exports are otherwise exact.
8. **Minor observations.**
   - The researcher-role check at sign-in is client-side only. `/api/auth/login` issues a token to a trainer on the
     researcher page and logs nothing; the API itself does refuse and log (A-15).
   - The participant page shows the Trainer and Researcher navigation links.
   - The sign-in pages print the demo credentials.
