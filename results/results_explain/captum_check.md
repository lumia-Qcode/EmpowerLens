# D22 — Captum cross-check (val)

> **Internal check only.** Captum scores are never shown to a trainer.

## Setup

- Model: D12 winner, MentalRoBERTa multilabel, seed 42 (D12 recipe: lr 3e-5, 5 epochs, weight decay 0.1, max length 512, batch 16); checkpoint `/kaggle/temp/ckpt_d22/multilabel_mental-roberta-base_42`.
- Method: LayerIntegratedGradients over `model.roberta.embeddings` (as `src/explain.py`), 50 steps, baseline = `<pad>` tokens with `<s>`/`</s>` kept. Token score = attribution summed over the embedding.
- Label explained: the top-scoring distortion of each row (argmax of the 10 logits).
- Rows: val rows with a distortion (`y_bin == 1`). Test is not used.
- Sentences: the selector's own sentences (`results/results_span_selector/predictions_val_42.csv`), aligned by `Id_Number` + `sent_idx` using character offsets from `results/results_sentence_label/sentence_labels_val.csv`. Sentence score = signed sum of its tokens.

## 1. Agreement with the D03 selector (seed 42)

Agreement: **57 of 157 rows (36.3%)** — the selector's top-scoring sentence is also the sentence with the highest total Captum score.

For reference, picking a sentence at random would agree in 17.2% of these rows (mean of 1 / number of sentences). Rows are those in both files; the selector file excludes the T12 rows whose quote could not be located.

## 2. Cue or topic? (20 rows with absolute words)

Selection rule: val rows with `y_bin == 1` whose text contains at least one of `never`, `always`, `everyone`, `nobody`, `everything`, `nothing`, `completely`, `all`, `every`, `totally` (whole word, any case); eligible rows sorted by `Id_Number`, 20 drawn without replacement with `numpy.random.default_rng(42)`, shown sorted by `Id_Number`. Top 5 words = highest signed Captum score towards the attributed label.

Verdict: **cue** = top words are the distortion's wording (absolute words, "should", labels, ...); **topic** = top words are the subject (people, jobs, bodies, ...); **mixed** = both.

| Id_Number | Absolute words | Attributed label | Top 5 words (Captum) | Verdict |
|---|---|---|---|---|
| 131 | never, always, every | mind_reading | scared / criticizing / to / always / supportive | mixed |
| 165 | everyone, nothing, all | mind_reading | predator / prey / myself / isolate / by | topic |
| 219 | never, everything | emotional_reasoning | flashbacks / flashbacks / flashbacks / to / everything | mixed |
| 334 | never | should_statements | humiliated / masochism / I / masochism / humiliated | topic |
| 697 | always, everything | mind_reading | don’t / insensitive / When / i / mother | topic |
| 920 | all | mind_reading | like / like / me / angry / the | mixed |
| 1053 | never, everything, nothing, all, every | mind_reading | criticizing / He / the / acting / critiques | topic |
| 1066 | never, every | emotional_reasoning | I’m / suicidal / urges / of / I | topic |
| 1172 | always | mind_reading | think / offensive / offend / because / me | mixed |
| 1582 | all | personalization | aromantic / what / wasted / if / sick | topic |
| 1696 | never, always, all | all_or_nothing | retarded / I’m / talk / succeeding / mumble | mixed |
| 1736 | always | mind_reading | doesn’t / takers / care / care / relationship | mixed |
| 1756 | never, all | mental_filter | don’t / I’m / couldn’t / can’t / out | cue |
| 1802 | never, always, every | personalization | infuriated / Sometimes / will / Dear / narcissist | topic |
| 1904 | all | should_statements | From / Germany / wouldn’t / voluntary / myself | topic |
| 1932 | never, always, nobody, every | mind_reading | thought / ugly / lame / job / fustration | mixed |
| 2417 | everything | emotional_reasoning | Prozac / haven’t / 40mg / very / the | topic |
| 2532 | every | magnification | can’t / embarrassed / myself / can’t / I | cue |
| 4522 | always, all | personalization | But / She’s / She / She / She | topic |
| 4549 | always, all | mind_reading | stonewalls / irrational / mistakes / ready / being | topic |

Verdict counts: cue 2, topic 11, mixed 7, blank 0.

## Summary

<!-- summary:start -->
**Verdicts were assigned by Claude (AI assistant) at Lumia's request because of time limits, not by an
independent human reviewer.** Each row was judged by reading the full val text, its gold distortion and quoted
distorted part, and the five words with the highest Captum score.

**Agreement.** The selector's top sentence is also the sentence with the highest total Captum score in 57 of 157
distorted val rows (36.3%), about twice the 17.2% expected by picking a sentence at random. The two methods point to
the same place more often than chance, but in most rows they do not.

**Cue or topic?** Of the 20 rows with absolute words: cue 2, mixed 7,
topic 11. The absolute words themselves (never, always, everything, ...) reach the top 5 in
only 2 of the 20 rows (131 'always', 219 'everything'). Where a cue does show up, it is more often mind-reading or
labelling wording ('feel like', 'think', 'retarded', 'doesn't care') or helpless negation ('can't', 'couldn't')
than an absolute. In most rows the strongest words name the subject: people (mother, She, He), conditions
(masochism, suicidal, aromantic), medication (Prozac, 40mg), or even the post's origin (From Germany). So the
model leans more on topic than on distortion wording, which fits the low clustering ARI (0.003-0.016) and the
gap between the full-text and gold-span results in D17.

**Limits.** One seed (42), one attributed label per row (the top-scoring one, which is one of the row's gold
labels in only 6 of these 20 rows, and 51 of all 158 attributed rows), and a sample restricted to texts containing absolute words. This is an internal
check only; Captum scores are never shown to a trainer.

Per-row reasons:
- 131: **mixed** - 'always' (absolute) is in the top 5, but next to family-topic words (criticizing, supportive).
- 165: **topic** - predator / prey / isolate are the content of the questions asked, not the self-label 'filth'.
- 219: **mixed** - 'flashbacks' (topic) dominates, but 'everything' from the distorted part is also there.
- 334: **topic** - humiliated / masochism: the subject of the post.
- 697: **topic** - mother / insensitive: who the post is about.
- 920: **mixed** - 'like' x2 comes from 'I feel like he hates me' (mind-reading phrase) plus 'angry' (topic).
- 1053: **topic** - criticizing / critiques / He: the husband's behaviour.
- 1066: **topic** - suicidal / urges: the subject, not the 'I'm so scared ... turning into reality' prediction.
- 1172: **mixed** - 'think' from 'they think I am going to...' (mind-reading cue) plus offensive / offend (topic).
- 1582: **topic** - aromantic / sick: identity and illness words; 'wasted' is close to the distorted part but the rest is topic.
- 1696: **mixed** - 'retarded' is the self-label ('I felt like I was retarded'), with talk / mumble (topic).
- 1736: **mixed** - 'doesn't care' / 'takers' (labelling and mind-reading wording) plus 'relationship' (topic).
- 1756: **cue** - don't / couldn't / can't: helpless-negation wording rather than the medical subject.
- 1802: **topic** - infuriated / narcissist / Dear: the mother and the letter form.
- 1904: **topic** - From / Germany / voluntary: location and setting artefacts.
- 1932: **mixed** - 'thought' (mind-reading) and lame / ugly (labelling the pill) next to job / frustration (topic).
- 2417: **topic** - Prozac / 40mg: medication details.
- 2532: **cue** - can't x2 / embarrassed / myself: self-evaluative wording, not the trauma subject.
- 4522: **topic** - She x4: focus on the therapist, not the 'I have to' comparison.
- 4549: **topic** - stonewalls / irrational / mistakes: the conflict's content.
<!-- summary:end -->

## Files

`attributions_val.csv` (per sentence), `attributions_val_words.csv` (per word), `attributions_val_rows.csv` (per row: label, probability, top words), `attributions_val_config.json`, `captum_check_rows.csv` (the 20-row table). Fill the verdicts in that CSV or in the table above, and the summary between the markers; re-running `report` keeps both (the CSV wins if they differ).
