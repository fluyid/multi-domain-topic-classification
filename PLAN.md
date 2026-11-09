# Project Plan: Multi-Domain Topic Classification (CSEBAIPNLP01, Task 3)

**Owner:** Kailash Nath (Kai) · **GitHub:** fluyid · **Started:** 9 Nov 2026
**Pair programmer:** Claude (teaching mode: assume Kai is new to programming)

---

## How to use this file

- This is the master plan. Every new chat about this project starts by reading it.
- Tick boxes `[x]` as you finish things, and update the **Progress log** at the bottom at the end of every session (2 or 3 lines is enough).
- Each phase has: **Steps**, **Concepts to learn**, **What can go wrong**, and **Done when**.
- The file lives in two places: in the NLP Project on claude.ai (so Claude sees it) and in the repo as `PLAN.md` (so it's versioned with the code). Keep them in sync: when you update one, paste the change into the other.
- Kai runs all git commands himself. Claude suggests the command and the commit message.

### Starting a new chat (copy this prompt)

> "We're continuing the NLP Task 3 project. Read the project plan doc, check the Progress log, and let's continue with the next unticked step. Teach me as if I'm new to programming."

---

## The goal in one paragraph

Build an NLP system that sorts documents into topics (multi-class classification). Train it on labelled datasets, preprocess the text (stop words, stemming or lemmatization, encoding), train several models (from Naive Bayes up to a transformer), evaluate them on held-out test data, show how performance changes as the dataset grows, and show how well it handles **different types of text**: forum posts, news articles, tweets and academic abstracts. Everything goes in a GitHub repo with the report.

### Brief requirements checklist (what the examiner will look for)

- [ ] Collect a dataset of annotated documents
- [ ] Preprocess: stop words, stemming or lemmatization, encoding
- [ ] Train supervised models (Naive Bayes, SVM, others)
- [ ] Evaluate on a separate test set (accuracy and generalization)
- [ ] Use the trained model to classify new documents (multi-class output)
- [ ] Handle different text types (news, academic, social media)
- [ ] Evaluate each step in different scenarios, small dataset first, then larger
- [ ] GitHub link with code, report and test-set performance metrics

---

## Key decisions (and why)

| Decision | Choice | Why (plain language) |
|---|---|---|
| Local Python | **Python 3.15** (released 9 Oct 2026) via `uv` | Latest stable. numpy, pandas, scipy and scikit-learn already have 3.15 wheels. |
| Transformer models | **Google Colab GPU** for training, local for small tests | PyTorch 2.15 (28 Oct 2026) supports Python 3.11 to 3.15, so torch also installs locally. But neither laptop has an NVIDIA GPU, so real fine-tuning goes on Colab. |
| Device choice | `get_device()` helper: CUDA, then Apple MPS, then CPU | One function picks the best hardware on any machine, so the same code runs on Colab, the Mac and the ROG. Only matters for PyTorch (Phase 8); scikit-learn models run on CPU anyway. |
| Local machine | Either; code is machine-agnostic | ROG (32GB) preferred for AG News and big grids. The ROG's AMD GPU falls back to CPU (AMD GPU setup on Windows isn't worth the time this week). |
| Repo visibility | **Public** (Kai's choice) | Flag: IU warns that shared solutions can raise plagiarism suspicion. Safer option is private while working, public just before submitting. Kai decides. |
| small-text library | **Optional Phase 9b**, scikit-learn backend only | Our data is already labelled, so active learning is a *simulation* (like the paper's experiment). Nice extra, not core. Latest stable is 1.4.1 (Aug 2024); test the install on day 1. |
| Stemming / lemmatization | **NLTK** first (check spaCy works on 3.15) | NLTK is pure Python, so it installs on any version. |
| Code layout | Reusable code in `src/`, exploring in `notebooks/` | Notebooks are great for looking at data, bad for code you reuse. Put it in `src/` once and import it everywhere. |
| Data in git | **Never commit datasets** | They're big, and they can be re-downloaded by a script. The repo stores code, results and figures. |
| Scope | **Both A and B**, then compare | B = each domain trained and tested on its own labels. A = shared labels + mixed and cross-domain training. Comparing them answers: "does one shared model lose accuracy versus specialists?" |

**Note on "labelling":** the datasets come already labelled by humans. Our job is not to label them; it's to **clean** them, **map** labels into a shared set (if we pick Option A), and **split** them properly.

---

## This week's schedule (coding done by Sun 15 Nov)

This is tight. If a day slips, cut in this order: Phase 9 (POS + small-text), then SetFit (8.4), then the reverse cross-domain runs (half of 7.1).

| Day | Phases | Goal by end of day |
|---|---|---|
| Mon 9 | 0 | Environment, repo, `get_device()`, first push |
| Tue 10 | 1, 2 | 20 Newsgroups explored, preprocessing variants working |
| Wed 11 | 3, 4, 5 | Baselines + learning curves + preprocessing table on 20NG |
| Thu 12 | 6 | All 4 datasets loaded; Option B results; Option A label mapping |
| Fri 13 | 7, start 8 | Cross-domain matrix; Colab notebook running |
| Sat 14 | 8 | Embeddings + DistilBERT results in the same tables |
| Sun 15 | 10, 11 (+ 9 if ahead) | `predict.py`, README, figures, final push |

---

## Planned repo structure

```
multi-domain-topic-classification/
├── PLAN.md                  # this file
├── README.md                # project summary, how to run, results table
├── pyproject.toml           # dependencies (managed by uv)
├── .gitignore               # ignores data/, .venv/, model files
├── data/                    # downloaded datasets (NOT in git)
├── src/topicclf/
│   ├── data.py              # load each dataset into one common format
│   ├── preprocess.py        # cleaning, stop words, stemming, lemmatization
│   ├── features.py          # bag of words, TF-IDF
│   ├── models.py            # model definitions
│   ├── evaluate.py          # metrics, confusion matrix, learning curves
│   └── predict.py           # classify new documents
├── experiments/             # scripts that run one experiment each
├── notebooks/               # exploration + Colab transformer notebook
├── results/                 # CSV metrics + figures (in git)
└── report/                  # report drafts
```

---

## Phase 0: Setup

**Steps**
- [x] 0.1 Pick the machine for local work (Mac M2 8GB or Windows ROG 32GB)
- [x] 0.2 Install `uv`, then install Python 3.15 with `uv python install 3.15`
- [x] 0.3 Create the repo on GitHub (`multi-domain-topic-classification`, package name `topicclf`)
- [x] 0.4 `uv init`, then add the first packages: `scikit-learn pandas numpy matplotlib nltk datasets`
- [x] 0.5 Make the folder structure above and add a `.gitignore`
- [x] 0.6 Copy this plan in as `PLAN.md`
- [x] 0.7 First commit and push (Kai types the git commands)
- [x] 0.8 Check that NLTK (spaCy not needed so far) install and import on 3.15
- [x] 0.9 Write `src/topicclf/device.py` with `get_device()`: use CUDA if available, else Apple MPS, else CPU; print which one was chosen
- [x] 0.10 Test-install `small-text` in a throwaway environment: small-text 1.4.1 installs and imports on Python 3.15.2

**Concepts to learn:** terminal basics, virtual environments (why each project gets its own "toy box" of packages), what `pyproject.toml` is, git basics (`add`, `commit`, `push`, `status`), what `.gitignore` does.

**What can go wrong**
- Installing packages globally instead of in the project's environment. Fix: always use `uv add` / `uv run`.
- Committing the `data/` folder by accident. Fix: write `.gitignore` before the first `git add`.
- **Committing the IU task PDF.** IU objects to publishing the task. Never put it in the repo.

**Done when:** `uv run python -c "import sklearn, nltk; print('ok')"` prints `ok`, and the empty structure is on GitHub.

---

## Phase 1: Explore the first dataset (20 Newsgroups)

**Steps**
- [ ] 1.1 Load 20 Newsgroups with `remove=('headers', 'footers', 'quotes')`
- [ ] 1.2 Look at real examples from 3 or 4 classes (read them yourself)
- [ ] 1.3 Count documents per class (is it balanced?)
- [ ] 1.4 Look at document lengths (words per doc), and plot a histogram
- [ ] 1.5 Find empty or near-empty documents after the headers are removed
- [ ] 1.6 Write down observations (these go in the report's Data section)
- [ ] 1.7 Write `data.py`: a function that returns a table with columns `text`, `label`, `domain`, `split`

**Concepts to learn:** datasets and labels, classes, class balance, train/test split, pandas DataFrames, histograms.

**What can go wrong**
- **Label leakage:** without `remove=...`, email headers literally contain the newsgroup name, so the model "cheats".
- Empty documents after cleaning. Decide: drop them and record how many were dropped.

**Done when:** you can describe the dataset in 5 sentences with numbers, and `data.py` loads it in the common format.

---

## Phase 2: Preprocessing pipeline

**Steps**
- [ ] 2.1 Basic cleaning: lowercase, strip URLs, emails, numbers, extra whitespace
- [ ] 2.2 Tokenization (splitting text into words)
- [ ] 2.3 Stop word removal (with an on/off switch)
- [ ] 2.4 Stemming (Porter or Snowball) as option 1
- [ ] 2.5 Lemmatization (WordNet) as option 2
- [ ] 2.6 Make each step a configurable option so we can compare variants:
  - `raw` (only lowercase)
  - `stop` (stop words removed)
  - `stop+stem`
  - `stop+lemma`
- [ ] 2.7 Cache the preprocessed text to disk so it isn't recomputed every run
- [ ] 2.8 Unit-test the functions on 3 or 4 hand-written sentences

**Concepts to learn:** tokens, stop words, stemming vs lemmatization (and why "studies" becomes "studi" vs "study"), functions with parameters, caching, simple tests.

**What can go wrong**
- Lemmatization is slow on 120k documents. Fix: caching (2.7), and it's also the environmentally friendly option, since it computes once instead of every run.
- Over-cleaning: removing numbers or symbols that carry topic meaning (like "C++" or "COVID-19"). Check examples before and after.
- Stop word removal can hurt very short texts (tweets). This is why it's a switch, not a rule.

**Done when:** one function call turns raw text into any of the 4 variants, and the tests pass.

---

## Phase 3: Turning text into numbers (encoding)

**Steps**
- [ ] 3.1 Bag of words with `CountVectorizer`
- [ ] 3.2 TF-IDF with `TfidfVectorizer`
- [ ] 3.3 Try unigrams vs unigrams + bigrams (`ngram_range`)
- [ ] 3.4 Limit vocabulary (`min_df`, `max_features`) and see the effect on size and speed
- [ ] 3.5 Look at the top TF-IDF words per class (great figure for the report)

**Concepts to learn:** vectors, vocabulary, sparse matrices (why most of the numbers are zero), TF-IDF intuition, n-grams.

**What can go wrong**
- **Fitting the vectorizer on all the data before splitting** (data leakage). Fix: always use a scikit-learn `Pipeline`.
- Calling `.toarray()` on a big sparse matrix will fill up RAM (especially on 8GB). Keep it sparse.

**Done when:** you can explain TF-IDF to a 6-year-old, and the top words per class make sense.

---

## Phase 4: Baseline models (classical machine learning)

**Steps**
- [ ] 4.1 `DummyClassifier` (most frequent class): the floor every model must beat
- [ ] 4.2 Multinomial Naive Bayes
- [ ] 4.3 Complement Naive Bayes (often better on text)
- [ ] 4.4 Logistic Regression
- [ ] 4.5 Linear SVM (`LinearSVC`)
- [ ] 4.6 Optional: Random Forest (to show a model that is usually *worse* on sparse text, and why)
- [ ] 4.7 Small hyperparameter search on a **validation** split (e.g. `alpha` for NB, `C` for SVM/LR)
- [ ] 4.8 Save each fitted pipeline with `joblib`

**Concepts to learn:** supervised learning, what each model does (intuition, not heavy maths), hyperparameters, validation set vs test set, overfitting.

**What can go wrong**
- Tuning on the test set. The test set is touched **once**, at the very end of each experiment.
- Comparing models trained with different preprocessing. Change one thing at a time.

**Done when:** a results table shows every model beating the Dummy baseline on 20 Newsgroups.

---

## Phase 5: Evaluation framework + "start small, then scale up"

**Steps**
- [ ] 5.1 `evaluate.py`: accuracy, macro-F1, per-class precision/recall, confusion matrix
- [ ] 5.2 Run every experiment with 5 random seeds and report mean ± std (like the Small-Text paper)
- [ ] 5.3 **Learning curves:** train on 500, 1k, 2k, 5k, 10k and all docs, then plot accuracy and macro-F1 against training size
- [ ] 5.4 **Preprocessing comparison:** the 4 variants from Phase 2 × the best 2 or 3 models
- [ ] 5.5 Record training time per model (speed and energy matter too)
- [ ] 5.6 Save every result as a row in `results/*.csv` (never just print it)
- [ ] 5.7 Commit results and figures

**Concepts to learn:** accuracy vs precision vs recall vs F1, macro vs micro averaging, confusion matrices, random seeds, mean and standard deviation, learning curves.

**What can go wrong**
- Reporting a single lucky run. Fix: 5 seeds.
- Too big an experiment grid that takes forever. Keep it to: 4 preprocessing variants × 3 models × 6 sizes × 5 seeds, and add more only if needed.
- Results only on screen and lost. Fix: always write to CSV.

**Done when:** you have the learning-curve figure and the preprocessing table for 20 Newsgroups.

---

## Phase 6: Add more text types

**Datasets**
| Domain | Dataset | Notes |
|---|---|---|
| Forum / social | 20 Newsgroups | already done |
| News | AG News | 4 classes, 120k train, great for scaling |
| Social media | TweetTopic (`cardiffnlp/tweet_topic_single`) | 6 classes, about 4.4k train tweets, short texts |
| Academic | Web of Science (WOS) or arXiv abstracts | research fields; confirm source and licence when we get here |

**Steps**
- [ ] 6.1 Add a loader for each dataset to `data.py` (same common format)
- [ ] 6.2 Quick exploration of each (class balance, lengths, examples)
- [ ] 6.3 Remove duplicates, especially across train and test
- [ ] 6.4 Run the Phase 5 pipeline on each domain separately (this is Option B, and it's needed either way)
- [ ] 6.5 **Option A only:** design the shared label set (e.g. Sports, Business, Sci/Tech, Politics/World) and write the mapping table from each dataset's labels; drop classes that don't map. Explain each mapping in the report.
- [ ] 6.6 Balance the classes/domains where needed (e.g. downsample AG News when comparing with tweets)

**Concepts to learn:** domains, domain shift, label mapping, deduplication, downsampling.

**What can go wrong**
- Bad label mapping (e.g. putting "pop culture" into "Business"). Every mapping needs a one-line justification.
- Comparing accuracy across datasets with different numbers of classes. Always show each dataset's Dummy baseline next to it.
- Tweets contain placeholders like `{{URL}}` and `{@user@}`. Decide how to handle them in preprocessing.

**Done when:** a results table per domain, plus (Option A) the mapping table.

---

## Phase 7: Cross-domain experiments (Option A)

**Steps**
- [ ] 7.1 Train on news, test on tweets (and the reverse)
- [ ] 7.2 Train on news + forum, test on tweets
- [ ] 7.3 Train on all domains mixed, test on each domain separately
- [ ] 7.4 Make a matrix figure: rows = train domain, columns = test domain
- [ ] 7.5 Error analysis: read 20 misclassified documents and group the reasons

**Concepts to learn:** generalization, out-of-domain evaluation, error analysis.

**What can go wrong:** expecting high numbers. The drop **is** the finding; explaining it is what earns marks.

**Done when:** the train/test domain matrix and a short written error analysis.

---

## Phase 8: Advanced models (Google Colab)

**Steps**
- [ ] 8.1 Colab notebook in the repo: clone the repo, install deps, load data with the same `data.py`
- [ ] 8.2 Sentence embeddings (e.g. a small sentence-transformers model) + Logistic Regression
- [ ] 8.3 Fine-tune DistilBERT (or similar) on each domain
- [ ] 8.4 Optional: SetFit (from the Small-Text paper) on the small datasets
- [ ] 8.5 Evaluate with the **same** splits and metrics, then add rows to the same results CSVs
- [ ] 8.6 Compare accuracy vs training time vs compute cost against the classical models

**Concepts to learn:** embeddings, pretrained models, transfer learning, fine-tuning, GPU vs CPU, epochs and batches.

**What can go wrong**
- Burning compute units: a runtime left idle still counts. Fix: disconnect when done, save checkpoints to Google Drive, start with a small subset.
- Different splits from the local runs, which makes comparisons meaningless. Fix: save split IDs to a file and load them in Colab.
- Colab sessions disconnect. Fix: save progress to Drive often.

**Done when:** the transformer models are in the same results table as the classical ones.

---

## Phase 9 (optional): The genre side quest (POS paper)

- [ ] 9.1 Turn each document into a POS-tag frequency vector (like Schulman and Barbosa)
- [ ] 9.2 Classify **text type** (news vs tweet vs academic vs forum) using only POS features
- [ ] 9.3 Check that POS features are weak for **topic**, but strong for **genre**
- [ ] 9.4 One short report section linking back to the paper

**Concepts:** parts of speech, genre vs topic, syntax vs vocabulary.

## Phase 9b (optional): Active learning with small-text

- [ ] 9b.1 Pretend AG News (or TweetTopic) is unlabelled; start with 25 labelled docs
- [ ] 9b.2 Loop: model picks the docs it's least sure about, we "reveal" their labels, retrain (scikit-learn SVM backend, CPU only)
- [ ] 9b.3 Compare query strategies: least confidence vs breaking ties vs random sampling
- [ ] 9b.4 Plot active learning curve vs the Phase 5 random learning curve: how many labels does smart picking save?

**Concepts:** active learning, query strategy, pool-based learning, uncertainty.
**What can go wrong:** dependency clashes on Python 3.15 (checked in 0.10); spending a day on it before the core is finished.

---

## Phase 10: Classify new documents (required by the brief)

- [ ] 10.1 `predict.py`: load the best saved model, take text (or a file), output the topic + confidence
- [ ] 10.2 Try it on texts you write yourself and on real headlines/tweets
- [ ] 10.3 Show a few example outputs in the README and report

**Done when:** `uv run python -m topicclf.predict "Your text here"` prints a topic.

---

## Phase 11: Results + repo polish

- [ ] 11.1 Final results tables (per model, per domain, per preprocessing variant)
- [ ] 11.2 Final figures (learning curves, confusion matrices, cross-domain matrix)
- [ ] 11.3 README: what the project does, how to install, how to run, results summary
- [ ] 11.4 Make sure a fresh clone + documented commands reproduces the results
- [ ] 11.5 Clean commit history, final push, then decide on public vs private + access for the tutor

---

## Phase 12: Report

- [ ] 12.1 Read the IU writing guidelines (page limits, structure, citation style)
- [ ] 12.2 Outline: Introduction, Related work (both papers + the textbook), Data, Method, Experiments, Results, Discussion (limitations, cross-domain, energy), Conclusion
- [ ] 12.3 Write the sections in order of easiest first: Data, Method, Results
- [ ] 12.4 Add the GitHub link and the test-set metrics
- [ ] 12.5 Proofread (no em dashes)

---

## Concepts tracker (tick when you can explain it simply)

- [ ] Virtual environment · [ ] git commit/push · [ ] DataFrame
- [ ] Tokenization · [ ] Stop words · [ ] Stemming vs lemmatization
- [ ] Bag of words · [ ] TF-IDF · [ ] Sparse matrix · [ ] n-grams
- [ ] Train / validation / test split · [ ] Data leakage · [ ] Overfitting
- [ ] Naive Bayes · [ ] Logistic Regression · [ ] SVM · [ ] Hyperparameters
- [ ] Accuracy · [ ] Precision / recall / F1 · [ ] Macro vs micro · [ ] Confusion matrix
- [ ] Random seed · [ ] Learning curve · [ ] Domain shift
- [ ] Embeddings · [ ] Transformers / fine-tuning · [ ] GPU

---

## Progress log

| Date | Session summary | Next step |
|---|---|---|
| 2026-11-09 | Read both papers, chose data tiers (20NG, AG News, TweetTopic, academic), wrote this plan | Answer the open questions, then Phase 0 |
| 2026-11-09 | Decided: both Option A and B, `get_device()` helper, coding done by Sun 15 Nov, public repo, small-text as optional 9b | Phase 0 |
| 2026-11-09 | Phase 0 done on Windows (ROG): uv 0.13.7, Python 3.15.2 pinned, requires-python >=3.12, deps added, .gitignore/.gitattributes, repo pushed to github.com/fluyid/multi-domain-topic-classification, `get_device()` + first pytest passing, small-text 1.4.1 works on 3.15 | Phase 1: explore 20 Newsgroups |

## Open questions

- Report requirements: guidelines PDF is now in the Project; read it before Phase 12
- Public now, or private until just before submission? (Kai's call, see Key decisions)
