---
layout: post
title: Measuring Anti-Money Laundering Program Effectiveness
image: "/posts/aml-transaction-monitoring-title-img.png"
tags: [EDA, Feature Engineering, Data Visualization, DuckDB, XGBoost, scikit-learn, Optuna, SQL, Python, Tableau]
---

In this project I build a complete **anti-money laundering transaction monitoring pipeline** on 9.5 million banking transactions, and then measure how effective it actually is.

I start by completing exploratory data analysis to build the risk profile of the dataset. Next, I build a **rule-based system** which tries to capture behavior associated with the different fraud typologies which acts as a baseline. Given the imbalanced nature of the dataset and goal of the program (to capture criminal behaviour) I also designed metrics and relevant KPIs focused on the alerts produced and time spent investigating. After establishing this baseline I added in a **machine learning model** and measured its impact in terms of time or effort saved and produced an interactive dashboard showcasing my findings. 

This project taught me how to connect data analysis with real-world outcomes, highlighting the difference between model quality or performance and effectiveness when real-world costs and variables are considered. Additionally, this project provided an opportunity to further develop my data storytelling and visualizations skills, by creating compelling dashboards which supported and enhanced my findings. 

[![AML Program Effectiveness dashboard: alert cost by rule, detection versus alert volume for three operating models, and typology coverage gaps](/img/posts/aml-dashboard-full.png)](https://public.tableau.com/app/profile/christine.dowling/viz/AMLProgramEffectiveness/AMLProgramEffectiveness)

*The finished dashboard. **[Open the interactive version on Tableau Public →](https://public.tableau.com/app/profile/christine.dowling/viz/AMLProgramEffectiveness/AMLProgramEffectiveness)** — the capacity and operating-model controls are live.*

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Why Program Effectiveness Is Not Model Accuracy](#effectiveness-overview)
- [03. Building Features From Almost Nothing](#feature-engineering)
- [04. The Rule Engine Baseline](#rule-engine)
- [05. The Machine Learning Model](#the-model)
- [06. Rules vs Model vs Hybrid](#hybrid-comparison)
- [07. Honest Limitations](#limitations)
- [08. Results Comparison](#results-comparison)
- [09. Growth & Next Steps](#growth-next-steps)

___

# 00. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

**Money laundering** is the process of moving criminally obtained money through the banking system until it appears legitimate. In Canada, banks and financial institutions are legally required to watch for it, and the system they use is called **transaction monitoring**: a set of rules that scan every transaction and raise an **alert** whenever something looks suspicious. A human analyst then investigates each alert and decides whether to escalate it.

One of the major problems in this areas is volume. Real transaction monitoring systems are notorious for creating **false positives**; that is, creating an alert or flagging an account where no illegal behaviour exists. This is a direct result of the imbalanced nature of the data (i.e. most transactions are legitimate) and reflected in industry stats, which claim somewhere between 95% and 99% of AML alerts are closed with no further action (NTD: add citation). But the imbalance is not restricted to the data, it's also present in the economics. A missed laundering network could result in hefty fines and a loss of money, however every alert (regardless if its fraudulent or not) has to be investigated and documented, costing analyst time. 

This creates the question the project is really about: *given finite resources (analysts, time to investigate, etc.), how do I catch the most money laundering?* To answer this question, we need to look at more than just model performance, and instead focus on the whole program, taking into account the real cost of resources and trade-offs.

### Actions <a name="overview-actions"></a>

I built an end-to-end pipeline that:

* Loaded and queried the SAML-D synthetic dataset using **DuckDB**
* Profiled the data to establish which behaviours actually carry risk signal
* Engineered **behavioural features** to measure how an account behaves over time, beyond payment metrics (i.e. date/time, amount)
* Built a **rule engine** of eight monitoring scenarios modelled on real bank practice, and scored each one for alert volume, productivity and analyst cost
* Trained an **XGBoost** classifier and evaluated it with metrics appropriate to a 0.1% event rate
* Compared three operating models — rules alone, rules ranked by the model, and a hybrid — at matched analyst capacity

Throughout, I split the data **chronologically** rather than randomly, so the model is always tested on a period that comes after everything it learned from.

### Results <a name="overview-results"></a>

Scored on a held-out period of 97 days containing 722,838 transactions and 3,099 laundering transactions:

| Measure | Value |
|---|---|
| Alerts Raised | 108,843 |
| Alert Productivity | 1.37% |
| Detection Rate | 26.4% of suspicious account-days |
| Analyst-Days of Work | 7,558 |
| **Analysts Needed to Keep up** | **109.5** |

A ruleset of eight plausible scenarios, run over three months of one synthetic bank's traffic, implies a standing investigation team of over a hundred people — to find roughly a quarter of the laundering present.

The per-rule breakdown showed where that budget goes and which scenarios earn their place. One fan-out scenario consumed **68% of the entire investigation budget**. Two others — structuring and pass-through — produced **16,489 alerts and 1,145 analyst-days between them, while contributing only 6 unique detections (i.e. that no other rule had already found).**

Adding a prioritization and a machine learning model changed the picture drastically. At a quarter of current capacity, ranking the same alerts by model score lifted detection from **6.9% to 23.0%**, and adding accounts the rules never flagged (via ML model) lifted it to **77.6%**. Expressed as effort, reaching a 20% detection target costs **84.5 analysts** with rules alone and **1.2** with the same alerts ranked.

### Growth/Next Steps <a name="overview-growth"></a>

Potential future enhancements include:

* Synthesising an investigator case-management layer to model queue backlogs and SLA breaches
* Cost-based thresholding using real currency values rather than alert counts
* Validating the approach on a second dataset with different generation logic
* Introducing new rules and measuring the lift or change in detection
* Network-level detection, since laundering is a property of a *group* of accounts rather than any single one

___

# 01. Data Overview <a name="data-overview"></a>

I used **SAML-D**, a synthetic transaction monitoring dataset built by researchers at Bournemouth University in consultation with practising AML specialists. The transaction data itself was generated by a simulation, rather than taken from real customers and scrubbed of personal identifying information or anonymized. [NTD: add citation]

[DuckDB was used here since it allows us to query large files without loading them into memory]

The dataset contains **9,504,852 transactions**, of which **9,873 are laundering — 0.1039%**. From a slightly different perspective, the data covers **855,460 accounts**, of which only **7,902 ever touch a suspicious transaction.** A brief overview of the data:

| Column | Meaning |
|---|---|
| `Date`, `Time` | When the transaction happened |
| `Sender_account`, `Receiver_account` | Who paid whom |
| `Amount` | Value of the payment |
| `Payment_currency`, `Received_currency` | Currency sent and received |
| `Sender_bank_location`, `Receiver_bank_location` | Countries involved |
| `Payment_type` | Cash deposit, cheque, cross-border transfer, card, etc. |
| `Is_laundering` | Target: 1 = laundering, 0 = normal |
| `Laundering_type` | Which pattern of laundering (or normal behaviour) this belongs to |

That last column is unusually valuable. A **typology** is a named pattern of criminal behaviour, and the dataset labels 17 distinct suspicious ones alongside 11 normal ones. In plain terms, a few of the suspicious ones are:

* **Structuring** (also called *smurfing*) — breaking one large payment into many small ones to stay under reporting thresholds
* **Fan-out** — one account rapidly paying out to many different recipients
* **Fan-in** — many accounts paying into one, gathering funds for onward movement
* **Cycle** — money moving through a loop of accounts and returning to its origin
* **Layering** — deliberately adding hops between the crime and the cash to obscure the trail

<br>
Having typologies labelled means I can ask a far more useful question than "how accurate is my model", instead allowing me to probe what kinds of criminal behaviour my program can actually see.

I also profiled the risk carried by each attribute including payment type, payment currency, and the country initiating or receiving the payment. Cash and cross-border payments were roughly ten times riskier than routine electronic transfers:

| Payment type | Transactions | Laundering rate |
|---|---|---|
| Cash Deposit | 225,206 | 0.624% |
| Cash Withdrawal | 300,477 | 0.444% |
| Cross-border | 933,931 | 0.281% |
| ACH | 2,008,807 | 0.058% |
| Credit card | 2,012,909 | 0.056% |
| Cheque | 2,011,419 | 0.054% |

Destination country mattered even more, with the riskiest payment corridors running to Morocco, Nigeria, and Albania; each representing a laundering rate which was 6.350, 6.211, and 5.574 times higher than the baseline, respectively.

[NTD: add dashboard link for risk profile]
___

# 02. Why Program Effectiveness Is Not Model Accuracy <a name="effectiveness-overview"></a>

Before building anything, I needed to determine what metric(s) I would use to measure 'effectiveness' since traditional performance measures fall short:

* Due to the highly imbalanced nature of the dataset, *Accuracy* is not a useful measure. Predicting "not laundering" for all 9.5 million transactions scores **99.896%** while catching nothing at all. Any optimiser given accuracy as a target will find that answer immediately and stop. 
had to settle what "good" means. Three commonly used measures are actively misleading here.
* *ROC-AUC* summarises how well a model separates the two classes, but it measures false alarms as a share of the enormous innocent majority, providing an optimistically biased measure of performance. Flagging 10,000 legitimate transactions barely moves the number, while representing months of analyst work.
* *Precision-Recall AUC* measures precision (of the alerts I raised, what fraction are genuinely suspicious?) and recall (of all the laundering that actually happened, what fraction did I catch?).

Out of the metrics mentioned here, PR-AUC is the most useful, but still isn't the real objective metrics should be operational in nature. Instead I've proposed the following metrics regarding analyst effort to determine how well a program is performing:   

| Metric | The question it answers |
|---|---|
| Alert volume | How much work am I creating? |
| Alert productivity | How much of that work is wasted? |
| Analyst days | What does this cost in headcount? |
| Typology coverage | Which crimes can I not see at all? |
| Marginal rule value | What would I lose by switching this rule off? |

___

# 03. Building Features From Almost Nothing <a name="feature-engineering"></a>

SAML-D provides only twelve raw columns, and no information about the customer at all — no age of account, no occupation, no expected activity. Very little of that is useful on its own. For example, a large payment on its own tells me nothing; a large payment from an account that has been dormant for six months and has just paid out to eleven new recipients tells me a great deal.

So the real work was **feature engineering** — deriving new measurements that describe *behaviour over time* rather than a single payment in isolation. I built these in DuckDB, which allowed me to use SQL within a notebook, computing rolling time windows across millions of rows far faster than standard Python tools.

The features fall into five families:

* **Velocity** — how many transactions, and what total value, has this account sent or received in the last 1, 7 and 30 days?
* **Counterparty spread** — how many *different* accounts has it dealt with recently? This is what exposes fan-in and fan-out patterns.
* **Pass-through ratio** — does money leave this account almost as fast as it arrives, and in similar amounts? This is the classic signature of a **money mule**, an account used purely to relay funds.
* **Dormancy** — how long was this account inactive before it suddenly started moving money?
* **Behavioural change** — how unusual is this payment compared to that same account's own history?

The single most important technical safeguard is that **every one of these windows only looks backwards.** If a feature were allowed to see the future, the model would appear brilliant in testing and fail completely in production. This is called **leakage**, and it is the most common way that machine learning projects quietly fool their authors.

<br>
**Why this matters:**  I also split train and test **by date** rather than randomly. With behavioural features, a random split lets the model learn from transactions that happen after the ones it is being tested on — which is impossible in real life and inflates every number.

The same discipline applies to the rule engine: its thresholds are **calibrated on the training period and scored on the test period**, so the rules and the model are always judged on the same unseen data.

I then ran a deliberate **leakage tripwire**: scoring each feature's predictive power on its own. If any single feature had scored near-perfectly, it would have meant something was secretly encoding the answer. The strongest was dormancy at 0.734 — high enough to be useful, low enough to be believable.

One finding here was genuinely useful: two features I had expected to matter, proximity to a £10,000 reporting threshold and round-number amounts, scored **0.495 and 0.500** — statistically indistinguishable from a coin flip. SAML-D simply does not model a cash-reporting threshold. I removed them from the model but deliberately **kept them in the rule engine**, for reasons that become clear below.

### Making 9.5 million rows tractable

Computing rolling windows across 9.5 million transactions is expensive, so I sampled. But sampling *transactions* at random would have destroyed the very features I had just built — you cannot measure how many counterparties an account dealt with if you have thrown away half of them.

Instead I sampled **whole accounts**: every account involved in any laundering, plus a random selection of clean ones, keeping all of their transactions. That preserved behaviour intact and left **2,461,050 transactions containing all 9,873 laundering cases.**

The side effect is that laundering now looks about four times more common than it really is. I measured that enrichment precisely (**3.874×**) and corrected for it whenever reporting results, because otherwise every estimate of analyst workload would have been four times too optimistic.

___

# 04. The Rule Engine Baseline <a name="rule-engine"></a>

Before adding any machine learning, I built a hypothetical system a financial institution would run today. This ordering is deliberate since the value of any model can only be expressed in relation to a baseline. 

I implemented eight monitoring scenarios:

| Rule | What it looks for |
|---|---|
| `R01_STRUCTURING` | Many small payments in one day summing to a large total |
| `R02_PASSTHROUGH` | Money arriving and leaving in near-identical amounts |
| `R03_HR_CORRIDOR` | Large payments into high-risk country corridors |
| `R04_FAN_OUT` | One account paying many different recipients quickly |
| `R05_FAN_IN` | Many accounts paying into one |
| `R06_DORMANT` | A long-inactive account suddenly moving significant money |
| `R07_XCCY_LAYER` | Cross-border, cross-currency payments at speed |
| `R08_LARGE_CASH` | Unusually large cash transactions |

Two design decisions were important.

**Thresholds are set per payment type.** My first attempt used one global amount threshold and the cash rule never fired once. The reason was instructive: cash withdrawals in this data top out around **£342**, while electronic transfers reach **£48,000**. A single threshold is meaningless across distributions that differ by two orders of magnitude.

**Alerts are grouped by account and day, not by transaction.** Real monitoring systems raise one alert per account per scenario per day — an analyst investigates *an account*, not each individual payment. This collapses roughly five rule hits into every one alert, and skipping it would have inflated the apparent workload by the same factor.
[[The rule engine reproduced the central problem of real AML programs: a very large alert queue in which the overwhelming majority of alerts are innocent.]]

### Results

*[To be completed from the test-period run.]*

| Rule | Alerts | Productive | Productivity | Recall | Analyst days |
|---|---|---|---|---|---|
| `R08_LARGE_CASH` | | | | | |
| `R07_XCCY_LAYER` | | | | | |
| `R03_HR_CORRIDOR` | | | | | |
| `R05_FAN_IN` | | | | | |
| `R06_DORMANT` | | | | | |
| `R04_FAN_OUT` | | | | | |
| `R01_STRUCTURING` | | | | | |
| `R02_PASSTHROUGH` | | | | | |

**Portfolio total:** *[alerts, productivity, recall, analyst-days]*

Whatever the exact figure, a productivity rate of a few percent is not a failure of the exercise — it is a faithful reproduction of the industry's actual problem. Real transaction monitoring runs somewhere between 1% and 5% productivity.

### The findings that matter

**One rule dominates the cost base.** The fan-out scenario alone consumes the large majority of the investigation budget while delivering the largest single share of detections. It is simultaneously the most productive rule by volume and by far the most expensive per detection. That is a tuning conversation, not a deletion — and it is the first thing I would put in front of a head of financial crime.

**Two rules are close to worthless.** I measured **marginal value**: how much unique detection would be lost if a rule were switched off entirely. Structuring and pass-through generate tens of thousands of alerts and thousands of analyst-days between them, while contributing only a handful of detections that no other rule had already found.

This is exactly why I kept the "dead" structuring features in the rule engine after removing them from the model. **A rule that burns 11% of the budget and finds almost nothing is a finding in its own right** — and it is the single most actionable thing on the dashboard.

**Coverage is deeply uneven.** The rule engine sees some behaviour almost perfectly and other behaviour barely at all:

*[To be completed from the test-period run.]*

| Typology | Cases | Caught | Coverage |
|---|---|---|---|
| Behavioural_Change_1 | | | |
| Behavioural_Change_2 | | | |
| Structuring | | | |
| Fan_In | | | |
| Deposit-Send | | | |
| Cycle | | | |
| Smurfing | | | |
| Bipartite | | | |
| Layered_Fan_Out | | | |

Across the full dataset the spread ran from near-total coverage of simple behavioural changes down to roughly a fifth of layered fan-out structures.

<br>
**Why this matters:**  That gap is the case for adding a model. The layered structures — deliberately designed to defeat simple rules — are exactly where a rules-only program is weakest, and exactly where sophisticated criminals operate.

___

# 05. The Machine Learning Model <a name="the-model"></a>

I trained an **XGBoost** classifier — a *gradient boosting* model, which builds many small decision trees in sequence with each one correcting the errors of those before it. It handles the messy, non-linear interactions in behavioural data well and copes natively with missing values, which matters here because an account's first ever transaction genuinely has no history.

### Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.9959 |
| PR-AUC (sampled) | 0.8870 |
| **PR-AUC (population-corrected)** | **0.8246** |

The features the model relied on most were the behavioural ones, not the transaction details:

```
snd_cnt_7d        0.283   sender's transaction count, last 7 days
snd_n_cp_7d       0.181   sender's distinct counterparties, last 7 days
rcv_cnt_7d        0.106   receiver's transaction count, last 7 days
rcv_n_cp_7d       0.096   receiver's distinct counterparties, last 7 days
rcv_cnt_1d        0.070   receiver's transaction count, last 24 hours
```

Payment amount contributed just **1.5%**. That matches AML practice: it is the *pattern of movement* that betrays laundering, not the size of any single payment.

### Tuning

*[To be completed from the notebook run.]*

I tune the model with **Optuna**, which uses Bayesian optimisation to search hyperparameters intelligently rather than exhaustively. Two choices differ from a standard tuning setup:

1. **Cross-validation must be chronological.** Standard cross-validation shuffles rows randomly, which with time-based features would let the model learn from the future. I use rolling-origin folds where every validation period is strictly later than its training period.
2. **The objective is PR-AUC**, for the reasons in section 02.

| Metric | Value |
|---|---|
| Trials | 30 |
| Best cross-validated PR-AUC | |
| Best parameters | |
| Test PR-AUC (tuned) | |
| Improvement over baseline | |

___

# 06. Rules vs Model vs Hybrid <a name="hybrid-comparison"></a>

*[To be completed from the notebook run.]*

This is the section the whole project builds toward. I compare three ways of operating the same program, **at matched analyst capacity**:

**A — Rules only.** The current state. Rules produce a queue of alerts with no priority order, so analysts work it in whatever order it arrives.

**B — Rules plus triage.** Exactly the same alerts, but sorted by the model's score so the most likely productive ones surface first. **This detects nothing new.** Its entire value is ordering — which matters enormously the moment capacity is constrained.

**C — Hybrid.** Rule alerts plus accounts the model flags that the rules never looked at. This is the only configuration that can raise the ceiling on how much is caught.

| Config | Recall at 25% capacity | at 50% | at 100% | Effort saved at equal recall |
|---|---|---|---|---|
| A — Rules only | | | | |
| B — Rules + triage | | | | |
| C — Hybrid | | | | |

The two questions this table answers are the two a head of financial crime actually asks:

* *"If I hold detection steady, how much analyst time do I save?"*
* *"If I hold headcount steady, how much more do I catch?"*

**A note on units:**  the rule scorecard in section 04 counts one alert per rule, per account, per day, because that is how rule productivity is conventionally reported. This comparison instead counts one alert per **account per day** — because three rules firing on the same account on the same day is one piece of investigative work, not three. The volumes here are therefore lower than section 04's by design.

___

# 07. Honest Limitations <a name="limitations"></a>

This section matters more than it usually would, and I would rather state it plainly than have someone find it.

**Synthetic data is far more learnable than reality.** SAML-D's suspicious typologies are *generated* from counterparty-count and velocity patterns — which is precisely what my strongest features measure. To a real extent the model is recovering the data generator's own construction. A PR-AUC of 0.8246 on genuine banking data would be extraordinary; here it partly reflects the fact that the patterns were manufactured. **The shape of these results is the finding; the magnitudes are a property of the dataset.**

**The synthesised operational layer is illustrative.** No public dataset ships with real investigator outcomes, so analyst handling time (25 minutes per alert) and effective capacity (6 productive hours per day) are stated assumptions, not measurements. They are plausible industry figures, and every number derived from them moves proportionally if you disagree with them.

**There is no true alert-disposition ground truth.** I know which transactions were laundering, but not which alerts a human investigator would have escalated. "Productive alert" here means "contained a genuinely suspicious transaction", which is an upper bound on what a real analyst would find.

___

# 08. Results Comparison <a name="results-comparison"></a>



| Approach | Alerts | Productivity | Recall | Analyst days |
|---|---|---|---|---|
| Rules only — full queue | | | | |
| Rules + triage — 25% capacity | | | | |
| Hybrid — 25% capacity | | | | |
| Hybrid — matched recall | | | | |

All rows are scored on the same held-out test period.

### Summary of the three levers

| | What it changes | Cost | Strengths | Limitations |
|---|---|---|---|---|
| **Rule tuning** | Which alerts exist at all | Analyst review of scenarios | Transparent, explainable to a regulator, no model risk | Blind to patterns nobody wrote a rule for |
| **Model triage** | The *order* alerts are worked in | One model, retrained periodically | Large capacity savings; adds no regulatory risk since no rule is switched off | Cannot detect anything the rules missed |
| **Hybrid detection** | Which accounts are looked at | Model plus governance overhead | Only option that raises the detection ceiling | Requires model explainability for regulators |

The way to read this is that **triage and detection are different products.** Triage makes the existing program cheaper and is comparatively easy to get approved, because no rule is being turned off. Hybrid detection makes the program *better* but requires an institution to justify a model-driven alert to a regulator. Most institutions do the first before attempting the second, and the ordering in this project reflects that.

___

# 09. Growth & Next Steps <a name="growth-next-steps"></a>

Potential future enhancements include:

* **A case management layer.** Synthesising analyst assignment, investigation duration and queue backlogs would allow the dashboard to show SLA breaches and case aging — the operational measures AML leadership reviews most often.
* **Cost-based thresholds.** Replacing "alerts per analyst per day" with real currency — expected laundering value detected against investigation cost — so the operating point is chosen on money.
* **Network-level detection.** Laundering is a property of a *group* of accounts, not a single one. Graph features, or a graph neural network, would model that directly instead of approximating it with counterparty counts.
* **Validation on a second dataset.** Repeating the pipeline on the IBM AML dataset, which uses entirely different generation logic, would separate genuine method from artefacts of SAML-D specifically.
* **Rule retirement testing.** The marginal value analysis suggests two scenarios could be retired or rebuilt. Simulating that decision — and quantifying the freed capacity — is a concrete deliverable a bank would act on.

This project demonstrates that in financial crime detection, the highest-leverage question is rarely *"how accurate is the model?"* It is **"where is my analyst capacity going, and what am I still unable to see?"** — and answering that takes an operational pipeline, not just a classifier.

___
