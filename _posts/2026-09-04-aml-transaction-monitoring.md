---
layout: post
title: Measuring Anti-Money Laundering Program Effectiveness
image: "/posts/aml-transaction-monitoring-title-img.png"
tags: [EDA, Feature Engineering, Data Visualization, DuckDB, XGBoost, scikit-learn, Optuna, SQL, Python, Tableau]
---

In this project I build a complete **anti-money laundering transaction monitoring pipeline** on a synthetic dataset, and then measure how effective it actually is.

I start by building a baseline (deterministic) **rule-based system** which tries to capture fraudulent behavior patterns by producing alerts or flags (which would be investigated by an analyst). I also design metrics and KPIs to measure the effectiveness of this baseline which focus on the quality of the alerts produced and time to investigate. 

Next, I add in **machine learning layer** and study its impact on the metrics and KPIs; producing an **interactive dashboard** to display the findings. 

This project taught me how to connect data analysis with real-world outcomes, highlighting the difference between model quality or performance and effectiveness when real-world costs and variables are considered. Additionally, this project provided an opportunity to further develop my data storytelling and visualizations skills, by creating compelling dashboards which supported my findings.

[![AML Program Effectiveness dashboard: alert cost by rule, detection versus alert volume for three operating models, and typology coverage gaps](/img/posts/aml-dashboard-full.png)](https://public.tableau.com/app/profile/christine.dowling/viz/AMLProgramEffectiveness/AMLProgramEffectiveness)

*The interactive version of the dashboard is available **[here.](https://public.tableau.com/app/profile/christine.dowling/viz/AMLProgramEffectiveness/AMLProgramEffectiveness)** Capacity and operating-model can be toggled to see the impact the metrics.*

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Measuring Program Effectiveness](#effectiveness-overview)
- [03. Feature Engineering](#feature-engineering)
- [04. Rule Engine Baseline](#rule-engine)
- [05. Machine Learning Model Layer](#the-model)
- [06. Rules vs Model vs Hybrid](#hybrid-comparison)
- [07. Limitations](#limitations)
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

* Loaded and queried the SAML-D synthetic dataset
* Conducted **exploratory data analysis** to better understand the risk profile of the dataset
* Engineered **behavioural features** to measure how an account behaves over time, beyond payment metrics (e.g. date/time, amount)
* Built a **rule engine** of eight monitoring scenarios and scored each one for alert volume, productivity and analyst cost
* Trained and tuned an **XGBoost classifier** and evaluated it with metrics appropriate to a 0.1% event rate
* Compared three operating models — rules alone, rules ranked by the model, and a hybrid — at matched analyst capacity

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

The dataset contains **9,504,852 transactions**, of which **0.1039% or 9,873 are laundering**. A brief overview of the schema:

| Column | Meaning |
|---|---|
| `Date`, `Time` | When the transaction happened |
| `Sender_account`, `Receiver_account` | Transaction parties  |
| `Amount` | Value of the payment |
| `Payment_currency`, `Received_currency` | Currency sent and received |
| `Sender_bank_location`, `Receiver_bank_location` | Countries involved |
| `Payment_type` | Method use (i.e. Cash deposit, cheque, cross-border transfer, card, etc.) |
| `Is_laundering` | Target: 1 = laundering, 0 = normal |
| `Laundering_type` | Which pattern of laundering (or normal behaviour) this belongs to |

That last column is unusually valuable. A **typology** is a named pattern of criminal behaviour, and the dataset labels 17 distinct suspicious ones alongside 11 normal ones. In plain terms, a few of the suspicious ones are:

* **Structuring** (also called *smurfing*) — breaking one large payment into many small ones to stay under reporting thresholds
* **Fan-out** — one account rapidly paying out to many different recipients
* **Fan-in** — many accounts paying into one, gathering funds for onward movement
* **Cycle** — money moving through a loop of accounts and returning to its origin
* **Layering** — deliberately adding hops between the crime and the cash to obscure the trail

Remainder of the typologies and their definitions are found in this paper [NTD: add citation]

Having typologies labelled means I can look deeper into the model performance, beyond accuracy, instead allowing me to probe what kinds of criminal behaviour my program can actually see.

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

Data covers **855,460 accounts**, of which only **7,902 ever touch a suspicious transaction.**

[NTD: add dashboard link for risk profile]

___

# 02. Measuring Program Effectiveness <a name="effectiveness-overview"></a>

Before building anything, I needed to determine what metric(s) I would use to measure 'effectiveness' since traditional performance measures fall short:

* *Accuracy* measures the overall performance however, due to the highly imbalanced nature of the dataset, this is not a useful measure. Predicting "not laundering" for all 9.5 million transactions scores **99.896%** while catching no suspicious transactions. Any optimizer given accuracy as a target will find that answer immediately and stop. 
* *ROC-AUC* summarises how well a model separates the two classes, but it measures false alarms as a share of the non-fraudulent majority, providing an optimistically biased measure of performance. Flagging 10,000 legitimate transactions barely moves the number, while representing months of analyst work.
* *Precision-Recall AUC* measures precision (of the alerts I raised, what fraction are genuinely suspicious?) and recall (of all the laundering that actually happened, what fraction did I catch?).

Out of the metrics mentioned here, PR-AUC is the most useful, but still isn't the real objective as metrics should be operational in nature. Instead I proposed the following metrics surrounding **analyst effort** to determine how well a program is performing:   

| Metric | The question it answers |
|---|---|
| Alert volume | How much work am I creating? |
| Alert productivity | How much of that work is wasted? |
| Analyst days | What does this cost in headcount? |
| Typology coverage | Which crimes can I not see at all? |
| Marginal rule value | What would I lose by switching this rule off? |

___

# 03. Engineering Features<a name="feature-engineering"></a>

SAML-D provides only twelve raw columns, and no information about the customer at all (i.e. age of account, occupation, expected activity), which provides very little information on its own. For example, a large payment by itself tells me nothing; a large payment from an account that has been dormant for six months and has just paid out to eleven new recipients is far more informative.

Therefore, in order to make use of this data I need to derive new measurements that describe an account's (client's) *behaviour over time* - also known as **feature engineering**. Rather than using Pandas to complete this task (given the size of the dataset), I built these in DuckDB, which allowed me to run  SQL-based queries within a notebook, computing rolling time windows across millions of rows far faster than standard Python tools.

The features broadly fall into five families:

* **Velocity** — how many transactions, and what total value, has this account sent or received within a defined time period?
* **Counterparty spread** — how many *different* accounts has it dealt with recently?
* **Pass-through ratio** — does money leave this account almost as fast as it arrives, and in similar amounts? 
* **Dormancy** — how long was this account inactive before it suddenly started moving money?
* **Behavioural change** — how unusual is this payment compared to that same account's own history?

An important safeguard which became apparent after designing these features was that chronology would be important to preserve. Every one of these windows only looks **backwards.** This has implications for splitting the data into training and testing, any cross-validation, as well as sampling. Effectively, I needed to ensure that features didn't contain a mix of past and future data (also known as **leakage**) which could artificially inflate the model's performance. To deal with this, I split train and test **by date** rather than randomly, so that the model doesn't learn from 'future transactions'. This split was also applied to the rule engine to ensure all options were scored on the same unseen data.  

### Sampling

Computing rolling windows across 9.5 million transactions is expensive, so I sampled. But sampling *transactions* at random would have destroyed the features above including velocity and counterparties.

Here I sampled **whole accounts**: every account involved in any laundering, plus a random selection of clean ones, keeping all of their transactions. That preserved behaviour intact and left **2,461,050 transactions containing all 9,873 laundering cases.**

The side effect is that laundering now looks about four times more common than it really is. I measured that enrichment precisely (**3.874×**) and corrected for it whenever reporting results, otherwise every estimate of analyst workload would have been almost four times too optimistic.

___

# 04. The Rule Engine Baseline <a name="rule-engine"></a>

Before adding any machine learning, I built a simple, hypothetical, rule-based system akin to a transaction monitoring system which might exist today. This provided insight on the amount of coverage a system like this might have, as well as a baseline for assessing the value of any machine learning model additions down the road. For simplicity, I implemented eight rules which targeted different monitoring scenarios:

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

**Thresholds are set per payment type.** My first attempt used one global amount threshold and the cash rule never fired once. The reason for this was that different payment types had vastly different caps, making a single threshold useless.

**Alerts are grouped by account and day, not by transaction.** Real monitoring systems raise one alert per account per scenario per day since analysts investigate *an account*, not each individual payment. This collapses roughly five rule hits into every one alert, and prevents artificial inflation of the workload.


### Results

| Rule | Alerts | Productive | Productivity | Recall | Analyst days | Unique |
|---|---|---|---|---|---|---|
| `R08_LARGE_CASH` | 380 | 87 | **22.89%** | 2.8% | 26 | 68 |
| `R07_XCCY_LAYER` | 4,022 | 230 | 5.72% | 8.2% | 279 | 132 |
| `R03_HR_CORRIDOR` | 1,510 | 76 | 5.03% | 2.5% | 105 | 34 |
| `R05_FAN_IN` | 11,980 | 267 | 2.23% | 9.2% | 832 | 143 |
| `R06_DORMANT` | 6,732 | 70 | 1.04% | 2.3% | 468 | 55 |
| `R04_FAN_OUT` | 85,757 | 653 | 0.76% | 25.0% | **5,955** | 587 |
| `R01_STRUCTURING` | 13,540 | 62 | 0.46% | 2.5% | 940 | **4** |
| `R02_PASSTHROUGH` | 2,949 | 6 | **0.20%** | 0.2% | 205 | **2** |

**Portfolio total: 126,870 rule-alerts, 1.14% productive, 8,810 analyst-days.**

NOTE: While productivity rate appears very low, this is comparable to real transaction monitoring systems which run ~1-5% productivity. 

### The findings that matter

**One rule dominates the cost base.** The fan-out scenario alone consumes the large majority of the investigation budget (5,955 analyst days, **68% percent of the budget**) while delivering the largest single share of detections (25% recall). It is simultaneously the most productive rule by volume and the most expensive per detection.

**Two rules are close to worthless.** I measured **marginal value** by looking at how many unique detections would be lost if a rule were switched off entirely. Structuring contributed **4 unique detections** for 13,540 alerts and 940 analyst-days. Pass-through contributed **2**, for 2,949 alerts. Between them: 16,489 alerts, 1,145 analyst-days, and 6 detections nothing else already found. 

While findings might look like easy targets for cutting costs, removing or altering these rules should be a business decision which takes into account the real and full cost (i.e. the cost of missed laundering transactions). 

**Coverage is deeply uneven.** The rule engine sees some behaviour almost perfectly and other behaviour barely at all:

| Typology | Cases | Caught | Coverage |
|---|---|---|---|
| Behavioural_Change_1 | 133 | 133 | **100.0%** |
| Behavioural_Change_2 | 148 | 148 | 100.0% |
| Fan_In | 102 | 61 | 59.8% |
| Structuring | 556 | 321 | 57.7% |
| Deposit-Send | 297 | 135 | 45.5% |
| Cycle | 114 | 48 | 42.1% |
| Layered_Fan_In | 179 | 40 | 22.3% |
| Smurfing | 279 | 61 | 21.9% |
| Bipartite | 63 | 10 | **15.9%** |

<br>
**The gap above is the case for adding a model.** Layered structures are deliberately designed to defeat simple rules and are exactly where sophisticated criminals operate.

___

# 05. The Machine Learning Model <a name="the-model"></a>

Taking some lessons from the Fraud Detection project, I trained an **XGBoost** classifier since it is appropriate for the messy, non-linear interactions in behavioural data. I also used Bayesian optimization (Optuna) to tune hyperparameters rather than using brute force (e.g. GridSearch). Two points differed from the previous project:

* **Cross-validation must be chronological.** As mentioned in the feature engineering section above, preserving the chronology of transactions is important to prevent leakage. Standard cross-validation shuffles rows randomly, therefore I use rolling-origin folds where every validation period is strictly later than its training period.
* **The objective is PR-AUC**, rather than accuracy or ROC-AUC for the reasons stated in section 02.

### Results

| Metric | Value |
|---|---|
| Trials | 30 |
| Best cross-validated PR-AUC | 0.8320 |
| Baseline test PR-AUC (population-adjusted) | 0.8259 |
| **Tuned test PR-AUC (population-adjusted)** | **0.8576** |

The features the model relied on were behavioural, not transactional:

```
snd_cnt_7d        sender's transaction count, last 7 days
snd_n_cp_7d       sender's distinct counterparties, last 7 days
rcv_cnt_7d        receiver's transaction count, last 7 days
rcv_n_cp_7d       receiver's distinct counterparties, last 7 days
rcv_cnt_1d        receiver's transaction count, last 24 hours
```

Payment amount contributed under 2%. That is not surprising since the *pattern of movement* is typically what demonstrates laundering, not the size of any single payment.

# 06. Thresholding <a name="thresholding"></a>

The model produces a **score**, not a decision. Turning a score into an alert requires a cut-off, and that cut-off is not a hyperparameter — it is chosen after training, and it is the cheapest lever in the pipeline.

The trap is choosing it on the test set. Sweeping thresholds on test and reporting the best one leaks the test set into the decision. So I swept **out-of-fold** predictions from the training period, reusing the same rolling-origin folds, so each score came from a model that never saw that row.

| Strategy | Threshold | PR-AUC | Precision | Recall | Alerts | Missed |
|---|---|---|---|---|---|---|
| Default 0.5 | 0.5000 | 0.8576 | 0.9672 | 0.7699 | 2,467 | 713 |
| Best F1 (OOF) | 0.2693 | 0.8576 | 0.9144 | 0.8203 | 2,780 | 557 |
| Capacity (OOF) | 0.0223 | 0.8576 | 0.4106 | 0.9264 | 6,992 | 228 |

**PR-AUC is identical on every row.** That is the clearest demonstration that thresholding does not improve the model. It cannot change how well the model *ranks*; it only chooses where to cut that ranking. Any PR-AUC gain has to come from the model, and any precision-recall trade comes from here.

The default 0.5 is badly wrong, but in the opposite direction from most fraud problems: near-perfect precision while missing 23% of cases. It is tuned for a balanced problem that does not exist.

<br>
**Why this matters:**  The capacity-based row is the one an AML team actually operates under. The review team can work N alerts a day, so the threshold is set by headcount, not by any statistical criterion.
___

# 07. Rules vs Model vs Hybrid <a name="hybrid-comparison"></a>

I compared three ways of operating the same program on the same held-out period.

**A — Rules only.** The current state. Rules produce a queue with no priority order, so analysts work it in whatever order it arrives.

**B — Rules plus triage.** Exactly the same alerts, sorted by model score so the most likely productive ones surface first. **This detects nothing new.** Its entire value is ordering.

**C — Hybrid.** Rule alerts plus accounts the model flags that the rules never looked at. The only configuration that raises the ceiling.

Triage works by reordering, not by adding or removing. Each account-day alert inherits the **highest** model score among its transactions, and the queue is sorted by that score. An account-day containing one highly suspicious payment among twenty routine ones ranks high, where averaging would dilute it.

### At matched capacity

| Config | Alerts | Analysts | Caught | Detection | Productivity |
|---|---|---|---|---|---|
| **At 25% of current capacity** ||||||
| A — Rules only | 27,210 | 27.4 | 389 | 6.9% | 1.43% |
| B — Rules + triage | 27,210 | 27.4 | 1,297 | **23.0%** | 4.77% |
| C — Hybrid | 27,210 | 27.4 | 4,377 | **77.6%** | 16.09% |
| **At 100% of current capacity** ||||||
| A — Rules only | 108,843 | 109.5 | 1,487 | 26.4% | 1.37% |
| B — Rules + triage | 108,843 | 109.5 | 1,487 | 26.4% | 1.37% |
| C — Hybrid | 108,843 | 109.5 | 4,600 | **81.6%** | 4.23% |

Two things stand out. At a quarter of capacity, ranking alone more than triples detection for identical cost. And at full capacity A and B are **identical** — because if you work every alert, the order is irrelevant. All of triage's value exists under constraint, which is the permanent condition of every real AML team.

### Inverted: what does a detection target cost?

| Detection target | A — Rules only | B — Rules + triage | C — Hybrid |
|---|---|---|---|
| 5% | 19.6 analysts | 0.3 | 0.3 |
| 10% | 39.6 | 0.6 | 0.6 |
| 20% | **84.5** | **1.2** | **1.1** |
| 25% | 103.5 | 70.2 | 1.4 |
| 50% | beyond ceiling | beyond ceiling | 2.8 |
| 80% | beyond ceiling | beyond ceiling | 65.7 |

The 25% row is worth pausing on. Triage jumps from 1.2 analysts to 70.2, because ranking reaches ~23% almost free and then must grind the remaining queue for the last two points. That diminishing return is real, and it is what makes the case for the hybrid rather than merely better sorting.

The blanks are also a finding. Rules cannot reach 30% at any cost — **26.4% is the structural ceiling of the ruleset**, and no amount of headcount changes it.

### Closing the coverage gaps

Compared at equal alert volume, the hybrid closes exactly the gaps the rules could not see:

| Typology | Rules only | Hybrid | Gain |
|---|---|---|---|
| Bipartite | 10.3% | 94.9% | **+84.6pp** |
| Layered_Fan_In | 15.4% | 91.7% | +76.3pp |
| Smurfing | 17.9% | 91.9% | +74.1pp |
| Cash_Withdrawal | 19.3% | 90.8% | +71.5pp |
| Gather-Scatter | 17.0% | 88.3% | +71.3pp |

<br>
**[Explore this comparison interactively →](https://public.tableau.com/app/profile/christine.dowling/viz/AMLProgramEffectiveness/AMLProgramEffectiveness)**
Move the capacity control to see detection and analyst headcount change across all three operating models.

<br>
**Why this matters:**  Triage makes the existing program cheaper and is comparatively easy to get approved, because no rule is switched off and no alert suppressed. Hybrid detection makes the program *better* but requires justifying a model-driven alert to a regulator. Most institutions do the first before attempting the second.


The two questions this table answers are the two a head of financial crime actually asks:

* *"If I hold detection steady, how much analyst time do I save?"*
* *"If I hold headcount steady, how much more do I catch?"*

**A note on units:**  the rule scorecard in section 04 counts one alert per rule, per account, per day, because that is how rule productivity is conventionally reported. This comparison instead counts one alert per **account per day** — because three rules firing on the same account on the same day is one piece of investigative work, not three. The volumes here are therefore lower than section 04's by design.

___

# 08. Counting Honestly <a name="counting-honestly"></a>

Two figures in this project both look like "the answer" and differ by a third. Both are correct; they count different things, and being explicit about which is which is most of the work.

| | Rule scorecard | Program comparison |
|---|---|---|
| Alert unit | rule × account × day | account × day |
| Alert count | 126,870 | 108,843 |
| Analyst-days | 8,810 | 7,558 |
| What "caught" means | suspicious *transactions* | suspicious *account-days* |
| Denominator | 3,099 | 5,640 |

When three rules fire on one account on one day, the scorecard counts three — that is how rule productivity is conventionally reported — but an analyst does one piece of work. And each suspicious transaction touches up to two account-days, one for the sender and one for the receiver, so 3,099 transactions expand to 5,640 distinct account-days.

The dashboard uses **account-days throughout**, because that is the unit an analyst investigates.

### The staffing arithmetic

```
alerts × 25 minutes ÷ 60 = analyst-hours
analyst-hours ÷ 6 productive hours = analyst-days
analyst-days ÷ 69 working days = analysts required
```

Six productive hours represents an eight-hour day less meetings, admin and training. The divisor is **working days, not calendar days** — the test period spans 97 calendar days but only 69 weekdays. Alerts accumulate seven days a week; the capacity to work them does not. Using calendar days would have understated headcount by 41%.

<br>
**Why this matters:**  Effort here measures what a detection target costs, **not a staffing recommendation**. All rule alerts still require disposition however they are prioritised — working only the top 1,210 leaves the rest un-dispositioned, which is a regulatory finding rather than an efficiency win.

___

# 09. Limitations <a name="limitations"></a>

**Synthetic data is far more learnable than reality.** SAML-D's typologies are *generated* from counterparty-count and velocity patterns — precisely what my strongest features measure. To a real extent the model recovers the generator's own construction. A PR-AUC of 0.8576 on genuine banking data would be extraordinary. **The shape of these results is the finding; the magnitudes are a property of the dataset.**

**The operational layer is illustrative.** No public dataset ships with real investigator outcomes, so 25 minutes per alert and 6 productive hours per day are stated assumptions, not measurements. Every derived number moves proportionally if you disagree with them.

**There is no true alert-disposition ground truth.** I know which transactions were laundering, not which alerts a human would have escalated. "Productive alert" means "contained a genuinely suspicious transaction", which is an upper bound on what a real analyst would find.

**Sub-one-analyst figures are arithmetic, not staffing.** No AML function can be staffed at 1.2 people regardless of queue mathematics — quality assurance, four-eyes review, escalation and holiday cover impose a floor unrelated to alert volume.

___

# 10. Results Comparison <a name="results-comparison"></a>

| Approach | Alerts | Analysts | Detection | Productivity |
|---|---|---|---|---|
| Rules only — full queue | 108,843 | 109.5 | 26.4% | 1.37% |
| Rules + triage — 25% capacity | 27,210 | 27.4 | 23.0% | 4.77% |
| Hybrid — 25% capacity | 27,210 | 27.4 | 77.6% | 16.09% |
| Rules only — 20% detection target | 84,627 | 84.5 | 20% | — |
| Rules + triage — 20% detection target | 1,210 | 1.2 | 20% | — |

All rows scored on the same held-out test period, in account-day units.

### Summary of the three levers

| | What it changes | Cost | Strengths | Limitations |
|---|---|---|---|---|
| **Rule tuning** | Which alerts exist | Analyst review of scenarios | Transparent, explainable to a regulator, no model risk | Blind to patterns nobody wrote a rule for |
| **Model triage** | The *order* alerts are worked | One model, retrained periodically | Large capacity savings; no rule switched off | Cannot detect anything the rules missed |
| **Hybrid detection** | Which accounts are looked at | Model plus governance overhead | Only option that raises the ceiling | Requires explainability for regulators |

**Triage and detection are different products.** Triage makes the existing program cheaper and is comparatively easy to approve. Hybrid detection makes it *better* but requires a bank to defend a model-driven alert. The ordering in this project reflects the order most institutions attempt them.

___

# 11. Growth & Next Steps <a name="growth-next-steps"></a>

* **A case management layer.** Synthesising analyst assignment, investigation duration and queue backlogs would let the dashboard show SLA breaches and case aging — the measures AML leadership reviews most often.
* **Cost-based thresholds.** Replacing "alerts per analyst per day" with currency — expected laundering value detected against investigation cost — so the operating point is chosen on money.
* **Network-level detection.** Laundering is a property of a *group* of accounts. Graph features, or a graph neural network, would model that directly instead of approximating it with counterparty counts.
* **Validation on a second dataset.** Repeating the pipeline on the IBM AML dataset, which uses different generation logic, would separate genuine method from artefacts of SAML-D.
* **Rule retirement simulation.** The marginal value analysis suggests two scenarios could be retired or rebuilt. Quantifying the freed capacity is a concrete deliverable a bank would act on.

This project demonstrates that in financial crime detection the highest-leverage question is rarely *"how accurate is the model?"* It is **"where is my analyst capacity going, and what am I still unable to see?"** — and answering that takes an operational pipeline, not just a classifier.
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
