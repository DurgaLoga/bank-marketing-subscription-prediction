# Bank Marketing Subscription Prediction
### From Historical Customer Data to Smarter Marketing Decisions

**Python | Exploratory Data Analysis | Logistic Regression | Machine Learning | Tableau | Customer Analytics**

An end-to-end academic data mining project exploring how customer characteristics, previous campaign outcomes, and marketing interactions can support more effective term-deposit campaign targeting.

The project combines data auditing, exploratory analysis, predictive modeling, and business intelligence to examine an important business question:

> **Can banks use historical customer information to identify promising prospects before making a marketing call?**

**Project Type:** Academic Team Project  
**My Role:** Project Lead | Data Auditing | Exploratory Data Analysis | Tableau Visualization | Logistic Regression Baseline Modeling  
**Dataset:** Portuguese Bank Marketing Dataset  
**Status:** Completed Academic Project | Research and Portfolio Demonstration

---

## 1. Business Problem

Banks conduct outbound marketing campaigns to promote financial products such as term deposits.

However, contacting customers requires time, operational resources, and marketing expenditure. Not every customer contacted will subscribe.

The business challenge is to use historical information to support better customer prioritization.

### Project Objectives

- Understand the characteristics of customers who subscribe to term deposits.
- Identify patterns associated with successful marketing campaigns.
- Analyze customer and campaign behavior.
- Develop predictive classification models.
- Evaluate models using metrics appropriate for imbalanced data.
- Explore how predictions could support operational targeting decisions.

The objective is to support better marketing decisions, not to assume that a prediction guarantees a subscription.

---

## 2. Dataset Overview

The project uses the Portuguese Bank Marketing dataset.

| Attribute | Description |
|---|---|
| Total Records | 45,211 |
| Original Predictors | 16 |
| Target Variable | `y` — Term Deposit Subscription |
| Positive Class | 5,289 subscriptions |
| Negative Class | 39,922 non-subscriptions |
| Subscription Rate | 11.7% |
| Data Quality | No null values or duplicate records reported |

### Key Data Categories

**Customer Information**
- Age
- Job
- Marital status
- Education
- Account balance
- Housing loan
- Personal loan

**Campaign Information**
- Contact method
- Contact month and day
- Number of campaign contacts
- Previous campaign outcomes
- Previous contact history

**Target Variable**

`y = yes` indicates that the customer subscribed to a term deposit.

The dataset is imbalanced, with approximately 88.3% of observations belonging to the non-subscription class.

This makes model evaluation using accuracy alone potentially misleading.

---

## 3. Project Workflow

The overall team project followed an end-to-end analytical process:

**Business Understanding → Data Auditing → Exploratory Analysis → Feature Assessment → Predictive Modeling → Model Evaluation → Business Recommendations**

### My Individual Contributions

I led the academic project and personally developed the following analytical components:

| Area | Contribution |
|---|---|
| Project Leadership | Coordinated the team-based analytical project |
| Data Auditing | Developed Python scripts to inspect dataset structure, quality, and target distribution |
| Exploratory Data Analysis | Analyzed customer demographics, financial characteristics, and campaign behavior using Python |
| Business Intelligence | Developed Tableau visualizations to communicate customer and campaign insights |
| Baseline Modeling | Implemented logistic regression classification and a majority-class baseline |
| Model Evaluation | Used stratified cross-validation and Matthews correlation coefficient to assess baseline performance |

Other classification models and advanced experiments are presented as **team-level contributions**, not as individually implemented work.

---

## 4. Data Auditing and Quality Assessment

The initial analytical stage examined the reliability and structure of the dataset.

### Key Activities

- Reviewed dataset dimensions and column types.
- Examined numerical distributions.
- Checked duplicate records.
- Investigated categorical values recorded as `unknown`.
- Evaluated the target variable distribution.
- Identified special values requiring careful interpretation.

### Important Finding

The dataset contains no reported null values or duplicate records.

However, categorical `unknown` values require interpretation rather than automatic deletion.

Additionally, `pdays = -1` represents customers who were not previously contacted in an earlier campaign.

These distinctions are important when preparing features for modeling.

**Python implementation:** [`Code/data_audit.py`](Code/data_audit.py)

---

## 5. Exploratory Data Analysis

Exploratory analysis investigated how customer characteristics and campaign activity were associated with subscription outcomes.

### Customer Analysis

Examined subscription patterns across:

- Age groups
- Employment categories
- Marital status
- Education
- Account balance
- Housing and personal loan status

### Campaign Analysis

Investigated:

- Contact frequency
- Previous campaign outcomes
- Contact timing
- Campaign response patterns
- Call duration

### Selected Historical Findings

| Finding | Observed Result |
|---|---|
| Overall Subscription Rate | 11.7% |
| Previous Campaign Success | Approximately 64.7% subscription rate |
| Student Customers | Approximately 28.7% subscription rate |
| Retired Customers | Approximately 22.8% subscription rate |
| First Campaign Contact | Approximately 14.6% subscription rate |
| 11+ Campaign Contacts | Approximately 3.9% subscription rate |

These findings describe historical associations, not causal relationships.

**Python implementation:** [`Code/exploratory_data_analysis.py`](Code/exploratory_data_analysis.py)

---

## 6. Interactive Tableau Dashboard

An interactive Tableau dashboard was developed to communicate key customer and campaign insights.

### Dashboard Focus

- Customer segmentation
- Subscription behavior
- Campaign effectiveness
- Customer financial characteristics
- Previous campaign outcomes
- Marketing performance patterns

### Explore the Dashboard

**[View Interactive Bank Marketing Campaign Dashboard](https://public.tableau.com/app/profile/durga.s1183/viz/Bank-TermDepositCampaignDashboard/CampaignDashboard)**

The dashboard complements the Python analysis by allowing users to explore historical patterns interactively.

---

## 7. Predictive Modeling

The team evaluated four classification approaches:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

### My Individual Modeling Work

I implemented logistic regression as an interpretable baseline classifier.

The workflow included:

- Numerical feature scaling
- Categorical one-hot encoding
- Stratified five-fold cross-validation
- Majority-class baseline comparison
- Matthews correlation coefficient evaluation
- Comparison of models with and without call duration

**Python implementation:** [`Code/logistic_regression.py`](Code/logistic_regression.py)

### Why Call Duration Matters

Call duration is only known after a marketing conversation begins.

Therefore, it may be useful for understanding historical subscription outcomes but cannot be used to decide whom to contact before a call.

The project distinguishes between:

**Full-information modeling:** Includes call duration for diagnostic comparison.

**Pre-contact modeling:** Excludes call duration to better reflect information available before customer contact.

This distinction demonstrates why operational feature availability matters as much as predictive performance.

---

## 8. Team-Level Model Comparison

The broader project compared four classification algorithms.

The following results summarize the team's reported tuned models using five-fold cross-validation without call duration.

| Algorithm | Mean MCC |
|---|---:|
| Logistic Regression | 0.302 |
| Decision Tree | 0.317 |
| Random Forest | 0.331 |
| XGBoost | 0.355 |

**Key observation:** XGBoost achieved the highest MCC in the controlled comparison.

The team subsequently investigated feature engineering, class imbalance treatment, and operational feature availability.

These experiments represent the collective project findings.

---

## 9. Final Team Model and Evaluation

The team's final reported configuration used:

- Tuned XGBoost
- Positive-class weight of 4
- Fourteen original pre-contact predictors
- Exclusion of call duration and current-campaign contact count
- Five-fold stratified cross-validation

### Reported Historical Performance

| Metric | Result |
|---|---:|
| Matthews Correlation Coefficient | 0.4211 |
| Precision | 46.9% |
| Recall | 51.6% |
| F1 Score | 0.4914 |

Historical out-of-fold ranking showed that the top-ranked 20% of customers contained approximately 62.2% of observed subscribers.

These results are retrospective and should not be interpreted as guaranteed performance on future campaigns.

### Important Limitation: Temporal Drift

Forward-time evaluation showed substantially weaker performance than shuffled cross-validation.

This indicates that historical patterns may not remain stable across future campaigns.

A production deployment would require:

- Validation on later campaign periods
- Monitoring for changing customer behavior
- Assessment of operational costs
- Prospective pilot testing
- Evaluation of fairness and customer impact

The project did not establish production readiness or verified financial ROI.

---

## 10. Business Recommendations

The analysis suggests several opportunities for improving marketing decisions.

**Customer Prioritization**

Use validated model rankings to support the selection of customers for marketing outreach.

**Campaign Efficiency**

Investigate contact frequency and previous campaign outcomes when developing targeting strategies.

**Operationally Realistic Predictions**

Only use customer information that is available at the time a targeting decision is made.

**Model Monitoring**

Track predictive performance and changing customer behavior across campaign periods.

**Controlled Business Testing**

Conduct future campaign pilots to measure whether model-assisted targeting improves outcomes compared with existing practices.

---

## 11. Technologies and Analytical Methods

| Category | Technologies / Methods |
|---|---|
| Programming | Python |
| Data Manipulation | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Tableau |
| Machine Learning | scikit-learn |
| Classification | Logistic Regression; team-level Decision Tree, Random Forest, XGBoost |
| Model Evaluation | Stratified Cross-Validation, MCC, Precision, Recall, F1 |
| Data Preparation | StandardScaler, OneHotEncoder |
| Business Domain | Customer Analytics, Marketing Analytics |

---

## 12. Repository Structure

```text
bank-marketing-subscription-prediction/
│
├── code/
│   ├── README.md
│   ├── data_audit.py
│   ├── exploratory_data_analysis.py
│   └── logistic_regression.py
│
├── documentation/
│   ├── README.md
│   ├── Bank Mkt Marketing Prediction.pdf
│   └── Bank_Marketing_Complete_Project_Report.pdf
│
├── .gitignore
└── README.md
```

The Python scripts document my individual analytical implementations.

The PDF documents provide a broader synthesis of the academic team's project.

The original dataset is not included in this repository.

---

## 13. Project Documentation

**[View Project Presentation](documentation/Bank%20Mkt%20Marketing%20Prediction.pdf)**

**[View Comprehensive Project Report](documentation/Bank_Marketing_Complete_Project_Report.pdf)**

Both documents summarize the overall team's analytical findings.

---

## 14. Project Ownership and Attribution

This project was completed collaboratively as part of an academic data mining assignment.

**My individual work included:**

- Project leadership
- Python-based data auditing
- Exploratory data analysis
- Tableau visualization
- Logistic regression baseline modeling

Additional algorithms, advanced modeling experiments, and final team-level findings are included for project completeness and are attributed to the collaborative team effort.

The repository is intended to demonstrate analytical methodology, business understanding, technical implementation, and responsible interpretation of predictive models.

---

## 15. Key Takeaways

This project reinforced five important analytical principles:

1. **Data understanding comes first.** Reliable analysis requires careful inspection of data quality and meaning.
2. **Business context matters.** A highly predictive feature may not be available when a real decision must be made.
3. **Accuracy is not enough.** Imbalanced classification requires more informative evaluation metrics.
4. **Historical performance is not future performance.** Temporal validation is essential before operational deployment.
5. **Analytics should support decisions.** Predictive modeling creates value only when integrated with realistic business processes.

**From prediction to smarter targeting — connecting data science with practical business decisions.**
