# From Subcription prediction to smart marketing

#EDA before doing the tableau dashboard

import pandas as pd

df = pd.read_csv("bank-full.csv", sep=";")

df["Subscribed"] = (df["y"] == "yes").astype(int)

#For each job category, how many customers said yes and how many said no?
job_analysis = pd.crosstab(
    df["job"],
    df["y"],
    margins=True
)

print(job_analysis)

#For each job category, calculate the percentage that said no and yes
job_subscription_rate = pd.crosstab(
    df["job"],
    df["y"],
    normalize="index"
) * 100

print("\nSubscription rate by job:")
print(job_subscription_rate.round(2))

#Does subscription likeliness vary across customer age groups?
age_bins = [17, 29, 39, 49, 59, 69, 100]

age_labels = [
    "18-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+"
]

df["age_group"] = pd.cut(
    df["age"],
    bins=age_bins,
    labels=age_labels
)

age_analysis = pd.crosstab(
    df["age_group"],
    df["y"],
    margins=True
)

print("\nAge group counts:")
print(age_analysis)

age_subscription_rate = pd.crosstab(
    df["age_group"],
    df["y"],
    normalize="index"
) * 100

print("\nSubscription rate by age group:")
print(age_subscription_rate.round(2))

#Do customers with different account balances show different term-deposit subscription behaviour?
balance_analysis = df.groupby("y")["balance"].agg(
    ["count", "mean", "median", "min", "max"]
)

print("\nBalance analysis by subscription:")
print(balance_analysis.round(2))

#How does subscription rate change across different balance levels?
df["balance_group"] = pd.qcut(
    df["balance"],
    q=5,
    duplicates="drop"
)

balance_group_analysis = df.groupby(
    "balance_group",
    observed=True
).agg(
    customers=("y", "size"),
    subscribers=("Subscribed", "sum"),
    subscription_rate=("Subscribed", "mean"),
    median_balance=("balance", "median")
)

balance_group_analysis["subscription_rate"] *= 100

print("\nBalance groups and subscription:")
print(balance_group_analysis.round(2))

#Customer Profiling
customer_variables = [
    "marital",
    "education",
    "housing",
    "loan",
    "default"
]

for variable in customer_variables:

    analysis = df.groupby(variable).agg(
        customers=("y", "size"),
        subscribers=("Subscribed", "sum"),
        subscription_rate=("Subscribed", "mean")
    )

    analysis["subscription_rate"] *= 100

    print(f"\nSubscription analysis by {variable}:")
    print(analysis.round(2))

#Does overall debt exposure relate to term-deposit subscription?
def create_loan_profile(row):
    if row["housing"] == "no" and row["loan"] == "no":
        return "No Loans"
    elif row["housing"] == "yes" and row["loan"] == "no":
        return "Housing Loan Only"
    elif row["housing"] == "no" and row["loan"] == "yes":
        return "Personal Loan Only"
    else:
        return "Both Loans"

df["loan_profile"] = df.apply(create_loan_profile, axis=1)

loan_profile_analysis = df.groupby("loan_profile").agg(
    customers=("y", "size"),
    subscribers=("Subscribed", "sum"),
    subscription_rate=("Subscribed", "mean")
)

loan_profile_analysis["subscription_rate"] *= 100

print("\nLoan profile analysis:")
print(loan_profile_analysis.round(2))

#Campaign Behaviour
#What campaign characteristics are associated with customer response?

campaign_variables = [
    "contact",
    "month"
]

for variable in campaign_variables:

    analysis = df.groupby(variable).agg(
        records=("y", "size"),
        subscribers=("Subscribed", "sum"),
        subscription_rate=("Subscribed", "mean")
    )

    analysis["subscription_rate"] *= 100

    print(f"\nSubscription analysis by {variable}:")
    print(analysis.round(2))

#How much contact is too much?
campaign_analysis = df.groupby("campaign").agg(
    records=("y", "size"),
    subscribers=("Subscribed", "sum"),
    subscription_rate=("Subscribed", "mean")
)

campaign_analysis["subscription_rate"] *= 100

print("\nSubscription analysis by no. of Campaign Contacts:")
print(campaign_analysis.round(2))

def contact_intensity(campaign):
    if campaign == 1:
        return "1 - First Contact"
    elif campaign <= 3:
        return "2 - Light Follow-up"
    elif campaign <= 5:
        return "3 - Moderate Follow-up"
    elif campaign <= 10:
        return "4 - High Follow-up"
    else:
        return "5 - Very High Follow-up"

df["contact_intensity"] = df["campaign"].apply(contact_intensity)

contact_intensity_analysis = df.groupby("contact_intensity").agg(
    records=("y", "size"),
    subscribers=("Subscribed", "sum"),
    subscription_rate=("Subscribed", "mean"),
    avg_contacts=("campaign", "mean")
)

contact_intensity_analysis["subscription_rate"] *= 100

print("\nContact intensity analysis:")
print(contact_intensity_analysis.round(2))

#Does previous campaign experience relate to the likelihood of subscribing in the current campaign?
poutcome_analysis = df.groupby("poutcome").agg(
    records=("y", "size"),
    subscribers=("Subscribed", "sum"),
    subscription_rate=("Subscribed", "mean")
)

poutcome_analysis["subscription_rate"] *= 100

print("\nPrevious campaign outcome analysis:")
print(poutcome_analysis.round(2))

#Were the customers previously contacted?
df["previously_contacted"] = df["pdays"] != -1

previous_contact_analysis = df.groupby("previously_contacted").agg(
    records=("y", "size"),
    subscribers=("Subscribed", "sum"),
    subscription_rate=("Subscribed", "mean"),
    avg_previous_contacts=("previous", "mean")
)

previous_contact_analysis["subscription_rate"] *= 100

print("\nPreviously contacted analysis:")
print(previous_contact_analysis.round(2))

print("\nPrevious contact vs previous outcome:")
print(
    pd.crosstab(
        df["previously_contacted"],
        df["poutcome"]
    )
)

def previous_campaign_status(row):
    if row["pdays"] == -1:
        return "No Previous Contact"
    elif row["poutcome"] == "success":
        return "Previous Success"
    elif row["poutcome"] == "failure":
        return "Previous Failure"
    else:
        return "Previous Other Outcome"

df["previous_campaign_status"] = df.apply(
    previous_campaign_status,
    axis=1
)

previous_status_analysis = df.groupby(
    "previous_campaign_status"
).agg(
    records=("y", "size"),
    subscribers=("Subscribed", "sum"),
    subscription_rate=("Subscribed", "mean")
)

previous_status_analysis["subscription_rate"] *= 100

print("\nPrevious campaign status analysis:")
print(previous_status_analysis.round(2))

previously_contacted_df = df[df["pdays"] != -1]

print("\nPdays distribution for previously contacted customers:")
print(
    previously_contacted_df["pdays"].describe()
)

pdays_by_subscription = previously_contacted_df.groupby("y")["pdays"].agg(
    ["count", "mean", "median", "min", "max"]
)

print("\nPdays by current subscription:")
print(pdays_by_subscription.round(2))

def recency_segment(pdays):
    if pdays <= 90:
        return "1 - Recent (≤90 days)"
    elif pdays <= 180:
        return "2 - Moderately Recent (91-180 days)"
    elif pdays <= 365:
        return "3 - Older Contact (181-365 days)"
    else:
        return "4 - Long Ago (>365 days)"

previously_contacted_df = previously_contacted_df.copy()

previously_contacted_df["recency_segment"] = (
    previously_contacted_df["pdays"].apply(recency_segment)
)

recency_analysis = previously_contacted_df.groupby(
    "recency_segment"
).agg(
    records=("y", "size"),
    subscribers=("Subscribed", "sum"),
    subscription_rate=("Subscribed", "mean"),
    median_days=("pdays", "median")
)

recency_analysis["subscription_rate"] *= 100

print("\nPrevious contact recency analysis:")
print(recency_analysis.round(2))

recency_outcome = pd.crosstab(
    previously_contacted_df["recency_segment"],
    previously_contacted_df["poutcome"],
    normalize="index"
) * 100

print("\nPrevious outcome distribution within each recency segment:")
print(recency_outcome.round(2))

recency_outcome_rate = previously_contacted_df.groupby(
    ["recency_segment", "poutcome"]
).agg(
    records=("y", "size"),
    subscription_rate=("Subscribed", "mean")
)

recency_outcome_rate["subscription_rate"] *= 100

print("\nSubscription rate by recency and previous outcome:")
print(recency_outcome_rate.round(2))

duration_analysis = df.groupby("y")["duration"].agg(
    ["count", "mean", "median", "min", "max"]
)

print("\nCall duration by subscription outcome:")
print(duration_analysis.round(2))

def duration_segment(duration):
    if duration < 60:
        return "1 - Under 1 min"
    elif duration < 180:
        return "2 - 1 to 3 min"
    elif duration < 300:
        return "3 - 3 to 5 min"
    elif duration < 600:
        return "4 - 5 to 10 min"
    else:
        return "5 - Over 10 min"

df["duration_segment"] = df["duration"].apply(duration_segment)

duration_segment_analysis = df.groupby(
    "duration_segment"
).agg(
    records=("y", "size"),
    subscribers=("Subscribed", "sum"),
    subscription_rate=("Subscribed", "mean"),
    median_duration=("duration", "median")
)

duration_segment_analysis["subscription_rate"] *= 100

print("\nCall duration segment analysis:")
print(duration_segment_analysis.round(2))

# CAMPAIGN TIMING - MONTH
month_order = [
    "jan", "feb", "mar", "apr",
    "may", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec"
]

month_analysis = df.groupby("month").agg(
    records=("y", "size"),
    subscribers=("Subscribed", "sum"),
    subscription_rate=("Subscribed", "mean")
)

month_analysis["subscription_rate"] *= 100

month_analysis = month_analysis.reindex(month_order)

print("\nMonthly Campaign Performance:")
print(month_analysis.round(2))

# CAMPAIGN TIMING - DAY
day_analysis = df.groupby("day").agg(
    records=("y", "size"),
    subscribers=("Subscribed", "sum"),
    subscription_rate=("Subscribed", "mean")
)

day_analysis["subscription_rate"] *= 100

print("\nDaily Campaign Performance:")
print(day_analysis.round(2))
