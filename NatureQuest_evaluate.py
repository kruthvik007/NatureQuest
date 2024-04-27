import numpy as np
import pandas as pd

# Load data
data = pd.read_csv("NatureQuestData.csv")

def thompson_sampling_with_details(num_options, user_age, user_budget, user_duration, user_state_preference):
    user_age_group = None
    if user_age < 15:
        user_age_group = "Less than 15"
    elif user_age >= 15 and user_age <= 20:
        user_age_group = "15-20"
    elif user_age > 60:
        user_age_group = "More than 60"
    else:
        user_age_group = f"{((user_age-1)//5)*5+1}-{((user_age-1)//5)*5+5}"

    filtered_data = data.copy()

    if user_age_group:
        filtered_data = filtered_data[filtered_data["Age group"] == user_age_group]
    if user_budget != np.inf:
        filtered_data = filtered_data[filtered_data["Budget"].astype(float) <= user_budget]
    if user_duration != np.inf:
        filtered_data = filtered_data[filtered_data["Duration"].astype(int) <= user_duration]
    if user_state_preference:
        filtered_data = filtered_data[filtered_data["State"].str.lower() == user_state_preference]

    num_arms = len(filtered_data)
    arm_successes = np.ones(num_arms)
    arm_failures = np.ones(num_arms)

    num_iterations = 1000
    if not filtered_data.empty:
        for i in range(num_iterations):
            theta_samples = np.random.beta(arm_successes, arm_failures)
            chosen_arm = np.argmax(theta_samples)
            reward = filtered_data.iloc[chosen_arm]["Rating"]
            if reward > 3:
                arm_successes[chosen_arm] += 1
            else:
                arm_failures[chosen_arm] += 1

    if filtered_data.empty:
        return []
    else:
        theta_estimates = arm_successes / (arm_successes + arm_failures)
        sorted_arms = np.argsort(theta_estimates)[::-1]
        recommendations = []
        for i in range(min(num_options, len(sorted_arms))):
            option = filtered_data.iloc[sorted_arms[i]]
            recommendation = {
                "Location": option["Location"],
                "State": option["State"],
                "Activity": option["Activity"],
                "Duration": option["Duration"],
                "Budget": option["Budget"],
                "Rating": option["Rating"]
            }
            recommendations.append(recommendation)
        return recommendations

def recommend_places_with_details(num_recommendations, user_age, user_budget, user_duration, user_state_preference):
    user_age_group = None
    if user_age < 15:
        user_age_group = "Less than 15"
    elif user_age >= 15 and user_age <= 20:
        user_age_group = "15-20"
    elif user_age > 60:
        user_age_group = "More than 60"
    else:
        user_age_group = f"{((user_age-1)//5)*5+1}-{((user_age-1)//5)*5+5}"

    filtered_data = data.copy()

    if user_age_group:
        filtered_data = filtered_data[filtered_data["Age group"] == user_age_group]
    if user_budget != np.inf:
        filtered_data = filtered_data[filtered_data["Budget"].astype(float) <= user_budget]
    if user_duration != np.inf:
        filtered_data = filtered_data[filtered_data["Duration"].astype(int) <= user_duration]
    if user_state_preference:
        filtered_data = filtered_data[filtered_data["State"].str.lower() == user_state_preference]

    if not filtered_data.empty:
        rating_weight = 0.1
        duration_weight = 0.5
        budget_weight = 0.4

        max_duration = filtered_data["Duration"].max()
        max_budget = filtered_data["Budget"].max()

        filtered_data["Score"] = (filtered_data["Rating"] * rating_weight) + \
                                  ((max_duration - filtered_data["Duration"]) / max_duration) * duration_weight + \
                                  ((max_budget - filtered_data["Budget"]) / max_budget) * budget_weight

        sorted_data = filtered_data.sort_values(by="Score", ascending=False).head(num_recommendations)

        recommendations = []
        for _, option in sorted_data.iterrows():
            recommendation = {
                "Location": option["Location"],
                "State": option["State"],
                "Activity": option["Activity"],
                "Duration": option["Duration"],
                "Budget": option["Budget"],
                "Rating": option["Rating"]
            }
            recommendations.append(recommendation)
        return recommendations
    else:
        return []

# Example usage
user_age = 25
user_budget = 1000
user_duration = 5
user_state_preference = "california"

thompson_recommendations = thompson_sampling_with_details(5, user_age, user_budget, user_duration, user_state_preference)
print("Thompson Sampling Recommendations:")
for recommendation in thompson_recommendations:
    print(recommendation)

cf_recommendations = recommend_places_with_details(5, user_age, user_budget, user_duration, user_state_preference)
print("MultiFactor CF Recommendations:")
for recommendation in cf_recommendations:
    print(recommendation)


def calculate_map(recommendations, ground_truth):
    average_precision = 0
    num_relevant_recommendations = 0
    for i, recommendation in enumerate(recommendations):
        if recommendation["Location"] in ground_truth:
            num_relevant_recommendations += 1
            precision_at_i = num_relevant_recommendations / (i + 1)
            average_precision += precision_at_i
    if num_relevant_recommendations == 0:
        return 0
    else:
        return average_precision / num_relevant_recommendations


def calculate_ndcg(recommendations, ground_truth, k):
    dcg = 0
    for i, recommendation in enumerate(recommendations[:k]):
        if recommendation["Location"] in ground_truth:
            relevance = 1
        else:
            relevance = 0
        dcg += (2 ** relevance - 1) / np.log2(i + 2)
    idcg = sum((2 ** 1 - 1) / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
    if idcg == 0:
        return 0
    else:
        return dcg / idcg

# Example usage
ground_truth = ["Houston", "Las Vegas", "Miami"]  # Example ground truth


# MAP
thompson_map = calculate_map(thompson_recommendations, ground_truth)
cf_map = calculate_map(cf_recommendations, ground_truth)
print("Thompson Sampling MAP:", thompson_map)
print("MultiFactor CF MAP:", cf_map)


# NDCG
thompson_ndcg = calculate_ndcg(thompson_recommendations, ground_truth, 5)
cf_ndcg = calculate_ndcg(cf_recommendations, ground_truth, 5)
print("Thompson Sampling NDCG:", thompson_ndcg)
print("MultiFactor CF NDCG:", cf_ndcg)











