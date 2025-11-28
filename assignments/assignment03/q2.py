import copy

# Defining the MDP environment based on the assignment story.
STATES = ["Top of the Hill", "Rolling down the Hill", "Bottom of the Hill"]
ACTIONS = ["Drive", "Don't Drive"]

# These are the base rewards for being in a state.
# The driving cost (-1) is applied dynamically inside the loops later.
REWARDS = {
    "Top of the Hill": 3,
    "Rolling down the Hill": 1,
    "Bottom of the Hill": 1
}

# Transition probabilities formatted as: State -> Action -> List of (Probability, Next_State)
TRANSITION = {
    "Top of the Hill": {
        "Drive": [(0.9, "Top of the Hill"), (0.1, "Rolling down the Hill")],
        "Don't Drive": [(0.7, "Top of the Hill"), (0.3, "Rolling down the Hill")]
    },
    "Rolling down the Hill": {
        "Drive": [(0.3, "Top of the Hill"), (0.6, "Rolling down the Hill"), (0.1, "Bottom of the Hill")],
        "Don't Drive": [(1.0, "Bottom of the Hill")]
    },
    "Bottom of the Hill": {
        "Drive": [(0.6, "Top of the Hill"), (0.4, "Bottom of the Hill")],
        "Don't Drive": [(1.0, "Bottom of the Hill")]
    }
}

DRIVING_COST = -1
MAX_ITERS = 60
DISCOUNT_FACTOR = 0.8

def value_iteration():
    # Start with all values at zero
    VALUES = {
        "Top of the Hill": 0.0,
        "Rolling down the Hill": 0.0,
        "Bottom of the Hill": 0.0
    }

    policy = {}

    for iter in range(MAX_ITERS):
        # CRITICAL: Creating a deepcopy of the current values.
        # As I noted in the report, this ensures we calculate the next iteration (k+1)
        # using ONLY the values from the previous iteration (k), preventing in-place updates.
        current_values = copy.deepcopy(VALUES)

        for state in STATES:
            max_action_value = -float('inf')

            for action in ACTIONS:
                # Base reward comes from the state, but we subtract 1 if the action is 'Drive'.
                immediate_reward = REWARDS[state]
                if action == "Drive":
                    immediate_reward += DRIVING_COST

                P = TRANSITION[state][action]
                action_value = 0

                # Bellman Update
                for prob, state_hat in P:
                    action_value += prob * (immediate_reward + DISCOUNT_FACTOR * current_values[state_hat])

                if action_value > max_action_value:
                    max_action_value = action_value
                    policy[state] = action

            # Update the state value after checking all possible actions
            VALUES[state] = max_action_value

        print(f"[{iter + 1}]:")
        print(" VALUES:")
        for k, v in VALUES.items():
            print(f"    {k}: {v}")

        print(" POLICY:")
        for k, v in policy.items():
            print(f"    {k}: {v}")

        print()


def policy_iteration():
    VALUES = {
        "Top of the Hill": 0.0,
        "Rolling down the Hill": 0.0,
        "Bottom of the Hill": 0.0
    }

    # Initializing with the "Don't Drive" policy
    policy = {s: "Don't Drive" for s in STATES}

    outer_iter = 0
    theta = 1e-6


    while True:
        # ------------------------------------------------
        # STEP 1: POLICY EVALUATION
        # I need to calculate the true value of the current policy (V_pi).
        # I loop here until the values stabilize for the *fixed* policy.
        # ------------------------------------------------
        while True:
            delta = 0
            current_values = copy.deepcopy(VALUES)

            for state in STATES:
                action = policy[state] # We only look at the action dictated by the current policy

                immediate_reward = REWARDS[state]
                if action == "Drive":
                    immediate_reward += DRIVING_COST

                P = TRANSITION[state][action]
                new_value = 0

                # Calculate the value for this specific state-action pair
                for prob, state_hat in P:
                    new_value += prob * (immediate_reward + DISCOUNT_FACTOR * current_values[state_hat])

                # Tracking delta to check if the evaluation has converged
                delta = max(delta, abs(new_value - VALUES[state]))
                VALUES[state] = new_value

            # If values stopped changing for this policy, proceed to improvement
            if delta < theta:
                break

        # ------------------------------------------------
        # STEP 2: POLICY IMPROVEMENT
        # Now I check if there is a greedy action that is better than the current policy.
        # ------------------------------------------------
        updated_policy = {}
        policy_stable = True # Flag to track stability, as mentioned in the report challenges.

        for state in STATES:
            old_action = policy[state]
            max_action_value = -float('inf')
            best_action = None

            # Iterate through ALL actions to find the argmax Q(s,a)
            for action in ACTIONS:
                immediate_reward = REWARDS[state]
                if action == "Drive":
                    immediate_reward += DRIVING_COST

                P = TRANSITION[state][action]
                action_value = 0
                for prob, state_hat in P:
                    # We use the VALUES that converged in Step 1
                    action_value += prob * (immediate_reward + DISCOUNT_FACTOR * VALUES[state_hat])

                if action_value > max_action_value:
                    max_action_value = action_value
                    best_action = action

            updated_policy[state] = best_action

            # If the best action is different from what we were doing, the policy isn't optimal yet.
            if best_action != old_action:
                policy_stable = False

        policy = updated_policy
        outer_iter += 1

        print(f"[{outer_iter}]:")
        print(" VALUES:")
        for k, v in VALUES.items():
            print(f"    {k}: {v}")

        print(" POLICY:")
        for k, v in policy.items():
            print(f"    {k}: {v}")

        print()

        if policy_stable:
            print("Optimal policy found!")
            break

if __name__ == "__main__":

    print("*********VALUE ITERATION*********")
    value_iteration()

    print("*********POLICY ITERATION*********")
    policy_iteration()