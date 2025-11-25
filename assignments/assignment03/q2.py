import copy


STATES = ["Top of the Hill", "Rolling down the Hill", "Bottom of the Hill"]
ACTIONS = ["Drive", "Don't Drive"]

REWARDS = {
    "Top of the Hill": 3,
    "Rolling down the Hill": 1,
    "Bottom of the Hill": 1
}

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

VALUES= {
    "Top of the Hill": 0.0,
    "Rolling down the Hill": 0.0,
    "Bottom of the Hill": 0.0
}

def value_iteration():
    policy = {}
    for iter in range(MAX_ITERS):
        current_values = copy.deepcopy(VALUES)
        for state in STATES:
            max_action_value = -float('inf')

            for action in ACTIONS:
                immediate_reward = REWARDS[state]
                if action == "Drive":
                    immediate_reward += DRIVING_COST

                P = TRANSITION[state][action]
                action_value = 0
                for prob, state_hat in P:
                    action_value += prob * (immediate_reward + DISCOUNT_FACTOR * current_values[state_hat])

                if action_value > max_action_value:
                    max_action_value = action_value
                    policy[state] = action

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
    policy = {
        s:"Don't Drive" for s in STATES
    }
    iterations = 0
    while True:
        # Value Iteration
        for _ in range(MAX_ITERS):
            current_values = copy.deepcopy(VALUES)

            for state in STATES:
                action = policy[state]
                immediate_reward = REWARDS[state]
                if action == "Drive":
                    immediate_reward += DRIVING_COST

                P = TRANSITION[state][action]
                new_value = 0
                for prob, state_hat in P:
                    new_value += prob * (immediate_reward + DISCOUNT_FACTOR * current_values[state_hat])

                VALUES[state] = new_value

        # Policy Improvement
        updated_policy = {}
        policy_stable = True
        for state in STATES:
            max_action_value = -float('inf')
            for action in ACTIONS:
                immediate_reward = REWARDS[state]
                if action == "Drive":
                    immediate_reward += DRIVING_COST

                P = TRANSITION[state][action]
                action_value = 0
                for prob, state_hat in P:
                    action_value += prob * (immediate_reward + DISCOUNT_FACTOR * VALUES[state_hat])

                if action_value > max_action_value:
                    max_action_value = action_value
                    updated_policy[state] = action

            if updated_policy[state] != policy[state]:
                policy_stable = False

        policy = updated_policy

        print(f"[{iterations}]:")
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

        iterations += 1

if __name__ == "__main__":

    print("*********VALUE ITERATION*********")
    value_iteration()

    print("*********POLICY ITERATION*********")
    policy_iteration()

