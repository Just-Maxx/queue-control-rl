def policy_match_ratio(policy_a, policy_b, states):
    matches = 0

    for state in states:
        if policy_a[state] == policy_b[state]:
            matches += 1

    return matches / len(states)


def q_to_value(Q, states, actions):
    V = {}

    for state in states:
        V[state] = max(Q[(state, action)] for action in actions)

    return V


def value_sse(V_est, V_ref, states):
    error_sum = 0.0

    for state in states:
        error = V_est[state] - V_ref[state]
        error_sum += error ** 2

    return float(error_sum)


def discounted_episode_return(episode, gamma):
    total_return = 0.0
    discount = 1.0

    for _, _, reward, _ in episode:
        total_return += discount * reward
        discount *= gamma

    return total_return
