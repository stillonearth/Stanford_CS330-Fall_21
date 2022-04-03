"""Run the Q-network on the environment for fixed steps.

Complete the code marked TODO."""
import numpy as np # pylint: disable=unused-import
import torch # pylint: disable=unused-import

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_episode(
    env,
    q_net, # pylint: disable=unused-argument
    steps_per_episode,
):
    """Runs the current policy on the given environment.

    Args:
        env (gym): environment to generate the state transition
        q_net (QNetwork): Q-Network used for computing the next action
        steps_per_episode (int): number of steps to run the policy for

    Returns:
        episode_experience (list): list containing the transitions
                        (state, action, reward, next_state, goal_state)
        episodic_return (float): reward collected during the episode
        succeeded (bool): DQN succeeded to reach the goal state or not
    """

    # list for recording what happened in the episode
    episode_experience = []
    succeeded = False
    episodic_return = 0.0

    # reset the environment to get the initial state
    state, goal_state = env.reset() # pylint: disable=unused-variable

    for _ in range(steps_per_episode):

        # print(state)
        # print(goal_state)

        # ======================== TODO modify code ========================

        # append goal state to input, and prepare for feeding to the q-network

        x = np.hstack([state, goal_state])
        x = torch.from_numpy(x).float()

        # print(x.shape)

        # forward pass to find action
        q_net.eval()
        with torch.no_grad():
            action_values = q_net(x)
        q_net.train()
            
        # greedy action
        action = np.argmax(action_values.cpu().data.numpy())

        # take action, use env.step
        (next_state, reward, done, info) = env.step(action)

        # add transition to episode_experience as a tuple of
        # (state, action, reward, next_state, goal)

        episode_experience.append(
            (state, action, reward, next_state, goal_state)
        )

        # update episodic return
        episodic_return += reward

        # update state
        state = next_state

        # update succeeded bool from the info returned by env.step
        succeeded = succeeded or info['successful_this_state']

        # break the episode if done=True
        if done:
            break

        # ========================      END TODO       ========================

    return episode_experience, episodic_return, succeeded
