import pandas as pd
import numpy as np
import dqn_single_schedule as dqn

input("Press Enter to acknowledge and continue...")
print("Beginning Demo...")

dam = pd.read_parquet("..\\data-pull\\combined_data\\dam_data_2023_2026.parquet")


# 5. Verification Test
def test_input(state):
    dqn.q_net.eval()
    with dqn.torch.no_grad():
        inp = dqn.torch.FloatTensor(state).unsqueeze(0)
        test_env = dqn.BatteryEnv()
        action_mask = test_env.get_action_mask(state)
        q_vals = dqn.q_net(inp).numpy()[0]
        masked_q_vals = q_vals.copy()
        masked_q_vals[~action_mask] = float('-inf')
        chosen_action = int(masked_q_vals.argmax())
    return chosen_action, q_vals

print("Test: First 24 hours of 2026")
for hour in range(24):
    print("Selected Action: ", test_input(np.append(dam["lmp"][hour: hour + 24],0)))  # Expected: Unknown
    print("Price: ", dam["lmp"][hour])