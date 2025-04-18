from scipy.stats import vonmises_line
import numpy as np
import torch

'''
    Foot periodicity using Vonmise distribution.
'''
def limit_input_vonmise_cdf(x, loc, kappa):
    # Ranges: x in [0, 1]
    # assert np.min(x) >= 0.0 and np.max(x) <= 1.0
    return vonmises_line.cdf(x = 2*np.pi*x, loc = 2*np.pi*loc, kappa = kappa)


def prob_phase_indicator(phi : np.ndarray, start : np.ndarray, end : np.ndarray, kappa : np.ndarray):
    # P(I = 1)
    # start, end, kappa: [num_envs, 1]; phi: [num_envs, num_feet]
    temp = np.stack([start, end, start - 1.0, end - 1.0, start + 1.0, end + 1.0], axis = 0) # For faster computation on vonmise cdf, shape [6, num_envs, 1]
    Ps = limit_input_vonmise_cdf(phi[None], temp, kappa[None])
    return Ps[0] * (1 - Ps[1]) + Ps[2] * (1 - Ps[3]) + Ps[4] * (1 - Ps[5])


def E_phase_indicator(phi, start, end, kappa):
    # E_I = 1 * P(I = 1) + 0 * P(I = 0)
    return prob_phase_indicator(phi, start, end, kappa)


def E_periodic_property(phi, duty_factor, kappa, c_swing, c_stance):
    return c_swing * E_phase_indicator(phi, np.zeros_like(duty_factor), 1.0 - duty_factor, kappa) + c_stance * E_phase_indicator(phi, 1.0 - duty_factor, np.ones_like(duty_factor), kappa)


def _get_foot_phis(i, period, commands, foot_thetas):
    phi = i / period
    sign_indicator = torch.ones_like(commands[..., 0:1])
    sign = torch.where(commands[..., 0:1] >= 0, sign_indicator, -sign_indicator)
    return torch.abs(((phi.unsqueeze(-1) + foot_thetas) * sign) % sign)



def test():
    E_C_frcs = []
    E_C_spds = []

    kappa_ts = torch.ones(1) * 16
    duty_factor_ts = torch.ones(1) * 0.37
    period_ts = torch.ones(1) * 250
    init_foot_thetas_ts = torch.as_tensor([0, 0.5, 0.5, 0]).unsqueeze(0) # Trotting
    commands_ts = torch.ones((1, 1))
    for i in range(250):
        foot_phis = _get_foot_phis(i, period_ts, commands_ts, init_foot_thetas_ts).cpu().numpy()
        duty_factor = duty_factor_ts.cpu().numpy()[..., None] # [num_envs, 1]
        kappa = kappa_ts.cpu().numpy()[..., None] # [num_envs, 1]

        E_C_frc = E_periodic_property(foot_phis, duty_factor, kappa, -1, 0)
        E_C_frc = torch.from_numpy(E_C_frc)
        E_C_frcs.append(E_C_frc)
        E_C_spd = E_periodic_property(foot_phis, duty_factor, kappa, 0, -1)
        E_C_spd = torch.from_numpy(E_C_spd)
        E_C_spds.append(E_C_spd)

    
    E_C_frcs = np.concatenate(E_C_frcs, axis = 0)
    E_C_spds = np.concatenate(E_C_spds, axis = 0)


    # start_idx = 0
    # end_idx = E_C_frcs.shape[0]
    # foot_seq = ['FL', 'FR', 'RL', 'RR']
    # import matplotlib.pyplot as plt
    # plt.figure(figsize = (12, 4))
    # for i, ft in enumerate(foot_seq):
    #     plt.plot(np.arange(start_idx, end_idx), E_C_frcs[start_idx:end_idx, i], label = ft)
    # plt.legend()
    # plt.savefig('E_C_frcs.png')
    # plt.close()

    # E_C_spds = np.concatenate(E_C_spds, axis = 0)
    # print(E_C_spds.shape)

    # start_idx = 0
    # end_idx = E_C_spds.shape[0]
    # foot_seq = ['FL', 'FR', 'RL', 'RR']
    # import matplotlib.pyplot as plt
    # plt.figure(figsize = (12, 4))
    # for i, ft in enumerate(foot_seq):
    #     plt.plot(np.arange(start_idx, end_idx), E_C_spds[start_idx:end_idx, i], label = ft)
    # plt.legend()
    # plt.savefig('E_C_spds.png')
    # plt.close()

    import matplotlib.pyplot as plt
    xs = np.linspace(0, 1, 250)
    plt.plot(xs, E_C_frcs[:, 0], label = r'$C_F(\cdot)$')
    plt.plot(xs, E_C_spds[:, 0], label = r'$C_v(\cdot)$')
    plt.axvline(x=1 - duty_factor.item(), color='r', linestyle='--')
    plt.legend(loc='right', fontsize=20)
    plt.text(x=0.25, y=-0.2, s='Swing', fontsize=20)
    plt.text(x=0.75, y=-0.2, s='Stance', fontsize=20)
    # plt.xlabel(r'$\phi$')
    # plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig('temporal.png')
    plt.close()


if __name__ == '__main__':
    test()