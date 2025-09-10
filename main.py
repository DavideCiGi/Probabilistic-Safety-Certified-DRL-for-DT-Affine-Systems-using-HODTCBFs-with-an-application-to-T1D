import numpy as np
from agent import Agent
from compensator import BarrierCompensator
# from cbfcontroller_gen import (DT_CBFs, CBF_with_GPs_SOCP)
from cbfcontroller import (DT_CBFs, CBF_with_GPs_SOCP)
from gps import (GP_Delta_kernel, compute_mean_and_square_root_covariance, beta,
                 preliminary_computations_for_mnsrc)
from utils import (plot_learning_curve, manage_memory, DT_Bergman_dynamics, plot_evaluation_run_with_GPs,
                   calculate_reward, HistoryTracker, plot_reward_function, plot_violation_curve, meal_schedule)
import gpflow
import tensorflow as tf
import time

# warmup 1000 CBF2 0.30 CBF1 0.35 works quite well for N 500, GP_max_size 500

if __name__ == '__main__':
    # Our knowledge about dynamics:
    nominal_model_params = {
        'p_1': 2.3e-6,
        'G_b': 75.0,
        'p_2': 0.088,
        'p_3': 0.63e-3,
        'n': 0.09,
        'I_b': 15.0,
        'tau_G': 47.0,
        'V_G': 253.0
    }

    CBF1_params = {
        'gamma_11': 0.30,
        'gamma_21': 0.30,
        'gamma_31': 0.30
    }

    CBF2_params = {
        'gamma_12': 0.35,
        'gamma_22': 0.35,
        'gamma_32': 0.35  # up to 50-45 it was working, even to 30-25
    }

    np.random.seed(42)
    # np.random.seed(40)

    # the true model dynamics
    true_model_params = {
        k: v if k in ('tau_G', 'V_G')  # , 'p_1', 'p_2', 'p_3', 'n')
        else np.random.choice([0.7, 1.3]) * v  # ideal variation 0.5-1.5, 0.7-1.3 good enough
        for k, v in nominal_model_params.items()
    }
    print(f'\nThe true model parameters:\n{true_model_params}')

    np.random.seed(None)

    answer1 = input("Do you want to do an evaluation run? ")
    while not (answer1 == 'yes' or answer1 == 'Yes' or answer1 == 'YES'
               or answer1 == 'no' or answer1 == 'No' or answer1 == 'NO'):
        answer1 = input("Please provide a yes or a no as an answer! ")
    if answer1 == 'yes' or answer1 == 'Yes' or answer1 == 'YES':
        evaluate = True
        restore_training = False
    else:
        evaluate = False
        answer2 = input("Do you want to do restore training? ")
        while not (answer2 == 'yes' or answer2 == 'Yes' or answer2 == 'YES'
                   or answer2 == 'no' or answer2 == 'No' or answer2 == 'NO'):
            answer2 = input("Please provide a yes or a no as an answer! ")
        if answer2 == 'yes' or answer2 == 'Yes' or answer2 == 'YES':
            restore_training = True
        else:
            restore_training = False

    # every time should be thought in minutes, unless specified differently
    dt = 1  # 0 < dt < 60

    N = 34 * 60
    # N = 500
    # N = 24 * 60
    print(f'Total number of steps: {N}.')
    CHO_max = 65_000  # YOU ASSUME TO KNOW THE MAXIMAL CHO INTAKE, old is 60k g

    past_bg = 0
    sample_time_bg = 1 * dt  # it's a sample time used to store prior glucose information, not obtained from sensors
    # to obtain the usual case --> past_bg = 0 and sample_time_bg = dt

    action_dim = 1
    state_dim = 5
    ext_state_dim = 5 + past_bg
    max_action = 30.  # important to guarantee it's a float
    min_action = 0.
    size_action = (max_action - min_action) / 2
    manage_memory()
    agent = Agent(state_dim=ext_state_dim,
                  action_dim=action_dim,
                  max_action=max_action/(24*60)*10,
                  min_action=min_action, alr=1e-3, clr=1e-3,
                  max_size=100_000, tau=5e-3, d=2, explore_sigma=0.1*size_action, smooth_sigma=0.2*size_action,
                  c=0.5*size_action, fc1_dims=256, fc2_dims=256, batch_size=256)

    compensator = BarrierCompensator(state_dim=state_dim,
                                     action_dim=action_dim,
                                     max_action=max_action,
                                     min_action=min_action, lr=1e-3,
                                     max_size=100_000, fc1_dims=30, fc2_dims=40, batch_size=100)

    # GP_max_size = 500
    GP_max_size = 1000
    dt_cbfs = DT_CBFs(dt, nominal_model_params, CBF1_params, CBF2_params, CHO_max, GP_max_size, N, true_model_params)

    kernels_psi = GP_Delta_kernel(state_dim, action_dim, dt_cbfs.r)
    kernels_phi = GP_Delta_kernel(state_dim, action_dim, dt_cbfs.r)
    GP_psi_dir = 'models/GP_psi'
    GP_phi_dir = 'models/GP_phi'

    k_delta = 1.
    K_eps_1 = 1e5  # old 1e6, decent initial 1e7
    K_eps_2 = 1e5  # old 1e6, decent initial 1e7
    cbf_socp = CBF_with_GPs_SOCP(k_delta, K_eps_1, K_eps_2, max_action, min_action, CBF1_params, CBF2_params)
    # a change in the parameters above require necessarily the SOCP problem to be re-written again in C

    G0 = 140
    best_score = calculate_reward(0.1) * N  # value that will get updated ofc
    print(f'Almost worst case scenario score: {best_score}.')
    print(f'Best policy reward: {calculate_reward(G0) * N}.')  # best reward we can hope
    print(f'Decent policy reward: {calculate_reward(90) * N}.')  # a reward worse than this, means something went wrong
    reward_avg_window = 50
    score_history = []
    max_violation_history = []

    if evaluate:
        n_games = 1

        states = []
        controls = []
        worst_CBF_psi = []
        worst_CBF_phi = []
        mean_CBF_psi = []
        mean_CBF_phi = []
        nominal_CBF_psi = []
        nominal_CBF_phi = []
        epsilon_psi = []
        epsilon_phi = []
        true_CBF_psi = []
        true_CBF_phi = []

        agent.load_models()
        compensator.load_model()

    else:
        n_games = 100
        # n_games = 85
        print(f'Number of episodes: {n_games}.')
        print(f'Collecting for the neural networks starts after {past_bg * sample_time_bg} steps for each episode.')
        plot_reward_function(figure_file='plots/RewardFunction.png')
        if restore_training:
            agent.load_models()
            compensator.load_model()

    time.sleep(5)

    for j in range(n_games):
        g_state = np.clip(np.random.normal(loc=G0, scale=15), a_min=dt_cbfs.BG_min_value, a_max=dt_cbfs.BG_max_value)
        print(f'Episode {j} started.')
        CHOs, eating = meal_schedule(N, dt, CHO_max)

        if j > 0 or evaluate or restore_training:
            psi_model = tf.saved_model.load(GP_psi_dir)
            phi_model = tf.saved_model.load(GP_phi_dir)

        score = 0
        episode_violations = []
        glu_ht = HistoryTracker(past_bg, sample_time_bg, 1)
        state = np.array([g_state, 0., true_model_params['I_b'], 0., 0.])
        glu_ht.add(state[0])
        ext_state = np.hstack((np.squeeze(glu_ht.history), state[1:]))
        print(f'Starting extended state: {ext_state}')
        for i in range(N):
            if state[0] <= 0:
                print('The blood glucose level has reached zero! Terminating the execution of the program.\nTry again!')
                break
            episode_violations.append(max(0., - state[0] + dt_cbfs.BG_min_value, + state[0] - dt_cbfs.BG_max_value))
            print(f'Step {i} started.')
            print(f'Current state: {state}')
            if i >= past_bg * sample_time_bg:  # Don't collect these steps into the NNs buffer? Then...
                u_RL = agent.choose_action(ext_state, evaluate)[0].numpy()
            else:  # ... don't act on what you don't know, leave the CBF operating alone!
                u_RL = 0.
            # u_RL = 0.
            print(f'u_RL: {u_RL}')

            if j > 0 or evaluate or restore_training:
                u_bar = compensator.compensation(state)[0].numpy()
                # u_bar = 0.
                print(f'u_bar: {u_bar}')

                # t0 = time.time()
                (psi_m_r_temp, psi_m_1_temp, psi_Lr_bar_temp,
                 psi_L1_bar_temp) = psi_model.compiled_mean_and_square_root_covariance(state.reshape((1, 5)))
                # t1 = time.time()
                # print('\nPsi_r mean&cov calc\nSolve time: %.3f ms' % (1000 * (t1 - t0)))

                if tf.reduce_any(tf.stack([tf.reduce_any(tf.math.is_nan(psi_Lr_bar_temp)),
                                           tf.reduce_any(tf.math.is_nan(psi_L1_bar_temp))])):
                    tf.print("\n!!! Cholesky produced NaNs for Sigma_psi!")

                (phi_m_r_temp, phi_m_1_temp, phi_Lr_bar_temp,
                 phi_L1_bar_temp) = phi_model.compiled_mean_and_square_root_covariance(state.reshape((1, 5)))

                if tf.reduce_any(tf.stack([tf.reduce_any(tf.math.is_nan(phi_Lr_bar_temp)),
                                           tf.reduce_any(tf.math.is_nan(phi_L1_bar_temp))])):
                    tf.print("\n!!! Cholesky produced NaNs for Sigma_phi!")

                psi_m_r, psi_m_1, psi_Lr_bar, psi_L1_bar = (psi_m_r_temp.numpy(), psi_m_1_temp.numpy(),
                                                            psi_Lr_bar_temp.numpy(), psi_L1_bar_temp.numpy())

                phi_m_r, phi_m_1, phi_Lr_bar, phi_L1_bar = (phi_m_r_temp.numpy(), phi_m_1_temp.numpy(),
                                                            phi_Lr_bar_temp.numpy(), phi_L1_bar_temp.numpy())

                # psi_m_r, psi_m_1, psi_Lr_bar, psi_L1_bar = (np.zeros((dt_cbfs.r, 1)),
                #                                             np.zeros((1, 1)), np.zeros((dt_cbfs.r + 1, dt_cbfs.r)),
                #                                             np.zeros((dt_cbfs.r + 1, 1)))
                # phi_m_r, phi_m_1, phi_Lr_bar, phi_L1_bar = (np.zeros((dt_cbfs.r, 1)),
                #                                             np.zeros((1, 1)), np.zeros((dt_cbfs.r + 1, dt_cbfs.r)),
                #                                             np.zeros((dt_cbfs.r + 1, 1)))

            else:  # basically if we are in the first step of the training phase and not in evaluation,
                # we restrict ourselves to a simple QP basically
                u_bar = 0.
                psi_m_r, psi_m_1, psi_Lr_bar, psi_L1_bar = (np.zeros((dt_cbfs.r, 1)),
                                                            np.zeros((1, 1)), np.zeros((dt_cbfs.r + 1, dt_cbfs.r)),
                                                            np.zeros((dt_cbfs.r + 1, 1)))
                phi_m_r, phi_m_1, phi_Lr_bar, phi_L1_bar = (np.zeros((dt_cbfs.r, 1)),
                                                            np.zeros((1, 1)), np.zeros((dt_cbfs.r + 1, dt_cbfs.r)),
                                                            np.zeros((dt_cbfs.r + 1, 1)))

            # print(f'Addition to the mean, when ID=0, to the psi CBF: {psi_m_r}')
            # print(f'Addition to the mean, when ID=0, to the phi CBF: {phi_m_r}')

            # t0 = time.time()
            a_11, a_21, a_12, a_22 = dt_cbfs.online_CBF_parameters(state)
            # t1 = time.time()
            # print('\nOnline CBF params\nSolve time: %.3f ms' % (1000 * (t1 - t0)))
            # print(f'a params: {a_11}, {a_21}, {a_12}, {a_22}')

            # t0 = time.time()
            sol, A_CBF, b_CBF, c_CBF, d_CBF = cbf_socp.solve(a_11, a_21, a_12, a_22, psi_m_r, psi_m_1, phi_m_r, phi_m_1,
                                                             psi_Lr_bar, phi_Lr_bar, psi_L1_bar, phi_L1_bar,
                                                             u_RL, u_bar)
            # t1 = time.time()
            # print('\nSolving online SOCP\nSolve time: %.3f ms' % (1000 * (t1 - t0)))
            # time.sleep(5)

            u_CBF = sol[0]
            print(f'u_CBF: {u_CBF}')

            ID = np.clip(u_bar + u_RL + u_CBF, a_min=min_action, a_max=max_action)  # pump limit
            control = np.array([ID, CHOs[i, 0]])

            if evaluate:
                sol_no_violations = sol.copy()
                sol_no_violations[-3] = 0.
                sol_no_violations[-2] = 0.

                states.append(state)
                controls.append(control)
                worst_CBF_psi.append(np.squeeze(c_CBF[0].T @ sol_no_violations + d_CBF[0] -
                                                np.linalg.norm(A_CBF[0] @ sol_no_violations + b_CBF[0])))
                worst_CBF_phi.append(np.squeeze(c_CBF[1].T @ sol_no_violations + d_CBF[1] -
                                                np.linalg.norm(A_CBF[1] @ sol_no_violations + b_CBF[1])))
                mean_CBF_psi.append(np.squeeze(c_CBF[0].T @ sol_no_violations + d_CBF[0]))
                mean_CBF_phi.append(np.squeeze(c_CBF[1].T @ sol_no_violations + d_CBF[1]))
                nominal_CBF_psi.append(a_11 * (sol_no_violations[0] + u_RL + u_bar) + a_21)
                nominal_CBF_phi.append(a_12 * (sol_no_violations[0] + u_RL + u_bar) + a_22)
                epsilon_psi.append(sol[-3])
                epsilon_phi.append(sol[-2])
                at_11, at_21, at_12, at_22 = dt_cbfs.online_true_CBF_parameters(state)
                true_CBF_psi.append(at_11 * (sol_no_violations[0] + u_RL + u_bar) + at_21)
                true_CBF_phi.append(at_12 * (sol_no_violations[0] + u_RL + u_bar) + at_22)

            new_state = DT_Bergman_dynamics(state, control, true_model_params, dt)
            glu_ht.add(new_state[0])
            new_ext_state = np.hstack((np.squeeze(glu_ht.history), new_state[1:]))  # CHECK LATER ON
            reward = calculate_reward(new_state[0])

            if not evaluate:
                state_action = np.concatenate((state, control[:1]), axis=0)  # action needs to be (1,)
                dt_cbfs.X_ht.add(state_action)
                    
                if i >= dt_cbfs.r:
                    dt_cbfs.GP_collect_data(eating[i - dt_cbfs.r])  # OR just dt_cbfs.r?

                compensator.collect_transition(state, u_bar + u_CBF)

                if i >= past_bg * sample_time_bg:
                    agent.collect_transition(ext_state, control[0], reward, new_ext_state, False)
                    # done=True --> for transitions where the episode terminates by reaching some failure state,
                    # and not due to the episode running until the max horizon (TD3 paper appendix)
            else:
                if i < dt_cbfs.r:
                    dt_cbfs.init_X_ht.add(state)
                if i == dt_cbfs.r-1:
                    dt_cbfs.initial_check_CBF()
                    time.sleep(5)

            score += reward
            state = new_state
            ext_state = new_ext_state
            # END EPISODE CYCLE

        if state[0] <= 0:
            break

        if evaluate:
            print('Episode terminated. Score {:.1f}. Max violation {:.1f}\n'.format(score,
                                                                                    max(episode_violations)))
        else:
            max_violation_history.append(max(episode_violations))
            score_history.append(score)
            avg_score = np.mean(score_history[-reward_avg_window:])
            print('Episode {} terminated. Score {:.1f}. Avg score {:.1f}. Max violation {:.1f}\n'.format(
                j, score, avg_score, max(episode_violations)))

            for k in range(N):  # learning done off-policy
                compensator.learn()
                agent.learn()

            if avg_score > best_score:
                best_score = avg_score
                agent_saved = agent.save_models()
                print(f"Is the agent model saved? {agent_saved}.\n")
                compensator_saved = compensator.save_model()
                print(f"Is the compensator model saved? {compensator_saved}.\n")

            psi_betas = np.array([beta(j, dt_cbfs.r, list(CBF1_params.values())) for j in range(1, dt_cbfs.r + 1)])
            phi_betas = np.array([beta(j, dt_cbfs.r, list(CBF2_params.values())) for j in range(1, dt_cbfs.r + 1)])

            X, Y_psi, Y_phi = dt_cbfs.GP_memory.special_sample_buffer()
            _, idx = np.unique(X, axis=0, return_index=True)
            if len(idx) < X.shape[0]:
                print("Beware: in X there are", X.shape[0] - len(idx), "duplicated rows.")
            # print(f'X shape: {X.shape}, take a look: {X[:10, :]}\n')
            # print(f'Y_psi shape: {Y_psi.shape}, take a look: {Y_psi[:10, :]}\n')
            # print(f'Y_phi shape: {Y_phi.shape}, take a look: {Y_phi[:10, :]}\n')
            
            psi_betas_array = np.tile(psi_betas.reshape(1, -1), (X.shape[0], 1))
            phi_betas_array = np.tile(phi_betas.reshape(1, -1), (X.shape[0], 1))
            X_psi_ext = np.concatenate((X[:, :state_dim], psi_betas_array,
                                        X[:, state_dim:state_dim + action_dim]), axis=1)
            X_phi_ext = np.concatenate((X[:, :state_dim], phi_betas_array,
                                        X[:, state_dim:state_dim + action_dim]), axis=1)

            psi_model = gpflow.models.GPR((X_psi_ext, Y_psi), kernel=kernels_psi)
            phi_model = gpflow.models.GPR((X_phi_ext, Y_phi), kernel=kernels_phi)

            opt = gpflow.optimizers.Scipy()
            opt.minimize(psi_model.training_loss, psi_model.trainable_variables)
            opt.minimize(phi_model.training_loss, phi_model.trainable_variables)

            gpflow.utilities.print_summary(psi_model)
            gpflow.utilities.print_summary(phi_model)

            psi_beta_rows, psi_m_right_factor, psi_L_hat = preliminary_computations_for_mnsrc(psi_model, action_dim)
            phi_beta_rows, phi_m_right_factor, phi_L_hat = preliminary_computations_for_mnsrc(phi_model, action_dim)

            psi_model.compiled_mean_and_square_root_covariance = tf.function(
                lambda x: compute_mean_and_square_root_covariance(x, model=psi_model, beta_rows=psi_beta_rows,
                                                                  m_right_factor=psi_m_right_factor, L_hat=psi_L_hat),
                input_signature=[tf.TensorSpec(shape=[1, state_dim], dtype=tf.float64)],
            )

            phi_model.compiled_mean_and_square_root_covariance = tf.function(
                lambda x: compute_mean_and_square_root_covariance(x, model=phi_model, beta_rows=phi_beta_rows,
                                                                  m_right_factor=phi_m_right_factor, L_hat=phi_L_hat),
                input_signature=[tf.TensorSpec(shape=[1, state_dim], dtype=tf.float64)],
            )

            tf.saved_model.save(psi_model, GP_psi_dir)
            tf.saved_model.save(phi_model, GP_phi_dir)
        # END GAMES CYCLE

    if evaluate:
        plot_evaluation_run_with_GPs(dt, states, controls, worst_CBF_psi, worst_CBF_phi, mean_CBF_psi, mean_CBF_phi,
                                     nominal_CBF_psi, nominal_CBF_phi, epsilon_psi, epsilon_phi,
                                     k_delta, true_CBF_psi, true_CBF_phi,
                                     figure_file='plots/BergmanEvaluationRunTD3.png')
    else:
        x = [i + 1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, reward_avg_window, figure_file='plots/BergmanTrainingScoreTD3.png')
        plot_violation_curve(x, max_violation_history, figure_file='plots/BergmanTrainingViolationsTD3.png')
