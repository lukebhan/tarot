from gym.envs.registration import register

register(
        id='Tarot-v0',
        entry_point='gym_tarot.envs:TarotBaseEnv')
