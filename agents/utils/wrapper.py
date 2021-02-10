from gym3.wrapper import Wrapper

class RgbWrapper(Wrapper):
    """
        A wrapper that extracts rgb from the original env.
    """
    def __init__(self, env, ob_space=None, ac_space=None):
        super().__init__(env, ob_space, ac_space)

    def observe(self):
        r, s, first = self.env.observe()
        return r, s['rgb'].astype('float32'), first